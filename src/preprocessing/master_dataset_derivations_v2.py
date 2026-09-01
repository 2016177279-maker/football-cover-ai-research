"""Master Dataset v2 — Derived Outcome Variables.

Deterministic derivations from raw propagation counts.

This module implements the derivation contracts from the Master Dataset v2 Schema Contract.
All formulas are deterministic. NaN means "unknown" (never converted to 0).

DERIVATION FORMULAS
-------------------
log_view_count     = log1p(view_count)
log_like_count     = log1p(like_count)
log_comment_count  = log1p(comment_count)
video_age_days     = (collection_time - publish_time).total_seconds() / 86400
views_per_day      = view_count / max(video_age_days, 1)    [NaN when age ≤ 0 or NaN]
likes_per_1000_views  = like_count / view_count * 1000     [NaN when view_count == 0 or NaN]
comments_per_1000_views = comment_count / view_count * 1000 [NaN when view_count == 0 or NaN]
engagement         = (like_count + comment_count) / view_count  [NaN when view_count == 0 or NaN]

ZERO DENOMINATOR HANDLING
-------------------------
- view_count == 0      → log1p(0) = 0 (VALID, not NaN)
- view_count == 0      → rates → NaN (not inf, not 0)
- view_count == NaN    → rates → NaN
- video_age_days == 0  → views_per_day → NaN
- video_age_days == NaN → views_per_day → NaN
- video_age_days < 0    → NaN (future timestamp, invalid)

NaN HANDLING
------------
- NaN means "unknown/missing" — NEVER silently converted to 0
- 0 is a valid count (video has zero likes/comments)
- Negative counts are invalid → sanitized to NaN

AUTHORITATIVE IMPLEMENTATION
---------------------------
This module IS the authoritative source for v2 derivation formulas.
Cross-reference with: reports/master_dataset_v2/master_dataset_schema_contract_v1.md

PROHIBITED DERIVATIONS
-----------------------
- CTR (not a propagation measure; no authoritative source data)
- shares (NOT_AVAILABLE for YouTube)
- saves (NOT_AVAILABLE for YouTube)
- paid_boost (no authoritative source)
- roi (no authoritative source)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DERIVED_OUTCOME_COLUMNS: tuple[str, ...] = (
    "video_age_days",
    "log_view_count",
    "log_like_count",
    "log_comment_count",
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement",
)


# ---------------------------------------------------------------------------
# Dataclass for QC report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivedOutcomeReport:
    rows: int
    video_age_valid: int
    log_view_valid: int
    log_like_valid: int
    log_comment_valid: int
    views_per_day_valid: int
    likes_per_1000_valid: int
    comments_per_1000_valid: int
    engagement_valid: int
    invalid_publish_time: int
    invalid_collection_time: int
    negative_video_age: int
    view_count_zero: int
    view_count_missing: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_nonnegative_float(series: pd.Series) -> pd.Series:
    """Convert to float, coerce negative values and non-numeric to NaN.

    Negative counts are invalid data, not real negative engagement.
    They are excluded from derived metric computation (produce NaN).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0)


def _safe_log1p(values: pd.Series | None) -> pd.Series:
    """Return log1p of a non-negative series; NaN stays NaN.

    Note: log1p(0) = 0, which is VALID.
    Zero views/likes/comments are not errors.
    """
    if values is None:
        return pd.Series(dtype=float)
    result = np.log1p(values)
    return result


def _safe_rate_per_1000(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    """Compute rate per 1000 with NaN on zero or missing denominator.

    Returns NaN when denominator is 0, NaN, or negative.
    """
    safe_denom = denominator.where(denominator > 0)
    return (numerator / safe_denom) * 1000


def _safe_rate(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    """Safe division; returns NaN when denominator is 0, NaN, or negative."""
    safe_denom = denominator.where(denominator > 0)
    return numerator / safe_denom


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------

def add_derived_outcomes(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, DerivedOutcomeReport]:
    """Add v2 derived outcome columns to a DataFrame.

    Parameters
    ----------
    frame : pd.DataFrame
        Must contain:
        - view_count      : Raw view count (may contain NaN, 0, or positive)
        - like_count      : Raw like count
        - comment_count   : Raw comment count
        - publish_time    : ISO-8601 timestamp
        - collection_time : ISO-8601 timestamp

    Returns
    -------
    tuple[pd.DataFrame, DerivedOutcomeReport]
        DataFrame with derived columns added, and a QC report.

    Formulas
    --------
    log_view_count        = log1p(view_count)
    log_like_count        = log1p(like_count)
    log_comment_count     = log1p(comment_count)
    video_age_days        = (collection_time - publish_time).days
    views_per_day         = view_count / max(video_age_days, 1)
                            [NaN when age ≤ 0]
    likes_per_1000_views  = like_count / view_count * 1000
                            [NaN when view_count == 0]
    comments_per_1000_views = comment_count / view_count * 1000
                            [NaN when view_count == 0]
    engagement            = (like_count + comment_count) / view_count
                            [NaN when view_count == 0]

    Missing Data Policy
    -------------------
    - NaN source → NaN derived
    - Zero is valid for all counts (log1p(0) = 0)
    - Zero in denominator → NaN (not inf or 0)
    - Negative counts → NaN (invalid data)
    """
    required_cols = {
        "view_count",
        "like_count",
        "comment_count",
        "publish_time",
        "collection_time",
    }
    missing = sorted(required_cols.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = frame.copy()

    # ── 1. Parse timestamps ────────────────────────────────────────────────
    publish_ts = pd.to_datetime(
        result["publish_time"], errors="coerce", utc=True, format="mixed"
    )
    collection_ts = pd.to_datetime(
        result["collection_time"], errors="coerce", utc=True, format="mixed"
    )

    # ── 2. Compute video_age_days ──────────────────────────────────────────
    raw_age_days = (collection_ts - publish_ts).dt.total_seconds() / 86_400
    negative_age_mask = raw_age_days < 0
    result["video_age_days"] = raw_age_days.where(~negative_age_mask)

    # ── 3. Sanitize raw counts (negative → NaN) ───────────────────────────
    view_count = _to_nonnegative_float(result["view_count"])
    like_count = _to_nonnegative_float(result["like_count"])
    comment_count = _to_nonnegative_float(result["comment_count"])

    # ── 4. Log-transformed outcomes ───────────────────────────────────────
    result["log_view_count"] = _safe_log1p(view_count)
    result["log_like_count"] = _safe_log1p(like_count)
    result["log_comment_count"] = _safe_log1p(comment_count)

    # ── 5. Views per day ───────────────────────────────────────────────────
    valid_age_mask = result["video_age_days"].notna() & (result["video_age_days"] > 0)
    age_denom = result["video_age_days"].where(valid_age_mask).clip(lower=1)
    result["views_per_day"] = _safe_rate(view_count, age_denom)

    # ── 6. Per-1000-views rates ──────────────────────────────────────────
    result["likes_per_1000_views"] = _safe_rate_per_1000(like_count, view_count)
    result["comments_per_1000_views"] = _safe_rate_per_1000(comment_count, view_count)

    # ── 7. Engagement ─────────────────────────────────────────────────────
    # engagement = (likes + comments) / views
    # NaN when views is 0, NaN, or negative
    result["engagement"] = _safe_rate(like_count + comment_count, view_count)

    # ── 8. Sanitize inf → NaN ─────────────────────────────────────────────
    derived_cols = list(DERIVED_OUTCOME_COLUMNS)
    result[derived_cols] = result[derived_cols].replace([np.inf, -np.inf], np.nan)

    # ── 9. Build QC report ────────────────────────────────────────────────
    invalid_publish_mask = publish_ts.isna()
    invalid_collect_mask = collection_ts.isna()
    negative_age_filled = negative_age_mask.fillna(False)

    report = DerivedOutcomeReport(
        rows=len(result),
        video_age_valid=int(result["video_age_days"].notna().sum()),
        log_view_valid=int(result["log_view_count"].notna().sum()),
        log_like_valid=int(result["log_like_count"].notna().sum()),
        log_comment_valid=int(result["log_comment_count"].notna().sum()),
        views_per_day_valid=int(result["views_per_day"].notna().sum()),
        likes_per_1000_valid=int(result["likes_per_1000_views"].notna().sum()),
        comments_per_1000_valid=int(result["comments_per_1000_views"].notna().sum()),
        engagement_valid=int(result["engagement"].notna().sum()),
        invalid_publish_time=int(invalid_publish_mask.sum()),
        invalid_collection_time=int(invalid_collect_mask.sum()),
        negative_video_age=int(negative_age_filled.sum()),
        view_count_zero=int((view_count == 0).sum()),
        view_count_missing=int(view_count.isna().sum()),
    )

    return result, report


# ---------------------------------------------------------------------------
# Verification helpers (for tests)
# ---------------------------------------------------------------------------

def verify_derivation_formulas() -> dict[str, object]:
    """Return human-readable formula reference for documentation."""
    return {
        "log_view_count": {
            "formula": "log1p(view_count)",
            "zero_handling": "log1p(0) = 0 (valid)",
            "nan_handling": "NaN when view_count is NaN",
            "negative_handling": "NaN (invalid data)",
        },
        "log_like_count": {
            "formula": "log1p(like_count)",
            "zero_handling": "log1p(0) = 0 (valid)",
            "nan_handling": "NaN when like_count is NaN",
            "negative_handling": "NaN (invalid data)",
        },
        "log_comment_count": {
            "formula": "log1p(comment_count)",
            "zero_handling": "log1p(0) = 0 (valid)",
            "nan_handling": "NaN when comment_count is NaN",
            "negative_handling": "NaN (invalid data)",
        },
        "video_age_days": {
            "formula": "(collection_time - publish_time).total_seconds() / 86400",
            "zero_handling": "0 days (valid)",
            "negative_handling": "NaN (future timestamp, invalid)",
            "nan_handling": "NaN when either timestamp is NaN",
        },
        "views_per_day": {
            "formula": "view_count / max(video_age_days, 1)",
            "zero_handling": "NaN when video_age_days == 0",
            "nan_handling": "NaN when video_age_days is NaN",
            "negative_handling": "NaN when video_age_days < 0",
        },
        "likes_per_1000_views": {
            "formula": "like_count / view_count * 1000",
            "zero_handling": "NaN when view_count == 0",
            "nan_handling": "NaN when view_count is NaN",
            "negative_handling": "NaN (invalid data)",
        },
        "comments_per_1000_views": {
            "formula": "comment_count / view_count * 1000",
            "zero_handling": "NaN when view_count == 0",
            "nan_handling": "NaN when view_count is NaN",
            "negative_handling": "NaN (invalid data)",
        },
        "engagement": {
            "formula": "(like_count + comment_count) / view_count",
            "zero_handling": "NaN when view_count == 0",
            "nan_handling": "NaN when view_count is NaN, or likes/comments are NaN",
            "negative_handling": "NaN (invalid data)",
        },
    }
