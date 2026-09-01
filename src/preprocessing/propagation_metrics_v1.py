"""Propagation Metrics v1 for formal research datasets.

This module calculates research outcome variables only. It does not select
a research outcome, fit a model, or support substantive interpretation.

DESIGN PRINCIPLES
-----------------
1. Raw outcomes are NEVER modified; they remain in their original columns.
2. Derived metrics are always added as new columns.
3. Missing values (NaN) are NEVER silently converted to 0.
4. Zero values are distinct from missing values.
5. All computations are deterministic given the same input data.
6. video_age_days is computed from explicit collection_time, never from today's date.

METRIC TIERS
------------
RAW      : views, likes, comments, author_followers (source columns, unmodified)
DERIVED  : video_age_days, log_views, log_likes, log_comments, like_rate,
           comment_rate, engagement_rate, views_per_day, log_author_followers
OPTIONAL : follower_normalized_views, log_views_per_day, log_engagement_count,
           log_views_per_follower

OUTCOME HIERARCHY (CANDIDATE ONLY — NOT FINALIZED)
--------------------------------------------------
Primary candidates   : log_views, engagement_rate, views_per_day
Secondary candidates: log_likes, log_comments, like_rate, comment_rate,
                      views_per_follower, follower_normalized_views

MISSING DATA POLICY
-------------------
NaN != 0. A NaN like_count means "unknown", not "zero likes".
Each derived metric has explicit handling rules documented in its docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

PROPAGATION_METRIC_COLUMNS = (
    "video_age_days",
    "log_views",
    "log_likes",
    "log_comments",
    "like_rate",
    "comment_rate",
    "engagement_rate",
    "views_per_day",
    "log_author_followers",
    "follower_normalized_views",
)

OPTIONAL_PROPAGATION_METRIC_COLUMNS = (
    "log_views_per_day",
    "log_engagement_count",
    "log_views_per_follower",
)

ALL_DERIVED_METRICS = PROPAGATION_METRIC_COLUMNS + OPTIONAL_PROPAGATION_METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Dataclasses for reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingDataSummary:
    """Summarizes missing data patterns for propagation source columns."""

    views_missing: int
    views_zero: int
    views_negative: int
    likes_missing: int
    likes_zero: int
    likes_negative: int
    comments_missing: int
    comments_zero: int
    comments_negative: int
    followers_missing: int
    followers_zero: int
    followers_negative: int
    publish_time_invalid: int
    collection_time_invalid: int
    video_age_negative: int


@dataclass(frozen=True)
class PropagationMetricReport:
    """Quality-control report for propagation metric computation."""

    rows: int
    successful_derived_rows: int
    success_rate: float
    all_required_derived_present: int
    all_required_derived_present_rate: float
    invalid_publish_time_rows: tuple[int, ...]
    invalid_collection_time_rows: tuple[int, ...]
    negative_video_age_rows: tuple[int, ...]
    missing_data: MissingDataSummary
    extreme_views_per_day_rows: tuple[int, ...]
    extreme_engagement_rate_rows: tuple[int, ...]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_nonnegative_float(series: pd.Series) -> pd.Series:
    """Convert to float, coerce negative values and non-numeric to NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0)


def _safe_log1p(values: pd.Series | None) -> pd.Series:
    """Return log1p of a non-negative series; NaN stays NaN."""
    if values is None:
        return pd.Series(dtype=float)
    result = np.log1p(values)
    return result


def _compute_rate(
    numerator: pd.Series, denominator: pd.Series, default: float = np.nan
) -> pd.Series:
    """Safe division; returns default when denominator is 0, NaN, or negative."""
    valid_denom = denominator.where(denominator > 0)
    return numerator / valid_denom


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------


def add_propagation_metrics(
    frame: pd.DataFrame,
    include_optional: bool = False,
    extreme_views_per_day_threshold: float = 1_000_000.0,
    extreme_engagement_rate_threshold: float = 1.0,
) -> tuple[pd.DataFrame, PropagationMetricReport]:
    """Return a copy with v1 propagation metrics plus a quality report.

    Parameters
    ----------
    frame : pd.DataFrame
        Input DataFrame with required source columns.
    include_optional : bool
        If True, also compute OPTIONAL propagation metrics
        (log_views_per_day, log_engagement_count, log_views_per_follower).
    extreme_views_per_day_threshold : float
        Flag rows where views_per_day exceeds this value.
    extreme_engagement_rate_threshold : float
        Flag rows where engagement_rate exceeds this value (e.g., 1.0 = 100%).

    Returns
    -------
    tuple[pd.DataFrame, PropagationMetricReport]
        Modified DataFrame (copy) with derived columns added, and a QC report.

    Required Source Columns
    -----------------------
    - publish_time      : ISO-8601 timestamp of video publication
    - collection_time    : ISO-8601 timestamp when data was collected
    - views             : Raw view count (RAW, never modified)
    - likes             : Raw like count (RAW, never modified)
    - comments          : Raw comment count (RAW, never modified)
    - author_followers  : Raw follower count (RAW, never modified)

    Missing Data Policy
    -------------------
    - NaN source values propagate to NaN in derived metrics.
    - Zero is a VALID value; 0 likes means the video has zero likes.
    - NaN means the value is unknown or invalid.
    - Rate denominators of 0 produce NaN (not 0 or inf).
    - video_age_days of 0 or negative produces NaN for views_per_day.

    Determinism
    -----------
    All derived metrics are deterministic given the same input.
    video_age_days uses collection_time from the row, never today's date.
    """
    required = {
        "publish_time",
        "collection_time",
        "views",
        "likes",
        "comments",
        "author_followers",
    }
    missing_cols = sorted(required.difference(frame.columns))
    if missing_cols:
        raise ValueError(f"Missing propagation source columns: {', '.join(missing_cols)}")

    result = frame.copy()

    # ── 1. Parse timestamps ────────────────────────────────────────────────
    publish_ts = pd.to_datetime(
        result["publish_time"], errors="coerce", utc=True, format="mixed"
    )
    collection_ts = pd.to_datetime(
        result["collection_time"], errors="coerce", utc=True, format="mixed"
    )

    # ── 2. Compute video_age_days ─────────────────────────────────────────
    raw_age_days = (collection_ts - publish_ts).dt.total_seconds() / 86_400
    negative_age_mask = raw_age_days < 0
    result["video_age_days"] = raw_age_days.where(~negative_age_mask)

    # ── 3. Convert raw counts to non-negative numerics ─────────────────────
    views = _to_nonnegative_float(result["views"])
    likes = _to_nonnegative_float(result["likes"])
    comments = _to_nonnegative_float(result["comments"])
    followers = _to_nonnegative_float(result["author_followers"])

    # NOTE: Raw columns (views, likes, comments, author_followers) are preserved
    # as-is. Negative values are invalid data, not silently corrected to 0.
    # They produce NaN in derived metrics rather than corrupting raw data.

    # ── 4. Log-transformed outcomes ────────────────────────────────────────
    result["log_views"] = _safe_log1p(views)
    result["log_likes"] = _safe_log1p(likes)
    result["log_comments"] = _safe_log1p(comments)

    # ── 5. Engagement rates ────────────────────────────────────────────────
    # Use views > 0 as denominator; NaN when views is 0, missing, or negative
    valid_views_for_rate = views.where(views > 0)
    result["like_rate"] = _compute_rate(likes, valid_views_for_rate)
    result["comment_rate"] = _compute_rate(comments, valid_views_for_rate)
    result["engagement_rate"] = _compute_rate(
        likes + comments, valid_views_for_rate
    )

    # ── 6. Exposure-time metrics ──────────────────────────────────────────
    # views_per_day: NaN when video_age_days is NaN or <= 0.
    # For valid ages, denominator is max(video_age_days, 1).
    # CRITICAL: clip AFTER where(), not before, to avoid converting NaN to 1.
    valid_age_mask = result["video_age_days"].notna() & (result["video_age_days"] > 0)
    age_denom = result["video_age_days"].where(valid_age_mask).clip(lower=1)
    result["views_per_day"] = _compute_rate(views, age_denom)

    # ── 7. Creator-scale metrics ──────────────────────────────────────────
    result["log_author_followers"] = _safe_log1p(followers)

    # ── 8. Follower-normalized views ─────────────────────────────────────
    # views / followers; NaN when followers is 0, missing, or negative
    result["follower_normalized_views"] = _compute_rate(
        views, followers.where(followers > 0)
    )

    # ── 9. Optional derived metrics ───────────────────────────────────────
    if include_optional:
        result["log_views_per_day"] = _safe_log1p(result["views_per_day"])
        result["log_engagement_count"] = _safe_log1p(likes + comments)
        result["log_views_per_follower"] = _safe_log1p(
            _compute_rate(views, followers.where(followers > 0))
        )

    # ── 9. Sanitize: replace any inf/-inf with NaN ────────────────────────
    derived_cols = (
        list(PROPAGATION_METRIC_COLUMNS)
        + (list(OPTIONAL_PROPAGATION_METRIC_COLUMNS) if include_optional else [])
    )
    result[derived_cols] = result[derived_cols].replace([np.inf, -np.inf], np.nan)

    # ── 10. Build quality report ───────────────────────────────────────────
    invalid_publish_mask = publish_ts.isna()
    invalid_collect_mask = collection_ts.isna()
    negative_age_mask_filled = negative_age_mask.fillna(False)

    extreme_vpd_mask = result["views_per_day"] > extreme_views_per_day_threshold
    extreme_er_mask = result["engagement_rate"] > extreme_engagement_rate_threshold

    # Count missing/zero/negative patterns
    views_is_nan = views.isna()
    likes_is_nan = likes.isna()
    comments_is_nan = comments.isna()
    followers_is_nan = followers.isna()

    missing_summary = MissingDataSummary(
        views_missing=int(views_is_nan.sum()),
        views_zero=int((views == 0).sum()),
        views_negative=int((views < 0).sum()),
        likes_missing=int(likes_is_nan.sum()),
        likes_zero=int((likes == 0).sum()),
        likes_negative=int((likes < 0).sum()),
        comments_missing=int(comments_is_nan.sum()),
        comments_zero=int((comments == 0).sum()),
        comments_negative=int((comments < 0).sum()),
        followers_missing=int(followers_is_nan.sum()),
        followers_zero=int((followers == 0).sum()),
        followers_negative=int((followers < 0).sum()),
        publish_time_invalid=int(invalid_publish_mask.sum()),
        collection_time_invalid=int(invalid_collect_mask.sum()),
        video_age_negative=int(negative_age_mask_filled.sum()),
    )

    all_derived = list(PROPAGATION_METRIC_COLUMNS)
    all_derived_present = result[all_derived].notna().all(axis=1)
    successful_derived = result[all_derived].notna().any(axis=1)

    report = PropagationMetricReport(
        rows=len(result),
        successful_derived_rows=int(successful_derived.sum()),
        success_rate=(
            float(successful_derived.mean()) if len(result) > 0 else 1.0
        ),
        all_required_derived_present=int(all_derived_present.sum()),
        all_required_derived_present_rate=(
            float(all_derived_present.mean()) if len(result) > 0 else 1.0
        ),
        invalid_publish_time_rows=tuple(
            result.index[invalid_publish_mask].tolist()
        ),
        invalid_collection_time_rows=tuple(
            result.index[invalid_collect_mask].tolist()
        ),
        negative_video_age_rows=tuple(
            result.index[negative_age_mask_filled].tolist()
        ),
        missing_data=missing_summary,
        extreme_views_per_day_rows=tuple(
            result.index[extreme_vpd_mask].tolist()
        ),
        extreme_engagement_rate_rows=tuple(
            result.index[extreme_er_mask].tolist()
        ),
    )

    return result, report


# ---------------------------------------------------------------------------
# Standalone CLI for batch processing
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for propagation metrics computation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Compute Propagation Metrics v1 from a CSV file."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional derived metrics (log_views_per_day, etc.)",
    )
    parser.add_argument(
        "--report-json",
        "-r",
        help="Optional path to write a JSON quality report",
    )
    args = parser.parse_args()

    print(f"Reading input: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows.")

    df_out, report = add_propagation_metrics(
        df, include_optional=args.include_optional
    )

    print(f"Writing output: {args.output}")
    df_out.to_csv(args.output, index=False)

    print(f"\n--- Propagation Metrics v1 QC Report ---")
    print(f"Rows processed          : {report.rows}")
    print(f"Successful derived rows : {report.successful_derived_rows} ({report.success_rate:.1%})")
    print(
        f"All required present    : {report.all_required_derived_present} "
        f"({report.all_required_derived_present_rate:.1%})"
    )
    print(f"Invalid publish_time    : {report.missing_data.publish_time_invalid}")
    print(f"Invalid collection_time : {report.missing_data.collection_time_invalid}")
    print(f"Negative video_age_days : {report.missing_data.video_age_negative}")
    print(f"Missing views           : {report.missing_data.views_missing}")
    print(f"Zero views              : {report.missing_data.views_zero}")
    print(f"Missing likes           : {report.missing_data.likes_missing}")
    print(f"Zero likes              : {report.missing_data.likes_zero}")
    print(f"Missing comments        : {report.missing_data.comments_missing}")
    print(f"Zero comments           : {report.missing_data.comments_zero}")
    print(f"Missing followers       : {report.missing_data.followers_missing}")
    print(f"Zero followers          : {report.missing_data.followers_zero}")
    print(f"Extreme views_per_day   : {len(report.extreme_views_per_day_rows)} rows")
    print(f"Extreme engagement_rate : {len(report.extreme_engagement_rate_rows)} rows")

    if args.report_json:
        import json

        report_dict = {
            "rows": report.rows,
            "successful_derived_rows": report.successful_derived_rows,
            "success_rate": report.success_rate,
            "all_required_derived_present": report.all_required_derived_present,
            "all_required_derived_present_rate": (
                report.all_required_derived_present_rate
            ),
            "invalid_publish_time_rows": report.invalid_publish_time_rows,
            "invalid_collection_time_rows": report.invalid_collection_time_rows,
            "negative_video_age_rows": report.negative_video_age_rows,
            "missing_data": {
                "views_missing": report.missing_data.views_missing,
                "views_zero": report.missing_data.views_zero,
                "views_negative": report.missing_data.views_negative,
                "likes_missing": report.missing_data.likes_missing,
                "likes_zero": report.missing_data.likes_zero,
                "likes_negative": report.missing_data.likes_negative,
                "comments_missing": report.missing_data.comments_missing,
                "comments_zero": report.missing_data.comments_zero,
                "comments_negative": report.missing_data.comments_negative,
                "followers_missing": report.missing_data.followers_missing,
                "followers_zero": report.missing_data.followers_zero,
                "followers_negative": report.missing_data.followers_negative,
                "publish_time_invalid": report.missing_data.publish_time_invalid,
                "collection_time_invalid": report.missing_data.collection_time_invalid,
                "video_age_negative": report.missing_data.video_age_negative,
            },
            "extreme_views_per_day_rows": report.extreme_views_per_day_rows,
            "extreme_engagement_rate_rows": report.extreme_engagement_rate_rows,
        }
        with open(args.report_json, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Report written to: {args.report_json}")


if __name__ == "__main__":
    main()
