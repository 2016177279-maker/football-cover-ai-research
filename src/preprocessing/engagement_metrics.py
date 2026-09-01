"""Candidate engagement metrics for research exploration, not model targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGAGEMENT_COLUMNS = ("likes", "comments", "shares", "saves")
RATE_COLUMNS = ("like_rate", "comment_rate", "share_rate", "save_rate", "engagement_rate")


def add_engagement_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with safe rate metrics and ``log_views``.

    Missing values remain missing. Rates are NaN whenever views are zero,
    negative, or missing; infinite values are never emitted.
    """
    required = {"views", *ENGAGEMENT_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"缺少传播指标字段: {', '.join(missing)}")
    result = frame.copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid_views = result["views"].where(result["views"] > 0)
    for source, target in zip(ENGAGEMENT_COLUMNS, RATE_COLUMNS[:4], strict=True):
        result[target] = result[source] / valid_views
    result["engagement_rate"] = result[list(ENGAGEMENT_COLUMNS)].sum(axis=1, min_count=4) / valid_views
    result["log_views"] = np.log1p(result["views"].where(result["views"] >= 0))
    result[list(RATE_COLUMNS)] = result[list(RATE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    return result
