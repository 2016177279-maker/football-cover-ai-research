"""Calendar-based recency filtering helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def recent_months_cutoff(as_of: datetime, months: int = 12) -> pd.Timestamp:
    """Return the inclusive calendar-month cutoff relative to ``as_of``."""
    if months <= 0:
        raise ValueError("months 必须大于 0")
    return pd.Timestamp(as_of) - pd.DateOffset(months=months)


def filter_recent_posts(
    frame: pd.DataFrame,
    *,
    as_of: datetime,
    months: int = 12,
    time_column: str = "publish_time",
) -> pd.DataFrame:
    """Return rows published in the inclusive calendar window ending at ``as_of``."""
    if time_column not in frame.columns:
        raise ValueError(f"缺少时间字段: {time_column}")
    result = frame.copy()
    parsed = pd.to_datetime(result[time_column], utc=True, errors="coerce")
    end = pd.Timestamp(as_of)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    cutoff = recent_months_cutoff(end.to_pydatetime(), months)
    return result.loc[parsed.between(cutoff, end, inclusive="both")].copy()
