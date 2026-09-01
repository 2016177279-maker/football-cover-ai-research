from __future__ import annotations

import math

import pandas as pd

from src.preprocessing.engagement_metrics import add_engagement_metrics


def test_rates_are_nan_when_views_zero_and_log_is_safe() -> None:
    source = pd.DataFrame({"views": [100, 0], "likes": [10, 1], "comments": [2, 1], "shares": [1, 1], "saves": [7, 1]})
    result = add_engagement_metrics(source)
    assert result.loc[0, "like_rate"] == 0.1
    assert result.loc[0, "engagement_rate"] == 0.2
    assert math.isnan(result.loc[1, "like_rate"])
    assert math.isnan(result.loc[1, "engagement_rate"])
    assert result.loc[1, "log_views"] == 0
    assert "like_rate" not in source.columns
