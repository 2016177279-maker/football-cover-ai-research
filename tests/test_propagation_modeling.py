from __future__ import annotations

import pandas as pd
import pytest

from src.models.propagation_modeling import (
    PIPELINE_WARNING,
    default_model_specifications,
    dry_run,
    explain_model,
)


def _modeling_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_views": [4.0, 5.0, 6.0],
            "log_author_followers": [2.0, 3.0, None],
            "video_age_days": [2.0, 3.0, 4.0],
            "content_subtype": ["highlights", "news", "news"],
            "publish_month": [1, 1, 2],
            "publish_weekday": [0, 1, 2],
            "publish_hour": [12, 18, 20],
            "query_keyword": ["football", "soccer", "soccer"],
            "brightness_mean": [100.0, 120.0, 110.0],
        }
    )


def test_pipeline_dry_run_prepares_data_without_fitting() -> None:
    specification = default_model_specifications(["brightness_mean"])[1]
    result = dry_run(_modeling_frame(), specification)
    assert result["status"] == PIPELINE_WARNING
    assert result["input_rows"] == 3
    assert result["usable_rows"] == 2
    assert result["dropped_rows"] == 1
    assert result["model_fitted"] is False
    assert "brightness_mean" in result["design_columns"]


def test_pipeline_rejects_missing_required_columns() -> None:
    specification = default_model_specifications(["missing_visual"])[1]
    with pytest.raises(ValueError, match="missing_visual"):
        dry_run(_modeling_frame(), specification)


def test_shap_interface_is_explicitly_deferred() -> None:
    with pytest.raises(NotImplementedError, match=">=5000"):
        explain_model()
