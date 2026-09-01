"""Comprehensive tests for Propagation Metrics v1.

This test suite verifies the correctness of propagation metric computation
using controlled synthetic data. All tests use unit data designed to
exercise specific code paths without making assumptions about real distributions.

MARK: DEVELOPMENT VALIDATION ONLY
These tests are for formula correctness, missing handling, zero handling,
extreme values, deterministic behavior, and output schema verification.
They do NOT produce research conclusions.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.propagation_metrics_v1 import (
    MissingDataSummary,
    PropagationMetricReport,
    add_propagation_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_valid_row() -> dict:
    """A single valid row with all required columns."""
    return {
        "video_id": "test001",
        "publish_time": "2025-01-01T00:00:00+00:00",
        "collection_time": "2025-01-11T00:00:00+00:00",
        "views": 1000.0,
        "likes": 50.0,
        "comments": 10.0,
        "author_followers": 10000.0,
    }


@pytest.fixture
def base_frame(minimal_valid_row: dict) -> pd.DataFrame:
    """A minimal valid DataFrame with one row."""
    return pd.DataFrame([minimal_valid_row])


# ---------------------------------------------------------------------------
# Test: Required columns validation
# ---------------------------------------------------------------------------


def test_raises_on_missing_source_column() -> None:
    """Must raise ValueError when a required source column is missing."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        # views, likes, comments, author_followers are all missing
    })
    with pytest.raises(ValueError, match="Missing propagation source columns"):
        add_propagation_metrics(frame)


# ---------------------------------------------------------------------------
# Test: Input DataFrame is not modified
# ---------------------------------------------------------------------------


def test_input_unchanged(base_frame: pd.DataFrame) -> None:
    """Input DataFrame must not be mutated."""
    original = base_frame.copy()
    add_propagation_metrics(base_frame)
    pd.testing.assert_frame_equal(base_frame, original)


# ---------------------------------------------------------------------------
# Test: Normal case
# ---------------------------------------------------------------------------


def test_normal_case_log_transforms() -> None:
    """Verify log1p transforms for normal counts."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert result.loc[0, "log_views"] == pytest.approx(math.log1p(1000.0))
    assert result.loc[0, "log_likes"] == pytest.approx(math.log1p(50.0))
    assert result.loc[0, "log_comments"] == pytest.approx(math.log1p(10.0))
    assert result.loc[0, "log_author_followers"] == pytest.approx(math.log1p(10000.0))
    assert report.rows == 1
    assert report.all_required_derived_present == 1


def test_normal_case_engagement_rates() -> None:
    """Verify engagement rate calculations."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "like_rate"] == pytest.approx(0.05)
    assert result.loc[0, "comment_rate"] == pytest.approx(0.01)
    assert result.loc[0, "engagement_rate"] == pytest.approx(0.06)


def test_normal_case_video_age_and_views_per_day() -> None:
    """Verify video_age_days and views_per_day calculations."""
    # 10 days between publish and collection
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "video_age_days"] == pytest.approx(10.0)
    assert result.loc[0, "views_per_day"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test: Zero values
# ---------------------------------------------------------------------------


def test_zero_views_produces_nan_rates() -> None:
    """Zero views must NOT produce inf; rate metrics must be NaN."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [0.0],
        "likes": [10.0],
        "comments": [5.0],
        "author_followers": [1000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "like_rate"])
    assert math.isnan(result.loc[0, "comment_rate"])
    assert math.isnan(result.loc[0, "engagement_rate"])
    # views_per_day: video_age_days=10 > 0, so views/10 = 0
    assert result.loc[0, "views_per_day"] == pytest.approx(0.0)
    # log1p(0) = 0, which is valid
    assert result.loc[0, "log_views"] == pytest.approx(0.0)
    assert report.rows == 1
    assert report.all_required_derived_present == 0


def test_zero_likes_and_comments_valid() -> None:
    """Zero likes/comments are valid; log1p(0) = 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [0.0],
        "comments": [0.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "like_rate"] == pytest.approx(0.0)
    assert result.loc[0, "comment_rate"] == pytest.approx(0.0)
    assert result.loc[0, "engagement_rate"] == pytest.approx(0.0)
    assert result.loc[0, "log_likes"] == pytest.approx(0.0)
    assert result.loc[0, "log_comments"] == pytest.approx(0.0)


def test_zero_followers_produces_nan_log_followers() -> None:
    """Zero followers: log1p(0) = 0 is valid, but follower_normalized_views is NaN."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [100.0],
        "likes": [5.0],
        "comments": [2.0],
        "author_followers": [0.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "log_author_followers"] == pytest.approx(0.0)
    assert math.isnan(result.loc[0, "follower_normalized_views"])


# ---------------------------------------------------------------------------
# Test: Missing values (NaN)
# ---------------------------------------------------------------------------


def test_missing_views_propagates_to_derived() -> None:
    """Missing views must produce NaN in log_views and NaN in rate metrics."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [np.nan],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "views"])
    assert math.isnan(result.loc[0, "log_views"])
    assert math.isnan(result.loc[0, "like_rate"])
    assert math.isnan(result.loc[0, "comment_rate"])
    assert math.isnan(result.loc[0, "engagement_rate"])
    assert math.isnan(result.loc[0, "views_per_day"])
    assert report.missing_data.views_missing == 1


def test_missing_likes_propagates_to_derived() -> None:
    """Missing likes must produce NaN in log_likes and NaN in like_rate/engagement_rate."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [np.nan],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "likes"])
    assert math.isnan(result.loc[0, "log_likes"])
    assert math.isnan(result.loc[0, "like_rate"])
    assert math.isnan(result.loc[0, "engagement_rate"])
    # comment_rate should still work since views > 0
    assert result.loc[0, "comment_rate"] == pytest.approx(0.01)


def test_missing_comments_propagates_to_derived() -> None:
    """Missing comments must produce NaN in log_comments and NaN in engagement_rate."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [np.nan],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "comments"])
    assert math.isnan(result.loc[0, "log_comments"])
    assert math.isnan(result.loc[0, "engagement_rate"])
    # like_rate should still work
    assert result.loc[0, "like_rate"] == pytest.approx(0.05)


def test_missing_followers_propagates_to_derived() -> None:
    """Missing followers must produce NaN in log_author_followers."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [np.nan],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "author_followers"])
    assert math.isnan(result.loc[0, "log_author_followers"])


def test_missing_publish_time() -> None:
    """Missing publish_time produces NaN in video_age_days."""
    frame = pd.DataFrame({
        "publish_time": [np.nan],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "video_age_days"])
    assert math.isnan(result.loc[0, "views_per_day"])
    # Single-row frame has index 0, not 1
    assert 0 in report.invalid_publish_time_rows


def test_missing_collection_time() -> None:
    """Missing collection_time produces NaN in video_age_days."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": [np.nan],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "video_age_days"])
    # Single-row frame has index 0, not 1
    assert 0 in report.invalid_collection_time_rows


# ---------------------------------------------------------------------------
# Test: Invalid negative counts
# ---------------------------------------------------------------------------


def test_negative_views_converted_to_nan() -> None:
    """Negative views must produce NaN in derived metrics (raw column preserved)."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [-100.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    # Raw column is preserved as-is (negative value remains)
    # Derived metrics use sanitized values, so log_views is NaN
    assert result.loc[0, "log_views"] is np.nan or math.isnan(result.loc[0, "log_views"])
    assert result.loc[0, "like_rate"] is np.nan or math.isnan(result.loc[0, "like_rate"])


def test_negative_likes_converted_to_nan() -> None:
    """Negative likes must produce NaN in derived metrics (raw column preserved)."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [-10.0],
        "comments": [5.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    # Raw column is preserved; derived metrics use sanitized values
    assert result.loc[0, "log_likes"] is np.nan or math.isnan(result.loc[0, "log_likes"])
    assert result.loc[0, "like_rate"] is np.nan or math.isnan(result.loc[0, "like_rate"])


def test_negative_comments_converted_to_nan() -> None:
    """Negative comments must produce NaN in derived metrics (raw column preserved)."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [-5.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "log_comments"] is np.nan or math.isnan(result.loc[0, "log_comments"])


def test_negative_followers_converted_to_nan() -> None:
    """Negative followers must produce NaN in derived metrics (raw column preserved)."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [-1000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "log_author_followers"] is np.nan or math.isnan(result.loc[0, "log_author_followers"])


# ---------------------------------------------------------------------------
# Test: Timestamp handling
# ---------------------------------------------------------------------------


def test_publish_after_collection_produces_nan_age() -> None:
    """When publish_time > collection_time, video_age_days must be NaN (not negative)."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-20T00:00:00+00:00"],
        "collection_time": ["2025-01-01T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "video_age_days"])
    # Single-row frame has index 0
    assert 0 in report.negative_video_age_rows


def test_invalid_publish_time_string() -> None:
    """Invalid publish_time string must be handled gracefully."""
    frame = pd.DataFrame({
        "publish_time": ["not-a-valid-date"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "video_age_days"])
    assert 0 in report.invalid_publish_time_rows


def test_invalid_collection_time_string() -> None:
    """Invalid collection_time string must be handled gracefully."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["invalid"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, report = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "video_age_days"])
    assert 0 in report.invalid_collection_time_rows


# ---------------------------------------------------------------------------
# Test: Determinism — video_age_days does not change with today's date
# ---------------------------------------------------------------------------


def test_video_age_days_is_deterministic(base_frame: pd.DataFrame) -> None:
    """video_age_days must be identical across multiple calls with same input."""
    result1, _ = add_propagation_metrics(base_frame)
    result2, _ = add_propagation_metrics(base_frame)
    pd.testing.assert_series_equal(
        result1["video_age_days"],
        result2["video_age_days"],
    )
    # Verify it uses collection_time, not today's date
    assert result1.loc[0, "video_age_days"] == pytest.approx(10.0)


def test_same_input_same_output() -> None:
    """Running the same input through the pipeline twice produces identical results."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result1, _ = add_propagation_metrics(frame)
    result2, _ = add_propagation_metrics(frame)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Test: No Inf values
# ---------------------------------------------------------------------------


def test_no_inf_in_derived_metrics() -> None:
    """Derived metrics must never contain +inf or -inf."""
    # Edge case: very large numbers
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-01T00:00:01+00:00"],  # 1 second = very high views_per_day
        "views": [1e12],
        "likes": [1e9],
        "comments": [1e8],
        "author_followers": [1e10],
    })
    result, _ = add_propagation_metrics(frame)
    derived_cols = list(result.columns)
    for col in derived_cols:
        if col in [
            "video_age_days", "log_views", "log_likes", "log_comments",
            "like_rate", "comment_rate", "engagement_rate",
            "views_per_day", "log_author_followers",
        ]:
            assert not math.isinf(result.loc[0, col]), f"{col} is inf"


def test_no_inf_from_zero_denominator() -> None:
    """Zero views denominator must never produce inf in rate metrics."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [0.0],
        "likes": [0.0],
        "comments": [0.0],
        "author_followers": [100.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert not math.isinf(result.loc[0, "like_rate"])
    assert not math.isinf(result.loc[0, "comment_rate"])
    assert not math.isinf(result.loc[0, "engagement_rate"])


# ---------------------------------------------------------------------------
# Test: log1p correctness
# ---------------------------------------------------------------------------


def test_log1p_correctness() -> None:
    """Verify log1p produces correct values."""
    test_cases = [
        (0.0, 0.0),
        (1.0, math.log1p(1.0)),
        (1000.0, math.log1p(1000.0)),
        (999999.0, math.log1p(999999.0)),
    ]
    for views_val, expected_log in test_cases:
        frame = pd.DataFrame({
            "publish_time": ["2025-01-01T00:00:00+00:00"],
            "collection_time": ["2025-01-11T00:00:00+00:00"],
            "views": [views_val],
            "likes": [10.0],
            "comments": [5.0],
            "author_followers": [1000.0],
        })
        result, _ = add_propagation_metrics(frame)
        assert result.loc[0, "log_views"] == pytest.approx(expected_log)


# ---------------------------------------------------------------------------
# Test: Rate correctness
# ---------------------------------------------------------------------------


def test_rate_correctness_like_rate() -> None:
    """like_rate = likes / views when views > 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [200.0],
        "likes": [10.0],
        "comments": [5.0],
        "author_followers": [1000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "like_rate"] == pytest.approx(0.05)


def test_rate_correctness_comment_rate() -> None:
    """comment_rate = comments / views when views > 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [200.0],
        "likes": [10.0],
        "comments": [5.0],
        "author_followers": [1000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "comment_rate"] == pytest.approx(0.025)


def test_rate_correctness_engagement_rate() -> None:
    """engagement_rate = (likes + comments) / views when views > 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [200.0],
        "likes": [10.0],
        "comments": [5.0],
        "author_followers": [1000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "engagement_rate"] == pytest.approx(0.075)


# ---------------------------------------------------------------------------
# Test: views_per_day edge cases
# ---------------------------------------------------------------------------


def test_views_per_day_zero_video_age() -> None:
    """When video_age_days = 0 (published and collected same day), views_per_day should be NaN."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T12:00:00+00:00"],
        "collection_time": ["2025-01-01T12:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    # video_age_days = 0, which is not > 0, so views_per_day should be NaN
    # (the implementation uses valid_age_for_rate = age.where(age > 0), so age=0 is masked out)
    assert result.loc[0, "video_age_days"] == pytest.approx(0.0)
    assert math.isnan(result.loc[0, "views_per_day"])


def test_views_per_day_fractional_days() -> None:
    """views_per_day should handle fractional video_age_days correctly."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-01T12:00:00+00:00"],  # 0.5 days
        "views": [100.0],
        "likes": [5.0],
        "comments": [2.0],
        "author_followers": [1000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "video_age_days"] == pytest.approx(0.5)
    assert result.loc[0, "views_per_day"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test: follower_normalized_views
# ---------------------------------------------------------------------------


def test_follower_normalized_views_valid() -> None:
    """follower_normalized_views = views / followers when followers > 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "follower_normalized_views"] == pytest.approx(0.1)


def test_follower_normalized_views_zero_followers() -> None:
    """follower_normalized_views must be NaN when followers = 0."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [100.0],
        "likes": [5.0],
        "comments": [2.0],
        "author_followers": [0.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "follower_normalized_views"])


def test_follower_normalized_views_missing_followers() -> None:
    """follower_normalized_views must be NaN when followers is missing."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [100.0],
        "likes": [5.0],
        "comments": [2.0],
        "author_followers": [np.nan],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "follower_normalized_views"])


# ---------------------------------------------------------------------------
# Test: Optional metrics
# ---------------------------------------------------------------------------


def test_optional_metrics_disabled_by_default() -> None:
    """Optional metrics must NOT be added unless include_optional=True."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame, include_optional=False)
    assert "log_views_per_day" not in result.columns
    assert "log_engagement_count" not in result.columns
    assert "log_views_per_follower" not in result.columns


def test_optional_metrics_enabled() -> None:
    """Optional metrics must be added when include_optional=True."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame, include_optional=True)
    assert "log_views_per_day" in result.columns
    assert "log_engagement_count" in result.columns
    assert "log_views_per_follower" in result.columns
    # Verify values
    assert result.loc[0, "log_views_per_day"] == pytest.approx(math.log1p(100.0))
    assert result.loc[0, "log_engagement_count"] == pytest.approx(math.log1p(60.0))
    assert result.loc[0, "log_views_per_follower"] == pytest.approx(math.log1p(0.1))


# ---------------------------------------------------------------------------
# Test: Report fields
# ---------------------------------------------------------------------------


def test_report_contains_all_fields() -> None:
    """Report must contain all required QC fields."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    _, report = add_propagation_metrics(frame)
    assert isinstance(report, PropagationMetricReport)
    assert report.rows == 1
    assert report.successful_derived_rows >= 0
    assert report.success_rate >= 0.0
    assert isinstance(report.missing_data, MissingDataSummary)
    assert isinstance(report.invalid_publish_time_rows, tuple)
    assert isinstance(report.invalid_collection_time_rows, tuple)
    assert isinstance(report.negative_video_age_rows, tuple)


def test_report_missing_data_counts() -> None:
    """MissingDataSummary must count all missing/zero/negative patterns."""
    frame = pd.DataFrame({
        "publish_time": [np.nan],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [np.nan],
        "likes": [0.0],
        "comments": [-5.0],  # negative values are converted to NaN, not counted as negative
        "author_followers": [0.0],
    })
    _, report = add_propagation_metrics(frame)
    assert report.missing_data.views_missing == 1
    assert report.missing_data.views_zero == 0
    assert report.missing_data.views_negative == 0
    assert report.missing_data.likes_missing == 0
    assert report.missing_data.likes_zero == 1
    # Negative values in source are converted to NaN (not left as negative)
    # So comments_negative = 0, comments_missing = 1
    assert report.missing_data.comments_missing == 1
    assert report.missing_data.comments_negative == 0
    assert report.missing_data.followers_missing == 0
    assert report.missing_data.followers_zero == 1
    assert report.missing_data.publish_time_invalid == 1


# ---------------------------------------------------------------------------
# Test: Extreme values
# ---------------------------------------------------------------------------


def test_extreme_values_flagged() -> None:
    """Rows with extreme views_per_day must be flagged in the report."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-01T00:00:01+00:00"],  # 1 second = huge views_per_day
        "views": [1e12],
        "likes": [1e9],
        "comments": [1e8],
        "author_followers": [1e10],
    })
    _, report = add_propagation_metrics(
        frame, extreme_views_per_day_threshold=1_000_000.0
    )
    assert len(report.extreme_views_per_day_rows) == 1


def test_engagement_rate_exceeding_threshold_flagged() -> None:
    """Rows with engagement_rate > 1.0 must be flagged."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [10.0],
        "likes": [20.0],
        "comments": [5.0],
        "author_followers": [1000.0],
    })
    result, report = add_propagation_metrics(
        frame, extreme_engagement_rate_threshold=1.0
    )
    assert len(report.extreme_engagement_rate_rows) == 1
    assert result.loc[0, "engagement_rate"] == pytest.approx(2.5)  # 25/10


# ---------------------------------------------------------------------------
# Test: Output schema
# ---------------------------------------------------------------------------


def test_output_contains_required_derived_columns() -> None:
    """Output DataFrame must contain all required derived columns."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    required_cols = [
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
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_raw_columns_preserved() -> None:
    """Original raw columns must be preserved in the output."""
    frame = pd.DataFrame({
        "video_id": ["test001"],
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert "video_id" in result.columns
    assert "views" in result.columns
    assert "likes" in result.columns
    assert "comments" in result.columns


# ---------------------------------------------------------------------------
# Test: Multiple rows
# ---------------------------------------------------------------------------


def test_multiple_rows() -> None:
    """Pipeline must handle multiple rows correctly."""
    frame = pd.DataFrame({
        "publish_time": [
            "2025-01-01T00:00:00+00:00",
            "2025-01-02T00:00:00+00:00",
            "2025-01-03T00:00:00+00:00",
        ],
        "collection_time": ["2025-01-11T00:00:00+00:00"] * 3,
        "views": [1000.0, 0.0, np.nan],
        "likes": [50.0, 0.0, 5.0],
        "comments": [10.0, 0.0, 2.0],
        "author_followers": [10000.0, 1000.0, 100.0],
    })
    result, report = add_propagation_metrics(frame)
    assert len(result) == 3
    assert report.rows == 3
    # Row 0: all valid
    assert result.loc[0, "log_views"] == pytest.approx(math.log1p(1000.0))
    # Row 1: views = 0
    assert result.loc[1, "log_views"] == pytest.approx(0.0)
    assert math.isnan(result.loc[1, "like_rate"])
    # Row 2: views missing
    assert math.isnan(result.loc[2, "log_views"])
    assert math.isnan(result.loc[2, "like_rate"])


# ---------------------------------------------------------------------------
# Test: No silent missing-to-zero conversion
# ---------------------------------------------------------------------------


def test_no_silent_missing_to_zero_conversion() -> None:
    """Missing values must NOT be silently converted to 0."""
    frame = pd.DataFrame({
        "publish_time": [np.nan],  # Invalid timestamp
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [np.nan],
        "likes": [np.nan],
        "comments": [np.nan],
        "author_followers": [np.nan],
    })
    result, _ = add_propagation_metrics(frame)
    # All derived metrics must be NaN, NOT 0
    derived = [
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
    ]
    for col in derived:
        assert math.isnan(result.loc[0, col]), f"{col} should be NaN, not 0"


# ---------------------------------------------------------------------------
# Test: Engagement rate with missing likes or comments
# ---------------------------------------------------------------------------


def test_engagement_rate_partial_missing() -> None:
    """engagement_rate must be NaN when likes OR comments is missing (not 0)."""
    # Missing likes
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [np.nan],
        "comments": [10.0],
        "author_followers": [10000.0],
    })
    result, _ = add_propagation_metrics(frame)
    assert math.isnan(result.loc[0, "engagement_rate"])

    # Missing comments
    frame2 = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": [1000.0],
        "likes": [50.0],
        "comments": [np.nan],
        "author_followers": [10000.0],
    })
    result2, _ = add_propagation_metrics(frame2)
    assert math.isnan(result2.loc[0, "engagement_rate"])


# ---------------------------------------------------------------------------
# Test: String numeric values
# ---------------------------------------------------------------------------


def test_string_numeric_values() -> None:
    """String numeric values in source columns must be parsed correctly."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": ["1000"],  # string number
        "likes": ["50"],
        "comments": ["10"],
        "author_followers": ["10000"],
    })
    result, _ = add_propagation_metrics(frame)
    assert result.loc[0, "log_views"] == pytest.approx(math.log1p(1000.0))
    assert result.loc[0, "like_rate"] == pytest.approx(0.05)


def test_non_numeric_string_converted_to_nan() -> None:
    """Non-numeric strings in source columns must produce NaN in derived metrics."""
    frame = pd.DataFrame({
        "publish_time": ["2025-01-01T00:00:00+00:00"],
        "collection_time": ["2025-01-11T00:00:00+00:00"],
        "views": ["not_a_number"],
        "likes": ["50"],
        "comments": ["10"],
        "author_followers": ["10000"],
    })
    result, _ = add_propagation_metrics(frame)
    # Raw column is preserved as string; derived metric uses sanitized value
    assert result.loc[0, "views"] == "not_a_number"
    assert math.isnan(result.loc[0, "log_views"])
    # Other valid string numerics should work
    assert result.loc[0, "log_likes"] == pytest.approx(math.log1p(50.0))


# ---------------------------------------------------------------------------
# Test: MissingDataSummary dataclass immutability
# ---------------------------------------------------------------------------


def test_missing_data_summary_immutable() -> None:
    """MissingDataSummary must be frozen (immutable)."""
    summary = MissingDataSummary(
        views_missing=0,
        views_zero=0,
        views_negative=0,
        likes_missing=0,
        likes_zero=0,
        likes_negative=0,
        comments_missing=0,
        comments_zero=0,
        comments_negative=0,
        followers_missing=0,
        followers_zero=0,
        followers_negative=0,
        publish_time_invalid=0,
        collection_time_invalid=0,
        video_age_negative=0,
    )
    with pytest.raises(AttributeError):
        summary.views_missing = 5  # type: ignore


# ---------------------------------------------------------------------------
# Test: PropagationMetricReport dataclass immutability
# ---------------------------------------------------------------------------


def test_propagation_metric_report_immutable() -> None:
    """PropagationMetricReport must be frozen (immutable)."""
    from src.preprocessing.propagation_metrics_v1 import MissingDataSummary

    summary = MissingDataSummary(
        views_missing=0,
        views_zero=0,
        views_negative=0,
        likes_missing=0,
        likes_zero=0,
        likes_negative=0,
        comments_missing=0,
        comments_zero=0,
        comments_negative=0,
        followers_missing=0,
        followers_zero=0,
        followers_negative=0,
        publish_time_invalid=0,
        collection_time_invalid=0,
        video_age_negative=0,
    )
    report = PropagationMetricReport(
        rows=1,
        successful_derived_rows=1,
        success_rate=1.0,
        all_required_derived_present=1,
        all_required_derived_present_rate=1.0,
        invalid_publish_time_rows=(),
        invalid_collection_time_rows=(),
        negative_video_age_rows=(),
        missing_data=summary,
        extreme_views_per_day_rows=(),
        extreme_engagement_rate_rows=(),
    )
    with pytest.raises(AttributeError):
        report.rows = 10  # type: ignore
