"""Propagation modeling pipeline skeleton v1.

This module defines model specifications and validates model inputs.  It is
deliberately not a research analysis: importing it or calling :func:`dry_run`
does not fit a model, calculate significance, or produce feature explanations.
Optional modeling libraries are imported only when their adapters are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


PIPELINE_WARNING = "PIPELINE SMOKE TEST ONLY - NOT FOR RESEARCH INTERPRETATION"

PRIMARY_TARGET_CANDIDATES = ("log_views", "views_per_day", "engagement_rate")
SECONDARY_TARGET_CANDIDATES = ("like_rate", "comment_rate", "log_likes", "log_comments")
TARGET_CANDIDATES = PRIMARY_TARGET_CANDIDATES + SECONDARY_TARGET_CANDIDATES

DEFAULT_CONTROLS = (
    "log_author_followers",
    "video_age_days",
    "content_subtype",
    "publish_month",
    "publish_weekday",
    "publish_hour",
    "query_keyword",
)
DEFAULT_CATEGORICAL_CONTROLS = (
    "content_subtype", "publish_month", "publish_weekday", "publish_hour", "query_keyword"
)


@dataclass(frozen=True)
class ModelSpecification:
    """Declarative specification for one future propagation model."""

    name: str
    target: str
    controls: tuple[str, ...]
    visual_features: tuple[str, ...] = ()
    estimator_family: str = "ols"
    group_column: str = "channel_id"

    @property
    def predictors(self) -> tuple[str, ...]:
        """Return predictors once, preserving their declared order."""
        return tuple(dict.fromkeys((*self.controls, *self.visual_features)))


def default_model_specifications(
    visual_features: Sequence[str], target: str = "log_views"
) -> tuple[ModelSpecification, ...]:
    """Create the three model specifications required by pipeline v1."""
    if target not in TARGET_CANDIDATES:
        raise ValueError(f"Unsupported candidate target: {target}")
    visual = tuple(visual_features)
    return (
        ModelSpecification("baseline_ols", target, DEFAULT_CONTROLS),
        ModelSpecification("visual_plus_controls_ols", target, DEFAULT_CONTROLS, visual),
        ModelSpecification("tree_based_predictive", target, DEFAULT_CONTROLS, visual, "tree"),
    )


def add_publish_time_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive auditable UTC calendar controls without modifying the input."""
    if "publish_time" not in frame:
        raise ValueError("Missing modeling column: publish_time")
    result = frame.copy()
    publish = pd.to_datetime(result["publish_time"], errors="coerce", utc=True, format="mixed")
    result["publish_month"] = publish.dt.month.astype("Int64")
    result["publish_weekday"] = publish.dt.weekday.astype("Int64")
    result["publish_hour"] = publish.dt.hour.astype("Int64")
    return result


def validate_model_data(frame: pd.DataFrame, specification: ModelSpecification) -> None:
    """Validate required columns and reject duplicated predictor declarations."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    required = {specification.target, *specification.predictors}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing modeling columns: {', '.join(missing)}")
    if specification.target in specification.predictors:
        raise ValueError("Target must not also be used as a predictor")


def prepare_design_matrix(
    frame: pd.DataFrame,
    specification: ModelSpecification,
    *,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_CONTROLS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare complete-case numeric X/y without fitting or interpreting a model.

    Categorical controls are one-hot encoded with an explicit reference level.
    Rows with missing or non-finite model inputs are retained as missing until
    the final complete-case selection, making the dropped-row count auditable.
    """
    validate_model_data(frame, specification)
    columns = [specification.target, *specification.predictors]
    working = frame.loc[:, columns].copy()
    categorical = [column for column in categorical_columns if column in specification.predictors]
    numeric = [column for column in specification.predictors if column not in categorical]

    working[specification.target] = pd.to_numeric(working[specification.target], errors="coerce")
    for column in numeric:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)

    complete_inputs = working[[specification.target, *specification.predictors]].notna().all(axis=1)
    x = pd.get_dummies(working[list(specification.predictors)], columns=categorical, drop_first=True, dtype=float)
    y = working[specification.target]
    complete = complete_inputs & x.notna().all(axis=1)
    return x.loc[complete].astype(float), y.loc[complete].astype(float)


def fit_ols(
    frame: pd.DataFrame,
    specification: ModelSpecification,
    *,
    covariance_type: str = "HC3",
) -> Any:
    """Fit an OLS adapter when optional ``statsmodels`` is available.

    Returned results are un-interpreted. Formal research should additionally
    consider channel-clustered standard errors and the validation protocol.
    """
    if specification.estimator_family != "ols":
        raise ValueError("fit_ols requires an OLS specification")
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - dependency intentionally optional
        raise RuntimeError("OLS fitting requires the optional 'statsmodels' package") from exc
    x, y = prepare_design_matrix(frame, specification)
    if x.empty:
        raise ValueError("No complete rows are available for OLS fitting")
    return sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type=covariance_type)


def fit_tree_model(frame: pd.DataFrame, specification: ModelSpecification, estimator: Any) -> Any:
    """Fit a caller-provided tree estimator implementing the sklearn-style API."""
    if specification.estimator_family != "tree":
        raise ValueError("fit_tree_model requires a tree specification")
    if not hasattr(estimator, "fit"):
        raise TypeError("estimator must provide a fit(X, y) method")
    x, y = prepare_design_matrix(frame, specification)
    if x.empty:
        raise ValueError("No complete rows are available for tree-model fitting")
    return estimator.fit(x, y)


def explain_model(*_: Any, **__: Any) -> None:
    """Reserved interface for future SHAP/feature-importance analysis."""
    raise NotImplementedError("SHAP/feature importance is reserved for the formal >=5000 sample")


def dry_run(frame: pd.DataFrame, specification: ModelSpecification) -> dict[str, Any]:
    """Validate and prepare inputs, but never fit a model."""
    x, y = prepare_design_matrix(frame, specification)
    return {
        "status": PIPELINE_WARNING,
        "model_name": specification.name,
        "target": specification.target,
        "input_rows": int(len(frame)),
        "usable_rows": int(len(y)),
        "dropped_rows": int(len(frame) - len(y)),
        "design_columns": list(x.columns),
        "model_fitted": False,
    }


__all__ = [
    "DEFAULT_CONTROLS",
    "ModelSpecification",
    "PIPELINE_WARNING",
    "PRIMARY_TARGET_CANDIDATES",
    "SECONDARY_TARGET_CANDIDATES",
    "TARGET_CANDIDATES",
    "default_model_specifications",
    "add_publish_time_controls",
    "dry_run",
    "explain_model",
    "fit_ols",
    "fit_tree_model",
    "prepare_design_matrix",
    "validate_model_data",
]
