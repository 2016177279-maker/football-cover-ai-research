"""Econometric Specification v2 — Model Skeleton.

STATUS: PREPARATION_ONLY — DO NOT EXECUTE
ACTIVATION: After FINAL_DATASET_V2_12M freeze

This module defines the explicit model specifications M0–M5
as frozen execution contracts. No regression is run here.

CROSS-REFERENCE: reports/statistical_model_specification_v1.md
  - This module extends the v1 specification for Master Dataset v2
  - Does NOT overwrite an already-frozen specification
  - The v1 statistical_model_specification_v1.md remains authoritative for v1 datasets

MODEL SPECIFICATIONS
------------------
M0 : Descriptive / Univariate (single visual predictor, exploratory)
M1 : Visual-only (all Basic CV features, no controls)
M2 : + Author controls (log_subscriber_count)
M3 : + Topic fixed effects
M4 : + Temporal controls (publish_month, video_age_days)
M5 : Full specification (primary inference model)

PRIMARY CLUSTERING
------------------
cluster_robust = channel_id

This is the PRIMARY inference requirement.
HC1 clustered by channel_id.

ROBUSTNESS HOOKS
----------------
- Alternative Y variables
- Alternative visual subsets
- Missingness sensitivity
- Winsorization sensitivity (if approved)

HETEROGENEITY HOOKS
--------------------
- By topic
- By author_tier (long_tail, mid, head)
- By publish period

FIELD NAME MAPPING
-------------------
This specification uses Master Dataset v2 field names:
  - view_count      (not views)
  - like_count      (not likes)
  - comment_count   (not comments)
  - subscriber_count (not author_followers)
  - log_subscriber_count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Frozen model hierarchy
# ---------------------------------------------------------------------------

MODEL_NAMES: tuple[str, ...] = (
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
)


# ---------------------------------------------------------------------------
# Y Variable Specification
# ---------------------------------------------------------------------------

class YVariable:
    """Specification for a dependent variable (Y)."""

    def __init__(
        self,
        name: str,
        field_name: str,
        description: str,
        is_primary: bool = False,
        is_logged: bool = True,
    ):
        self.name = name
        self.field_name = field_name
        self.description = description
        self.is_primary = is_primary
        self.is_logged = is_logged

    def __repr__(self) -> str:
        return f"YVariable({self.name}, primary={self.is_primary})"


Y_VARIABLES: dict[str, YVariable] = {
    "log_views": YVariable(
        name="log_views",
        field_name="log_view_count",
        description="log1p(view_count) — cumulative exposure proxy",
        is_primary=True,
        is_logged=True,
    ),
    "views_per_day": YVariable(
        name="views_per_day",
        field_name="views_per_day",
        description="view_count / video_age_days — time-normalized velocity",
        is_primary=True,
        is_logged=False,
    ),
    "engagement": YVariable(
        name="engagement",
        field_name="engagement",
        description="(like_count + comment_count) / view_count — interaction efficiency",
        is_primary=True,
        is_logged=False,
    ),
    "log_likes": YVariable(
        name="log_likes",
        field_name="log_like_count",
        description="log1p(like_count) — secondary engagement",
        is_primary=False,
        is_logged=True,
    ),
    "log_comments": YVariable(
        name="log_comments",
        field_name="log_comment_count",
        description="log1p(comment_count) — secondary engagement",
        is_primary=False,
        is_logged=True,
    ),
    "likes_per_1000_views": YVariable(
        name="likes_per_1000_views",
        field_name="likes_per_1000_views",
        description="like_count / view_count × 1000 — normalized like intensity",
        is_primary=False,
        is_logged=False,
    ),
    "comments_per_1000_views": YVariable(
        name="comments_per_1000_views",
        field_name="comments_per_1000_views",
        description="comment_count / view_count × 1000 — normalized comment intensity",
        is_primary=False,
        is_logged=False,
    ),
}

# Primary Y candidates
PRIMARY_Y = {k: v for k, v in Y_VARIABLES.items() if v.is_primary}
# Secondary Y (robustness only)
SECONDARY_Y = {k: v for k, v in Y_VARIABLES.items() if not v.is_primary}


# ---------------------------------------------------------------------------
# Visual Predictor Specification
# ---------------------------------------------------------------------------

# Basic CV features (21 fields)
BASIC_CV_FEATURES: tuple[str, ...] = (
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "colorfulness",
    "contrast",
    "sharpness_laplacian",
    "edge_density",
    "entropy",
    "center_brightness",
    "border_brightness",
    "center_edge_density",
    "dominant_color_r",
    "dominant_color_g",
    "dominant_color_b",
    "warm_color_ratio",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
)

# Advanced CV scalar features (from Advanced CV v1)
ADVANCED_CV_AESTHETIC: tuple[str, ...] = (
    "aesthetic_score",
)

ADVANCED_CV_SALIENCY: tuple[str, ...] = (
    "saliency_concentration_top10",
    "saliency_center_share",
    "saliency_peak_region_share",
    "saliency_subject_share",
    "saliency_background_share",
)

ADVANCED_CV_TEXT: tuple[str, ...] = (
    "text_region_count",
    "text_area_ratio",
    "text_largest_region_area_ratio",
    "text_subject_overlap_ratio",
)

ADVANCED_CV_GEOMETRY: tuple[str, ...] = (
    "face_count",
    "face_area_ratio",
    "largest_face_area_ratio",
    "person_count",
    "person_area_ratio",
    "largest_person_area_ratio",
    "subject_count",
    "subject_area_ratio",
    "largest_subject_area_ratio",
)

ALL_ADVANCED_CV_SCALAR: tuple[str, ...] = (
    *ADVANCED_CV_AESTHETIC,
    *ADVANCED_CV_SALIENCY,
    *ADVANCED_CV_TEXT,
    *ADVANCED_CV_GEOMETRY,
)


# ---------------------------------------------------------------------------
# Control Variable Specification
# ---------------------------------------------------------------------------

# Subscriber controls
SUBSCRIBER_CONTROLS: tuple[str, ...] = (
    "log_subscriber_count",
)

# Temporal controls
TEMPORAL_CONTROLS: tuple[str, ...] = (
    "publish_month",
    "publish_weekday",
    "video_age_days",
)

# Content controls
TOPIC_CONTROL = "topic"

# Author-size stratum
AUTHOR_TIER_CONTROL = "author_tier"

# Query lineage
QUERY_LINEAGE_CONTROL = "query_keyword"

# Channel ID (for clustering)
CHANNEL_ID_VAR = "channel_id"

# All control variables
ALL_CONTROLS: tuple[str, ...] = (
    *SUBSCRIBER_CONTROLS,
    *TEMPORAL_CONTROLS,
    TOPIC_CONTROL,
    AUTHOR_TIER_CONTROL,
)

# Categorical controls (need dummy encoding)
CATEGORICAL_CONTROLS: tuple[str, ...] = (
    "publish_month",
    "publish_weekday",
    "topic",
    "author_tier",
)

# Continuous controls
CONTINUOUS_CONTROLS: tuple[str, ...] = (
    "log_subscriber_count",
    "video_age_days",
)


# ---------------------------------------------------------------------------
# Reference Groups (Deterministic)
# ---------------------------------------------------------------------------

REFERENCE_GROUPS: dict[str, str] = {
    "topic": "club",
    "publish_month": "1",    # January
    "publish_weekday": "0",   # Monday
    "author_tier": "long_tail",
    "person_present": "unknown",
    "text_present": "false",
    "scene_type": "unknown",
    "action_cue": "unknown",
}


# ---------------------------------------------------------------------------
# Model Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Frozen model specification."""

    name: str
    description: str
    y_vars: list[str]  # Y variable names
    visual_predictors: list[str]  # Field names for visual features
    author_controls: list[str]  # Author/scale controls
    topic_controls: list[str]  # Topic fixed effects (or empty)
    temporal_controls: list[str]  # Temporal controls (or empty)
    author_tier_control: bool  # Include author_tier?
    cluster_by: str  # Clustering variable
    se_type: str  # Standard error type
    notes: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "M0": ModelSpec(
        name="M0",
        description=(
            "Descriptive / Univariate. "
            "Single visual predictor vs Y. "
            "Exploratory only; no inference claims."
        ),
        y_vars=["log_views"],
        visual_predictors=["brightness_mean"],  # Single representative feature
        author_controls=[],
        topic_controls=[],
        temporal_controls=[],
        author_tier_control=False,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Exploratory; run for each visual feature individually",
    ),

    "M1": ModelSpec(
        name="M1",
        description=(
            "Visual-only. "
            "All Basic CV features as predictors. "
            "No controls. Tests visual features in isolation."
        ),
        y_vars=["log_views"],
        visual_predictors=list(BASIC_CV_FEATURES),
        author_controls=[],
        topic_controls=[],
        temporal_controls=[],
        author_tier_control=False,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Reports R², individual coefficients, VIF",
    ),

    "M2": ModelSpec(
        name="M2",
        description=(
            "Visual + Author Controls. "
            "M1 + log_subscriber_count. "
            "Tests if visual effects survive author-scale confound."
        ),
        y_vars=["log_views"],
        visual_predictors=list(BASIC_CV_FEATURES),
        author_controls=list(SUBSCRIBER_CONTROLS),
        topic_controls=[],
        temporal_controls=[],
        author_tier_control=False,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Primary comparison: M1 vs M2",
    ),

    "M3": ModelSpec(
        name="M3",
        description=(
            "Visual + Author + Topic. "
            "M2 + topic fixed effects. "
            "Tests content heterogeneity."
        ),
        y_vars=["log_views"],
        visual_predictors=list(BASIC_CV_FEATURES),
        author_controls=list(SUBSCRIBER_CONTROLS),
        topic_controls=[TOPIC_CONTROL],
        temporal_controls=[],
        author_tier_control=False,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Reference group for topic: 'club'",
    ),

    "M4": ModelSpec(
        name="M4",
        description=(
            "Visual + Author + Topic + Temporal. "
            "M3 + publish_month + publish_weekday + video_age_days. "
            "Tests temporal patterns."
        ),
        y_vars=["log_views"],
        visual_predictors=list(BASIC_CV_FEATURES),
        author_controls=list(SUBSCRIBER_CONTROLS),
        topic_controls=[TOPIC_CONTROL],
        temporal_controls=list(TEMPORAL_CONTROLS),
        author_tier_control=False,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Reference groups: month='1', weekday='0'",
    ),

    "M5": ModelSpec(
        name="M5",
        description=(
            "Full Specification (Primary Inference Model). "
            "M4 + author_tier fixed effects. "
            "Full visual + all approved controls."
        ),
        y_vars=["log_views"],
        visual_predictors=list(BASIC_CV_FEATURES),
        author_controls=list(SUBSCRIBER_CONTROLS),
        topic_controls=[TOPIC_CONTROL],
        temporal_controls=list(TEMPORAL_CONTROLS),
        author_tier_control=True,
        cluster_by=CHANNEL_ID_VAR,
        se_type="HC1_clustered",
        notes="Primary inference model. Reference: author_tier='long_tail'",
    ),
}


# ---------------------------------------------------------------------------
# Standard Error Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SESpec:
    """Standard error specification."""

    name: str
    type: str  # HC0, HC1, HC3, clustered, two_way
    cluster_vars: tuple[str, ...] = ()
    min_clusters: int = 50
    notes: str = ""


SE_SPECS: dict[str, SESpec] = {
    "HC1_clustered": SESpec(
        name="HC1_clustered",
        type="HC1",
        cluster_vars=(CHANNEL_ID_VAR,),
        min_clusters=50,
        notes="Primary: HC1 with channel_id clustering",
    ),
    "HC0_clustered": SESpec(
        name="HC0_clustered",
        type="HC0",
        cluster_vars=(CHANNEL_ID_VAR,),
        min_clusters=50,
        notes="Alternative: HC0 without finite-sample correction",
    ),
    "HC3_clustered": SESpec(
        name="HC3_clustered",
        type="HC3",
        cluster_vars=(CHANNEL_ID_VAR,),
        min_clusters=50,
        notes="Sensitivity: more conservative",
    ),
    "clustered_topic": SESpec(
        name="clustered_topic",
        type="HC1",
        cluster_vars=("topic",),
        min_clusters=6,
        notes="Alternative clustering by topic (for robustness)",
    ),
    "two_way_channel_month": SESpec(
        name="two_way_channel_month",
        type="two_way_cluster",
        cluster_vars=(CHANNEL_ID_VAR, "publish_month"),
        min_clusters=50,
        notes="Two-way clustering: channel × month",
    ),
}

PRIMARY_SE_SPEC = SE_SPECS["HC1_clustered"]


# ---------------------------------------------------------------------------
# Robustness Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustnessSpec:
    """Robustness check specification."""

    name: str
    description: str
    changes: dict[str, object]  # What changes relative to M5
    expected_effect: str  # "should be similar" or "expected difference"


ROBUSTNESS_SPECS: dict[str, RobustnessSpec] = {
    "alternative_y_log_likes": RobustnessSpec(
        name="alternative_y_log_likes",
        description="Use log_like_count as Y instead of log_view_count",
        changes={"y_vars": ["log_likes"]},
        expected_effect="Coefficient direction should be consistent",
    ),
    "alternative_y_engagement": RobustnessSpec(
        name="alternative_y_engagement",
        description="Use engagement as Y instead of log_view_count",
        changes={"y_vars": ["engagement"]},
        expected_effect="Different scale; effect directions comparable",
    ),
    "advanced_cv_only": RobustnessSpec(
        name="advanced_cv_only",
        description="Use only Advanced CV features as visual predictors",
        changes={
            "visual_predictors": list(ALL_ADVANCED_CV_SCALAR),
            "drop_basic_cv": True,
        },
        expected_effect="Different features; test robustness of visual signal",
    ),
    "basic_cv_subset": RobustnessSpec(
        name="basic_cv_subset",
        description="Use only brightness, saturation, contrast features",
        changes={
            "visual_predictors": [
                "brightness_mean", "saturation_mean", "contrast",
                "brightness_std", "saturation_std",
            ],
            "drop_other_basic_cv": True,
        },
        expected_effect="Coefficients should remain significant",
    ),
    "without_temporal_controls": RobustnessSpec(
        name="without_temporal_controls",
        description="M5 without temporal controls (M3 baseline)",
        changes={
            "temporal_controls": [],
        },
        expected_effect="Larger SEs expected; coefficients may shift",
    ),
    "without_author_controls": RobustnessSpec(
        name="without_author_controls",
        description="M5 without subscriber controls",
        changes={
            "author_controls": [],
        },
        expected_effect="Coefficients may change due to omitted variable bias",
    ),
    "without_topic_controls": RobustnessSpec(
        name="without_topic_controls",
        description="M4 without topic fixed effects",
        changes={
            "topic_controls": [],
        },
        expected_effect="Coefficients may shift; test topic confound",
    ),
    "winsorized_y": RobustnessSpec(
        name="winsorized_y",
        description="Winsorize Y at 1st/99th percentile",
        changes={
            "winsorize_y": True,
            "winsorize_pct": (1, 99),
        },
        expected_effect="Should be similar to main results",
    ),
    "missingness_complete_case": RobustnessSpec(
        name="missingness_complete_case",
        description="Complete-case analysis (drop rows with missing features)",
        changes={
            "complete_case_only": True,
        },
        expected_effect="Compare with main results; assess selection",
    ),
}


# ---------------------------------------------------------------------------
# Heterogeneity Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeterogeneitySpec:
    """Heterogeneity / subgroup analysis specification."""

    name: str
    description: str
    subgroup_by: str  # Variable name
    subgroups: dict[str, str]  # value → label
    model_base: str  # Base model (M5)
    notes: str = ""


HETEROGENEITY_SPECS: dict[str, HeterogeneitySpec] = {
    "by_topic": HeterogeneitySpec(
        name="by_topic",
        description="Run M5 separately for each topic",
        subgroup_by="topic",
        subgroups={
            "club": "Club content",
            "competition": "Competition content",
            "core_football_content": "Core football",
            "entertainment_derivative": "Entertainment derivative",
            "multilingual": "Multilingual",
            "player": "Player content",
        },
        model_base="M5",
        notes="Small-N subgroups may have unreliable SEs",
    ),
    "by_author_tier": HeterogeneitySpec(
        name="by_author_tier",
        description="Run M5 separately for each author tier",
        subgroup_by="author_tier",
        subgroups={
            "long_tail": "Long-tail channels (≤P33)",
            "mid": "Mid-tier channels (P33–P67)",
            "head": "Head channels (>P67)",
        },
        model_base="M5",
        notes="Tests if visual effects differ by channel size",
    ),
    "by_publish_period": HeterogeneitySpec(
        name="by_publish_period",
        description="Run M5 separately for pre/post 2026-01",
        subgroup_by="publish_period",
        subgroups={
            "pre_2026": "Published before 2026-01",
            "post_2026": "Published 2026-01 onwards",
        },
        model_base="M5",
        notes="Temporal heterogeneity test",
    ),
    "by_video_age": HeterogeneitySpec(
        name="by_video_age",
        description="Run M5 separately for new vs old videos",
        subgroup_by="video_age_bucket",
        subgroups={
            "new": "video_age_days ≤ 90",
            "medium": "90 < video_age_days ≤ 180",
            "old": "video_age_days > 180",
        },
        model_base="M5",
        notes="Tests age-dependent visual effects",
    ),
}


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def get_model_predictors(model_name: str) -> list[str]:
    """Return all predictor variable names for a model."""
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}")

    predictors: list[str] = []
    predictors.extend(spec.visual_predictors)
    predictors.extend(spec.author_controls)
    predictors.extend(spec.topic_controls)
    predictors.extend(spec.temporal_controls)
    if spec.author_tier_control:
        predictors.append(AUTHOR_TIER_CONTROL)
    return predictors


def get_model_formula_components(model_name: str) -> dict[str, list[str]]:
    """Return formula components for a model (for documentation)."""
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name}")

    return {
        "visual": spec.visual_predictors,
        "author_controls": spec.author_controls,
        "topic_controls": spec.topic_controls,
        "temporal_controls": spec.temporal_controls,
        "author_tier": [AUTHOR_TIER_CONTROL] if spec.author_tier_control else [],
        "cluster": [spec.cluster_by],
    }


def get_cluster_var() -> str:
    """Return the primary clustering variable."""
    return CHANNEL_ID_VAR


def get_se_spec() -> SESpec:
    """Return the primary SE specification."""
    return PRIMARY_SE_SPEC


def model_summary() -> dict[str, dict]:
    """Return a human-readable summary of all model specifications."""
    return {
        name: {
            "description": spec.description,
            "n_predictors": (
                len(spec.visual_predictors)
                + len(spec.author_controls)
                + len(spec.topic_controls)
                + len(spec.temporal_controls)
                + (1 if spec.author_tier_control else 0)
            ),
            "n_visual": len(spec.visual_predictors),
            "cluster_by": spec.cluster_by,
            "se_type": spec.se_type,
            "notes": spec.notes,
        }
        for name, spec in MODEL_SPECS.items()
    }
