"""Master Dataset v2 Schema Contract.

This module defines the canonical field structure for the Master Dataset v2,
the input to the Econometric Pipeline.

STATUS: PREPARATION_ONLY — NOT FINAL
ACTIVATION: After FINAL_DATASET_V2_12M freeze

KEY PRINCIPLES
--------------
1. One row = one FINAL V2 video_id (primary key)
2. NaN means unknown/missing; NEVER convert to zero
3. subscriber_count is a collection-time snapshot only
4. CLIP 512-dimensional embeddings are NOT in this CSV
5. Outcomes are derived deterministically; propagation_data_available = True
   does NOT mean outcomes are positive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

DATASET_VERSION = "master_dataset_v2"
SCHEMA_VERSION = "master_dataset_v2_v1"


# ---------------------------------------------------------------------------
# Field Groups (ordered by pipeline stage)
# ---------------------------------------------------------------------------

# 1. IDENTIFIERS / LINEAGE
IDENTIFIER_FIELDS: tuple[str, ...] = (
    "video_id",
    "source_platform",
    "collection_batch",
    "collection_time",
    "dataset_version",
    "schema_version",
)

# 2. SAMPLING VARIABLES
SAMPLING_FIELDS: tuple[str, ...] = (
    "month_bucket",
    "topic",
    "query_keyword",
    "publish_time",
    "title",
    "description_snippet",
)

# 3. ELIGIBILITY VARIABLES
ELIGIBILITY_FIELDS: tuple[str, ...] = (
    "channel_id",
    "author_tier",
    "football_final_status",
    "relevance_decision",
    "decision_source",
)

# 4. PROPAGATION OUTCOMES
OUTCOME_FIELDS: tuple[str, ...] = (
    "view_count",
    "like_count",
    "comment_count",
    "shares",   # NOT_AVAILABLE for YouTube
    "saves",    # NOT_AVAILABLE for YouTube
)

# 5. CHANNEL CONTROLS
CHANNEL_CONTROL_FIELDS: tuple[str, ...] = (
    "channel_title",
    "subscriber_count",
    "log_subscriber_count",
)

# 6. DERIVED OUTCOMES (computed after propagation outcomes)
DERIVED_OUTCOME_FIELDS: tuple[str, ...] = (
    "video_age_days",
    "log_view_count",
    "log_like_count",
    "log_comment_count",
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement",
)

# 7. BASIC CV
BASIC_CV_LINEAGE_FIELDS: tuple[str, ...] = (
    "basic_cv_version",
    "basic_cv_source_path",
)

BASIC_CV_IMAGE_FIELDS: tuple[str, ...] = (
    "image_path",
    "image_width",
    "image_height",
)

BASIC_CV_ADMIN_FIELDS: tuple[str, ...] = (
    "feature_extraction_status",
    "feature_extraction_error",
)

BASIC_CV_FEATURE_FIELDS: tuple[str, ...] = (
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

# 8. ADVANCED CV
ADVANCED_CV_LINEAGE_FIELDS: tuple[str, ...] = (
    "advanced_cv_schema_version",
    "advanced_cv_extractor_version",
    "advanced_cv_source_path",
    "advanced_cv_run_id",
    "advanced_cv_extracted_at_utc",
)

ADVANCED_CV_ADMIN_FIELDS: tuple[str, ...] = (
    "advanced_cv_status",
    "advanced_cv_error",
)

ADVANCED_CV_AESTHETIC_FIELDS: tuple[str, ...] = (
    "aesthetic_score",
)

ADVANCED_CV_SALIENCY_FIELDS: tuple[str, ...] = (
    "saliency_concentration_top10",
    "saliency_center_share",
    "saliency_peak_region_share",
    "saliency_subject_share",
    "saliency_background_share",
)

ADVANCED_CV_TEXT_FIELDS: tuple[str, ...] = (
    "text_region_count",
    "text_area_ratio",
    "text_largest_region_area_ratio",
    "text_subject_overlap_ratio",
)

ADVANCED_CV_FACE_FIELDS: tuple[str, ...] = (
    "face_count",
    "face_area_ratio",
    "largest_face_area_ratio",
)

ADVANCED_CV_PERSON_FIELDS: tuple[str, ...] = (
    "person_count",
    "person_area_ratio",
    "largest_person_area_ratio",
)

ADVANCED_CV_SUBJECT_FIELDS: tuple[str, ...] = (
    "subject_count",
    "subject_area_ratio",
    "largest_subject_area_ratio",
)

# CLIP EMBEDDINGS: stored in separate Parquet, NOT in main CSV
# Path: data/research/master_dataset_v2/embeddings/clip_embeddings_v1.parquet
# Shape: (N, 512) float32 normalized vectors keyed by video_id

# 9. AVAILABILITY FLAGS
AVAILABILITY_FLAG_FIELDS: tuple[str, ...] = (
    "propagation_data_available",
    "channel_stats_available",
    "subscriber_count_available",
    "visual_data_available",
    "like_data_available",
    "comment_data_available",
    "thumbnail_available",
    "advanced_cv_available",
    "clip_embedding_available",
)


# ---------------------------------------------------------------------------
# All non-lineage scalar columns (for assembler reference)
# ---------------------------------------------------------------------------

ALL_SCALAR_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_FIELDS,
    *SAMPLING_FIELDS,
    *ELIGIBILITY_FIELDS,
    *OUTCOME_FIELDS,
    *CHANNEL_CONTROL_FIELDS,
    *DERIVED_OUTCOME_FIELDS,
    *BASIC_CV_IMAGE_FIELDS,
    *BASIC_CV_ADMIN_FIELDS,
    *BASIC_CV_FEATURE_FIELDS,
    *ADVANCED_CV_ADMIN_FIELDS,
    *ADVANCED_CV_AESTHETIC_FIELDS,
    *ADVANCED_CV_SALIENCY_FIELDS,
    *ADVANCED_CV_TEXT_FIELDS,
    *ADVANCED_CV_FACE_FIELDS,
    *ADVANCED_CV_PERSON_FIELDS,
    *ADVANCED_CV_SUBJECT_FIELDS,
    *AVAILABILITY_FLAG_FIELDS,
)


# ---------------------------------------------------------------------------
# Required Fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: tuple[str, ...] = (
    # Identifiers
    "video_id",
    "source_platform",
    "collection_time",
    "dataset_version",
    "schema_version",
    # Sampling
    "month_bucket",
    "topic",
    "query_keyword",
    "publish_time",
    "title",
    # Eligibility
    "channel_id",
    "author_tier",
    "football_final_status",
    "relevance_decision",
    "decision_source",
    # Outcomes
    "view_count",
    "like_count",
    "comment_count",
    # Channel controls
    "subscriber_count",
    "log_subscriber_count",
    # Derived outcomes
    "video_age_days",
    "log_view_count",
    "log_like_count",
    "log_comment_count",
    # Basic CV
    "feature_extraction_status",
    # Availability flags
    "propagation_data_available",
    "channel_stats_available",
    "subscriber_count_available",
    "visual_data_available",
    "like_data_available",
    "comment_data_available",
    "thumbnail_available",
    "advanced_cv_available",
    "clip_embedding_available",
)


# ---------------------------------------------------------------------------
# Optional Fields
# ---------------------------------------------------------------------------

OPTIONAL_FIELDS: tuple[str, ...] = (
    "collection_batch",
    "description_snippet",
    "channel_title",
    "shares",
    "saves",
    "image_path",
    "image_width",
    "image_height",
    "feature_extraction_error",
    *BASIC_CV_FEATURE_FIELDS,
    *ADVANCED_CV_AESTHETIC_FIELDS,
    *ADVANCED_CV_SALIENCY_FIELDS,
    *ADVANCED_CV_TEXT_FIELDS,
    *ADVANCED_CV_FACE_FIELDS,
    *ADVANCED_CV_PERSON_FIELDS,
    *ADVANCED_CV_SUBJECT_FIELDS,
    "views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement",
)


# ---------------------------------------------------------------------------
# Dtype Rules
# ---------------------------------------------------------------------------

DTYPE_RULES: dict[str, type] = {
    "video_id": str,
    "source_platform": str,
    "collection_batch": str,
    "collection_time": str,
    "dataset_version": str,
    "schema_version": str,
    "month_bucket": str,
    "topic": str,
    "query_keyword": str,
    "publish_time": str,
    "title": str,
    "description_snippet": str,
    "channel_id": str,
    "author_tier": str,
    "football_final_status": str,
    "relevance_decision": str,
    "decision_source": str,
    "channel_title": str,
    "subscriber_count": float,
    "view_count": float,
    "like_count": float,
    "comment_count": float,
    "shares": float,
    "saves": float,
    "image_path": str,
    "image_width": int,
    "image_height": int,
    "feature_extraction_status": str,
    "feature_extraction_error": str,
    "aspect_ratio": float,
    "brightness_mean": float,
    "brightness_std": float,
    "saturation_mean": float,
    "saturation_std": float,
    "colorfulness": float,
    "contrast": float,
    "sharpness_laplacian": float,
    "edge_density": float,
    "entropy": float,
    "center_brightness": float,
    "border_brightness": float,
    "center_edge_density": float,
    "dominant_color_r": float,
    "dominant_color_g": float,
    "dominant_color_b": float,
    "warm_color_ratio": float,
    "dark_pixel_ratio": float,
    "bright_pixel_ratio": float,
    "advanced_cv_status": str,
    "advanced_cv_error": str,
    "aesthetic_score": float,
    "saliency_concentration_top10": float,
    "saliency_center_share": float,
    "saliency_peak_region_share": float,
    "saliency_subject_share": float,
    "saliency_background_share": float,
    "text_region_count": int,
    "text_area_ratio": float,
    "text_largest_region_area_ratio": float,
    "text_subject_overlap_ratio": float,
    "face_count": float,
    "face_area_ratio": float,
    "largest_face_area_ratio": float,
    "person_count": float,
    "person_area_ratio": float,
    "largest_person_area_ratio": float,
    "subject_count": float,
    "subject_area_ratio": float,
    "largest_subject_area_ratio": float,
    "video_age_days": float,
    "log_subscriber_count": float,
    "log_view_count": float,
    "log_like_count": float,
    "log_comment_count": float,
    "views_per_day": float,
    "likes_per_1000_views": float,
    "comments_per_1000_views": float,
    "engagement": float,
    "propagation_data_available": bool,
    "channel_stats_available": bool,
    "subscriber_count_available": bool,
    "visual_data_available": bool,
    "like_data_available": bool,
    "comment_data_available": bool,
    "thumbnail_available": bool,
    "advanced_cv_available": bool,
    "clip_embedding_available": bool,
}


# ---------------------------------------------------------------------------
# Enum Allowed Values
# ---------------------------------------------------------------------------

ENUM_ALLOWED_VALUES: dict[str, set[str]] = {
    "source_platform": {"youtube"},
    "author_tier": {"long_tail", "mid", "head"},
    "football_final_status": {"accept"},
    "relevance_decision": {"Accept", "Reject", "Abstain"},
    "decision_source": {
        "vlm_auto",
        "vlm_auto_high_confidence",
        "manual_override_accept",
        "manual_override_reject",
        "unanimous_vlm",
        "mixed_vlm_manual",
        "not_applicable",
    },
    "feature_extraction_status": {"success", "failed"},
    "advanced_cv_status": {"success", "failed", "not_run"},
    "topic": {
        "club",
        "competition",
        "core_football_content",
        "entertainment_derivative",
        "multilingual",
        "player",
    },
}


# ---------------------------------------------------------------------------
# Non-Negative Numeric Fields
# ---------------------------------------------------------------------------

NON_NEGATIVE_FIELDS: tuple[str, ...] = (
    "subscriber_count",
    "view_count",
    "like_count",
    "comment_count",
    "shares",
    "saves",
    "image_width",
    "image_height",
    "video_age_days",
)


# ---------------------------------------------------------------------------
# Prohibited Fields (must NOT appear in master dataset)
# ---------------------------------------------------------------------------

PROHIBITED_FIELDS: tuple[str, ...] = (
    "ctr",
    "click_through_rate",
    "shares",
    "saves",
    "paid_boost",
    "roi",
    "historical_follower_count",
    "publication_time_followers",
)


# ---------------------------------------------------------------------------
# Data Class for Schema Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MasterDatasetFieldMetadata:
    name: str
    tier: Literal["RAW", "DERIVED", "AI_PREDICTED", "QC", "OPTIONAL", "NOT_AVAILABLE"]
    dtype: str
    unit: str
    source_module: str
    missing_policy: str
    research_role: str
    leakage_risk: bool


# ---------------------------------------------------------------------------
# Frozen Source Paths (for lineage)
# ---------------------------------------------------------------------------

FROZEN_SOURCE_PATHS: dict[str, str] = {
    "propagation_metrics": "data/research/propagation_metrics_v1/video_propagation_metrics_v1.csv",
    "channel_statistics": "data/research/propagation_metrics_v1/channel_statistics_v1.csv",
    "basic_cv": "data/research/basic_cv_v1/basic_cv_features_frozen5000_v1.csv",
    "thumbnail_manifest": "data/research/thumbnails_frozen5000_v1/thumbnail_manifest_v1_reconciled.csv",
    "advanced_cv_scalar": "data/research/advanced_cv_v1/advanced_cv_v1_features_*.csv",
    "advanced_cv_clip": "data/research/advanced_cv_v1/clip_embeddings_*.parquet",
    "master_dataset_output": "data/research/master_dataset_v2/master_dataset_v2.csv",
    "clip_embeddings_output": "data/research/master_dataset_v2/embeddings/clip_embeddings_v1.parquet",
}


# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

OUTPUT_PATHS: dict[str, str] = {
    "master_dataset_csv": "data/research/master_dataset_v2/master_dataset_v2.csv",
    "clip_embeddings_parquet": "data/research/master_dataset_v2/embeddings/clip_embeddings_v1.parquet",
    "qc_summary_json": "data/research/master_dataset_v2/qc_summary.json",
    "lineage_manifest": "data/research/master_dataset_v2/lineage_manifest.json",
    "rejected_ids_log": "data/research/master_dataset_v2/rejected_ids.log",
}


# ---------------------------------------------------------------------------
# Frozen Dependencies
# ---------------------------------------------------------------------------

FROZEN_DEPENDENCIES: dict[str, str] = {
    "formal_sampling_protocol": "formal_sampling_protocol_v1_1",
    "propagation_metrics": "v1",
    "basic_cv": "v1",
    "advanced_cv_schema": "advanced_cv_v1.1.0-minimal",
    "advanced_cv_extractor": "advanced_cv_v1.1.0",
    "research_schema": "v1",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def missing_fields(
    record: dict | object, required: tuple[str, ...]
) -> set[str]:
    """Return required keys absent from a record."""
    if isinstance(record, dict):
        present = set(record.keys())
    else:
        present = set(vars(record).keys())
    return set(required).difference(present)


def prohibited_fields_present(record: dict) -> set[str]:
    """Return prohibited field names that appear in a record."""
    if isinstance(record, dict):
        present = set(record.keys())
    else:
        present = set(vars(record).keys())
    return present.intersection(set(PROHIBITED_FIELDS))


def schema_summary() -> dict[str, int]:
    """Return counts per field group."""
    return {
        "identifiers": len(IDENTIFIER_FIELDS),
        "sampling": len(SAMPLING_FIELDS),
        "eligibility": len(ELIGIBILITY_FIELDS),
        "outcomes": len(OUTCOME_FIELDS),
        "channel_controls": len(CHANNEL_CONTROL_FIELDS),
        "derived_outcomes": len(DERIVED_OUTCOME_FIELDS),
        "basic_cv_lineage": len(BASIC_CV_LINEAGE_FIELDS),
        "basic_cv_image": len(BASIC_CV_IMAGE_FIELDS),
        "basic_cv_admin": len(BASIC_CV_ADMIN_FIELDS),
        "basic_cv_features": len(BASIC_CV_FEATURE_FIELDS),
        "advanced_cv_lineage": len(ADVANCED_CV_LINEAGE_FIELDS),
        "advanced_cv_admin": len(ADVANCED_CV_ADMIN_FIELDS),
        "advanced_cv_aesthetic": len(ADVANCED_CV_AESTHETIC_FIELDS),
        "advanced_cv_saliency": len(ADVANCED_CV_SALIENCY_FIELDS),
        "advanced_cv_text": len(ADVANCED_CV_TEXT_FIELDS),
        "advanced_cv_face": len(ADVANCED_CV_FACE_FIELDS),
        "advanced_cv_person": len(ADVANCED_CV_PERSON_FIELDS),
        "advanced_cv_subject": len(ADVANCED_CV_SUBJECT_FIELDS),
        "availability_flags": len(AVAILABILITY_FLAG_FIELDS),
        "total_scalar_columns": len(ALL_SCALAR_COLUMNS),
        "required_fields": len(REQUIRED_FIELDS),
        "optional_fields": len(OPTIONAL_FIELDS),
    }
