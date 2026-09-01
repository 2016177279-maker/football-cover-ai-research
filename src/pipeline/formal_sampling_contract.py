"""Schema constants and pure validation helpers for Formal Sampling Protocol v1.

This module performs no collection, network access, filtering, or file writes.
Threshold values intentionally remain in the versioned YAML configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PROTOCOL_VERSION = "formal_sampling_protocol_v1"
DATASET_VERSION = "football_cover_formal_v1"
FILTER_DECISIONS = frozenset({"accept", "reject", "abstain"})
MANUAL_DECISIONS = frozenset({"manual_accept", "manual_reject"})

BATCH_REQUIRED_FIELDS = frozenset(
    {
        "batch_id",
        "collection_date",
        "query_group",
        "raw_candidate_count",
        "unique_candidate_count",
    }
)

CANDIDATE_REQUIRED_FIELDS = frozenset(
    {
        "video_id",
        "publish_time",
        "collection_time",
        "video_age_days",
        "batch_id",
        "query_group",
        "query_keyword",
        "channel_id",
        "author_followers",
        "channel_sample_count",
        "thumbnail_url",
        "image_path",
        "views",
        "development_overlap",
        "filter_status",
        "manual_review_status",
        "thumbnail_download_status",
        "thumbnail_decode_status",
        "thumbnail_qc_status",
    }
)

LINEAGE_REQUIRED_FIELDS = frozenset(
    {
        "dataset_version",
        "sampling_protocol_version",
        "filter_bundle_sha",
        "visual_feature_version",
        "metrics_version",
        "schema_version",
    }
)


def missing_fields(record: Mapping[str, object], required: Sequence[str] | frozenset[str]) -> set[str]:
    """Return required keys absent from a record; values are validated downstream."""

    return set(required).difference(record)


def relevance_eligible(filter_status: str, manual_review_status: str | None) -> bool:
    """Apply the protocol's only permitted relevance eligibility rule."""

    if filter_status not in FILTER_DECISIONS:
        raise ValueError(f"Unknown filter decision: {filter_status!r}")
    if manual_review_status is not None and manual_review_status not in MANUAL_DECISIONS:
        raise ValueError(f"Unknown manual decision: {manual_review_status!r}")
    return filter_status == "accept" or (
        filter_status == "abstain" and manual_review_status == "manual_accept"
    )


def validate_unique_video_ids(records: Sequence[Mapping[str, object]]) -> list[str]:
    """Return duplicate non-empty video IDs in first-repeat order."""

    seen: set[str] = set()
    duplicates: list[str] = []
    reported: set[str] = set()
    for record in records:
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        if video_id in seen and video_id not in reported:
            duplicates.append(video_id)
            reported.add(video_id)
        seen.add(video_id)
    return duplicates
