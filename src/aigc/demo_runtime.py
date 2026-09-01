"""Reusable orchestration for the Streamlit research demo.

This module contains no Streamlit calls.  It adapts the frozen AIGC prototype
APIs for both built-in FINAL V2 rows and newly uploaded images.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_adapter import VISUAL_MODEL_FIELDS, extract_frozen_features, features_from_frozen_row
from .generation import OptionalGeneratorProvider, generate_candidates
from .offline_ab import compare_arms
from .rules import rule_assessment, validate_candidate
from .schemas import PrototypeInput, TEMPORAL_WARNING
from .scoring import apply_research_heuristic, score_candidate

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "reports/final_v2_aigc_prototype_v1/aigc_smoke_sample_v1.csv"
MASTER = ROOT / "data/research/final_master_dataset_v2/final_master_dataset_v2.parquet"
CLIP = ROOT / "data/research/advanced_cv_final_v2_v1/advanced_cv_final_v2_clip_embeddings_v1.parquet"
UPLOAD_DIR = ROOT / "demo/streamlit_demo_v1/uploads"
DISPLAY_FEATURES = {
    "advanced_cv_text_area_ratio": "Text coverage",
    "advanced_cv_aesthetic_score": "Aesthetic score",
    "basic_cv_entropy": "Entropy / complexity",
    "advanced_cv_saliency_center_share": "Centered saliency",
    "advanced_cv_face_count": "Face presence (count)",
    "advanced_cv_face_area_ratio": "Face prominence",
    "advanced_cv_person_area_ratio": "Person prominence",
    "advanced_cv_largest_subject_area_ratio": "Largest subject prominence",
}
PROHIBITED_UI_PHRASES = (
    "ctr prediction", "expected ctr uplift", "guaranteed views", "future views forecast",
    "views will increase by",
)


def load_builtin_examples(sample_csv: Path = SAMPLE_CSV) -> list[dict[str, Any]]:
    """Load outcome-blind deterministic examples and resolve local thumbnails."""
    rows = pd.read_csv(sample_csv).fillna("").to_dict("records")
    examples = []
    for row in rows:
        path = Path(str(row["original_thumbnail"]))
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            row["original_thumbnail"] = str(path)
            row["label"] = f"{row['video_id']} · {row['author_tier']} · {row['topic']}"
            examples.append(row)
    if not 2 <= len(examples) <= 4:
        raise ValueError(f"expected 2–4 usable built-in examples, found {len(examples)}")
    return examples


def make_request(example: dict[str, Any], **overrides: Any) -> PrototypeInput:
    values = {**example, **overrides}
    return PrototypeInput(
        video_id=str(values.get("video_id") or "uploaded"),
        original_thumbnail=str(values["original_thumbnail"]),
        title=str(values.get("title") or ""), topic=str(values.get("topic") or "football"),
        author_tier=str(values.get("author_tier") or "MID"),
        subscriber_count=float(values.get("subscriber_count") or 0),
        publish_month=str(values.get("publish_month") or "2026-08"),
        content_brief=str(values.get("content_brief") or ""),
    )


def load_builtin_features(video_id: str) -> dict[str, Any]:
    master = pd.read_parquet(MASTER, filters=[("video_id", "==", video_id)])
    clip = pd.read_parquet(CLIP, filters=[("video_id", "==", video_id)])
    if len(master) != 1 or len(clip) != 1:
        raise ValueError(f"frozen data unavailable or non-unique for {video_id}")
    vector = clip.iloc[0][[f"clip_{i:03d}" for i in range(512)]].to_numpy(dtype=np.float32)
    return features_from_frozen_row(master.iloc[0].to_dict(), vector)


def build_advanced_pipeline():
    """Load the pinned offline-only Advanced CV pipeline for uploaded images."""
    from src.advanced_cv_v1.model_assets import CORE_ASSET_COMPONENTS, load_and_verify_assets
    from src.advanced_cv_v1.pipeline import AdvancedCVPipeline, PipelineConfig, core_component_specs_from_assets
    assets = load_and_verify_assets(ROOT / "configs/advanced_cv_v1_model_assets.json", ROOT, CORE_ASSET_COMPONENTS)
    cache = ROOT / "demo/streamlit_demo_v1/cache"
    return AdvancedCVPipeline(
        PipelineConfig(cache_root=cache, embeddings_dir=cache / "embeddings",
                       checkpoint_path=cache / "checkpoint.json", metrics_path=cache / "metrics.json"),
        component_specs=core_component_specs_from_assets(assets),
    )


def persist_upload(data: bytes, filename: str, arm_id: str) -> Path:
    if not data:
        raise ValueError("empty upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("unsupported image type; use JPG, PNG, or WEBP")
    digest = hashlib.sha256(data).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{arm_id}_{digest[:20]}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    validity = validate_candidate(str(path))
    if not validity.gates["IMAGE_DECODABLE"] or not validity.gates["VALID_DIMENSIONS"]:
        path.unlink(missing_ok=True)
        raise ValueError("missing, corrupt, or zero-dimension image upload")
    return path


def analyze_features(features: dict[str, Any], context: dict[str, Any], *, candidate_id: str,
                     original_features: dict[str, Any] | None = None, bundle=None,
                     candidate_status: str = "ELIGIBLE") -> dict[str, Any]:
    assessment = rule_assessment(features, original_features)
    if candidate_id == "A":
        assessment["tier_1_compliance"] = True
    scored = score_candidate(features, context, bundle)
    return {
        "candidate_id": candidate_id, "candidate_status": candidate_status,
        "nonvisual_context": context, "features": features,
        "feature_deltas_vs_A": {} if original_features is None else {
            key: float(features[key]) - float(original_features[key])
            for key in DISPLAY_FEATURES if key in features and key in original_features
        },
        **assessment, **scored,
        "evidence_explanation": evidence_explanation(candidate_id, assessment),
    }


def analyze_uploaded(path: str | Path, request: PrototypeInput, *, candidate_id: str,
                     original_features: dict[str, Any] | None, pipeline, bundle,
                     review: dict[str, bool]) -> dict[str, Any]:
    validity = validate_candidate(str(path), **review)
    if validity.status.value != "ELIGIBLE":
        return unscored_arm(candidate_id, request.nonvisual_context(), validity.status.value,
                            "; ".join(validity.reasons) or "candidate validity gates did not pass")
    features = extract_frozen_features(str(path), video_id=f"{request.video_id}_{candidate_id}", advanced_pipeline=pipeline)
    if features.get("feature_extraction_status") != "SUCCESS":
        return unscored_arm(candidate_id, request.nonvisual_context(), "REJECT_NO_SCORE",
                            features.get("failure_reason", "frozen feature extraction failed"))
    return analyze_features(features, request.nonvisual_context(), candidate_id=candidate_id,
                            original_features=original_features, bundle=bundle)


def unscored_arm(candidate_id: str, context: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id, "candidate_status": status, "nonvisual_context": context,
        "features": {}, "feature_deltas_vs_A": {}, "tier_1_compliance": None,
        "tier_2_alignment": {}, "experimental_diagnostics": {}, "point_prediction": None,
        "research_propagation_score": None, "prediction_interval_lower": None,
        "prediction_interval_upper": None, "temporal_risk_warning": TEMPORAL_WARNING,
        "evidence_explanation": reason,
    }


def generate_instruction_set(request: PrototypeInput) -> list[dict[str, Any]]:
    return [x.to_dict() for x in generate_candidates(request)]


def optional_generator(request: PrototypeInput, generator=None) -> dict[str, Any]:
    if generator is None:
        return {"status": "GENERATOR_NOT_CONFIGURED", "instructions": generate_instruction_set(request), "paths": []}
    items = generate_candidates(request, OptionalGeneratorProvider(generator))
    return {"status": "GENERATED", "instructions": [x.to_dict() for x in items],
            "paths": [x.image_path for x in items]}


def compare_same_context(arms: list[dict[str, Any]]) -> dict[str, Any]:
    apply_research_heuristic(arms)
    return compare_arms(arms)


def evidence_explanation(candidate_id: str, assessment: dict[str, Any]) -> str:
    if candidate_id == "A":
        return "Original reference arm scored with frozen features and the frozen research model."
    t1 = "passes" if assessment.get("tier_1_compliance") else "does not improve"
    aligned = [k.replace("_", " ") for k, v in assessment.get("tier_2_alignment", {}).items() if v is True]
    detail = ", ".join(aligned) if aligned else "no clear Tier-2 improvement"
    return f"Candidate {t1} the text-coverage rule relative to Original; Tier-2 evidence: {detail}."


def compact_diagnostics(features: dict[str, Any]) -> dict[str, Any]:
    return {label: features.get(key) for key, label in DISPLAY_FEATURES.items()}


def format_interval(row: dict[str, Any]) -> str:
    lo, hi = row.get("prediction_interval_lower"), row.get("prediction_interval_upper")
    return "Not scored" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"


def assert_safe_ui_copy(text: str) -> None:
    lowered = text.lower()
    found = [phrase for phrase in PROHIBITED_UI_PHRASES if phrase in lowered]
    if found:
        raise ValueError(f"prohibited score language: {found}")


def manifest_summary() -> dict[str, Any]:
    return {"built_in_examples": len(load_builtin_examples()), "temporal_warning": TEMPORAL_WARNING,
            "generator_configured": False, "provider_interface": "src.aigc.generation.GenerationProvider"}
