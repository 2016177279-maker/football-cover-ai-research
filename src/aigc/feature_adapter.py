from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.visual_features import extract_visual_features
from .schemas import FEATURE_PIPELINE_VERSION

VISUAL_MODEL_FIELDS=("basic_cv_brightness_mean","basic_cv_saturation_mean","basic_cv_colorfulness","basic_cv_warm_color_ratio","basic_cv_contrast","basic_cv_edge_density","basic_cv_entropy","basic_cv_center_edge_density","advanced_cv_aesthetic_score","advanced_cv_saliency_center_share","advanced_cv_text_area_ratio","advanced_cv_text_subject_overlap_ratio","advanced_cv_face_count","advanced_cv_face_area_ratio","advanced_cv_person_area_ratio","advanced_cv_largest_subject_area_ratio")

def sha256(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def extract_frozen_features(image_path: str, *, video_id: str, advanced_pipeline=None) -> dict:
    """Run unchanged Basic CV v1 plus the unchanged Advanced CV v1 pipeline.

    The injected advanced_pipeline makes the adapter testable; production callers pass a
    configured ``AdvancedCVPipeline``. No PCA is fit here.
    """
    p=Path(image_path)
    record={"video_id":video_id,"input_image_path":str(p),"image_sha256":"",
            "feature_pipeline_version":FEATURE_PIPELINE_VERSION,
            "feature_extraction_timestamp":datetime.now(timezone.utc).isoformat(),"feature_extraction_status":"FAILED"}
    if not p.is_file(): record["failure_reason"]="missing image"; return record
    try:
        basic=extract_visual_features(p)
        record.update({f"basic_cv_{k}":v for k,v in basic.items()}); record["image_sha256"]=sha256(p)
        if advanced_pipeline is None:
            record["feature_extraction_status"]="PARTIAL_ADVANCED_PIPELINE_NOT_CONFIGURED"
            record["failure_reason"]="Advanced CV v1 configured assets are required for a new candidate image"
            return record
        result=advanced_pipeline.process_one(video_id,p)
        record.update({f"advanced_cv_{k}":v for k,v in result.row.items() if k not in ("video_id",)})
        emb=result.embeddings.get("clip")
        if isinstance(emb, dict):
            emb=emb.get("emb")
        record["clip_embedding"]=np.asarray(emb,dtype=np.float32).tolist() if emb is not None else None
        record["feature_extraction_status"]="SUCCESS" if result.row.get("overall_extraction_status")=="success" and emb is not None else "FAILED"
        return record
    except Exception as exc:
        record["failure_reason"]=f"{type(exc).__name__}: {exc}"; return record

def assert_feature_schema(features: dict) -> None:
    missing=[x for x in VISUAL_MODEL_FIELDS if x not in features]
    emb=features.get("clip_embedding")
    if missing or not isinstance(emb,(list,tuple,np.ndarray)) or len(emb)!=512:
        raise ValueError(f"frozen feature schema incompatible; missing={missing}, clip_dim={len(emb) if emb is not None else None}")

def features_from_frozen_row(row: dict, embedding) -> dict:
    out={k:row[k] for k in VISUAL_MODEL_FIELDS}; out["clip_embedding"]=np.asarray(embedding,dtype=np.float32).tolist()
    assert_feature_schema(out); return out
