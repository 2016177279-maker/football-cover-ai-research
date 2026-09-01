from __future__ import annotations

from pathlib import Path
from PIL import Image
from .schemas import CandidateStatus, ValidityResult

GATE_NAMES = ("IMAGE_DECODABLE", "VALID_DIMENSIONS", "SEMANTIC_IDENTITY_PRESERVED",
              "NO_SEVERE_TEXT_CORRUPTION", "NO_OBVIOUS_VISUAL_ARTIFACT", "CONTENT_RELEVANCE_PASS")

def validate_candidate(image_path: str, *, semantic_identity_preserved: bool | None = None,
                       no_severe_text_corruption: bool | None = None,
                       no_obvious_visual_artifact: bool | None = None,
                       content_relevance_pass: bool | None = None) -> ValidityResult:
    gates = dict.fromkeys(GATE_NAMES, False); reasons=[]
    p=Path(image_path)
    if not p.is_file():
        return ValidityResult(gates, CandidateStatus.REJECT_NO_SCORE, ["missing image"])
    try:
        with Image.open(p) as im: im.load(); w,h=im.size
        gates["IMAGE_DECODABLE"] = True; gates["VALID_DIMENSIONS"] = w > 0 and h > 0
    except Exception as exc:
        return ValidityResult(gates, CandidateStatus.REJECT_NO_SCORE, [f"image decode failed: {exc}"])
    review = {"SEMANTIC_IDENTITY_PRESERVED":semantic_identity_preserved,
              "NO_SEVERE_TEXT_CORRUPTION":no_severe_text_corruption,
              "NO_OBVIOUS_VISUAL_ARTIFACT":no_obvious_visual_artifact,
              "CONTENT_RELEVANCE_PASS":content_relevance_pass}
    for k,v in review.items(): gates[k] = v is True
    if any(v is None for v in review.values()): reasons.append("required semantic/quality review not supplied")
    if all(gates.values()): status=CandidateStatus.ELIGIBLE
    elif gates["IMAGE_DECODABLE"] and gates["VALID_DIMENSIONS"]: status=CandidateStatus.REJECT_AND_REGENERATE
    else: status=CandidateStatus.REJECT_NO_SCORE
    return ValidityResult(gates,status,reasons)

def rule_assessment(features: dict, original: dict | None = None) -> dict:
    delta = {} if original is None else {
        k: float(features[k])-float(original[k])
        for k in features if k in original
        and isinstance(features[k], (int, float)) and isinstance(original[k], (int, float))
    }
    hard = None if original is None else delta.get("advanced_cv_text_area_ratio",0) <= 0
    soft = {"aesthetic_coherence": None if original is None else delta.get("advanced_cv_aesthetic_score",0)>0,
            "visual_entropy": None if original is None else delta.get("basic_cv_entropy",0)<0,
            "limited_face_presence":"DIAGNOSTIC_ONLY", "centered_saliency": None if original is None else delta.get("advanced_cv_saliency_center_share",0)>0}
    return {"tier_1_compliance":hard,"tier_2_alignment":soft,
            "experimental_diagnostics": {"saturation_direction":delta.get("basic_cv_saturation_mean"),"global_edge_complexity":delta.get("basic_cv_edge_density"),"joint_center_global_edge_allocation":delta.get("basic_cv_center_edge_density")},
            "experimental_in_primary_score":False}
