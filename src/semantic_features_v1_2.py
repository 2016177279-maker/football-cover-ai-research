"""Visual Semantic Feature v1.2 development implementation.

No API calls or generic-caption decisions occur here. Learned evidence is supplied
by explicit targeted detector/classifier adapters; missing evidence abstains.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.visual_features import extract_visual_features

VERSION = "visual-semantic-v1.2"
OBJECT_FIELDS = (
    "person_present", "football_ball_present", "football_pitch_present",
    "football_kit_present", "stadium_present", "trophy_present",
    "club_logo_present", "collage_layout", "strong_foreground_subject",
)
OBJECT_THRESHOLDS = {
    "person_present": .70, "football_ball_present": .75,
    "football_pitch_present": .70, "football_kit_present": .75,
    "stadium_present": .70, "trophy_present": .75,
    "club_logo_present": .80, "collage_layout": .70,
    "strong_foreground_subject": .70,
}
SCENE_VALUES = {"match_action", "portrait", "studio_press", "ceremony", "training", "tactical_graphic", "other"}
ACTION_VALUES = {"active_play", "celebration", "static"}


@dataclass(frozen=True)
class Evidence:
    value: Any
    confidence: float
    source: str
    status: str = "success"
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    scores: tuple[float, ...] = ()
    alternatives: tuple[tuple[str, float], ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class FeatureResult:
    value: Any
    confidence: float
    evidence_source: str
    status: str
    abstained: bool
    detail: str = ""


def image_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gray_small(path: str | Path, width: int = 320) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        h = max(1, round(im.height * width / im.width))
        return np.asarray(im.convert("L").resize((width, h)), dtype=np.float32)


def detect_text_regions(path: str | Path) -> tuple[list[tuple[int, int, int, int]], float, dict[str, Any]]:
    """Find coherent glyph-like tiles with fixed, versioned thresholds.

    This is text-region detection, not OCR/transcription. Thresholds are design
    constants and were not selected from the frozen human-validation set.
    """
    gray = _gray_small(path)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    tile_h, tile_w = max(6, gray.shape[0] // 18), 10
    rows, cols = gray.shape[0] // tile_h, gray.shape[1] // tile_w
    active = np.zeros((rows, cols), dtype=bool)
    for y in range(rows):
        for x in range(cols):
            ys, xs = slice(y*tile_h, (y+1)*tile_h), slice(x*tile_w, (x+1)*tile_w)
            block, eb = gray[ys, xs], edge[ys, xs]
            # Text tiles combine frequent strokes, tonal variation, and both axes.
            dx = np.abs(gx[ys, xs]); dy = np.abs(gy[ys, xs])
            active[y, x] = ((eb >= 32).mean() >= .12 and block.std() >= 24
                            and (dx >= 24).mean() >= .045 and (dy >= 24).mean() >= .035)
    # Keep active tiles with a horizontal/vertical neighbour; isolated texture is rejected.
    coherent = np.zeros_like(active)
    for y, x in zip(*np.where(active)):
        y0, y1, x0, x1 = max(0,y-1), min(rows,y+2), max(0,x-1), min(cols,x+2)
        coherent[y, x] = active[y0:y1, x0:x1].sum() >= 2
    boxes = [(x*tile_w, y*tile_h, (x+1)*tile_w, (y+1)*tile_h)
             for y, x in zip(*np.where(coherent))]
    ratio = float(coherent.sum() * tile_h * tile_w / gray.size)
    return boxes, ratio, {"resize_width": 320, "tile": [tile_w, tile_h], "coherent_tiles": int(coherent.sum())}


def extract_text_features(path: str | Path) -> dict[str, FeatureResult]:
    boxes, ratio, meta = detect_text_regions(path)
    present = ratio >= .015 and len(boxes) >= 2
    category = "none" if not present else "low" if ratio < .06 else "medium" if ratio < .15 else "high"
    distance = abs(ratio - .015)
    confidence = float(min(.99, .55 + distance / .06))
    detail = json.dumps({"text_region_ratio": round(ratio, 6), **meta}, separators=(",", ":"))
    return {
        "text_present": FeatureResult(present, confidence, "deterministic:text_region_v1.2", "success", False, detail),
        "text_amount_category": FeatureResult(category, confidence, "deterministic:text_region_area_v1.2", "success", False, detail),
    }


def _unknown(source: str, detail: str) -> FeatureResult:
    return FeatureResult("unknown", 0.0, source, "abstained", True, detail)


def _box_union_ratio(boxes: Sequence[Sequence[float]], width: int = 256, height: int = 256) -> float:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        if len(box) != 4 or not all(0 <= float(v) <= 1 for v in box):
            raise ValueError("boxes must be normalized xyxy values")
        x0,y0,x1,y1 = box
        if x1 <= x0 or y1 <= y0: raise ValueError("invalid xyxy box")
        xa,xb = int(x0*width), max(1, int(np.ceil(x1*width)))
        ya,yb = int(y0*height), max(1, int(np.ceil(y1*height)))
        mask[ya:min(yb,height), xa:min(xb,width)] = True
    return float(mask.mean())


def resolve_object_features(evidence: Mapping[str, Evidence] | None) -> dict[str, FeatureResult]:
    evidence = evidence or {}
    out: dict[str, FeatureResult] = {}
    for field in OBJECT_FIELDS:
        ev = evidence.get(field)
        if ev is None or ev.status != "success":
            out[field] = _unknown(f"detector:{field}", "targeted evidence unavailable")
        elif ev.confidence < OBJECT_THRESHOLDS[field] or not isinstance(ev.value, bool):
            out[field] = _unknown(ev.source, f"below threshold or invalid value; confidence={ev.confidence:.3f}")
        else:
            out[field] = FeatureResult(ev.value, ev.confidence, ev.source, "success", False, ev.detail)
    person = evidence.get("person_present")
    if person is None or person.status != "success" or person.confidence < .70:
        out["person_area_category"] = _unknown("spatial:person_boxes", "accepted person evidence unavailable")
    elif person.value is False:
        out["person_area_category"] = FeatureResult("none", person.confidence, person.source, "success", False, "person detector explicit negative")
    else:
        try:
            boxes = [b for i,b in enumerate(person.boxes) if not person.scores or person.scores[i] >= .70]
            if not boxes: raise ValueError("positive person evidence has no accepted boxes")
            ratio = _box_union_ratio(boxes)
            value = "small" if ratio < .08 else "medium" if ratio < .25 else "large"
            out["person_area_category"] = FeatureResult(value, person.confidence, "spatial:accepted_person_box_union_v1.2", "success", False, f"union_ratio={ratio:.6f}")
        except (ValueError, IndexError) as exc:
            out["person_area_category"] = _unknown("spatial:person_boxes", str(exc))
    return out


def _restricted(ev: Evidence | None, allowed: set[str], name: str) -> FeatureResult:
    if ev is None or ev.status != "success" or ev.value not in allowed:
        return _unknown(f"classifier:{name}", "restricted classifier evidence unavailable/invalid")
    runner_up = max((score for label,score in ev.alternatives if label != ev.value), default=0.0)
    if ev.confidence < .75 or ev.confidence - runner_up < .15:
        return _unknown(ev.source, f"confidence={ev.confidence:.3f};runner_up={runner_up:.3f}")
    return FeatureResult(ev.value, ev.confidence, ev.source, "success", False, ev.detail)


def extract_semantic_features(
    path: str | Path,
    detector_evidence: Mapping[str, Evidence] | None = None,
    classifier_evidence: Mapping[str, Evidence] | None = None,
) -> dict[str, Any]:
    """Extract 21 legacy deterministic CV values plus 14 v1.2 semantic results."""
    result: dict[str, Any] = {"version": VERSION, "image_sha256": image_sha256(path)}
    failures: list[str] = []
    try: result["deterministic_visual_v1"] = extract_visual_features(path)
    except Exception as exc: result["deterministic_visual_v1"] = {}; failures.append(f"visual:{type(exc).__name__}:{exc}")
    features: dict[str, FeatureResult] = {}
    try: features.update(extract_text_features(path))
    except Exception as exc:
        failures.append(f"text:{type(exc).__name__}:{exc}")
        features["text_present"] = _unknown("deterministic:text_region_v1.2", str(exc))
        features["text_amount_category"] = _unknown("deterministic:text_region_area_v1.2", str(exc))
    features.update(resolve_object_features(detector_evidence))
    cev = classifier_evidence or {}
    features["scene_type"] = _restricted(cev.get("scene_type"), SCENE_VALUES, "scene_type")
    features["action_cue"] = _restricted(cev.get("action_cue"), ACTION_VALUES, "action_cue")
    result["features"] = {k: asdict(v) for k,v in features.items()}
    result["status"] = "partial" if failures or any(v.abstained for v in features.values()) else "success"
    result["failures"] = failures
    return result


def evidence_from_json(payload: Mapping[str, Any]) -> Evidence:
    return Evidence(
        value=payload.get("value"), confidence=float(payload.get("confidence", 0)),
        source=str(payload.get("source", "external:unspecified")), status=str(payload.get("status", "success")),
        boxes=tuple(tuple(map(float,b)) for b in payload.get("boxes", ())),
        scores=tuple(map(float,payload.get("scores", ()))),
        alternatives=tuple((str(x[0]),float(x[1])) for x in payload.get("alternatives", ())),
        detail=str(payload.get("detail", "")),
    )
