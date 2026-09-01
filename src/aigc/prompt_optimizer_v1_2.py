"""Diagnosis-conditioned integrated prompt construction for Demo V1.2.

V1.1 remains the authoritative implementation of the isolated B1/B2/B3
research mechanisms.  This module adds a user-facing integrated strategy
without changing those mechanisms or any frozen research artifact.
"""
from __future__ import annotations

from typing import Any

from src.aigc.prompt_optimizer_v1_1 import (
    EVIDENCE_POLICY,
    FEATURES,
    MASTER_POLICY,
    NEGATIVE_PROMPT,
    STRATEGIES,
    StructuredPrompt,
    _protected,
    build_prompt,
    build_prompt_set,
    diagnose,
    generator_status,
)

INTEGRATED_OPTIMIZATION = "INTEGRATED_OPTIMIZATION"
INTEGRATED_LABEL = "综合优化（推荐）"


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def select_integrated_interventions(features: dict[str, Any]) -> list[dict[str, str]]:
    """Select only relevant high/moderate-evidence interventions.

    Thresholds are conservative presentation heuristics for deciding whether
    to include a direction in a prompt; they are not claimed universal optima.
    """
    m = diagnose(features)["measurements"]
    text = _number(m.get("text_coverage"))
    aesthetic = _number(m.get("aesthetic_coherence"))
    entropy = _number(m.get("visual_entropy"))
    center = _number(m.get("center_saliency"))
    faces = _number(m.get("face_presence"))
    subject = _number(m.get("subject_prominence"))
    selected: list[dict[str, str]] = []

    if text is not None and text >= 0.08:
        selected.append({"id": "REDUCE_EXCESSIVE_TEXT", "evidence": "HIGH",
                         "direction": "Reduce redundant or competing text while preserving the factual title meaning."})
    if aesthetic is not None and aesthetic < 0.65:
        selected.append({"id": "IMPROVE_AESTHETIC_COHERENCE", "evidence": "MODERATE",
                         "direction": "Improve hierarchy, balance, tonal coherence and foreground/background separation."})
    if entropy is not None and entropy > 0.65:
        selected.append({"id": "REDUCE_EXCESS_VISUAL_ENTROPY", "evidence": "MODERATE",
                         "direction": "Remove avoidable clutter and competing micro-elements without stripping necessary context."})
    if center is not None and center < 0.45:
        selected.append({"id": "STRENGTHEN_FOCAL_ATTENTION", "evidence": "MODERATE",
                         "direction": "Strengthen one clear focal subject through scale, separation, contrast and framing; exact centering is not required."})
    if faces is not None and faces > 0 and (subject is None or subject < 0.35):
        selected.append({"id": "PRESERVE_CONTEXTUAL_FACE_PRESENCE", "evidence": "MODERATE",
                         "direction": "Preserve the same visible person's identity and improve subject legibility without creating a different person."})
    return selected


def build_integrated_prompt(
    features: dict[str, Any], *, title: str = "", topic: str = "football", content_brief: str = ""
) -> dict[str, Any]:
    diagnosis = diagnose(features)
    m = diagnosis["measurements"]
    interventions = select_integrated_interventions(features)
    protected = _protected(title, topic)
    editable = [
        "layout hierarchy and spacing",
        "typographic hierarchy and overlay placement",
        "subject scale, separation, contrast and framing",
        "background clutter, lighting and tonal treatment",
    ]

    if interventions:
        intervention_text = "\n".join(f"- [{x['evidence']}] {x['direction']}" for x in interventions)
        expected = [x["direction"] for x in interventions]
    else:
        intervention_text = "- No high/moderate-evidence weakness is clearly triggered; preserve the current visual structure and make only minimal polish changes."
        expected = ["preserve satisfactory measured dimensions", "minimal non-destructive visual polish"]

    preservation: list[str] = []
    text = _number(m.get("text_coverage"))
    if text is not None and text < 0.08:
        preservation.append("Text coverage is already low; preserve the compact text structure and do not add copy.")
    center = _number(m.get("center_saliency"))
    if center is not None and center >= 0.45:
        preservation.append("Focal attention is already comparatively strong; preserve it and do not force recentering.")
    aesthetic = _number(m.get("aesthetic_coherence"))
    if aesthetic is not None and aesthetic >= 0.65:
        preservation.append("Aesthetic coherence is already comparatively strong; avoid destructive restyling.")
    entropy = _number(m.get("visual_entropy"))
    if entropy is not None and entropy <= 0.65:
        preservation.append("Visual entropy is not excessive; do not over-simplify the necessary scene context.")
    faces = _number(m.get("face_presence"))
    if faces is not None and faces <= 0:
        preservation.append("No face is detected; do not invent a person or face.")

    context = f"Source context — title: {title or 'not supplied'}; topic: {topic or 'football'}"
    if content_brief:
        context += f"; brief: {content_brief}"
    prompt = "\n\n".join([
        MASTER_POLICY,
        context,
        "CURRENT VISUAL DIAGNOSIS:\n" + "\n".join(f"- {k}: {v}" for k, v in m.items()),
        "SELECTED EVIDENCE-BACKED INTERVENTIONS:\n" + intervention_text,
        "PRESERVE SATISFACTORY DIMENSIONS:\n" + ("\n".join(f"- {x}" for x in preservation) or "- Make no unnecessary changes."),
        "PROTECTED SEMANTIC ELEMENTS:\n- " + "\n- ".join(protected),
        "EDITABLE VISUAL ELEMENTS:\n- " + "\n- ".join(editable),
        "INTEGRATED GENERATION INSTRUCTION: Produce one coherent same-content football thumbnail by applying only the selected interventions together. Keep all unselected dimensions stable. If any factual detail is uncertain, simplify or omit it rather than inventing it.",
        "EXPLORATORY DIAGNOSTICS ONLY: saturation and edge density may be observed during evaluation, but must not be optimized, maximized, minimized, or treated as hard generation directions.",
    ])
    return StructuredPrompt(
        strategy_id=INTEGRATED_OPTIMIZATION,
        strategy_name=INTEGRATED_LABEL,
        diagnosis=diagnosis,
        evidence_level="HIGH / MODERATE (diagnosis-selected)",
        evidence_summary="Automatically selects only relevant high/moderate-evidence directions; exploratory factors remain diagnostic-only.",
        primary_objective="Improve the current thumbnail through the smallest relevant evidence-backed set of changes while preserving its semantics.",
        protected_elements=protected,
        editable_elements=editable,
        generation_prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        expected_feature_direction=expected,
        research_caveat="Observational research guidance only. This is not a CTR estimate or guaranteed improvement; evaluate only same-content candidates. Temporal generalization remains weak.",
    ).to_dict() | {"selected_interventions": interventions}


__all__ = [
    "EVIDENCE_POLICY", "FEATURES", "STRATEGIES", "INTEGRATED_OPTIMIZATION", "INTEGRATED_LABEL",
    "build_prompt", "build_prompt_set", "build_integrated_prompt", "select_integrated_interventions",
    "diagnose", "generator_status",
]
