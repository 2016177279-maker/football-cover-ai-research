"""Evidence-guided, diagnosis-conditioned prompt construction for Demo V1.1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EVIDENCE_POLICY = {
    "excessive_text_coverage": "HIGH",
    "aesthetic_coherence": "MODERATE",
    "visual_entropy": "MODERATE",
    "limited_face_presence": "MODERATE",
    "centered_saliency": "MODERATE",
    "saturation_direction": "EXPLORATORY",
    "global_edge_complexity": "EXPLORATORY",
    "joint_center_global_edge_allocation": "EXPLORATORY",
}

FEATURES = {
    "text_coverage": "advanced_cv_text_area_ratio",
    "aesthetic_coherence": "advanced_cv_aesthetic_score",
    "visual_entropy": "basic_cv_entropy",
    "center_saliency": "advanced_cv_saliency_center_share",
    "face_presence": "advanced_cv_face_count",
    "subject_prominence": "advanced_cv_person_area_ratio",
}

STRATEGIES = {
    "B1": ("TEXT_SIMPLIFICATION", "HIGH", "Reduce excessive textual visual load while preserving the original content and message."),
    "B2": ("AESTHETIC_COHERENCE", "MODERATE", "Improve visual hierarchy, coherence, polish and balance without changing the football semantics."),
    "B3": ("CENTERED_ATTENTION", "MODERATE", "Strengthen a clear focal point around the most important football subject."),
}

MASTER_POLICY = (
    "Create a research candidate thumbnail for the SAME underlying football content. Preserve the original topic, "
    "team/player/event context and semantic meaning. Do not invent a different match, player, score, transfer, "
    "injury, trophy, quote, news event, club affiliation or factual claim. Visual presentation may change for a "
    "controlled same-content comparison. Use only research-supported design directions. Do not claim expected "
    "performance, future views, guaranteed uplift or causal performance."
)

NEGATIVE_PROMPT = (
    "excessive text, multiple competing headlines, fake breaking news, fabricated score, fabricated statistics, "
    "fabricated transfer or injury claim, incorrect football identity, wrong club context, duplicated player, "
    "deformed face, extra limbs, distorted football, illegible typography, overcrowded composition, unnecessary "
    "stickers, random arrows, excessive circles, extreme saturation, watermarks, misleading claims"
)


@dataclass(frozen=True)
class StructuredPrompt:
    strategy_id: str
    strategy_name: str
    diagnosis: dict[str, Any]
    evidence_level: str
    evidence_summary: str
    primary_objective: str
    protected_elements: list[str]
    editable_elements: list[str]
    generation_prompt: str
    negative_prompt: str
    expected_feature_direction: list[str]
    research_caveat: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose(features: dict[str, Any]) -> dict[str, Any]:
    """Return measurements plus relative/qualitative readings—never universal cutoffs."""
    vals = {name: features.get(key) for name, key in FEATURES.items()}
    return {
        "measurements": vals,
        "text_guidance": "Measured for directional comparison; excessive coverage should be reduced, without a universal numeric target.",
        "entropy_guidance": "Use as a relative clutter signal in same-content comparisons.",
        "center_guidance": "Central or near-central attention is useful when compositionally appropriate.",
        "face_guidance": "Face presence is contextual; never invent a person.",
    }


def _protected(title: str, topic: str) -> list[str]:
    return [
        f"factual meaning of source title: {title or 'not supplied'}",
        f"underlying football topic: {topic or 'football'}",
        "main football subject and relevant player/team identity when visible or supplied",
        "event or competition context when grounded in the source",
        "If an uncertain factual element cannot be preserved reliably, simplify or omit it instead of inventing it.",
    ]


def build_prompt(strategy_id: str, features: dict[str, Any], *, title: str = "", topic: str = "football", content_brief: str = "") -> dict[str, Any]:
    if strategy_id not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_id}")
    name, level, objective = STRATEGIES[strategy_id]
    d = diagnose(features)
    m = d["measurements"]
    safeguards = []
    entropy = m.get("visual_entropy")
    center = m.get("center_saliency")
    faces = m.get("face_presence")
    aesthetic = m.get("aesthetic_coherence")
    # These are presentation diagnoses, not claimed universal optima.
    if entropy is not None and float(entropy) > 0.65:
        safeguards.append("The measured entropy is comparatively high; prevent new clutter and competing micro-elements.")
    if center is not None and float(center) > 0.45:
        safeguards.append("Focal attention is already comparatively strong; preserve it and do not recenter mechanically.")
    if aesthetic is not None and float(aesthetic) > 0.65:
        safeguards.append("Aesthetic coherence is comparatively strong; avoid destructive restyling.")
    if faces is not None:
        safeguards.append("No face is detected; do not invent one unless the supplied context clearly requires it." if float(faces) <= 0 else
                          "A face is detected; preserve the same identity and context rather than creating a different person.")

    if strategy_id == "B1":
        text = m.get("text_coverage")
        if text is not None and float(text) < 0.08:
            mechanism = ("Text coverage is not identified as the primary visual weakness. Preserve the compact text structure and focus on hierarchy. ")
        else:
            mechanism = ("Remove redundant wording; preserve only core textual meaning; reduce competing text blocks and decorative pseudo-news labels. ")
        mechanism += ("Use spacing and hierarchy instead of more text; keep typography away from the player, face, ball and main subject; maintain football-subject dominance.")
        editable = ["wording redundancy", "text-block count", "typographic hierarchy", "spacing and overlay placement"]
        direction = ["directionally lower excessive text coverage", "clearer text hierarchy", "preserved subject prominence"]
        summary = "High-confidence directional evidence supports avoiding excessive text coverage; it does not establish a universal percentage."
    elif strategy_id == "B2":
        mechanism = ("Use one dominant primary subject; subordinate secondary elements; reduce unnecessary overlap; clarify foreground/background separation; "
                     "use coherent lighting and tonal treatment, consistent visual language, helpful negative space and balanced placement; integrate text into the hierarchy; do not maximize saturation for attention.")
        editable = ["layout hierarchy", "overlap", "background separation", "lighting and tonal treatment", "negative space"]
        direction = ["higher relative aesthetic coherence", "lower avoidable clutter", "balanced spatial hierarchy"]
        summary = "Moderate cross-method evidence supports testing aesthetic coherence and avoiding excessive entropy."
    else:
        mechanism = ("Identify one dominant football subject and strengthen immediate focus through scale, separation, depth, contrast and framing; "
                     "reduce competing high-saliency background elements; preserve necessary football context; keep text away from the focal region; use central or near-central attention only when compositionally appropriate.")
        editable = ["subject scale", "depth and separation", "contrast", "framing", "background saliency", "text placement"]
        direction = ["clearer focal attention", "reduced competing saliency", "preserved football context"]
        summary = "Moderate evidence supports testing clear focal attention; exact centering is not a universal rule."

    context = f"Source context — title: {title or 'not supplied'}; topic: {topic or 'football'}"
    if content_brief:
        context += f"; brief: {content_brief}"
    prompt = "\n\n".join([MASTER_POLICY, context, f"PRIMARY OBJECTIVE: {objective}", mechanism, *safeguards,
                            "PROTECTED ELEMENTS:\n- " + "\n- ".join(_protected(title, topic))])
    return StructuredPrompt(strategy_id, name, d, level, summary, objective, _protected(title, topic), editable,
                            prompt, NEGATIVE_PROMPT, direction,
                            "Observational research guidance only. Evaluate only same-content candidates; exploratory findings are diagnostics, not hard commands.").to_dict()


def build_prompt_set(features: dict[str, Any], **context: str) -> list[dict[str, Any]]:
    return [build_prompt(sid, features, **context) for sid in ("B1", "B2", "B3")]


def generator_status() -> dict[str, str]:
    return {"status": "GENERATOR_NOT_CONFIGURED", "message": "The evidence-guided generation instructions remain fully usable and candidate images may be generated externally and uploaded manually."}
