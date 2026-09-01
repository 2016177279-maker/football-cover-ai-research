"""Offline-first Streamlit proof of concept for the public portfolio."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:  # Allows fixture/prompt helpers to be tested without the UI extra.
    st = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aigc.prompt_optimizer_v1_1 import diagnose
from src.aigc.prompt_optimizer_v1_2 import build_integrated_prompt, build_prompt_set

FEATURE_FILE = ROOT / "examples" / "synthetic_features.csv"
OFFLINE_DEMO = "OFFLINE_DEMO"
OPTIONAL_EXTERNAL_MODEL = "OPTIONAL_EXTERNAL_MODEL"


def load_synthetic_features(path: Path = FEATURE_FILE) -> pd.DataFrame:
    """Load the deliberately synthetic, credential-free demo fixture."""
    frame = pd.read_csv(path)
    if frame.empty or not frame["video_id"].astype(str).str.startswith("demo_video_").all():
        raise ValueError("demo input must contain only synthetic demo_video_* rows")
    return frame


def feature_payload(row: pd.Series) -> dict[str, float]:
    return {
        "advanced_cv_text_area_ratio": float(row["text_coverage"]),
        "advanced_cv_aesthetic_score": float(row["aesthetic_coherence"]),
        "basic_cv_entropy": float(row["visual_entropy"]),
        "advanced_cv_saliency_center_share": float(row["center_saliency"]),
        "advanced_cv_face_count": float(row["face_presence"]),
        "advanced_cv_face_area_ratio": float(row["face_presence"]) * 0.08,
        "advanced_cv_person_area_ratio": float(row["subject_prominence"]),
        "advanced_cv_largest_subject_area_ratio": float(row["subject_prominence"]),
    }


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required for the UI; install requirements.txt")
    st.set_page_config(page_title="Football Cover AI — Offline PoC", page_icon="⚽", layout="wide")
    st.title("Football Cover AI — evidence-guided optimization PoC")
    st.warning(
        "Research prototype only: this is not a CTR predictor and does not guarantee "
        "views, engagement, or uplift. Generated candidates require human review."
    )
    mode = st.radio("Runtime mode", [OFFLINE_DEMO, OPTIONAL_EXTERNAL_MODEL], horizontal=True)
    if mode == OPTIONAL_EXTERNAL_MODEL:
        st.info(
            "No external model is bundled. Export the prompt to a provider you are "
            "authorized to use, then review its output manually."
        )

    frame = load_synthetic_features()
    selected_id = st.selectbox("Synthetic example", frame["video_id"].tolist())
    row = frame.loc[frame["video_id"] == selected_id].iloc[0]
    features = feature_payload(row)
    title = st.text_input("Content title", "Demo FC match analysis")
    topic = st.text_input("Topic", "football analysis")
    brief = st.text_area("Protected content brief", "Keep the same fictional team, player, and event context.")

    st.subheader("Synthetic feature diagnostics")
    st.dataframe(pd.DataFrame([diagnose(features)["measurements"]]), use_container_width=True)
    prompt = build_integrated_prompt(features, title=title, topic=topic, content_brief=brief)
    st.subheader("Evidence-guided prompt")
    st.write(prompt["evidence_summary"])
    st.code(prompt["generation_prompt"], language="text")
    st.caption(prompt["research_caveat"])

    with st.expander("Show isolated B1 / B2 / B3 mechanisms"):
        for item in build_prompt_set(features, title=title, topic=topic, content_brief=brief):
            st.markdown(f"**{item['strategy_id']}**")
            st.code(item["generation_prompt"], language="text")

    st.subheader("Human review")
    st.checkbox("Identity and event semantics are preserved")
    st.checkbox("No fabricated score, transfer, injury, trophy, quote, or news event")
    st.checkbox("Text is readable and no obvious visual artifact is present")
    st.selectbox("Decision", ["Needs review", "Keep original", "Keep candidate for testing"])
    st.caption("Human decisions are not fed back into model training in this public demo.")


if __name__ == "__main__":
    main()
