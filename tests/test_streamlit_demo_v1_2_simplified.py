from pathlib import Path

from demo.app import FEATURE_FILE, OFFLINE_DEMO, feature_payload, load_synthetic_features
from src.aigc.prompt_optimizer_v1_2 import build_integrated_prompt


def test_offline_fixture_is_synthetic_and_small():
    frame = load_synthetic_features()
    assert 10 <= len(frame) <= 20
    assert frame["video_id"].str.startswith("demo_video_").all()
    assert FEATURE_FILE == Path(__file__).resolve().parents[1] / "examples" / "synthetic_features.csv"


def test_offline_feature_mapping_and_prompt_generation():
    row = load_synthetic_features().iloc[0]
    prompt = build_integrated_prompt(feature_payload(row), title="Demo title", topic="football", content_brief="Fictional event")
    assert prompt["generation_prompt"]
    assert prompt["research_caveat"]
    assert OFFLINE_DEMO == "OFFLINE_DEMO"


def test_demo_copy_is_conservative_and_credential_free():
    source = (Path(__file__).resolve().parents[1] / "demo" / "app.py").read_text()
    lowered = source.lower()
    assert "not a ctr predictor" in lowered
    assert "does not guarantee" in lowered
    assert "youtube_api_key" not in lowered
    assert "/users/" not in lowered
