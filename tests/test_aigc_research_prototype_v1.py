from pathlib import Path

import pytest
from PIL import Image

from src.aigc.generation import InstructionOnlyProvider
from src.aigc.offline_ab import assert_identical_context, compare_arms
from src.aigc.ranking import rank_arms
from src.aigc.rules import rule_assessment, validate_candidate
from src.aigc.schemas import CandidateStatus, PrototypeInput, TEMPORAL_WARNING
from src.aigc.uncertainty import prediction_interval


def request() -> PrototypeInput:
    return PrototypeInput("demo_video_001", "fixture.png", "Demo football title", "match", "DEMO", 1000, "2026-01")


def test_instruction_only_provider_is_offline_and_distinct():
    items = InstructionOnlyProvider().generate_candidates(request())
    assert len({item.strategy for item in items}) == 3
    assert all("mandatory" in item.tier_1_constraint.lower() for item in items)


def test_identical_context_is_enforced():
    with pytest.raises(ValueError):
        assert_identical_context([{"nonvisual_context": {"x": 1}}, {"nonvisual_context": {"x": 2}}])


def test_uncertainty_and_conservative_ranking():
    assert prediction_interval(5)[1] > prediction_interval(5)[0]
    rows = [
        {"candidate_id": "A", "candidate_status": "ELIGIBLE", "point_prediction": 1, "research_propagation_score": 1, "prediction_interval_lower": 0, "prediction_interval_upper": 2},
        {"candidate_id": "B1", "candidate_status": "ELIGIBLE", "point_prediction": 1.1, "research_propagation_score": 1.1, "prediction_interval_lower": 0.1, "prediction_interval_upper": 2.1},
    ]
    assert rank_arms(rows)[1] == "NO_CLEAR_MODEL_PREFERENCE"


def test_candidate_validity_handles_invalid_images(tmp_path: Path):
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (10, 10)).save(tiny)
    assert validate_candidate(str(tiny)).status == CandidateStatus.REJECT_AND_REGENERATE
    assert validate_candidate(str(tmp_path / "missing.png")).status == CandidateStatus.REJECT_NO_SCORE


def test_assessment_copy_is_conservative():
    result = rule_assessment({"basic_cv_saturation_mean": 1}, {"basic_cv_saturation_mean": 0})
    assert result["experimental_in_primary_score"] is False
    text = " ".join(item.instruction for item in InstructionOnlyProvider().generate_candidates(request()))
    assert "will increase views" not in text.lower()
    assert "temporal" in TEMPORAL_WARNING.lower()


def test_offline_ab_arm_accounting():
    base = {"candidate_status": "REJECT_NO_SCORE", "point_prediction": None, "research_propagation_score": None, "nonvisual_context": {"x": 1}}
    result = compare_arms([{"candidate_id": arm, **base} for arm in ("A", "B1", "B2", "B3")])
    assert len(result["arms"]) == 4
    assert result["observed_ab_test"] is False
