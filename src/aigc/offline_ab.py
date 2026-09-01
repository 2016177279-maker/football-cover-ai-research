from __future__ import annotations
from .ranking import rank_arms
from .schemas import TEMPORAL_WARNING

def assert_identical_context(arms: list[dict]) -> None:
    contexts=[a["nonvisual_context"] for a in arms]
    if any(c!=contexts[0] for c in contexts[1:]): raise ValueError("nonvisual context differs across arms")

def compare_arms(arms: list[dict]) -> dict:
    assert_identical_context(arms)
    ids=[x["candidate_id"] for x in arms]
    if len(ids)!=len(set(ids)) or "A" not in ids: raise ValueError("offline A/B arms must be unique and include A")
    ranked,best=rank_arms(arms)
    return {"simulation_type":"OFFLINE_A_B_SIMULATION","observed_ab_test":False,"arms":ranked,
            "BEST_RESEARCH_ARM":best,"temporal_risk_warning":TEMPORAL_WARNING}
