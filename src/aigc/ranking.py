from __future__ import annotations
from .uncertainty import interval_confidence

def rank_arms(rows: list[dict]) -> tuple[list[dict],str]:
    eligible=[r for r in rows if r.get("candidate_status")=="ELIGIBLE" and r.get("point_prediction") is not None]
    for r in rows: r["relative_rank"]=None; r["ranking_confidence"]="NOT_COMPARABLE"
    eligible.sort(key=lambda r:r["research_propagation_score"],reverse=True)
    for i,r in enumerate(eligible,1): r["relative_rank"]=i
    if len(eligible)<2: return rows,"NO_CLEAR_MODEL_PREFERENCE"
    top,next_=eligible[:2]
    conf,_=interval_confidence((top["prediction_interval_lower"],top["prediction_interval_upper"]),(next_["prediction_interval_lower"],next_["prediction_interval_upper"]))
    top["ranking_confidence"]=conf
    return rows, top["candidate_id"] if conf != "LOW" and top["point_prediction"] != next_["point_prediction"] else "NO_CLEAR_MODEL_PREFERENCE"
