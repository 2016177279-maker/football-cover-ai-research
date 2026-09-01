from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from .feature_adapter import VISUAL_MODEL_FIELDS, assert_feature_schema
from .schemas import TEMPORAL_WARNING
from .uncertainty import prediction_interval

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=ROOT/"reports/final_v2_ml_shap_v1/models/velocity_C_VISUAL_CLIP_HistGradientBoosting_v1.joblib"
MODEL_VERSION="velocity_C_VISUAL_CLIP_HistGradientBoosting_v1"

def load_frozen_model(path: str|Path=MODEL_PATH):
    local=ROOT/".ml_deps"
    # Prefer compatible packages in the active canonical CV environment.  The
    # local wheel bundle is a fallback for lean environments and may contain
    # extension modules built for a different Python minor version.
    try:
        import joblib
        import sklearn  # noqa: F401
    except ImportError:
        if str(local) not in sys.path: sys.path.insert(0,str(local))
        import joblib
    # Bundles were serialized by the original script as __main__.Design.
    spec=importlib.util.spec_from_file_location("frozen_v1_pipeline",ROOT/"scripts/run_final_v2_ml_shap_v1.py")
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    import __main__; prior=getattr(__main__,"Design",None); __main__.Design=mod.Design
    try: bundle=joblib.load(path)
    finally:
        if prior is None: delattr(__main__,"Design")
        else: __main__.Design=prior
    if bundle.get("target")!="log1p_views_per_day" or bundle["design"].pca is None: raise ValueError("not the frozen selected velocity C_VISUAL_CLIP bundle")
    return bundle

def score_candidate(features: dict, context: dict, bundle=None) -> dict:
    assert_feature_schema(features); bundle=bundle or load_frozen_model()
    row={k:context[k] for k in ("topic","author_tier","publish_month")}
    row["log1p_subscriber_count"]=float(np.log1p(float(context["subscriber_count"])))
    row.update({k:features[k] for k in VISUAL_MODEL_FIELDS})
    frame=pd.DataFrame([row]); clip=pd.DataFrame([features["clip_embedding"]],columns=bundle["design"].clip_cols)
    point=float(bundle["model"].predict(bundle["design"].transform(frame,clip))[0]); lo,hi=prediction_interval(point)
    return {"point_prediction":point,"prediction_interval_lower":lo,"prediction_interval_upper":hi,
            "prediction_interval_nominal_coverage":.90,"research_propagation_score":point,
            "model_version":MODEL_VERSION,"temporal_risk_warning":TEMPORAL_WARNING}

def apply_research_heuristic(rows: list[dict]) -> list[dict]:
    """Apply frozen 65/20/10/5 policy; experimental diagnostics get zero weight."""
    valid=[r for r in rows if r.get("candidate_status")=="ELIGIBLE" and r.get("point_prediction") is not None]
    order={id(r):i for i,r in enumerate(sorted(valid,key=lambda x:x["point_prediction"]))}
    denom=max(1,len(valid)-1)
    for r in rows:
        if r not in valid: r["research_propagation_score"]=None; continue
        model_rank=order[id(r)]/denom if len(valid)>1 else .5
        t1=1.0 if r.get("tier_1_compliance") is True else 0.0
        vals=[v for v in r.get("tier_2_alignment",{}).values() if isinstance(v,bool)]
        t2=sum(vals)/len(vals) if vals else .5
        validity=1.0
        r["research_propagation_score"]=.65*model_rank+.20*t1+.10*t2+.05*validity
    return rows
