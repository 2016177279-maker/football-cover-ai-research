from src.preprocessing.football_relevance_v5_3 import build_evidence, classify_subtype_v3, fuse_evidence


def decide(**kwargs):
    title=kwargs.pop("title","");description=kwargs.pop("description","")
    return fuse_evidence(build_evidence(title=title,description=description,channel=kwargs.pop("channel",""),**kwargs),title=title,description=description)


def test_independent_sources_not_keyword_count() -> None:
    repeated=decide(title="football football football goals")
    assert repeated.independent_positive_count==1 and repeated.relevance_decision=="abstain"
    combined=decide(title="Barcelona vs Real Madrid match",channel="FC Barcelona",visual_positive="soccer match on a football pitch",vlm_confidence=.95)
    assert set(combined.independent_positive_sources)>={"text","entity","channel","visual"}
    assert combined.relevance_decision=="accept"


def test_typed_conflicts_have_severity_and_guardrail() -> None:
    result=decide(title="NFL Super Bowl quarterback highlights",visual_positive="football player",vlm_confidence=.9)
    assert result.relevance_decision=="reject"
    assert any(x.kind=="cross_sport_conflict" and x.severity=="strong" for x in result.typed_conflicts)


def test_vague_conflict_does_not_veto_real_match() -> None:
    result=decide(title="Barcelona vs Real Madrid match",channel="FC Barcelona",visual_positive="soccer match and goalpost",raw_conflicts="general news",vlm_confidence=.95)
    assert result.relevance_decision=="accept"


def test_protected_visual_rescue_requires_corroboration() -> None:
    visual_only=decide(title="Amazing",visual_positive="soccer match on pitch",vlm_confidence=.95)
    assert visual_only.relevance_decision!="accept"
    rescued=decide(title="match highlights",visual_positive="soccer match on pitch",vlm_confidence=.95)
    assert rescued.relevance_decision=="accept"
    blocked=decide(title="movie scenes match",visual_positive="soccer match on pitch",vlm_confidence=.95)
    assert blocked.relevance_decision=="reject"


def test_abstain_risk_stratification() -> None:
    low=decide(title="football tactics")
    assert low.relevance_decision=="abstain" and low.abstain_risk=="abstain_low_risk"
    high=decide(title="Barcelona football",raw_conflicts="Barcelona city versus club ambiguity")
    assert high.relevance_decision=="abstain" and high.abstain_risk=="abstain_high_risk"


def test_subtype_v3_runs_only_after_accept() -> None:
    derivative=decide(title="Official World Cup anthem",description="opening ceremony",channel="FIFA",visual_positive="soccer stadium match",vlm_confidence=.95)
    assert derivative.relevance_decision=="accept"
    assert derivative.content_subtype=="football_entertainment_derivative"
    fiction=decide(title="Soccer movie recap",description="full movie scenes",visual_positive="soccer match",vlm_confidence=.95)
    assert fiction.relevance_decision=="reject" and fiction.content_subtype is None
    assert classify_subtype_v3("Charity football match","players score goals")=="football_core"


def test_game_rugby_cricket_workout_regressions() -> None:
    cases=[("eFootball PS5 gameplay",""),("All Blacks match","rugby highlights"),("T20 World Cup","cricket innings"),("Interactive warm-up","full body cardio")]
    for title,description in cases:
        assert decide(title=title,description=description,visual_positive="football field",vlm_confidence=.9).relevance_decision=="reject"
