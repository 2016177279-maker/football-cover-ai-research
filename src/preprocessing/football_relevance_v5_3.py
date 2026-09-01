"""Football Filter v5.3: typed conflicts and independent evidence fusion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


def normalize(value: object) -> str:
    raw = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", raw)
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c)).lower()).strip()


def contains(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    found=[]
    for term in terms:
        token=normalize(term)
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) if len(token)<=4 else token in text:
            found.append(term)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class TypedConflict:
    kind: str
    severity: str
    evidence: str
    grounded: bool = True


@dataclass(frozen=True, slots=True)
class V53Evidence:
    positive_sources: tuple[str, ...]
    positive_details: tuple[str, ...]
    conflicts: tuple[TypedConflict, ...]
    visual_strength: str
    multilingual: bool


@dataclass(frozen=True, slots=True)
class V53Result:
    relevance_decision: str
    content_subtype: str | None
    independent_positive_sources: tuple[str, ...]
    independent_positive_count: int
    typed_conflicts: tuple[TypedConflict, ...]
    abstain_risk: str | None
    abstain_reason: str | None
    decision_reason: str
    confidence: float


SOCCER=("soccer","futbol","fútbol","futebol","association football","calcio")
ACTIONS=("match","matches","highlights","goal","goals","tactics","tactical","prediction","predictions","transfer","training","drill","fixture","resumen","goles","golazo","partido","partida","maç","maca","jornada","assist")
EVENTS=("champions league","premier league","la liga","laliga","liga mx","fifa world cup","uefa","europa league","copa","bundesliga","serie a","nwsl","mls","community shield","fa cup","world cup")
ENTITIES=("arsenal","barcelona","real madrid","bayern","psg","manchester city","man city","manchester united","man united","liverpool","chelsea","everton","juventus","inter milan","ac milan","dortmund","atletico","ronaldinho","messi","ronaldo","harry kane","lamine yamal","villarreal","schalke")
FOOTBALL_CHANNEL=("football","soccer","futbol","fútbol","laliga","liga mx","sky sports football","tudn","espn deportes","unisport","nwsl","fc barcelona")
VISUAL_STRONG=("soccer match","soccer player","soccer field","football match","football pitch","goalkeeper","goalpost","scoring a goal","players in jerseys","football game strategy")
VISUAL_MEDIUM=("soccer","football","pitch","field","player","jersey","kit","goal","stadium","sports ball","match")


def _strong_negative(text: str, visual: str) -> list[TypedConflict]:
    combined=f"{text} {visual}"; out=[]
    rules=(
        ("cross_sport_conflict",("nfl","super bowl","quarterback","touchdown","college football","american football")),
        ("cross_sport_conflict",("rugby","nrl","all blacks","betfred super league","australian football league","afl")),
        ("cross_sport_conflict",("cricket"," t20 ","wicket","innings","ipl","tnpl")),
        ("gaming_conflict",("gameplay","video game","efootball","e-football","esports","e-sports","ps5","pes 21","simulation")),
        ("movie_fiction_conflict",("full movie","movie recap","movie scene","movie scenes","full scene","fictional","plot recap","alur cerita")),
        ("geography_conflict",("sagrada familia","barcelona travel","barcelona tourism","things to do in barcelona","barcelona hotel")),
    )
    for kind,terms in rules:
        hits=contains(combined,terms)
        if hits: out.append(TypedConflict(kind,"strong",", ".join(hits)))
    workout=contains(text,("full body cardio","interactive warm up","interactive warm-up","generic workout","fitness viral"))
    if workout: out.append(TypedConflict("workout_scope_conflict","strong",", ".join(workout)))
    return out


def build_evidence(*, title: object, description: object, channel: object, text_positive: object="", entity_positive: object="", visual_positive: object="", visual_negative: object="", raw_conflicts: object="", hard_negative: object="", vlm_confidence: object=None) -> V53Evidence:
    t,d,c=map(normalize,(title,description,channel)); meta=f"{t} {d}"; visual=normalize(f"{visual_positive} {visual_negative}")
    sources=[];details=[]
    soccer=contains(meta,SOCCER); actions=contains(meta,ACTIONS); events=contains(meta,EVENTS); entities=contains(meta,ENTITIES)
    legacy_text=normalize(text_positive);legacy_entity=normalize(entity_positive)
    if soccer or actions or legacy_text:
        sources.append("text");details.append(f"text:{','.join((*soccer,*actions)) or legacy_text[:120]}")
    if entities or legacy_entity:
        sources.append("entity");details.append(f"entity:{','.join(entities) or legacy_entity[:120]}")
    if events or "competition:" in legacy_entity:
        sources.append("event");details.append(f"event:{','.join(events) or legacy_entity[:120]}")
    channel_hits=contains(c,FOOTBALL_CHANNEL)
    if channel_hits:
        sources.append("channel");details.append(f"channel:{','.join(channel_hits)}")
    visual_strength="none"
    strong_visual_hits=contains(visual,VISUAL_STRONG); medium_visual_hits=contains(visual,VISUAL_MEDIUM)
    if strong_visual_hits or (len(medium_visual_hits)>=2 and contains(visual,("soccer","association football"))): visual_strength="strong"
    elif medium_visual_hits: visual_strength="medium"
    confidence=float(vlm_confidence) if str(vlm_confidence).strip() not in {"","nan","None"} else 0.0
    if visual_strength!="none" and confidence>=.70:
        sources.append("visual");details.append(f"visual:{visual[:180]}")
    conflicts=_strong_negative(meta,visual)
    hard=normalize(hard_negative)
    kind_map={"american_football":"cross_sport_conflict","rugby":"cross_sport_conflict","cricket":"cross_sport_conflict","fictional_media_center":"movie_fiction_conflict","tourism_center":"geography_conflict","general_fitness_center":"workout_scope_conflict","video_game_center":"gaming_conflict"}
    for key,kind in kind_map.items():
        if key in hard and not any(x.kind==kind for x in conflicts): conflicts.append(TypedConflict(kind,"strong",key))
    raw=normalize(raw_conflicts)
    if raw:
        if "barcelona city" in raw: conflicts.append(TypedConflict("geography_conflict","medium",raw))
        elif any(x in raw for x in ("rugby","nfl","american football","cricket","movie","fiction","tourism")):
            conflicts.append(TypedConflict("weak_semantic_conflict","medium",raw,False))
        elif raw not in {"no conflict","match","soccer player","general news","champions league news","la liga noticias"}:
            conflicts.append(TypedConflict("weak_semantic_conflict","weak",raw,False))
    # Specific football reporting is not general news. Only explicit generic-news center with no event/entity is medium.
    general=contains(meta,("world news","general news","actualidad nacional e internacional","noticias de argentina y el mundo"))
    if general and not (events or entities): conflicts.append(TypedConflict("general_news_conflict","strong",", ".join(general)))
    return V53Evidence(tuple(dict.fromkeys(sources)),tuple(details),tuple(conflicts),visual_strength,bool(soccer and any(ord(ch)>127 for ch in str(title)+str(description))))


def classify_subtype_v3(title: object, description: object) -> str:
    text=normalize(f"{title} {description}")
    derivative=contains(text,("opening ceremony","closing ceremony","world cup anthem","official anthem","halftime show","music performance","world cup music","football-event concert"))
    return "football_entertainment_derivative" if derivative else "football_core"


def fuse_evidence(e: V53Evidence, *, title: object, description: object) -> V53Result:
    sources=e.positive_sources; count=len(sources); strong=[x for x in e.conflicts if x.severity=="strong" and x.grounded]; medium=[x for x in e.conflicts if x.severity=="medium"]
    specific=bool(set(sources)&{"entity","event","visual"}); visual_rescue=e.visual_strength=="strong" and "visual" in sources and len(set(sources)-{"visual"})>=1
    if strong:
        return V53Result("reject",None,sources,count,e.conflicts,None,None,"grounded strong conflict: "+", ".join(sorted({x.kind for x in strong})),.98)
    if count>=2 and specific and not medium or count>=4 and specific:
        return V53Result("accept",classify_subtype_v3(title,description),sources,count,e.conflicts,None,None,"independent positive evidence accumulated",min(.97,.72+.05*count))
    if visual_rescue and count>=2 and not medium:
        return V53Result("accept",classify_subtype_v3(title,description),sources,count,e.conflicts,None,None,"protected visual rescue with independent corroboration",.88)
    if count==0 and not medium:
        return V53Result("reject",None,sources,count,e.conflicts,None,None,"no association-football evidence",.88)
    risk="abstain_high_risk" if medium else "abstain_low_risk"
    reason="typed positive/negative conflict" if medium else "positive evidence just below acceptance proof"
    return V53Result("abstain",None,sources,count,e.conflicts,risk,reason,reason,.52 if medium else .62)
