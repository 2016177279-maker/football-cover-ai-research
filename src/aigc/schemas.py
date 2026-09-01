from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

TEMPORAL_WARNING = ("The model showed useful grouped held-out performance but weak "
                    "later-period temporal generalization. Scores are research-oriented "
                    "relative rankings, not validated prospective performance forecasts.")
PIPELINE_VERSION = "aigc_research_prototype_v1"
FEATURE_PIPELINE_VERSION = "basic_cv_v1+advanced_cv_v1+clip_vit_b_32"

class CandidateStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECT_AND_REGENERATE = "REJECT_AND_REGENERATE"
    REJECT_NO_SCORE = "REJECT_NO_SCORE"

@dataclass(frozen=True)
class PrototypeInput:
    video_id: str
    original_thumbnail: str
    title: str
    topic: str
    author_tier: str
    subscriber_count: float
    publish_month: str
    content_brief: str = ""

    def nonvisual_context(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "title": self.title, "topic": self.topic,
                "author_tier": self.author_tier, "subscriber_count": self.subscriber_count,
                "publish_month": self.publish_month, "content_brief": self.content_brief}

@dataclass
class CandidateInstruction:
    candidate_id: str
    strategy: str
    instruction: str
    preserved_semantic_content_identity: list[str]
    tier_1_constraint: str
    tier_2_preferences: list[str]
    excluded_primary_drivers: list[str]
    mode: str = "INSTRUCTION_ONLY"
    image_path: str = ""

    def to_dict(self): return asdict(self)

@dataclass
class ValidityResult:
    gates: dict[str, bool]
    status: CandidateStatus
    reasons: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self); d["status"] = self.status.value; return d
