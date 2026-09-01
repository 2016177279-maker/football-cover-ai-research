from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
from .schemas import CandidateInstruction, PrototypeInput

STRATEGIES = (
    ("B1", "TEXT_SIMPLIFICATION", "Reduce excessive text coverage; retain only essential, correctly spelled title text and keep the core football subject, teams, event, and visual identity recognizable.", ["avoid excessive text coverage"]),
    ("B2", "AESTHETIC_COHERENCE", "Improve compositional and color coherence and remove distracting clutter while preserving all match, player, club, and event semantics.", ["increase aesthetic coherence", "avoid excessive visual entropy"]),
    ("B3", "CENTERED_ATTENTION", "Strengthen centered attention around the principal football subject; use limited face/person emphasis only where appropriate and preserve content identity.", ["test centered saliency", "test limited face presence"]),
)

class GenerationProvider(ABC):
    @abstractmethod
    def generate_candidates(self, request: PrototypeInput) -> list[CandidateInstruction]: ...

class InstructionOnlyProvider(GenerationProvider):
    def generate_candidates(self, request: PrototypeInput) -> list[CandidateInstruction]:
        identity = [request.title or "content theme", request.topic, "principal football subjects/teams/event"]
        return [CandidateInstruction(cid, strategy, text, identity,
                "Tier 1 mandatory: avoid excessive text coverage; do not add unnecessary copy.", soft,
                ["saturation direction", "global edge complexity", "joint center/global edge allocation"])
                for cid, strategy, text, soft in STRATEGIES]

class OptionalGeneratorProvider(GenerationProvider):
    """Adapter for an authorized callable; credentials are never required by the core."""
    def __init__(self, generator: Callable[[PrototypeInput, list[CandidateInstruction]], list[str]] | None = None):
        self.generator = generator
    def generate_candidates(self, request: PrototypeInput) -> list[CandidateInstruction]:
        instructions = InstructionOnlyProvider().generate_candidates(request)
        if self.generator is None:
            return instructions
        paths = self.generator(request, instructions)
        if len(paths) != len(instructions): raise ValueError("generator returned wrong candidate count")
        for item, path in zip(instructions, paths): item.mode, item.image_path = "OPTIONAL_GENERATOR", str(path)
        return instructions

def generate_candidates(request: PrototypeInput, provider: GenerationProvider | None = None):
    return (provider or InstructionOnlyProvider()).generate_candidates(request)
