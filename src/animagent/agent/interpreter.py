"""GPT-6-assisted interpretation of ambiguous anatomy (optional, off by default).

The deterministic analysis raises `Ambiguity` records. An `AnatomyInterpreter` may propose
`Resolution`s for them. Resolutions are stored next to the deterministic results and are
only marked `accepted` when they pick one of the offered candidates with enough confidence;
they never overwrite deterministic data.

Only a compact, numeric summary of the relevant skeleton neighbourhood is sent to the model:
no meshes, vertices or weights.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from animagent.analysis.geom import unit, v
from animagent.config import LLMConfig
from animagent.logging_setup import get_logger
from animagent.models.anatomy import AnatomyProfile, Resolution, Skeleton

log = get_logger(__name__)

ACCEPT_THRESHOLD = 0.7

SYSTEM_PROMPT = """You are a character rigging and anatomy expert assisting an automated animation pipeline.
You receive ambiguities that a deterministic skeleton analysis could not resolve, together with a
compact numeric description of the relevant bones in the character's body frame
(x = character left, y = forward, z = up; lengths relative to body height).
For each ambiguity pick exactly one of the offered candidates, give a calibrated confidence in [0, 1]
and a one-sentence rationale grounded in the numbers. Use low confidence when the data does not
support a clear decision. Never invent candidates."""


class _LLMResolution(BaseModel):
    ambiguity_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class _LLMResolutionBatch(BaseModel):
    resolutions: list[_LLMResolution]


class AnatomyInterpreter(Protocol):
    def resolve(self, skeleton: Skeleton, anatomy: AnatomyProfile) -> list[Resolution]: ...


def build_context(skeleton: Skeleton, anatomy: AnatomyProfile, max_bones: int = 60) -> dict[str, Any]:
    """Compact numeric context for the ambiguities (body-frame coordinates, rounded)."""
    f = anatomy.body_frame
    R = np.array([v(f.left), v(f.forward), v(f.up)])
    origin, H = v(f.origin), max(f.height, 1e-6)
    bm = skeleton.bone_map()
    wanted: list[str] = []
    for a in anatomy.ambiguities:
        for name in a.subject:
            if name in bm:
                cur: str | None = name
                for _ in range(2):  # subject, its subtree root and parent for context
                    if cur and cur not in wanted:
                        wanted.append(cur)
                    cur = bm[cur].parent if cur else None
                stack = list(bm[name].children)
                while stack and len(wanted) < max_bones:
                    n = stack.pop()
                    if n not in wanted:
                        wanted.append(n)
                    stack.extend(bm[n].children)
    seg_of = {b: s.id for s in anatomy.segments for b in s.source_bones}

    def body(p: list[float]) -> list[float]:
        return [round(float(x), 3) for x in R @ ((v(p) - origin) / H)]

    bones = []
    for n in wanted[:max_bones]:
        b = bm[n]
        bones.append(
            {
                "name": n,
                "parent": b.parent,
                "side": b.side.value,
                "segment": seg_of.get(n),
                "head": body(b.head),
                "dir": [round(float(x), 2) for x in R @ unit(v(b.tail) - v(b.head))],
                "length": round(b.length / H, 3),
                "children": len(b.children),
            }
        )
    return {
        "body_height": round(H, 3),
        "segments": [
            {"id": s.id, "kind": s.kind.value, "side": s.side.value}
            for s in anatomy.segments
            if s.kind.value not in ("digit",)
        ],
        "bones": bones,
        "ambiguities": [
            {
                "id": a.id,
                "question": a.question,
                "candidates": a.candidates,
                "deterministic_best": a.deterministic_best,
                "context": a.context,
            }
            for a in anatomy.ambiguities
        ],
    }


class NullInterpreter:
    def resolve(self, skeleton: Skeleton, anatomy: AnatomyProfile) -> list[Resolution]:
        return []


class OpenAIAnatomyInterpreter:
    """Resolves ambiguities with GPT-6 through the OpenAI Responses API (structured output)."""

    def __init__(self, config: LLMConfig | None = None, client: Any = None) -> None:
        self.config = config or LLMConfig.from_env()
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def resolve(self, skeleton: Skeleton, anatomy: AnatomyProfile) -> list[Resolution]:
        ctx = build_context(skeleton, anatomy)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps(ctx),
            "text_format": _LLMResolutionBatch,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        log.info(
            "llm.request",
            model=self.config.model,
            ambiguities=len(ctx["ambiguities"]),
            context_chars=len(kwargs["input"]),
        )
        response = self.client.responses.parse(**kwargs)
        batch: _LLMResolutionBatch | None = response.output_parsed
        if batch is None:
            log.warning("llm.no_parsed_output")
            return []
        return [
            Resolution(
                ambiguity_id=r.ambiguity_id,
                label=r.label,
                confidence=r.confidence,
                rationale=r.rationale,
                source=f"llm:{self.config.model}",
            )
            for r in batch.resolutions
        ]


def apply_resolutions(anatomy: AnatomyProfile, resolutions: list[Resolution]) -> None:
    """Attach resolutions; accept only valid candidates above the confidence threshold."""
    amb = {a.id: a for a in anatomy.ambiguities}
    for r in resolutions:
        a = amb.get(r.ambiguity_id)
        r.accepted = bool(a is not None and r.label in a.candidates and r.confidence >= ACCEPT_THRESHOLD)
        anatomy.resolutions.append(r)
