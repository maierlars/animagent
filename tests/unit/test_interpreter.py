import json
from types import SimpleNamespace
from typing import Any

from animagent.agent.interpreter import (
    OpenAIAnatomyInterpreter,
    _LLMResolution,
    _LLMResolutionBatch,
    apply_resolutions,
    build_context,
)
from animagent.analysis.anatomy import analyze_anatomy
from animagent.analysis.skeleton import build_skeleton
from animagent.config import LLMConfig
from animagent.models.anatomy import Resolution
from tests.fakes.synthetic import humanoid_extraction


class FakeResponses:
    def __init__(self, batch: _LLMResolutionBatch) -> None:
        self.batch = batch
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.batch)


def _setup():  # type: ignore[no-untyped-def]
    ex = humanoid_extraction(accessory=True, knee_backward=True)
    sk = build_skeleton(ex)
    return sk, analyze_anatomy(ex, sk)


def test_context_is_compact_and_numeric() -> None:
    sk, a = _setup()
    ctx = build_context(sk, a)
    text = json.dumps(ctx)
    assert len(text) < 20_000
    assert {x["id"] for x in ctx["ambiguities"]} == {x.id for x in a.ambiguities}
    names = {b["name"] for b in ctx["bones"]}
    assert "feather.L" in names and "thigh.L" in names
    assert "weights" not in text and "vertex" not in text


def test_openai_interpreter_uses_responses_api_and_keeps_results_separate() -> None:
    sk, a = _setup()
    batch = _LLMResolutionBatch(
        resolutions=[
            _LLMResolution(ambiguity_id="appendage:feather", label="feather_group", confidence=0.9, rationale="r"),
            _LLMResolution(
                ambiguity_id="bend:knee.L", label="anatomical_reversed_joint", confidence=0.5, rationale="r"
            ),
            _LLMResolution(ambiguity_id="bend:knee.R", label="invented_label", confidence=0.95, rationale="r"),
        ]
    )
    fake = FakeResponses(batch)
    interp = OpenAIAnatomyInterpreter(LLMConfig(model="gpt-6"), client=SimpleNamespace(responses=fake))
    before = a.model_dump(exclude={"resolutions"})
    res = interp.resolve(sk, a)
    apply_resolutions(a, res)
    assert fake.calls[0]["model"] == "gpt-6"
    assert fake.calls[0]["text_format"] is _LLMResolutionBatch
    accepted = {r.ambiguity_id: r.accepted for r in a.resolutions}
    assert accepted == {"appendage:feather": True, "bend:knee.L": False, "bend:knee.R": False}
    assert all(r.source == "llm:gpt-6" for r in a.resolutions)
    assert a.model_dump(exclude={"resolutions"}) == before  # deterministic data untouched


def test_apply_resolutions_rejects_unknown_ambiguity() -> None:
    _, a = _setup()
    apply_resolutions(a, [Resolution(ambiguity_id="nope", label="x", confidence=1.0, rationale="", source="t")])
    assert a.resolutions[0].accepted is False
