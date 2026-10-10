"""Opt-in tests against the user's *running* Blender (ANIMAGENT_LIVE_TESTS=1).

Strictly read-only: they never modify or save the open file.
"""

from __future__ import annotations

import os

import pytest

from animagent.pipeline import analyze_rig
from tests.integration.conftest import run_session

pytestmark = [pytest.mark.live, pytest.mark.timeout(300)]


def _armature() -> str:
    name = os.environ.get("ANIMAGENT_LIVE_ARMATURE")
    if name:
        return name
    info = run_session(None, lambda s: s.run_script("scene_info", {}))
    rigs = [a["name"] for a in info["armatures"] if a["rig_id"]]
    if not rigs:
        pytest.skip("no Rigify rig in the running scene")
    return rigs[0]


def test_live_rig_analysis_is_read_only() -> None:
    name = _armature()
    profile = run_session(None, lambda s: analyze_rig(s, name))
    assert profile.readonly_check is not None and profile.readonly_check.identical, profile.readonly_check
    kinds = {lb.kind.value for lb in profile.anatomy.limbs}
    assert {"arm", "leg"} <= kinds
    for lc in profile.controls.limbs:
        assert lc.fk_chain, lc.limb_id
