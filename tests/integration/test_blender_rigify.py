"""Integration tests against a real (background) Blender through the official MCP server."""

from __future__ import annotations

from pathlib import Path

import pytest

from animagent.cli import main
from animagent.models.profile import RigProfile
from animagent.pipeline import analyze_rig, extract, fingerprint
from tests.integration.conftest import run_session

pytestmark = [pytest.mark.blender, pytest.mark.timeout(300)]


def test_analyze_generated_rigify_human(rigify_port: int) -> None:
    profile = run_session(rigify_port, lambda s: analyze_rig(s, "Rig"))
    a, cm = profile.anatomy, profile.controls
    assert profile.skeleton.strategy == "rigify_org"
    assert profile.extraction.rigify.metarig == "metarig"
    # Body frame: Rigify characters face -Y.
    assert a.body_frame.forward == pytest.approx([0, -1, 0], abs=0.05)
    assert a.body_frame.confidence >= 0.9
    segs = a.segment_map()
    assert segs["pelvis"].source_bones == ["ORG-spine"]
    assert segs["head"].source_bones == ["ORG-spine.006"]
    assert segs["neck"].source_bones == ["ORG-spine.004", "ORG-spine.005"]
    limbs = {lb.id: lb for lb in a.limbs}
    assert {"arm.L", "arm.R", "leg.L", "leg.R"} <= set(limbs)
    assert limbs["arm.L"].segments == ["clavicle.L", "upper_arm.L", "forearm.L", "hand.L"]
    assert limbs["leg.R"].segments == ["thigh.R", "shin.R", "foot.R", "toe.R"]
    assert len(limbs["arm.L"].digits) == 5
    j = a.joint_map()
    assert j["knee.L"].bend and j["knee.L"].bend.matches_prior is True
    assert j["elbow.R"].bend and j["elbow.R"].bend.matches_prior is True
    assert j["knee.L"].bend.hinge_axis_local in ("+X", "-X")
    # Control mapping.
    arm = next(lc for lc in cm.limbs if lc.limb_id == "arm.L")
    assert [c.bone for c in arm.fk_chain] == ["shoulder.L", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]
    assert arm.ik_effector and arm.ik_effector.bone == "hand_ik.L"
    assert arm.ik_pole and arm.ik_pole.bone == "upper_arm_ik_target.L"
    sw = arm.ik_fk_switch
    assert sw and sw.bone == "upper_arm_parent.L" and sw.property == "IK_FK"
    assert sw.ik_value is not None and sw.fk_value is not None and sw.ik_value != sw.fk_value
    leg = next(lc for lc in cm.limbs if lc.limb_id == "leg.L")
    assert leg.ik_effector and leg.ik_effector.bone == "foot_ik.L"
    assert cm.root_control and cm.root_control.bone == "root"
    assert any(c.role == "torso" and c.bone == "torso" for c in cm.body_controls)
    # Weight summaries reached the anatomy.
    assert segs["thigh.L"].mass and segs["thigh.L"].mass.relative_mass > 0


def test_inspection_does_not_modify_scene(rigify_port: int) -> None:
    async def go(s):  # type: ignore[no-untyped-def]
        before = await fingerprint(s, "Rig")
        await extract(s, "Rig")
        await s.run_script("scene_info", {})
        mid = await fingerprint(s, "Rig")
        profile = await analyze_rig(s, "Rig")
        after = await fingerprint(s, "Rig")
        return before, mid, after, profile

    before, mid, after, profile = run_session(rigify_port, go)
    assert before.hashes == mid.hashes == after.hashes
    assert set(before.hashes) >= {"pose", "rest", "animation", "meshes", "vertex_weights", "objects", "datablocks"}
    assert profile.readonly_check and profile.readonly_check.identical
    # Freshly opened file stays clean: inspection never marks it as modified.
    assert before.info["is_dirty"] is False and after.info["is_dirty"] is False


def test_fingerprint_detects_real_changes(rigify_port: int) -> None:
    """Sanity check that the fingerprint is sensitive: change the pose, compare, undo exactly."""

    async def go(s):  # type: ignore[no-untyped-def]
        before = await fingerprint(s, "Rig")
        await s.execute(
            "import bpy\npb = bpy.data.objects['Rig'].pose.bones['hand_ik.L']\n"
            "bpy.app.driver_namespace['_t'] = tuple(pb.location)\npb.location.x += 0.1\nresult = {}"
        )
        changed = await fingerprint(s, "Rig")
        await s.execute(
            "import bpy\npb = bpy.data.objects['Rig'].pose.bones['hand_ik.L']\n"
            "pb.location = bpy.app.driver_namespace.pop('_t')\nresult = {}"
        )
        restored = await fingerprint(s, "Rig")
        return before, changed, restored

    before, changed, restored = run_session(rigify_port, go)
    assert changed.hashes["pose"] != before.hashes["pose"]
    assert restored.hashes["pose"] == before.hashes["pose"]


def test_metarig_analysed_as_generic_deform_rig(rigify_port: int) -> None:
    profile = run_session(rigify_port, lambda s: analyze_rig(s, "metarig"))
    assert profile.skeleton.strategy == "deform"
    assert profile.controls.rig_kind == "generic"
    assert {"arm.L", "arm.R", "leg.L", "leg.R"} <= {lb.id for lb in profile.anatomy.limbs}


def test_cli_against_real_blender(rigify_port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLENDER_MCP_PORT", str(rigify_port))
    assert main(["doctor"]) == 0
    out = tmp_path / "profile.json"
    assert main(["analyze-rig", "--armature", "Rig", "--output", str(out)]) == 0
    p = RigProfile.load(out)
    assert p.readonly_check and p.readonly_check.identical
    assert main(["verify-readonly", "--armature", "Rig"]) == 0
    # Overlay/viewport scripts degrade gracefully in background mode.
    assert main(["overlay", "status"]) == 0
