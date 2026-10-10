import numpy as np
import pytest

from animagent.analysis.anatomy import analyze_anatomy
from animagent.analysis.skeleton import build_skeleton
from animagent.models.anatomy import SegmentKind
from tests.fakes.synthetic import humanoid_extraction, rigify_like_extraction


def _anatomy(**kw):  # type: ignore[no-untyped-def]
    ex = humanoid_extraction(**kw)
    return analyze_anatomy(ex, build_skeleton(ex))


def test_body_frame_humanoid() -> None:
    a = _anatomy()
    f = a.body_frame
    assert f.up == [0, 0, 1]
    assert np.allclose(f.forward, [0, -1, 0])
    assert np.allclose(f.left, [1, 0, 0])
    assert f.height == pytest.approx(1.75)
    assert f.confidence > 0.9
    assert any(e.kind == "feet" and e.weight > 0 for e in f.evidence)


def test_segments_and_limbs() -> None:
    a = _anatomy()
    segs = a.segment_map()
    assert segs["pelvis"].source_bones == ["hips"]
    assert segs["spine"].source_bones == ["spine"]
    assert segs["chest"].source_bones == ["chest"]
    assert segs["neck"].source_bones == ["neck"]
    assert segs["head"].source_bones == ["head"]
    assert {lb.id for lb in a.limbs} == {"arm.L", "arm.R", "leg.L", "leg.R"}
    arm = next(lb for lb in a.limbs if lb.id == "arm.L")
    assert arm.segments == ["clavicle.L", "upper_arm.L", "forearm.L", "hand.L"]
    assert len(arm.digits) == 2
    leg = next(lb for lb in a.limbs if lb.id == "leg.L")
    assert leg.segments == ["thigh.L", "shin.L", "foot.L", "toe.L"]
    assert leg.reaches_ground
    assert segs["foot.L"].source_bones == ["foot.L", "heel.L"]
    assert not a.ambiguities
    assert not a.unassigned_bones


def test_joints_and_contacts() -> None:
    a = _anatomy()
    j = a.joint_map()
    for jid in ("hip.L", "knee.L", "ankle.L", "ball.L", "shoulder.R", "elbow.R", "wrist.R", "neck_base", "skull_base"):
        assert jid in j
    assert j["knee.L"].position == [0.1, -0.03, 0.55]
    contacts = {c.id: c for c in a.contacts}
    assert contacts["heel.L"].position == pytest.approx([0.1, 0.03, 0.0])
    assert contacts["toe_tip.L"].position == pytest.approx([0.1, -0.18, 0.03])


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return vec * np.cos(angle) + np.cross(axis, vec) * np.sin(angle) + axis * (axis @ vec) * (1 - np.cos(angle))


def test_bend_directions_follow_humanoid_prior() -> None:
    a = _anatomy()
    j = a.joint_map()
    knee, elbow = j["knee.L"].bend, j["elbow.L"].bend
    assert knee is not None and elbow is not None
    assert knee.body_relative["forward"] > 0.9 and knee.matches_prior
    assert elbow.body_relative["forward"] < -0.9 and elbow.matches_prior
    # The hinge axis is oriented so that a positive rotation of the lower segment increases flexion.
    for jid, upper, lower in (("knee.L", "thigh.L", "shin.L"), ("elbow.L", "upper_arm.L", "forearm.L")):
        b = j[jid].bend
        segs = a.segment_map()
        u = np.subtract(segs[upper].tail, segs[upper].head)
        low = np.subtract(segs[lower].tail, segs[lower].head)
        before = np.arccos(u @ low / np.linalg.norm(u) / np.linalg.norm(low))
        rotated = _rotate(low, np.array(b.hinge_axis), 0.2)
        after = np.arccos(u @ rotated / np.linalg.norm(u) / np.linalg.norm(rotated))
        assert after > before
    assert knee.hinge_axis_local in ("+X", "-X")


def test_reversed_knee_is_flagged_not_overridden() -> None:
    a = _anatomy(knee_backward=True)
    knee = a.joint_map()["knee.L"].bend
    assert knee is not None and knee.matches_prior is False
    assert knee.body_relative["forward"] < -0.9
    amb = {x.id: x for x in a.ambiguities}
    assert "bend:knee.L" in amb
    assert "anatomical_reversed_joint" in amb["bend:knee.L"].candidates


def test_swapped_side_names_reduce_frame_confidence() -> None:
    a = _anatomy(swap_sides=True)
    assert a.body_frame.confidence <= 0.7
    assert "forward_direction" in {x.id for x in a.ambiguities}


def test_tail_and_accessory() -> None:
    a = _anatomy(tail=True, accessory=True)
    segs = a.segment_map()
    assert segs["tail"].kind == SegmentKind.TAIL
    assert segs["tail"].source_bones == ["tail", "tail.001"]
    assert "tail_tip" in {e.id for e in a.end_effectors}
    assert segs["appendage:feather.L"].kind == SegmentKind.APPENDAGE
    amb = [x for x in a.ambiguities if x.id.startswith("appendage:feather")]
    assert len(amb) == 1  # mirrored pair raised once
    assert set(amb[0].subject) == {"feather.L", "feather.R"}


def test_no_arms_raises_ambiguity() -> None:
    ex = humanoid_extraction()
    ex.bones = [b for b in ex.bones if not any(k in b.name for k in ("shoulder", "arm", "hand", "f_"))]
    for b in ex.bones:
        b.children = [c for c in b.children if c in {x.name for x in ex.bones}]
    a = analyze_anatomy(ex, build_skeleton(ex))
    assert {lb.kind.value for lb in a.limbs} == {"leg"}
    assert "chest" in {x.id for x in a.ambiguities}


def test_mass_shares_sum_to_at_most_one() -> None:
    a = _anatomy(tail=True)
    total = sum(s.mass.relative_mass for s in a.segments if s.mass)
    assert 0.95 <= total <= 1.0001
    assert a.segment_map()["thigh.L"].mass.relative_mass > a.segment_map()["hand.L"].mass.relative_mass


def test_rigify_like_rig_anatomy_matches_plain() -> None:
    ex = rigify_like_extraction()
    a = analyze_anatomy(ex, build_skeleton(ex))
    assert {lb.id for lb in a.limbs} == {"arm.L", "arm.R", "leg.L", "leg.R"}
    assert a.segment_map()["upper_arm.L"].source_bones == ["ORG-upper_arm.L"]
