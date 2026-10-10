import pytest

from animagent.analysis.anatomy import analyze_anatomy
from animagent.analysis.controls import evaluate_driver, map_controls
from animagent.analysis.skeleton import build_skeleton
from animagent.models.rig import DriverData
from tests.fakes.synthetic import humanoid_extraction, rigify_like_extraction


def _map(ex):  # type: ignore[no-untyped-def]
    sk = build_skeleton(ex)
    return map_controls(ex, sk, analyze_anatomy(ex, sk))


def test_rigify_like_limb_controls() -> None:
    cm = _map(rigify_like_extraction())
    assert cm.rig_kind == "rigify"
    arm = next(lc for lc in cm.limbs if lc.limb_id == "arm.L")
    assert [c.bone for c in arm.fk_chain] == ["shoulder.L", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]
    assert arm.ik_effector and arm.ik_effector.bone == "hand_ik.L"
    assert arm.ik_pole and arm.ik_pole.bone == "upper_arm_ik_target.L"
    assert [c.bone for c in arm.ik_chain_controls] == ["upper_arm_ik.L"]
    sw = arm.ik_fk_switch
    assert sw and (sw.bone, sw.property, sw.ik_value, sw.fk_value) == ("upper_arm_parent.L", "IK_FK", 0.0, 1.0)
    assert any(e.kind == "pose_observation" and e.weight > 0 for e in sw.evidence)
    leg = next(lc for lc in cm.limbs if lc.limb_id == "leg.R")
    assert leg.ik_effector and leg.ik_effector.bone == "foot_ik.R"
    assert "toe_ik.R" in [c.bone for c in leg.ik_chain_controls]
    assert {p.property for p in leg.settings} == {"IK_Stretch"}
    assert cm.root_control and cm.root_control.bone == "root"
    assert cm.unmapped_controls == []


def test_switch_semantics_follow_driver_modifiers() -> None:
    # Inverted generator: IK_FK=1 means IK; pose observation (FK at value 0) agrees.
    cm = _map(rigify_like_extraction(switch_generator=(0.0, 1.0), pose_follows="fk"))
    sw = cm.limbs[0].ik_fk_switch
    assert sw and sw.ik_value == 1.0 and sw.fk_value == 0.0


def test_pose_observation_overrides_wrong_driver_model() -> None:
    # Driver model says IK_FK=0 -> FK, but the evaluated pose follows IK: observation wins.
    cm = _map(rigify_like_extraction(switch_generator=(0.0, 1.0), pose_follows="ik"))
    sw = cm.limbs[0].ik_fk_switch
    assert sw and sw.ik_value == 0.0
    assert any(e.kind == "pose_observation" and e.weight < 0 for e in sw.evidence)


def test_generic_rig_has_no_controls_layer() -> None:
    cm = _map(humanoid_extraction())
    assert cm.rig_kind == "generic"
    assert all(not lc.ik_chains for lc in cm.limbs)


@pytest.mark.parametrize(
    ("dtype", "expr", "mods", "x", "expected"),
    [
        ("SUM", "", [], 0.3, 0.3),
        ("AVERAGE", "", [{"type": "GENERATOR", "mode": "POLYNOMIAL", "coefficients": [1.0, -1.0]}], 0.0, 1.0),
        ("SCRIPTED", "1 - var * 2", [], 0.25, 0.5),
        ("SCRIPTED", "max(0, min(1, var))", [], 3.0, 1.0),
        ("SCRIPTED", "__import__('os')", [], 0.0, None),
    ],
)
def test_evaluate_driver(dtype: str, expr: str, mods: list, x: float, expected: float | None) -> None:
    d = DriverData.model_validate(
        {
            "data_path": "x",
            "type": dtype,
            "expression": expr,
            "modifiers": mods,
            "variables": [{"name": "var", "type": "SINGLE_PROP"}],
        }
    )
    got = evaluate_driver(d, {"var": x})
    assert got == (pytest.approx(expected) if expected is not None else None)
