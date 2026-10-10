import numpy as np

from animagent.pipeline import analyze
from animagent.validation.checks import control_provenance, joint_table, run_checks
from animagent.validation.inputs import experiment_args, landmark_specs
from tests.fakes.synthetic import rigify_like_extraction


def _probe(p):  # type: ignore[no-untyped-def]
    """A fabricated probe consistent with the synthetic rig (as Blender would measure it)."""
    conn = {}
    bm = p.extraction.bone_map()
    for b in p.extraction.bones:
        if b.parent:
            conn[b.name] = [b.parent, float(np.linalg.norm(np.subtract(bm[b.parent].tail_world, b.head_world))), False]
    lms = []
    for lm in landmark_specs(p):
        pos = np.array(lm["position"])
        lms.append(
            {
                "id": lm["id"],
                "vertex_count": 10,
                "centroid": pos.tolist(),
                "bbox_min": (pos - 0.05).tolist(),
                "bbox_max": (pos + 0.05).tolist(),
                "nearest_vertex_distance": 0.01,
                "distance_outside_bbox": 0.0,
                "lowest_point": (pos - [0, 0, 0.03]).tolist(),
                "forward_most_point": (pos + [0, -0.04, 0]).tolist(),
                "backward_most_point": (pos + [0, 0.04, 0]).tolist(),
                "extent": [0.1, 0.1, 0.1],
            }
        )
    return {
        "bone_frames": {
            "bones": 1,
            "max_orthonormality_error": 0.0,
            "min_det": 1.0,
            "max_det": 1.0,
            "max_y_axis_misalignment": 0.0,
            "armature_matrix_det": 1.0,
        },
        "connectivity": conn,
        "mesh_ground_z": 0.0,
        "landmarks": lms,
        "drivers": [],
    }


def test_checks_run_and_classify_methods() -> None:
    p = analyze(rigify_like_extraction())
    cl = run_checks(p, _probe(p), None)
    by_id = {c["id"]: c for c in cl.items}
    assert by_id["A2"]["status"] == "pass"
    assert by_id["A3"]["method"] == "independent_measurement"
    assert by_id["D1:knee.L"]["status"] == "pass"
    assert by_id["F2"]["status"] == "pass"
    assert {c["category"] for c in cl.items} >= set("ABCDEFG")
    # Without an experiment, experimental claims are not reported as verified.
    assert by_id["G2:arm.L"]["status"] == "fail" and "no experiment" in by_id["G2:arm.L"]["note"]


def test_joint_table_has_bones_and_local_positions() -> None:
    p = analyze(rigify_like_extraction())
    rows = {r["id"]: r for r in joint_table(p)}
    knee = rows["knee.L"]
    assert knee["blender_bones"]["structural"] == "ORG-shin.L"
    assert knee["blender_bones"]["structural_parent"] == "ORG-thigh.L"
    # The knee sits at the tail of the thigh: local position (0, length, 0) in the thigh's rest frame.
    thigh_len = p.extraction.bone_map()["ORG-thigh.L"].length
    assert np.allclose(knee["position_in_parent_bone_rest_space"], [0, thigh_len, 0], atol=1e-4)
    assert knee["bend"]["hinge_axis_local"] in ("+X", "-X")


def test_experiment_args_and_provenance() -> None:
    p = analyze(rigify_like_extraction())
    ea = experiment_args(p)
    arm = next(x for x in ea["limbs"] if x["limb_id"] == "arm.L")
    assert (arm["upper"], arm["lower"], arm["fk_lower"], arm["ik_effector"]) == (
        "ORG-upper_arm.L",
        "ORG-forearm.L",
        "forearm_fk.L",
        "hand_ik.L",
    )
    prov = {c["control"]: c for c in control_provenance(p, None)}
    assert prov["upper_arm_fk.L"]["mapped_as"][0]["evidence_level"] == "constraint_or_driver"
    assert prov["upper_arm_fk.L"]["mapping_confirmed_experimentally"] is None
