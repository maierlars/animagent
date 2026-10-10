import numpy as np

from animagent.pipeline import analyze
from animagent.visualize import framing, overlay_spec
from tests.fakes.synthetic import humanoid_extraction


def test_overlay_spec_rest_matches_anatomy() -> None:
    p = analyze(humanoid_extraction(tail=True))
    spec = overlay_spec(p, pose="rest")
    assert spec["lines"] and spec["points"] and spec["labels"]
    knee = p.anatomy.joint_map()["knee.L"].position
    assert any(np.allclose(pt[0], knee) for pt in spec["points"])
    assert {lbl[1] for lbl in spec["labels"]} >= {"UP", "FORWARD", "LEFT", "head"}


def test_overlay_follows_current_pose() -> None:
    ex = humanoid_extraction()
    # Translate the whole current pose by +1 m in X: posed overlay points must follow.
    shift = np.eye(4)
    shift[0, 3] = 1.0
    for b in ex.bones:
        b.pose.matrix_armature = (shift @ np.asarray(b.matrix_world_rest)).tolist()
    p = analyze(ex)
    knee = np.asarray(p.anatomy.joint_map()["knee.L"].position)
    posed = overlay_spec(p, pose="current")
    assert any(np.allclose(pt[0], knee + [1, 0, 0]) for pt in posed["points"])
    assert not any(np.allclose(pt[0], knee) for pt in posed["points"])


def test_framing() -> None:
    f = framing(analyze(humanoid_extraction()))
    assert f["forward"] == [0.0, -1.0, 0.0] and f["size"] > 1.5
