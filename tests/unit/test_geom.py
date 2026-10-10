import numpy as np
import pytest

from animagent.analysis.geom import closest_local_axis, mirror_name, point_segment_distance, side_from_name
from animagent.models.anatomy import Side


@pytest.mark.parametrize(
    ("name", "side"),
    [
        ("hand.L", Side.LEFT),
        ("ORG-thigh.R", Side.RIGHT),
        ("f_index.01.L.001", Side.LEFT),
        ("hand_l", Side.LEFT),
        ("Left_Arm", Side.LEFT),
        ("spine.003", Side.CENTER),
        ("Lid", Side.CENTER),
        ("tail", Side.CENTER),
    ],
)
def test_side_from_name(name: str, side: Side) -> None:
    assert side_from_name(name) == side


def test_mirror_name() -> None:
    assert mirror_name("ORG-hand.L") == "ORG-hand.R"
    assert mirror_name("thumb.01.R.001") == "thumb.01.L.001"
    assert mirror_name("Left_foot") == "Right_foot"
    assert mirror_name("spine") is None


def test_point_segment_distance() -> None:
    d, t = point_segment_distance(np.array([0.5, 1.0, 0]), np.zeros(3), np.array([1.0, 0, 0]))
    assert d == pytest.approx(1.0)
    assert t == pytest.approx(0.5)


def test_closest_local_axis() -> None:
    m = np.eye(4)
    m[:3, 0] = [0, 0, 1]
    m[:3, 2] = [-1, 0, 0]
    assert closest_local_axis(np.array([0, 0, -1.0]), m.tolist()) == ("-X", pytest.approx(1.0))
