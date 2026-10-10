import numpy as np
import pytest

from animagent.analysis.bending import hinge_bend

AXES = {"forward": np.array([0, -1.0, 0]), "left": np.array([1.0, 0, 0]), "up": np.array([0, 0, 1.0])}


def test_bent_chain_uses_geometry() -> None:
    b = hinge_bend([0, 0, 1], [0, -0.1, 0.5], [0, 0, 0], frame_axes=AXES, prior_points="forward")
    assert b.source == "rest_geometry"
    assert b.body_relative["forward"] == pytest.approx(1.0, abs=1e-3)
    assert b.matches_prior is True
    assert b.rest_flexion_deg > 10


def test_straight_chain_falls_back_to_pole() -> None:
    b = hinge_bend([0, 0, 1], [0, 0, 0.5], [0, 0, 0], frame_axes=AXES, pole_position=[0, -1, 0.5])
    assert b.source == "ik_pole"
    assert b.joint_points_toward == pytest.approx([0, -1, 0])
    assert b.confidence < 0.9


def test_straight_chain_without_pole_uses_local_axis_with_low_confidence() -> None:
    m = np.eye(4)
    b = hinge_bend([0, 0, 1], [0, 0, 0.5], [0, 0, 0], frame_axes=AXES, lower_matrix_world=m.tolist())
    assert b.source == "local_axis_convention"
    assert b.confidence < 0.5
    assert b.hinge_axis_local == "+X"


def test_disagreeing_pole_lowers_confidence() -> None:
    agree = hinge_bend([0, 0, 1], [0, -0.1, 0.5], [0, 0, 0], frame_axes=AXES, pole_position=[0, -1, 0.5])
    disagree = hinge_bend([0, 0, 1], [0, -0.1, 0.5], [0, 0, 0], frame_axes=AXES, pole_position=[0, 1, 0.5])
    assert agree.confidence > disagree.confidence
