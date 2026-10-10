"""Preferred bending directions of hinge-like joints (deterministic).

For a two-segment chain a -> b -> c (e.g. shoulder -> elbow -> wrist):

* rest flexion is the angle between (b - a) and (c - b);
* the joint "points toward" the offset of b from the line a-c;
* the hinge axis h satisfies h = p x l (p: points-toward, l: lower direction), so that a positive
  right-handed rotation of the lower segment about h increases flexion.

When the rest pose is (nearly) straight, an IK pole position, or finally the bone's local
axis convention (Rigify bends limbs around local X), is used instead with lower confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from animagent.analysis.geom import angle_deg, closest_local_axis, rnd, unit, v
from animagent.models.anatomy import BendInfo, Evidence

STRAIGHT_DEG = 2.0


def _body_relative(p: np.ndarray, frame_axes: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: round(float(p @ a), 3) for k, a in frame_axes.items()}


def hinge_bend(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    *,
    frame_axes: dict[str, np.ndarray],
    lower_bone: str | None = None,
    lower_matrix_world: Sequence[Sequence[float]] | None = None,
    prior_points: str | None = None,
    pole_position: Sequence[float] | None = None,
) -> BendInfo:
    """Compute bend info. `prior_points` is e.g. "forward" (knee) or "backward" (elbow)."""
    A, B, C = v(a), v(b), v(c)
    u, low = B - A, C - B
    l_hat = unit(low)
    flex = angle_deg(u, low)
    evidence: list[Evidence] = []
    line = C - A
    t = float((B - A) @ line / max(float(line @ line), 1e-18))
    offset = B - (A + t * line)

    pole_dir = None
    if pole_position is not None:
        pp = v(pole_position) - B
        pole_dir = unit(pp - (pp @ l_hat) * l_hat)

    source = "rest_geometry"
    if flex >= STRAIGHT_DEG and np.linalg.norm(offset) > 1e-9:
        p = unit(offset)
        conf = 0.9 if flex >= 10 else (0.75 if flex >= 4 else 0.55)
        evidence.append(Evidence(kind="rest_geometry", detail=f"rest flexion {flex:.1f} deg", weight=min(flex / 10, 1)))
    elif pole_dir is not None and np.linalg.norm(pole_dir) > 0:
        p, conf, source = pole_dir, 0.6, "ik_pole"
        evidence.append(Evidence(kind="ik_pole", detail="chain is straight at rest; using IK pole side", weight=0.6))
    elif lower_matrix_world is not None:
        axis_x = unit(np.asarray(lower_matrix_world, dtype=float)[:3, 0])
        p, conf, source = unit(np.cross(l_hat, axis_x)), 0.35, "local_axis_convention"
        evidence.append(
            Evidence(kind="convention", detail="straight chain, no pole: assuming bend about local +X", weight=0.3)
        )
    else:
        p, conf, source = np.zeros(3), 0.0, "prior"
    axis = unit(np.cross(p, l_hat))

    if pole_dir is not None and source == "rest_geometry" and np.linalg.norm(pole_dir) > 0:
        agree = float(p @ pole_dir)
        if agree > 0.5:
            conf = min(conf + 0.05, 0.98)
            evidence.append(Evidence(kind="ik_pole", detail=f"IK pole agrees (cos={agree:.2f})", weight=0.5))
        else:
            conf = max(conf - 0.25, 0.1)
            evidence.append(Evidence(kind="ik_pole", detail=f"IK pole disagrees (cos={agree:.2f})", weight=-0.7))

    matches = None
    if prior_points and np.linalg.norm(p) > 0:
        d = float(p @ frame_axes["forward"])
        if abs(d) >= 0.3:
            matches = (d > 0) == (prior_points == "forward")
            evidence.append(
                Evidence(
                    kind="anatomical_prior",
                    detail=f"humanoid prior: joint points {prior_points}; measured forward component {d:+.2f}",
                    weight=0.3 if matches else -0.3,
                )
            )

    local_label, local_align = None, None
    if lower_matrix_world is not None and np.linalg.norm(axis) > 0:
        local_label, local_align = closest_local_axis(axis, lower_matrix_world)

    return BendInfo(
        hinge_axis=rnd(axis),
        joint_points_toward=rnd(p),
        rest_flexion_deg=round(flex, 2),
        local_axis_bone=lower_bone,
        hinge_axis_local=local_label,
        hinge_axis_local_alignment=round(local_align, 3) if local_align is not None else None,
        body_relative=_body_relative(p, frame_axes),
        source=source,  # type: ignore[arg-type]
        matches_prior=matches,
        confidence=round(conf, 3),
        evidence=evidence,
    )
