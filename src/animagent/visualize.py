"""Build debug-overlay specs (lines/points/labels in world space) from a rig profile."""

from __future__ import annotations

from typing import Any

import numpy as np

from animagent.analysis.geom import v
from animagent.models.anatomy import Side
from animagent.models.profile import RigProfile

COLORS = {
    "trunk": (1.0, 0.85, 0.2, 1.0),
    "head": (1.0, 0.3, 0.9, 1.0),
    "L": (1.0, 0.3, 0.25, 1.0),
    "R": (0.25, 0.55, 1.0, 1.0),
    "tail": (0.3, 1.0, 0.4, 1.0),
    "digit": (0.7, 0.7, 0.7, 0.8),
    "appendage": (0.6, 0.6, 0.45, 0.6),
    "bend": (1.0, 1.0, 1.0, 1.0),
    "hinge": (0.8, 0.3, 1.0, 1.0),
    "contact": (0.2, 1.0, 0.6, 1.0),
    "up": (0.3, 1.0, 0.3, 1.0),
    "forward": (0.3, 0.5, 1.0, 1.0),
    "left": (1.0, 0.3, 0.3, 1.0),
}


def _seg_color(kind: str, side: Side) -> tuple[float, ...]:
    if kind in ("pelvis", "spine", "chest", "neck"):
        return COLORS["trunk"]
    if kind in ("head", "jaw"):
        return COLORS["head"]
    if kind == "tail":
        return COLORS["tail"]
    if kind in ("digit", "appendage"):
        return COLORS[kind]
    return COLORS[side.value] if side != Side.CENTER else COLORS["trunk"]


class _Posed:
    """Maps rest-pose world data to the extracted *current* pose (or leaves it at rest)."""

    def __init__(self, profile: RigProfile, use_pose: bool) -> None:
        self.use_pose = use_pose
        self.bones = profile.extraction.bone_map()
        self.arm = np.asarray(profile.extraction.armature.matrix_world)

    def _pose_world(self, bone: str) -> np.ndarray | None:
        b = self.bones.get(bone)
        if not self.use_pose or b is None or b.pose.matrix_armature is None:
            return None
        return self.arm @ np.asarray(b.pose.matrix_armature)

    def point(self, bone: str, p: list[float]) -> list[float]:
        """Rigidly carry a rest-space point attached to `bone` into the current pose."""
        pw = self._pose_world(bone)
        if pw is None:
            return list(p)
        rest = np.asarray(self.bones[bone].matrix_world_rest)
        local = np.linalg.solve(rest, np.append(np.asarray(p, float), 1.0))
        return (pw @ local)[:3].tolist()

    def vector(self, bone: str, d: list[float]) -> np.ndarray:
        pw = self._pose_world(bone)
        if pw is None:
            return v(d)
        rest = np.asarray(self.bones[bone].matrix_world_rest)[:3, :3]
        r = pw[:3, :3] @ np.linalg.inv(rest)
        out = r @ v(d)
        n = np.linalg.norm(out)
        return out / n * np.linalg.norm(v(d)) if n > 1e-12 else out


def overlay_spec(profile: RigProfile, labels: bool = True, pose: str = "current") -> dict[str, Any]:
    """Overlay geometry. `pose="current"` follows the extracted current pose, `"rest"` the rest pose."""
    a = profile.anatomy
    bm = profile.skeleton.bone_map()
    H = max(a.body_frame.height, 1e-6)
    P = _Posed(profile, pose == "current")
    joint_bone = {
        j.id: next((s.source_bones[0] for s in a.segments if s.id == j.child_segment), None) for j in a.joints
    }
    for j in a.joints:  # the bone whose head is the joint
        cands = [
            b
            for s in a.segments
            if s.id == j.child_segment
            for b in s.source_bones
            if b in bm and np.allclose(bm[b].head, j.position, atol=1e-5)
        ]
        if cands:
            joint_bone[j.id] = cands[0]
    lines: list[list[Any]] = []
    points: list[list[Any]] = []
    texts: list[list[Any]] = []
    for s in a.segments:
        col = _seg_color(s.kind.value, s.side)
        for b in s.source_bones:
            if b in bm:
                lines.append([P.point(b, bm[b].head), P.point(b, bm[b].tail), col])
        if labels and s.kind.value not in ("digit", "appendage"):
            b0 = s.source_bones[0]
            mid = (0.5 * (v(P.point(b0, bm[b0].head)) + v(P.point(b0, bm[b0].tail)))).tolist()
            texts.append([mid, s.id, (1, 1, 1, 0.9)])
    for j in a.joints:
        major = j.kind.value not in ("digit", "generic")
        jb = joint_bone.get(j.id)
        pos = P.point(jb, j.position) if jb else j.position
        points.append([pos, (1, 1, 1, 1) if major else (0.8, 0.8, 0.8, 0.7), 9.0 if major else 4.0])
        if j.bend and major:
            p = v(pos)
            lb = j.bend.local_axis_bone or jb
            toward = P.vector(lb, j.bend.joint_points_toward) if lb else v(j.bend.joint_points_toward)
            ax = P.vector(lb, j.bend.hinge_axis) if lb else v(j.bend.hinge_axis)
            lines.append([pos, (p + 0.1 * H * toward).tolist(), COLORS["bend"]])
            lines.append([(p - 0.04 * H * ax).tolist(), (p + 0.04 * H * ax).tolist(), COLORS["hinge"]])
            if labels:
                texts.append([pos, f"{j.id} ({j.bend.hinge_axis_local})", (1, 1, 0.6, 1)])
    seg_bone = {s.id: s.source_bones[0] for s in a.segments}
    for c in a.contacts:
        if c.kind in ("heel", "ball", "toe_tip", "palm", "head_top"):
            cb = seg_bone.get(c.segment)
            points.append([P.point(cb, c.position) if cb else c.position, COLORS["contact"], 7.0])
    f = a.body_frame
    o = v(f.origin)
    for key in ("up", "forward", "left"):
        lines.append([o.tolist(), (o + 0.25 * H * v(getattr(f, key))).tolist(), COLORS[key]])
        if labels:
            texts.append([(o + 0.27 * H * v(getattr(f, key))).tolist(), key.upper(), COLORS[key]])
    return {
        "lines": lines,
        "points": points,
        "labels": texts,
        "title": f"animagent: {profile.source.armature} ({pose} pose)",
    }


def framing(profile: RigProfile) -> dict[str, Any]:
    f = profile.anatomy.body_frame
    pts = np.array([b.head for b in profile.skeleton.bones] + [b.tail for b in profile.skeleton.bones])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {
        "center": (0.5 * (lo + hi)).tolist(),
        "size": float(max(hi - lo)),
        "forward": f.forward,
        "up": f.up,
        "left": f.left,
    }
