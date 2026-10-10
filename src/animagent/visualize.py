"""Build debug-overlay specs (lines/points/labels/legend in world space) from a rig profile.

Layers: skeleton, joints, bend, contacts, frame, ambiguity, controls. All geometry can follow
the pose captured in the profile (`pose="current"`) or the rest pose.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    "digit": (0.85, 0.85, 0.85, 0.9),
    "appendage": (0.6, 0.6, 0.6, 0.85),
    "bend": (1.0, 1.0, 1.0, 1.0),
    "hinge": (0.8, 0.3, 1.0, 1.0),
    "contact": (0.2, 1.0, 0.6, 1.0),
    "up": (0.3, 1.0, 0.3, 1.0),
    "forward": (0.3, 0.5, 1.0, 1.0),
    "left": (1.0, 0.3, 0.3, 1.0),
    "conf_high": (0.2, 1.0, 0.3, 1.0),
    "conf_mid": (1.0, 0.9, 0.2, 1.0),
    "conf_low": (1.0, 0.35, 0.1, 1.0),
    "ctrl_fk": (0.3, 0.6, 1.0, 1.0),
    "ctrl_ik": (1.0, 0.25, 0.25, 1.0),
    "ctrl_pole": (1.0, 0.3, 1.0, 1.0),
    "ctrl_ikchain": (1.0, 0.6, 0.15, 1.0),
    "ctrl_switch": (1.0, 1.0, 0.2, 1.0),
    "ctrl_body": (0.95, 0.95, 0.95, 1.0),
    "ctrl_tweak": (0.55, 0.55, 0.55, 0.9),
}
ALL_LAYERS = frozenset({"skeleton", "joints", "bend", "contacts", "frame", "ambiguity", "labels"})


def _conf_color(c: float) -> tuple[float, ...]:
    return COLORS["conf_high"] if c >= 0.8 else (COLORS["conf_mid"] if c >= 0.6 else COLORS["conf_low"])


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
        out = pw[:3, :3] @ np.linalg.inv(rest) @ v(d)
        n = np.linalg.norm(out)
        return out / n * np.linalg.norm(v(d)) if n > 1e-12 else out

    def head_tail(self, bone: str) -> tuple[list[float], list[float]]:
        b = self.bones[bone]
        return self.point(bone, b.head_world), self.point(bone, b.tail_world)


def _dashed(p0: list[float], p1: list[float], col: tuple[float, ...], n: int = 8) -> list[list[Any]]:
    a, b = v(p0), v(p1)
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append([(a + (b - a) * i / n).tolist(), (a + (b - a) * (i + 1) / n).tolist(), col])
    return out


def legend_for(layers: Iterable[str], title: str) -> dict[str, Any]:
    layers = set(layers)
    items: list[list[Any]] = []
    if "skeleton" in layers:
        items += [
            ["trunk (pelvis/spine/chest/neck)", COLORS["trunk"]],
            ["head / jaw", COLORS["head"]],
            ["left limb (character's left)", COLORS["L"]],
            ["right limb", COLORS["R"]],
            ["tail", COLORS["tail"]],
            ["digits (fingers / toes)", COLORS["digit"]],
        ]
    if "ambiguity" in layers:
        items += [["? dashed grey: ambiguous appendage (not interpreted)", COLORS["appendage"]]]
    if "joints" in layers:
        items += [
            ["joint, confidence >= 0.8", COLORS["conf_high"]],
            ["joint, confidence 0.6-0.8", COLORS["conf_mid"]],
            ["joint, confidence < 0.6", COLORS["conf_low"]],
        ]
    if "bend" in layers:
        items += [
            ["white arrow: direction the joint apex points when flexed", COLORS["bend"]],
            ["purple bar: hinge axis (+rotation = more flexion); label = local axis", COLORS["hinge"]],
        ]
    if "contacts" in layers:
        items += [["contact point (heel / ball / toe tip / palm / head top)", COLORS["contact"]]]
    if "frame" in layers:
        items += [["UP", COLORS["up"]], ["FORWARD", COLORS["forward"]], ["LEFT (character's left)", COLORS["left"]]]
    if "controls" in layers:
        items += [
            ["FK control", COLORS["ctrl_fk"]],
            ["IK effector control", COLORS["ctrl_ik"]],
            ["IK pole control", COLORS["ctrl_pole"]],
            ["IK chain / foot-roll control", COLORS["ctrl_ikchain"]],
            ["IK/FK switch holder (label shows IK value)", COLORS["ctrl_switch"]],
            ["root / torso / trunk control", COLORS["ctrl_body"]],
            ["dotted line: control -> segment it drives (structural trace)", COLORS["ctrl_tweak"]],
        ]
    return {"title": title, "items": items, "size": 13}


def overlay_spec(
    profile: RigProfile,
    labels: bool = True,
    pose: str = "current",
    layers: Iterable[str] | None = None,
    segments: Iterable[str] | None = None,
    legend_title: str | None = None,
) -> dict[str, Any]:
    """Overlay geometry. `segments` restricts drawing to a subset (close-ups)."""
    layers = set(layers) if layers is not None else set(ALL_LAYERS)
    if not labels:
        layers.discard("labels")
    a = profile.anatomy
    bm = profile.skeleton.bone_map()
    H = max(a.body_frame.height, 1e-6)
    P = _Posed(profile, pose == "current")
    seg_filter = set(segments) if segments is not None else None
    lines: list[list[Any]] = []
    points: list[list[Any]] = []
    texts: list[list[Any]] = []
    show_labels = "labels" in layers

    def keep(seg_id: str | None) -> bool:
        return seg_filter is None or (seg_id is not None and seg_id in seg_filter)

    for s in a.segments:
        if not keep(s.id):
            continue
        ambiguous = s.kind.value == "appendage"
        if ambiguous and "ambiguity" not in layers:
            continue
        if not ambiguous and "skeleton" not in layers:
            continue
        col = _seg_color(s.kind.value, s.side)
        for b in s.source_bones:
            if b in bm:
                h, t = P.point(b, bm[b].head), P.point(b, bm[b].tail)
                lines.extend(_dashed(h, t, col) if ambiguous else [[h, t, col]])
        if show_labels and s.kind.value != "digit":
            if ambiguous:  # label at the chain tip, short name
                bl = s.source_bones[-1]
                texts.append([P.point(bl, bm[bl].tail), "? " + s.id.removeprefix("appendage:"), (0.8, 0.8, 0.8, 1)])
            else:
                b0 = s.source_bones[0]
                mid = (0.5 * (v(P.point(b0, bm[b0].head)) + v(P.point(b0, bm[b0].tail)))).tolist()
                texts.append([mid, s.id, (1, 1, 1, 0.95)])

    joint_bone: dict[str, str] = {}
    for j in a.joints:
        cands = [
            b
            for s in a.segments
            if s.id == j.child_segment
            for b in s.source_bones
            if b in bm and np.allclose(bm[b].head, j.position, atol=1e-5)
        ]
        if cands:
            joint_bone[j.id] = cands[0]
    for j in a.joints:
        if not keep(j.child_segment):
            continue
        major = j.kind.value not in ("digit", "generic")
        jb = joint_bone.get(j.id)
        pos = P.point(jb, j.position) if jb else j.position
        if "joints" in layers and (major or seg_filter is not None):
            points.append([pos, _conf_color(j.confidence), 10.0 if major else 6.0])
            if show_labels and major and not ("bend" in layers and j.bend):
                texts.append([pos, f"{j.id} {j.confidence:.2f}", (1, 1, 1, 1)])
        if j.bend and "bend" in layers and (major or seg_filter is not None):
            p = v(pos)
            lb = j.bend.local_axis_bone or jb
            toward = P.vector(lb, j.bend.joint_points_toward) if lb else v(j.bend.joint_points_toward)
            ax = P.vector(lb, j.bend.hinge_axis) if lb else v(j.bend.hinge_axis)
            scale = 1.0 if major else 0.3
            tip = p + 0.1 * scale * H * toward
            lines.append([pos, tip.tolist(), COLORS["bend"]])
            side = np.cross(toward, ax)
            if np.linalg.norm(side) > 1e-9:
                side = side / np.linalg.norm(side)
                for sgn in (+1, -1):  # arrow head
                    lines.append(
                        [
                            tip.tolist(),
                            (tip - 0.025 * scale * H * toward + sgn * 0.012 * scale * H * side).tolist(),
                            COLORS["bend"],
                        ]
                    )
            lines.append([(p - 0.05 * scale * H * ax).tolist(), (p + 0.05 * scale * H * ax).tolist(), COLORS["hinge"]])
            if show_labels and (major or seg_filter is not None and len(seg_filter) < 8):
                prior = (
                    "" if j.bend.matches_prior is None else (" prior-ok" if j.bend.matches_prior else " PRIOR-MISMATCH")
                )
                texts.append(
                    [pos, f"{j.id} axis {j.bend.hinge_axis_local} c={j.bend.confidence:.2f}{prior}", (1, 1, 0.6, 1)]
                )
    if "contacts" in layers:
        seg_bone = {s.id: s.source_bones[0] for s in a.segments}
        for c in a.contacts:
            if c.kind in ("heel", "ball", "toe_tip", "palm", "head_top") and keep(c.segment):
                cb = seg_bone.get(c.segment)
                cp = P.point(cb, c.position) if cb else c.position
                points.append([cp, COLORS["contact"], 8.0])
                if show_labels and seg_filter is not None:
                    texts.append([cp, c.id, COLORS["contact"]])
    if "frame" in layers:
        f = a.body_frame
        o = v(f.origin)
        for key in ("up", "forward", "left"):
            lines.append([o.tolist(), (o + 0.25 * H * v(getattr(f, key))).tolist(), COLORS[key]])
            if show_labels:
                texts.append([(o + 0.27 * H * v(getattr(f, key))).tolist(), key.upper(), COLORS[key]])
    if "controls" in layers:
        _controls_layer(profile, P, keep, lines, points, texts, show_labels)
    spec: dict[str, Any] = {
        "lines": lines,
        "points": points,
        "labels": texts,
        "label_size": 13,
        "line_width": 3.0,
        "title": f"animagent: {profile.source.armature} ({pose} pose)",
    }
    if legend_title:
        spec["legend"] = legend_for(layers, legend_title)
    return spec


def _controls_layer(
    profile: RigProfile, P: _Posed, keep: Any, lines: list[Any], points: list[Any], texts: list[Any], show_labels: bool
) -> None:
    a, cm = profile.anatomy, profile.controls
    segs = a.segment_map()

    def ctrl(bone: str, col: tuple[float, ...], label: str | None, segment: str | None) -> None:
        if bone not in P.bones:
            return
        h, t = P.head_tail(bone)
        lines.append([h, t, col])
        points.append([h, col, 9.0])
        if show_labels and label:
            texts.append([h, label, col])
        if segment and segment in segs:
            sb = segs[segment].source_bones[0]
            if sb in P.bones:
                sh, st = P.head_tail(sb)
                target = (0.5 * (v(sh) + v(st))).tolist()
                lines.extend(_dashed(h, target, COLORS["ctrl_tweak"], n=10))

    for lc in cm.limbs:
        limb = next(lb for lb in a.limbs if lb.id == lc.limb_id)
        if not any(keep(s) for s in limb.segments):
            continue
        for c in lc.fk_chain:
            ctrl(c.bone, COLORS["ctrl_fk"], c.bone, c.segment)
        if lc.ik_effector:
            ctrl(lc.ik_effector.bone, COLORS["ctrl_ik"], f"IK {lc.ik_effector.bone}", lc.ik_effector.segment)
        if lc.ik_pole:
            ctrl(lc.ik_pole.bone, COLORS["ctrl_pole"], f"pole {lc.ik_pole.bone}", None)
        for c in lc.ik_chain_controls:
            ctrl(c.bone, COLORS["ctrl_ikchain"], c.bone, None)
        if lc.ik_fk_switch:
            sw = lc.ik_fk_switch
            ctrl(
                sw.bone,
                COLORS["ctrl_switch"],
                f"{sw.bone}[{sw.property}] IK={sw.ik_value:g} now={sw.current_value}",
                None,
            )
    for c in cm.body_controls:
        if keep(c.segment) or c.segment is None:
            ctrl(c.bone, COLORS["ctrl_body"], f"{c.bone} ({c.role})", c.segment)
    if cm.root_control:
        ctrl(cm.root_control.bone, COLORS["ctrl_body"], f"{cm.root_control.bone} (root)", None)


def framing(
    profile: RigProfile, segments: Iterable[str] | None = None, pose: str = "current", margin: float = 1.0
) -> dict[str, Any]:
    """View centre/size for the whole character or a subset of segments (posed positions)."""
    f = profile.anatomy.body_frame
    P = _Posed(profile, pose == "current")
    segs = profile.anatomy.segment_map()
    bones = (
        [b for s in segments if s in segs for b in segs[s].source_bones]
        if segments is not None
        else [b.name for b in profile.skeleton.bones]
    )
    pts = np.array([q for b in bones if b in P.bones for q in P.head_tail(b)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {
        "center": (0.5 * (lo + hi)).tolist(),
        "size": float(max(max(hi - lo), 0.05)) * margin,
        "forward": f.forward,
        "up": f.up,
        "left": f.left,
    }
