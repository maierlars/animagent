"""Build the arguments for the Blender-side validation probe and experiment from a profile."""

from __future__ import annotations

import re
from typing import Any

from animagent.analysis.controls import _CON_PATH, _prop_refs, control_bones
from animagent.models.profile import RigProfile

_MECH = re.compile(r"^(ORG|DEF|MCH|VIS|WGT)[-_]")


def segment_deform_bones(profile: RigProfile, segment_ids: list[str]) -> list[str]:
    bm = profile.skeleton.bone_map()
    segs = profile.anatomy.segment_map()
    out: list[str] = []
    for sid in segment_ids:
        if sid in segs:
            for b in segs[sid].source_bones:
                out += bm[b].deform_bones if b in bm else []
    return sorted(set(out))


def landmark_specs(profile: RigProfile) -> list[dict[str, Any]]:
    """Points to compare against skin geometry, each with the deform bones of its region."""
    a = profile.anatomy
    fwd = a.body_frame.forward
    seg_of_limb = {lb.id: lb.segments for lb in a.limbs}
    specs: list[dict[str, Any]] = []
    for e in a.end_effectors:
        region = [e.segment]
        if e.kind == "foot":
            limb = next((lb for lb in a.limbs if lb.end_effector == e.id), None)
            if limb:
                region = [s for s in limb.segments if s.startswith(("foot", "toe"))] + limb.digits
        elif e.kind == "hand":
            limb = next((lb for lb in a.limbs if lb.end_effector == e.id), None)
            if limb:
                region = [s for s in limb.segments if s.startswith(("forearm", "hand"))] + limb.digits
        elif e.kind == "head":
            region = ["head", "jaw"] + [s.id for s in a.segments if s.parent_segment == "head"]
        specs.append(
            {
                "id": f"effector:{e.id}",
                "position": e.position,
                "deform_bones": segment_deform_bones(profile, region),
                "forward": fwd,
            }
        )
    for c in a.contacts:
        if c.kind in ("heel", "ball", "toe_tip", "palm", "head_top"):
            limb_segs: list[str] = []
            for lid, segs in seg_of_limb.items():
                if c.segment in segs:
                    lb = next(x for x in a.limbs if x.id == lid)
                    limb_segs = [s for s in segs if s.startswith(("foot", "toe", "hand"))] + lb.digits
            region = limb_segs or [c.segment]
            if c.kind == "head_top":
                region = ["head"] + [s.id for s in a.segments if s.parent_segment == "head"]
            specs.append(
                {
                    "id": f"contact:{c.id}",
                    "position": c.position,
                    "deform_bones": segment_deform_bones(profile, region),
                    "forward": fwd,
                }
            )
    return specs


def switch_driver_paths(profile: RigProfile) -> list[str]:
    paths = []
    for lc in profile.controls.limbs:
        sw = lc.ik_fk_switch
        if sw is None:
            continue
        for d in profile.extraction.drivers:
            if any((r.bone, r.prop) == (sw.bone, sw.property) for r in _prop_refs(d).values()):
                m = _CON_PATH.match(d.data_path)
                if m and m.group("attr") == "influence":
                    paths.append(d.data_path)
    return paths


def pole_switch(profile: RigProfile, chain_owner: str) -> dict[str, Any] | None:
    """Property that toggles (mutes) the IK constraints of `chain_owner`, found from drivers."""
    for d in profile.extraction.drivers:
        m = _CON_PATH.match(d.data_path)
        if m and m.group("bone") == chain_owner and m.group("attr") in ("mute", "influence"):
            refs = list(_prop_refs(d).values())
            if refs:
                b = profile.extraction.bone_map().get(refs[0].bone)
                cp = b.custom_props.get(refs[0].prop) if b else None
                if cp is not None:
                    lo, hi = cp.ui.get("min", 0), cp.ui.get("max", 1)
                    return {
                        "bone": refs[0].bone,
                        "property": refs[0].prop,
                        "values": [lo, hi],
                        "original": cp.value,
                        "driver": d.data_path,
                    }
    return None


def experiment_args(profile: RigProfile) -> dict[str, Any]:
    a, cm = profile.anatomy, profile.controls
    segs = a.segment_map()
    joints = a.joint_map()
    switches: list[dict[str, Any]] = []
    limbs = []
    for lc in cm.limbs:
        limb = next(lb for lb in a.limbs if lb.id == lc.limb_id)
        upper_seg, lower_seg = (
            (limb.segments[1], limb.segments[2])
            if limb.segments[0].startswith(("clavicle", "hip_bone"))
            else (limb.segments[0], limb.segments[1])
        )
        end_seg = (
            limb.segments[limb.segments.index(lower_seg) + 1]
            if limb.segments.index(lower_seg) + 1 < len(limb.segments)
            else lower_seg
        )
        mid = next(j for j in joints.values() if j.child_segment == lower_seg)
        si = None
        if lc.ik_fk_switch:
            sw = lc.ik_fk_switch
            switches.append({"bone": sw.bone, "property": sw.property, "min": sw.min, "max": sw.max})
            si = len(switches) - 1
        fk = {c.segment: c.bone for c in lc.fk_chain}
        chain = next((ch for ch in lc.ik_chains if not ch.muted), lc.ik_chains[0] if lc.ik_chains else None)
        limbs.append(
            {
                "limb_id": lc.limb_id,
                "switch_index": si,
                "upper": segs[upper_seg].source_bones[0],
                "lower": segs[lower_seg].source_bones[0],
                "end": segs[end_seg].source_bones[0],
                "fk_upper": fk.get(upper_seg),
                "fk_lower": fk.get(lower_seg),
                "ik_effector": lc.ik_effector.bone if lc.ik_effector else None,
                "ik_pole": lc.ik_pole.bone if lc.ik_pole else None,
                "hinge_axis_world": mid.bend.hinge_axis if mid.bend else [1, 0, 0],
                "points_toward": mid.bend.joint_points_toward if mid.bend else [0, 0, 0],
                "joint_id": mid.id,
                "pole_switch": pole_switch(profile, chain.constraint_bone) if chain else None,
            }
        )
    measured = [b.name for b in profile.skeleton.bones] + [b.name for b in profile.extraction.bones if b.use_deform]
    return {
        "armature": profile.source.armature,
        "isolated_copy": True,
        "structural": measured,
        "switches": switches,
        "limbs": limbs,
        "controls": control_bones(profile.extraction),
        "angle_deg": 10.0,
        "offset": 0.03,
    }


def probe_args(profile: RigProfile) -> dict[str, Any]:
    return {
        "armature": profile.source.armature,
        "landmarks": landmark_specs(profile),
        "drivers": [{"data_path": p} for p in switch_driver_paths(profile)],
        "sample_inputs": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
