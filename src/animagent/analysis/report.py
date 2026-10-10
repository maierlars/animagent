"""Compact, human/LLM-readable JSON report of a rig profile (schema `rig_report/1`).

Small by design (a few KB): no matrices, no vertex data, rounded numbers.
"""

from __future__ import annotations

from typing import Any

from animagent.analysis.geom import rnd
from animagent.models.controls import ControlRef
from animagent.models.profile import RigProfile


def _ctrl(c: ControlRef | None) -> str | None:
    return c.bone if c else None


def build_report(p: RigProfile) -> dict[str, Any]:
    a, cm, ex = p.anatomy, p.controls, p.extraction
    joints = a.joint_map()
    segs = a.segment_map()
    limb_ctrls = {lc.limb_id: lc for lc in cm.limbs}
    limbs = []
    for lb in a.limbs:
        if lb.kind.value == "appendage":
            continue
        entry: dict[str, Any] = {
            "id": lb.id,
            "kind": lb.kind.value,
            "side": lb.side.value,
            "confidence": lb.confidence,
            "segments": {s: segs[s].source_bones[0] for s in lb.segments},
            "chain_length": round(lb.chain_length, 3),
            "end_effector": lb.end_effector,
            "digits": len(lb.digits),
            "joints": [],
        }
        for jid in lb.joints:
            j = joints[jid]
            je: dict[str, Any] = {"id": j.id, "kind": j.kind.value, "dof": j.dof_hint, "position": rnd(j.position, 3)}
            if j.bend:
                je["bend"] = {
                    "points_toward_body": j.bend.body_relative,
                    "hinge_axis_local": f"{j.bend.hinge_axis_local} of {j.bend.local_axis_bone}",
                    "rest_flexion_deg": j.bend.rest_flexion_deg,
                    "matches_humanoid_prior": j.bend.matches_prior,
                    "source": j.bend.source,
                    "confidence": j.bend.confidence,
                }
            entry["joints"].append(je)
        lc = limb_ctrls.get(lb.id)
        if lc:
            sw = lc.ik_fk_switch
            entry["controls"] = {
                "fk": [c.bone for c in lc.fk_chain],
                "ik_effector": _ctrl(lc.ik_effector),
                "ik_pole": _ctrl(lc.ik_pole),
                "ik_chain": [c.bone for c in lc.ik_chain_controls],
                "ik_fk_switch": (
                    {
                        "bone": sw.bone,
                        "property": sw.property,
                        "ik_value": sw.ik_value,
                        "fk_value": sw.fk_value,
                        "current": sw.current_value,
                    }
                    if sw
                    else None
                ),
                "tweaks": [c.bone for c in lc.other_controls if c.role == "tweak"],
                "confidence": lc.confidence,
            }
        limbs.append(entry)
    seg_ctrl = {sc.segment_id: sc.controls for sc in cm.segments}
    trunk = []
    for s in a.segments:
        if s.kind.value in ("digit", "appendage") or any(
            s.id in lb.segments for lb in a.limbs if lb.kind.value in ("arm", "leg")
        ):
            continue
        trunk.append(
            {
                "id": s.id,
                "kind": s.kind.value,
                "bones": s.source_bones,
                "confidence": s.confidence,
                "relative_mass": s.mass.relative_mass if s.mass else None,
                "controls": [f"{c.bone} ({c.role})" for c in seg_ctrl.get(s.id, [])[:4]],
            }
        )
    appendages = [
        {
            "id": s.id,
            "bones": len(s.source_bones),
            "attached_to": s.parent_segment,
            "controls": [c.bone for c in seg_ctrl.get(s.id, []) if c.role == "fk"][:4],
        }
        for s in a.segments
        if s.kind.value == "appendage"
    ]
    mass = sorted(((s.id, s.mass.relative_mass) for s in a.segments if s.mass), key=lambda x: -x[1])
    f = a.body_frame
    return {
        "schema_version": "rig_report/1",
        "armature": ex.armature.name,
        "blend_file": ex.blender.file_path,
        "blender": ex.blender.version,
        "rig_kind": cm.rig_kind,
        "skeleton_strategy": p.skeleton.strategy,
        "counts": {
            "bones": len(ex.bones),
            "structural": len(p.skeleton.bones),
            "controls": len(cm.control_bones),
            "deform": sum(b.use_deform for b in ex.bones),
            "drivers": len(ex.drivers),
            "meshes": len(ex.meshes),
            "segments": len(a.segments),
            "joints": len(a.joints),
        },
        "body_frame": {
            "up": f.up,
            "forward": f.forward,
            "left": f.left,
            "height": f.height,
            "ground": f.ground_height,
            "confidence": f.confidence,
        },
        "symmetry": {
            "pairs": a.symmetry.pairs,
            "median_mirror_error": a.symmetry.median_mirror_error,
            "asymmetric_at_rest": a.symmetry.asymmetric_pairs[:10],
        },
        "trunk": trunk,
        "limbs": limbs,
        "appendages": appendages,
        "root_control": _ctrl(cm.root_control),
        "body_controls": [f"{c.bone} ({c.role}{', ' + c.segment if c.segment else ''})" for c in cm.body_controls],
        "global_properties": [f"{gp.bone}[{gp.property}]={gp.current_value}" for gp in cm.global_properties][:20],
        "prop_mounts": [c.bone for c in cm.prop_mounts],
        "heaviest_segments": [{"id": k, "relative_mass": m} for k, m in mass[:8]],
        "ambiguities": [
            {"id": x.id, "question": x.question, "best": x.deterministic_best, "confidence": x.confidence}
            for x in a.ambiguities
        ],
        "resolutions": [r.model_dump() for r in a.resolutions],
        "readonly_check": (
            {"identical": p.readonly_check.identical, "changed": p.readonly_check.changed_categories}
            if p.readonly_check
            else None
        ),
        "diagnostics": [d.model_dump() for d in p.diagnostics],
    }
