# Executed INSIDE Blender via MCP. Read-only.
# Independent measurements used to validate the analysis. Nothing here uses the analyzer's
# interpretation except the list of points/bones to measure, which the caller passes in.
#
# Contract: main({
#   "armature": str,
#   "landmarks": [{"id": str, "position": [x,y,z], "deform_bones": [str], "forward": [x,y,z]}],
#   "drivers": [{"data_path": str}],       # constraint-influence drivers to evaluate with Blender's F-curves
#   "sample_inputs": [float, ...],
# }) -> dict

import math

import bpy
import numpy as np


def _bone_frames(arm):
    mw = np.array(arm.matrix_world)
    worst_ortho, dets, gaps = 0.0, [], []
    for b in arm.data.bones:
        m = mw @ np.array(b.matrix_local)
        r = m[:3, :3]
        rn = r / np.linalg.norm(r, axis=0)
        worst_ortho = max(worst_ortho, float(np.abs(rn.T @ rn - np.eye(3)).max()))
        dets.append(float(np.linalg.det(rn)))
        # Y axis must point from head to tail (Blender bone convention).
        head = m[:3, 3]
        tail = mw @ np.append(np.array(b.tail_local), 1.0)
        d = tail[:3] - head
        if np.linalg.norm(d) > 1e-9:
            gaps.append(float(1.0 - (rn[:, 1] @ (d / np.linalg.norm(d)))))
    return {
        "bones": len(arm.data.bones),
        "max_orthonormality_error": worst_ortho,
        "min_det": min(dets), "max_det": max(dets),
        "max_y_axis_misalignment": max(gaps) if gaps else 0.0,
        "armature_matrix_det": float(np.linalg.det(mw[:3, :3])),
    }


def _connectivity(arm):
    """Raw parent-tail to child-head distances for every bone (no interpretation)."""
    mw = np.array(arm.matrix_world)
    out = {}
    for b in arm.data.bones:
        if b.parent is None:
            continue
        ph = mw @ np.append(np.array(b.parent.tail_local), 1.0)
        ch = mw @ np.append(np.array(b.head_local), 1.0)
        out[b.name] = [b.parent.name, float(np.linalg.norm(ph[:3] - ch[:3])), bool(b.use_connect)]
    return out


def _dominant_clouds(arm):
    """World rest positions of skin vertices grouped by their dominant deform group."""
    deform = {b.name for b in arm.data.bones if b.use_deform}
    clouds = {}
    ground = math.inf
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not any(m.type == "ARMATURE" and m.object is arm for m in ob.modifiers):
            continue
        if not ob.visible_get():
            continue
        names = [g.name for g in ob.vertex_groups]
        me = ob.data
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(ob.matrix_world)
        world = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        ground = min(ground, float(world[:, 2].min()))
        for v in me.vertices:
            best, bw = None, 0.0
            for g in v.groups:
                if g.weight > bw and g.group < len(names) and names[g.group] in deform:
                    best, bw = names[g.group], g.weight
            if best is not None:
                clouds.setdefault(best, []).append(world[v.index])
    return {k: np.array(v) for k, v in clouds.items()}, ground


def _landmarks(clouds, landmarks):
    out = []
    for lm in landmarks:
        pts = [clouds[b] for b in lm["deform_bones"] if b in clouds]
        p = np.array(lm["position"], dtype=float)
        entry = {"id": lm["id"], "deform_bones": lm["deform_bones"]}
        if not pts:
            entry["vertex_count"] = 0
            out.append(entry)
            continue
        cloud = np.concatenate(pts)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        outside = np.maximum(0.0, np.maximum(lo - p, p - hi))
        fwd = np.array(lm.get("forward", [0, -1, 0]), dtype=float)
        proj = cloud @ fwd
        entry.update({
            "vertex_count": int(len(cloud)),
            "centroid": cloud.mean(axis=0).tolist(),
            "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
            "nearest_vertex_distance": float(np.linalg.norm(cloud - p, axis=1).min()),
            "distance_outside_bbox": float(np.linalg.norm(outside)),
            "lowest_point": cloud[np.argmin(cloud[:, 2])].tolist(),
            "forward_most_point": cloud[np.argmax(proj)].tolist(),
            "backward_most_point": cloud[np.argmin(proj)].tolist(),
            "extent": (hi - lo).tolist(),
        })
        out.append(entry)
    return out


def _drivers(arm, wanted, inputs):
    """Blender's own evaluation of the driver F-curve mapping (keyframes + modifiers) at sample
    inputs, plus the *currently evaluated* constraint influence and driver variable values."""
    ad = arm.animation_data
    out = []
    if ad is None:
        return out
    by_path = {fc.data_path: fc for fc in ad.drivers}
    dg = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm.evaluated_get(dg)
    for w in wanted:
        fc = by_path.get(w["data_path"])
        if fc is None:
            out.append({"data_path": w["data_path"], "found": False})
            continue
        d = fc.driver
        entry = {
            "data_path": fc.data_path, "found": True, "type": d.type, "expression": d.expression,
            "is_valid": bool(d.is_valid),
            "variables": [{"name": v.name, "type": v.type,
                           "targets": [{"id": t.id.name if t.id else None, "data_path": t.data_path} for t in v.targets[:1]]}
                          for v in d.variables],
            "modifiers": [{"type": m.type, "mode": getattr(m, "mode", None),
                           "coefficients": list(getattr(m, "coefficients", []))} for m in fc.modifiers],
            # F-curve mapping of the raw driver value -> property value, evaluated by Blender.
            "fcurve_mapping": {str(x): float(fc.evaluate(x)) for x in inputs},
        }
        # Current values: the switch property and the evaluated constraint influence.
        try:
            bone = fc.data_path.split('"')[1]
            con = fc.data_path.split('"')[3]
            entry["current_influence_original"] = float(arm.pose.bones[bone].constraints[con].influence)
            entry["current_influence_evaluated"] = float(arm_eval.pose.bones[bone].constraints[con].influence)
        except Exception as ex:
            entry["influence_error"] = str(ex)
        vals = []
        for v in d.variables:
            t = v.targets[0]
            try:
                vals.append(float(t.id.path_resolve(t.data_path)) if t.id and t.data_path else None)
            except Exception:
                vals.append(None)
        entry["current_variable_values"] = vals
        out.append(entry)
    return out


def main(args):
    arm = bpy.data.objects[args["armature"]]
    clouds, ground = _dominant_clouds(arm)
    return {
        "bone_frames": _bone_frames(arm),
        "connectivity": _connectivity(arm),
        "mesh_ground_z": ground,
        "landmarks": _landmarks(clouds, args.get("landmarks", [])),
        "drivers": _drivers(arm, args.get("drivers", []), args.get("sample_inputs", [0.0, 0.5, 1.0])),
        "dominant_cloud_sizes": {k: int(len(v)) for k, v in clouds.items()},
    }
