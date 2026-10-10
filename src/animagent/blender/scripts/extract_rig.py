# Executed INSIDE Blender via the MCP `execute_blender_code` tool.
# Read-only extraction of an armature, its deformed meshes and Rigify metadata.
# Must never modify the scene: no operators, no mode switches, no property writes.
#
# Contract: `main(args) -> dict` (JSON-serialisable), args = {"armature": str, "weight_threshold": float}

import math

import bpy
import numpy as np

SCHEMA_VERSION = "rig_extraction/1"


def _vec(v):
    return [float(x) for x in v]


def _mat(m):
    return [[float(x) for x in row] for row in m]


def _jsonable(value, depth=0):
    """Convert ID-property / RNA values into plain JSON values (best effort)."""
    if depth > 4:
        return repr(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if hasattr(value, "to_dict"):
        try:
            return {k: _jsonable(v, depth + 1) for k, v in value.to_dict().items()}
        except Exception:
            pass
    if hasattr(value, "to_list"):
        try:
            return [_jsonable(v, depth + 1) for v in value.to_list()]
        except Exception:
            pass
    if isinstance(value, bpy.types.ID):
        return {"id_type": type(value).__name__, "name": value.name}
    try:
        return [_jsonable(v, depth + 1) for v in value]
    except TypeError:
        return repr(value)


def _custom_props(owner):
    out = {}
    for key in owner.keys():
        if key.startswith("_") or key in {"cycles"}:
            continue
        entry = {"value": _jsonable(owner[key])}
        try:
            ui = owner.id_properties_ui(key).as_dict()
            entry["ui"] = {k: _jsonable(v) for k, v in ui.items() if k in ("min", "max", "soft_min", "soft_max", "default", "description", "subtype")}
        except Exception:
            pass
        out[key] = entry
    return out


_SKIP_RNA = {"rna_type", "name", "type", "influence", "mute", "is_valid", "active", "show_expanded",
             "is_override_data_editable", "error_location", "error_rotation", "target", "subtarget",
             "pole_target", "pole_subtarget", "owner_space", "target_space", "enabled"}


def _rna_scalars(struct):
    """Generic dump of scalar/enum/vector RNA properties of a constraint-like struct."""
    out = {}
    for prop in struct.bl_rna.properties:
        pid = prop.identifier
        if pid in _SKIP_RNA:
            continue
        try:
            val = getattr(struct, pid)
        except Exception:
            continue
        if prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            if getattr(prop, "is_array", False) or (prop.type in {"INT", "FLOAT", "BOOLEAN"} and getattr(prop, "array_length", 0) > 0):
                try:
                    out[pid] = _jsonable(list(val))
                except TypeError:
                    out[pid] = _jsonable(val)
            elif prop.type == "ENUM" and getattr(prop, "is_enum_flag", False):
                out[pid] = sorted(val)
            else:
                out[pid] = _jsonable(val)
        elif prop.type == "POINTER" and isinstance(val, bpy.types.ID):
            out[pid] = {"id_type": type(val).__name__, "name": val.name}
    return out


def _constraint_targets(c):
    targets = []
    if hasattr(c, "target") and getattr(c, "target", None) is not None:
        targets.append({"object": c.target.name, "subtarget": getattr(c, "subtarget", "") or "", "role": "target"})
    if getattr(c, "pole_target", None) is not None:
        targets.append({"object": c.pole_target.name, "subtarget": c.pole_subtarget or "", "role": "pole"})
    if c.type == "ARMATURE":
        for t in c.targets:
            if t.target is not None:
                targets.append({"object": t.target.name, "subtarget": t.subtarget or "", "role": "armature", "weight": float(t.weight)})
    return targets


def _constraint(c):
    d = {
        "name": c.name,
        "type": c.type,
        "influence": float(c.influence),
        "mute": bool(c.mute),
        "enabled": bool(getattr(c, "enabled", not c.mute)),
        "is_valid": bool(c.is_valid),
        "owner_space": getattr(c, "owner_space", None),
        "target_space": getattr(c, "target_space", None),
        "targets": _constraint_targets(c),
        "settings": _rna_scalars(c),
    }
    return d


def _driver(fc):
    drv = fc.driver
    variables = []
    for v in drv.variables:
        targets = []
        for t in list(v.targets)[: (2 if v.type in {"LOC_DIFF", "ROTATION_DIFF"} else 1)]:
            targets.append({
                "id_type": t.id_type,
                "id_name": t.id.name if t.id else None,
                "data_path": t.data_path,
                "bone_target": t.bone_target,
                "transform_type": t.transform_type,
                "transform_space": t.transform_space,
                "rotation_mode": t.rotation_mode,
            })
        variables.append({"name": v.name, "type": v.type, "targets": targets})
    return {
        "data_path": fc.data_path,
        "array_index": int(fc.array_index),
        "type": drv.type,
        "expression": drv.expression,
        "use_self": bool(drv.use_self),
        "is_valid": bool(drv.is_valid) and bool(fc.is_valid),
        "variables": variables,
        "has_modifiers": len(fc.modifiers) > 0,
        "keyframe_count": len(fc.keyframe_points),
        "keyframes": [[float(k.co[0]), float(k.co[1])] for k in list(fc.keyframe_points)[:32]],
        "modifiers": [_fmodifier(m) for m in fc.modifiers],
    }


def _fmodifier(m):
    d = {"type": m.type, "mute": bool(m.mute), "active": bool(getattr(m, "active", True)),
         "use_influence": bool(m.use_influence), "influence": float(m.influence),
         "use_restricted_range": bool(m.use_restricted_range)}
    if m.type == "GENERATOR":
        d.update({"mode": m.mode, "poly_order": int(m.poly_order), "use_additive": bool(m.use_additive),
                  "coefficients": [float(c) for c in m.coefficients]})
    return d


def _iter_action_fcurves(action, slot=None):
    """Iterate F-Curves of an action, supporting layered actions (Blender 4.4+)."""
    layers = getattr(action, "layers", None)
    if layers is not None and len(layers) > 0:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    if slot is not None and getattr(bag, "slot", None) is not None and bag.slot != slot:
                        continue
                    yield from bag.fcurves
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy


def _animation_summary(ad):
    if ad is None:
        return None
    out = {"action": None, "slot": None, "fcurve_count": 0, "keyed_bones": [], "frame_range": None,
           "nla_tracks": [t.name for t in ad.nla_tracks], "driver_count": len(ad.drivers)}
    act = ad.action
    if act is not None:
        slot = getattr(ad, "action_slot", None)
        fcs = list(_iter_action_fcurves(act, slot))
        bones = set()
        for fc in fcs:
            if fc.data_path.startswith('pose.bones["'):
                bones.add(fc.data_path.split('"')[1])
        out.update({
            "action": act.name,
            "slot": getattr(slot, "identifier", None) if slot else None,
            "fcurve_count": len(fcs),
            "keyed_bones": sorted(bones),
            "frame_range": _vec(act.frame_range),
        })
    return out


def _bone(arm_ob, bone, pbone):
    mw = arm_ob.matrix_world
    rest_world = mw @ bone.matrix_local
    pose = {
        "location": _vec(pbone.location),
        "rotation_mode": pbone.rotation_mode,
        "rotation_quaternion": _vec(pbone.rotation_quaternion),
        "rotation_euler": _vec(pbone.rotation_euler),
        "rotation_axis_angle": _vec(pbone.rotation_axis_angle),
        "scale": _vec(pbone.scale),
        "matrix_armature": _mat(pbone.matrix),
        "head_world": _vec(mw @ pbone.head),
        "tail_world": _vec(mw @ pbone.tail),
        "lock_location": [bool(x) for x in pbone.lock_location],
        "lock_rotation": [bool(x) for x in pbone.lock_rotation],
        "lock_rotation_w": bool(pbone.lock_rotation_w),
        "lock_rotations_4d": bool(pbone.lock_rotations_4d),
        "lock_scale": [bool(x) for x in pbone.lock_scale],
        "ik": {
            "stretch": float(pbone.ik_stretch),
            "lock": [bool(pbone.lock_ik_x), bool(pbone.lock_ik_y), bool(pbone.lock_ik_z)],
            "use_limit": [bool(pbone.use_ik_limit_x), bool(pbone.use_ik_limit_y), bool(pbone.use_ik_limit_z)],
            "min": [float(pbone.ik_min_x), float(pbone.ik_min_y), float(pbone.ik_min_z)],
            "max": [float(pbone.ik_max_x), float(pbone.ik_max_y), float(pbone.ik_max_z)],
            "stiffness": [float(pbone.ik_stiffness_x), float(pbone.ik_stiffness_y), float(pbone.ik_stiffness_z)],
        },
    }
    return {
        "name": bone.name,
        "parent": bone.parent.name if bone.parent else None,
        "children": [c.name for c in bone.children],
        "use_deform": bool(bone.use_deform),
        "use_connect": bool(bone.use_connect),
        "inherit_rotation": bool(bone.use_inherit_rotation),
        "inherit_scale": bone.inherit_scale,
        "use_local_location": bool(bone.use_local_location),
        "bbone_segments": int(bone.bbone_segments),
        "head_world": _vec(rest_world.translation),
        "tail_world": _vec(mw @ bone.tail_local),
        "length": float(bone.length * mw.to_scale()[1]),
        "matrix_local": _mat(bone.matrix_local),
        "matrix_world_rest": _mat(rest_world),
        "collections": [c.name for c in bone.collections],
        "hidden": bool(bone.hide),
        "is_visible": bool(getattr(bone, "is_visible", True) if hasattr(bone, "is_visible") else True),
        "custom_shape": pbone.custom_shape.name if pbone.custom_shape else None,
        "bone_group_color": getattr(pbone.color, "palette", None) if hasattr(pbone, "color") else None,
        "pose": pose,
        "constraints": [_constraint(c) for c in pbone.constraints],
        "custom_props": _custom_props(pbone),
    }


def _weight_summary(ob, deform_bones, threshold):
    """Compact per-vertex-group statistics; never returns per-vertex arrays."""
    me = ob.data
    n = len(me.vertices)
    M = np.array(ob.matrix_world, dtype=np.float64)
    out = {"vertex_count": n, "polygon_count": len(me.polygons), "groups": [], "unweighted_vertex_count": 0,
           "max_influences": 0, "mean_influences": 0.0, "bbox_min": None, "bbox_max": None, "surface_area": 0.0}
    if n == 0:
        return out
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    world = co @ M[:3, :3].T + M[:3, 3]
    out["bbox_min"] = world.min(axis=0).tolist()
    out["bbox_max"] = world.max(axis=0).tolist()

    # Per-vertex surface area (world space approximation) used as a mass proxy.
    vert_area = np.zeros(n)
    npoly = len(me.polygons)
    if npoly:
        area = np.empty(npoly)
        me.polygons.foreach_get("area", area)
        loop_total = np.empty(npoly, dtype=np.int64)
        me.polygons.foreach_get("loop_total", loop_total)
        loop_vert = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", loop_vert)
        scale2 = abs(np.linalg.det(M[:3, :3])) ** (2.0 / 3.0)
        per_loop = np.repeat(area * scale2 / np.maximum(loop_total, 1), loop_total)
        np.add.at(vert_area, loop_vert, per_loop)
    else:
        vert_area[:] = 1.0
    out["surface_area"] = float(vert_area.sum())

    names = [g.name for g in ob.vertex_groups]
    vi, gi, ww = [], [], []
    for v in me.vertices:
        for g in v.groups:
            if g.weight > 0.0:
                vi.append(v.index)
                gi.append(g.group)
                ww.append(g.weight)
    G = len(names)
    if not vi or G == 0:
        out["unweighted_vertex_count"] = n
        out["groups"] = [{"name": nm, "is_deform_bone": nm in deform_bones, "vertex_count": 0} for nm in names]
        return out
    vi = np.array(vi, dtype=np.int64)
    gi = np.array(gi, dtype=np.int64)
    ww = np.array(ww, dtype=np.float64)
    is_def = np.array([nm in deform_bones for nm in names], dtype=bool)
    valid = gi < G
    vi, gi, ww = vi[valid], gi[valid], ww[valid]
    dmask = is_def[gi]
    infl = np.bincount(vi[dmask], minlength=n)
    out["unweighted_vertex_count"] = int((infl == 0).sum())
    out["max_influences"] = int(infl.max())
    out["mean_influences"] = float(infl[infl > 0].mean()) if (infl > 0).any() else 0.0

    a = vert_area[vi]
    p = world[vi]
    cnt = np.bincount(gi, minlength=G)
    wsum = np.bincount(gi, weights=ww, minlength=G)
    wa = np.bincount(gi, weights=ww * a, minlength=G)
    cen = np.zeros((G, 3))
    for k in range(3):
        cen[:, k] = np.bincount(gi, weights=ww * a * p[:, k], minlength=G) / np.maximum(wa, 1e-12)
    d2 = ((p - cen[gi]) ** 2).sum(axis=1)
    rms = np.sqrt(np.bincount(gi, weights=ww * a * d2, minlength=G) / np.maximum(wa, 1e-12))
    strong = ww >= threshold
    bmin = np.full((G, 3), np.inf)
    bmax = np.full((G, 3), -np.inf)
    np.minimum.at(bmin, gi[strong], p[strong])
    np.maximum.at(bmax, gi[strong], p[strong])
    strong_cnt = np.bincount(gi[strong], minlength=G)
    for g, nm in enumerate(names):
        entry = {"name": nm, "is_deform_bone": bool(is_def[g]), "vertex_count": int(cnt[g])}
        if cnt[g]:
            entry.update({
                "weight_sum": float(wsum[g]),
                "area_weighted": float(wa[g]),
                "centroid_world": cen[g].tolist(),
                "rms_radius": float(rms[g]),
                "strong_vertex_count": int(strong_cnt[g]),
            })
            if strong_cnt[g]:
                entry["bbox_min"] = bmin[g].tolist()
                entry["bbox_max"] = bmax[g].tolist()
        out["groups"].append(entry)
    return out


def _mesh_entry(ob, arm_ob, deform_bones, threshold):
    mods = []
    for i, m in enumerate(ob.modifiers):
        e = {"index": i, "name": m.name, "type": m.type, "show_viewport": bool(m.show_viewport), "show_render": bool(m.show_render)}
        if m.type == "ARMATURE":
            e.update({
                "object": m.object.name if m.object else None,
                "use_vertex_groups": bool(m.use_vertex_groups),
                "use_bone_envelopes": bool(m.use_bone_envelopes),
                "use_deform_preserve_volume": bool(m.use_deform_preserve_volume),
                "use_multi_modifier": bool(m.use_multi_modifier),
                "vertex_group": m.vertex_group or None,
            })
        mods.append(e)
    shape_keys = ob.data.shape_keys
    entry = {
        "name": ob.name,
        "data_name": ob.data.name,
        "data_users": int(ob.data.users),
        "parent": ob.parent.name if ob.parent else None,
        "parent_type": ob.parent_type,
        "parent_bone": ob.parent_bone or None,
        "matrix_world": _mat(ob.matrix_world),
        "hide_viewport": bool(ob.hide_viewport),
        "visible": bool(ob.visible_get()) if hasattr(ob, "visible_get") else True,
        "modifiers": mods,
        "shape_key_count": len(shape_keys.key_blocks) if shape_keys else 0,
        "shape_key_driver_count": len(shape_keys.animation_data.drivers) if shape_keys and shape_keys.animation_data else 0,
        "material_count": len(ob.material_slots),
    }
    entry["weights"] = _weight_summary(ob, deform_bones, threshold)
    return entry


def _find_metarig(arm_ob):
    for ob in bpy.data.objects:
        if ob.type == "ARMATURE" and ob is not arm_ob:
            tgt = getattr(ob.data, "rigify_target_rig", None)
            if tgt is arm_ob:
                return ob
    return None


def _rigify_info(arm_ob):
    rig_id = arm_ob.data.get("rig_id")
    info = {"rig_id": rig_id, "is_rigify_generated": rig_id is not None, "rig_ui_text": None,
            "metarig": None, "metarig_bones": []}
    ui = arm_ob.get("rig_ui")
    if ui is not None and isinstance(ui, bpy.types.ID):
        info["rig_ui_text"] = ui.name
    meta = _find_metarig(arm_ob)
    if meta is not None:
        info["metarig"] = meta.name
        mw = meta.matrix_world
        for b in meta.data.bones:
            pb = meta.pose.bones[b.name]
            info["metarig_bones"].append({
                "name": b.name,
                "parent": b.parent.name if b.parent else None,
                "use_connect": bool(b.use_connect),
                "rigify_type": getattr(pb, "rigify_type", "") or "",
                "head_world": _vec(mw @ b.head_local),
                "tail_world": _vec(mw @ b.tail_local),
            })
    return info


def _attachments(arm_ob):
    """Objects attached to the rig other than deformed meshes (bone parents, constraints)."""
    out = []
    for ob in bpy.data.objects:
        if ob is arm_ob:
            continue
        rels = []
        if ob.parent is arm_ob:
            rels.append({"kind": "parent", "parent_type": ob.parent_type, "bone": ob.parent_bone or None})
        for c in ob.constraints:
            if getattr(c, "target", None) is arm_ob:
                rels.append({"kind": "constraint", "type": c.type, "bone": getattr(c, "subtarget", "") or None, "influence": float(c.influence), "mute": bool(c.mute)})
        if rels:
            out.append({"name": ob.name, "type": ob.type, "relations": rels, "hide_viewport": bool(ob.hide_viewport)})
    return out


def main(args):
    name = args["armature"]
    threshold = float(args.get("weight_threshold", 0.1))
    arm_ob = bpy.data.objects.get(name)
    if arm_ob is None:
        armatures = [o.name for o in bpy.data.objects if o.type == "ARMATURE"]
        raise ValueError("No object named {!r}. Armatures in file: {}".format(name, armatures))
    if arm_ob.type != "ARMATURE":
        raise ValueError("Object {!r} is a {}, not an ARMATURE".format(name, arm_ob.type))

    scene = bpy.context.scene
    arm = arm_ob.data
    bones = [_bone(arm_ob, b, arm_ob.pose.bones[b.name]) for b in arm.bones]
    deform_bones = {b.name for b in arm.bones if b.use_deform}

    drivers = []
    if arm_ob.animation_data:
        drivers += [dict(_driver(fc), owner="object") for fc in arm_ob.animation_data.drivers]
    if arm.animation_data:
        drivers += [dict(_driver(fc), owner="armature_data") for fc in arm.animation_data.drivers]

    meshes = []
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        deformed = any(m.type == "ARMATURE" and m.object is arm_ob for m in ob.modifiers)
        if deformed:
            meshes.append(_mesh_entry(ob, arm_ob, deform_bones, threshold))

    return {
        "schema_version": SCHEMA_VERSION,
        "blender": {
            "version": bpy.app.version_string,
            "file_path": bpy.data.filepath,
            "is_dirty": bool(bpy.data.is_dirty),
            "scene": scene.name,
            "frame_current": int(scene.frame_current),
            "fps": float(scene.render.fps / scene.render.fps_base),
            "unit_system": scene.unit_settings.system,
            "unit_scale_length": float(scene.unit_settings.scale_length),
            "background": bool(bpy.app.background),
        },
        "armature": {
            "name": arm_ob.name,
            "data_name": arm.name,
            "matrix_world": _mat(arm_ob.matrix_world),
            "pose_position": arm.pose_position,
            "display_type": arm.display_type,
            "bone_count": len(arm.bones),
            "collections": [
                {"name": c.name, "visible": bool(c.is_visible), "parent": c.parent.name if c.parent else None,
                 "bone_count": len(c.bones)}
                for c in arm.collections_all
            ],
            "custom_props": _custom_props(arm_ob),
            "data_custom_props": _custom_props(arm),
            "animation": _animation_summary(arm_ob.animation_data),
            "hide_viewport": bool(arm_ob.hide_viewport),
        },
        "rigify": _rigify_info(arm_ob),
        "bones": bones,
        "drivers": drivers,
        "meshes": meshes,
        "attachments": _attachments(arm_ob),
    }
