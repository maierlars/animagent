# Executed INSIDE Blender via MCP. Read-only.
# Computes hashes of everything an inspection must not change: pose, rest pose, custom
# properties, animation (keyframes + drivers), object transforms, mesh geometry/weights,
# data-block inventory, text contents, scene/render/view-layer settings, collections,
# materials/images/node groups (structure), ID properties, mode/selection/frame.
# NOT covered: pixel data, full node-socket values, UI/screen layout, preferences, packed files. Used before/after analysis to prove reversibility.
#
# Contract: main({"armature": str | None}) -> {"hashes": {category: hex}, "info": {...}}

import hashlib

import bpy
import numpy as np


def _h():
    return hashlib.sha256()


def _upd(h, *vals):
    for v in vals:
        h.update(repr(v).encode("utf-8"))
        h.update(b"\x1f")


def _upd_array(h, arr):
    h.update(np.ascontiguousarray(np.round(arr, 7)).tobytes())


def _iter_action_fcurves(action):
    layers = getattr(action, "layers", None)
    if layers is not None and len(layers) > 0:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    yield from bag.fcurves
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy


def _fcurve_hash(h, fc):
    _upd(h, fc.data_path, fc.array_index, fc.mute, len(fc.keyframe_points), len(fc.modifiers))
    n = len(fc.keyframe_points)
    if n:
        for attr in ("co", "handle_left", "handle_right"):
            buf = np.empty(n * 2)
            fc.keyframe_points.foreach_get(attr, buf)
            _upd_array(h, buf)
        _upd(h, [k.interpolation for k in fc.keyframe_points])


def _props(owner):
    out = []
    for k in sorted(owner.keys()):
        v = owner[k]
        if hasattr(v, "to_dict"):
            v = v.to_dict()
        elif hasattr(v, "to_list"):
            v = v.to_list()
        elif isinstance(v, bpy.types.ID):
            v = ("ID", v.name)
        out.append((k, v if isinstance(v, (int, float, str, bool, list, dict, tuple)) else repr(v)))
    return out


def _pose_hash(ob):
    h = _h()
    for pb in ob.pose.bones:
        _upd(h, pb.name, pb.rotation_mode, tuple(pb.location), tuple(pb.rotation_quaternion),
             tuple(pb.rotation_euler), tuple(pb.rotation_axis_angle), tuple(pb.scale), _props(pb))
        for c in pb.constraints:
            _upd(h, c.name, c.type, round(c.influence, 7), c.mute)
    return h.hexdigest()


def _rest_hash(ob):
    h = _h()
    for b in ob.data.bones:
        _upd(h, b.name, b.parent.name if b.parent else None, b.use_deform, b.use_connect, b.hide)
        _upd_array(h, np.array(b.matrix_local, dtype=np.float64))
        _upd_array(h, np.array(b.tail_local, dtype=np.float64))
    _upd(h, ob.data.pose_position, [c.name for c in ob.data.collections_all],
         [c.is_visible for c in ob.data.collections_all], _props(ob), _props(ob.data))
    return h.hexdigest()


def _anim_hash():
    h = _h()
    for act in sorted(bpy.data.actions, key=lambda a: a.name):
        _upd(h, act.name, act.users)
        for fc in _iter_action_fcurves(act):
            _fcurve_hash(h, fc)
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        for owner in (ob, ob.data):
            ad = getattr(owner, "animation_data", None) if owner is not None else None
            if ad is None:
                continue
            _upd(h, ob.name, ad.action.name if ad.action else None, [t.name for t in ad.nla_tracks])
            for fc in ad.drivers:
                d = fc.driver
                _upd(h, fc.data_path, fc.array_index, d.type, d.expression, d.is_valid,
                     [(v.name, v.type, [(t.id.name if t.id else None, t.data_path, t.bone_target) for t in v.targets]) for v in d.variables])
                _fcurve_hash(h, fc)
    return h.hexdigest()


def _objects_hash():
    h = _h()
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        _upd(h, ob.name, ob.type, ob.parent.name if ob.parent else None, ob.parent_type, ob.parent_bone,
             ob.hide_viewport, ob.hide_render, ob.data.name if ob.data else None,
             [(m.name, m.type, m.show_viewport) for m in ob.modifiers],
             [(c.name, c.type, round(c.influence, 7), c.mute) for c in ob.constraints], _props(ob))
        _upd_array(h, np.array(ob.matrix_basis, dtype=np.float64))
    return h.hexdigest()


def _mesh_hash():
    h = _h()
    for me in sorted(bpy.data.meshes, key=lambda m: m.name):
        n = len(me.vertices)
        _upd(h, me.name, n, len(me.edges), len(me.polygons))
        if n:
            co = np.empty(n * 3)
            me.vertices.foreach_get("co", co)
            _upd_array(h, co)
        if me.shape_keys:
            for kb in me.shape_keys.key_blocks:
                _upd(h, kb.name, round(kb.value, 7), kb.mute)
    return h.hexdigest()


def _weights_hash():
    h = _h()
    for ob in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
        _upd(h, ob.name, [g.name for g in ob.vertex_groups])
        if ob.vertex_groups:
            acc = []
            for v in ob.data.vertices:
                for g in v.groups:
                    acc.append((v.index, g.group, g.weight))
            if acc:
                _upd_array(h, np.array(acc, dtype=np.float64))
    return h.hexdigest()


def _datablock_hash():
    h = _h()
    counts = {}
    for attr in ("objects", "meshes", "armatures", "actions", "materials", "collections", "images",
                 "texts", "node_groups", "curves", "grease_pencils", "cameras", "lights", "scenes"):
        coll = getattr(bpy.data, attr, None)
        if coll is not None:
            counts[attr] = len(coll)
            _upd(h, attr, sorted(x.name for x in coll))
    return h.hexdigest(), counts


def _texts_hash():
    h = _h()
    for t in sorted(bpy.data.texts, key=lambda t: t.name):
        _upd(h, t.name, t.use_module, t.as_string())
    return h.hexdigest()


def _scene_hash():
    h = _h()
    for sc in sorted(bpy.data.scenes, key=lambda s: s.name):
        r = sc.render
        _upd(h, sc.name, sc.frame_start, sc.frame_end, sc.frame_step, r.fps, r.fps_base, r.engine,
             r.resolution_x, r.resolution_y, sc.camera.name if sc.camera else None,
             sc.world.name if sc.world else None, sc.unit_settings.system, sc.unit_settings.scale_length,
             sc.tool_settings.use_keyframe_insert_auto, _props(sc))
        for vl in sc.view_layers:
            stack = [vl.layer_collection]
            while stack:
                lc = stack.pop()
                _upd(h, vl.name, lc.name, lc.exclude, lc.hide_viewport, lc.holdout, lc.indirect_only)
                stack.extend(lc.children)
    return h.hexdigest()


def _collections_hash():
    h = _h()
    for c in sorted(bpy.data.collections, key=lambda c: c.name):
        _upd(h, c.name, sorted(o.name for o in c.objects), sorted(ch.name for ch in c.children),
             c.hide_viewport, c.hide_render, c.hide_select, _props(c))
    return h.hexdigest()


def _materials_images_hash():
    h = _h()
    for m in sorted(bpy.data.materials, key=lambda m: m.name):
        nodes = sorted((n.name, n.bl_idname, len(n.inputs)) for n in m.node_tree.nodes) if m.node_tree else []
        _upd(h, m.name, m.users, nodes, _props(m))
    for i in sorted(bpy.data.images, key=lambda i: i.name):
        _upd(h, i.name, i.filepath, i.source, i.packed_file is not None, _props(i))
    for ng in sorted(bpy.data.node_groups, key=lambda n: n.name):
        _upd(h, ng.name, sorted((n.name, n.bl_idname) for n in ng.nodes))
    return h.hexdigest()


def _id_props_hash():
    """Custom properties on every ID type (catches add-ons storing data on datablocks)."""
    h = _h()
    for attr in ("objects", "meshes", "armatures", "actions", "materials", "collections", "scenes", "texts",
                 "images", "node_groups", "worlds", "cameras", "lights"):
        for idb in sorted(getattr(bpy.data, attr, []), key=lambda x: x.name):
            p = _props(idb)
            if p:
                _upd(h, attr, idb.name, p)
    for a in sorted(bpy.data.actions, key=lambda a: a.name):
        _upd(h, a.name, a.use_fake_user, len(getattr(a, "layers", [])), [s.identifier for s in getattr(a, "slots", [])])
    return h.hexdigest()


def _ui_state():
    ctx = bpy.context
    view_layer = ctx.view_layer
    active = view_layer.objects.active
    return {
        "mode": ctx.mode,
        "active_object": active.name if active else None,
        "selected": sorted(o.name for o in view_layer.objects if o.select_get()),
        "frame_current": ctx.scene.frame_current,
        "scene": ctx.scene.name,
    }


def main(args):
    db_hash, counts = _datablock_hash()
    hashes = {
        "animation": _anim_hash(),
        "objects": _objects_hash(),
        "meshes": _mesh_hash(),
        "vertex_weights": _weights_hash(),
        "datablocks": db_hash,
        "texts": _texts_hash(),
        "scene_settings": _scene_hash(),
        "collections": _collections_hash(),
        "materials_images": _materials_images_hash(),
        "id_properties": _id_props_hash(),
    }
    name = args.get("armature")
    if name:
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != "ARMATURE":
            raise ValueError("No armature named {!r}".format(name))
        hashes["pose"] = _pose_hash(ob)
        hashes["rest"] = _rest_hash(ob)
    ui = _ui_state()
    h = _h()
    _upd(h, sorted(ui.items()))
    hashes["ui_state"] = h.hexdigest()
    return {"hashes": hashes, "info": {"datablock_counts": counts, "ui_state": ui, "is_dirty": bool(bpy.data.is_dirty)}}
