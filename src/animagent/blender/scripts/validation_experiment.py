# Executed INSIDE an ISOLATED BACKGROUND Blender (a copy opened from disk, never saved).
# Perturbs controls and measures the evaluated motion of structural bones. This MODIFIES the
# pose of the loaded copy, so it refuses to run in an interactive session.
#
# Contract: main({
#   "armature": str, "isolated_copy": true,
#   "structural": [bone names],                  # bones whose motion is measured
#   "switches": [{"bone": str, "property": str, "min": float, "max": float}],
#   "limbs": [{"limb_id", "switch_index": int|null, "upper", "lower", "end",
#              "fk_upper", "fk_lower", "ik_effector", "ik_pole", "hinge_axis_world", "points_toward",
#              "pole_switch": {"bone", "property", "values": [..]} | null}],
#   "controls": [control names], "angle_deg": float, "offset": float,
# }) -> dict

import math

import bpy
from mathutils import Matrix, Vector

TOL = 1e-4


def _update():
    bpy.context.view_layer.update()


def _snapshot(arm, names):
    mw = arm.matrix_world
    out = {}
    for n in names:
        pb = arm.pose.bones.get(n)
        if pb is not None:
            out[n] = (mw @ pb.head, mw @ pb.tail, (mw @ pb.matrix).to_quaternion())
    return out


def _moved(before, after):
    moved = {}
    for n, (h0, t0, q0) in before.items():
        h1, t1, q1 = after[n]
        d = max((h1 - h0).length, (t1 - t0).length)
        ang = q0.rotation_difference(q1).angle
        if d > TOL or ang > 1e-3:
            moved[n] = [round(d, 6), round(ang, 6)]
    return moved


def _with_basis(pb, new_basis, measure):
    old = pb.matrix_basis.copy()
    pb.matrix_basis = new_basis
    _update()
    try:
        return measure()
    finally:
        pb.matrix_basis = old
        _update()


def _with_armature_translation(pb, vec, measure):
    """Translate a pose bone by `vec` in ARMATURE space (independent of its location settings)."""
    old = pb.matrix_basis.copy()
    pb.matrix = Matrix.Translation(vec) @ pb.matrix
    _update()
    try:
        return measure()
    finally:
        pb.matrix_basis = old
        _update()


def _flex(arm, upper, lower):
    a, b = arm.pose.bones[upper], arm.pose.bones[lower]
    u = (b.head - a.head).normalized() if (b.head - a.head).length > 1e-9 else (a.tail - a.head).normalized()
    l_ = (b.tail - b.head).normalized()
    return math.degrees(u.angle(l_))


def _set_prop(arm, sw, value):
    arm.pose.bones[sw["bone"]][sw["property"]] = type(arm.pose.bones[sw["bone"]][sw["property"]])(value)
    _update()


def main(args):
    if not bpy.app.background or not args.get("isolated_copy"):
        raise RuntimeError("validation_experiment modifies the pose; it only runs in an isolated background copy")
    arm = bpy.data.objects[args["armature"]]
    angle = math.radians(float(args.get("angle_deg", 10.0)))
    offset = float(args.get("offset", 0.03))
    structural = [n for n in args["structural"] if n in arm.pose.bones]
    mwi = arm.matrix_world.inverted()

    # Neutral state: no action, all pose channels at rest; rig mechanics (constraints/drivers) stay active.
    notes = []
    if arm.animation_data and arm.animation_data.action:
        notes.append(f"unassigned action {arm.animation_data.action.name!r} in the isolated copy")
        arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    _update()
    switches = args.get("switches", [])
    original_props = [arm.pose.bones[s["bone"]][s["property"]] for s in switches]

    out = {"notes": notes, "switch_tests": [], "hinge_tests": [], "ik_bend_tests": [], "control_sweep": {}}

    # 1. IK/FK switch semantics, purely by motion.
    for si, sw in enumerate(switches):
        limb = next((lb for lb in args["limbs"] if lb.get("switch_index") == si), None)
        if limb is None:
            continue
        limb_bones = [limb["upper"], limb["lower"], limb["end"]]
        res = {"limb_id": limb["limb_id"], "switch": sw, "by_value": {}}
        for val in (sw["min"], sw["max"]):
            _set_prop(arm, sw, val)
            base = _snapshot(arm, limb_bones)
            entry = {}
            eff = arm.pose.bones.get(limb.get("ik_effector") or "")
            if eff is not None:
                toward = (arm.pose.bones[limb["upper"]].head - eff.head).normalized() * offset
                entry["effector_motion"] = _with_armature_translation(
                    eff, toward, lambda: _moved(base, _snapshot(arm, limb_bones)))
            fk = arm.pose.bones.get(limb.get("fk_upper") or "")
            if fk is not None:
                entry["fk_motion"] = _with_basis(fk, fk.matrix_basis @ Matrix.Rotation(angle, 4, "X"),
                                                 lambda: _moved(base, _snapshot(arm, limb_bones)))
            res["by_value"][str(val)] = entry
        _set_prop(arm, sw, original_props[si])
        out["switch_tests"].append(res)

    def mode_value(limb, want):
        si = limb.get("switch_index")
        if si is None:
            return None
        t = out["switch_tests"][[x["limb_id"] for x in out["switch_tests"]].index(limb["limb_id"])]
        for val, e in t["by_value"].items():
            eff = bool(e.get("effector_motion"))
            fk = bool(e.get("fk_motion"))
            if (want == "ik" and eff and not fk) or (want == "fk" and fk and not eff):
                return float(val)
        return None

    # 2. Hinge direction in FK: rotate the lower FK control about the inferred world hinge axis.
    for limb in args["limbs"]:
        fkv = mode_value(limb, "fk")
        sw = switches[limb["switch_index"]] if limb.get("switch_index") is not None else None
        if sw is not None and fkv is not None:
            _set_prop(arm, sw, fkv)
        pb = arm.pose.bones.get(limb.get("fk_lower") or "")
        res = {"limb_id": limb["limb_id"], "mode_value": fkv, "control": limb.get("fk_lower")}
        if pb is not None:
            joint = arm.pose.bones[limb["lower"]].head.copy()
            axis = (mwi.to_3x3() @ Vector(limb["hinge_axis_world"])).normalized()
            flex0 = _flex(arm, limb["upper"], limb["lower"])
            end0 = arm.matrix_world @ arm.pose.bones[limb["lower"]].tail.copy()
            for sign in (+1, -1):
                rot = Matrix.Translation(joint) @ Matrix.Rotation(sign * angle, 4, axis) @ Matrix.Translation(-joint)
                old = pb.matrix_basis.copy()
                pb.matrix = rot @ pb.matrix
                _update()
                flex1 = _flex(arm, limb["upper"], limb["lower"])
                end1 = arm.matrix_world @ arm.pose.bones[limb["lower"]].tail
                d = end1 - end0
                res["plus" if sign > 0 else "minus"] = {
                    "flexion_before_deg": flex0, "flexion_after_deg": flex1,
                    "end_displacement": list(d),
                    "end_displacement_dot_points_toward": (d.normalized().dot(Vector(limb["points_toward"]))
                                                           if d.length > 1e-9 else 0.0),
                }
                pb.matrix_basis = old
                _update()
        if sw is not None:
            _set_prop(arm, sw, original_props[limb["switch_index"]])
        out["hinge_tests"].append(res)

    # 3. Bend direction in IK: pull the effector toward the limb root; the middle joint pops out.
    for limb in args["limbs"]:
        ikv = mode_value(limb, "ik")
        sw = switches[limb["switch_index"]] if limb.get("switch_index") is not None else None
        eff = arm.pose.bones.get(limb.get("ik_effector") or "")
        if sw is None or ikv is None or eff is None:
            out["ik_bend_tests"].append({"limb_id": limb["limb_id"], "skipped": "no IK mode/effector"})
            continue
        _set_prop(arm, sw, ikv)
        variants = [("pole_as_is", None)]
        ps = limb.get("pole_switch")
        if ps:
            variants += [(f"{ps['property']}={v}", v) for v in ps["values"]]
        for label, pval in variants:
            if pval is not None:
                _set_prop(arm, {"bone": ps["bone"], "property": ps["property"]}, pval)
            root = arm.pose.bones[limb["upper"]].head.copy()
            knee0 = arm.matrix_world @ arm.pose.bones[limb["lower"]].head.copy()
            flex0 = _flex(arm, limb["upper"], limb["lower"])
            pull = (root - eff.head) * 0.15

            def measure():
                knee1 = arm.matrix_world @ arm.pose.bones[limb["lower"]].head
                d = knee1 - knee0
                return {"knee_displacement": list(d), "flexion_before_deg": flex0,
                        "flexion_after_deg": _flex(arm, limb["upper"], limb["lower"]),
                        "dot_points_toward": d.normalized().dot(Vector(limb["points_toward"])) if d.length > 1e-9 else 0.0}

            eff_before = eff.head.copy()

            def measure_with_effector():
                r = measure()
                r["effector_displacement"] = list(eff.head - eff_before)
                r["requested_effector_displacement"] = list(pull)
                return r

            entry = {"limb_id": limb["limb_id"], "variant": label,
                     **_with_armature_translation(eff, pull, measure_with_effector)}
            pole = arm.pose.bones.get(limb.get("ik_pole") or "")
            if pole is not None:
                base = _snapshot(arm, [limb["lower"]])
                entry["pole_moves_knee"] = bool(_with_armature_translation(
                    pole, Vector((offset * 3, 0, 0)), lambda: _moved(base, _snapshot(arm, [limb["lower"]]))))
            out["ik_bend_tests"].append(entry)
            if pval is not None:
                _set_prop(arm, {"bone": ps["bone"], "property": ps["property"]},
                          type(arm.pose.bones[ps["bone"]][ps["property"]])(ps["original"]))
        _set_prop(arm, sw, original_props[limb["switch_index"]])

    # 4. Control sweep: which structural bones does each control move (in each switch mode)?
    for mode in ("min", "max"):
        for sw in switches:
            _set_prop(arm, sw, sw[mode])
        base = _snapshot(arm, structural)
        sweep = {}
        for name in args.get("controls", []):
            pb = arm.pose.bones.get(name)
            if pb is None:
                continue
            r = {}
            r["rotate"] = _with_basis(pb, pb.matrix_basis @ Matrix.Rotation(angle, 4, "X") @ Matrix.Rotation(angle, 4, "Z"),
                                      lambda: _moved(base, _snapshot(arm, structural)))
            r["scale"] = _with_basis(pb, pb.matrix_basis @ Matrix.Diagonal((1.15, 1.15, 1.15, 1.0)),
                                     lambda: _moved(base, _snapshot(arm, structural)))
            r["translate"] = _with_armature_translation(pb, Vector((offset, offset, offset)),
                                                        lambda: _moved(base, _snapshot(arm, structural)))
            sweep[name] = r
        out["control_sweep"][mode] = sweep
    for sw, v in zip(switches, original_props):
        _set_prop(arm, sw, v)
    out["file"] = bpy.data.filepath
    out["is_dirty_after"] = bool(bpy.data.is_dirty)
    return out
