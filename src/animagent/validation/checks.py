"""Independent validation checks (A-H) for a rig profile.

Every check records its *method* so a reviewer can tell evidence strength apart:

* ``independent_measurement``: measured in Blender (or from raw extraction data) without using the
  analyzer's interpretation beyond selecting what to measure.
* ``experimental``: perturbation of controls in an isolated background copy, measuring evaluated motion.
* ``raw_data``: arithmetic on raw extracted Blender values (matrices, heads/tails, drivers).
* ``consistency``: internal consistency of the analyzer's own outputs (weakest evidence).
* ``heuristic``: compares against a prior/assumption; informational.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from animagent.analysis.controls import _CON_PATH, _prop_refs, control_bones, evaluate_driver
from animagent.analysis.geom import angle_deg, mirror_name, unit, v
from animagent.models.anatomy import Side
from animagent.models.profile import RigProfile

_MECH = re.compile(r"^(ORG|DEF|MCH|VIS|WGT)[-_]")


class CheckList:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        cid: str,
        category: str,
        title: str,
        method: str,
        status: str,
        measured: Any = None,
        tolerance: Any = None,
        expected: Any = None,
        note: str = "",
    ) -> None:
        self.items.append(
            {
                "id": cid,
                "category": category,
                "title": title,
                "method": method,
                "status": status,
                "measured": measured,
                "expected": expected,
                "tolerance": tolerance,
                "note": note,
            }
        )

    def check(
        self,
        cid: str,
        category: str,
        title: str,
        method: str,
        ok: bool,
        measured: Any = None,
        tolerance: Any = None,
        expected: Any = None,
        note: str = "",
        fail_status: str = "fail",
    ) -> bool:
        self.add(cid, category, title, method, "pass" if ok else fail_status, measured, tolerance, expected, note)
        return ok


def _r(x: Any, nd: int = 5) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(float(y), nd) for y in x]
    return x


# --------------------------------------------------------------------------------------------
# Joint table
# --------------------------------------------------------------------------------------------


def joint_table(p: RigProfile) -> list[dict[str, Any]]:
    a = p.anatomy
    sk = p.skeleton.bone_map()
    ex = p.extraction.bone_map()
    segs = a.segment_map()
    arm_world = np.asarray(p.extraction.armature.matrix_world)
    amb_by_subject: dict[str, list[str]] = {}
    for amb in a.ambiguities:
        for s in amb.subject:
            amb_by_subject.setdefault(s, []).append(amb.id)
    fk_of: dict[str, str] = {}
    for lc in p.controls.limbs:
        for c in lc.fk_chain:
            if c.segment:
                fk_of[c.segment] = c.bone
    rows = []
    for j in a.joints:
        child = segs[j.child_segment]
        bone = next(
            (b for b in child.source_bones if b in sk and np.allclose(sk[b].head, j.position, atol=1e-5)),
            child.source_bones[0],
        )
        sb = sk.get(bone)
        parent_bone = sb.parent if sb else None
        pos = np.asarray(j.position)
        local_in_parent = None
        if parent_bone and parent_bone in ex:
            m = np.asarray(ex[parent_bone].matrix_world_rest)
            local_in_parent = (np.linalg.solve(m, np.append(pos, 1.0))[:3]).tolist()
        arm_space = (np.linalg.solve(arm_world, np.append(pos, 1.0))[:3]).tolist()
        row: dict[str, Any] = {
            "id": j.id,
            "anatomical_name": j.kind.value,
            "side": j.side.value,
            "blender_bones": {
                "structural": bone,
                "structural_parent": parent_bone,
                "deform": sb.deform_bones if sb else [],
                "fk_control_of_child_segment": fk_of.get(j.child_segment),
            },
            "world_position": _r(j.position),
            "armature_space_position": _r(arm_space),
            "position_in_parent_bone_rest_space": _r(local_in_parent) if local_in_parent else None,
            "parent_segment": j.parent_segment,
            "child_segment": j.child_segment,
            "dof_hint": j.dof_hint,
            "confidence": j.confidence,
            "evidence": [e.model_dump() for e in j.evidence],
            "ambiguities": sorted(
                {x for b in child.source_bones for x in amb_by_subject.get(b, [])}
                | ({f"bend:{j.id}"} & {x.id for x in a.ambiguities})
            ),
        }
        if j.bend:
            row["bend"] = {
                "hinge_axis_world": j.bend.hinge_axis,
                "joint_points_toward_world": j.bend.joint_points_toward,
                "hinge_axis_local": j.bend.hinge_axis_local,
                "local_axis_bone": j.bend.local_axis_bone,
                "local_axis_alignment": j.bend.hinge_axis_local_alignment,
                "points_toward_body_frame": j.bend.body_relative,
                "rest_flexion_deg": j.bend.rest_flexion_deg,
                "source": j.bend.source,
                "matches_humanoid_prior": j.bend.matches_prior,
                "confidence": j.bend.confidence,
                "evidence": [e.model_dump() for e in j.bend.evidence],
            }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def run_checks(p: RigProfile, probe: dict[str, Any], experiment: dict[str, Any] | None) -> CheckList:
    cl = CheckList()
    a, ex = p.anatomy, p.extraction
    f = a.body_frame
    H = f.height
    up, fwd, left = v(f.up), v(f.forward), v(f.left)
    bm = ex.bone_map()
    sk = p.skeleton.bone_map()
    segs = a.segment_map()
    joints = a.joint_map()
    lm = {x["id"]: x for x in probe.get("landmarks", [])}

    # ---------------- A. coordinate frames ----------------
    basis = np.column_stack([-left, fwd, up])  # (right, forward, up)
    ortho = float(np.abs(basis.T @ basis - np.eye(3)).max())
    cl.check("A1", "A", "Body frame orthonormal", "consistency", ortho < 1e-3, _r(ortho), 1e-3)
    det = float(np.linalg.det(basis))
    cl.check(
        "A2",
        "A",
        "Body frame (right, forward, up) is right-handed",
        "consistency",
        abs(det - 1) < 1e-3,
        _r(det),
        "det = +1 +- 1e-3",
        note="right = -left; forward = left x up by construction",
    )
    bf = probe["bone_frames"]
    cl.check(
        "A3",
        "A",
        "All bone rest matrices orthonormal and right-handed (measured in Blender)",
        "independent_measurement",
        bf["max_orthonormality_error"] < 1e-4 and abs(bf["min_det"] - 1) < 1e-4 and abs(bf["max_det"] - 1) < 1e-4,
        {k: _r(x, 8) for k, x in bf.items()},
        "ortho < 1e-4, det = 1 +- 1e-4",
    )
    cl.check(
        "A4",
        "A",
        "Bone Y axis points head->tail for all bones (Blender convention)",
        "independent_measurement",
        bf["max_y_axis_misalignment"] < 1e-5,
        _r(bf["max_y_axis_misalignment"], 9),
        1e-5,
    )
    cl.check(
        "A5",
        "A",
        "Armature object transform is a proper rotation (no mirroring)",
        "independent_measurement",
        bf["armature_matrix_det"] > 0,
        _r(bf["armature_matrix_det"]),
        "> 0",
    )
    # Forward vs. skin geometry, without using the forward axis: farthest skin point from the head
    # centroid (beak) and from the ankles (toes), projected to the ground plane.
    head = lm.get("effector:head")
    if head and head.get("vertex_count"):
        c = v(head["centroid"])
        far = max(
            (v(head["forward_most_point"]), v(head["backward_most_point"])), key=lambda q: float(np.linalg.norm(q - c))
        )
        d = unit((far - c) - ((far - c) @ up) * up)
        cl.check(
            "A6",
            "A",
            "Forward axis agrees with the beak/face direction (farthest head-region skin point)",
            "independent_measurement",
            float(d @ fwd) > 0.7,
            _r(float(d @ fwd)),
            "> 0.7",
            note=f"farthest point {_r(list(far), 3)} from head-region centroid {_r(list(c), 3)}",
        )
    dots = []
    for e in a.end_effectors:
        if e.kind != "foot":
            continue
        m = lm.get(f"effector:{e.id}")
        if m and m.get("vertex_count"):
            ank = v(e.position)
            pts = [v(m["forward_most_point"]), v(m["backward_most_point"])]
            far = max(pts, key=lambda q: float(np.linalg.norm((q - ank) - ((q - ank) @ up) * up)))
            d = unit((far - ank) - ((far - ank) @ up) * up)
            dots.append(float(d @ fwd))
    if dots:
        cl.check(
            "A7",
            "A",
            "Forward axis agrees with toe direction of the foot skin",
            "independent_measurement",
            min(dots) > 0.7,
            _r(dots),
            "> 0.7 per foot",
        )
    g = probe.get("mesh_ground_z")
    cl.check(
        "A8",
        "A",
        "Ground height equals the lowest skin vertex",
        "independent_measurement",
        abs(f.ground_height - g) < 1e-3,
        {"frame_ground": f.ground_height, "mesh_min_z": _r(g)},
        1e-3,
    )
    lr = [b for b in sk.values() if b.side != Side.CENTER]
    dead = 0.01 * H
    offs = {b.name: float((v(b.head) - v(f.origin)) @ left) for b in lr}
    wrong = sorted(n for n, s_ in offs.items() if abs(s_) > dead and (sk[n].side == Side.LEFT) != (s_ > 0))
    near = sorted((n, _r(s_, 4)) for n, s_ in offs.items() if abs(s_) <= dead)
    near_wrong = [x for x in near if (sk[x[0]].side == Side.LEFT) != (x[1] > 0)]
    cl.check(
        "A9",
        "A",
        ".L/.R names lie on the character's left/right side (left = +X here)",
        "raw_data",
        not wrong,
        {"sided_bones": len(lr), "wrong_side": wrong, "near_midline_wrong_sign": near_wrong},
        f"dead zone +-{dead:.3f} m (1% height) around the mid-plane",
        note="Blender/Rigify convention: character faces -Y, its left is +X. Bones within the dead zone are "
        "listed; they sit on the midline (rig placement quirk) rather than on the wrong side.",
    )

    # ---------------- B. connectivity ----------------
    conn = probe["connectivity"]
    tol = 1e-4 * H
    for lb in a.limbs:
        if lb.kind.value in ("appendage",):
            continue
        bones = [segs[s].source_bones[0] for s in lb.segments]
        gaps = []
        for b in bones[1:]:
            if b in conn:
                gaps.append([conn[b][0], b, _r(conn[b][1], 7)])
        bad = [x for x in gaps if x[2] > tol and x[0] in bones]
        cl.check(
            f"B1:{lb.id}",
            "B",
            f"Chain {lb.id} is continuous (parent tail == child head, measured)",
            "independent_measurement",
            not bad,
            gaps,
            f"<= {tol:.2e} m",
            note="clavicle->upper_arm and hip attachment are allowed to be disconnected" if not bad else "",
        )
    trunk = [s for s in ("pelvis", "spine", "chest", "neck", "head") if s in segs]
    tb = [b for s in trunk for b in segs[s].source_bones]
    gaps = [[conn[b][0], b, _r(conn[b][1], 7)] for b in tb[1:] if b in conn]
    struct_gaps = [x for x in gaps if x[2] > tol]
    geo_gaps = []
    for x, y in zip(tb, tb[1:], strict=False):
        geo_gaps.append([x, y, _r(float(np.linalg.norm(v(sk[x].tail) - v(sk[y].head))), 7)])
    cl.check(
        "B2",
        "B",
        "Trunk chain pelvis->head is geometrically continuous",
        "raw_data",
        all(gg[2] <= tol for gg in geo_gaps),
        geo_gaps,
        f"<= {tol:.2e} m",
        note="Rigify parents ORG spine bones to tweak controls, so hierarchy gaps are reported separately: "
        + str(struct_gaps[:4]),
    )
    cov = []
    for sbn, sbone in sk.items():
        if sbone.deform_bones and sbn not in sbone.deform_bones:
            total = sum(bm[d].length for d in sbone.deform_bones)
            cov.append((sbn, _r(total / max(sbone.length, 1e-9), 4)))
    bad = [c for c in cov if abs(c[1] - 1) > 0.02]
    cl.check(
        "B3",
        "B",
        "Deform bones mapped to each structural bone cover its length (sum within 2%)",
        "raw_data",
        not bad,
        {"checked": len(cov), "outside_tolerance": bad[:20]},
        "ratio 1 +- 0.02",
        fail_status="warn",
        note="segments with additional deform bones (e.g. heel/palm helpers) may differ",
    )

    # ---------------- C. symmetry ----------------
    o = v(f.origin)
    pair_err = []
    for j in a.joints:
        if j.side != Side.LEFT:
            continue
        m = joints.get(mirror_name(j.id) or "") if mirror_name(j.id) else None
        if m is None:
            m = joints.get(j.id[:-2] + ".R") if j.id.endswith(".L") else None
        if m is None:
            continue
        pl = v(j.position)
        mirrored = pl - 2 * float((pl - o) @ left) * left
        pair_err.append((j.id, float(np.linalg.norm(mirrored - v(m.position)))))
    worst = sorted(pair_err, key=lambda x: -x[1])
    lim = 0.01 * H
    cl.check(
        "C1",
        "C",
        "Left/right joint pairs mirror across the body mid-plane",
        "raw_data",
        all(e <= lim for _, e in pair_err),
        {"pairs": len(pair_err), "worst": [(k, _r(e, 4)) for k, e in worst[:8]]},
        f"<= {lim:.4f} m (1% height)",
        fail_status="warn",
        note="asymmetric rest poses (e.g. finger curls) show up here",
    )
    bad_side = [
        (j.id, _r(float((v(j.position) - o) @ left), 4))
        for j in a.joints
        if j.side != Side.CENTER
        and abs(float((v(j.position) - o) @ left)) > 0.01 * H
        and (j.side == Side.LEFT) != (float((v(j.position) - o) @ left) > 0)
    ]
    cl.check(
        "C2",
        "C",
        "Every sided joint lies on its own side of the mid-plane (1% height dead zone)",
        "raw_data",
        not bad_side,
        bad_side,
    )
    for kind in ("arm", "leg"):
        lens = {lb.side.value: lb.chain_length for lb in a.limbs if lb.kind.value == kind}
        if {"L", "R"} <= set(lens):
            ratio = lens["L"] / lens["R"]
            cl.check(
                f"C3:{kind}",
                "C",
                f"{kind} upper+lower length L/R ratio",
                "raw_data",
                abs(ratio - 1) < 0.01,
                _r(ratio, 5),
                "1 +- 0.01",
            )
    cl.add(
        "C4",
        "C",
        "Symmetry assumption",
        "heuristic",
        "info",
        note="Left axis = mean L-R offset of mirrored bone NAME pairs; requires .L/.R naming. "
        f"Median mirror error {a.symmetry.median_mirror_error}, asymmetric pairs: {a.symmetry.asymmetric_pairs[:12]}",
    )

    # ---------------- D. bending ----------------
    exp_hinge = {h["limb_id"]: h for h in (experiment or {}).get("hinge_tests", [])}
    exp_ik = {}
    for b in (experiment or {}).get("ik_bend_tests", []):
        exp_ik.setdefault(b["limb_id"], []).append(b)
    for lb in a.limbs:
        if lb.kind.value not in ("arm", "leg"):
            continue
        mid_kind = "elbow" if lb.kind.value == "arm" else "knee"
        j = next((joints[x] for x in lb.joints if joints[x].kind.value == mid_kind), None)
        if j is None or j.bend is None:
            continue
        up_seg, low_seg = j.parent_segment, j.child_segment
        A_, B_ = v(sk[segs[up_seg].source_bones[0]].head), v(sk[segs[low_seg].source_bones[0]].head)
        end_seg_idx = lb.segments.index(low_seg) + 1
        C_ = (
            v(sk[segs[lb.segments[end_seg_idx]].source_bones[0]].head)
            if end_seg_idx < len(lb.segments)
            else v(sk[segs[low_seg].source_bones[0]].tail)
        )
        line = C_ - A_
        off = B_ - (A_ + float((B_ - A_) @ line) / float(line @ line) * line)
        cos = float(unit(off) @ v(j.bend.joint_points_toward))
        cl.check(
            f"D1:{j.id}",
            "D",
            f"{j.id}: bend direction recomputed from raw rest heads",
            "raw_data",
            cos > 0.99,
            {
                "cos": _r(cos),
                "rest_offset_m": _r(float(np.linalg.norm(off)), 5),
                "rest_flexion_deg": j.bend.rest_flexion_deg,
            },
            "> 0.99",
        )
        lc = next((x for x in p.controls.limbs if x.limb_id == lb.id), None)
        if lc and lc.ik_pole and lc.ik_pole.bone in bm:
            pole = v(bm[lc.ik_pole.bone].head_world)
            pp = pole - (A_ + float((pole - A_) @ line) / float(line @ line) * line)
            pc = float(unit(pp) @ v(j.bend.joint_points_toward))
            muted = all(ch.muted for ch in lc.ik_chains if ch.pole_control)
            cl.check(
                f"D2:{j.id}",
                "D",
                f"{j.id}: IK pole control lies on the inferred bend side (rest)",
                "raw_data",
                pc > 0.7,
                {"cos": _r(pc), "pole": lc.ik_pole.bone, "pole_constraint_muted_now": muted},
                "> 0.7",
            )
        h = exp_hinge.get(lb.id)
        if h and "plus" in h:
            pl, mi = h["plus"], h["minus"]
            ok = (
                pl["flexion_after_deg"] > pl["flexion_before_deg"]
                and mi["flexion_after_deg"] < mi["flexion_before_deg"]
                and pl["end_displacement_dot_points_toward"] < -0.7
            )
            cl.check(
                f"D3:{j.id}",
                "D",
                f"{j.id}: +10 deg about inferred hinge axis (FK control {h['control']}) "
                "increases flexion; distal end moves away from the apex side",
                "experimental",
                ok,
                {"plus": {k: _r(x, 3) for k, x in pl.items()}, "minus": {k: _r(x, 3) for k, x in mi.items()}},
                "flexion +10/-10 deg; end dot apex < -0.7",
            )
        for t in exp_ik.get(lb.id, []):
            if "dot_points_toward" not in t:
                continue
            req, got = v(t["requested_effector_displacement"]), v(t["effector_displacement"])
            cl.check(
                f"D4:{j.id}:{t['variant']}",
                "D",
                f"{j.id}: pulling the IK effector toward the limb root makes the "
                f"joint move to the inferred side ({t['variant']})",
                "experimental",
                t["dot_points_toward"] > 0.7 and t["flexion_after_deg"] > t["flexion_before_deg"],
                {
                    "cos": _r(t["dot_points_toward"]),
                    "flexion_before": _r(t["flexion_before_deg"], 2),
                    "flexion_after": _r(t["flexion_after_deg"], 2),
                    "effector_moved_as_requested": bool(np.linalg.norm(req - got) < 1e-4),
                    "pole_moves_joint": t.get("pole_moves_knee"),
                },
                "cos > 0.7",
            )
        cl.check(
            f"D5:{j.id}",
            "D",
            f"{j.id}: hinge axis aligned with a local bone axis",
            "raw_data",
            (j.bend.hinge_axis_local_alignment or 0) > 0.95,
            {"axis": j.bend.hinge_axis_local, "alignment": j.bend.hinge_axis_local_alignment},
            "> 0.95",
            fail_status="warn",
        )
        cl.add(
            f"D6:{j.id}",
            "D",
            f"{j.id}: matches humanoid prior ({'elbow back' if mid_kind == 'elbow' else 'knee forward'})",
            "heuristic",
            "info",
            j.bend.matches_prior,
            note="a mismatch is not an error (e.g. reversed avian ankle); only flagged as ambiguity",
        )

    # ---------------- E. landmarks vs skin ----------------
    def lm_check(cid: str, key: str, title: str, cond: Any, tol_txt: str, extra: dict[str, Any] | None = None) -> None:
        m = lm.get(key)
        if not m or not m.get("vertex_count"):
            cl.add(cid, "E", title, "independent_measurement", "warn", note="no skin vertices dominated by region")
            return
        ok, meas = cond(m)
        cl.check(cid, "E", title, "independent_measurement", ok, {**meas, **(extra or {})}, tol_txt)

    for e in a.end_effectors:
        key = f"effector:{e.id}"
        if e.kind in ("hand", "foot"):
            lm_check(
                f"E1:{e.id}",
                key,
                f"{e.id} landmark (joint centre) lies inside its skin region",
                lambda m: (
                    m["distance_outside_bbox"] <= 0.01 * H,
                    {
                        "distance_outside_bbox": _r(m["distance_outside_bbox"]),
                        "nearest_vertex": _r(m["nearest_vertex_distance"]),
                        "vertices": m["vertex_count"],
                    },
                ),
                f"outside bbox <= {0.01 * H:.3f} m",
            )
        if e.kind == "head":
            lm_check(
                "E1:head",
                key,
                "Head landmark lies inside the head-region skin",
                lambda m, e=e: (
                    m["distance_outside_bbox"] <= 0.01 * H,
                    {
                        "distance_outside_bbox": _r(m["distance_outside_bbox"]),
                        "head_region_centroid": _r(m["centroid"], 4),
                        "landmark": _r(e.position, 4),
                    },
                ),
                f"<= {0.01 * H:.3f} m",
            )
    for c in a.contacts:
        key = f"contact:{c.id}"
        if c.kind == "heel":
            lm_check(
                f"E2:{c.id}",
                key,
                f"{c.id}: heel contact near the lowest/back-most foot skin",
                lambda m, c=c: (
                    abs(c.position[2] - m["lowest_point"][2]) <= 0.03 * H
                    and float((v(c.position) - v(m["backward_most_point"])) @ fwd) <= 0.06 * H,
                    {
                        "heel_z": _r(c.position[2], 4),
                        "lowest_skin_z": _r(m["lowest_point"][2], 4),
                        "behind_by_m": _r(float((v(c.position) - v(m["backward_most_point"])) @ fwd), 4),
                    },
                ),
                "dz <= 3% H; within 6% H of back-most point",
            )
        if c.kind == "toe_tip" and not c.id.startswith(("leg.", "arm.")):
            lm_check(
                f"E3:{c.id}",
                key,
                f"{c.id}: toe tip near the forward-most foot skin",
                lambda m, c=c: (
                    float((v(m["forward_most_point"]) - v(c.position)) @ fwd) <= 0.05 * H,
                    {"skin_ahead_of_tip_m": _r(float((v(m["forward_most_point"]) - v(c.position)) @ fwd), 4)},
                ),
                "skin extends <= 5% H beyond the tip",
            )
        if c.kind == "ball":
            lm_check(
                f"E4:{c.id}",
                key,
                f"{c.id}: ball of foot close to the ground",
                lambda m, c=c: (
                    c.position[2] - f.ground_height <= 0.05 * H,
                    {"height_above_ground": _r(c.position[2] - f.ground_height, 4)},
                ),
                "<= 5% H",
            )
        if c.kind == "palm":
            lm_check(
                f"E5:{c.id}",
                key,
                f"{c.id}: palm point inside hand skin region",
                lambda m: (
                    m["distance_outside_bbox"] <= 0.005 * H,
                    {
                        "distance_outside_bbox": _r(m["distance_outside_bbox"]),
                        "nearest_vertex": _r(m["nearest_vertex_distance"]),
                    },
                ),
                "<= 0.5% H",
            )
        if c.kind == "head_top":
            lm_check(
                "E6:head_top",
                key,
                "head_top vs highest head-region skin",
                lambda m, c=c: (
                    True,
                    {
                        "head_top_z": _r(c.position[2], 4),
                        "skin_top_z": _r(m["bbox_max"][2], 4),
                        "gap_m": _r(m["bbox_max"][2] - c.position[2], 4),
                    },
                ),
                "info only",
            )

    digit_dirs = []
    for lb in a.limbs:
        for d in lb.digits:
            sd = segs[d]
            dvec = unit(v(sk[sd.source_bones[-1]].tail) - v(sk[sd.source_bones[0]].head))
            digit_dirs.append(
                {
                    "digit": d,
                    "bones": sd.source_bones,
                    "direction_body": {
                        "forward": _r(float(dvec @ fwd), 3),
                        "left": _r(float(dvec @ left), 3),
                        "up": _r(float(dvec @ up), 3),
                    },
                }
            )
    back = [x["digit"] for x in digit_dirs if x["digit"].startswith("leg") and x["direction_body"]["forward"] < -0.3]
    cl.add(
        "E7",
        "E",
        "Digit chain directions (toes pointing backward indicate a zygodactyl/avian foot)",
        "raw_data",
        "info",
        {"backward_pointing_toes": back, "digits": digit_dirs},
        note="toe tips are only compared to the forward-most skin for the main toe; per-digit tips are reported here",
    )

    # ---------------- F. separation of anatomy vs rig machinery ----------------
    controls = set(control_bones(ex))
    src = {b for s in a.segments for b in s.source_bones}
    wrong = sorted(b for b in src if b not in sk)
    cl.check("F1", "F", "Anatomy references only structural-skeleton bones", "consistency", not wrong, wrong[:10])
    cats: dict[str, int] = {}
    for b in src:
        m = _MECH.match(b)
        cats[m.group(1) if m else ("CONTROL" if b in controls else "OTHER")] = (
            cats.get(m.group(1) if m else ("CONTROL" if b in controls else "OTHER"), 0) + 1
        )
    cl.check(
        "F2",
        "F",
        "Anatomical segments are built from ORG bones only (no DEF/MCH/controls)",
        "raw_data",
        set(cats) == {"ORG"} if p.skeleton.strategy == "rigify_org" else True,
        cats,
    )
    shaped = sorted(b for b in src if bm[b].custom_shape)
    cl.check(
        "F3", "F", "No anatomical bone is an animator control (no custom shape)", "raw_data", not shaped, shaped[:10]
    )
    mapped_ctrls = set()
    for lc in p.controls.limbs:
        for c in [*lc.fk_chain, *lc.ik_chain_controls, *lc.other_controls]:
            mapped_ctrls.add(c.bone)
        mapped_ctrls |= {x.bone for x in (lc.ik_effector, lc.ik_pole) if x}
    for sc in p.controls.segments:
        mapped_ctrls |= {c.bone for c in sc.controls}
    nonctrl = sorted(b for b in mapped_ctrls if b not in controls)
    cl.check("F4", "F", "Control mapping references only control bones", "raw_data", not nonctrl, nonctrl[:10])
    deform = {b.name for b in ex.bones if b.use_deform}
    mapped_def = [d for b in sk.values() for d in b.deform_bones]
    dup = sorted({d for d in mapped_def if mapped_def.count(d) > 1})
    unm = sorted(deform - set(mapped_def))
    cl.check(
        "F5",
        "F",
        "Every deform bone maps to exactly one structural bone",
        "raw_data",
        not dup and not unm,
        {"deform": len(deform), "duplicates": dup, "unmapped": unm[:20]},
    )
    cl.check(
        "F6",
        "F",
        "Every structural bone is assigned to a segment",
        "consistency",
        not a.unassigned_bones,
        a.unassigned_bones[:20],
    )

    # ---------------- G. control coverage ----------------
    cov = control_provenance(p, experiment)
    n_exp = sum(1 for c in cov if c["experimental"]["moves_structural"])
    n_conf = sum(1 for c in cov if c["mapping_confirmed_experimentally"] is True)
    n_contra = [c["control"] for c in cov if c["mapping_confirmed_experimentally"] is False]
    cl.add(
        "G1",
        "G",
        "Control coverage summary",
        "experimental" if experiment else "raw_data",
        "info",
        {
            "controls": len(cov),
            "mapped": sum(1 for c in cov if c["mapped_as"]),
            "unmapped": [c["control"] for c in cov if not c["mapped_as"]],
            "move_structural_bones_experimentally": n_exp,
            "mapping_confirmed_experimentally": n_conf,
            "mapping_contradicted_experimentally": n_contra,
            "only_move_deform_bones": [
                c["control"]
                for c in cov
                if not c["experimental"]["moves_structural"] and c["experimental"]["moves_deform"]
            ],
            "no_measured_effect": [
                c["control"]
                for c in cov
                if not c["experimental"]["moves_structural"] and not c["experimental"]["moves_deform"]
            ],
        },
    )
    for lc in p.controls.limbs:
        claims = [(c.bone, c.segment) for c in lc.fk_chain] + (
            [(lc.ik_effector.bone, lc.ik_effector.segment)] if lc.ik_effector else []
        )
        bad = [
            (b, s)
            for b, s in claims
            if next((c for c in cov if c["control"] == b), {}).get("mapping_confirmed_experimentally") is False
        ]
        cl.check(
            f"G2:{lc.limb_id}",
            "G",
            f"{lc.limb_id}: FK chain and IK effector move their claimed segment (evaluated motion)",
            "experimental",
            experiment is not None and not bad,
            {"claims": claims, "contradicted": bad},
            note="" if experiment else "no experiment available",
        )

    # ---------------- H. IK/FK switch ----------------
    drv = {d["data_path"]: d for d in probe.get("drivers", [])}
    ex_drv = {d.data_path: d for d in ex.drivers}
    sw_tests = {t["limb_id"]: t for t in (experiment or {}).get("switch_tests", [])}
    for lc in p.controls.limbs:
        sw = lc.ik_fk_switch
        if sw is None:
            continue
        paths = (
            [
                pth
                for pth in drv
                if any((r.bone, r.prop) == (sw.bone, sw.property) for r in _prop_refs(ex_drv[pth]).values())
            ]
            if drv
            else []
        )
        diffs = []
        for pth in paths:
            for x, val in drv[pth]["fcurve_mapping"].items():
                ours = evaluate_driver(ex_drv[pth], {vn.name: float(x) for vn in ex_drv[pth].variables})
                diffs.append(abs((ours if ours is not None else np.nan) - val))
        maxdiff = float(np.nanmax(diffs)) if diffs else float("nan")
        cl.check(
            f"H1:{lc.limb_id}",
            "H",
            f"{lc.limb_id}: analyzer's driver evaluation equals Blender's F-curve evaluation",
            "independent_measurement",
            bool(diffs) and maxdiff < 1e-6,
            {
                "drivers": len(paths),
                "max_abs_diff": _r(maxdiff, 9),
                "blender_mapping_example": drv[paths[0]]["fcurve_mapping"] if paths else None,
                "expression": drv[paths[0]]["expression"] if paths else None,
                "type": drv[paths[0]]["type"] if paths else None,
                "modifiers": drv[paths[0]]["modifiers"] if paths else None,
            },
            "< 1e-6",
        )
        cur = [drv[pth].get("current_influence_evaluated") for pth in paths]
        cl.add(
            f"H2:{lc.limb_id}",
            "H",
            f"{lc.limb_id}: current switch value and evaluated IK-copy influences",
            "independent_measurement",
            "info",
            {"property": f"{sw.bone}[{sw.property}]", "value": sw.current_value, "evaluated_influences": cur},
        )
        t = sw_tests.get(lc.limb_id)
        if t:
            ik = t["by_value"].get(str(float(sw.ik_value)))
            fk = t["by_value"].get(str(float(sw.fk_value)))
            ok = bool(
                ik
                and fk
                and ik.get("effector_motion")
                and not ik.get("fk_motion")
                and fk.get("fk_motion")
                and not fk.get("effector_motion")
            )
            cl.check(
                f"H3:{lc.limb_id}",
                "H",
                f"{lc.limb_id}: {sw.property}={sw.ik_value:g} is IK and "
                f"{sw.property}={sw.fk_value:g} is FK (effector vs FK-control motion)",
                "experimental",
                ok,
                {
                    k: {
                        "effector_moves_limb": bool(x.get("effector_motion")),
                        "fk_control_moves_limb": bool(x.get("fk_motion")),
                    }
                    for k, x in t["by_value"].items()
                },
            )
    return cl


# --------------------------------------------------------------------------------------------
# Control provenance (G)
# --------------------------------------------------------------------------------------------


def control_provenance(p: RigProfile, experiment: dict[str, Any] | None) -> list[dict[str, Any]]:
    ex = p.extraction
    bm = ex.bone_map()
    sk = p.skeleton.bone_map()
    segs = p.anatomy.segment_map()
    deform = {b.name for b in ex.bones if b.use_deform}
    claims: dict[str, list[dict[str, Any]]] = {}

    def claim(bone: str, role: str, segment: str | None, src: str, evidence: list[Any], path: list[Any]) -> None:
        kinds = sorted({e.kind for e in evidence})
        via = sorted({lk.via.split(":")[0] for lk in path})
        structural = any(
            k in kinds for k in ("dependency", "ik_constraint", "driver_evaluation", "custom_props")
        ) or any(x in via for x in ("constraint", "ik_target", "ik_pole", "driver"))
        name_only = not structural and ("collection_hint" in kinds or "name_hint" in kinds)
        claims.setdefault(bone, []).append(
            {
                "source": src,
                "role": role,
                "segment": segment,
                "evidence_level": "name_or_collection"
                if name_only
                else (
                    "constraint_or_driver"
                    if any(x in via for x in ("constraint", "ik_target", "ik_pole", "driver"))
                    or any(k in kinds for k in ("ik_constraint", "driver_evaluation"))
                    else "hierarchy"
                ),
                "path": [f"{lk.from_bone} -[{lk.via}]-> {lk.to_bone}" for lk in path][:8],
            }
        )

    for lc in p.controls.limbs:
        for c in lc.fk_chain:
            claim(c.bone, "fk", c.segment, f"limb:{lc.limb_id}", c.evidence, c.path)
        for c in lc.ik_chain_controls:
            claim(c.bone, "ik_chain", c.segment, f"limb:{lc.limb_id}", c.evidence, c.path)
        for c in lc.other_controls:
            claim(c.bone, c.role, c.segment, f"limb:{lc.limb_id}", c.evidence, c.path)
        for c, role in ((lc.ik_effector, "ik_effector"), (lc.ik_pole, "ik_pole")):
            if c:
                claim(c.bone, role, c.segment, f"limb:{lc.limb_id}", c.evidence, c.path)
    for sc in p.controls.segments:
        for c in sc.controls:
            claim(c.bone, c.role, sc.segment_id, f"segment:{sc.segment_id}", c.evidence, c.path)
    if p.controls.root_control:
        c = p.controls.root_control
        claim(c.bone, "root", None, "root", c.evidence, c.path)
    for c in p.controls.body_controls:
        claim(c.bone, c.role, c.segment, "body", c.evidence, c.path)
    for c in p.controls.prop_mounts:
        claim(c.bone, "prop_mount", None, "attachment", c.evidence, c.path)

    sweep = (experiment or {}).get("control_sweep", {})
    out = []
    for name in control_bones(ex):
        moved: dict[str, set[str]] = {"structural": set(), "deform": set()}
        modes = {}
        for mode, res in sweep.items():
            r = res.get(name, {})
            ms = set(r.get("rotate", {})) | set(r.get("translate", {})) | set(r.get("scale", {}))
            modes[mode] = sorted(ms & set(sk))[:40]
            moved["structural"] |= ms & set(sk)
            moved["deform"] |= ms & deform
        cl = claims.get(name, [])
        confirmed: bool | None = None
        if experiment is not None:
            primary = [c for c in cl if c["role"] in ("fk", "ik_effector", "master", "ik_chain") and c["segment"]]
            if primary:
                confirmed = all(
                    any(b in moved["structural"] for b in segs[c["segment"]].source_bones)
                    for c in primary
                    if c["segment"] in segs
                )
        out.append(
            {
                "control": name,
                "collections": bm[name].collections,
                "mapped_as": [{k: c[k] for k in ("source", "role", "segment", "evidence_level")} for c in cl],
                "structural_paths": cl[0]["path"] if cl else [],
                "experimental": {
                    "performed": experiment is not None,
                    "moves_structural": sorted(moved["structural"])[:60],
                    "moves_deform_count": len(moved["deform"]),
                    "moves_deform": bool(moved["deform"]),
                    "structural_by_switch_mode": modes,
                },
                "mapping_confirmed_experimentally": confirmed,
            }
        )
    return out


def drivers_report(p: RigProfile, probe: dict[str, Any]) -> list[dict[str, Any]]:
    """Every driver touching the limbs' switch properties, with Blender's evaluated values."""
    out = []
    ex_drv = {d.data_path: d for d in p.extraction.drivers}
    for d in probe.get("drivers", []):
        m = _CON_PATH.match(d["data_path"])
        ours = (
            {
                x: evaluate_driver(
                    ex_drv[d["data_path"]], {vn.name: float(x) for vn in ex_drv[d["data_path"]].variables}
                )
                for x in d.get("fcurve_mapping", {})
            }
            if d["data_path"] in ex_drv
            else {}
        )
        out.append(
            {
                **d,
                "bone": m.group("bone") if m else None,
                "constraint": m.group("con") if m else None,
                "analyzer_evaluation": ours,
            }
        )
    return out


def frame_report(p: RigProfile) -> dict[str, Any]:
    f = p.anatomy.body_frame
    left = v(f.left)
    return {
        "convention": "world space, metres; Blender Z-up; character left = +X, forward = -Y for this rig",
        "origin": f.origin,
        "up": f.up,
        "forward": f.forward,
        "left": f.left,
        "right": _r(list(-left)),
        "height": f.height,
        "ground_height": f.ground_height,
        "confidence": f.confidence,
        "evidence": [e.model_dump() for e in f.evidence],
        "armature_matrix_world": p.extraction.armature.matrix_world,
        "angle_forward_vs_minus_Y_deg": _r(angle_deg(v(f.forward), np.array([0, -1.0, 0])), 3),
    }
