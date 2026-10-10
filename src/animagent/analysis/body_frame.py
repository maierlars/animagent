"""Body coordinate system: up, character-left, forward, ground and height."""

from __future__ import annotations

import numpy as np

from animagent.analysis.geom import mirror_name, rnd, unit, v
from animagent.models.anatomy import BodyFrame, Evidence, Side, Skeleton, SymmetryInfo
from animagent.models.rig import RigExtraction

WORLD_UP = np.array([0.0, 0.0, 1.0])


def symmetric_pairs(skel: Skeleton) -> list[tuple[str, str]]:
    names = skel.bone_map()
    pairs = []
    for b in skel.bones:
        if b.side == Side.LEFT:
            m = mirror_name(b.name)
            if m and m in names and names[m].side == Side.RIGHT:
                pairs.append((b.name, m))
    return pairs


def mesh_bounds(ex: RigExtraction, skel: Skeleton) -> tuple[np.ndarray, np.ndarray, str]:
    mins, maxs = [], []
    for m in ex.meshes:
        if m.weights.bbox_min is not None and m.weights.bbox_max is not None and m.visible:
            mins.append(v(m.weights.bbox_min))
            maxs.append(v(m.weights.bbox_max))
    if mins:
        return np.min(mins, axis=0), np.max(maxs, axis=0), "deformed_meshes"
    pts = np.array([b.head for b in skel.bones] + [b.tail for b in skel.bones])
    return pts.min(axis=0), pts.max(axis=0), "skeleton"


def compute_body_frame(ex: RigExtraction, skel: Skeleton) -> tuple[BodyFrame, SymmetryInfo]:
    evidence: list[Evidence] = []
    bmap = skel.bone_map()
    up = WORLD_UP.copy()
    lo, hi, bounds_src = mesh_bounds(ex, skel)
    ground = float(lo @ up)
    height = float(hi @ up) - ground

    # Up: Blender is Z-up. Verify that the extremities of sided chains are the lowest points.
    pts_z = [min(b.head[2], b.tail[2]) for b in skel.bones]
    lowest = sorted(skel.bones, key=lambda b: min(b.head[2], b.tail[2]))[: max(2, len(skel.bones) // 20)]
    sided_low = sum(b.side != Side.CENTER for b in lowest) / max(len(lowest), 1)
    up_conf = 0.6 + 0.35 * sided_low
    evidence.append(Evidence(kind="convention", detail="Blender world is Z-up", weight=0.6))
    evidence.append(
        Evidence(
            kind="geometry",
            detail=f"{sided_low:.0%} of the lowest structural bones are sided (feet-like); "
            f"skeleton z-range {min(pts_z):.3f}..{max(max(b.head[2], b.tail[2]) for b in skel.bones):.3f}",
            weight=sided_low,
        )
    )

    # Character left from L/R pairs (names), verified by mirror geometry.
    pairs = symmetric_pairs(skel)
    unpaired = [b.name for b in skel.bones if b.side != Side.CENTER and mirror_name(b.name) not in bmap]
    sym = SymmetryInfo(pairs=len(pairs), unpaired_sided=unpaired)
    left_conf = 0.3
    if pairs:
        diffs = np.array([v(bmap[left].head) - v(bmap[right].head) for left, right in pairs])
        left = unit(diffs.mean(axis=0) - (diffs.mean(axis=0) @ up) * up)
        mid_x = float(np.mean([(v(bmap[a].head) + v(bmap[b].head)) @ left / 2 for a, b in pairs]))
        errs: list[tuple[float, str]] = []
        for a, b in pairs:
            e = 0.0
            for attr in ("head", "tail"):
                pa, pb = v(getattr(bmap[a], attr)), v(getattr(bmap[b], attr))
                mirrored = pa - 2 * ((pa @ left) - mid_x) * left
                e = max(e, float(np.linalg.norm(mirrored - pb)))
            errs.append((e, a))
        max_err, worst = max(errs)
        median_err = float(np.median([e for e, _ in errs]))
        tol = 0.02 * max(height, 1e-6)
        asym = sorted(n for e, n in errs if e > tol)
        sym.max_mirror_error = round(max_err, 5)
        sym.median_mirror_error = round(median_err, 5)
        sym.asymmetric_pairs = asym
        sym.mirror_plane_normal = rnd(left)
        agree = float(np.mean([(v(bmap[a].head) - v(bmap[b].head)) @ left > 0 for a, b in pairs]))
        left_conf = 0.5 + 0.4 * agree * (1.0 if median_err < tol else 0.6)
        evidence.append(
            Evidence(
                kind="symmetry",
                detail=f"{len(pairs)} L/R bone pairs; mean L-R offset {rnd(diffs.mean(axis=0), 3)}; "
                f"median mirror error {median_err:.4f}, max {max_err:.4f} ({worst}); "
                f"{len(asym)} pairs asymmetric at rest",
                weight=agree,
            )
        )
    else:
        left = np.array([1.0, 0.0, 0.0])
        evidence.append(Evidence(kind="convention", detail="No L/R pairs; assuming +X is character left", weight=0.2))

    forward = unit(np.cross(left, up))
    evidence.append(
        Evidence(
            kind="derived", detail=f"forward = left x up = {rnd(forward, 3)} (Rigify convention is -Y)", weight=0.5
        )
    )
    center = [b for b in skel.bones if b.side == Side.CENTER] or skel.bones
    c = np.mean([v(b.head) for b in center], axis=0)
    origin = c - ((c @ up) - ground) * up
    conf = float(min(up_conf, left_conf))
    frame = BodyFrame(
        origin=rnd(origin),
        up=rnd(up),
        forward=rnd(forward),
        left=rnd(left),
        height=round(height, 4),
        ground_height=round(ground, 4),
        confidence=round(conf, 3),
        evidence=evidence + [Evidence(kind="bounds", detail=f"height from {bounds_src}: {height:.3f}", weight=0.5)],
    )
    return frame, sym
