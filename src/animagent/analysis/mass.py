"""Approximate per-segment mass distribution from skin-weight summaries.

Skin weights are NOT masses. We use skin-weighted *surface area* per deform bone as an
artistic proxy: relative_mass(segment) = sum(weight * vertex_area over its deform bones) /
sum over all deform bones. Good enough for "the head is heavy, the hand is light" guidance.
"""

from __future__ import annotations

import numpy as np

from animagent.analysis.geom import rnd, v
from animagent.models.anatomy import MassEstimate, Segment, Skeleton
from animagent.models.rig import RigExtraction


def deform_bone_stats(ex: RigExtraction) -> dict[str, tuple[float, np.ndarray, float]]:
    """deform bone -> (area_weighted, centroid, rms_radius) accumulated over all deformed meshes."""
    acc: dict[str, list[tuple[float, np.ndarray, float]]] = {}
    for m in ex.meshes:
        if not m.visible:
            continue
        for g in m.weights.groups:
            if g.is_deform_bone and g.area_weighted > 0 and g.centroid_world is not None:
                acc.setdefault(g.name, []).append((g.area_weighted, v(g.centroid_world), g.rms_radius or 0.0))
    out = {}
    for name, items in acc.items():
        w = sum(i[0] for i in items)
        cen = sum(i[0] * i[1] for i in items) / w
        rms = float(np.sqrt(sum(i[0] * (i[2] ** 2 + float(np.sum((i[1] - cen) ** 2))) for i in items) / w))
        out[name] = (w, cen, rms)
    return out


def estimate_segment_masses(ex: RigExtraction, skel: Skeleton, segments: list[Segment]) -> dict[str, MassEstimate]:
    stats = deform_bone_stats(ex)
    total = sum(s[0] for s in stats.values())
    if total <= 0:
        return {}
    bm = skel.bone_map()
    out = {}
    for seg in segments:
        deform = {d for b in seg.source_bones if b in bm for d in bm[b].deform_bones}
        items = [stats[d] for d in deform if d in stats]
        if not items:
            continue
        w = sum(i[0] for i in items)
        cen = sum(i[0] * i[1] for i in items) / w
        rms = float(np.sqrt(sum(i[0] * (i[2] ** 2 + float(np.sum((i[1] - cen) ** 2))) for i in items) / w))
        out[seg.id] = MassEstimate(relative_mass=round(w / total, 4), center_of_mass=rnd(cen), rms_radius=round(rms, 4))
    return out
