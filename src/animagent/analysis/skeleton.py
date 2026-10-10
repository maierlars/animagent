"""Stage 1: derive the clean structural skeleton from a raw extraction.

Strategies:
* `rigify_org`: Rigify-generated rigs keep a copy of the metarig as ORG bones. Their
  hierarchy is the cleanest anatomical skeleton. Structural parents come from the metarig
  (if present in the file), otherwise from the generated hierarchy, skipping mechanism bones.
* `deform`: any other rig; deform bones form the skeleton.

Each structural bone also lists the deform bones that lie on it (geometric containment),
which links skin weights to anatomy without relying on names.
"""

from __future__ import annotations

import re

import numpy as np

from animagent.analysis.geom import point_segment_distance, side_from_name, unit, v
from animagent.models.anatomy import Skeleton, StructuralBone
from animagent.models.rig import BoneData, RigExtraction

_MECH_PREFIX = re.compile(r"^(DEF|MCH|ORG|VIS|WGT)-")


def _strip_numeric_suffix(name: str) -> str:
    return re.sub(r"(\.\d{3})+$", "", name)


def choose_strategy(ex: RigExtraction) -> str:
    org = sum(b.name.startswith("ORG-") for b in ex.bones)
    if ex.rigify.is_rigify_generated and org >= 5:
        return "rigify_org"
    return "deform"


def _near(child: BoneData, parent: BoneData) -> bool:
    """True if `child` starts on or close to `parent` (a plausible anatomical attachment)."""
    dist, _ = point_segment_distance(v(child.head_world), v(parent.head_world), v(parent.tail_world))
    return dist <= 0.5 * max(parent.length, 1e-6)


def _would_cycle(parent_of: dict[str, str | None], child: str, parent: str) -> bool:
    cur: str | None = parent
    seen = set()
    while cur is not None and cur not in seen:
        if cur == child:
            return True
        seen.add(cur)
        cur = parent_of.get(cur)
    return False


def build_skeleton(ex: RigExtraction) -> Skeleton:
    strategy = choose_strategy(ex)
    bones = ex.bone_map()
    notes: list[str] = []
    if strategy == "rigify_org":
        struct = [b for b in ex.bones if b.name.startswith("ORG-")]
    else:
        struct = [b for b in ex.bones if b.use_deform]
        if not struct:
            struct = list(ex.bones)
            notes.append("No deform bones; using all bones as structural skeleton.")
    names = {b.name for b in struct}
    scale = max((b.length for b in struct), default=1.0)
    tol = 1e-3 * max(scale, 1e-6) + 1e-5

    parent_of: dict[str, str | None] = {}
    source: dict[str, str] = {}

    meta_parent: dict[str, str | None] = {}
    if strategy == "rigify_org" and ex.rigify.metarig_bones:
        meta_parent = {m.name: m.parent for m in ex.rigify.metarig_bones}
        notes.append(f"Structural parents taken from metarig {ex.rigify.metarig!r}.")

    def via_hierarchy(b: BoneData) -> str | None:
        cur = b.parent
        while cur is not None:
            if cur in names and cur != b.name:
                return cur
            # Mechanism / deform copies of a structural bone stand in for it (DEF-hand -> ORG-hand).
            if strategy == "rigify_org":
                base = _MECH_PREFIX.sub("", cur)
                for cand in (f"ORG-{base}", f"ORG-{_strip_numeric_suffix(base)}"):
                    if cand in names and cand != b.name and _near(b, bones[cand]):
                        return cand
            cur = bones[cur].parent if cur in bones else None
        return None

    for b in struct:
        base = b.name[4:] if b.name.startswith("ORG-") else b.name
        p: str | None = None
        if base in meta_parent:
            mp = meta_parent[base]
            if mp is not None and f"ORG-{mp}" in names:
                p, source[b.name] = f"ORG-{mp}", "metarig"
            elif mp is None:
                p, source[b.name] = None, "root"
        if b.name not in source:
            p = via_hierarchy(b)
            source[b.name] = "hierarchy" if p else "root"
        parent_of[b.name] = p

    # Geometric repair: a bone whose head coincides with another structural bone's tail is
    # treated as connected to it when the hierarchy gave no parent or a non-adjacent one.
    tails = {b.name: v(b.tail_world) for b in struct}
    for b in struct:
        if source[b.name] == "metarig":
            continue
        head = v(b.head_world)
        p = parent_of[b.name]
        if p is not None and np.linalg.norm(tails[p] - head) <= tol:
            continue
        cands = [n for n, t in tails.items() if n != b.name and np.linalg.norm(t - head) <= tol]
        cands = [c for c in cands if not _would_cycle(parent_of, b.name, c)]
        if len(cands) == 1 and (p is None or not np.allclose(v(bones[p].head_world), head, atol=tol)):
            parent_of[b.name] = cands[0]
            source[b.name] = "geometry"

    for n in sorted(names):
        if parent_of[n] is not None and _would_cycle(parent_of, n, parent_of[n]):  # type: ignore[arg-type]
            notes.append(f"Broke structural cycle at {n!r} (was parented to {parent_of[n]!r}).")
            parent_of[n], source[n] = None, "root"

    children: dict[str, list[str]] = {n: [] for n in names}
    for n, p in parent_of.items():
        if p is not None:
            children[p].append(n)

    # Map deform bones onto structural bones by geometric containment.
    deform_map: dict[str, list[str]] = {n: [] for n in names}
    seg = {b.name: (v(b.head_world), v(b.tail_world)) for b in struct}
    for d in ex.bones:
        if not d.use_deform:
            continue
        if d.name in names:
            deform_map[d.name].append(d.name)
            continue
        dh, dt = v(d.head_world), v(d.tail_world)
        mid, ddir = 0.5 * (dh + dt), unit(dt - dh)
        best, best_cost = None, np.inf
        for n, (a, bb) in seg.items():
            dist, _ = point_segment_distance(mid, a, bb)
            dh_dist, _ = point_segment_distance(dh, a, bb)
            align = abs(float(ddir @ unit(bb - a)))
            cost = dist + dh_dist + 0.1 * (1.0 - align) * float(np.linalg.norm(bb - a))
            if cost < best_cost:
                best, best_cost = n, cost
        if best is not None and best_cost <= 0.25 * max(float(np.linalg.norm(seg[best][1] - seg[best][0])), tol):
            deform_map[best].append(d.name)

    unmapped = [d.name for d in ex.bones if d.use_deform and not any(d.name in x for x in deform_map.values())]
    if unmapped:
        notes.append(f"{len(unmapped)} deform bones not mapped to a structural bone: {unmapped[:10]}")

    out = [
        StructuralBone(
            name=b.name,
            parent=parent_of[b.name],
            children=sorted(children[b.name]),
            side=side_from_name(b.name),
            head=b.head_world,
            tail=b.tail_world,
            length=b.length,
            parent_source=source[b.name],  # type: ignore[arg-type]
            deform_bones=sorted(deform_map[b.name]),
        )
        for b in struct
    ]
    roots = sorted(n for n, p in parent_of.items() if p is None)
    return Skeleton(strategy=strategy, bones=out, roots=roots, notes=notes)  # type: ignore[arg-type]
