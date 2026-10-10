"""Stage 3: rig-specific control mapping.

Maps anatomy (segments / limbs) to animator controls by analysing the rig's dependency
graph: parenting, constraint targets, IK constraints and the drivers that switch
constraint influences (IK/FK switches). Names and bone collections are recorded only as
supporting evidence.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections import deque
from dataclasses import dataclass

import numpy as np

from animagent.models.anatomy import AnatomyProfile, Evidence, LimbKind, Skeleton
from animagent.models.controls import (
    ControlMapping,
    ControlRef,
    DependencyLink,
    IKChainInfo,
    LimbControls,
    SegmentControls,
    SwitchProperty,
)
from animagent.models.rig import BoneData, ConstraintData, DriverData, RigExtraction

_MECH = re.compile(r"^(ORG|DEF|MCH|VIS|WGT)[-_]")
# Constraints that restrict rather than transfer motion are not dependency edges.
_NON_TRANSFER = {"LIMIT_DISTANCE", "LIMIT_LOCATION", "LIMIT_ROTATION", "LIMIT_SCALE", "MAINTAIN_VOLUME"}
_TWEAK_HINT = re.compile(r"tweak", re.I)
_PROP_PATH = re.compile(r'^pose\.bones\["(?P<bone>.+?)"\]\["(?P<prop>.+?)"\]$')
_CON_PATH = re.compile(r'^pose\.bones\["(?P<bone>.+?)"\]\.constraints\["(?P<con>.+?)"\]\.(?P<attr>\w+)$')
MAX_DEPTH = 7


def is_rigify(ex: RigExtraction) -> bool:
    return ex.rigify.is_rigify_generated or sum(b.name.startswith("ORG-") for b in ex.bones) >= 5


def control_bones(ex: RigExtraction) -> list[str]:
    rigify = is_rigify(ex)
    out = []
    for b in ex.bones:
        if rigify:
            if _MECH.match(b.name):
                continue
            if b.custom_shape is not None or not b.hidden:
                out.append(b.name)
        elif b.custom_shape is not None:
            out.append(b.name)
    if not out and not rigify:
        out = [b.name for b in ex.bones if not b.hidden]
    return out


# --------------------------------------------------------------------------------------
# Driver evaluation (deterministic, sandboxed)
# --------------------------------------------------------------------------------------

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_FUNCS = {
    "min": min,
    "max": max,
    "abs": abs,
    "pow": pow,
    "sqrt": math.sqrt,
    "clamp": lambda x, a=0.0, b=1.0: min(max(x, a), b),
}


def _eval_expr(expr: str, env: dict[str, float]) -> float | None:
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = ev(n.left), ev(n.comparators[0])
            ops = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b, ast.Eq: a == b, ast.NotEq: a != b}
            return float(ops.get(type(n.ops[0]), 0.0))
        raise ValueError(f"unsupported expression node {type(n).__name__}")

    try:
        return float(ev(ast.parse(expr, mode="eval")))
    except Exception:
        return None


def _driver_raw(d: DriverData, env: dict[str, float]) -> float | None:
    vals = [env.get(var.name) for var in d.variables]
    if d.type == "SCRIPTED":
        return _eval_expr(d.expression, {k: x for k, x in env.items() if x is not None})
    if any(x is None for x in vals) or not vals:
        return None
    nums = [float(x) for x in vals if x is not None]
    return {"SUM": sum(nums), "AVERAGE": sum(nums) / len(nums), "MIN": min(nums), "MAX": max(nums)}.get(d.type)


def evaluate_driver(d: DriverData, env: dict[str, float]) -> float | None:
    """Driver value incl. its F-curve mapping (keyframes, generator modifiers).

    Returns None for anything not modelled (e.g. non-generator modifiers, restricted ranges)."""
    x = _driver_raw(d, env)
    if x is None:
        return None
    y = x
    if d.keyframes:
        kx = [k[0] for k in d.keyframes]
        ky = [k[1] for k in d.keyframes]
        y = float(np.interp(x, kx, ky))
    for m in d.modifiers:
        if m.mute or not m.active:
            continue
        if m.type != "GENERATOR" or m.mode != "POLYNOMIAL" or m.use_restricted_range:
            return None
        poly = sum(c * x**i for i, c in enumerate(m.coefficients))
        val = (y + poly) if m.use_additive else poly
        y = (y + (val - y) * m.influence) if m.use_influence else val
    return y


@dataclass
class _PropRef:
    bone: str
    prop: str


def _prop_refs(d: DriverData) -> dict[str, _PropRef]:
    out = {}
    for var in d.variables:
        if var.type == "SINGLE_PROP" and var.targets:
            m = _PROP_PATH.match(var.targets[0].data_path or "")
            if m:
                out[var.name] = _PropRef(m.group("bone"), m.group("prop"))
    return out


# --------------------------------------------------------------------------------------
# Dependency graph
# --------------------------------------------------------------------------------------


class RigGraph:
    def __init__(self, ex: RigExtraction, deform_of: dict[str, list[str]] | None = None) -> None:
        self.ex = ex
        self.deform_of = deform_of or {}
        self.bones = ex.bone_map()
        self.arm = ex.armature.name
        self.controls = set(control_bones(ex))
        self.rigify = is_rigify(ex)
        self.drivers_by_con: dict[tuple[str, str], DriverData] = {}
        for d in ex.drivers:
            m = _CON_PATH.match(d.data_path)
            if m and m.group("attr") == "influence":
                self.drivers_by_con[(m.group("bone"), m.group("con"))] = d
        self.ik_constraints: list[tuple[BoneData, ConstraintData]] = [
            (b, c) for b in ex.bones for c in b.constraints if c.type == "IK"
        ]
        # Motion dependencies created by drivers reading bone transforms (e.g. beak_master).
        self.driver_deps: dict[str, list[tuple[str, str]]] = {}
        for d in ex.drivers:
            m = re.match(r'^pose\.bones\["(.+?)"\]', d.data_path)
            if not m:
                continue
            for var in d.variables:
                if var.type in {"TRANSFORMS", "ROTATION_DIFF", "LOC_DIFF"}:
                    for t in var.targets:
                        if t.id_name == self.arm and t.bone_target in self.bones:
                            self.driver_deps.setdefault(m.group(1), []).append((t.bone_target, f"driver:{var.type}"))

    def is_structural(self, name: str) -> bool:
        if self.rigify:
            return name.startswith(("ORG-", "DEF-"))
        return self.bones[name].use_deform and name not in self.controls

    def sources(self, name: str) -> list[tuple[str, str]]:
        b = self.bones[name]
        out: list[tuple[str, str]] = []
        for c in b.constraints:
            if c.mute or c.type in _NON_TRANSFER:
                continue
            for t in c.targets:
                if t.object == self.arm and t.subtarget in self.bones:
                    via = (
                        "ik_pole"
                        if t.role == "pole"
                        else ("ik_target" if c.type == "IK" else f"constraint:{c.type}:{c.name}")
                    )
                    out.append((t.subtarget, via))
        # Bones inside an IK chain depend on that IK's target and pole.
        for owner, con in self.ik_constraints:
            if con.mute:
                continue
            chain = self.ik_chain(owner.name, con)
            if name in chain and name != owner.name:
                for t in con.targets:
                    if t.object == self.arm and t.subtarget in self.bones:
                        out.append((t.subtarget, "ik_pole" if t.role == "pole" else "ik_target"))
        out.extend(self.driver_deps.get(name, []))
        if b.parent:
            out.append((b.parent, "parent"))
        return out

    def ik_chain(self, owner: str, con: ConstraintData) -> list[str]:
        n = int(con.settings.get("chain_count", 0) or 0)
        chain, cur = [], owner
        while cur is not None and (n == 0 or len(chain) < n):
            chain.append(cur)
            cur = self.bones[cur].parent
        return chain

    def resolve(
        self, start: str, *, through_controls: bool = True, max_depth: int = MAX_DEPTH
    ) -> list[tuple[str, list[DependencyLink], bool]]:
        """BFS from `start` to the controls that move it.

        Returns (control, path, via_control). Expansion stops at other structural bones
        (inherited motion from upstream anatomy) and at `max_depth`."""
        found: dict[str, tuple[list[DependencyLink], bool]] = {}
        queue: deque[tuple[str, list[DependencyLink], bool]] = deque([(start, [], False)])
        seen = {start}
        while queue:
            node, path, via_ctrl = queue.popleft()
            if len(path) >= max_depth:
                continue
            for src, via in self.sources(node):
                if src in seen:
                    continue
                seen.add(src)
                link = DependencyLink(from_bone=node, to_bone=src, via=via)
                p = [*path, link]
                if src in self.controls:
                    found.setdefault(src, (p, via_ctrl))
                    if through_controls:
                        queue.append((src, p, True))
                elif not self.is_structural(src):
                    queue.append((src, p, via_ctrl))
        return [(k, p, vc) for k, (p, vc) in found.items()]

    def coincident(self, a: str, b: str, tol: float = 1e-4) -> bool:
        ba, bb = self.bones[a], self.bones[b]
        return bool(
            np.allclose(ba.head_world, bb.head_world, atol=tol) and np.allclose(ba.tail_world, bb.tail_world, atol=tol)
        )

    def control_ref(
        self,
        bone: str,
        role: str,
        conf: float,
        evidence: list[Evidence],
        path: list[DependencyLink] | None = None,
        segment: str | None = None,
    ) -> ControlRef:
        b = self.bones[bone]
        locks = []
        for ch, flags in (("loc", b.pose.lock_location), ("rot", b.pose.lock_rotation), ("scale", b.pose.lock_scale)):
            locks += [f"{ch}.{ax}" for ax, f in zip("xyz", flags, strict=False) if f]
        hints = [c for c in b.collections if re.search(r"tweak|ik|fk", c, re.I)]
        ev = list(evidence)
        if hints:
            ev.append(Evidence(kind="collection_hint", detail=f"bone collections {b.collections}", weight=0.2))
        return ControlRef(
            bone=bone,
            role=role,
            confidence=round(conf, 3),
            evidence=ev,
            path=path or [],  # type: ignore[arg-type]
            rotation_mode=b.pose.rotation_mode,
            locked_channels=locks,
            collections=b.collections,
            segment=segment,
        )

    def constraint_link(self, bone: str, target: str | None) -> list[DependencyLink]:
        """The constraint edge from `bone` to `target`, if one exists (for provenance)."""
        if target is None:
            return []
        for c in self.bones[bone].constraints:
            if any(t.subtarget == target for t in c.targets):
                return [DependencyLink(from_bone=bone, to_bone=target, via=f"constraint:{c.type}:{c.name}")]
        return []

    def observed_source(self, bone: str, tol: float = 1e-3) -> str | None:
        """Constraint target whose *current evaluated* pose matrix equals this bone's (read-only check)."""
        b = self.bones[bone]
        if b.pose.matrix_armature is None:
            return None
        mb = np.asarray(b.pose.matrix_armature)
        hits = []
        for c in b.constraints:
            if c.type != "COPY_TRANSFORMS" or not c.targets:
                continue
            t = self.bones.get(c.targets[0].subtarget)
            if t is not None and t.pose.matrix_armature is not None:
                if np.abs(np.asarray(t.pose.matrix_armature) - mb).max() <= tol:
                    hits.append(t.name)
        return hits[0] if len(hits) == 1 else None

    def effective_source(self, bone: str, env_prop: tuple[str, str], value: float) -> str | None:
        """Constraint target that wins on `bone` when switch property = value (last full-influence copy)."""
        b = self.bones[bone]
        winner = None
        for c in b.constraints:
            if c.mute or c.type not in {"COPY_TRANSFORMS", "COPY_ROTATION", "CHILD_OF", "ARMATURE", "COPY_LOCATION"}:
                continue
            infl = c.influence
            d = self.drivers_by_con.get((bone, c.name))
            if d is not None:
                refs = _prop_refs(d)
                env = {}
                for var, ref in refs.items():
                    if (ref.bone, ref.prop) == env_prop:
                        env[var] = value
                    else:
                        cur = self.bones.get(ref.bone)
                        pv = cur.custom_props.get(ref.prop) if cur else None
                        if pv is not None and isinstance(pv.value, (int, float)):
                            env[var] = float(pv.value)
                ev = evaluate_driver(d, env)
                if ev is None:
                    continue
                infl = ev
            if infl >= 0.999 and c.targets:
                winner = c.targets[0].subtarget
        return winner


# --------------------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------------------


def _switches_for(g: RigGraph, bones: list[str]) -> dict[tuple[str, str], list[str]]:
    """Switch properties whose drivers modulate constraint influences on `bones`."""
    out: dict[tuple[str, str], list[str]] = {}
    for (bone, con), d in g.drivers_by_con.items():
        if bone not in bones:
            continue
        for ref in _prop_refs(d).values():
            out.setdefault((ref.bone, ref.prop), []).append(f"{bone}:{con}")
    return out


def _prop_range(g: RigGraph, bone: str, prop: str) -> tuple[float, float, object]:
    cp = g.bones[bone].custom_props.get(prop) if bone in g.bones else None
    if cp is None:
        return 0.0, 1.0, None
    lo = cp.ui.get("min", 0.0)
    hi = cp.ui.get("max", 1.0)
    try:
        return float(lo), float(hi), cp.value
    except (TypeError, ValueError):
        return 0.0, 1.0, cp.value


def _ik_related(g: RigGraph) -> tuple[set[str], list[IKChainInfo]]:
    """All bones taking part in IK machinery, and a description of every IK constraint."""
    related: set[str] = set()
    chains = []
    for owner, con in g.ik_constraints:
        chain = g.ik_chain(owner.name, con)
        tgt = next((t.subtarget for t in con.targets if t.role == "target" and t.subtarget), None)
        pole = next((t.subtarget for t in con.targets if t.role == "pole" and t.subtarget), None)
        pctrl, _ = _resolve_one(g, pole)
        related.update(chain)
        for x in (tgt, pole, pctrl):
            if x:
                related.add(x)
        if tgt:
            related.update(k for k, _, _ in g.resolve(tgt, through_controls=False))
        pa = con.settings.get("pole_angle")
        chains.append(
            IKChainInfo(
                constraint_bone=owner.name,
                constraint_name=con.name,
                chain_count=len(chain),
                chain_bones=chain,
                target_bone=tgt,
                target_control=None,
                pole_bone=pole,
                pole_control=pctrl,
                pole_angle_deg=round(math.degrees(float(pa)), 3) if isinstance(pa, (int, float)) else None,
                influence=con.influence,
                muted=con.mute,
                use_tail=bool(con.settings.get("use_tail", True)),
                use_stretch=con.settings.get("use_stretch"),
            )
        )
    return related, chains


def _check_switch_against_pose(g: RigGraph, sw: SwitchProperty, struct: list[str], ik_related: set[str]) -> None:
    """Verify the driver-derived switch semantics against the *current evaluated pose*.

    Read-only functional check: at the property's current value, the structural bones must
    coincide with the sources the inferred mode predicts. On disagreement, observation wins."""
    try:
        cur = float(sw.current_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if sw.ik_value is None or sw.fk_value is None:
        return
    predicted_ik = abs(cur - sw.ik_value) < abs(cur - sw.fk_value)
    obs_ik = obs_fk = 0
    for b in struct:
        src = g.observed_source(b)
        if src is None:
            continue
        is_ik = src in ik_related or any(k in ik_related for k, _, _ in g.resolve(src, through_controls=False))
        obs_ik += is_ik
        obs_fk += not is_ik
    if obs_ik + obs_fk == 0:
        sw.evidence.append(Evidence(kind="pose_observation", detail="current pose cannot discriminate modes", weight=0))
        return
    observed_ik = obs_ik > obs_fk
    if observed_ik == predicted_ik:
        sw.evidence.append(
            Evidence(
                kind="pose_observation",
                detail=f"at {sw.property}={cur:g} the evaluated pose "
                f"matches the {'IK' if observed_ik else 'FK'} sources ({obs_ik} IK / {obs_fk} FK)",
                weight=0.9,
            )
        )
    else:
        sw.ik_value, sw.fk_value = sw.fk_value, sw.ik_value
        sw.evidence.append(
            Evidence(
                kind="pose_observation",
                detail=f"driver model predicted the opposite mode at "
                f"{sw.property}={cur:g}; evaluated pose follows {'IK' if observed_ik else 'FK'} "
                f"({obs_ik} IK / {obs_fk} FK). Semantics flipped to match observation.",
                weight=-0.5,
            )
        )


def _topmost(g: RigGraph, cands: list[str]) -> str | None:
    """Among candidate controls, the one that is a parent-ancestor of most others."""
    if not cands:
        return None

    def ancestors(n: str) -> set[str]:
        out, cur = set(), g.bones[n].parent
        while cur is not None:
            out.add(cur)
            cur = g.bones[cur].parent
        return out

    score = {c: sum(c in ancestors(o) for o in cands if o != c) for c in cands}
    return max(cands, key=lambda c: (score[c], -cands.index(c)))


def _resolve_one(g: RigGraph, bone: str | None) -> tuple[str | None, list[DependencyLink]]:
    if bone is None:
        return None, []
    if bone in g.controls:
        return bone, []
    found = g.resolve(bone, through_controls=False)
    if not found:
        return None, []
    top = _topmost(g, [k for k, _, _ in found])
    path = next(p for k, p, _ in found if k == top)
    return top, path


def _limb_controls(
    g: RigGraph, anatomy: AnatomyProfile, limb_id: str, ik_related: set[str], ik_chains: list[IKChainInfo]
) -> LimbControls:
    segs = anatomy.segment_map()
    limb = next(lb for lb in anatomy.limbs if lb.id == limb_id)
    main = [segs[s] for s in limb.segments]
    struct = [s.source_bones[0] for s in main]
    lc = LimbControls(limb_id=limb_id)
    effector_segment = next((e.segment for e in anatomy.end_effectors if e.id == limb.end_effector), limb.segments[-1])

    # Controls that move the limb's attachment (torso, root, ...) are upstream, not limb controls.
    attach_bone = segs[limb.attached_to].source_bones[0] if limb.attached_to in segs else None
    upstream = {k for k, _, _ in g.resolve(attach_bone)} if attach_bone else set()
    if attach_bone and attach_bone in g.controls:
        upstream.add(attach_bone)

    # IK chains whose bones coincide with this limb's structural bones.
    effector_extras: list[tuple[str, list[DependencyLink]]] = []
    for ch in ik_chains:
        if any(g.coincident(cb, sb) for cb in ch.chain_bones for sb in struct):
            ch = ch.model_copy(deep=True)
            if ch.target_bone:
                if ch.target_bone in g.controls:
                    ch.target_control = ch.target_bone
                else:
                    found = [(k, p) for k, p, _ in g.resolve(ch.target_bone) if k not in upstream]
                    ch.target_control = _topmost(g, [k for k, _ in found])
                    effector_extras += [(k, p) for k, p in found if k != ch.target_control]
            lc.ik_chains.append(ch)

    # IK/FK switch from drivers on the limb's structural bones.
    switches = _switches_for(g, struct)
    switch_key = None
    ik_value = fk_value = None
    for (pb, prop), driven in sorted(switches.items(), key=lambda kv: -len(kv[1])):
        lo, hi, cur = _prop_range(g, pb, prop)
        mode = {}
        for val in (lo, hi):
            srcs = [g.effective_source(b, (pb, prop), val) for b in struct]
            ik_hits = sum(
                1
                for s in srcs
                if s and (s in ik_related or any(k in ik_related for k, _, _ in g.resolve(s, through_controls=False)))
            )
            mode[val] = ik_hits
        if mode[lo] != mode[hi]:
            switch_key = (pb, prop)
            ik_value, fk_value = (lo, hi) if mode[lo] > mode[hi] else (hi, lo)
            lc.ik_fk_switch = SwitchProperty(
                bone=pb,
                property=prop,
                current_value=cur,
                min=lo,
                max=hi,
                fk_value=fk_value,
                ik_value=ik_value,
                driven=driven,
                evidence=[
                    Evidence(
                        kind="driver_evaluation",
                        detail=f"at {prop}={ik_value} {mode[ik_value]} limb bones follow "
                        f"IK machinery; at {prop}={fk_value} {mode[fk_value]}",
                        weight=0.9,
                    )
                ],
            )
            break
    if switch_key is not None and lc.ik_fk_switch is not None:
        _check_switch_against_pose(g, lc.ik_fk_switch, struct, ik_related)
        ik_value, fk_value = lc.ik_fk_switch.ik_value, lc.ik_fk_switch.fk_value
        pbone = g.bones[switch_key[0]]
        for prop in pbone.custom_props:
            if prop == switch_key[1]:
                continue
            lo, hi, cur = _prop_range(g, switch_key[0], prop)
            driven = [
                d.data_path
                for d in g.ex.drivers
                if any((r.bone, r.prop) == (switch_key[0], prop) for r in _prop_refs(d).values())
            ]
            lc.settings.append(
                SwitchProperty(bone=switch_key[0], property=prop, current_value=cur, min=lo, max=hi, driven=driven[:8])
            )
        lc.other_controls.append(
            g.control_ref(
                switch_key[0],
                "parent",
                0.7,
                [Evidence(kind="custom_props", detail="holds the IK/FK switch and limb settings", weight=0.6)],
            )
        )

    # FK chain and IK-mode controls per segment bone.
    for seg, sb in zip(main, struct, strict=True):
        fk_src = g.effective_source(sb, switch_key, fk_value) if switch_key and fk_value is not None else None
        ik_src = g.effective_source(sb, switch_key, ik_value) if switch_key and ik_value is not None else None
        if fk_src is None:
            # No switch: nearest coincident control that is not IK machinery.
            cands = [(k, p) for k, p, vc in g.resolve(sb) if not vc and k not in ik_related]
            co = [(k, p) for k, p in cands if g.coincident(k, sb)]
            if co:
                fk_src = co[0][0]
            elif cands:
                fk_src = cands[0][0]
        fk_ctrl, fk_path = _resolve_one(g, fk_src)
        fk_path = g.constraint_link(sb, fk_src) + fk_path
        if fk_ctrl:
            coin = g.coincident(fk_ctrl, sb)
            ev = [Evidence(kind="dependency", detail=f"{sb} follows {fk_src} in FK mode", weight=0.7)]
            if coin:
                ev.append(
                    Evidence(kind="geometry", detail="control coincides with structural bone at rest", weight=0.5)
                )
            lc.fk_chain.append(g.control_ref(fk_ctrl, "fk", 0.9 if coin else 0.6, ev, fk_path, seg.id))
        if ik_src is not None and ik_src != fk_src:
            ik_ctrl, ik_path = _resolve_one(g, ik_src)
            ik_path = g.constraint_link(sb, ik_src) + ik_path
            if ik_ctrl and ik_ctrl not in {c.bone for c in lc.ik_chain_controls}:
                is_chain = any(ik_ctrl in ch.chain_bones for ch in lc.ik_chains) or any(
                    g.bones[cb].parent == ik_ctrl for ch in lc.ik_chains for cb in ch.chain_bones
                )
                is_target = any(ik_ctrl == ch.target_control for ch in lc.ik_chains)
                if not is_target:
                    lc.ik_chain_controls.append(
                        g.control_ref(
                            ik_ctrl,
                            "ik_chain",
                            0.8 if is_chain else 0.6,
                            [Evidence(kind="dependency", detail=f"{sb} follows {ik_src} in IK mode", weight=0.7)],
                            ik_path,
                            seg.id,
                        )
                    )
    if lc.ik_chains:
        main_chain = next((ch for ch in lc.ik_chains if not ch.muted), lc.ik_chains[0])
        pole_chain = next((ch for ch in lc.ik_chains if ch.pole_control), None)
        if main_chain.target_control:
            lc.ik_effector = g.control_ref(
                main_chain.target_control,
                "ik_effector",
                0.9,
                [
                    Evidence(
                        kind="ik_constraint",
                        detail=f"{main_chain.constraint_bone}:{main_chain.constraint_name} target "
                        f"{main_chain.target_bone} resolves to this control",
                        weight=0.9,
                    )
                ],
                segment=effector_segment,
            )
        if pole_chain is not None and pole_chain.pole_control:
            state = "currently muted (pole mode switched off)" if pole_chain.muted else "active"
            lc.ik_pole = g.control_ref(
                pole_chain.pole_control,
                "ik_pole",
                0.9 if not pole_chain.muted else 0.8,
                [
                    Evidence(
                        kind="ik_constraint",
                        detail=f"pole of {pole_chain.constraint_bone}:{pole_chain.constraint_name}, {state}",
                        weight=0.9,
                    )
                ],
            )
        special = {x.bone for x in (lc.ik_effector, lc.ik_pole) if x}
        lc.ik_chain_controls = [c for c in lc.ik_chain_controls if c.bone not in special]
        seen_extra = {c.bone for c in lc.ik_chain_controls} | special
        for k, p in effector_extras:
            if k not in seen_extra:
                seen_extra.add(k)
                lc.ik_chain_controls.append(
                    g.control_ref(
                        k,
                        "ik_chain",
                        0.6,
                        [
                            Evidence(
                                kind="dependency",
                                detail="between the IK effector control and the IK target (e.g. foot roll)",
                                weight=0.6,
                            )
                        ],
                        p,
                        effector_segment,
                    )
                )
    # Tweak / other controls reachable from deform bones of the limb.
    known = {c.bone for c in [*lc.fk_chain, *lc.ik_chain_controls, *lc.other_controls]}
    known |= {x.bone for x in (lc.ik_effector, lc.ik_pole) if x}
    for seg in main:
        for sbn in seg.source_bones:
            for d in g.deform_of.get(sbn, []):
                for k, p, vc in g.resolve(d, through_controls=False):
                    if k not in known and not vc:
                        known.add(k)
                        lc.other_controls.append(
                            g.control_ref(
                                k,
                                "tweak",
                                0.6,
                                [
                                    Evidence(
                                        kind="dependency", detail=f"deform bone {d} follows it directly", weight=0.6
                                    )
                                ],
                                p,
                                seg.id,
                            )
                        )
    n_ok = len(lc.fk_chain) / max(len(main), 1)
    lc.confidence = round(min(limb.confidence, 0.5 + 0.4 * n_ok + (0.1 if lc.ik_effector else 0.0)), 3)
    lc.evidence = [
        Evidence(
            kind="coverage",
            detail=f"FK controls for {len(lc.fk_chain)}/{len(main)} segments, "
            f"{len(lc.ik_chains)} IK constraints, switch={'yes' if lc.ik_fk_switch else 'no'}",
            weight=n_ok,
        )
    ]
    return lc


def map_controls(ex: RigExtraction, skel: Skeleton, anatomy: AnatomyProfile) -> ControlMapping:
    g = RigGraph(ex, deform_of={b.name: [d for d in b.deform_bones if d != b.name] for b in skel.bones})
    ik_related, ik_chains = _ik_related(g)
    mapping = ControlMapping(rig_kind="rigify" if g.rigify else "generic", control_bones=sorted(g.controls))

    for limb in anatomy.limbs:
        if limb.kind in (LimbKind.ARM, LimbKind.LEG):
            mapping.limbs.append(_limb_controls(g, anatomy, limb.id, ik_related, ik_chains))

    # Per-segment candidate controls (all segments, including trunk, tail, digits, appendages).
    used: set[str] = set()
    for lc in mapping.limbs:
        for c in [*lc.fk_chain, *lc.ik_chain_controls, *lc.other_controls]:
            used.add(c.bone)
        used |= {x.bone for x in (lc.ik_effector, lc.ik_pole) if x}
    found_by_seg: dict[str, dict[str, tuple[list[DependencyLink], bool, str]]] = {}
    for seg in anatomy.segments:
        found: dict[str, tuple[list[DependencyLink], bool, str]] = {}
        for sb in seg.source_bones:
            for st in [sb] + g.deform_of.get(sb, []):
                for k, p, vc in g.resolve(st):
                    found.setdefault(k, (p, vc, st))
        found_by_seg[seg.id] = found
    for seg in anatomy.segments:
        found = found_by_seg[seg.id]
        upstream = set(found_by_seg.get(seg.parent_segment or "", {}))
        refs: list[ControlRef] = []
        for k, (p, vc, st) in found.items():
            coincident = any(g.coincident(k, sb) for sb in seg.source_bones)
            tweak_hint = any(_TWEAK_HINT.search(c) for c in g.bones[k].collections)
            via_deform_only = st not in seg.source_bones
            if coincident and not tweak_hint:
                role, conf, why = "fk", 0.85, "coincides with a structural bone of the segment"
            elif tweak_hint or (via_deform_only and not vc):
                role, conf, why = "tweak", 0.6, "drives deform bones directly / tweak collection"
            elif via_deform_only:
                role, conf, why = "deform_only", 0.4, "reaches only the segment's deform bones (e.g. via stretch)"
            elif k in upstream:
                role, conf, why = "parent", 0.4, "also moves the parent segment (inherited)"
            else:
                role, conf, why = "master", 0.65 if not vc else 0.55, "moves this segment but not its parent"
            refs.append(
                g.control_ref(
                    k,
                    role,
                    conf,
                    [
                        Evidence(
                            kind="dependency",
                            detail=f"{st} {'indirectly ' if vc else ''}follows {k} ({len(p)} links); {why}",
                            weight=conf,
                        )
                    ],
                    p,
                    seg.id,
                )
            )
        order = {"fk": 0, "master": 1, "tweak": 2, "parent": 3, "deform_only": 4}
        refs.sort(key=lambda r: (order.get(r.role, 4), len(r.path)))
        mapping.segments.append(SegmentControls(segment_id=seg.id, controls=refs[:16]))
        used |= set(found)

    # Root: control without control ancestors from which most controls descend.
    parents = {n: g.bones[n].parent for n in g.controls}

    def ctrl_ancestors(n: str) -> list[str]:
        out, cur = [], g.bones[n].parent
        while cur is not None:
            if cur in g.controls:
                out.append(cur)
            cur = g.bones[cur].parent
        return out

    roots = [n for n in g.controls if parents[n] is None]
    if roots:
        reach = {r: 0 for r in roots}
        for n in g.controls:
            for k, _, _ in g.resolve(n, max_depth=12):
                if k in reach:
                    reach[k] += 1
        root = max(roots, key=lambda r: reach[r])
        mapping.root_control = g.control_ref(
            root,
            "root",
            0.8 if reach[root] > len(g.controls) // 3 else 0.5,
            [Evidence(kind="dependency", detail=f"unparented; {reach[root]} controls depend on it", weight=0.8)],
        )
        used.add(root)

    # Torso / centre-of-gravity: nearest common control ancestor of pelvis and chest controls.
    pelvis_ctrls = [r.bone for s in mapping.segments if s.segment_id == "pelvis" for r in s.controls]
    chest_ctrls = [r.bone for s in mapping.segments if s.segment_id == "chest" for r in s.controls]
    if pelvis_ctrls and chest_ctrls:
        common = [
            c
            for c in pelvis_ctrls
            if c in chest_ctrls and c != (mapping.root_control.bone if mapping.root_control else None)
        ]
        common.sort(key=lambda c: len(ctrl_ancestors(c)), reverse=True)
        if common:
            mapping.body_controls.append(
                g.control_ref(
                    common[0],
                    "torso",
                    0.75,
                    [
                        Evidence(
                            kind="dependency", detail="deepest control that moves both pelvis and chest", weight=0.7
                        )
                    ],
                )
            )
    seg_refs = {sc.segment_id: sc.controls for sc in mapping.segments}
    for sid in ("pelvis", "spine", "chest", "neck", "head", "jaw", "tail"):
        refs = [r for r in seg_refs.get(sid, []) if r.role in ("fk", "master")]
        if refs:
            r = refs[0]
            mapping.body_controls.append(
                r.model_copy(
                    update={
                        "evidence": [
                            *r.evidence,
                            Evidence(
                                kind="ranking", detail=f"top-ranked {r.role} control of segment {sid}", weight=0.5
                            ),
                        ]
                    }
                )
            )

    # Global switch properties referenced by drivers but not attached to a limb.
    limb_switch_bones = {lc.ik_fk_switch.bone for lc in mapping.limbs if lc.ik_fk_switch}
    props: dict[tuple[str, str], list[str]] = {}
    for d in ex.drivers:
        for ref in _prop_refs(d).values():
            if ref.bone not in limb_switch_bones and ref.bone in g.controls:
                props.setdefault((ref.bone, ref.prop), []).append(d.data_path)
    for (b, p), paths in sorted(props.items()):
        lo, hi, cur = _prop_range(g, b, p)
        mapping.global_properties.append(
            SwitchProperty(bone=b, property=p, current_value=cur, min=lo, max=hi, driven=paths[:8])
        )

    # Prop mounts: bones that other objects (weapons, props) are attached to.
    for att in ex.attachments:
        for rel in att.relations:
            if rel.bone and rel.bone in g.bones:
                mapping.prop_mounts.append(
                    g.control_ref(
                        rel.bone,
                        "generic",
                        0.8,
                        [
                            Evidence(
                                kind="attachment",
                                detail=f"object {att.name!r} ({att.type}) attached via {rel.kind}"
                                f"{':' + rel.type if rel.type else ''}",
                                weight=0.8,
                            )
                        ],
                    )
                )
                used.add(rel.bone)

    mapping.unmapped_controls = sorted(c for c in g.controls if c not in used)
    if mapping.unmapped_controls:
        mapping.notes.append(f"{len(mapping.unmapped_controls)} controls were not linked to any anatomical segment.")
    return mapping
