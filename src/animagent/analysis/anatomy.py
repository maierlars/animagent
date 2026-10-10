"""Stage 2: deterministic anatomical interpretation of a structural skeleton.

Works purely on geometry + topology of the structural skeleton (`models.anatomy.Skeleton`).
Bone names are only used for L/R side detection and as weak, explicitly recorded hints.
Anything that cannot be decided with confidence becomes an `Ambiguity` instead of a guess.

Algorithm outline:
1. Lateral branches = sided subtrees attached to the centre line. Branches whose subtree
   reaches the ground are legs; long, multi-bone branches above the legs are arms;
   everything else is an appendage (feathers, ears, accessories ...).
2. Pelvis = lowest common ancestor (LCA) of leg attachments; chest = LCA of arm attachments;
   spine = path between them. Neck/head continue the centre chain above the chest until a
   terminal or branching bone (the head). Backward-pointing centre chains below are tails.
3. Limb chains are split at the longest consecutive bone pair (upper/lower segment);
   a preceding bone is a clavicle/hip, following bones hand/foot/toe; hand/toe subtrees
   are digits.
4. Joints, bending directions, contacts, end effectors and mass shares are derived.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from animagent.analysis.bending import hinge_bend
from animagent.analysis.body_frame import compute_body_frame
from animagent.analysis.geom import mirror_name, rnd, unit, v
from animagent.analysis.mass import estimate_segment_masses
from animagent.models.anatomy import (
    Ambiguity,
    AnatomyProfile,
    BodyFrame,
    ContactPoint,
    EndEffector,
    Evidence,
    Joint,
    JointKind,
    Limb,
    LimbKind,
    Segment,
    SegmentKind,
    Side,
    Skeleton,
    StructuralBone,
)
from animagent.models.rig import RigExtraction

GROUND_FRACTION = 0.08  # subtree within this fraction of body height from ground "reaches ground"
ARM_MIN_FRACTION = 0.2  # minimum chain length (fraction of height) for an arm
APPENDAGE_MIN_FRACTION = 0.04  # smaller appendages are listed but not raised as ambiguities


def _label(name: str) -> str:
    """Human-readable label for ids: 'ORG-f_index.01.L' -> 'f_index'."""
    base = re.sub(r"^(ORG|DEF|MCH)-", "", name)
    base = re.sub(r"([._\- ](L|R|l|r|Left|Right|left|right))?(\.\d+)*$", "", base)
    return base or name


@dataclass
class _Ctx:
    ex: RigExtraction
    skel: Skeleton
    frame: BodyFrame
    bm: dict[str, StructuralBone]
    up: np.ndarray
    fwd: np.ndarray
    left: np.ndarray
    segments: list[Segment] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    limbs: list[Limb] = field(default_factory=list)
    effectors: list[EndEffector] = field(default_factory=list)
    contacts: list[ContactPoint] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    assigned: dict[str, str] = field(default_factory=dict)  # bone -> segment id

    @property
    def height(self) -> float:
        return max(self.frame.height, 1e-6)

    def axes(self) -> dict[str, np.ndarray]:
        return {"forward": self.fwd, "left": self.left, "up": self.up}

    def h(self, p: list[float] | np.ndarray) -> float:
        """Height of a point above ground."""
        return float(v(p) @ self.up) - self.frame.ground_height

    # --- tree helpers -------------------------------------------------------------------
    def subtree(self, name: str) -> list[str]:
        out, stack = [], [name]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(self.bm[n].children)
        return out

    def ancestors(self, name: str) -> list[str]:
        out, cur = [], self.bm[name].parent
        while cur is not None:
            out.append(cur)
            cur = self.bm[cur].parent
        return out

    def lca(self, names: list[str]) -> str | None:
        if not names:
            return None
        paths = [[n, *self.ancestors(n)] for n in names]
        common = set(paths[0]).intersection(*map(set, paths[1:]))
        for n in paths[0]:
            if n in common:
                return n
        return None

    def extent(self, name: str, origin: np.ndarray) -> float:
        return max(
            max(np.linalg.norm(v(self.bm[n].tail) - origin), np.linalg.norm(v(self.bm[n].head) - origin))
            for n in self.subtree(name)
        )

    def main_chain(self, start: str) -> list[str]:
        chain, cur = [start], start
        origin = v(self.bm[start].head)
        while True:
            kids = self.bm[cur].children
            if not kids or len(kids) >= 3:
                return chain
            tail = v(self.bm[cur].tail)
            tol = 1e-3 * max(self.bm[cur].length, 1e-6) + 1e-6

            def rank(k: str, tail: np.ndarray = tail, tol: float = tol) -> tuple[bool, float]:
                connected = bool(np.linalg.norm(v(self.bm[k].head) - tail) <= tol)
                return connected, self.extent(k, origin)

            nxt = max(kids, key=rank)
            chain.append(nxt)
            cur = nxt

    def chain_len(self, chain: list[str]) -> float:
        return float(sum(self.bm[n].length for n in chain))

    # --- builders -----------------------------------------------------------------------
    def unique_segment_id(self, base: str) -> str:
        ids = {s.id for s in self.segments}
        sid, k = base, 2
        while sid in ids:
            sid, k = f"{base}#{k}", k + 1
        return sid

    def unique_ambiguity_id(self, base: str) -> str:
        ids = {a.id for a in self.ambiguities}
        aid, k = base, 2
        while aid in ids:
            aid, k = f"{base}#{k}", k + 1
        return aid

    def add_segment(
        self,
        sid: str,
        kind: SegmentKind,
        bones: list[str],
        side: Side,
        conf: float,
        evidence: list[Evidence],
        parent: str | None = None,
    ) -> Segment:
        first, last = self.bm[bones[0]], self.bm[bones[-1]]
        seg = Segment(
            id=sid,
            kind=kind,
            side=side,
            source_bones=bones,
            head=first.head,
            tail=last.tail,
            length=round(self.chain_len(bones), 5),
            parent_segment=parent,
            confidence=round(conf, 3),
            evidence=evidence,
        )
        self.segments.append(seg)
        for b in bones:
            self.assigned.setdefault(b, sid)
        return seg

    def add_joint(
        self,
        jid: str,
        kind: JointKind,
        bone: str,
        parent_seg: str | None,
        child_seg: str,
        side: Side,
        conf: float,
        dof: str = "unknown",
        detail: str = "",
    ) -> Joint:
        j = Joint(
            id=jid,
            kind=kind,
            side=side,
            position=self.bm[bone].head,
            parent_segment=parent_seg,
            child_segment=child_seg,
            dof_hint=dof,  # type: ignore[arg-type]
            confidence=round(conf, 3),
            evidence=[Evidence(kind="structure", detail=detail or f"head of {bone}", weight=conf)],
        )
        self.joints.append(j)
        return j


def _side_tag(side: Side) -> str:
    return "" if side == Side.CENTER else f".{side.value}"


# ----------------------------------------------------------------------------------------
# Lateral branches: legs / arms / appendages
# ----------------------------------------------------------------------------------------


@dataclass
class _Branch:
    root: str
    side: Side
    chain: list[str]
    attach: str | None
    lowest_h: float
    length: float
    kind: str = "appendage"
    conf: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)


def _lateral_branches(c: _Ctx) -> list[_Branch]:
    out = []
    for b in c.skel.bones:
        if b.side == Side.CENTER:
            continue
        parent = c.bm[b.parent] if b.parent else None
        if parent is not None and parent.side != Side.CENTER:
            continue
        sub = c.subtree(b.name)
        lowest = min(min(c.h(c.bm[n].head), c.h(c.bm[n].tail)) for n in sub)
        chain = c.main_chain(b.name)
        out.append(_Branch(b.name, b.side, chain, b.parent, lowest, c.chain_len(chain)))
    return out


def _classify_branches(c: _Ctx, branches: list[_Branch]) -> None:
    H = c.height
    for br in branches:
        reaches = br.lowest_h <= GROUND_FRACTION * H
        if reaches and len(br.chain) >= 3:
            br.kind = "leg"
            br.conf = 0.85 if len(br.chain) >= 4 else 0.75
            br.evidence.append(
                Evidence(
                    kind="geometry",
                    detail=f"subtree reaches {br.lowest_h:.3f} above ground (<= {GROUND_FRACTION:.0%} of height), "
                    f"chain of {len(br.chain)} bones",
                    weight=0.8,
                )
            )
    leg_attach_h = [c.h(c.bm[br.root].head) for br in branches if br.kind == "leg"]
    min_leg_attach = max(leg_attach_h) if leg_attach_h else -np.inf
    for side in (Side.LEFT, Side.RIGHT):
        cands = [
            br
            for br in branches
            if br.side == side
            and br.kind == "appendage"
            and len(br.chain) >= 3
            and br.length >= ARM_MIN_FRACTION * H
            and c.h(c.bm[br.root].head) > min_leg_attach
        ]
        cands.sort(key=lambda br: -br.length)
        if not cands:
            continue
        best = cands[0]
        best.kind = "arm"
        best.conf = 0.85
        best.evidence.append(
            Evidence(
                kind="geometry",
                detail=f"longest lateral non-ground chain on side {side.value} "
                f"({len(best.chain)} bones, {best.length:.3f} = {best.length / H:.0%} of height), attached above legs",
                weight=0.8,
            )
        )
        for other in cands[1:]:
            if other.length >= 0.7 * best.length:
                best.conf = 0.6
                c.ambiguities.append(
                    Ambiguity(
                        id=f"arm_choice{_side_tag(side)}",
                        subject=[best.root, other.root],
                        question="Two arm-like lateral chains; which one is the primary arm?",
                        candidates=[best.root, other.root],
                        deterministic_best=best.root,
                        confidence=0.6,
                        evidence=[Evidence(kind="geometry", detail="chains have similar length", weight=0.5)],
                    )
                )


# ----------------------------------------------------------------------------------------
# Trunk: pelvis, spine, chest, neck, head, tail
# ----------------------------------------------------------------------------------------


def _center_children(c: _Ctx, name: str, exclude: set[str]) -> list[str]:
    return [k for k in c.bm[name].children if c.bm[k].side == Side.CENTER and k not in exclude]


def _trunk(c: _Ctx, branches: list[_Branch]) -> dict[str, object]:
    legs = [br for br in branches if br.kind == "leg"]
    arms = [br for br in branches if br.kind == "arm"]
    centers = [b.name for b in c.skel.bones if b.side == Side.CENTER]
    info: dict[str, object] = {}

    leg_attach = [br.attach for br in legs if br.attach]
    pelvis = c.lca(leg_attach) if leg_attach else None
    if pelvis is None:
        roots = [r for r in c.skel.roots if c.bm[r].side == Side.CENTER] or c.skel.roots
        pelvis = max(roots, key=lambda r: len(c.subtree(r))) if roots else None
        c.ambiguities.append(
            Ambiguity(
                id="pelvis",
                subject=[pelvis] if pelvis else [],
                question="No legs detected; which bone is the pelvis/hips?",
                candidates=centers[:12],
                deterministic_best=pelvis,
                confidence=0.3,
            )
        )
        pelvis_conf = 0.3
    else:
        pelvis_conf = 0.9 if len(leg_attach) >= 2 else 0.6
    if pelvis is None:
        return info

    arm_attach = [br.attach for br in arms if br.attach]
    chest = c.lca(arm_attach) if arm_attach else None
    chest_conf = 0.85 if len(arm_attach) >= 2 else 0.55
    if chest is not None and pelvis not in [chest, *c.ancestors(chest)]:
        c.notes.append(f"Arm attachment LCA {chest!r} is not above pelvis {pelvis!r}; ignoring.")
        chest = None
    if chest is None:
        # Fall back to the centre chain: chest = last spine bone before the neck can't be known.
        c.ambiguities.append(
            Ambiguity(
                id="chest",
                subject=[pelvis],
                question="No arms detected; where does the chest end and the neck begin?",
                candidates=centers[:12],
                confidence=0.2,
            )
        )

    spine_path: list[str] = []
    if chest is not None:
        spine_path = [chest]
        cur = chest
        while cur != pelvis:
            cur = c.bm[cur].parent  # type: ignore[assignment]
            spine_path.append(cur)
        spine_path.reverse()  # pelvis ... chest

    pelvis_ev = [
        Evidence(kind="structure", detail=f"lowest common ancestor of leg attachments {leg_attach}", weight=0.9)
    ]
    c.add_segment("pelvis", SegmentKind.PELVIS, [pelvis], Side.CENTER, pelvis_conf, pelvis_ev)
    prev_seg = "pelvis"
    if len(spine_path) > 2:
        c.add_segment(
            "spine",
            SegmentKind.SPINE,
            spine_path[1:-1],
            Side.CENTER,
            min(pelvis_conf, chest_conf),
            [Evidence(kind="structure", detail="bones between pelvis and chest", weight=0.8)],
            parent="pelvis",
        )
        for i, bn in enumerate(spine_path[1:-1]):
            c.add_joint(
                f"spine.{i}", JointKind.SPINE, bn, prev_seg if i == 0 else "spine", "spine", Side.CENTER, 0.8, "ball"
            )
        prev_seg = "spine"
    if chest is not None and chest != pelvis:
        c.add_segment(
            "chest",
            SegmentKind.CHEST,
            [chest],
            Side.CENTER,
            chest_conf,
            [Evidence(kind="structure", detail=f"lowest common ancestor of arm attachments {arm_attach}", weight=0.85)],
            parent=prev_seg,
        )
        c.add_joint("spine.chest", JointKind.SPINE, chest, prev_seg, "chest", Side.CENTER, chest_conf, "ball")
        prev_seg = "chest"
    elif chest == pelvis:
        c.notes.append("Arms and legs attach to the same centre bone; chest == pelvis.")

    # Neck / head: continue the centre chain above the chest.
    top = chest or (spine_path[-1] if spine_path else pelvis)
    exclude = set(spine_path)
    chain: list[str] = []
    cur = top
    while True:
        kids = _center_children(c, cur, exclude)
        if chain or cur == chest:
            # Only children starting near the current tail continue the chain; bones rooted at the
            # head's base (e.g. a face rig) are sub-structures, not continuations.
            reach = 0.25 * max(c.bm[cur].length, 1e-6)
            cont = [k for k in kids if np.linalg.norm(v(c.bm[k].head) - v(c.bm[cur].tail)) <= reach]
            if chain and not cont:
                break
            kids = cont or kids
        if chest is None and cur == pelvis:
            # pick the upward child
            kids = [k for k in kids if (v(c.bm[k].tail) - v(c.bm[k].head)) @ c.up > 0] or kids
        if not kids:
            break
        if len(kids) >= 2 and chain:
            break  # branching: cur is the head
        nxt = max(kids, key=lambda k: len(c.subtree(k)))
        chain.append(nxt)
        cur = nxt
        if len(kids) >= 2:
            # branching directly above the chest: take the largest subtree as neck/head
            continue
    head = chain[-1] if chain else None
    neck = chain[:-1]
    if head is not None:
        if neck:
            c.add_segment(
                "neck",
                SegmentKind.NECK,
                neck,
                Side.CENTER,
                0.8,
                [Evidence(kind="structure", detail="centre chain between chest and head", weight=0.8)],
                parent=prev_seg,
            )
            c.add_joint("neck_base", JointKind.NECK_BASE, neck[0], prev_seg, "neck", Side.CENTER, 0.8, "ball")
            for i, bn in enumerate(neck[1:]):
                c.add_joint(f"neck.{i + 1}", JointKind.SPINE, bn, "neck", "neck", Side.CENTER, 0.75, "ball")
            prev_seg = "neck"
        else:
            c.ambiguities.append(
                Ambiguity(
                    id="neck",
                    subject=[head],
                    question="Only one centre bone above the chest; is it the head (no neck bone) or the neck?",
                    candidates=["head", "neck"],
                    deterministic_best="head",
                    confidence=0.5,
                )
            )
        hk = c.bm[head]
        nkids = len(_center_children(c, head, exclude))
        shape = f"branches into {nkids} centre children" if nkids else "terminal bone"
        head_ev = [
            Evidence(
                kind="structure",
                detail=f"end of centre chain above chest ({shape})",
                weight=0.8,
            )
        ]
        c.add_segment("head", SegmentKind.HEAD, [head], Side.CENTER, 0.85 if neck else 0.6, head_ev, parent=prev_seg)
        c.add_joint("skull_base", JointKind.SKULL_BASE, head, prev_seg, "head", Side.CENTER, 0.8, "ball")
        top_pt = v(hk.tail) if (v(hk.tail) - v(hk.head)) @ c.up > 0 else v(hk.head)
        c.effectors.append(
            EndEffector(
                id="head",
                kind="head",
                segment="head",
                position=rnd(0.5 * (v(hk.head) + v(hk.tail))),
                contacts=["head_top"],
                confidence=0.8,
            )
        )
        c.contacts.append(
            ContactPoint(id="head_top", kind="head_top", segment="head", position=rnd(top_pt), confidence=0.6)
        )
        _head_children(c, head)

    # Tails: centre chains hanging off pelvis/spine that point backward or down.
    for sb in [pelvis, *spine_path[1:]]:
        for k in _center_children(c, sb, exclude | set(chain)):
            if k in c.assigned:
                continue
            tchain = c.main_chain(k)
            d = unit(v(c.bm[tchain[-1]].tail) - v(c.bm[k].head))
            back = float(-(d @ c.fwd))
            down = float(-(d @ c.up))
            if back > 0.3 or (down > 0.5 and back > -0.2):
                conf = 0.6 + 0.3 * min(back, 1.0) * (1.0 if len(tchain) >= 2 else 0.5)
                tid = "tail" if not any(s.id == "tail" for s in c.segments) else f"tail:{_label(k)}"
                c.add_segment(
                    tid,
                    SegmentKind.TAIL,
                    tchain,
                    Side.CENTER,
                    conf,
                    [
                        Evidence(
                            kind="geometry",
                            detail=f"centre chain from {sb} pointing back ({back:+.2f}) / down "
                            f"({down:+.2f}), {len(tchain)} bones",
                            weight=0.7,
                        )
                    ],
                    parent=c.assigned.get(sb),
                )
                for i, bn in enumerate(tchain):
                    c.add_joint(
                        f"{tid}.{i}",
                        JointKind.TAIL,
                        bn,
                        c.assigned.get(sb) if i == 0 else tid,
                        tid,
                        Side.CENTER,
                        conf,
                        "ball",
                    )
                tip = c.bm[tchain[-1]].tail
                c.effectors.append(
                    EndEffector(id=f"{tid}_tip", kind="tail_tip", segment=tid, position=tip, confidence=conf)
                )
                c.limbs.append(
                    Limb(
                        id=tid,
                        kind=LimbKind.TAIL,
                        side=Side.CENTER,
                        segments=[tid],
                        joints=[f"{tid}.{i}" for i in range(len(tchain))],
                        attached_to=c.assigned.get(sb),
                        end_effector=f"{tid}_tip",
                        chain_length=round(c.chain_len(tchain), 5),
                        confidence=round(conf, 3),
                    )
                )
    info.update({"pelvis": pelvis, "chest": chest, "head": head, "spine_path": spine_path})
    return info


def _head_children(c: _Ctx, head: str) -> None:
    kids = [k for k in c.bm[head].children if c.bm[k].side == Side.CENTER]
    jaw_cands = []
    for k in kids:
        d = unit(v(c.bm[k].tail) - v(c.bm[k].head))
        if d @ c.fwd > 0.3 and d @ c.up < 0.2:
            jaw_cands.append(k)
    named = [k for k in jaw_cands if "jaw" in k.lower()]
    if len(jaw_cands) == 1 or len(named) == 1:
        jaw = named[0] if named else jaw_cands[0]
        ev = [Evidence(kind="geometry", detail="centre child of head pointing forward/down", weight=0.5)]
        conf = 0.6
        if named:
            ev.append(Evidence(kind="name_hint", detail=f"bone name contains 'jaw' ({jaw})", weight=0.3))
            conf = 0.8
        c.add_segment("jaw", SegmentKind.JAW, c.subtree(jaw), Side.CENTER, conf, ev, parent="head")
        c.add_joint("jaw", JointKind.JAW, jaw, "head", "jaw", Side.CENTER, conf, "hinge")
    elif len(jaw_cands) > 1:
        c.ambiguities.append(
            Ambiguity(
                id="jaw",
                subject=jaw_cands,
                question="Several forward/down-pointing centre children of the head; which (if any) is the jaw?",
                candidates=[*jaw_cands, "none"],
                confidence=0.3,
            )
        )


# ----------------------------------------------------------------------------------------
# Limbs
# ----------------------------------------------------------------------------------------


def _split_chain(c: _Ctx, chain: list[str]) -> tuple[int, str]:
    """Index of the upper segment: the longest consecutive pair among the first bones."""
    best_i, best = 0, -1.0
    for i in range(min(2, len(chain) - 1)):
        s = c.bm[chain[i]].length + c.bm[chain[i + 1]].length
        if s > best:
            best_i, best = i, s
    return best_i, f"longest consecutive pair {chain[best_i]} + {chain[best_i + 1]} ({best:.3f})"


def _digits(c: _Ctx, hub: str, exclude: set[str]) -> tuple[list[list[str]], list[str]]:
    """Split the subtree below `hub` into digit chains; bones shared by several digits
    (palm / metacarpals) are returned separately."""
    sub = [n for n in c.subtree(hub) if n != hub and n not in exclude]
    subset = set(sub)
    leaves = [n for n in sub if not [k for k in c.bm[n].children if k in subset]]
    paths = []
    for leaf in leaves:
        p, cur = [], leaf
        while cur is not None and cur != hub and cur in subset:
            p.append(cur)
            cur = c.bm[cur].parent
        paths.append(list(reversed(p)))
    count: dict[str, int] = {}
    for p in paths:
        for n in p:
            count[n] = count.get(n, 0) + 1
    shared = sorted(n for n, k in count.items() if k > 1)
    digits = [[n for n in p if count[n] == 1] for p in paths]
    digits = [d for d in digits if d]
    # Single-bone leaves that are not digit-like (e.g. heel helpers) are kept as digits too;
    # callers filter by geometry.
    return digits, shared


def _add_digits(c: _Ctx, limb_id: str, hub_seg: str, side: Side, digits: list[list[str]], kind_tip: str) -> list[str]:
    ids = []
    for d in digits:
        did = c.unique_segment_id(f"{limb_id}.{_label(d[-1])}")
        c.add_segment(
            did,
            SegmentKind.DIGIT,
            d,
            side,
            0.7,
            [Evidence(kind="structure", detail=f"chain below {hub_seg}", weight=0.6)],
            parent=hub_seg,
        )
        jids = []
        for i, bn in enumerate(d):
            j = c.add_joint(
                f"{did}.{i}",
                JointKind.DIGIT,
                bn,
                hub_seg if i == 0 else did,
                did,
                side,
                0.7,
                "hinge" if i > 0 else "saddle",
            )
            jids.append(j.id)
        if len(d) >= 3:
            j = c.joints[-1]
            prev2 = c.bm[d[-2]]
            j.bend = hinge_bend(
                prev2.head,
                c.bm[d[-1]].head,
                c.bm[d[-1]].tail,
                frame_axes=c.axes(),
                lower_bone=d[-1],
                lower_matrix_world=_mw(c, d[-1]),
            )
        tip = c.bm[d[-1]].tail
        c.contacts.append(
            ContactPoint(
                id=f"{did}.tip",
                kind=kind_tip,
                segment=did,
                position=tip,  # type: ignore[arg-type]
                confidence=0.6,
            )
        )
        ids.append(did)
    return ids


def _mw(c: _Ctx, bone: str) -> list[list[float]] | None:
    b = c.ex.bone_map().get(bone)
    return b.matrix_world_rest if b else None


def _build_limb(c: _Ctx, br: _Branch, poles: dict[str, list[float]]) -> None:
    side = br.side
    base_kind = "arm" if br.kind == "arm" else "leg"
    repeat = any(lb.id == f"{base_kind}{_side_tag(side)}" for lb in c.limbs)
    # A second limb pair of the same kind (e.g. quadruped front legs) gets a distinct id namespace.
    tag = (".fore" if base_kind == "leg" else ".extra") + _side_tag(side) if repeat else _side_tag(side)
    chain = br.chain
    attach_seg = c.assigned.get(br.attach or "", None)
    is_arm = br.kind == "arm"
    limb_id = f"{'arm' if is_arm else 'leg'}{tag}"
    if len(chain) < 2:
        return
    i, why = _split_chain(c, chain)
    names = (
        [SegmentKind.CLAVICLE, SegmentKind.UPPER_ARM, SegmentKind.FOREARM, SegmentKind.HAND]
        if is_arm
        else [SegmentKind.PELVIS, SegmentKind.THIGH, SegmentKind.SHIN, SegmentKind.FOOT, SegmentKind.TOE]
    )
    seg_ids: list[str] = []
    joint_ids: list[str] = []
    ev = [*br.evidence, Evidence(kind="structure", detail=f"limb split: {why}", weight=0.6)]
    prev = attach_seg
    if i == 1:
        k0 = names[0] if is_arm else SegmentKind.APPENDAGE
        sid = f"{'clavicle' if is_arm else 'hip_bone'}{tag}"
        c.add_segment(sid, k0, [chain[0]], side, br.conf * 0.9, ev, parent=prev)
        if is_arm:
            joint_ids.append(
                c.add_joint(
                    f"sternoclavicular{tag}",
                    JointKind.STERNOCLAVICULAR,
                    chain[0],
                    prev,
                    sid,
                    side,
                    br.conf * 0.9,
                    "ball",
                ).id
            )
        seg_ids.append(sid)
        prev = sid
    upper, lower = chain[i], chain[i + 1]
    rest = chain[i + 2 :]
    up_kind, low_kind = (names[1], names[2]) if is_arm else (names[1], names[2])
    up_id, low_id = f"{up_kind.value}{tag}", f"{low_kind.value}{tag}"
    c.add_segment(up_id, up_kind, [upper], side, br.conf, ev, parent=prev)
    c.add_segment(low_id, low_kind, [lower], side, br.conf, ev, parent=up_id)
    seg_ids += [up_id, low_id]
    j_root = c.add_joint(
        f"{'shoulder' if is_arm else 'hip'}{tag}",
        JointKind.SHOULDER if is_arm else JointKind.HIP,
        upper,
        prev,
        up_id,
        side,
        br.conf,
        "ball",
    )
    j_mid = c.add_joint(
        f"{'elbow' if is_arm else 'knee'}{tag}",
        JointKind.ELBOW if is_arm else JointKind.KNEE,
        lower,
        up_id,
        low_id,
        side,
        br.conf,
        "hinge",
    )
    joint_ids += [j_root.id, j_mid.id]
    end_bone = rest[0] if rest else None
    j_mid.bend = hinge_bend(
        c.bm[upper].head,
        c.bm[lower].head,
        c.bm[end_bone].head if end_bone else c.bm[lower].tail,
        frame_axes=c.axes(),
        lower_bone=lower,
        lower_matrix_world=_mw(c, lower),
        prior_points="backward" if is_arm else "forward",
        pole_position=poles.get(lower) or poles.get(upper),
    )
    if j_mid.bend.matches_prior is False:
        c.ambiguities.append(
            Ambiguity(
                id=f"bend:{j_mid.id}",
                subject=[upper, lower],
                question=(
                    f"The {j_mid.kind.value} points {'forward' if is_arm else 'backward'} relative to the body, "
                    "opposite to the humanoid prior. Is this anatomical (e.g. digitigrade/bird leg, reversed "
                    "joint) or is the body forward direction wrong?"
                ),
                candidates=["anatomical_reversed_joint", "forward_direction_wrong", "rest_pose_artifact"],
                deterministic_best="anatomical_reversed_joint",
                confidence=0.5,
                evidence=j_mid.bend.evidence,
                context={
                    "joint_points_toward": j_mid.bend.joint_points_toward,
                    "body_relative": j_mid.bend.body_relative,
                },
            )
        )
    effector_id = None
    if end_bone is not None:
        hub_kind = names[3] if is_arm else names[3]
        hub_id = f"{hub_kind.value}{tag}"
        c.add_segment(hub_id, hub_kind, [end_bone], side, br.conf * 0.95, ev, parent=low_id)
        seg_ids.append(hub_id)
        jw = c.add_joint(
            f"{'wrist' if is_arm else 'ankle'}{tag}",
            JointKind.WRIST if is_arm else JointKind.ANKLE,
            end_bone,
            low_id,
            hub_id,
            side,
            br.conf * 0.95,
            "ball" if is_arm else "hinge",
        )
        joint_ids.append(jw.id)
        effector_id = f"{'hand' if is_arm else 'foot'}{tag}"
        if is_arm:
            digits, shared = _digits(c, end_bone, set())
            if shared:
                seg = c.segments[-1] if c.segments[-1].id == hub_id else next(s for s in c.segments if s.id == hub_id)
                seg.source_bones = [end_bone, *shared]
                for s in shared:
                    c.assigned.setdefault(s, hub_id)
            digit_ids = _add_digits(c, limb_id, hub_id, side, digits, "fingertip")
            hb = c.bm[end_bone]
            pts = [v(c.bm[n].head) for n in [end_bone, *shared]] + [v(hb.tail)]
            palm = rnd(np.mean(pts, axis=0))
            c.contacts.append(ContactPoint(id=f"palm{tag}", kind="palm", segment=hub_id, position=palm, confidence=0.6))
            c.effectors.append(
                EndEffector(
                    id=effector_id,
                    kind="hand",
                    segment=hub_id,
                    position=hb.head,
                    contacts=[f"palm{tag}"],
                    confidence=round(br.conf * 0.95, 3),
                )
            )
        else:
            foot = c.bm[end_bone]
            # Ankle bend (dorsiflexion axis) from shin/foot geometry.
            jw.bend = hinge_bend(
                c.bm[lower].head,
                foot.head,
                foot.tail,
                frame_axes=c.axes(),
                lower_bone=end_bone,
                lower_matrix_world=_mw(c, end_bone),
            )
            toe = rest[1] if len(rest) >= 2 else None
            heel_bones = [
                k
                for k in foot.children
                if k != toe
                and c.h(c.bm[k].head) <= GROUND_FRACTION * c.height
                and float((0.5 * (v(c.bm[k].head) + v(c.bm[k].tail)) - v(foot.head)) @ c.fwd) <= 0.02 * c.height
            ]
            contact_ids = []
            if heel_bones:
                hb = c.bm[heel_bones[0]]
                heel_pos, heel_conf = rnd(0.5 * (v(hb.head) + v(hb.tail))), 0.75
                for hbn in heel_bones:
                    c.assigned.setdefault(hbn, hub_id)
                c.segments[-1].source_bones = (
                    [end_bone, *heel_bones] if c.segments[-1].id == hub_id else c.segments[-1].source_bones
                )
            else:
                a = v(foot.head)
                heel_pos, heel_conf = rnd(a - (c.h(a)) * c.up), 0.4
            c.contacts.append(
                ContactPoint(id=f"heel{tag}", kind="heel", segment=hub_id, position=heel_pos, confidence=heel_conf)
            )
            contact_ids.append(f"heel{tag}")
            if toe is not None:
                toe_id = f"toe{tag}"
                c.add_segment(toe_id, SegmentKind.TOE, [toe], side, br.conf * 0.9, ev, parent=hub_id)
                seg_ids.append(toe_id)
                joint_ids.append(
                    c.add_joint(f"ball{tag}", JointKind.BALL, toe, hub_id, toe_id, side, br.conf * 0.9, "hinge").id
                )
                c.contacts.append(
                    ContactPoint(id=f"ball{tag}", kind="ball", segment=toe_id, position=c.bm[toe].head, confidence=0.75)
                )
                contact_ids.append(f"ball{tag}")
                digits, shared = _digits(c, toe, set())
                digit_ids = _add_digits(c, limb_id, toe_id, side, digits, "toe_tip")
                tips = [v(c.bm[d[-1]].tail) for d in digits] or [v(c.bm[toe].tail)]
                tip = max(tips, key=lambda p: float(p @ c.fwd))
                c.contacts.append(
                    ContactPoint(id=f"toe_tip{tag}", kind="toe_tip", segment=toe_id, position=rnd(tip), confidence=0.7)
                )
                contact_ids.append(f"toe_tip{tag}")
            else:
                digit_ids = []
                c.contacts.append(
                    ContactPoint(id=f"toe_tip{tag}", kind="toe_tip", segment=hub_id, position=foot.tail, confidence=0.5)
                )
                contact_ids.append(f"toe_tip{tag}")
            c.effectors.append(
                EndEffector(
                    id=effector_id,
                    kind="foot",
                    segment=hub_id,
                    position=foot.head,
                    contacts=contact_ids,
                    confidence=round(br.conf * 0.95, 3),
                )
            )
    else:
        digit_ids = []
        effector_id = f"{limb_id}_tip"
        c.effectors.append(
            EndEffector(id=effector_id, kind="appendage_tip", segment=low_id, position=c.bm[lower].tail, confidence=0.4)
        )
        c.notes.append(f"Limb {limb_id} has no hand/foot bone after the lower segment.")
    c.limbs.append(
        Limb(
            id=limb_id,
            kind=LimbKind.ARM if is_arm else LimbKind.LEG,
            side=side,
            segments=seg_ids,
            joints=joint_ids,
            attached_to=attach_seg,
            end_effector=effector_id,
            chain_length=round(c.chain_len([upper, lower]), 5),
            reaches_ground=not is_arm,
            digits=digit_ids,
            confidence=round(br.conf, 3),
            evidence=ev,
        )
    )


def _appendages(c: _Ctx) -> None:
    """Everything not yet assigned becomes an appendage segment (grouped per subtree root)."""
    seen_pairs: set[str] = set()
    for b in c.skel.bones:
        if b.name in c.assigned:
            continue
        if b.parent is not None and b.parent not in c.assigned:
            continue  # handled with its subtree root
        sub = [n for n in c.subtree(b.name) if n not in c.assigned]
        chain = [n for n in c.main_chain(b.name) if n in sub]
        sid = c.unique_segment_id(f"appendage:{_label(b.name)}{_side_tag(b.side)}")
        parent_seg = c.assigned.get(b.parent or "")
        size = c.chain_len(chain)
        c.add_segment(
            sid,
            SegmentKind.APPENDAGE,
            sub,
            b.side,
            0.5,
            [
                Evidence(
                    kind="structure",
                    detail=f"unclassified subtree of {len(sub)} bones attached to {parent_seg}",
                    weight=0.4,
                )
            ],
            parent=parent_seg,
        )
        c.add_joint(f"{sid}.root", JointKind.GENERIC, b.name, parent_seg, sid, b.side, 0.5, "unknown")
        if len(chain) >= 2:
            c.limbs.append(
                Limb(
                    id=sid,
                    kind=LimbKind.APPENDAGE,
                    side=b.side,
                    segments=[sid],
                    joints=[f"{sid}.root"],
                    attached_to=parent_seg,
                    end_effector=None,
                    chain_length=round(size, 5),
                    confidence=0.4,
                )
            )
        if size < APPENDAGE_MIN_FRACTION * c.height:
            continue
        mirror = mirror_name(b.name)
        key = min(b.name, mirror) if mirror else b.name
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        d = unit(v(c.bm[chain[-1]].tail) - v(b.head))
        c.ambiguities.append(
            Ambiguity(
                id=c.unique_ambiguity_id(f"appendage:{_label(b.name)}"),
                subject=[b.name] + ([mirror] if mirror and mirror in c.bm else []),
                question=f"What anatomical/accessory structure is the chain starting at {b.name!r} "
                f"(attached to {parent_seg})?",
                candidates=[
                    "face_rig",
                    "wing",
                    "feather_group",
                    "ear",
                    "hair",
                    "breast",
                    "clothing",
                    "accessory",
                    "tail",
                    "other",
                ],
                confidence=0.0,
                context={
                    "bones": len(sub),
                    "chain_length_rel_height": round(size / c.height, 3),
                    "direction_body": {
                        "forward": round(float(d @ c.fwd), 2),
                        "left": round(float(d @ c.left), 2),
                        "up": round(float(d @ c.up), 2),
                    },
                    "attached_to": parent_seg,
                    "mirrored": bool(mirror and mirror in c.bm),
                },
            )
        )


def _validate_forward(c: _Ctx) -> None:
    feet = [e for e in c.effectors if e.kind == "foot"]
    dots = []
    for f in feet:
        tip = next((p for p in c.contacts if p.id in f.contacts and p.kind == "toe_tip"), None)
        if tip is None:
            continue
        d = v(tip.position) - v(f.position)
        d = unit(d - (d @ c.up) * c.up)
        dots.append(float(d @ c.fwd))
    if not dots:
        return
    m = float(np.mean(dots))
    if m > 0.3:
        c.frame.confidence = round(min(0.97, c.frame.confidence + 0.1), 3)
        c.frame.evidence.append(Evidence(kind="feet", detail=f"feet point forward (mean cos {m:+.2f})", weight=0.8))
    elif m < -0.3:
        c.frame.confidence = round(max(0.2, c.frame.confidence - 0.3), 3)
        c.frame.evidence.append(Evidence(kind="feet", detail=f"feet point BACKWARD (mean cos {m:+.2f})", weight=-0.8))
        c.ambiguities.append(
            Ambiguity(
                id="forward_direction",
                subject=[f.id for f in feet],
                question="Feet point opposite to the forward axis derived from L/R naming. Which way does the "
                "character face?",
                candidates=["as_computed", "reversed"],
                deterministic_best="as_computed",
                confidence=0.3,
            )
        )
    else:
        c.frame.evidence.append(Evidence(kind="feet", detail=f"feet direction inconclusive ({m:+.2f})", weight=0.0))


def _collect_poles(ex: RigExtraction, skel: Skeleton) -> dict[str, list[float]]:
    """IK pole positions keyed by structural bones coinciding with the IK chain's lower bone."""
    bones = ex.bone_map()
    out: dict[str, list[float]] = {}
    for b in ex.bones:
        for con in b.constraints:
            if con.type != "IK" or not con.active:
                continue
            pole = next((t for t in con.targets if t.role == "pole" and t.subtarget in bones), None)
            if pole is None:
                continue
            for s in skel.bones:
                if np.allclose(s.head, b.head_world, atol=1e-4) and np.allclose(s.tail, b.tail_world, atol=1e-4):
                    out[s.name] = bones[pole.subtarget].head_world
    return out


def analyze_anatomy(ex: RigExtraction, skel: Skeleton) -> AnatomyProfile:
    frame, sym = compute_body_frame(ex, skel)
    c = _Ctx(
        ex=ex, skel=skel, frame=frame, bm=skel.bone_map(), up=v(frame.up), fwd=v(frame.forward), left=v(frame.left)
    )
    branches = _lateral_branches(c)
    _classify_branches(c, branches)
    _trunk(c, branches)
    poles = _collect_poles(ex, skel)
    for br in sorted(branches, key=lambda b: (b.kind != "leg", b.kind != "arm", c.h(c.bm[b.root].head), b.side.value)):
        if br.kind in ("leg", "arm") and br.root not in c.assigned:
            _build_limb(c, br, poles)
    for kind in ("arm", "leg"):
        sides = {lb.side for lb in c.limbs if lb.kind.value == kind}
        if len(sides) == 1:
            c.ambiguities.append(
                Ambiguity(
                    id=f"asymmetric_{kind}s",
                    subject=[lb.id for lb in c.limbs if lb.kind.value == kind],
                    question=f"Only one {kind} was detected; is the rig asymmetric?",
                    candidates=["asymmetric_rig", "detection_failure"],
                    confidence=0.3,
                )
            )
        if not sides:
            c.notes.append(f"No {kind}s detected.")
    _appendages(c)
    _validate_forward(c)
    masses = estimate_segment_masses(ex, skel, c.segments)
    for s in c.segments:
        s.mass = masses.get(s.id)
    unassigned = sorted(b.name for b in skel.bones if b.name not in c.assigned)
    return AnatomyProfile(
        body_frame=c.frame,
        symmetry=sym,
        segments=c.segments,
        joints=c.joints,
        limbs=c.limbs,
        end_effectors=c.effectors,
        contacts=c.contacts,
        ambiguities=c.ambiguities,
        unassigned_bones=unassigned,
        notes=c.notes + skel.notes,
    )
