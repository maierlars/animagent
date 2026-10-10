"""Synthetic rigs (as `RigExtraction` objects) for Blender-free unit tests.

The humanoid faces -Y with its left side on +X (Blender/Rigify convention), height ~1.75.
`rigify=True` adds a Rigify-like control layer: ORG/DEF/MCH bones, FK and IK controls, IK
constraints with poles and an IK/FK switch property driven through a generator F-modifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from animagent.models.rig import RigExtraction

Vec = tuple[float, float, float]

# name: (head, tail, parent)
_CENTER: dict[str, tuple[Vec, Vec, str | None]] = {
    "hips": ((0, 0, 1.0), (0, 0, 1.1), None),
    "spine": ((0, 0, 1.1), (0, 0, 1.25), "hips"),
    "chest": ((0, 0, 1.25), (0, 0, 1.4), "spine"),
    "neck": ((0, 0, 1.4), (0, -0.01, 1.5), "chest"),
    "head": ((0, -0.01, 1.5), (0, -0.01, 1.7), "neck"),
}
_LEFT: dict[str, tuple[Vec, Vec, str]] = {
    "shoulder": ((0.02, 0, 1.38), (0.18, 0.02, 1.38), "chest"),
    "upper_arm": ((0.18, 0.02, 1.38), (0.45, 0.05, 1.38), "shoulder"),
    "forearm": ((0.45, 0.05, 1.38), (0.72, 0.0, 1.38), "upper_arm"),
    "hand": ((0.72, 0.0, 1.38), (0.8, 0.0, 1.38), "forearm"),
    "f_index.01": ((0.8, -0.02, 1.38), (0.83, -0.02, 1.38), "hand"),
    "f_index.02": ((0.83, -0.02, 1.38), (0.855, -0.02, 1.37), "f_index.01"),
    "f_index.03": ((0.855, -0.02, 1.37), (0.87, -0.02, 1.355), "f_index.02"),
    "f_ring.01": ((0.8, 0.02, 1.38), (0.83, 0.02, 1.38), "hand"),
    "f_ring.02": ((0.83, 0.02, 1.38), (0.855, 0.02, 1.37), "f_ring.01"),
    "f_ring.03": ((0.855, 0.02, 1.37), (0.87, 0.02, 1.355), "f_ring.02"),
    "thigh": ((0.1, 0, 1.0), (0.1, -0.03, 0.55), "hips"),
    "shin": ((0.1, -0.03, 0.55), (0.1, 0.0, 0.1), "thigh"),
    "foot": ((0.1, 0.0, 0.1), (0.1, -0.12, 0.03), "shin"),
    "toe": ((0.1, -0.12, 0.03), (0.1, -0.18, 0.03), "foot"),
    "heel": ((0.13, 0.03, 0.0), (0.07, 0.03, 0.0), "foot"),
}
_TAIL: dict[str, tuple[Vec, Vec, str]] = {
    "tail": ((0, 0.05, 1.0), (0, 0.25, 0.9), "hips"),
    "tail.001": ((0, 0.25, 0.9), (0, 0.45, 0.85), "tail"),
}
_WING: dict[str, tuple[Vec, Vec, str]] = {  # small accessory chain on the chest
    "feather": ((0.05, 0.08, 1.35), (0.12, 0.2, 1.3), "chest"),
    "feather.001": ((0.12, 0.2, 1.3), (0.18, 0.3, 1.22), "feather"),
}


def _side_name(base: str, side: str) -> str:
    return f"{base}.{side}"


def _mirror(p: Vec) -> Vec:
    return (-p[0], p[1], p[2])


def bone_matrix(head: Vec, tail: Vec) -> list[list[float]]:
    """Rest matrix with Y along the bone; X = world X projected (or -Y when parallel)."""
    h, t = np.array(head, float), np.array(tail, float)
    y = (t - h) / np.linalg.norm(t - h)
    ref = np.array([1.0, 0, 0]) if abs(y[0]) < 0.9 else np.array([0, -1.0, 0])
    x = ref - (ref @ y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, h
    return m.tolist()


def humanoid_geometry(
    *, tail: bool = False, accessory: bool = False, knee_backward: bool = False, swap_sides: bool = False
) -> dict[str, tuple[Vec, Vec, str | None]]:
    geo: dict[str, tuple[Vec, Vec, str | None]] = dict(_CENTER)
    left = dict(_LEFT)
    if knee_backward:  # digitigrade-like: knee points backward
        left["thigh"] = ((0.1, 0, 1.0), (0.1, 0.06, 0.55), "hips")
        left["shin"] = ((0.1, 0.06, 0.55), (0.1, 0.0, 0.1), "thigh")
    if accessory:
        left.update(_WING)
    for base, (h, t, p) in left.items():
        lname, rname = ("R", "L") if swap_sides else ("L", "R")
        par_l = p if p in _CENTER else _side_name(p, lname)
        par_r = p if p in _CENTER else _side_name(p, rname)
        geo[_side_name(base, lname)] = (h, t, par_l)
        geo[_side_name(base, rname)] = (_mirror(h), _mirror(t), par_r)
    if tail:
        geo.update(_TAIL)
    return geo


def _bone(name: str, head: Vec, tail: Vec, parent: str | None, deform: bool, **extra: Any) -> dict[str, Any]:
    m = bone_matrix(head, tail)
    d = {
        "name": name,
        "parent": parent,
        "children": [],
        "use_deform": deform,
        "head_world": list(head),
        "tail_world": list(tail),
        "length": float(np.linalg.norm(np.subtract(tail, head))),
        "matrix_local": m,
        "matrix_world_rest": m,
        "collections": [],
        "constraints": [],
        "custom_props": {},
        "pose": {"matrix_armature": m},
    }
    d.update(extra)
    return d


def _mesh(groups: dict[str, tuple[Vec, Vec]]) -> dict[str, Any]:
    gs = []
    for name, (h, t) in groups.items():
        length = float(np.linalg.norm(np.subtract(t, h)))
        gs.append(
            {
                "name": name,
                "is_deform_bone": True,
                "vertex_count": 20,
                "weight_sum": 20.0,
                "area_weighted": 0.05 * length,
                "centroid_world": list(0.5 * (np.array(h) + np.array(t))),
                "rms_radius": 0.04,
                "strong_vertex_count": 20,
            }
        )
    return {
        "name": "Body",
        "data_name": "Body",
        "matrix_world": np.eye(4).tolist(),
        "modifiers": [{"index": 0, "name": "Armature", "type": "ARMATURE", "object": "Rig"}],
        "weights": {
            "vertex_count": 20 * len(gs),
            "polygon_count": 10,
            "surface_area": 1.0,
            "bbox_min": [-0.9, -0.2, 0.0],
            "bbox_max": [0.9, 0.5, 1.75],
            "groups": gs,
        },
    }


def _finish(
    bones: list[dict[str, Any]],
    drivers: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    rigify: dict[str, Any] | None = None,
) -> RigExtraction:
    by = {b["name"]: b for b in bones}
    for b in bones:
        if b["parent"]:
            by[b["parent"]]["children"].append(b["name"])
    return RigExtraction.model_validate(
        {
            "blender": {"version": "5.2.2", "file_path": "/tmp/synthetic.blend"},
            "armature": {
                "name": "Rig",
                "data_name": "Rig",
                "matrix_world": np.eye(4).tolist(),
                "bone_count": len(bones),
            },
            "rigify": rigify or {},
            "bones": bones,
            "drivers": drivers,
            "meshes": meshes,
        }
    )


def humanoid_extraction(**kw: Any) -> RigExtraction:
    """A plain (non-Rigify) deform skeleton."""
    geo = humanoid_geometry(**kw)
    bones = [_bone(n, h, t, p, True, custom_shape=None) for n, (h, t, p) in geo.items()]
    return _finish(bones, [], [_mesh({n: (h, t) for n, (h, t, _) in geo.items()})])


def _ct(name: str, sub: str, influence: float = 1.0) -> dict[str, Any]:
    return {
        "name": name,
        "type": "COPY_TRANSFORMS",
        "influence": influence,
        "targets": [{"object": "Rig", "subtarget": sub, "role": "target"}],
    }


def rigify_like_extraction(
    *,
    switch_generator: tuple[float, float] = (1.0, -1.0),
    ik_fk_value: float = 0.0,
    pose_follows: str = "ik",
    tail: bool = False,
) -> RigExtraction:
    """Rigify-like rig: ORG skeleton, DEF copies, FK/IK controls, IK constraints and switches.

    `switch_generator` are the polynomial coefficients mapping IK_FK -> influence of the IK copy
    constraint (Rigify uses (1, -1): IK_FK=0 means IK). `pose_follows` sets which sources the
    evaluated ORG pose matrices coincide with (for the observation check).
    """
    geo = humanoid_geometry(tail=tail)
    bones: list[dict[str, Any]] = []
    drivers: list[dict[str, Any]] = []
    meta = []
    bones.append(_bone("root", (0, 0, 0), (0, 0.3, 0), None, False, custom_shape="WGT-root"))
    bones.append(_bone("torso", (0, 0.1, 1.05), (0, 0.4, 1.05), "root", False, custom_shape="WGT-torso"))
    for n, (h, t, p) in geo.items():
        org_parent = f"ORG-{p}" if p else "torso"
        bones.append(_bone(f"ORG-{n}", h, t, org_parent, False))
        bones.append(_bone(f"DEF-{n}", h, t, f"ORG-{n}", True, constraints=[_ct("Copy Transforms", f"ORG-{n}")]))
        meta.append({"name": n, "parent": p, "head_world": list(h), "tail_world": list(t), "rigify_type": ""})
    by = {b["name"]: b for b in bones}

    def ctrl(name: str, h: Vec, t: Vec, parent: str | None, **extra: Any) -> None:
        b = _bone(name, h, t, parent, False, custom_shape=f"WGT-{name}", **extra)
        bones.append(b)
        by[name] = b

    # Trunk controls: FK controls coincide with ORG bones.
    for n in _CENTER:
        h, t, p = geo[n]
        ctrl(f"{n}_fk" if n not in ("head", "neck") else n, h, t, "torso")
        by[f"ORG-{n}"]["constraints"] = [_ct("Copy Transforms", f"{n}_fk" if n not in ("head", "neck") else n)]
    if tail:
        for n in _TAIL:
            h, t, p = geo[n]
            ctrl(f"{n}_ctrl", h, t, "torso")
            by[f"ORG-{n}"]["constraints"] = [_ct("Copy Transforms", f"{n}_ctrl")]

    for side in ("L", "R"):

        def g(base: str, side: str = side) -> tuple[Vec, Vec, str | None]:
            return geo[f"{base}.{side}"]

        limbs = {"arm": ["upper_arm", "forearm", "hand"], "leg": ["thigh", "shin", "foot", "toe"]}
        for limb, chain in limbs.items():
            parent_ctrl = f"{chain[0]}_parent.{side}"
            h0, t0, _ = g(chain[0])
            ctrl(
                parent_ctrl,
                h0,
                (h0[0], h0[1] + 0.05, h0[2]),
                "torso",
                custom_props={
                    "IK_FK": {"value": ik_fk_value, "ui": {"min": 0.0, "max": 1.0}},
                    "IK_Stretch": {"value": 1.0, "ui": {"min": 0.0, "max": 1.0}},
                },
            )
            prev_fk = parent_ctrl
            for base in chain:
                h, t, _ = g(base)
                ctrl(f"{base}_fk.{side}", h, t, prev_fk)
                prev_fk = f"{base}_fk.{side}"
            # IK: chain controls and mechanism
            uh, ut, _ = g(chain[0])
            lh, lt, _ = g(chain[1])
            eh, et, _ = g(chain[2])
            ctrl(f"{chain[0]}_ik.{side}", uh, ut, parent_ctrl)
            ik_mch = f"MCH-{chain[1]}_ik.{side}"
            target_ctrl = f"{chain[2]}_ik.{side}"
            target_mch = f"MCH-{chain[0]}_ik_target.{side}"
            pole = f"{chain[0]}_ik_target.{side}"
            ctrl(target_ctrl, eh, et, "root")
            bones.append(_bone(target_mch, eh, et, target_ctrl, False))
            by[target_mch] = bones[-1]
            pole_pos = (lh[0], lh[1] + (0.5 if limb == "arm" else -0.5), lh[2])
            ctrl(pole, pole_pos, (pole_pos[0], pole_pos[1], pole_pos[2] + 0.05), "root")
            bones.append(
                _bone(
                    ik_mch,
                    lh,
                    lt,
                    f"{chain[0]}_ik.{side}",
                    False,
                    constraints=[
                        {
                            "name": "IK",
                            "type": "IK",
                            "settings": {"chain_count": 2, "pole_angle": -1.5708},
                            "targets": [
                                {"object": "Rig", "subtarget": target_mch, "role": "target"},
                                {"object": "Rig", "subtarget": pole, "role": "pole"},
                            ],
                        },
                    ],
                )
            )
            by[ik_mch] = bones[-1]
            ik_sources = {chain[0]: f"{chain[0]}_ik.{side}", chain[1]: ik_mch, chain[2]: target_mch}
            if len(chain) > 3:
                ctrl(f"{chain[3]}_ik.{side}", *g(chain[3])[:2], target_ctrl)
                ik_sources[chain[3]] = f"{chain[3]}_ik.{side}"
            for base in chain:
                org = by[f"ORG-{base}.{side}"]
                org["constraints"] = [
                    _ct("Copy Transforms", f"{base}_fk.{side}"),
                    _ct("Copy Transforms.001", ik_sources[base]),
                ]
                drivers.append(
                    {
                        "data_path": f'pose.bones["ORG-{base}.{side}"].constraints["Copy Transforms.001"].influence',
                        "type": "SUM",
                        "expression": "",
                        "variables": [
                            {
                                "name": "var",
                                "type": "SINGLE_PROP",
                                "targets": [{"id_name": "Rig", "data_path": f'pose.bones["{parent_ctrl}"]["IK_FK"]'}],
                            }
                        ],
                        "modifiers": [
                            {
                                "type": "GENERATOR",
                                "mode": "POLYNOMIAL",
                                "poly_order": 1,
                                "coefficients": list(switch_generator),
                            }
                        ],
                    }
                )
                # Evaluated pose: ORG coincides with the source of the followed mode.
                src = ik_sources[base] if pose_follows == "ik" else f"{base}_fk.{side}"
                offset = np.eye(4)
                offset[0, 3] = 0.37  # move both away from rest so the match is not trivial
                m = (np.array(by[src]["matrix_world_rest"]) @ offset).tolist()
                by[src]["pose"]["matrix_armature"] = m
                org["pose"]["matrix_armature"] = m
        # Fingers: plain FK controls.
        for base in ("f_index.01", "f_index.02", "f_index.03", "f_ring.01", "f_ring.02", "f_ring.03"):
            h, t, _ = g(base)
            ctrl(f"{base}.{side}", h, t, f"hand_fk.{side}")
            by[f"ORG-{base}.{side}"]["constraints"] = [_ct("Copy Transforms", f"{base}.{side}")]
        for base in ("shoulder",):
            h, t, _ = g(base)
            ctrl(f"{base}.{side}", h, t, "torso")
            by[f"ORG-{base}.{side}"]["constraints"] = [_ct("Copy Transforms", f"{base}.{side}")]

    groups = {f"DEF-{n}": (h, t) for n, (h, t, _) in geo.items()}
    return _finish(
        bones,
        drivers,
        [_mesh(groups)],
        rigify={"rig_id": "synthetic", "is_rigify_generated": True, "metarig": "metarig", "metarig_bones": meta},
    )
