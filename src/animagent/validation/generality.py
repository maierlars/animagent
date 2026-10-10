"""Run the analysis on stock Rigify rigs (generated in isolated background Blender) and summarise."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import anyio

from animagent.blender.background import BUILD_FIXTURE, background_blender
from animagent.blender.session import BlenderSession
from animagent.config import MCPServerConfig
from animagent.logging_setup import get_logger
from animagent.mcp_client import BlenderMCPClient
from animagent.models.profile import RigProfile
from animagent.pipeline import analyze_rig

log = get_logger(__name__)

METARIGS = ["human", "basic_human", "bird", "cat", "wolf", "horse", "basic_quadruped", "shark"]


def _session_call(port: int, fn):  # type: ignore[no-untyped-def]
    async def go():  # type: ignore[no-untyped-def]
        cfg = MCPServerConfig.from_env()
        cfg.blender_port = port
        async with BlenderMCPClient(cfg) as client:
            return await fn(BlenderSession(client))

    return anyio.run(go)


def summarize(p: RigProfile) -> dict[str, Any]:
    a, cm = p.anatomy, p.controls
    kinds: dict[str, int] = {}
    for s in a.segments:
        kinds[s.kind.value] = kinds.get(s.kind.value, 0) + 1
    limbs = []
    for lb in a.limbs:
        if lb.kind.value == "appendage":
            continue
        lc = next((x for x in cm.limbs if x.limb_id == lb.id), None)
        mid = next((j for j in a.joints if j.id in lb.joints and j.kind.value in ("elbow", "knee")), None)
        limbs.append(
            {
                "id": lb.id,
                "kind": lb.kind.value,
                "segments": lb.segments,
                "digits": len(lb.digits),
                "confidence": lb.confidence,
                "mid_joint_points": mid.bend.body_relative if mid and mid.bend else None,
                "matches_humanoid_prior": mid.bend.matches_prior if mid and mid.bend else None,
                "fk_chain": [c.bone for c in lc.fk_chain] if lc else None,
                "ik_effector": lc.ik_effector.bone if lc and lc.ik_effector else None,
                "ik_pole": lc.ik_pole.bone if lc and lc.ik_pole else None,
                "switch": (
                    {
                        "prop": f"{lc.ik_fk_switch.bone}[{lc.ik_fk_switch.property}]",
                        "ik_value": lc.ik_fk_switch.ik_value,
                    }
                    if lc and lc.ik_fk_switch
                    else None
                ),
            }
        )
    return {
        "armature": p.source.armature,
        "bones": len(p.extraction.bones),
        "structural": len(p.skeleton.bones),
        "skeleton_strategy": p.skeleton.strategy,
        "rig_kind": cm.rig_kind,
        "body_frame": {"forward": a.body_frame.forward, "confidence": a.body_frame.confidence},
        "trunk": {
            k: a.segment_map()[k].source_bones
            for k in ("pelvis", "spine", "chest", "neck", "head", "tail")
            if k in a.segment_map()
        },
        "segment_kinds": kinds,
        "limbs": limbs,
        "appendages": [s.id for s in a.segments if s.kind.value == "appendage"],
        "ambiguities": [{"id": x.id, "question": x.question} for x in a.ambiguities],
        "unmapped_controls": len(cm.unmapped_controls),
        "controls": len(cm.control_bones),
        "readonly_identical": p.readonly_check.identical if p.readonly_check else None,
        "notes": a.notes,
    }


def run_metarig(kind: str, workdir: Path) -> dict[str, Any]:
    blend = workdir / f"{kind}.blend"
    with background_blender() as port:
        built = _session_call(
            port,
            lambda s: s.execute(
                f"ARGS_PATH = {str(blend)!r}\nMETARIG_OP = 'armature_{kind}_metarig_add'\n" + BUILD_FIXTURE
            ),
        )
    out: dict[str, Any] = {"metarig": kind, "fixture": built}
    with background_blender(blend) as port:
        rig = _session_call(port, lambda s: analyze_rig(s, "Rig"))
        meta = _session_call(port, lambda s: analyze_rig(s, "metarig"))
    out["generated_rig"] = summarize(rig)
    out["metarig_as_plain_armature"] = summarize(meta)
    return out


def run_all(kinds: list[str] | None = None) -> list[dict[str, Any]]:
    results = []
    with tempfile.TemporaryDirectory(prefix="animagent_generality_") as tmp:
        for kind in kinds or METARIGS:
            try:
                results.append(run_metarig(kind, Path(tmp)))
            except Exception as ex:  # record failures as results, never hide them
                log.warning("generality.failed", metarig=kind, error=str(ex))
                results.append({"metarig": kind, "error": f"{type(ex).__name__}: {ex}"[:2000]})
    return results
