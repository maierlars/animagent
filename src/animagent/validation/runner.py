"""Orchestrates validation: live analysis + read-only probe, isolated experiment, checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from animagent.blender.background import background_blender
from animagent.blender.session import BlenderSession
from animagent.config import MCPServerConfig
from animagent.logging_setup import get_logger
from animagent.mcp_client import BlenderMCPClient
from animagent.models.profile import RigProfile
from animagent.pipeline import analyze_rig, fingerprint
from animagent.validation.checks import (
    control_provenance,
    drivers_report,
    frame_report,
    joint_table,
    run_checks,
)
from animagent.validation.inputs import experiment_args, probe_args

log = get_logger(__name__)


async def live_analysis(armature: str) -> tuple[RigProfile, dict[str, Any], dict[str, Any]]:
    """Analysis + probe on the running Blender; fingerprinted around the whole sequence."""
    async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
        s = BlenderSession(client)
        before = await fingerprint(s, armature)
        profile = await analyze_rig(s, armature)
        probe = await s.run_script("validation_probe", probe_args(profile))
        after = await fingerprint(s, armature)
    integrity = {
        "before": before.model_dump(),
        "after": after.model_dump(),
        "identical": before.hashes == after.hashes,
        "changed": sorted(k for k in before.hashes if before.hashes[k] != after.hashes.get(k)),
        "is_dirty_before": before.info.get("is_dirty"),
        "is_dirty_after": after.info.get("is_dirty"),
        "operations": [
            "state_fingerprint",
            "extract_rig",
            "state_fingerprint",
            "validation_probe",
            "state_fingerprint",
        ],
    }
    return profile, probe, integrity


async def isolated_experiment(profile: RigProfile, reference_blend: Path) -> dict[str, Any]:
    """Experiments on a separate background Blender that opens the reference file (never saved)."""
    import anyio

    def run() -> dict[str, Any]:
        with background_blender(reference_blend) as port:

            async def go() -> dict[str, Any]:
                cfg = MCPServerConfig.from_env()
                cfg.blender_port = port
                async with BlenderMCPClient(cfg) as client:
                    s = BlenderSession(client)
                    ref_fp = await fingerprint(s, profile.source.armature)
                    res = await s.run_script("validation_experiment", experiment_args(profile))
                    res["reference_fingerprint_before_experiment"] = ref_fp.model_dump()
                    return res

            return anyio.run(go)

    return await anyio.to_thread.run_sync(run)


def build_reports(profile: RigProfile, probe: dict[str, Any], experiment: dict[str, Any] | None) -> dict[str, Any]:
    cl = run_checks(profile, probe, experiment)
    counts: dict[str, dict[str, int]] = {}
    for c in cl.items:
        counts.setdefault(c["category"], {}).setdefault(c["status"], 0)
        counts[c["category"]][c["status"]] += 1
    validation = {
        "schema_version": "anatomy_validation/1",
        "armature": profile.source.armature,
        "status_counts_by_category": counts,
        "categories": {
            "A": "coordinate frames",
            "B": "chain connectivity",
            "C": "left/right symmetry",
            "D": "knee/elbow bending",
            "E": "landmarks vs skin geometry",
            "F": "anatomy vs rig machinery",
            "G": "control coverage",
            "H": "IK/FK switch semantics",
        },
        "methods": {
            "independent_measurement": "measured in Blender / raw data without the analyzer's interpretation",
            "experimental": "control perturbation in an isolated background copy, evaluated motion",
            "raw_data": "arithmetic on raw extracted Blender values",
            "consistency": "internal consistency of analyzer output (weak evidence)",
            "heuristic": "comparison with a prior; informational",
        },
        "checks": cl.items,
        "body_frame": frame_report(profile),
        "joints": joint_table(profile),
        "probe": probe,
    }
    controls = {
        "schema_version": "control_validation/1",
        "evidence_levels": {
            "name_or_collection": "inferred from names or bone-collection membership only",
            "hierarchy": "inferred from parenting",
            "constraint_or_driver": "verified from constraint targets, IK constraints or drivers",
            "experimental": "verified by evaluated motion in the isolated copy (see 'experimental')",
        },
        "mapping": profile.controls.model_dump(),
        "provenance": control_provenance(profile, experiment),
        "switch_drivers": drivers_report(profile, probe),
        "experiment_performed": experiment is not None,
    }
    return {"validation": validation, "controls": controls, "experiment": experiment}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str))
