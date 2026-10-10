"""Build the independent-review package (data part). Narrative documents are written separately.

Usage: animagent review-package --armature RIG --outdir out/review-m1 [--skip-tests]
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio

from animagent import __version__
from animagent.blender.background import background_blender
from animagent.blender.session import BlenderSession
from animagent.config import MCPServerConfig
from animagent.logging_setup import get_logger
from animagent.mcp_client import BlenderMCPClient
from animagent.models.profile import RigProfile
from animagent.pipeline import fingerprint
from animagent.validation.generality import run_all
from animagent.validation.runner import build_reports, isolated_experiment, live_analysis, write_json
from animagent.validation.screens import capture_all

log = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]

STATE_PROBE = """import bpy
wm = bpy.context.window_manager
result = {"is_dirty": bool(bpy.data.is_dirty), "file": bpy.data.filepath,
          "operators": [o.bl_idname for o in wm.operators], "frame": bpy.context.scene.frame_current,
          "mode": bpy.context.mode}"""


def _run(cmd: list[str], cwd: Path = ROOT) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()


def git_info() -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "clean_worktree": status == "",
        "dirty_files": status.splitlines()[:20],
        "remote": _run(["git", "remote", "get-url", "origin"]) or None,
    }


def env_info(blender: dict[str, Any], server: dict[str, Any] | None) -> dict[str, Any]:
    import importlib.metadata as md

    addon_manifest = None
    for p in Path.home().glob(".config/blender/*/extensions/*/mcp/blender_manifest.toml"):
        txt = p.read_text()
        version = next(
            (ln.split("=")[1].strip().strip('"') for ln in txt.splitlines() if ln.startswith("version")), None
        )
        addon_manifest = {"path": str(p), "version": version}
    upstream = _run(["git", "ls-remote", "https://projects.blender.org/lab/blender_mcp.git", "HEAD"]).split()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "animagent": __version__,
        "packages": {k: md.version(k) for k in ("mcp", "pydantic", "numpy", "openai", "structlog", "pytest")},
        "blender": {k: blender.get(k) for k in ("blender_version", "blender_build_hash", "python_version", "numpy")},
        "mcp_server": {
            "package": "blender-mcp (official Blender Lab MCP server)",
            "reported_version": server.get("version") if server else None,
            "protocol": server.get("protocol_version") if server else None,
            "launch_command": MCPServerConfig.from_env().command,
            "upstream_repository": "https://projects.blender.org/lab/blender_mcp",
            "upstream_head_commit_at_packaging": upstream[0] if upstream else None,
            "blender_addon": addon_manifest,
        },
    }


def run_tests(out: Path) -> dict[str, Any]:
    results = {}
    for name, args in (
        ("unit", ["tests/unit"]),
        ("integration_background", ["-m", "blender", "tests/integration"]),
        ("integration_live", ["-m", "live", "tests/integration"]),
    ):
        xml = out / f"results_{name}.xml"
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "ANIMAGENT_LIVE_TESTS": "1"},
        )
        suite = ET.parse(xml).getroot()
        suite = suite.find("testsuite") if suite.tag == "testsuites" else suite
        cases = [
            {
                "name": f"{c.get('classname')}::{c.get('name')}",
                "time_s": float(c.get("time", 0)),
                "status": "failed"
                if c.find("failure") is not None or c.find("error") is not None
                else ("skipped" if c.find("skipped") is not None else "passed"),
            }
            for c in suite.iter("testcase")
        ]
        results[name] = {
            "exit_code": proc.returncode,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "tests": int(suite.get("tests", 0)),
            "failures": int(suite.get("failures", 0)),
            "errors": int(suite.get("errors", 0)),
            "skipped": int(suite.get("skipped", 0)),
            "cases": cases,
            "tail": proc.stdout.strip().splitlines()[-3:],
        }
    # merged JUnit file
    merged = ET.Element("testsuites")
    for name in results:
        root = ET.parse(out / f"results_{name}.xml").getroot()
        for s in [root] if root.tag == "testsuite" else list(root):
            s.set("name", name)
            merged.append(s)
    ET.ElementTree(merged).write(out / "results.xml", encoding="utf-8", xml_declaration=True)
    return results


async def live_phase(armature: str, outdir: Path) -> dict[str, Any]:
    """All interactions with the user's running Blender, with dirty-state tracking after each phase."""
    timeline: list[dict[str, Any]] = []

    async def state(label: str, s: BlenderSession) -> None:
        st = await s.execute(STATE_PROBE, label="state_probe")
        timeline.append({"phase": label, "time": datetime.now(UTC).isoformat(timespec="seconds"), **st})

    async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
        s = BlenderSession(client)
        server = client.server_info.model_dump() if client.server_info else None
        await state("start", s)
        scene = await s.run_script("scene_info", {})
        fp_start = await fingerprint(s, armature)
        await state("after_scene_info_and_fingerprint", s)

    profile, probe, integrity = await live_analysis(armature)
    async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
        s = BlenderSession(client)
        await state("after_analysis_and_probe", s)
        fp_before_shots = await fingerprint(s, armature)
        shots = await capture_all(s, profile, outdir / "screenshots")
        fp_after_shots = await fingerprint(s, armature)
        await state("after_overlay_viewport_screenshots", s)
        overlay_status = await s.run_script("overlay", {"action": "status"})
        fp_end = await fingerprint(s, armature)
        await state("end", s)
    return {
        "server": server,
        "scene": scene,
        "profile": profile,
        "probe": probe,
        "analysis_integrity": integrity,
        "screenshots": shots,
        "timeline": timeline,
        "overlay_status_after": overlay_status,
        "fingerprints": {
            "start": fp_start.model_dump(),
            "before_screenshots": fp_before_shots.model_dump(),
            "after_screenshots": fp_after_shots.model_dump(),
            "end": fp_end.model_dump(),
        },
    }


def disk_comparison(profile: RigProfile, live_end: dict[str, Any]) -> dict[str, Any]:
    ref = Path(profile.source.blend_file)

    def run() -> dict[str, Any]:
        with background_blender(ref) as port:

            async def go() -> dict[str, Any]:
                cfg = MCPServerConfig.from_env()
                cfg.blender_port = port
                async with BlenderMCPClient(cfg) as client:
                    s = BlenderSession(client)
                    fp = await fingerprint(s, profile.source.armature)
                    st = await s.execute(STATE_PROBE, label="state_probe")
                    return {"fingerprint": fp.model_dump(), "state": st}

            return anyio.run(go)

    disk = run()
    dh, lh = disk["fingerprint"]["hashes"], live_end["hashes"]
    return {
        "reference_file": str(ref),
        "reference_mtime": datetime.fromtimestamp(ref.stat().st_mtime, UTC).isoformat(timespec="seconds"),
        "opened_in": "isolated background Blender (command-line load incl. UI; never saved)",
        "per_category": {
            k: ("identical" if dh.get(k) == lh.get(k) else "DIFFERENT") for k in sorted(set(dh) | set(lh))
        },
        "disk_state_after_open": disk["state"],
        "disk_fingerprint": disk["fingerprint"],
    }


def mass_derivation(p: RigProfile) -> dict[str, Any]:
    from animagent.analysis.mass import deform_bone_stats

    stats = deform_bone_stats(p.extraction)
    total = sum(x[0] for x in stats.values())
    bm = p.skeleton.bone_map()
    rows = []
    for s in p.anatomy.segments:
        deform = sorted({d for b in s.source_bones if b in bm for d in bm[b].deform_bones})
        rows.append(
            {
                "segment": s.id,
                "kind": s.kind.value,
                "deform_bones": deform,
                "skin_weighted_area_per_deform_bone": {d: round(stats[d][0], 6) for d in deform if d in stats},
                "relative_mass": s.mass.relative_mass if s.mass else None,
                "center_of_mass": s.mass.center_of_mass if s.mass else None,
            }
        )
    return {
        "method": "relative_mass(segment) = sum over its deform bones of sum_v(weight_v * area_v) / same sum over all "
        "deform bones; area_v = vertex share of adjacent polygon areas (world scale); rest-pose, "
        "pre-modifier mesh data; visible deformed meshes only. Skin weights are NOT masses: artistic "
        "guidance only.",
        "total_skin_weighted_area": round(total, 6),
        "unassigned_share": round(1 - sum(r["relative_mass"] or 0 for r in rows), 6),
        "segments": rows,
    }


def build(armature: str, outdir: Path, skip_tests: bool = False) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)
    (outdir / "tests").mkdir(exist_ok=True)
    git = git_info()
    tests = None if skip_tests else run_tests(outdir / "tests")
    live = anyio.run(live_phase, armature, outdir)
    profile: RigProfile = live["profile"]
    experiment = anyio.run(isolated_experiment, profile, Path(profile.source.blend_file))
    disk = disk_comparison(profile, live["fingerprints"]["end"])
    generality = run_all()
    reports = build_reports(profile, live["probe"], experiment)
    R = outdir / "reports"
    profile.save(R / "character_profile.json")
    write_json(R / "rig_extraction.json", profile.extraction.model_dump())
    write_json(
        R / "anatomy.json",
        {
            **profile.anatomy.model_dump(),
            "skeleton": profile.skeleton.model_dump(),
            "mass_derivation": mass_derivation(profile),
        },
    )
    write_json(R / "anatomy_validation.json", reports["validation"])
    write_json(R / "control_mapping.json", reports["controls"])
    write_json(R / "ambiguities.json", ambiguity_report(profile))
    write_json(R / "experiment_raw.json", experiment)
    write_json(R / "generality.json", generality)
    from animagent.analysis.report import build_report

    write_json(R / "compact_report.json", build_report(profile))
    integrity = {
        "timeline": live["timeline"],
        "analysis_integrity": live["analysis_integrity"],
        "fingerprints": live["fingerprints"],
        "overlay_status_after": live["overlay_status_after"],
        "live_vs_disk": disk,
        "experiment_copy": {
            "file": experiment.get("file"),
            "is_dirty_after": experiment.get("is_dirty_after"),
            "notes": experiment.get("notes"),
        },
    }
    write_json(outdir / "tests" / "scene_integrity.json", integrity)
    write_json(outdir / "screenshots" / "index.json", live["screenshots"])
    if tests:
        write_json(outdir / "tests" / "test_summary.json", tests)
    checks = reports["validation"]["checks"]
    status = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "fail", "warn", "info")}
    manifest = {
        "package": "review-m1",
        "milestone": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "analyzed_armature": armature,
        "source_scene": {
            "file": profile.source.blend_file,
            "scene": profile.source.scene,
            "frame_current": profile.source.frame_current,
            "reference_mtime": disk["reference_mtime"],
            "datablock_counts": live["fingerprints"]["end"]["info"].get("datablock_counts"),
        },
        "code_revision": git,
        "environment": env_info(live["scene"], live["server"]),
        "validation_status": {
            "checks": status,
            "tests": {
                k: {x: v[x] for x in ("tests", "failures", "errors", "skipped")} for k, v in (tests or {}).items()
            },
            "readonly_live_analysis_identical": live["analysis_integrity"]["identical"],
            "live_vs_disk_all_identical": all(v == "identical" for v in disk["per_category"].values()),
            "experiment_performed": True,
            "milestone_validated": "PENDING INDEPENDENT REVIEW (not self-certified)",
        },
    }
    write_json(outdir / "manifest.json", manifest)
    return manifest


def ambiguity_report(p: RigProfile) -> dict[str, Any]:
    a = p.anatomy
    low = [
        {"type": "segment", "id": s.id, "confidence": s.confidence, "bones": s.source_bones}
        for s in a.segments
        if s.confidence < 0.7
    ]
    low += [{"type": "joint", "id": j.id, "confidence": j.confidence} for j in a.joints if j.confidence < 0.7]
    low += [
        {"type": "bend", "id": j.id, "confidence": j.bend.confidence, "source": j.bend.source}
        for j in a.joints
        if j.bend and j.bend.confidence < 0.7
    ]
    return {
        "schema_version": "ambiguities/1",
        "unresolved_ambiguities": [x.model_dump() for x in a.ambiguities],
        "resolutions": [r.model_dump() for r in a.resolutions],
        "llm_interpretation": "not run (no OPENAI_API_KEY available); all ambiguities remain unresolved",
        "low_confidence_items": low,
        "prior_mismatches": [
            {"joint": j.id, "body_relative": j.bend.body_relative}
            for j in a.joints
            if j.bend and j.bend.matches_prior is False
        ],
        "body_frame_confidence": a.body_frame.confidence,
        "notes": a.notes,
    }
