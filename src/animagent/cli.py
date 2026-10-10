"""`animagent` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import anyio

from animagent import __version__
from animagent.config import LLMConfig, MCPServerConfig
from animagent.logging_setup import configure_logging, get_logger
from animagent.mcp_client import BlenderConnectionError, BlenderMCPClient

log = get_logger("animagent.cli")

REQUIRED_TOOLS = ("execute_blender_code",)
OPTIONAL_TOOLS = ("get_screenshot_of_area_as_image", "get_blendfile_summary_path_info", "get_objects_summary")


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=1, default=str))


# ---------------------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------------------


async def _doctor(args: argparse.Namespace) -> int:
    from animagent.blender.session import BlenderSession

    cfg = MCPServerConfig.from_env()
    report: dict[str, Any] = {"animagent": __version__, "mcp_command": cfg.command, "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "ok": ok, "detail": detail})

    t0 = time.monotonic()
    try:
        async with BlenderMCPClient(cfg) as client:
            report["connect_s"] = round(time.monotonic() - t0, 2)
            info = client.server_info
            report["server"] = info.model_dump() if info else None
            check(
                "mcp_server_started",
                True,
                f"{info.name} {info.version}, protocol {info.protocol_version}" if info else "",
            )
            tools = await client.list_tools()
            names = {t.name for t in tools}
            report["tools"] = [
                {
                    "name": t.name,
                    "read_only": t.read_only,
                    "destructive": t.destructive,
                    "args": sorted(t.input_schema.get("properties", {}).keys()),
                    "summary": t.description.split("\n")[0],
                }
                for t in tools
            ]
            for t in REQUIRED_TOOLS:
                check(f"tool:{t}", t in names, "required")
            for t in OPTIONAL_TOOLS:
                check(f"tool:{t}", t in names, "optional")
            session = BlenderSession(client)
            try:
                t1 = time.monotonic()
                scene = await session.run_script("scene_info", {})
                report["blender_roundtrip_s"] = round(time.monotonic() - t1, 3)
                report["blender"] = scene
                check("blender_reachable", True, f"Blender {scene['blender_version']}")
                check("numpy_in_blender", bool(scene.get("numpy")), str(scene.get("numpy")))
                check("rigify_addon", scene["rigify_addon"]["loaded_state"], "needed only for metarig lookup")
                rigs = [a for a in scene["armatures"] if a["rig_id"]]
                check("rigify_rig_in_scene", bool(rigs), ", ".join(a["name"] for a in rigs) or "none")
            except BlenderConnectionError as ex:
                check("blender_reachable", False, str(ex))
    except BlenderConnectionError as ex:
        check("mcp_server_started", False, str(ex))

    ok = all(
        c["ok"]
        for c in report["checks"]
        if c["check"] in ("mcp_server_started", "blender_reachable", *(f"tool:{t}" for t in REQUIRED_TOOLS))
    )
    report["status"] = "ok" if ok else "error"
    if args.json:
        _print_json(report)
    else:
        print(f"animagent {__version__} doctor")
        print(f"  MCP command : {' '.join(cfg.command)}")
        if "server" in report and report["server"]:
            s = report["server"]
            print(
                f"  MCP server  : {s['name']} {s['version']} (protocol {s['protocol_version']}), "
                f"connected in {report.get('connect_s')}s"
            )
        if "blender" in report:
            b = report["blender"]
            print(
                f"  Blender     : {b['blender_version']} ({'background' if b['background'] else 'interactive'}), "
                f"python {b['python_version']}, numpy {b['numpy']}, roundtrip {report['blender_roundtrip_s']}s"
            )
            print(f"  File        : {b['file_path'] or '<unsaved>'}{' (unsaved changes)' if b['is_dirty'] else ''}")
            print(
                f"  Scene       : {b['scene']}, frame {b['frame_current']}, mode {b['mode']}, "
                f"{b['object_count']} objects"
            )
            for a in b["armatures"]:
                kind = "rigify-generated" if a["rig_id"] else ("metarig" if a["is_metarig"] else "armature")
                print(
                    f"    - {a['name']}: {a['bone_count']} bones ({kind}; {a['deform_bones']} deform), "
                    f"meshes: {', '.join(a['deformed_meshes']) or '-'}"
                )
        if "tools" in report:
            print(f"  MCP tools ({len(report['tools'])}):")
            for t in report["tools"]:
                flag = "ro" if t["read_only"] else ("DESTRUCTIVE" if t["destructive"] else "")
                print(f"    - {t['name']}({', '.join(t['args'])}) {flag}")
        print("  Checks:")
        for c in report["checks"]:
            print(f"    [{'ok' if c['ok'] else '!!'}] {c['check']}: {c['detail']}")
        print(f"  Status: {report['status'].upper()}")
    return 0 if ok else 2


# ---------------------------------------------------------------------------------------
# extract / analyze
# ---------------------------------------------------------------------------------------


async def _extract(args: argparse.Namespace) -> int:
    from animagent.blender.session import BlenderSession
    from animagent.pipeline import extract

    async with BlenderMCPClient() as client:
        ex = await extract(BlenderSession(client), args.armature)
    Path(args.output).write_text(ex.model_dump_json(indent=1))
    print(f"Wrote {args.output}: {len(ex.bones)} bones, {len(ex.meshes)} meshes, {len(ex.drivers)} drivers")
    return 0


def _make_interpreter(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    if not getattr(args, "llm", False):
        return None
    from animagent.agent.interpreter import OpenAIAnatomyInterpreter

    cfg = LLMConfig.from_env()
    if args.llm_model:
        cfg.model = args.llm_model
    return OpenAIAnatomyInterpreter(cfg)


def _summarize(profile, out: Path, artifacts: dict[str, Path] | None) -> None:  # type: ignore[no-untyped-def]
    a, cm = profile.anatomy, profile.controls
    f = a.body_frame
    print(f"Profile written to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if artifacts:
        print(f"Stage artifacts in {next(iter(artifacts.values())).parent}/: {', '.join(sorted(artifacts))}")
    print(
        f"  armature   : {profile.source.armature} ({len(profile.extraction.bones)} bones, "
        f"{cm.rig_kind}, skeleton={profile.skeleton.strategy})"
    )
    print(f"  body frame : up={f.up} forward={f.forward} height={f.height:.3f} (confidence {f.confidence})")
    for lb in a.limbs:
        if lb.kind.value == "appendage":
            continue
        lc = next((x for x in cm.limbs if x.limb_id == lb.id), None)
        ctrl = ""
        if lc:
            sw = lc.ik_fk_switch
            ctrl = (
                f" | FK {[c.bone for c in lc.fk_chain]} IK {lc.ik_effector.bone if lc.ik_effector else '-'}"
                f" pole {lc.ik_pole.bone if lc.ik_pole else '-'}"
                + (f" switch {sw.bone}[{sw.property}] IK={sw.ik_value:g}" if sw and sw.ik_value is not None else "")
            )
        print(f"  limb {lb.id:<8}: {' > '.join(lb.segments)} ({lb.confidence}){ctrl}")
    print(
        f"  segments   : {len(a.segments)}, joints: {len(a.joints)}, ambiguities: {len(a.ambiguities)}, "
        f"resolutions: {len(a.resolutions)}"
    )
    if profile.readonly_check:
        rc = profile.readonly_check
        print(f"  read-only  : {'scene unchanged' if rc.identical else 'CHANGED: ' + ', '.join(rc.changed_categories)}")
    for d in profile.diagnostics:
        print(f"  {d.level:<7}: [{d.stage}] {d.message}")


async def _analyze(args: argparse.Namespace) -> int:
    from animagent.models.rig import RigExtraction
    from animagent.pipeline import analyze, analyze_rig, save_artifacts

    interpreter = _make_interpreter(args)
    if args.from_extraction:
        ex = RigExtraction.model_validate_json(Path(args.from_extraction).read_text())
        profile = analyze(ex, interpreter)
    else:
        from animagent.blender.session import BlenderSession

        async with BlenderMCPClient() as client:
            profile = await analyze_rig(BlenderSession(client), args.armature, interpreter)
    out = Path(args.output)
    profile.save(out)
    artifacts = None
    if not args.no_artifacts:
        art_dir = Path(args.artifacts) if args.artifacts else out.with_name(out.stem + "_stages")
        artifacts = save_artifacts(profile, art_dir)
    _summarize(profile, out, artifacts)
    if profile.readonly_check and not profile.readonly_check.identical:
        return 3
    return 0


def _report(args: argparse.Namespace) -> int:
    from animagent.analysis.report import build_report
    from animagent.models.profile import RigProfile

    rep = build_report(RigProfile.load(Path(args.profile)))
    if args.output:
        Path(args.output).write_text(json.dumps(rep, indent=1))
        print(f"Wrote {args.output}")
    else:
        _print_json(rep)
    return 0


# ---------------------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------------------


async def _overlay(args: argparse.Namespace) -> int:
    from animagent.blender.session import BlenderSession

    async with BlenderMCPClient() as client:
        session = BlenderSession(client)
        if args.action == "show":
            from animagent.models.profile import RigProfile
            from animagent.visualize import overlay_spec

            spec = overlay_spec(RigProfile.load(Path(args.profile)), labels=not args.no_labels, pose=args.pose)
            res = await session.run_script("overlay", {"action": "show", "spec": spec})
        else:
            res = await session.run_script("overlay", {"action": args.action})
    _print_json(res)
    return 0


async def _screenshot(args: argparse.Namespace) -> int:
    from animagent.blender.session import BlenderSession

    profile = None
    if args.profile:
        from animagent.models.profile import RigProfile

        profile = RigProfile.load(Path(args.profile))
    async with BlenderMCPClient() as client:
        session = BlenderSession(client)
        added_overlay = False
        framed = False
        try:
            if profile is not None and args.overlay:
                from animagent.visualize import overlay_spec

                spec = overlay_spec(profile, labels=not args.no_labels, pose=args.pose)
                await session.run_script("overlay", {"action": "show", "spec": spec})
                added_overlay = True
            if args.view != "current":
                if profile is None:
                    raise SystemExit("--view requires --profile (for the body frame)")
                from animagent.visualize import framing

                await session.run_script("viewport", {"action": "frame", "view": args.view, **framing(profile)})
                framed = True
            png = await session.screenshot_view3d()
        finally:
            if framed and not args.keep_view:
                await session.run_script("viewport", {"action": "restore"})
            if added_overlay and not args.keep_overlay:
                await session.run_script("overlay", {"action": "clear"})
    Path(args.output).write_bytes(png)
    print(f"Wrote {args.output} ({len(png) / 1024:.0f} KB)")
    return 0


async def _verify_readonly(args: argparse.Namespace) -> int:
    from animagent.blender.session import BlenderSession
    from animagent.models.profile import ReadOnlyCheck
    from animagent.pipeline import extract, fingerprint

    async with BlenderMCPClient() as client:
        s = BlenderSession(client)
        before = await fingerprint(s, args.armature)
        await extract(s, args.armature)
        after = await fingerprint(s, args.armature)
    rc = ReadOnlyCheck.compare(before, after)
    _print_json(
        {
            "identical": rc.identical,
            "changed": rc.changed_categories,
            "hashes": after.hashes,
            "is_dirty_before": before.info.get("is_dirty"),
            "is_dirty_after": after.info.get("is_dirty"),
        }
    )
    return 0 if rc.identical else 3


# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animagent", description="AI animation agent for Rigify characters in Blender.")
    p.add_argument("--version", action="version", version=f"animagent {__version__}")
    p.add_argument("--log-level", default="WARNING", help="DEBUG, INFO, WARNING (default), ERROR")
    p.add_argument("--json-logs", action="store_true", help="emit structured JSON logs on stderr")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check MCP/Blender connectivity and capabilities")
    d.add_argument("--json", action="store_true")

    e = sub.add_parser("extract-rig", help="raw read-only extraction of an armature")
    e.add_argument("--armature", required=True)
    e.add_argument("--output", required=True)

    a = sub.add_parser("analyze-rig", help="extract + anatomical analysis + control mapping")
    a.add_argument("--armature", help="armature object name (required unless --from-extraction)")
    a.add_argument("--output", required=True, help="rig profile JSON")
    a.add_argument("--artifacts", help="directory for per-stage results (default: <output>_stages)")
    a.add_argument("--no-artifacts", action="store_true")
    a.add_argument("--from-extraction", help="analyse a saved extraction offline (no Blender)")
    a.add_argument("--llm", action="store_true", help="ask GPT-6 to interpret ambiguities (needs OPENAI_API_KEY)")
    a.add_argument("--llm-model", help="override model (default: $ANIMAGENT_LLM_MODEL or gpt-6)")

    r = sub.add_parser("report", help="compact JSON report from a profile")
    r.add_argument("--profile", required=True)
    r.add_argument("--output")

    o = sub.add_parser("overlay", help="debug overlay in the Blender viewport (draw handlers, no scene data)")
    o.add_argument("action", choices=["show", "clear", "status"])
    o.add_argument("--profile")
    o.add_argument("--no-labels", action="store_true")
    o.add_argument(
        "--pose",
        choices=["current", "rest"],
        default="current",
        help="draw at the pose captured in the profile (default) or at rest",
    )

    s = sub.add_parser("screenshot", help="capture the 3D viewport (optionally framed, with overlay)")
    s.add_argument("--output", required=True)
    s.add_argument("--profile")
    s.add_argument("--overlay", action="store_true", help="draw the anatomy overlay for the capture")
    s.add_argument("--no-labels", action="store_true")
    s.add_argument("--pose", choices=["current", "rest"], default="current")
    s.add_argument("--view", default="current", choices=["current", "front", "back", "left", "right", "top", "persp"])
    s.add_argument("--keep-view", action="store_true", help="do not restore the previous viewport")
    s.add_argument("--keep-overlay", action="store_true")

    v = sub.add_parser("verify-readonly", help="prove that extraction leaves the scene unchanged")
    v.add_argument("--armature", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.json_logs)
    try:
        if args.command == "doctor":
            return anyio.run(_doctor, args)
        if args.command == "extract-rig":
            return anyio.run(_extract, args)
        if args.command == "analyze-rig":
            if not args.armature and not args.from_extraction:
                print("error: --armature is required unless --from-extraction is given", file=sys.stderr)
                return 1
            return anyio.run(_analyze, args)
        if args.command == "report":
            return _report(args)
        if args.command == "overlay":
            if args.action == "show" and not args.profile:
                print("error: overlay show requires --profile", file=sys.stderr)
                return 1
            return anyio.run(_overlay, args)
        if args.command == "screenshot":
            return anyio.run(_screenshot, args)
        if args.command == "verify-readonly":
            return anyio.run(_verify_readonly, args)
    except BlenderConnectionError as ex:
        print(f"error: cannot reach Blender via MCP: {ex}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
