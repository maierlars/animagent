"""MCP transport and CLI tests against a fake blender-mcp server over real stdio MCP."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import anyio
import pytest

from animagent.blender.session import BlenderScriptError, BlenderSession, compose_script
from animagent.cli import main
from animagent.config import MCPServerConfig
from animagent.mcp_client import BlenderConnectionError, BlenderMCPClient
from animagent.models.profile import RigProfile
from tests.fakes.synthetic import rigify_like_extraction

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_blender_mcp.py"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ex_path = tmp_path / "extraction.json"
    ex_path.write_text(rigify_like_extraction().model_dump_json())
    monkeypatch.setenv("FAKE_EXTRACTION", str(ex_path))
    monkeypatch.setenv("ANIMAGENT_MCP_COMMAND", f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE))}")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.delenv("FAKE_BLENDER_DOWN", raising=False)
    monkeypatch.delenv("FAKE_MUTATE", raising=False)
    return tmp_path


def test_compose_script_injects_args() -> None:
    code = compose_script("scene_info", {"x": "a'b\"c"})
    ns: dict = {}
    exec(code.split("\n", 2)[0] + "\n" + code.split("\n", 2)[1], ns)  # only the ARGS lines
    assert ns["ARGS"] == {"x": "a'b\"c"}
    assert code.rstrip().endswith("result = main(ARGS)")


@pytest.mark.timeout(60)
def test_client_lists_tools_and_executes(fake_env: Path) -> None:
    async def run() -> None:
        async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
            assert client.server_info and client.server_info.name == "blender-mcp-fake"
            tools = {t.name: t for t in await client.list_tools()}
            assert tools["execute_blender_code"].destructive is True
            assert tools["get_screenshot_of_area_as_image"].read_only is True
            session = BlenderSession(client)
            assert await session.execute("print(1)") == {"echo": 8}
            with pytest.raises(BlenderScriptError, match="boom"):
                await session.execute("FAKE_ERROR")
            png = await session.screenshot_view3d()
            assert png.startswith(b"\x89PNG")

    anyio.run(run)


@pytest.mark.timeout(60)
def test_unreachable_blender_raises_connection_error(fake_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_BLENDER_DOWN", "1")

    async def run() -> None:
        async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
            with pytest.raises(BlenderConnectionError, match="Cannot connect"):
                await BlenderSession(client).execute("x")

    anyio.run(run)


@pytest.mark.timeout(60)
def test_bad_server_command_raises_connection_error() -> None:
    async def run() -> None:
        async with BlenderMCPClient(
            MCPServerConfig(command=[sys.executable, "-c", "import sys; sys.exit(3)"], timeout_s=5)
        ):
            pass

    with pytest.raises(BlenderConnectionError):
        anyio.run(run)


@pytest.mark.timeout(60)
def test_cli_doctor(fake_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "ok"
    assert rep["blender"]["blender_version"] == "5.2.2 LTS"
    assert {t["name"] for t in rep["tools"]} >= {"execute_blender_code", "get_screenshot_of_area_as_image"}


@pytest.mark.timeout(60)
def test_cli_doctor_reports_unreachable_blender(fake_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_BLENDER_DOWN", "1")
    assert main(["doctor"]) == 2


@pytest.mark.timeout(60)
def test_cli_analyze_rig_end_to_end(fake_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = fake_env / "kyra_profile.json"
    assert main(["analyze-rig", "--armature", "Rig", "--output", str(out)]) == 0
    p = RigProfile.load(out)
    assert p.readonly_check and p.readonly_check.identical
    assert {lb.id for lb in p.anatomy.limbs} >= {"arm.L", "leg.R"}
    stages = fake_env / "kyra_profile_stages"
    assert (stages / "05_report.json").exists()
    assert "limb arm.L" in capsys.readouterr().out


@pytest.mark.timeout(60)
def test_cli_analyze_rig_detects_scene_mutation(fake_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MUTATE", "1")
    out = fake_env / "p.json"
    assert main(["analyze-rig", "--armature", "Rig", "--output", str(out)]) == 3
    p = RigProfile.load(out)
    assert p.readonly_check and p.readonly_check.changed_categories == ["pose"]


@pytest.mark.timeout(60)
def test_cli_analyze_unknown_armature_fails(fake_env: Path) -> None:
    with pytest.raises(BlenderScriptError, match="No object named"):
        main(["analyze-rig", "--armature", "Nope", "--output", str(fake_env / "x.json")])


def test_cli_offline_analysis_and_report(fake_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = fake_env / "p.json"
    assert (
        main(
            [
                "analyze-rig",
                "--from-extraction",
                str(fake_env / "extraction.json"),
                "--output",
                str(out),
                "--no-artifacts",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["report", "--profile", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["root_control"] == "root"


@pytest.mark.timeout(60)
def test_cli_screenshot(fake_env: Path) -> None:
    out = fake_env / "shot.png"
    assert main(["screenshot", "--output", str(out)]) == 0
    assert out.read_bytes().startswith(b"\x89PNG")
