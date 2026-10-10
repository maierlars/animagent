from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest

from animagent.blender.session import BlenderSession
from animagent.config import MCPServerConfig
from animagent.mcp_client import BlenderMCPClient
from tests.integration.blender_harness import BUILD_FIXTURE, background_blender, blender_binary


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "live" in item.keywords and not os.environ.get("ANIMAGENT_LIVE_TESTS"):
            item.add_marker(pytest.mark.skip(reason="set ANIMAGENT_LIVE_TESTS=1 to test the running Blender"))
        if "blender" in item.keywords and blender_binary() is None:
            item.add_marker(pytest.mark.skip(reason="blender binary not found"))


def run_session(port: int | None, fn):  # type: ignore[no-untyped-def]
    async def go():  # type: ignore[no-untyped-def]
        cfg = MCPServerConfig.from_env()
        if port is not None:
            cfg.blender_port = port
        async with BlenderMCPClient(cfg) as client:
            return await fn(BlenderSession(client))

    return anyio.run(go)


@pytest.fixture(scope="session")
def rigify_blend(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated Rigify human + skinned mesh, saved to a temporary .blend file."""
    path = tmp_path_factory.mktemp("blend") / "rigify_human.blend"
    with background_blender() as port:
        res = run_session(port, lambda s: s.execute(f"ARGS_PATH = {str(path)!r}\n" + BUILD_FIXTURE))
    assert res["rig"] == "Rig" and path.exists()
    return path


@pytest.fixture(scope="session")
def rigify_port(rigify_blend: Path) -> Iterator[int]:
    """Background Blender with the saved fixture freshly opened (clean, not dirty)."""
    with background_blender(rigify_blend) as port:
        yield port
