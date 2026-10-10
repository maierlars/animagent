"""High-level access to a running Blender through the MCP transport.

Blender-side logic lives in `animagent/blender/scripts/*.py`. Each script defines
`main(args: dict) -> dict` and runs *inside* Blender through the official
`execute_blender_code` tool; arguments are injected as JSON and the return value
travels back as JSON.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from animagent.logging_setup import get_logger
from animagent.mcp_client import BlenderConnectionError, MCPTransport

log = get_logger(__name__)

EXECUTE_TOOL = "execute_blender_code"


class BlenderScriptError(RuntimeError):
    """Python code executed inside Blender raised an exception."""

    def __init__(self, script: str, message: str) -> None:
        super().__init__(f"Blender-side error in {script!r}:\n{message}")
        self.script = script
        self.blender_traceback = message


def load_script(name: str) -> str:
    return resources.files("animagent.blender.scripts").joinpath(f"{name}.py").read_text("utf-8")


def compose_script(name: str, args: dict[str, Any]) -> str:
    """Build the code string sent to Blender: injected args + script body + call."""
    payload = json.dumps(args)
    return (
        "import json as _animagent_json\n"
        f"ARGS = _animagent_json.loads({payload!r})\n"
        f"{load_script(name)}\n"
        "result = main(ARGS)\n"
    )


def unwrap_execute_response(script: str, response: dict[str, Any] | None, text: str) -> Any:
    if response is None:
        try:
            response = json.loads(text)
        except json.JSONDecodeError as ex:
            raise BlenderConnectionError(f"Unparseable response from Blender: {text[:300]}") from ex
    status = response.get("status")
    if status == "ok":
        return response.get("result")
    message = str(response.get("message", response))
    raise BlenderScriptError(script, message)


class BlenderSession:
    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport

    async def execute(self, code: str, *, label: str = "<inline>") -> Any:
        res = await self.transport.call_tool(EXECUTE_TOOL, {"code": code})
        if res.is_error:
            # Tool-level errors (as opposed to Blender-side exceptions) mean the MCP server
            # could not reach Blender, or rejected the call.
            raise BlenderConnectionError(f"{EXECUTE_TOOL} failed: {res.text.strip()[:500]}")
        value = unwrap_execute_response(label, res.structured, res.text)
        log.debug("blender.execute", script=label, elapsed_s=round(res.elapsed_s, 3), chars=len(res.text))
        return value

    async def run_script(self, name: str, args: dict[str, Any] | None = None) -> Any:
        return await self.execute(compose_script(name, args or {}), label=name)

    async def call_tool_json(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call one of the server's summary tools and unwrap its {status, result} envelope."""
        res = await self.transport.call_tool(name, arguments or {})
        if res.is_error:
            raise BlenderConnectionError(f"{name} failed: {res.text.strip()[:500]}")
        return unwrap_execute_response(name, res.structured, res.text)

    async def screenshot_view3d(self) -> bytes:
        res = await self.transport.call_tool("get_screenshot_of_area_as_image", {"area_ui_type": "VIEW_3D"})
        if res.is_error or not res.images:
            raise BlenderConnectionError(f"Screenshot failed: {res.text.strip()[:500]}")
        return res.images[0]
