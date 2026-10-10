"""A fake `blender-mcp` stdio server for tests (mocked Blender responses over real MCP).

It recognises the animagent Blender-side scripts by their content and returns canned results.
Environment:
  FAKE_EXTRACTION   path to a JSON RigExtraction returned by `extract_rig`
  FAKE_BLENDER_DOWN when set, every Blender call fails like an unreachable Blender
  FAKE_MUTATE       when set, the second fingerprint differs (simulates a scene change)
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

server = MCPServer("blender-mcp-fake")
_calls = {"fingerprint": 0}
# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _guard() -> None:
    if os.environ.get("FAKE_BLENDER_DOWN"):
        raise ToolError("Cannot connect to Blender at localhost:9876. Ensure Blender is running.")


@server.tool(annotations=ToolAnnotations(title="Execute Python Code", destructiveHint=True))
def execute_blender_code(code: str) -> dict[str, Any]:
    """Execute Python code in the connected Blender instance."""
    _guard()
    if "FAKE_ERROR" in code:
        return {"status": "error", "message": "Traceback (most recent call last):\nValueError: boom\n"}
    if 'SCHEMA_VERSION = "rig_extraction/1"' in code:
        args = json.loads(code.split("_animagent_json.loads(", 1)[1].split(")\n", 1)[0].strip("'\""))
        data = json.loads(open(os.environ["FAKE_EXTRACTION"]).read())
        if args.get("armature") != data["armature"]["name"]:
            return {"status": "error", "message": f"ValueError: No object named {args.get('armature')!r}"}
        return {"status": "ok", "result": data}
    if "def _datablock_hash" in code:
        _calls["fingerprint"] += 1
        changed = os.environ.get("FAKE_MUTATE") and _calls["fingerprint"] > 1
        return {
            "status": "ok",
            "result": {"hashes": {"pose": "b" if changed else "a", "meshes": "m"}, "info": {"is_dirty": False}},
        }
    if 'addon_utils.check("rigify")' in code:
        return {
            "status": "ok",
            "result": {
                "blender_version": "5.2.2 LTS",
                "blender_build_hash": "fake",
                "python_version": "3.13.0",
                "background": True,
                "file_path": "/tmp/fake.blend",
                "is_dirty": False,
                "scene": "Scene",
                "frame_current": 1,
                "mode": "OBJECT",
                "object_count": 3,
                "numpy": "2.0",
                "rigify_addon": {"loaded_default": True, "loaded_state": True},
                "armatures": [
                    {
                        "name": "Rig",
                        "bone_count": 10,
                        "rig_id": "x",
                        "org_bones": 5,
                        "def_bones": 5,
                        "deform_bones": 5,
                        "is_metarig": False,
                        "visible": True,
                        "deformed_meshes": ["Body"],
                    }
                ],
            },
        }
    if "animagent_overlay" in code or "animagent_saved_view" in code:
        return {"status": "ok", "result": {"status": "fake"}}
    return {"status": "ok", "result": {"echo": len(code)}}


@server.tool(annotations=ToolAnnotations(title="Get Area Screenshot", readOnlyHint=True))
def get_screenshot_of_area_as_image(area_ui_type: str, size_limit_in_bytes: int = 0) -> Image:
    """Take a screenshot of a single Blender area and return it as a PNG image."""
    _guard()
    return Image(data=_PNG, format="png")


@server.tool(annotations=ToolAnnotations(title="Path info", readOnlyHint=True))
def get_blendfile_summary_path_info() -> dict[str, Any]:
    """Simple/fast access to the blend file's path."""
    return {"status": "ok", "result": {"filepath": "/tmp/fake.blend"}}


if __name__ == "__main__":
    server.run()
