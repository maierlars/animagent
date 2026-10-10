"""Runtime configuration, resolved from environment variables with sane defaults."""

from __future__ import annotations

import os
import shlex

from pydantic import BaseModel, Field

# The official Blender Lab MCP server, launched on demand via uvx.
DEFAULT_MCP_COMMAND = "uvx --from git+https://projects.blender.org/lab/blender_mcp.git#subdirectory=mcp blender-mcp"


class MCPServerConfig(BaseModel):
    """How to launch the official `blender-mcp` stdio server.

    `blender-mcp` itself connects to the Blender add-on's TCP socket, configured through
    BLENDER_MCP_HOST / BLENDER_MCP_PORT (defaults localhost:9876).
    """

    command: list[str] = Field(default_factory=lambda: shlex.split(DEFAULT_MCP_COMMAND))
    blender_host: str | None = None
    blender_port: int | None = None
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> MCPServerConfig:
        cmd = os.environ.get("ANIMAGENT_MCP_COMMAND")
        port = os.environ.get("BLENDER_MCP_PORT")
        return cls(
            command=shlex.split(cmd) if cmd else shlex.split(DEFAULT_MCP_COMMAND),
            blender_host=os.environ.get("BLENDER_MCP_HOST"),
            blender_port=int(port) if port else None,
            timeout_s=float(os.environ.get("ANIMAGENT_MCP_TIMEOUT", "300")),
        )

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.blender_host:
            env["BLENDER_MCP_HOST"] = self.blender_host
        if self.blender_port:
            env["BLENDER_MCP_PORT"] = str(self.blender_port)
        return env


class LLMConfig(BaseModel):
    model: str = "gpt-6"
    reasoning_effort: str | None = "medium"

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            model=os.environ.get("ANIMAGENT_LLM_MODEL", "gpt-6"),
            reasoning_effort=os.environ.get("ANIMAGENT_LLM_REASONING", "medium") or None,
        )
