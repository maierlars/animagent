"""Async MCP client speaking stdio to the official `blender-mcp` server.

This is the only module that knows about the MCP protocol. Everything above it uses the
`MCPTransport` protocol, which tests replace with fakes.
"""

from __future__ import annotations

import base64
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from animagent.config import MCPServerConfig
from animagent.logging_setup import get_logger

log = get_logger(__name__)


class BlenderConnectionError(RuntimeError):
    """The MCP server or the Blender add-on behind it is unreachable."""


class ToolInfo(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    read_only: bool | None = None
    destructive: bool | None = None


class ServerInfo(BaseModel):
    name: str
    version: str
    protocol_version: str
    instructions_excerpt: str = ""


class ToolCallResult(BaseModel):
    """Normalised MCP tool call result."""

    is_error: bool = False
    structured: dict[str, Any] | None = None
    text: str = ""
    images: list[bytes] = Field(default_factory=list)
    elapsed_s: float = 0.0


class MCPTransport(Protocol):
    server_info: ServerInfo | None

    async def list_tools(self) -> list[ToolInfo]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult: ...


class BlenderMCPClient:
    """Launches `blender-mcp` as a subprocess and talks MCP over its stdio.

    Usage::

        async with BlenderMCPClient(MCPServerConfig.from_env()) as client:
            tools = await client.list_tools()
    """

    def __init__(self, config: MCPServerConfig | None = None, stderr_path: str | None = None) -> None:
        self.config = config or MCPServerConfig.from_env()
        self.server_info: ServerInfo | None = None
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._stderr_path = stderr_path or os.environ.get("ANIMAGENT_MCP_STDERR", os.devnull)

    async def __aenter__(self) -> BlenderMCPClient:
        if not self.config.command:
            raise BlenderConnectionError("Empty MCP server command")
        params = StdioServerParameters(
            command=self.config.command[0],
            args=self.config.command[1:],
            env=self.config.server_env(),
        )
        stack = AsyncExitStack()
        try:
            errlog = stack.enter_context(open(self._stderr_path, "a"))  # noqa: SIM115
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=self.config.timeout_s)
            )
            t0 = time.monotonic()
            init = await session.initialize()
            log.debug("mcp.initialized", elapsed_s=round(time.monotonic() - t0, 3))
        except BaseException as ex:
            await stack.aclose()
            if isinstance(ex, Exception):
                raise BlenderConnectionError(
                    f"Failed to start/initialise MCP server {self.config.command!r}: {ex}"
                ) from ex
            raise
        self._stack = stack
        self._session = session
        self.server_info = ServerInfo(
            name=init.server_info.name,
            version=init.server_info.version,
            protocol_version=str(init.protocol_version),
            instructions_excerpt=(init.instructions or "")[:200],
        )
        log.info("mcp.connected", server=self.server_info.name, version=self.server_info.version)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise BlenderConnectionError("MCP client is not connected (use `async with`)")
        return self._session

    async def list_tools(self) -> list[ToolInfo]:
        tools: list[ToolInfo] = []
        cursor: str | None = None
        while True:
            res = await self.session.list_tools(cursor=cursor) if cursor else await self.session.list_tools()
            for t in res.tools:
                ann = t.annotations
                tools.append(
                    ToolInfo(
                        name=t.name,
                        description=(t.description or "").strip(),
                        input_schema=dict(t.input_schema or {}),
                        read_only=getattr(ann, "read_only_hint", None) if ann else None,
                        destructive=getattr(ann, "destructive_hint", None) if ann else None,
                    )
                )
            cursor = getattr(res, "next_cursor", None)
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        t0 = time.monotonic()
        res = await self.session.call_tool(name, arguments or {})
        elapsed = time.monotonic() - t0
        if not isinstance(res, mcp_types.CallToolResult):
            raise BlenderConnectionError(f"Unexpected MCP result type for {name}: {type(res).__name__}")
        texts: list[str] = []
        images: list[bytes] = []
        for block in res.content:
            if isinstance(block, mcp_types.TextContent):
                texts.append(block.text)
            elif isinstance(block, mcp_types.ImageContent):
                images.append(base64.b64decode(block.data))
        out = ToolCallResult(
            is_error=bool(res.is_error),
            structured=res.structured_content,
            text="\n".join(texts),
            images=images,
            elapsed_s=elapsed,
        )
        log.debug("mcp.call_tool", tool=name, is_error=out.is_error, elapsed_s=round(elapsed, 3))
        return out
