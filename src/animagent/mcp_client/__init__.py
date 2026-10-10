"""MCP transport: a thin client for the official Blender Lab MCP server."""

from animagent.mcp_client.client import (
    BlenderConnectionError,
    BlenderMCPClient,
    MCPTransport,
    ServerInfo,
    ToolCallResult,
    ToolInfo,
)

__all__ = [
    "BlenderConnectionError",
    "BlenderMCPClient",
    "MCPTransport",
    "ServerInfo",
    "ToolCallResult",
    "ToolInfo",
]
