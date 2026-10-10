# animagent

Agent-driven stylized animation for Rigify characters in Blender. Milestone 1: **rig inspection
and anatomical analysis** through the official Blender Lab MCP server.

## Connection procedure

1. Install the official Blender MCP add-on in Blender (≥ 5.1) from
   https://projects.blender.org/lab/blender_mcp (Blender Lab extension "MCP"). Enable it and start its
   server (preferences → MCP → Start, or auto-start). It listens on `localhost:9876`.
2. Have `uv` installed. The MCP server itself is launched on demand by animagent:
   `uvx --from git+https://projects.blender.org/lab/blender_mcp.git#subdirectory=mcp blender-mcp`
   (override with `ANIMAGENT_MCP_COMMAND`; the port with `BLENDER_MCP_PORT`).
3. `uv sync && uv run animagent doctor`

## Usage

```bash
uv run animagent doctor                                   # connectivity + capabilities
uv run animagent analyze-rig --armature Rig --output out/kyra_profile.json
#   -> out/kyra_profile.json  (rig_profile/1: extraction, skeleton, anatomy, controls, read-only proof)
#   -> out/kyra_profile_stages/00..05_*.json  (per-stage results, 05_report.json = compact report)
uv run animagent analyze-rig --from-extraction out/kyra_profile_stages/01_extraction.json --output p.json  # offline
uv run animagent report --profile out/kyra_profile.json
uv run animagent overlay show --profile out/kyra_profile.json   # debug drawing in the viewport
uv run animagent overlay clear
uv run animagent screenshot --profile out/kyra_profile.json --overlay --view front --output out/front.png
uv run animagent verify-readonly --armature Rig
uv run animagent analyze-rig --armature Rig --output p.json --llm   # GPT-6 for ambiguities (OPENAI_API_KEY)
```

Exit codes: 0 ok, 2 Blender/MCP unreachable, 3 scene changed during inspection.

## Tests

```bash
uv run pytest tests/unit                      # no Blender needed
uv run pytest -m blender                      # isolated background Blender + generated Rigify rig
ANIMAGENT_LIVE_TESTS=1 uv run pytest -m live  # your running Blender (read-only)
```

See `ARCHITECTURE.md` (design, data model, algorithms), `SPEC.md` (scope, acceptance),
`TASKS.md` (status, limitations, next milestone).
