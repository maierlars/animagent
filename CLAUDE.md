# CLAUDE.md

Agent-driven stylized fighting-game animation for Rigify characters in Blender.
Product spec: `SPEC` (original brief) and `SPEC.md` (milestones). Design: `ARCHITECTURE.md`. Status: `TASKS.md`.

## Rules
- Standalone Python app. Blender is reached **only** through the official Blender Lab MCP server
  (`blender-mcp`, launched by our own MCP client). No custom add-on, no Blender UI, no own MCP server.
- **Never modify or save the user's open .blend.** Inspection must be read-only. Every analysis is
  bracketed by `state_fingerprint`; keep it that way. Probing (later milestones) must restore state exactly.
- Blender-side code lives in `src/animagent/blender/scripts/*.py` (`main(args) -> dict`, JSON only,
  no operators in read-only scripts). Do not import these modules from Python; they run inside Blender.
- Keep analysis deterministic and Blender-free (`analysis/*` operates on `RigExtraction`).
  The LLM (GPT-6, OpenAI Responses API) only resolves `Ambiguity` records; it never overwrites
  deterministic data. Send compact summaries only, never meshes, vertices or weights.
- Anatomy (`models/anatomy.py`) must stay rig-independent; Rigify specifics belong in
  `analysis/controls.py` / `analysis/skeleton.py`. Names are weak evidence; record them as `Evidence`.
- Pydantic models are versioned (`schema_version`). Bump the version on breaking changes.
- Do not implement choreography, pose solving or curve generation before their milestone.

## Commands
- `uv sync`, `uv run animagent doctor`
- `uv run animagent analyze-rig --armature <name> --output out/profile.json`
- `uv run pytest tests/unit` (fast, no Blender) · `uv run pytest -m blender` (background Blender)
- `ANIMAGENT_LIVE_TESTS=1 uv run pytest -m live` (running Blender, read-only)
- `uv run ruff check src tests && uv run ruff format src tests`

## Environment
- `BLENDER_MCP_PORT`/`BLENDER_MCP_HOST`: add-on socket (default localhost:9876).
- `ANIMAGENT_MCP_COMMAND`: override the server command (default `uvx --from git+…/blender_mcp.git#subdirectory=mcp blender-mcp`).
- `OPENAI_API_KEY`, `ANIMAGENT_LLM_MODEL` (default `gpt-6`) for `--llm`.
- Background test Blender: the add-on must be installed (it's enabled per process by the test launcher).
