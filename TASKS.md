# Tasks

## Milestone 1: rig inspection and anatomical analysis ✅

- [x] Inspect the official Blender MCP server (repo `dbbf836`, server 1.30.0, add-on 1.0.3, 26 tools); verify the live connection
- [x] uv project skeleton, structured logging (structlog), config via env
- [x] MCP client over stdio (official `mcp` SDK 2.x), tool listing, error normalisation
- [x] `animagent doctor`
- [x] Blender-side read-only extraction (`extract_rig`), versioned `RigExtraction`
- [x] State fingerprint and read-only proof around every analysis
- [x] Structural skeleton (Rigify ORG plus metarig parents / deform fallback, geometric repair, deform mapping)
- [x] Body frame, symmetry, limbs, trunk, tail, jaw, digits, appendages, joints, contacts, effectors
- [x] Bending directions (geometry, pole fallback, local-axis fallback, humanoid prior as evidence)
- [x] Mass guidance from skin-weighted area
- [x] Control mapping: FK/IK, poles, foot-roll, switches (driver evaluation plus pose verification), tweaks, torso/root, prop mounts
- [x] GPT-6 interpreter interface (Responses API, structured output), off by default
- [x] Compact report, per-stage artifacts, offline re-analysis (`--from-extraction`)
- [x] Viewport overlay (draw handlers) and screenshots with temporary framing
- [x] Unit tests (synthetic rigs, fake MCP server); integration tests (background Blender, generated Rigify rig); opt-in live test
- [x] Docs

## Known limitations / open questions

- Tested on a Rigify toucan (live) and a generated Rigify human (integration). **Kyra has not been
  analysed yet**; we need her .blend.
- The GPT-6 path is tested only with a mocked client (no `OPENAI_API_KEY` in this environment);
  the model name `gpt-6` is configurable (`ANIMAGENT_LLM_MODEL`).
- The control roles of trunk controls (fk/master/tweak/parent) are structural guesses. Rigify
  spine controls (hips/chest/torso/tweaks) need functional discovery (milestone 2) to rank reliably.
- Driver evaluation supports SUM/AVERAGE/MIN/MAX/simple scripted expressions plus keyframe and generator
  F-curves; other modifiers make the driver "unknown". The pose-observation check only
  discriminates when the IK and FK poses currently differ.
- Mass is surface-area based (not volume) and uses pre-modifier rest geometry.
- The overlay and screenshots need an interactive Blender (no viewport in background mode).
- The background integration bridge needs the official add-on installed in the user's Blender
  config (`~/.config/blender/<ver>/extensions/...`). The test launcher enables it per process.
- Very large rigs: extraction is one `execute_blender_code` call (~4 MB JSON for 700 bones). No
  chunking yet.
- The live toucan file shows `is_dirty=True`, but a fingerprint comparison against the on-disk file shows identical
  content in every category. On a freshly opened file the pipeline provably keeps `is_dirty=False` (integration test).

## Next: milestone 2, functional discovery (proposed)

- `probe` scripts: for each control and channel, apply a small perturbation in Blender,
  evaluate the depsgraph, measure the world-space motion of every structural bone and effector
  (and optionally sampled skin points), then restore exactly. The scene fingerprint must stay identical.
- Build a control → effect Jacobian/sensitivity map; verify and replace the structural role guesses
  (e.g. which control moves the pelvis without the chest).
- Measure joint limits and preferred bend empirically (IK pole sweeps), and switch-property effects.
- Staff representation: rigid prop with two grip frames attached to hand controls (prop mounts).
- Interfaces for the pose solver (`PoseTarget` → control values) defined against the measured map.
