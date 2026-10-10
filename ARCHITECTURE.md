# Architecture

`animagent` is a standalone Python application. It drives a running Blender through the
**official Blender Lab MCP server** (`blender-mcp`, https://projects.blender.org/lab/blender_mcp),
which it launches and talks to with its own MCP client. Milestone 1 covers rig inspection
and anatomical analysis only.

```
animagent CLI ──► pipeline ──► BlenderSession ──► BlenderMCPClient ──MCP/stdio──► blender-mcp ──TCP──► Blender add-on ──► bpy
                    │                 │  (scripts/*.py run inside Blender)
                    ▼
       skeleton ► anatomy ► controls ► report / overlay        (pure Python, deterministic, no Blender needed)
                    ▲
       agent.interpreter (GPT-6, optional; resolves ambiguities only)
```

## Subsystems and modules

| Subsystem | Module(s) | Notes |
|---|---|---|
| MCP transport | `mcp_client/client.py` | The only module that knows MCP. Spawns `blender-mcp` over stdio (official `mcp` SDK, 2.x), lists tools, calls tools and normalises results (`ToolCallResult`). `MCPTransport` protocol for fakes. |
| Blender communication | `blender/session.py` | `BlenderSession.run_script(name, args)` sends `blender/scripts/<name>.py` through the `execute_blender_code` tool with JSON-injected `ARGS`, then unwraps `{status, result}`. Errors split into `BlenderConnectionError` (cannot reach Blender) and `BlenderScriptError` (exception inside Blender). |
| Blender-side extraction/manipulation | `blender/scripts/*.py` | Run **inside** Blender. `extract_rig` (read-only extraction), `state_fingerprint` (hashes for the reversibility proof), `scene_info` (doctor), `overlay` (GPU draw-handler overlay), `viewport` (temporary framing + restore). |
| Rig/anatomy representation | `models/rig.py`, `models/anatomy.py`, `models/controls.py`, `models/profile.py` | Versioned Pydantic models (`rig_extraction/1`, `skeleton/1`, `anatomy/1`, `controls/1`, `rig_profile/1`). |
| Structural analysis | `analysis/skeleton.py` | Clean structural skeleton: Rigify ORG bones (parents from the metarig when present) or deform bones for other rigs; geometric repair; deform bones mapped onto structural bones by containment. |
| Anatomical interpretation | `analysis/body_frame.py`, `analysis/anatomy.py`, `analysis/bending.py`, `analysis/mass.py` | Rig-independent: works only on skeleton geometry and topology. Names are used for L/R sides and as weak, recorded hints. |
| Rig-specific control mapping | `analysis/controls.py` | Dependency graph (parents, constraint targets, IK chains, transform-reading drivers) to FK/IK controls, poles, IK/FK switches, tweaks, torso/root, prop mounts. |
| GPT-6 orchestration | `agent/interpreter.py` | `AnatomyInterpreter` protocol; `OpenAIAnatomyInterpreter` uses the OpenAI **Responses API** with structured output (`responses.parse`, Pydantic schema). Off by default (`--llm`). |
| Validation & feedback | `models/profile.ReadOnlyCheck`, `pipeline.py`, tests | Fingerprint before/after every analysis; diagnostics in the profile. |
| Visualization | `visualize.py` + `scripts/overlay.py`, `scripts/viewport.py` | Overlay spec built in Python, drawn in Blender via `SpaceView3D.draw_handler_add` (no datablocks). Screenshots via the official `get_screenshot_of_area_as_image` tool. |
| Choreography, pose solving, curves | (not yet) | Deliberately absent in milestone 1. They will consume `RigProfile` (anatomy plus control mapping). |

## Pipeline (`analyze-rig`)

1. `state_fingerprint` (before): SHA-256 hashes of pose, rest pose, animation (keyframes,
   drivers), objects, mesh geometry, vertex weights, datablock inventory and UI state.
2. `extract_rig`: read-only extraction into `RigExtraction` (≈2 s / 600 bones).
3. `state_fingerprint` (after), then `ReadOnlyCheck.compare`. Any difference is an error diagnostic and exit code 3.
4. `build_skeleton` → `analyze_anatomy` → (optional GPT-6 `resolve`) → `map_controls`.
5. Results: `RigProfile` (everything) plus per-stage artifacts `NN_*.json` and the compact `05_report.json`.

Stages 4–5 are pure functions of the extraction (`analyze-rig --from-extraction raw.json`),
so analysis can be rerun or debugged offline without Blender.

## Key algorithms (deterministic)

* **Body frame**: up = world +Z (validated: the lowest bones are sided, i.e. feet).
  Left = mean L−R offset of mirrored bone pairs (symmetry error reported). Forward = left × up.
  Cross-checked with foot direction (ankle → toe tip); a contradiction lowers confidence and
  raises a `forward_direction` ambiguity.
* **Limbs**: sided subtrees attached to the centre line. A subtree that reaches the ground with
  ≥3 bones is a leg; the longest multi-bone non-ground chain per side above the legs is an arm;
  everything else is an appendage (an ambiguity for the interpreter). Chains follow *connected*
  children first. The upper/lower split is the longest consecutive bone pair; a preceding
  bone is a clavicle; following bones are hand/foot/toe; hand/toe subtrees are digits (shared
  bones go to the palm).
* **Trunk**: pelvis = LCA(leg attachments), chest = LCA(arm attachments), spine = the path between.
  Neck/head continue the centre chain only through children that start near the current tail,
  up to a terminal or branching bone (the head). Centre chains pointing back/down from the pelvis
  or spine are tails.
* **Bending**: for upper/lower segments, the joint apex offset from the root–end line gives
  `joint_points_toward`; `hinge_axis = points_toward × lower_dir`, so positive rotation increases flexion.
  Fallbacks for straight chains: IK pole side, then local-X convention, with lower confidence.
  Each hinge axis is also expressed as the closest local axis of the child bone (Rigify limbs: ±X).
  A humanoid prior (knees forward, elbows back) is evidence only: a mismatch raises an
  ambiguity and never overrides the measurement (e.g. digitigrade legs).
* **Mass**: skin-weighted surface area per deform bone, aggregated per segment, gives relative
  mass and centre of mass. This is guidance only; weights are not masses.
* **Controls**: BFS from each structural bone through constraint targets, IK targets/poles, drivers
  that read bone transforms, and mechanism parents, until controls are reached. IK/FK switches
  are found from drivers on constraint influences; their semantics come from **evaluating
  the driver**, including its F-curve generator modifiers, at the property's min and max. The result
  is then **verified against the evaluated pose**: the ORG bone must coincide with the predicted
  source. Observation wins over the driver model. Effector = topmost control in the IK
  target's dependency set, excluding controls upstream of the limb (torso, root).

## Reversibility and safety

* Extraction, fingerprint and scene-info scripts never call operators or write properties.
* The overlay uses GPU draw handlers stored in `bpy.app.driver_namespace`. It creates no objects or
  datablocks, is never saved, and is removed by `overlay clear` or a Blender restart.
* `screenshot --view` saves the 3D view's location/rotation/distance/perspective and restores them.
* The `.blend` file is never saved. Integration tests use generated fixtures in temporary files.

## Validation tooling (`animagent/validation/`)

`review-package` runs: tests -> live analysis + read-only probe (dirty state recorded after every phase) ->
validation screenshots -> live-vs-disk fingerprint comparison -> experiments on an isolated background copy
of the reference .blend -> generality runs -> checks A-H. Each check states its method so that analyzer
self-consistency is never confused with independent evidence.

## Testing

* `tests/unit`: synthetic skeletons (`tests/fakes/synthetic.py`: plain humanoid, tail,
  accessory, reversed knees, swapped sides, a Rigify-like rig with IK/FK switches) and a **fake
  `blender-mcp` stdio server** (`tests/fakes/fake_blender_mcp.py`) exercising the real MCP client
  and CLI (doctor, analyze, mutation detection, errors, screenshots).
* `tests/integration` (`-m blender`): launches an isolated **background Blender** with the official
  add-on's bridge on a free port, generates a Rigify human with a skinned mesh, saves and reopens it,
  and runs the full pipeline through the official MCP server. It asserts the anatomy, the controls,
  read-only behaviour (including `is_dirty` staying false) and that the fingerprint detects changes.
* `-m live` (opt-in `ANIMAGENT_LIVE_TESTS=1`): read-only analysis of the user's running Blender.
