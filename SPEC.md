# Specification

The original product brief is in `SPEC`. This document pins down scope, data contracts and
acceptance for each milestone as implemented.

## Product goal (summary)

Given a Rigify-rigged character and a natural-language description, produce editable,
exportable Blender Actions of **stylized, exaggerated fighting-game animation** (readable from
a distant gameplay camera, with strong anticipation, explosive timing and weight). GPT-6 (OpenAI
Responses API) plans and critiques. Numerical algorithms (pose solving, curves, validation)
stay deterministic. First character: Kyra (lioness, two-handed staff). Target: a three-hit staff combo.

## Milestones

1. **Rig inspection and anatomical analysis** (this milestone, implemented).
2. Functional discovery: controlled, fully reverted perturbation experiments that measure what each control moves (see `TASKS.md`).
3. Semantic pose to control solving (IK/FK aware, two-handed staff as a rigid object with grip frames).
4. Choreography planning (phases, contacts, blocking poses) with GPT-6.
5. Curve generation (anticipation, overshoot, overlap), secondary motion.
6. Numerical and visual evaluation loop; Action export.

## Milestone 1: scope

### A. Connectivity: `animagent doctor [--json]`
Starts the official `blender-mcp` server, lists its tools with read-only/destructive
annotations, runs a read-only scene probe and reports server version, MCP protocol version,
Blender version and mode, file and dirty state, armatures (Rigify-generated, metarig or plain) with
their deformed meshes, and numpy/Rigify availability. Exit code 2 if Blender is unreachable.

### B. Extraction: `animagent extract-rig --armature A --output raw.json`
`RigExtraction` (`rig_extraction/1`), all world space at rest unless noted:
- armature: world matrix, pose position, bone collections, custom properties, animation summary
  (action, slot, keyed bones, frame range, NLA tracks)
- per bone: hierarchy, deform/connect/inherit flags, head/tail/length, rest matrices, collections,
  custom shape, visibility; pose: loc/rot (all modes)/scale, evaluated matrix and head/tail, locks,
  IK limits; constraints (type, influence, mute, targets incl. poles, generic RNA settings);
  custom properties with UI ranges
- drivers: data path, type, expression, variables and targets, F-curve keyframes and modifiers
- Rigify: rig_id, rig UI text, metarig and its bones and `rigify_type`s
- deformed meshes: modifiers (armature settings and order), shape keys, bbox, surface area,
  influence statistics, and per vertex group: vertex count, weight sum, skin-weighted area,
  weighted centroid, RMS radius and strong-weight bbox. **No per-vertex arrays.**
- attachments: objects parented or constrained to the rig (prop mounts)

### C. Anatomy (`anatomy/1`) and control mapping (`controls/1`)
Anatomy (rig-independent): body frame (origin, up, forward, left, height, ground), symmetry;
segments (pelvis, spine, chest, neck, head, jaw, clavicle, upper_arm, forearm, hand, thigh,
shin, foot, toe, digit, tail, appendage) with source bones and mass estimates; joints (shoulder,
elbow, wrist, hip, knee, ankle, ball, spine, neck base, skull base, tail, digits) with DOF hints and
bend info for hinges; limbs; end effectors; contact points (heel, ball, toe tip, palm,
fingertips, head top); **ambiguities**, plus resolutions kept separate from deterministic results.
Every inference carries `confidence` and `evidence`.

Controls (rig-specific): per limb the FK chain, IK effector, IK pole, IK chain controls (e.g. foot roll),
IK constraints, IK/FK switch property with **verified** IK/FK values, limb settings and tweaks;
per segment the candidate controls with roles (fk/master/tweak/parent) and dependency paths;
root, torso, body controls, global switch properties, prop mounts and unmapped controls.

GPT-6 interface: `AnatomyInterpreter.resolve(skeleton, anatomy) -> [Resolution]`, enabled
with `--llm`. Only a compact numeric context is sent. A resolution is accepted only if it picks an
offered candidate with confidence ≥ 0.7.

### D. Visualization
- `05_report.json` / `animagent report`: compact JSON (~20 KB) of anatomy, controls and ambiguities.
- `animagent overlay show|clear|status --profile P [--pose current|rest]`: segments, joints,
  hinge axes, bend directions, contacts and body axes drawn via GPU draw handlers (no scene data).
- `animagent screenshot --output x.png [--profile P --overlay] [--view front|back|left|right|top|persp]`:
  temporarily frames the viewport, captures it via the official screenshot tool, and restores the view.

### E. Tests and documentation
Unit tests (synthetic skeletons, fake MCP server), integration tests on a generated Rigify
rig in background Blender through the official server, and an opt-in live test. Docs:
`ARCHITECTURE.md`, `TASKS.md`, `CLAUDE.md`, `README.md`.

## Acceptance (milestone 1)

`animagent analyze-rig --armature <Rig> --output kyra_profile.json` produces a `rig_profile/1`
JSON with the extraction, skeleton, anatomy, control mapping, confidences, the read-only
proof and diagnostics, plus per-stage artifacts. It runs against a real Blender via MCP and
exits non-zero if the scene changed.
