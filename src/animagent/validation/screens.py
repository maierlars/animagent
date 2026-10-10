"""Validation screenshots of the live character (UI-only changes, restored after each shot)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animagent.blender.session import BlenderSession
from animagent.models.profile import RigProfile
from animagent.visualize import framing, overlay_spec

ANATOMY = ("skeleton", "joints", "contacts", "frame", "ambiguity", "labels")
BEND = ("skeleton", "joints", "bend", "labels")
CONTROLS = ("controls", "labels")


def shot_list(p: RigProfile) -> list[dict[str, Any]]:
    a = p.anatomy
    seg_ids = {s.id for s in a.segments}
    head_group = [s.id for s in a.segments if s.id in ("neck", "head", "jaw") or s.parent_segment in ("head", "jaw")]
    legs = [s for lb in a.limbs if lb.kind.value == "leg" for s in [*lb.segments, *lb.digits]]
    arms = [s for lb in a.limbs if lb.kind.value == "arm" for s in lb.segments]
    left_hand = [s for lb in a.limbs if lb.id == "arm.L" for s in [lb.segments[-1], *lb.digits]]
    right_hand = [s for lb in a.limbs if lb.id == "arm.R" for s in [lb.segments[-1], *lb.digits]]
    chest_app = [s.id for s in a.segments if s.kind.value == "appendage" and s.parent_segment in ("chest", "spine")]
    tail = [s for s in ("pelvis", "tail") if s in seg_ids] + [s.id for s in a.segments if s.id.startswith("tail:")]
    shots: list[dict[str, Any]] = [
        {
            "name": "reference",
            "view": "front",
            "layers": (),
            "overlay": {"show_overlays": False},
            "caption": "Clean reference: front orthographic, no annotations, viewport overlays off.",
        },
        {
            "name": "reference_side",
            "view": "right",
            "layers": (),
            "overlay": {"show_overlays": False},
            "caption": "Clean reference: side (character's right) orthographic.",
        },
        {
            "name": "anatomy_front",
            "view": "front",
            "layers": ANATOMY,
            "legend": "Anatomy - front orthographic",
            "caption": "Segments, joints (by confidence), contacts, body axes; ambiguous appendages dashed grey.",
        },
        {
            "name": "anatomy_side",
            "view": "right",
            "layers": ANATOMY,
            "legend": "Anatomy - side orthographic (character's right)",
            "caption": "Side view: spine curve, leg chain incl. ankle/hock, tail, head/beak.",
        },
        {
            "name": "anatomy_perspective",
            "view": "persp",
            "layers": ANATOMY,
            "legend": "Anatomy - three-quarter perspective",
            "caption": "Three-quarter perspective.",
        },
        {
            "name": "bend_axes",
            "view": "persp",
            "layers": BEND,
            "segments": legs + arms,
            "legend": "Knee/elbow/ankle bend axes",
            "caption": "Hinge axes (purple) and apex directions (white) for elbows, knees, ankles, toes.",
        },
        {
            "name": "bend_axes_legs_side",
            "view": "right",
            "layers": BEND,
            "segments": legs,
            "legend": "Leg bending - side view",
            "caption": "Legs from the side: knee apex should point forward, ankle (hock) apex backward.",
        },
        {
            "name": "bend_axes_arms_top",
            "view": "top",
            "layers": BEND,
            "segments": arms,
            "legend": "Arm bending - top view",
            "caption": "Arms from above: elbow apex should point backward.",
        },
        {
            "name": "rig_controls",
            "view": "front",
            "layers": CONTROLS,
            "legend": "Rig control mapping (structural trace)",
            "caption": "Mapped controls at their current pose by role; dotted lines go to the driven segment.",
        },
        {
            "name": "rig_controls_side",
            "view": "right",
            "layers": CONTROLS,
            "legend": "Rig control mapping - side",
            "caption": "Same, side view (IK poles, foot roll, switch holders).",
        },
        {
            "name": "closeup_head_neck_beak",
            "view": "right",
            "layers": ANATOMY,
            "segments": head_group,
            "legend": "Head / neck / beak (side)",
            "margin": 1.2,
            "caption": "Head, neck, jaw (lower beak); upper beak, crest, tufts are ambiguous appendages.",
        },
        {
            "name": "closeup_legs_feet_side",
            "view": "right",
            "layers": BEND + ("contacts",),
            "segments": legs,
            "legend": "Legs & feet (side)",
            "caption": "Leg chain, ankle/hock, heel/ball/toe contacts.",
        },
        {
            "name": "closeup_feet_top",
            "view": "top",
            "layers": ANATOMY,
            "segments": [s for s in legs if not s.startswith(("thigh", "shin"))],
            "legend": "Feet from above (toe directions)",
            "margin": 1.1,
            "caption": "Toe chains from above: two forward, two backward = zygodactyl foot.",
        },
        {
            "name": "closeup_hand_left",
            "view": "persp",
            "layers": ANATOMY,
            "segments": left_hand,
            "legend": "Left hand & fingers",
            "margin": 1.2,
            "caption": "Left hand, palm contact, finger chains.",
        },
        {
            "name": "closeup_hand_right",
            "view": "persp",
            "layers": ANATOMY,
            "segments": right_hand,
            "legend": "Right hand & fingers",
            "margin": 1.2,
            "caption": "Right hand (holds the wand pose).",
        },
        {
            "name": "closeup_wings_feathers_back",
            "view": "back",
            "layers": ANATOMY,
            "segments": arms + chest_app,
            "legend": "Arms vs wing feathers (back view)",
            "caption": "Arms (coloured) vs feather appendages on the chest (dashed grey, uninterpreted).",
        },
        {
            "name": "closeup_tail",
            "view": "right",
            "layers": ANATOMY,
            "segments": tail,
            "legend": "Tail (side)",
            "margin": 1.3,
            "caption": "Tail chain hanging from the pelvis.",
        },
    ]
    return [s for s in shots if s.get("segments") is None or s["segments"]]


async def capture_all(session: BlenderSession, p: RigProfile, outdir: Path) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    done = []
    for shot in shot_list(p):
        layers = set(shot["layers"])
        if layers:
            spec = overlay_spec(p, layers=layers, segments=shot.get("segments"), legend_title=shot.get("legend"))
            await session.run_script("overlay", {"action": "show", "spec": spec})
        overlay_settings = {"show_bones": False, "show_text": False, **shot.get("overlay", {})}
        fr = framing(p, shot.get("segments"), margin=shot.get("margin", 1.0))
        await session.run_script(
            "viewport",
            {"action": "frame", "view": shot["view"], **fr, "hide_panels": True, "overlay": overlay_settings},
        )
        path = outdir / f"{shot['name']}.png"
        try:
            # Official MCP screenshot tool only. (A direct bpy.ops.screen.screenshot_area call from the
            # add-on's execution context crashed Blender 5.2.2 on 2026-10-10; never call it from here.)
            path.write_bytes(await session.screenshot_view3d())
        finally:
            await session.run_script("viewport", {"action": "restore"})
            if layers:
                await session.run_script("overlay", {"action": "clear"})
        done.append(
            {
                "file": path.name,
                "view": shot["view"],
                "layers": sorted(layers),
                "segments": shot.get("segments"),
                "caption": shot["caption"],
                "bytes": path.stat().st_size,
            }
        )
    return done
