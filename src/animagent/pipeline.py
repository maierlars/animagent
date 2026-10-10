"""Milestone-1 pipeline: fingerprint -> extract -> fingerprint -> skeleton -> anatomy -> controls.

Every stage result is saved separately to an artifacts directory so it can be inspected
on its own; the combined `RigProfile` is the main output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from animagent.agent.interpreter import AnatomyInterpreter, apply_resolutions
from animagent.analysis.anatomy import analyze_anatomy
from animagent.analysis.controls import map_controls
from animagent.analysis.report import build_report
from animagent.analysis.skeleton import build_skeleton
from animagent.blender.session import BlenderSession
from animagent.logging_setup import get_logger
from animagent.models.profile import Diagnostic, ProfileSource, ReadOnlyCheck, RigProfile, StateFingerprint
from animagent.models.rig import RigExtraction

log = get_logger(__name__)


async def fingerprint(session: BlenderSession, armature: str | None) -> StateFingerprint:
    return StateFingerprint.model_validate(await session.run_script("state_fingerprint", {"armature": armature}))


async def extract(session: BlenderSession, armature: str) -> RigExtraction:
    raw = await session.run_script("extract_rig", {"armature": armature})
    return RigExtraction.model_validate(raw)


def analyze(
    extraction: RigExtraction,
    interpreter: AnatomyInterpreter | None = None,
    readonly_check: ReadOnlyCheck | None = None,
) -> RigProfile:
    """Pure-Python analysis of an extraction (no Blender needed)."""
    diagnostics: list[Diagnostic] = []
    skeleton = build_skeleton(extraction)
    log.info("stage.skeleton", strategy=skeleton.strategy, bones=len(skeleton.bones), roots=skeleton.roots)
    anatomy = analyze_anatomy(extraction, skeleton)
    log.info(
        "stage.anatomy", segments=len(anatomy.segments), limbs=len(anatomy.limbs), ambiguities=len(anatomy.ambiguities)
    )
    if interpreter is not None and anatomy.ambiguities:
        resolutions = interpreter.resolve(skeleton, anatomy)
        apply_resolutions(anatomy, resolutions)
        log.info("stage.interpretation", resolutions=len(resolutions), accepted=sum(r.accepted for r in resolutions))
    controls = map_controls(extraction, skeleton, anatomy)
    log.info(
        "stage.controls",
        limbs=len(controls.limbs),
        controls=len(controls.control_bones),
        unmapped=len(controls.unmapped_controls),
    )
    for att in extraction.attachments:
        for rel in att.relations:
            if rel.bone and rel.bone not in extraction.bone_map():
                diagnostics.append(
                    Diagnostic(
                        level="warning",
                        stage="extraction",
                        message=f"Object {att.name!r} is attached to missing bone {rel.bone!r}",
                    )
                )
    if readonly_check is not None and not readonly_check.identical:
        diagnostics.append(
            Diagnostic(
                level="error",
                stage="readonly_check",
                message=f"Scene changed during inspection: {readonly_check.changed_categories}",
            )
        )
    ex = extraction
    return RigProfile(
        source=ProfileSource(
            blend_file=ex.blender.file_path,
            armature=ex.armature.name,
            blender_version=ex.blender.version,
            scene=ex.blender.scene,
            frame_current=ex.blender.frame_current,
        ),
        extraction=ex,
        skeleton=skeleton,
        anatomy=anatomy,
        controls=controls,
        readonly_check=readonly_check,
        diagnostics=diagnostics,
    )


async def analyze_rig(
    session: BlenderSession, armature: str, interpreter: AnatomyInterpreter | None = None
) -> RigProfile:
    before = await fingerprint(session, armature)
    extraction = await extract(session, armature)
    after = await fingerprint(session, armature)
    check = ReadOnlyCheck.compare(before, after)
    log.info(
        "stage.extraction", bones=len(extraction.bones), meshes=len(extraction.meshes), readonly_ok=check.identical
    )
    return analyze(extraction, interpreter, check)


def save_artifacts(profile: RigProfile, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {
        "01_extraction.json": profile.extraction,
        "02_skeleton.json": profile.skeleton,
        "03_anatomy.json": profile.anatomy,
        "04_controls.json": profile.controls,
    }
    out = {}
    for name, model in stages.items():
        path = directory / name
        path.write_text(model.model_dump_json(indent=1))
        out[name] = path
    if profile.readonly_check:
        path = directory / "00_readonly_check.json"
        path.write_text(profile.readonly_check.model_dump_json(indent=1))
        out[path.name] = path
    path = directory / "05_report.json"
    path.write_text(json.dumps(build_report(profile), indent=1))
    out[path.name] = path
    return out
