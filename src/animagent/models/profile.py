"""Top-level rig profile (schema `rig_profile/1`), the output of `animagent analyze-rig`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from animagent import __version__
from animagent.models.anatomy import AnatomyProfile, Skeleton
from animagent.models.controls import ControlMapping
from animagent.models.rig import RigExtraction

PROFILE_SCHEMA = "rig_profile/1"


class StateFingerprint(BaseModel):
    hashes: dict[str, str]
    info: dict[str, object] = Field(default_factory=dict)


class ReadOnlyCheck(BaseModel):
    """Proof that the analysis did not modify the Blender scene."""

    before: StateFingerprint
    after: StateFingerprint
    identical: bool
    changed_categories: list[str] = Field(default_factory=list)

    @classmethod
    def compare(cls, before: StateFingerprint, after: StateFingerprint) -> ReadOnlyCheck:
        keys = sorted(set(before.hashes) | set(after.hashes))
        changed = [k for k in keys if before.hashes.get(k) != after.hashes.get(k)]
        return cls(before=before, after=after, identical=not changed, changed_categories=changed)


class ProfileSource(BaseModel):
    blend_file: str
    armature: str
    blender_version: str
    scene: str
    frame_current: int


class Diagnostic(BaseModel):
    level: Literal["info", "warning", "error"]
    stage: str
    message: str


class RigProfile(BaseModel):
    schema_version: Literal["rig_profile/1"] = PROFILE_SCHEMA
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    tool_version: str = __version__
    source: ProfileSource
    extraction: RigExtraction
    skeleton: Skeleton
    anatomy: AnatomyProfile
    controls: ControlMapping
    readonly_check: ReadOnlyCheck | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path: Path) -> RigProfile:
        return cls.model_validate_json(path.read_text())
