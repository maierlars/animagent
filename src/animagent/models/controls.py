"""Rig-specific control mapping (schema `controls/1`).

Maps anatomical concepts (segment / limb ids from `models.anatomy`) to the actual
animator controls of a rig, IK/FK machinery and switch properties. Derived from the
rig's dependency graph (parenting, constraints, drivers), not from names alone.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from animagent.models.anatomy import Evidence

CONTROLS_SCHEMA = "controls/1"

ControlRole = Literal[
    "fk",
    "ik_effector",
    "ik_pole",
    "ik_chain",
    "tweak",
    "parent",
    "master",
    "root",
    "torso",
    "generic",
    "deform_only",
]


class DependencyLink(BaseModel):
    """One edge of the path from a structural bone back to a control."""

    from_bone: str
    to_bone: str
    via: str  # "parent" | "constraint:<TYPE>:<name>" | "ik_target" | "ik_pole"


class ControlRef(BaseModel):
    bone: str
    role: ControlRole
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    path: list[DependencyLink] = Field(default_factory=list)
    rotation_mode: str | None = None
    locked_channels: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    segment: str | None = None


class SwitchProperty(BaseModel):
    """A custom property that switches behaviour, discovered through drivers."""

    bone: str
    property: str
    current_value: Any = None
    min: float | None = None
    max: float | None = None
    fk_value: float | None = None
    ik_value: float | None = None
    driven: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class IKChainInfo(BaseModel):
    constraint_bone: str
    constraint_name: str
    chain_count: int
    chain_bones: list[str]
    target_bone: str | None = None
    target_control: str | None = None
    pole_bone: str | None = None
    pole_control: str | None = None
    pole_angle_deg: float | None = None
    influence: float = 1.0
    muted: bool = False
    use_tail: bool = True
    use_stretch: bool | None = None


class LimbControls(BaseModel):
    limb_id: str
    fk_chain: list[ControlRef] = Field(default_factory=list)
    ik_effector: ControlRef | None = None
    ik_pole: ControlRef | None = None
    ik_chain_controls: list[ControlRef] = Field(default_factory=list)
    ik_chains: list[IKChainInfo] = Field(default_factory=list)
    ik_fk_switch: SwitchProperty | None = None
    settings: list[SwitchProperty] = Field(default_factory=list)
    other_controls: list[ControlRef] = Field(default_factory=list)
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)


class SegmentControls(BaseModel):
    segment_id: str
    controls: list[ControlRef] = Field(default_factory=list)


class ControlMapping(BaseModel):
    schema_version: Literal["controls/1"] = CONTROLS_SCHEMA
    rig_kind: Literal["rigify", "generic"]
    control_bones: list[str]
    root_control: ControlRef | None = None
    body_controls: list[ControlRef] = Field(default_factory=list)
    limbs: list[LimbControls] = Field(default_factory=list)
    segments: list[SegmentControls] = Field(default_factory=list)
    global_properties: list[SwitchProperty] = Field(default_factory=list)
    prop_mounts: list[ControlRef] = Field(default_factory=list)
    unmapped_controls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
