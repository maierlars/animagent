"""Rig-independent anatomical model (schema `anatomy/1`).

Nothing in here refers to Rigify conventions. Bones are referenced only as *evidence*
(`source_bones`); the mapping to animator controls lives in `models.controls`.
All positions/directions are world space at rest pose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

ANATOMY_SCHEMA = "anatomy/1"

Vec3 = list[float]


class Side(StrEnum):
    LEFT = "L"
    RIGHT = "R"
    CENTER = "C"


class Evidence(BaseModel):
    """One observation supporting (weight > 0) or contradicting (weight < 0) a conclusion."""

    kind: str
    detail: str
    weight: float = 1.0


class Confident(BaseModel):
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class BodyFrame(Confident):
    """Character body coordinate system. `left` is the *character's* left."""

    origin: Vec3
    up: Vec3
    forward: Vec3
    left: Vec3
    height: float
    ground_height: float
    method: str = "structural"


class StructuralBone(BaseModel):
    """A bone of the clean anatomical skeleton (e.g. Rigify ORG bones or deform bones)."""

    name: str
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    side: Side = Side.CENTER
    head: Vec3
    tail: Vec3
    length: float
    parent_source: Literal["metarig", "hierarchy", "geometry", "root"] = "hierarchy"
    deform_bones: list[str] = Field(default_factory=list)


class Skeleton(BaseModel):
    """Stage output: the structural skeleton the anatomy analysis runs on."""

    schema_version: Literal["skeleton/1"] = "skeleton/1"
    strategy: Literal["rigify_org", "deform"] = "deform"
    bones: list[StructuralBone]
    roots: list[str]
    notes: list[str] = Field(default_factory=list)

    def bone_map(self) -> dict[str, StructuralBone]:
        return {b.name: b for b in self.bones}


class SegmentKind(StrEnum):
    PELVIS = "pelvis"
    SPINE = "spine"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    JAW = "jaw"
    CLAVICLE = "clavicle"
    UPPER_ARM = "upper_arm"
    FOREARM = "forearm"
    HAND = "hand"
    THIGH = "thigh"
    SHIN = "shin"
    FOOT = "foot"
    TOE = "toe"
    DIGIT = "digit"
    TAIL = "tail"
    APPENDAGE = "appendage"
    UNKNOWN = "unknown"


class MassEstimate(BaseModel):
    """Artistic mass guidance. Derived from skin-weighted surface area, NOT physical mass."""

    relative_mass: float
    center_of_mass: Vec3 | None = None
    rms_radius: float | None = None
    method: str = "skin_weighted_surface_area"


class Segment(Confident):
    id: str
    kind: SegmentKind
    side: Side = Side.CENTER
    source_bones: list[str]
    head: Vec3
    tail: Vec3
    length: float
    parent_segment: str | None = None
    mass: MassEstimate | None = None


class JointKind(StrEnum):
    SPINE = "spine"
    NECK_BASE = "neck_base"
    SKULL_BASE = "skull_base"
    JAW = "jaw"
    STERNOCLAVICULAR = "sternoclavicular"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    HIP = "hip"
    KNEE = "knee"
    ANKLE = "ankle"
    BALL = "ball"
    TAIL = "tail"
    DIGIT = "digit"
    GENERIC = "generic"


class BendInfo(Confident):
    """Preferred bending of a hinge-like joint.

    * `hinge_axis`: rotation axis (right-hand rule) that *increases* flexion of the child segment.
    * `joint_points_toward`: direction the joint apex points when flexed (knee -> forward for humans,
      elbow -> backward); the distal end moves the opposite way.
    * `hinge_axis_local`: the same axis expressed as the closest local axis of `local_axis_bone`.
    """

    hinge_axis: Vec3
    joint_points_toward: Vec3
    rest_flexion_deg: float
    local_axis_bone: str | None = None
    hinge_axis_local: str | None = None
    hinge_axis_local_alignment: float | None = None
    body_relative: dict[str, float] = Field(default_factory=dict)
    source: Literal["rest_geometry", "ik_pole", "local_axis_convention", "prior"] = "rest_geometry"
    matches_prior: bool | None = None


class Joint(Confident):
    id: str
    kind: JointKind
    side: Side = Side.CENTER
    position: Vec3
    parent_segment: str | None = None
    child_segment: str
    dof_hint: Literal["hinge", "ball", "saddle", "unknown"] = "unknown"
    bend: BendInfo | None = None


class LimbKind(StrEnum):
    ARM = "arm"
    LEG = "leg"
    TAIL = "tail"
    DIGIT = "digit"
    APPENDAGE = "appendage"


class ContactPoint(Confident):
    id: str
    kind: Literal["heel", "ball", "toe_tip", "sole_center", "palm", "fingertip", "tail_tip", "head_top"]
    segment: str
    position: Vec3


class EndEffector(Confident):
    id: str
    kind: Literal["hand", "foot", "head", "tail_tip", "digit_tip", "appendage_tip"]
    segment: str
    position: Vec3
    contacts: list[str] = Field(default_factory=list)


class Limb(Confident):
    id: str
    kind: LimbKind
    side: Side
    segments: list[str]
    joints: list[str]
    attached_to: str | None = None
    end_effector: str | None = None
    chain_length: float
    reaches_ground: bool = False
    digits: list[str] = Field(default_factory=list)


class Ambiguity(BaseModel):
    """Something the deterministic analysis could not decide with sufficient confidence."""

    id: str
    subject: list[str]
    question: str
    candidates: list[str]
    deterministic_best: str | None = None
    confidence: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    context: dict[str, object] = Field(default_factory=dict)


class Resolution(BaseModel):
    """An interpretation of an ambiguity, e.g. by GPT-6. Never silently overrides deterministic data."""

    ambiguity_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    source: str
    accepted: bool = False


class SymmetryInfo(BaseModel):
    pairs: int
    unpaired_sided: list[str] = Field(default_factory=list)
    max_mirror_error: float = 0.0
    median_mirror_error: float = 0.0
    asymmetric_pairs: list[str] = Field(default_factory=list)
    mirror_plane_normal: Vec3 | None = None


class AnatomyProfile(BaseModel):
    schema_version: Literal["anatomy/1"] = ANATOMY_SCHEMA
    body_frame: BodyFrame
    symmetry: SymmetryInfo
    segments: list[Segment]
    joints: list[Joint]
    limbs: list[Limb]
    end_effectors: list[EndEffector]
    contacts: list[ContactPoint]
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list)
    unassigned_bones: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def segment_map(self) -> dict[str, Segment]:
        return {s.id: s for s in self.segments}

    def joint_map(self) -> dict[str, Joint]:
        return {j.id: j for j in self.joints}
