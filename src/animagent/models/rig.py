"""Raw rig extraction data model (schema `rig_extraction/1`).

Mirrors what `blender/scripts/extract_rig.py` returns. All positions are world space at
rest pose unless stated otherwise. This model is deliberately "raw": no interpretation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RIG_EXTRACTION_SCHEMA = "rig_extraction/1"

Vec3 = list[float]
Mat4 = list[list[float]]


class _Model(BaseModel):
    # Unknown keys from newer Blender-side scripts are kept rather than dropped.
    model_config = ConfigDict(extra="allow")


class CustomProp(_Model):
    value: Any = None
    ui: dict[str, Any] = Field(default_factory=dict)


class ConstraintTarget(_Model):
    object: str
    subtarget: str = ""
    role: Literal["target", "pole", "armature"] = "target"
    weight: float | None = None


class ConstraintData(_Model):
    name: str
    type: str
    influence: float = 1.0
    mute: bool = False
    enabled: bool = True
    is_valid: bool = True
    owner_space: str | None = None
    target_space: str | None = None
    targets: list[ConstraintTarget] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.enabled and not self.mute and self.influence > 0.0


class IKSettings(_Model):
    stretch: float = 0.0
    lock: list[bool] = Field(default_factory=lambda: [False] * 3)
    use_limit: list[bool] = Field(default_factory=lambda: [False] * 3)
    min: Vec3 = Field(default_factory=lambda: [0.0] * 3)
    max: Vec3 = Field(default_factory=lambda: [0.0] * 3)
    stiffness: Vec3 = Field(default_factory=lambda: [0.0] * 3)


class PoseData(_Model):
    location: Vec3 = Field(default_factory=lambda: [0.0] * 3)
    rotation_mode: str = "QUATERNION"
    rotation_quaternion: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    rotation_euler: Vec3 = Field(default_factory=lambda: [0.0] * 3)
    rotation_axis_angle: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 0.0])
    scale: Vec3 = Field(default_factory=lambda: [1.0] * 3)
    matrix_armature: Mat4 | None = None
    head_world: Vec3 | None = None
    tail_world: Vec3 | None = None
    lock_location: list[bool] = Field(default_factory=lambda: [False] * 3)
    lock_rotation: list[bool] = Field(default_factory=lambda: [False] * 3)
    lock_rotation_w: bool = False
    lock_rotations_4d: bool = False
    lock_scale: list[bool] = Field(default_factory=lambda: [False] * 3)
    ik: IKSettings = Field(default_factory=IKSettings)


class BoneData(_Model):
    name: str
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    use_deform: bool = False
    use_connect: bool = False
    inherit_rotation: bool = True
    inherit_scale: str = "FULL"
    use_local_location: bool = True
    bbone_segments: int = 1
    head_world: Vec3
    tail_world: Vec3
    length: float
    matrix_local: Mat4 | None = None
    matrix_world_rest: Mat4
    collections: list[str] = Field(default_factory=list)
    hidden: bool = False
    is_visible: bool = True
    custom_shape: str | None = None
    pose: PoseData = Field(default_factory=PoseData)
    constraints: list[ConstraintData] = Field(default_factory=list)
    custom_props: dict[str, CustomProp] = Field(default_factory=dict)


class DriverTarget(_Model):
    id_type: str | None = None
    id_name: str | None = None
    data_path: str = ""
    bone_target: str = ""
    transform_type: str | None = None
    transform_space: str | None = None
    rotation_mode: str | None = None


class DriverVariable(_Model):
    name: str
    type: str
    targets: list[DriverTarget] = Field(default_factory=list)


class FModifierData(_Model):
    type: str
    mute: bool = False
    active: bool = True
    use_influence: bool = False
    influence: float = 1.0
    use_restricted_range: bool = False
    mode: str | None = None
    poly_order: int | None = None
    use_additive: bool = False
    coefficients: list[float] = Field(default_factory=list)


class DriverData(_Model):
    owner: Literal["object", "armature_data"] = "object"
    data_path: str
    array_index: int = 0
    type: str = "SCRIPTED"
    expression: str = ""
    use_self: bool = False
    is_valid: bool = True
    variables: list[DriverVariable] = Field(default_factory=list)
    has_modifiers: bool = False
    keyframe_count: int = 0
    keyframes: list[list[float]] = Field(default_factory=list)
    modifiers: list[FModifierData] = Field(default_factory=list)


class VertexGroupSummary(_Model):
    name: str
    is_deform_bone: bool = False
    vertex_count: int = 0
    weight_sum: float = 0.0
    area_weighted: float = 0.0
    centroid_world: Vec3 | None = None
    rms_radius: float | None = None
    strong_vertex_count: int = 0
    bbox_min: Vec3 | None = None
    bbox_max: Vec3 | None = None


class WeightSummary(_Model):
    vertex_count: int
    polygon_count: int = 0
    surface_area: float = 0.0
    unweighted_vertex_count: int = 0
    max_influences: int = 0
    mean_influences: float = 0.0
    bbox_min: Vec3 | None = None
    bbox_max: Vec3 | None = None
    groups: list[VertexGroupSummary] = Field(default_factory=list)


class ModifierData(_Model):
    index: int
    name: str
    type: str
    show_viewport: bool = True
    show_render: bool = True
    object: str | None = None


class MeshData(_Model):
    name: str
    data_name: str
    data_users: int = 1
    parent: str | None = None
    parent_type: str | None = None
    parent_bone: str | None = None
    matrix_world: Mat4
    hide_viewport: bool = False
    visible: bool = True
    modifiers: list[ModifierData] = Field(default_factory=list)
    shape_key_count: int = 0
    shape_key_driver_count: int = 0
    material_count: int = 0
    weights: WeightSummary


class BoneCollectionData(_Model):
    name: str
    visible: bool = True
    parent: str | None = None
    bone_count: int = 0


class AnimationSummary(_Model):
    action: str | None = None
    slot: str | None = None
    fcurve_count: int = 0
    keyed_bones: list[str] = Field(default_factory=list)
    frame_range: list[float] | None = None
    nla_tracks: list[str] = Field(default_factory=list)
    driver_count: int = 0


class ArmatureData(_Model):
    name: str
    data_name: str
    matrix_world: Mat4
    pose_position: str = "POSE"
    display_type: str | None = None
    bone_count: int
    collections: list[BoneCollectionData] = Field(default_factory=list)
    custom_props: dict[str, CustomProp] = Field(default_factory=dict)
    data_custom_props: dict[str, CustomProp] = Field(default_factory=dict)
    animation: AnimationSummary | None = None
    hide_viewport: bool = False


class MetarigBone(_Model):
    name: str
    parent: str | None = None
    use_connect: bool = False
    rigify_type: str = ""
    head_world: Vec3
    tail_world: Vec3


class RigifyInfo(_Model):
    rig_id: str | None = None
    is_rigify_generated: bool = False
    rig_ui_text: str | None = None
    metarig: str | None = None
    metarig_bones: list[MetarigBone] = Field(default_factory=list)


class AttachmentRelation(_Model):
    kind: Literal["parent", "constraint"]
    parent_type: str | None = None
    type: str | None = None
    bone: str | None = None
    influence: float | None = None
    mute: bool | None = None


class Attachment(_Model):
    name: str
    type: str
    relations: list[AttachmentRelation] = Field(default_factory=list)
    hide_viewport: bool = False


class BlenderInfo(_Model):
    version: str
    file_path: str = ""
    is_dirty: bool = False
    scene: str = "Scene"
    frame_current: int = 1
    fps: float = 24.0
    unit_system: str = "METRIC"
    unit_scale_length: float = 1.0
    background: bool = False


class RigExtraction(_Model):
    schema_version: Literal["rig_extraction/1"] = RIG_EXTRACTION_SCHEMA
    blender: BlenderInfo
    armature: ArmatureData
    rigify: RigifyInfo = Field(default_factory=RigifyInfo)
    bones: list[BoneData]
    drivers: list[DriverData] = Field(default_factory=list)
    meshes: list[MeshData] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)

    def bone_map(self) -> dict[str, BoneData]:
        return {b.name: b for b in self.bones}
