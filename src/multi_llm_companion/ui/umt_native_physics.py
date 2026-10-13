"""PMX作者物理をBullet 2.75で実行するネイティブ互換バックエンド。"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from multi_llm_companion.ui.bullet_pmx_physics import (
    _body_anchors,
    _BodyAnchor,
    _bone_depth,
    _interpolate_kinematic_anchors,
    _matrix_to_euler,
    _wrap_angle,
)
from multi_llm_companion.ui.models import UiAvatarModelPreview, UiAvatarRigidBody
from multi_llm_companion.ui.pmx_physics import PhysicsPose
from multi_llm_companion.ui.pmx_pose import (
    BoneTransform,
    identity_matrix,
    matrix_multiply,
    matrix_transpose,
    matrix_vector,
    rotation_matrix,
    subtract,
)

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]

_MMD_UNIT_SCALE = 0.08
_MMD_REFERENCE_GRAVITY = 98.0
_SOLVER_ITERATIONS = 4
_REFERENCE_SOLVER_ITERATIONS = 4
_GRAVITY = (0.0, -_MMD_REFERENCE_GRAVITY * _MMD_UNIT_SCALE, 0.0)
_FIXED_STEP_SECONDS = 1.0 / 120.0
_MAX_CATCH_UP_STEPS = 120
_MAX_SUB_STEPS = 120
_GROUND_COLLISION_GROUP = 15
_GROUND_COLLISION_MASK = -1


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("ground_normal", ctypes.c_float * 3),
        ("ground_constant", ctypes.c_float),
        ("mmd_unit_to_unity_unit", ctypes.c_float),
        ("convex_margin", ctypes.c_float),
        ("spring_translation_damping", ctypes.c_float),
        ("spring_rotation_damping", ctypes.c_float),
        ("locked_translation_reinforce", ctypes.c_int),
    ]


class _RigidBodySimulationData(ctypes.Structure):
    _fields_ = [
        ("rigid_body_index", ctypes.c_int),
        ("related_bone_index", ctypes.c_int),
        ("group_index", ctypes.c_uint8),
        ("collision_group_mask", ctypes.c_int16),
        ("shape", ctypes.c_uint8),
        ("size", ctypes.c_float * 3),
        ("position", ctypes.c_float * 3),
        ("rotation", ctypes.c_float * 3),
        ("mass", ctypes.c_float),
        ("linear_damping", ctypes.c_float),
        ("angular_damping", ctypes.c_float),
        ("restitution", ctypes.c_float),
        ("friction", ctypes.c_float),
        ("mode", ctypes.c_uint8),
        ("initial_transform", ctypes.c_float * 16),
        ("no_related_bone", ctypes.c_bool),
        ("bone_local_transform", ctypes.c_float * 16),
        ("bone_model_position", ctypes.c_float * 3),
        ("rest_world_transform", ctypes.c_float * 16),
        ("bone_transform_level", ctypes.c_int),
    ]


class _NativeJointData(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint8),
        ("rigid_body_a_index", ctypes.c_int),
        ("rigid_body_b_index", ctypes.c_int),
        ("frame_in_a", ctypes.c_float * 16),
        ("frame_in_b", ctypes.c_float * 16),
        ("translation_min", ctypes.c_float * 3),
        ("translation_max", ctypes.c_float * 3),
        ("rotation_min", ctypes.c_float * 3),
        ("rotation_max", ctypes.c_float * 3),
        ("spring_translation", ctypes.c_float * 3),
        ("spring_rotation", ctypes.c_float * 3),
    ]


@dataclass(frozen=True, slots=True)
class _NativeBody:
    source: UiAvatarRigidBody
    local_position: Vector3
    local_rotation: Matrix3
    inverse_local_rotation: Matrix3


def _euler_zxy_rotation(pitch: float, yaw: float, roll: float) -> Matrix3:
    """Unity MathematicsのEulerZXYと同じ順序でPMX回転を作る。"""

    rotation_x = rotation_matrix(pitch, 0.0, 0.0)
    rotation_y = rotation_matrix(0.0, yaw, 0.0)
    rotation_z = rotation_matrix(0.0, 0.0, roll)
    return matrix_multiply(matrix_multiply(rotation_y, rotation_x), rotation_z)


def _native_config() -> _NativeConfig:
    """UnityMMDTools 0.5.0と同じネイティブ設定を作る。"""

    result = _NativeConfig()
    result.ground_normal[:] = (0.0, 1.0, 0.0)
    result.ground_constant = 0.0
    result.mmd_unit_to_unity_unit = _MMD_UNIT_SCALE
    result.convex_margin = 0.04 * _MMD_UNIT_SCALE
    result.spring_translation_damping = _SOLVER_ITERATIONS / _REFERENCE_SOLVER_ITERATIONS
    result.spring_rotation_damping = _SOLVER_ITERATIONS / (
        _REFERENCE_SOLVER_ITERATIONS * _MMD_UNIT_SCALE * _MMD_UNIT_SCALE
    )
    result.locked_translation_reinforce = 2
    return result


def _prepare_native_bodies(preview: UiAvatarModelPreview) -> list[_NativeBody]:
    """PMX基準姿勢からボーン相対の剛体変換を準備する。"""

    result: list[_NativeBody] = []
    for body in preview.rigid_bodies:
        local_rotation = _euler_zxy_rotation(*body.rotation)
        local_position = (
            subtract(body.position, preview.bones[body.bone_index].position)
            if 0 <= body.bone_index < len(preview.bones)
            else body.position
        )
        result.append(
            _NativeBody(body, local_position, local_rotation, matrix_transpose(local_rotation))
        )
    return result


def native_backend_available() -> bool:
    """同梱済みの検証済みネイティブDLLを利用できるか返す。"""

    return _dll_path().is_file()


def _dll_path() -> Path:
    return Path(__file__).with_name("native") / "UMTNativePlugin.dll"


class UmtNativePmxPhysicsSimulator:
    """PMX剛体・6DoFばねをネイティブBullet 2.75で計算する。"""

    backend_status_text = "ネイティブBullet 2.75互換物理・PMX剛体/6DoFばね/ジョイント"

    def __init__(self, *, fixed_step_seconds: float = _FIXED_STEP_SECONDS) -> None:
        self._fixed_step_seconds = max(1.0 / 240.0, min(fixed_step_seconds, 1.0 / 20.0))
        self._preview: UiAvatarModelPreview | None = None
        self._dll: ctypes.CDLL | None = None
        self._context: ctypes.c_void_p | None = None
        self._bodies: list[_NativeBody] = []
        self._kinematic_body_indexes: tuple[int, ...] = ()
        self._aligned_body_indexes: tuple[int, ...] = ()
        self._dynamic_body_indexes: tuple[int, ...] = ()
        self._physics_bone_order: tuple[int, ...] = ()
        self._output_bone_indexes: frozenset[int] | None = None
        self._previous_kinematic_anchors: tuple[_BodyAnchor, ...] = ()
        self._accumulator = 0.0
        self._last_pose = PhysicsPose()
        self._last_substeps = 0
        self._dropped_seconds = 0.0

    @property
    def fixed_step_seconds(self) -> float:
        return self._fixed_step_seconds

    @property
    def last_substeps(self) -> int:
        return self._last_substeps

    @property
    def dropped_seconds(self) -> float:
        return self._dropped_seconds

    @property
    def is_initialized(self) -> bool:
        return self._context is not None

    def reset(
        self,
        preview: UiAvatarModelPreview | None,
        bone_transforms: dict[int, BoneTransform] | None = None,
    ) -> None:
        self._destroy_context()
        self._preview = preview
        self._bodies = []
        self._kinematic_body_indexes = ()
        self._aligned_body_indexes = ()
        self._dynamic_body_indexes = ()
        self._physics_bone_order = ()
        self._output_bone_indexes = None
        self._previous_kinematic_anchors = ()
        self._accumulator = 0.0
        self._last_pose = PhysicsPose()
        self._last_substeps = 0
        self._dropped_seconds = 0.0
        if preview is None or not preview.rigid_bodies:
            return
        self._build_world(preview, bone_transforms or {})

    def step(
        self, preview: UiAvatarModelPreview | None, delta_seconds: float, enabled: bool
    ) -> dict[int, Vector3]:
        return self.step_pose(preview, delta_seconds, enabled).translations

    def step_pose(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
        bone_transforms: dict[int, BoneTransform] | None = None,
    ) -> PhysicsPose:
        if preview is not self._preview:
            self.reset(preview, bone_transforms)
        if preview is None or self._context is None or not enabled:
            return PhysicsPose()
        anchors = _body_anchors(preview, bone_transforms or {}, self._bodies)  # type: ignore[arg-type]
        safe_delta = max(0.0, min(delta_seconds, 0.25))
        if safe_delta == 0.0:
            self._push_targets(anchors, clear_velocity=True)
            self._previous_kinematic_anchors = anchors
            return self._last_pose
        self._accumulator += safe_delta
        available_steps = int(self._accumulator / self._fixed_step_seconds)
        substeps = min(available_steps, _MAX_CATCH_UP_STEPS)
        self._last_substeps = substeps
        previous = self._previous_kinematic_anchors or anchors
        for step_index in range(substeps):
            targets = _interpolate_kinematic_anchors(
                previous,
                anchors,
                self._kinematic_body_indexes,
                (step_index + 1) / substeps,
            )
            self._push_targets(targets, clear_velocity=False)
            self._step_native()
            self._accumulator -= self._fixed_step_seconds
        if available_steps > _MAX_CATCH_UP_STEPS:
            dropped = self._accumulator - min(self._accumulator, self._fixed_step_seconds)
            self._dropped_seconds += max(0.0, dropped)
            self._accumulator = min(self._accumulator, self._fixed_step_seconds)
        if substeps:
            self._previous_kinematic_anchors = anchors
            self._last_pose = self._bone_pose(preview, anchors)
        return self._last_pose

    def thin_cloth_source_surfaces(self) -> tuple[object, ...]:
        """旧来の表示メッシュ補正はネイティブ経路で使用しない。"""

        return ()

    def _build_world(
        self, preview: UiAvatarModelPreview, transforms: dict[int, BoneTransform]
    ) -> None:
        self._dll = _load_dll()
        gravity = (ctypes.c_float * 3)(*_GRAVITY)
        context = self._dll.MMDBulletPhysicsCreate(
            gravity,
            _SOLVER_ITERATIONS,
            _MAX_SUB_STEPS,
            self._fixed_step_seconds,
        )
        if not context:
            raise RuntimeError("ネイティブBullet物理コンテキストを作成できませんでした")
        self._context = ctypes.c_void_p(context)
        config = _native_config()
        self._dll.MMDBulletPhysicsSetConfig(self._context, ctypes.byref(config))
        self._bodies = _prepare_native_bodies(preview)
        data = (_RigidBodySimulationData * len(self._bodies))()
        for index, body in enumerate(self._bodies):
            data[index] = _rigid_body_data(index, body, preview)
        self._dll.MMDBulletPhysicsBuildRigidBodies(self._context, data, len(data))
        joints = _native_joints(preview, self._bodies)
        if joints:
            self._dll.MMDBulletPhysicsBuildJoints(self._context, joints, len(joints))
        self._dll.MMDBulletPhysicsBuildGround(
            self._context,
            True,
            _GROUND_COLLISION_GROUP,
            _GROUND_COLLISION_MASK,
        )
        self._kinematic_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode == 0
        )
        self._aligned_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode == 2
        )
        self._dynamic_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode != 0
        )
        self._physics_bone_order = tuple(
            sorted(
                {
                    self._bodies[index].source.bone_index
                    for index in self._dynamic_body_indexes
                    if 0 <= self._bodies[index].source.bone_index < len(preview.bones)
                },
                key=lambda index: _bone_depth(preview, index),
            )
        )
        anchors = _body_anchors(preview, transforms, self._bodies)  # type: ignore[arg-type]
        self._dll.MMDBulletPhysicsReset(self._context, 0)
        self._push_all(anchors, clear_velocity=True)
        self._previous_kinematic_anchors = anchors

    def _push_targets(self, anchors: tuple[_BodyAnchor, ...], *, clear_velocity: bool) -> None:
        indexes = self._kinematic_body_indexes
        if not indexes:
            return
        transforms = [
            _native_matrix4(anchors[index].position, anchors[index].rotation) for index in indexes
        ]
        self._set_transforms(indexes, transforms, clear_velocity=clear_velocity)

    def _push_all(self, anchors: tuple[_BodyAnchor, ...], *, clear_velocity: bool) -> None:
        self._set_transforms(
            tuple(range(len(self._bodies))),
            [_native_matrix4(anchor.position, anchor.rotation) for anchor in anchors],
            clear_velocity=clear_velocity,
        )

    def _set_transforms(
        self,
        indexes: tuple[int, ...],
        transforms: list[ctypes.Array[ctypes.c_float]],
        *,
        clear_velocity: bool,
    ) -> None:
        if self._dll is None or self._context is None or not indexes:
            return
        index_data = (ctypes.c_int * len(indexes))(*indexes)
        values = (ctypes.c_float * (16 * len(transforms)))()
        for index, transform in enumerate(transforms):
            values[index * 16 : (index + 1) * 16] = transform
        self._dll.MMDBulletPhysicsSetRigidBodyTransforms(
            self._context, len(indexes), values, index_data, clear_velocity
        )

    def _step_native(self) -> None:
        if self._dll is not None and self._context is not None:
            self._dll.MMDBulletPhysicsStepSimulation(self._context, self._fixed_step_seconds)

    def _read_transform(self, index: int) -> tuple[Vector3, Matrix3]:
        return self._read_transforms((index,))[index]

    def _read_transforms(self, indexes: tuple[int, ...]) -> dict[int, tuple[Vector3, Matrix3]]:
        """既存DLLの一括取得を使い、剛体ごとの呼び出し往復を省く。"""

        if self._dll is None or self._context is None:
            return {index: ((0.0, 0.0, 0.0), identity_matrix()) for index in indexes}
        if not indexes:
            return {}
        indices = (ctypes.c_int * len(indexes))(*indexes)
        values = (ctypes.c_float * (16 * len(indexes)))()
        self._dll.MMDBulletPhysicsGetRigidBodyMotionTransforms(
            self._context, len(indexes), indices, values
        )
        return {
            index: _read_native_matrix4(values[offset * 16 : (offset + 1) * 16])
            for offset, index in enumerate(indexes)
        }

    def _shift_body_position(self, index: int, delta: Vector3) -> None:
        if self._dll is None or self._context is None:
            return
        native_delta = (ctypes.c_float * 3)(*_to_native_position(delta))
        self._dll.MMDBulletPhysicsShiftRigidBodyPosition(
            self._context,
            index,
            native_delta,
        )

    def _bone_pose(
        self, preview: UiAvatarModelPreview, anchors: tuple[_BodyAnchor, ...]
    ) -> PhysicsPose:
        desired: dict[int, tuple[Matrix3, Vector3, int]] = {}
        anchor_by_bone: dict[int, _BodyAnchor] = {}
        read_indexes = tuple(
            index
            for index in self._dynamic_body_indexes
            if 0 <= self._bodies[index].source.bone_index < len(preview.bones)
            and (
                self._output_bone_indexes is None
                or self._bodies[index].source.bone_index in self._output_bone_indexes
                or self._bodies[index].source.mode == 2
            )
        )
        transforms = self._read_transforms(read_indexes)
        for index in read_indexes:
            body = self._bodies[index]
            bone_index = body.source.bone_index
            needed = self._output_bone_indexes is None or bone_index in self._output_bone_indexes
            # 剛体とジョイントは計算し続け、未使用の結果取得だけを省く。
            position, body_rotation = transforms[index]
            desired_rotation = matrix_multiply(body_rotation, body.inverse_local_rotation)
            anchor = anchors[index]
            physical_bone_position = subtract(
                position,
                matrix_vector(desired_rotation, body.local_position),
            )
            if body.source.mode == 2:
                self._shift_body_position(
                    index,
                    subtract(physical_bone_position, anchor.bone_position),
                )
                desired_position = anchor.bone_position
            else:
                desired_position = physical_bone_position
            if not needed:
                continue
            desired[bone_index] = desired_rotation, desired_position, body.source.mode
            anchor_by_bone[bone_index] = anchor
        translations: dict[int, Vector3] = {}
        rotations: dict[int, Vector3] = {}
        for bone_index in self._physics_bone_order:
            if bone_index not in desired:
                continue
            bone = preview.bones[bone_index]
            anchor = anchor_by_bone[bone_index]
            desired_rotation, desired_position, mode = desired[bone_index]
            parent_rotation, parent_position, _parent_mode = desired.get(
                bone.parent_index,
                (anchor.parent_rotation, anchor.parent_position, 0),
            )
            base_parent_inverse = matrix_transpose(anchor.parent_rotation)
            desired_parent_inverse = matrix_transpose(parent_rotation)
            base_local = matrix_multiply(base_parent_inverse, anchor.bone_rotation)
            desired_local = matrix_multiply(desired_parent_inverse, desired_rotation)
            base_euler = _matrix_to_euler(base_local)
            desired_euler = _matrix_to_euler(desired_local)
            rotations[bone_index] = tuple(
                _wrap_angle(desired_euler[axis] - base_euler[axis]) for axis in range(3)
            )
            if mode == 1:
                base_position = matrix_vector(
                    base_parent_inverse, subtract(anchor.bone_position, anchor.parent_position)
                )
                actual_position = matrix_vector(
                    desired_parent_inverse, subtract(desired_position, parent_position)
                )
                translations[bone_index] = subtract(actual_position, base_position)
        return PhysicsPose(translations=translations, rotations=rotations)

    def _destroy_context(self) -> None:
        if self._dll is not None and self._context is not None:
            self._dll.MMDBulletPhysicsDestroy(self._context)
        self._context = None
        self._dll = None

    def __del__(self) -> None:
        self._destroy_context()


def _load_dll() -> ctypes.CDLL:
    path = _dll_path()
    if not path.is_file():
        raise RuntimeError(f"ネイティブBullet DLLがありません: {path}")
    dll = ctypes.CDLL(str(path))
    dll.MMDBulletPhysicsCreate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    dll.MMDBulletPhysicsCreate.restype = ctypes.c_void_p
    dll.MMDBulletPhysicsDestroy.argtypes = [ctypes.c_void_p]
    dll.MMDBulletPhysicsSetConfig.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeConfig),
    ]
    dll.MMDBulletPhysicsReset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    dll.MMDBulletPhysicsBuildRigidBodies.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_RigidBodySimulationData),
        ctypes.c_int,
    ]
    dll.MMDBulletPhysicsBuildJoints.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NativeJointData),
        ctypes.c_int,
    ]
    dll.MMDBulletPhysicsBuildGround.argtypes = [
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_uint8,
        ctypes.c_int16,
    ]
    dll.MMDBulletPhysicsSetRigidBodyTransforms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_bool,
    ]
    dll.MMDBulletPhysicsGetRigidBodyMotionTransforms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.MMDBulletPhysicsShiftRigidBodyPosition.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.MMDBulletPhysicsStepSimulation.argtypes = [ctypes.c_void_p, ctypes.c_float]
    return dll


def _rigid_body_data(
    index: int, body: _NativeBody, preview: UiAvatarModelPreview
) -> _RigidBodySimulationData:
    source = body.source
    has_related_bone = 0 <= source.bone_index < len(preview.bones)
    bone_position = (
        preview.bones[source.bone_index].position if has_related_bone else (0.0, 0.0, 0.0)
    )
    result = _RigidBodySimulationData()
    result.rigid_body_index = index
    result.related_bone_index = source.bone_index
    result.group_index = source.collision_group
    # PMXの保存値は、0のビットを非衝突として扱うBullet互換マスク。
    result.collision_group_mask = ctypes.c_int16(source.non_collision_mask).value
    result.shape = source.shape
    result.size[:] = tuple(value * _MMD_UNIT_SCALE for value in source.size)
    result.position[:] = _to_native_position(source.position)
    result.rotation[:] = _to_native_rotation(source.rotation)
    result.mass = source.mass
    result.linear_damping = source.linear_damping
    result.angular_damping = source.angular_damping
    result.restitution = source.restitution
    result.friction = source.friction
    result.mode = source.mode
    initial_transform = _native_matrix4(body.local_position, body.local_rotation)
    rest_world_transform = _native_matrix4(source.position, body.local_rotation)
    result.initial_transform = initial_transform
    result.no_related_bone = not has_related_bone
    result.bone_local_transform = _inverse_matrix4(initial_transform)
    result.bone_model_position[:] = _to_native_position(bone_position)
    result.rest_world_transform = rest_world_transform
    result.bone_transform_level = (
        preview.bones[source.bone_index].deform_layer if has_related_bone else -1
    )
    return result


def _native_joints(
    preview: UiAvatarModelPreview, bodies: list[_NativeBody]
) -> ctypes.Array[_NativeJointData] | None:
    result: list[_NativeJointData] = []
    for joint in preview.physics_joints:
        if not (0 <= joint.rigid_body_a < len(bodies) and 0 <= joint.rigid_body_b < len(bodies)):
            continue
        first = bodies[joint.rigid_body_a]
        second = bodies[joint.rigid_body_b]
        joint_rotation = _to_native_rotation_matrix(_euler_zxy_rotation(*joint.rotation))
        joint_position = _to_native_position(joint.position)
        first_position = _to_native_position(first.source.position)
        second_position = _to_native_position(second.source.position)
        first_rotation = _to_native_rotation_matrix(first.local_rotation)
        second_rotation = _to_native_rotation_matrix(second.local_rotation)
        item = _NativeJointData()
        item.type = 0
        item.rigid_body_a_index = joint.rigid_body_a
        item.rigid_body_b_index = joint.rigid_body_b
        item.frame_in_a = _relative_matrix(
            first_position, first_rotation, joint_position, joint_rotation
        )
        item.frame_in_b = _relative_matrix(
            second_position, second_rotation, joint_position, joint_rotation
        )
        translation_min, translation_max = _native_limit_pair(
            joint.translation_min,
            joint.translation_max,
            _MMD_UNIT_SCALE,
        )
        rotation_min, rotation_max = _native_limit_pair(
            joint.rotation_min,
            joint.rotation_max,
            1.0,
        )
        item.translation_min[:] = translation_min
        item.translation_max[:] = translation_max
        item.rotation_min[:] = rotation_min
        item.rotation_max[:] = rotation_max
        item.spring_translation[:] = joint.spring_translation
        item.spring_rotation[:] = tuple(
            value * _MMD_UNIT_SCALE * _MMD_UNIT_SCALE for value in joint.spring_rotation
        )
        result.append(item)
    return (_NativeJointData * len(result))(*result) if result else None


def _matrix4(position: Vector3, rotation: Matrix3) -> ctypes.Array[ctypes.c_float]:
    return (ctypes.c_float * 16)(
        rotation[0][0],
        rotation[1][0],
        rotation[2][0],
        0.0,
        rotation[0][1],
        rotation[1][1],
        rotation[2][1],
        0.0,
        rotation[0][2],
        rotation[1][2],
        rotation[2][2],
        0.0,
        position[0],
        position[1],
        position[2],
        1.0,
    )


def _to_native_position(position: Vector3) -> Vector3:
    """MMD座標をUnityMMDToolsの左手系・0.08単位へ移す。"""

    return (
        -position[0] * _MMD_UNIT_SCALE,
        position[1] * _MMD_UNIT_SCALE,
        -position[2] * _MMD_UNIT_SCALE,
    )


def _from_native_position(position: Vector3) -> Vector3:
    return (
        -position[0] / _MMD_UNIT_SCALE,
        position[1] / _MMD_UNIT_SCALE,
        -position[2] / _MMD_UNIT_SCALE,
    )


def _to_native_rotation(rotation: Vector3) -> Vector3:
    return (-rotation[0], rotation[1], -rotation[2])


def _to_native_rotation_matrix(rotation: Matrix3) -> Matrix3:
    signs = (-1.0, 1.0, -1.0)
    return tuple(
        tuple(signs[row] * rotation[row][column] * signs[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _native_matrix4(position: Vector3, rotation: Matrix3) -> ctypes.Array[ctypes.c_float]:
    return _matrix4(
        _to_native_position(position),
        _to_native_rotation_matrix(rotation),
    )


def _read_native_matrix4(
    values: ctypes.Array[ctypes.c_float] | list[float],
) -> tuple[Vector3, Matrix3]:
    position, rotation = _read_matrix4(values)
    return _from_native_position(position), _to_native_rotation_matrix(rotation)


def _native_limit_pair(
    lower: Vector3,
    upper: Vector3,
    scale: float,
) -> tuple[Vector3, Vector3]:
    return (
        (-upper[0] * scale, lower[1] * scale, -upper[2] * scale),
        (-lower[0] * scale, upper[1] * scale, -lower[2] * scale),
    )


def _read_matrix4(values: ctypes.Array[ctypes.c_float] | list[float]) -> tuple[Vector3, Matrix3]:
    return (
        (float(values[12]), float(values[13]), float(values[14])),
        (
            (float(values[0]), float(values[4]), float(values[8])),
            (float(values[1]), float(values[5]), float(values[9])),
            (float(values[2]), float(values[6]), float(values[10])),
        ),
    )


def _inverse_matrix4(values: ctypes.Array[ctypes.c_float]) -> ctypes.Array[ctypes.c_float]:
    position, rotation = _read_matrix4(values)
    inverse = matrix_transpose(rotation)
    inverse_position = tuple(-value for value in matrix_vector(inverse, position))
    return _matrix4(inverse_position, inverse)  # type: ignore[arg-type]


def _relative_matrix(
    origin_position: Vector3,
    origin_rotation: Matrix3,
    target_position: Vector3,
    target_rotation: Matrix3,
) -> ctypes.Array[ctypes.c_float]:
    inverse = matrix_transpose(origin_rotation)
    return _matrix4(
        matrix_vector(inverse, subtract(target_position, origin_position)),
        matrix_multiply(inverse, target_rotation),
    )
