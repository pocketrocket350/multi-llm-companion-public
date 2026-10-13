"""Panda3D Bulletを使うPMX剛体・衝突・ジョイント物理。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from panda3d.bullet import (
    BulletBoxShape,
    BulletCapsuleShape,
    BulletFilterCallbackData,
    BulletGenericConstraint,
    BulletRigidBodyNode,
    BulletSphereShape,
    BulletWorld,
    YUp,
)
from panda3d.core import (
    LPoint3f,
    LQuaternionf,
    LVector3f,
    NodePath,
    PythonCallbackObject,
    TransformState,
    load_prc_file_data,
)

from multi_llm_companion.ui.models import (
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
)
from multi_llm_companion.ui.pmx_physics import PhysicsPose
from multi_llm_companion.ui.pmx_pose import (
    BoneTransform,
    add,
    cross,
    identity_matrix,
    length,
    matrix_multiply,
    matrix_transpose,
    matrix_vector,
    normalize,
    rotation_matrix,
    subtract,
)

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]

_GRAVITY = (0.0, -9.8, 0.0)
_FIXED_STEP_SECONDS = 1.0 / 30.0
_MAX_CATCH_UP_STEPS = 2
_MAX_BODY_OFFSET = 6.0
_INITIAL_SETTLE_STEPS = 30
# Bullet標準ソルバーの既定値。ばねの速度係数を本家実装と同じ式で求める。
_SPRING_SOLVER_ITERATIONS = 10
# Panda3DはPMXのばね制約を公開していないため、汎用モーターへ渡す力を補正する。
# 実モデルの比較では0.5以上で衣装が発散し、0.25以下で安定した。
_TRANSLATION_SPRING_COMPATIBILITY_SCALE = 0.25

# 剛体ごとのPMX非衝突マスクを判定するため、グループ単位ではなく個別判定を使う。
load_prc_file_data("", "bullet-filter-algorithm callback")


@dataclass(frozen=True, slots=True)
class _BodyAnchor:
    position: Vector3
    rotation: Matrix3
    bone_position: Vector3
    bone_rotation: Matrix3
    parent_position: Vector3
    parent_rotation: Matrix3


@dataclass(slots=True)
class _BulletBody:
    source: UiAvatarRigidBody
    node: BulletRigidBodyNode
    path: NodePath
    local_position: Vector3
    local_rotation: Matrix3
    inverse_local_rotation: Matrix3
    maximum_offset: float


@dataclass(slots=True)
class _BulletJoint:
    source: UiAvatarPhysicsJoint
    constraint: BulletGenericConstraint
    translation_springs: tuple[tuple[int, float], ...] = ()
    rotation_springs: tuple[tuple[int, float], ...] = ()
    equilibrium_translation: Vector3 = (0.0, 0.0, 0.0)
    equilibrium_rotation: Vector3 = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class _HandCollider:
    """PMXに手剛体がない場合だけ補う、手のひら用の運動学的剛体。"""

    source: UiAvatarRigidBody
    node: BulletRigidBodyNode
    path: NodePath
    bone_index: int
    local_center: Vector3
    local_rotation: Matrix3


@dataclass(slots=True)
class _ThinClothCollider:
    """表示メッシュ補正へ渡す、指定済み腕・袖剛体の識別子。"""

    node: BulletRigidBodyNode
    path: NodePath
    source_body_index: int
    target_body_indexes: tuple[int, ...]
    maximum_correction_per_step: float


@dataclass(frozen=True, slots=True)
class ThinClothSourceSurface:
    """描画スレッドへ渡せる、腕・袖のカプセル近似。"""

    position: Vector3
    axis: Vector3
    radius: float
    half_length: float


class BulletPmxPhysicsSimulator:
    """PMXの剛体形状と6自由度ジョイントをBulletで計算する。"""

    backend_status_text = (
        "Panda3D Bullet固定刻み接触物理・PMXばね/剛体/ジョイント・"
        "つま先IK接地・補助手のひらの運動学的接触"
    )

    def __init__(
        self,
        *,
        fixed_step_seconds: float = _FIXED_STEP_SECONDS,
        max_catch_up_steps: int = _MAX_CATCH_UP_STEPS,
    ) -> None:
        self._fixed_step_seconds = max(1.0 / 240.0, min(fixed_step_seconds, 1.0 / 20.0))
        self._max_catch_up_steps = max(1, min(max_catch_up_steps, 8))
        self._preview: UiAvatarModelPreview | None = None
        self._world: BulletWorld | None = None
        self._root = NodePath("pmx-physics")
        self._bodies: list[_BulletBody] = []
        self._body_sources_by_node_name: dict[str, UiAvatarRigidBody] = {}
        self._joints: list[_BulletJoint] = []
        self._hand_colliders: list[_HandCollider] = []
        self._thin_cloth_colliders: list[_ThinClothCollider] = []
        self._thin_cloth_include_auxiliary_hands = False
        self._kinematic_body_indexes: tuple[int, ...] = ()
        self._aligned_body_indexes: tuple[int, ...] = ()
        self._dynamic_body_indexes: tuple[int, ...] = ()
        self._physics_bone_order: tuple[int, ...] = ()
        self._previous_kinematic_anchors: tuple[_BodyAnchor, ...] = ()
        self._spring_velocity_factor = 1.0 / self._fixed_step_seconds / _SPRING_SOLVER_ITERATIONS
        self._accumulator = 0.0
        self._last_pose = PhysicsPose()
        self._last_substeps = 0
        self._dropped_seconds = 0.0
        self._maximum_body_offset_observed = 0.0
        self._stabilization_reset_count = 0
        self._non_finite_reset_count = 0
        self._offset_limit_reset_count = 0

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
    def maximum_body_offset_observed(self) -> float:
        """現在のBullet世界で観測した動的剛体の最大移動量。"""

        return self._maximum_body_offset_observed

    @property
    def stabilization_reset_count(self) -> int:
        """非数値または安全上限超過で剛体を戻した回数。"""

        return self._stabilization_reset_count

    @property
    def non_finite_reset_count(self) -> int:
        return self._non_finite_reset_count

    @property
    def offset_limit_reset_count(self) -> int:
        return self._offset_limit_reset_count

    @property
    def is_initialized(self) -> bool:
        """PMX剛体と補助衝突形状をBulletへ登録済みか返す。"""

        return self._world is not None

    def reset(
        self,
        preview: UiAvatarModelPreview | None,
        bone_transforms: dict[int, BoneTransform] | None = None,
    ) -> None:
        """同じPMXを使い回し、モデル変更時だけBullet世界を作り直す。"""

        self._preview = preview
        self._world = None
        self._root.remove_node()
        self._root = NodePath("pmx-physics")
        self._bodies = []
        self._body_sources_by_node_name = {}
        self._joints = []
        self._hand_colliders = []
        self._thin_cloth_colliders = []
        self._thin_cloth_include_auxiliary_hands = False
        self._kinematic_body_indexes = ()
        self._aligned_body_indexes = ()
        self._dynamic_body_indexes = ()
        self._physics_bone_order = ()
        self._previous_kinematic_anchors = ()
        self._accumulator = 0.0
        self._last_pose = PhysicsPose()
        self._last_substeps = 0
        self._dropped_seconds = 0.0
        self._maximum_body_offset_observed = 0.0
        self._stabilization_reset_count = 0
        self._non_finite_reset_count = 0
        self._offset_limit_reset_count = 0
        if preview is None or not preview.rigid_bodies:
            return
        self._build_world(preview, bone_transforms or {})

    def step(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
    ) -> dict[int, Vector3]:
        return self.step_pose(preview, delta_seconds, enabled).translations

    def step_pose(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
        bone_transforms: dict[int, BoneTransform] | None = None,
    ) -> PhysicsPose:
        """固定刻みで剛体を進め、物理ボーンの差分姿勢を返す。"""

        if preview is not self._preview:
            # VMD姿勢へ切り替わる前のPMX初期姿勢で物理を作ると、初回だけ
            # 衣装へ大きな衝撃が入る。現在表示する姿勢から物理世界を作る。
            self.reset(preview, bone_transforms)
        if preview is None or self._world is None or not enabled:
            return PhysicsPose()

        anchors = _body_anchors(preview, bone_transforms or {}, self._bodies)
        hand_transforms = _hand_collider_transforms(self._hand_colliders, bone_transforms or {})
        safe_delta = max(0.0, min(delta_seconds, 0.25))
        if safe_delta == 0.0:
            # 停止中や初期化直後は、次の補間の始点を現在姿勢へそろえる。
            self._update_kinematic_bodies(
                anchors,
                hand_transforms,
                clear_kinematic_velocity=True,
            )
            self._previous_kinematic_anchors = anchors
            return self._last_pose
        self._accumulator += safe_delta
        available_steps = int(self._accumulator / self._fixed_step_seconds)
        substeps = min(available_steps, self._max_catch_up_steps)
        self._last_substeps = substeps
        previous_anchors = self._previous_kinematic_anchors or anchors
        for step_index in range(substeps):
            # VMDの1更新分を瞬間移動せず、固定刻みごとの中間姿勢をBulletへ渡す。
            # 速度を消さないため、運動学的剛体の移動速度をBulletが導出できる。
            step_anchors = _interpolate_kinematic_anchors(
                previous_anchors,
                anchors,
                self._kinematic_body_indexes,
                (step_index + 1) / substeps,
            )
            self._update_kinematic_bodies(step_anchors, hand_transforms)
            self._update_joint_springs()
            self._world.do_physics(
                self._fixed_step_seconds,
                0,
                self._fixed_step_seconds,
            )
            self._stabilize_bodies(anchors)
            self._accumulator -= self._fixed_step_seconds
        if available_steps > self._max_catch_up_steps:
            dropped = self._accumulator - min(self._accumulator, self._fixed_step_seconds)
            self._dropped_seconds += max(0.0, dropped)
            self._accumulator = min(self._accumulator, self._fixed_step_seconds)
        if substeps:
            self._previous_kinematic_anchors = anchors
            self._last_pose = self._bone_pose(preview, anchors)
        return self._last_pose

    def _build_world(
        self,
        preview: UiAvatarModelPreview,
        bone_transforms: dict[int, BoneTransform],
    ) -> None:
        world = BulletWorld()
        world.set_gravity(LVector3f(*_GRAVITY))
        world.set_filter_callback(PythonCallbackObject(self._filter_collision))
        self._world = world
        anchors = _body_anchors(preview, bone_transforms)
        for index, body in enumerate(preview.rigid_bodies):
            node_name = f"pmx_body_{index}"
            node = BulletRigidBodyNode(node_name)
            shape = _make_shape(body)
            shape.set_margin(min(0.03, max(0.003, _body_radius(body) * 0.03)))
            node.add_shape(shape)
            node.set_mass(0.0 if body.mode == 0 else body.mass)
            node.set_linear_damping(body.linear_damping)
            node.set_angular_damping(body.angular_damping)
            node.set_restitution(body.restitution)
            node.set_friction(body.friction)
            if body.mode == 0:
                node.set_kinematic(True)
            path = self._root.attach_new_node(node)
            # ジョイントの局所座標を正しく作るため、剛体とジョイントはまず
            # PMX基準姿勢で構築する。現在姿勢への移動は制約作成後に行う。
            local_rotation = rotation_matrix(*body.rotation)
            path.set_transform(_transform_state(body.position, local_rotation))
            # 画面描画を経由しないワーカー内でも、初期座標をBullet本体へ即時反映する。
            node.set_transform_dirty()
            world.attach_rigid_body(node)
            if body.mode != 0:
                node.set_active(True)
            local_position = (
                subtract(body.position, preview.bones[body.bone_index].position)
                if 0 <= body.bone_index < len(preview.bones)
                else (0.0, 0.0, 0.0)
            )
            self._bodies.append(
                _BulletBody(
                    body,
                    node,
                    path,
                    local_position,
                    local_rotation,
                    matrix_transpose(local_rotation),
                    max(_MAX_BODY_OFFSET, _body_radius(body) * 5.0),
                )
            )
            self._body_sources_by_node_name[node_name] = body

        self._kinematic_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode == 0
        )
        self._aligned_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode == 2
        )
        self._dynamic_body_indexes = tuple(
            index for index, body in enumerate(self._bodies) if body.source.mode != 0
        )
        physics_bones = dict.fromkeys(
            self._bodies[index].source.bone_index
            for index in self._dynamic_body_indexes
            if 0 <= self._bodies[index].source.bone_index < len(preview.bones)
        )
        self._physics_bone_order = tuple(
            sorted(physics_bones, key=lambda index: _bone_depth(preview, index))
        )

        self._build_hand_colliders(preview, bone_transforms)
        self._build_thin_cloth_colliders(preview)

        for joint in preview.physics_joints:
            if not (
                0 <= joint.rigid_body_a < len(self._bodies)
                and 0 <= joint.rigid_body_b < len(self._bodies)
            ):
                continue
            first = self._bodies[joint.rigid_body_a]
            second = self._bodies[joint.rigid_body_b]
            joint_world = _transform_state(joint.position, rotation_matrix(*joint.rotation))
            frame_a = first.path.get_transform().invert_compose(joint_world)
            frame_b = second.path.get_transform().invert_compose(joint_world)
            constraint = BulletGenericConstraint(
                first.node,
                second.node,
                frame_a,
                frame_b,
                True,
            )
            for axis in range(3):
                constraint.set_linear_limit(
                    axis,
                    joint.translation_min[axis],
                    joint.translation_max[axis],
                )
                constraint.set_angular_limit(
                    axis,
                    joint.rotation_min[axis],
                    joint.rotation_max[axis],
                )
                rotational = constraint.get_rotational_limit_motor(axis)
                rotational.set_damping(0.85)
                rotational.set_softness(0.9)
            translational = constraint.get_translational_limit_motor()
            translational.set_damping(0.85)
            translational.set_softness(0.9)
            world.attach_constraint(constraint, True)
            translation_springs = tuple(
                (
                    axis,
                    max(0.0, joint.spring_translation[axis])
                    * _TRANSLATION_SPRING_COMPATIBILITY_SCALE,
                )
                for axis in range(3)
                if joint.spring_translation[axis] > 0.0
            )
            rotation_springs = tuple(
                (axis, max(0.0, joint.spring_rotation[axis]))
                for axis in range(3)
                if joint.spring_rotation[axis] > 0.0
            )
            self._joints.append(
                _BulletJoint(
                    joint,
                    constraint,
                    translation_springs,
                    rotation_springs,
                )
            )
        for index, body in enumerate(self._bodies):
            body.path.set_transform(
                _transform_state(anchors[index].position, anchors[index].rotation)
            )
            body.node.set_transform_dirty()
            body.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
            body.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
            body.node.set_active(True)
        self._settle_initial_pose(preview, bone_transforms)
        self._previous_kinematic_anchors = anchors

    def _build_hand_colliders(
        self,
        preview: UiAvatarModelPreview,
        bone_transforms: dict[int, BoneTransform],
    ) -> None:
        """既存PMXに手剛体がない側だけ、手首から手のひら形状を作る。"""

        if self._world is None:
            return
        existing_bones = {
            body.bone_index
            for body in preview.rigid_bodies
            if 0 <= body.bone_index < len(preview.bones)
            and any(word in body.name for word in ("手首", "手のひら", "掌"))
        }
        indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
        for side in ("右", "左"):
            wrist_index = indexes.get(f"{side}手首")
            finger_index = indexes.get(f"{side}中指１")
            if wrist_index is None or finger_index is None or wrist_index in existing_bones:
                continue
            wrist = preview.bones[wrist_index]
            direction = subtract(preview.bones[finger_index].position, wrist.position)
            hand_length = length(direction)
            if hand_length < 0.2:
                continue
            radius = max(0.18, min(0.34, hand_length * 0.30))
            cylinder_height = max(0.08, hand_length - radius * 2.0)
            local_center = tuple(value * 0.52 for value in direction)
            local_rotation = _rotation_with_y_axis(direction)
            source = UiAvatarRigidBody(
                bone_index=wrist_index,
                position=wrist.position,
                mass=0.0,
                linear_damping=1.0,
                mode=0,
                name=f"補助{side}手のひら",
                collision_group=2,
                # 補助手のひらは、既存の手剛体が欠ける場合だけスカートへ接触させる。
                non_collision_mask=0,
                shape=2,
                size=(radius, cylinder_height, 0.0),
            )
            node_name = f"pmx_aux_hand_{side}"
            node = BulletRigidBodyNode(node_name)
            shape = _make_shape(source)
            shape.set_margin(min(0.02, max(0.003, radius * 0.04)))
            node.add_shape(shape)
            node.set_mass(0.0)
            node.set_kinematic(True)
            # PMXにない補助手のひらは、作者物理へ影響させない。
            node.set_collision_response(False)
            path = self._root.attach_new_node(node)
            collider = _HandCollider(
                source=source,
                node=node,
                path=path,
                bone_index=wrist_index,
                local_center=local_center,
                local_rotation=local_rotation,
            )
            transform = _hand_collider_transform(collider, bone_transforms)
            path.set_transform(_transform_state(*transform))
            node.set_transform_dirty()
            self._world.attach_rigid_body(node)
            self._body_sources_by_node_name[node_name] = source
            self._hand_colliders.append(collider)

    def _build_thin_cloth_colliders(self, preview: UiAvatarModelPreview) -> None:
        """作者マスクを変えず、指定された薄布だけを接触監視する。"""

        if self._world is None or not preview.thin_cloth_contact_enabled:
            return
        source_patterns = tuple(
            pattern.casefold() for pattern in preview.thin_cloth_source_body_patterns
        )
        self._thin_cloth_include_auxiliary_hands = any(
            "手" in pattern or "掌" in pattern for pattern in source_patterns
        )
        target_patterns = tuple(
            pattern.casefold() for pattern in preview.thin_cloth_target_body_patterns
        )
        target_indexes = tuple(
            index
            for index, body in enumerate(self._bodies)
            if body.source.mode != 0
            and any(pattern in body.source.name.casefold() for pattern in target_patterns)
        )
        if not source_patterns or not target_indexes:
            return
        for index, body in enumerate(self._bodies):
            if body.source.mode != 0 or not any(
                pattern in body.source.name.casefold() for pattern in source_patterns
            ):
                continue
            node_name = f"thin_cloth_sensor_{index}"
            node = BulletRigidBodyNode(node_name)
            shape = _make_shape(body.source)
            shape.set_margin(min(0.01, max(0.001, _body_radius(body.source) * 0.01)))
            node.add_shape(shape)
            node.set_mass(0.0)
            node.set_kinematic(True)
            node.set_collision_response(False)
            sensor_source = UiAvatarRigidBody(
                bone_index=body.source.bone_index,
                position=body.source.position,
                mass=0.0,
                linear_damping=0.0,
                mode=0,
                name=node_name,
                collision_group=0,
                non_collision_mask=0,
            )
            path = self._root.attach_new_node(node)
            path.set_transform(body.path.get_transform())
            node.set_transform_dirty()
            self._world.attach_rigid_body(node)
            self._body_sources_by_node_name[node_name] = sensor_source
            self._thin_cloth_colliders.append(
                _ThinClothCollider(
                    node=node,
                    path=path,
                    source_body_index=index,
                    target_body_indexes=target_indexes,
                    maximum_correction_per_step=preview.thin_cloth_maximum_correction_per_step,
                )
            )

    def _settle_initial_pose(
        self,
        preview: UiAvatarModelPreview,
        bone_transforms: dict[int, BoneTransform],
    ) -> None:
        """描画前にPMX制約を安定させ、初回フレームの衣装発散を防ぐ。"""

        if self._world is None:
            return
        anchors = _body_anchors(preview, bone_transforms, self._bodies)
        hand_transforms = _hand_collider_transforms(self._hand_colliders, bone_transforms)
        self._enable_joint_springs()
        for _ in range(_INITIAL_SETTLE_STEPS):
            self._update_kinematic_bodies(anchors, hand_transforms)
            self._update_joint_springs()
            self._world.do_physics(
                self._fixed_step_seconds,
                0,
                self._fixed_step_seconds,
            )
            self._stabilize_bodies(anchors)
        for body in self._bodies:
            if body.source.mode != 0:
                body.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
                body.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
                body.node.set_active(True)
        for item in self._joints:
            # 初期安定化で生じた接触ずれをばねの基準にしてはいけない。
            # PMXジョイントの構築時基準（ゼロ）を維持する。
            item.equilibrium_translation = (0.0, 0.0, 0.0)
            item.equilibrium_rotation = (0.0, 0.0, 0.0)
        self._enable_joint_springs()

    def _enable_joint_springs(self) -> None:
        """PMXジョイントのばねを、初期化中から通常更新まで有効にする。"""

        for item in self._joints:
            translational = item.constraint.get_translational_limit_motor()
            translation_axes = {axis for axis, _stiffness in item.translation_springs}
            rotation_axes = {axis for axis, _stiffness in item.rotation_springs}
            for axis in range(3):
                translational.set_motor_enabled(axis, axis in translation_axes)
                item.constraint.get_rotational_limit_motor(axis).set_motor_enabled(
                    axis in rotation_axes
                )

    def _filter_collision(self, data: BulletFilterCallbackData) -> None:
        """PMX作者が指定した衝突グループと非衝突マスクだけを使う。"""

        first = self._body_sources_by_node_name.get(data.get_node_0().get_name())
        second = self._body_sources_by_node_name.get(data.get_node_1().get_name())
        if first is None or second is None:
            data.set_collide(False)
            return
        data.set_collide(_authored_collision(first, second))

    def _update_joint_springs(self) -> None:
        """Panda3D未公開の6自由度ばねを、Bullet標準モーターで再現する。"""

        for item in self._joints:
            constraint = item.constraint
            translational = constraint.get_translational_limit_motor()
            current_diff = translational.get_current_diff()
            target_velocity = [0.0, 0.0, 0.0]
            maximum_force = [0.0, 0.0, 0.0]
            for axis, stiffness in item.translation_springs:
                # btGeneric6DofSpringConstraint::internalUpdateSprings と同じ符号。
                displacement = float(current_diff[axis]) - item.equilibrium_translation[axis]
                force = displacement * stiffness
                target_velocity[axis] = self._spring_velocity_factor * force
                maximum_force[axis] = abs(force)
            translational.set_target_velocity(LVector3f(*target_velocity))
            translational.set_max_motor_force(LVector3f(*maximum_force))

            for axis, stiffness in item.rotation_springs:
                rotational = constraint.get_rotational_limit_motor(axis)
                displacement = float(constraint.get_angle(axis)) - item.equilibrium_rotation[axis]
                force = -displacement * stiffness
                rotational.set_target_velocity(self._spring_velocity_factor * force)
                rotational.set_max_motor_force(abs(force))

    def _update_kinematic_bodies(
        self,
        anchors: tuple[_BodyAnchor, ...],
        hand_transforms: tuple[tuple[Vector3, Matrix3], ...] = (),
        *,
        clear_kinematic_velocity: bool = False,
    ) -> None:
        for index in self._kinematic_body_indexes:
            body = self._bodies[index]
            anchor = anchors[index]
            body.path.set_transform(_transform_state(anchor.position, anchor.rotation))
            body.node.set_transform_dirty()
            if clear_kinematic_velocity:
                body.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
                body.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
            body.node.set_active(True)
        for index in self._aligned_body_indexes:
            body = self._bodies[index]
            anchor = anchors[index]
            current = body.path.get_transform()
            body.path.set_transform(
                TransformState.make_pos_quat_scale(
                    LPoint3f(*anchor.position),
                    current.get_quat(),
                    LVector3f(1.0, 1.0, 1.0),
                )
            )
            body.node.set_transform_dirty()
            body.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
            body.node.set_active(True)
        for collider, (position, rotation) in zip(
            self._hand_colliders,
            hand_transforms,
            strict=True,
        ):
            collider.path.set_transform(_transform_state(position, rotation))
            collider.node.set_transform_dirty()
            collider.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
            collider.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
            collider.node.set_active(True)
        for collider in self._thin_cloth_colliders:
            source = self._bodies[collider.source_body_index]
            collider.path.set_transform(source.path.get_transform())
            collider.node.set_transform_dirty()
            collider.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
            collider.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
            collider.node.set_active(True)

    def _resolve_thin_cloth_contacts(self) -> None:
        """旧方式の互換用入口。剛体中心による補正は行わない。"""

    def thin_cloth_source_surfaces(self) -> tuple[ThinClothSourceSurface, ...]:
        """現在の腕・袖剛体を、描画用の不変な数値スナップショットで返す。"""

        surfaces: list[ThinClothSourceSurface] = []
        for collider in self._thin_cloth_colliders:
            source = self._bodies[collider.source_body_index]
            surfaces.append(_thin_cloth_surface(source.source, source.path))
        if self._thin_cloth_include_auxiliary_hands:
            surfaces.extend(
                _thin_cloth_surface(collider.source, collider.path)
                for collider in self._hand_colliders
            )
        return tuple(surfaces)

    def _stabilize_bodies(self, anchors: tuple[_BodyAnchor, ...]) -> None:
        for index in self._dynamic_body_indexes:
            body = self._bodies[index]
            position = _point(body.path.get_pos())
            finite = all(math.isfinite(value) for value in position)
            offset = length(subtract(position, anchors[index].position)) if finite else math.inf
            if finite:
                self._maximum_body_offset_observed = max(
                    self._maximum_body_offset_observed,
                    offset,
                )
            if not finite or offset > body.maximum_offset:
                self._stabilization_reset_count += 1
                if finite:
                    self._offset_limit_reset_count += 1
                else:
                    self._non_finite_reset_count += 1
                body.path.set_transform(
                    _transform_state(anchors[index].position, anchors[index].rotation)
                )
                body.node.set_transform_dirty()
                body.node.set_linear_velocity(LVector3f(0.0, 0.0, 0.0))
                body.node.set_angular_velocity(LVector3f(0.0, 0.0, 0.0))
                body.node.set_active(True)

    def auxiliary_hand_contact_count(self) -> int:
        """補助手のひらで検出中の接触点数を返す。"""

        if self._world is None:
            return 0
        return sum(
            self._world.contact_test(collider.node).get_num_contacts()
            for collider in self._hand_colliders
        )

    def reset_diagnostic_counters(self) -> None:
        """物理状態を変えず、基準測定用の観測値だけを初期化する。"""

        self._maximum_body_offset_observed = 0.0
        self._stabilization_reset_count = 0
        self._non_finite_reset_count = 0
        self._offset_limit_reset_count = 0

    def _bone_pose(
        self,
        preview: UiAvatarModelPreview,
        anchors: tuple[_BodyAnchor, ...],
    ) -> PhysicsPose:
        desired_transforms: dict[int, BoneTransform] = {}
        desired_modes: dict[int, int] = {}
        anchor_by_bone: dict[int, _BodyAnchor] = {}
        for index in self._dynamic_body_indexes:
            body = self._bodies[index]
            source = body.source
            bone_index = source.bone_index
            if not 0 <= bone_index < len(preview.bones):
                continue
            anchor = anchors[index]
            body_rotation = _matrix_from_quat(body.path.get_quat())
            desired_rotation = matrix_multiply(
                body_rotation,
                body.inverse_local_rotation,
            )
            desired_position = (
                subtract(
                    _point(body.path.get_pos()),
                    matrix_vector(desired_rotation, body.local_position),
                )
                if source.mode == 1
                else anchor.bone_position
            )
            desired_transforms[bone_index] = desired_rotation, desired_position
            desired_modes[bone_index] = source.mode
            anchor_by_bone[bone_index] = anchor

        translations: dict[int, Vector3] = {}
        rotations: dict[int, Vector3] = {}
        for bone_index in self._physics_bone_order:
            bone = preview.bones[bone_index]
            anchor = anchor_by_bone[bone_index]
            desired_rotation, desired_position = desired_transforms[bone_index]
            parent_index = bone.parent_index
            desired_parent_rotation, desired_parent_position = desired_transforms.get(
                parent_index,
                (anchor.parent_rotation, anchor.parent_position),
            )
            inverse_parent_rotation = matrix_transpose(anchor.parent_rotation)
            inverse_desired_parent_rotation = matrix_transpose(desired_parent_rotation)
            base_local_rotation = matrix_multiply(
                inverse_parent_rotation,
                anchor.bone_rotation,
            )
            desired_local_rotation = matrix_multiply(
                inverse_desired_parent_rotation,
                desired_rotation,
            )
            base_euler = _matrix_to_euler(base_local_rotation)
            desired_euler = _matrix_to_euler(desired_local_rotation)
            rotations[bone_index] = tuple(
                _wrap_angle(desired_euler[axis] - base_euler[axis]) for axis in range(3)
            )
            if desired_modes[bone_index] == 1:
                base_local_position = matrix_vector(
                    inverse_parent_rotation,
                    subtract(anchor.bone_position, anchor.parent_position),
                )
                desired_local_position = matrix_vector(
                    inverse_desired_parent_rotation,
                    subtract(desired_position, desired_parent_position),
                )
                translations[bone_index] = subtract(
                    desired_local_position,
                    base_local_position,
                )
        return PhysicsPose(translations=translations, rotations=rotations)


def _body_anchors(
    preview: UiAvatarModelPreview,
    transforms: dict[int, BoneTransform],
    prepared_bodies: list[_BulletBody] | None = None,
) -> tuple[_BodyAnchor, ...]:
    anchors: list[_BodyAnchor] = []
    for index, body in enumerate(preview.rigid_bodies):
        prepared = (
            prepared_bodies[index]
            if prepared_bodies is not None and index < len(prepared_bodies)
            else None
        )
        local_rotation = prepared.local_rotation if prepared else rotation_matrix(*body.rotation)
        if 0 <= body.bone_index < len(preview.bones) and body.bone_index in transforms:
            bone_rotation, bone_position = transforms[body.bone_index]
            parent_index = preview.bones[body.bone_index].parent_index
            parent_rotation = (
                transforms[parent_index][0]
                if 0 <= parent_index < len(preview.bones) and parent_index in transforms
                else identity_matrix()
            )
            parent_position = (
                transforms[parent_index][1]
                if 0 <= parent_index < len(preview.bones) and parent_index in transforms
                else (0.0, 0.0, 0.0)
            )
            local_offset = (
                prepared.local_position
                if prepared
                else subtract(body.position, preview.bones[body.bone_index].position)
            )
            anchors.append(
                _BodyAnchor(
                    add(bone_position, matrix_vector(bone_rotation, local_offset)),
                    matrix_multiply(bone_rotation, local_rotation),
                    bone_position,
                    bone_rotation,
                    parent_position,
                    parent_rotation,
                )
            )
        else:
            anchors.append(
                _BodyAnchor(
                    body.position,
                    local_rotation,
                    preview.bones[body.bone_index].position
                    if 0 <= body.bone_index < len(preview.bones)
                    else body.position,
                    identity_matrix(),
                    (0.0, 0.0, 0.0),
                    identity_matrix(),
                )
            )
    return tuple(anchors)


def _interpolate_kinematic_anchors(
    previous: tuple[_BodyAnchor, ...],
    current: tuple[_BodyAnchor, ...],
    kinematic_indexes: tuple[int, ...],
    ratio: float,
) -> tuple[_BodyAnchor, ...]:
    """固定刻みの運動学的剛体だけを、前回VMD姿勢から補間する。"""

    clamped_ratio = max(0.0, min(1.0, ratio))
    indexes = set(kinematic_indexes)
    result: list[_BodyAnchor] = []
    for index, anchor in enumerate(current):
        if index not in indexes or index >= len(previous):
            result.append(anchor)
            continue
        before = previous[index]
        position = tuple(
            before.position[axis] + (anchor.position[axis] - before.position[axis]) * clamped_ratio
            for axis in range(3)
        )
        result.append(
            _BodyAnchor(
                position,  # type: ignore[arg-type]
                _slerp_matrix(before.rotation, anchor.rotation, clamped_ratio),
                anchor.bone_position,
                anchor.bone_rotation,
                anchor.parent_position,
                anchor.parent_rotation,
            )
        )
    return tuple(result)


def _slerp_matrix(first: Matrix3, second: Matrix3, ratio: float) -> Matrix3:
    """回転行列を最短経路の球面補間でつなぐ。"""

    before = _quat_from_matrix(first)
    after = _quat_from_matrix(second)
    first_values = (
        float(before.get_r()),
        float(before.get_i()),
        float(before.get_j()),
        float(before.get_k()),
    )
    second_values = (
        float(after.get_r()),
        float(after.get_i()),
        float(after.get_j()),
        float(after.get_k()),
    )
    dot = sum(first_values[index] * second_values[index] for index in range(4))
    if dot < 0.0:
        second_values = tuple(-value for value in second_values)
        dot = -dot
    if dot > 0.9995:
        values = tuple(
            first_values[index] + (second_values[index] - first_values[index]) * ratio
            for index in range(4)
        )
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sine = math.sin(theta)
        before_weight = math.sin((1.0 - ratio) * theta) / sine
        after_weight = math.sin(ratio * theta) / sine
        values = tuple(
            first_values[index] * before_weight + second_values[index] * after_weight
            for index in range(4)
        )
    quaternion = LQuaternionf(*values)
    quaternion.normalize()
    return _matrix_from_quat(quaternion)


def _authored_collision(first: UiAvatarRigidBody, second: UiAvatarRigidBody) -> bool:
    if first.mode == 0 and second.mode == 0:
        return False
    return not bool(
        first.non_collision_mask & (1 << second.collision_group)
        or second.non_collision_mask & (1 << first.collision_group)
    )


def _hand_collider_transforms(
    colliders: list[_HandCollider],
    transforms: dict[int, BoneTransform],
) -> tuple[tuple[Vector3, Matrix3], ...]:
    return tuple(_hand_collider_transform(collider, transforms) for collider in colliders)


def _hand_collider_transform(
    collider: _HandCollider,
    transforms: dict[int, BoneTransform],
) -> tuple[Vector3, Matrix3]:
    rotation, position = transforms.get(
        collider.bone_index,
        (identity_matrix(), collider.source.position),
    )
    return (
        add(position, matrix_vector(rotation, collider.local_center)),
        matrix_multiply(rotation, collider.local_rotation),
    )


def _rotation_with_y_axis(direction: Vector3) -> Matrix3:
    """カプセルのY軸を手首から指へ向ける回転行列を作る。"""

    y_axis = normalize(direction)
    reference = (0.0, 0.0, 1.0) if abs(y_axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    x_axis = normalize(cross(y_axis, reference))
    z_axis = normalize(cross(x_axis, y_axis))
    if min(length(x_axis), length(y_axis), length(z_axis)) < 0.00001:
        return identity_matrix()
    # 各軸を列として並べ、ローカルYが手の長手方向へ写るようにする。
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def _bone_depth(preview: UiAvatarModelPreview, bone_index: int) -> int:
    depth = 0
    visited: set[int] = set()
    while 0 <= bone_index < len(preview.bones) and bone_index not in visited:
        visited.add(bone_index)
        bone_index = preview.bones[bone_index].parent_index
        depth += 1
    return depth


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _make_shape(body: UiAvatarRigidBody):  # type: ignore[no-untyped-def]
    if body.shape == 0:
        return BulletSphereShape(body.size[0])
    if body.shape == 1:
        return BulletBoxShape(LVector3f(*body.size))
    return BulletCapsuleShape(body.size[0], body.size[1], YUp)


def _body_radius(body: UiAvatarRigidBody) -> float:
    if body.shape == 0:
        return body.size[0]
    if body.shape == 1:
        return length(body.size)
    return body.size[0] + body.size[1] * 0.5


def _thin_cloth_surface(source: UiAvatarRigidBody, path: NodePath) -> ThinClothSourceSurface:
    """物理応答を使わず、表示補正用の形状スナップショットを作る。"""

    if source.shape == 0:
        radius, half_length = source.size[0], 0.0
    elif source.shape == 1:
        radius, half_length = max(source.size[0], source.size[2]), source.size[1]
    else:
        radius, half_length = source.size[0], source.size[1] * 0.5
    axis = _point(path.get_quat().get_up())
    axis = (0.0, 1.0, 0.0) if length(axis) <= 0.000001 else normalize(axis)
    return ThinClothSourceSurface(
        position=_point(path.get_pos()),
        axis=axis,
        radius=max(0.0, radius),
        half_length=max(0.0, half_length),
    )


def _transform_state(position: Vector3, rotation: Matrix3) -> TransformState:
    return TransformState.make_pos_quat_scale(
        LPoint3f(*position),
        _quat_from_matrix(rotation),
        LVector3f(1.0, 1.0, 1.0),
    )


def _quat_from_matrix(matrix: Matrix3) -> LQuaternionf:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    quaternion = LQuaternionf(w, x, y, z)
    quaternion.normalize()
    return quaternion


def _matrix_from_quat(quaternion: LQuaternionf) -> Matrix3:
    w = float(quaternion.get_r())
    x = float(quaternion.get_i())
    y = float(quaternion.get_j())
    z = float(quaternion.get_k())
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _matrix_to_euler(matrix: Matrix3) -> Vector3:
    yaw = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(yaw)) > 0.00001:
        pitch = math.atan2(matrix[2][1], matrix[2][2])
        roll = math.atan2(matrix[1][0], matrix[0][0])
    else:
        pitch = math.atan2(-matrix[1][2], matrix[1][1])
        roll = 0.0
    return pitch, yaw, roll


def _point(value: LPoint3f) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])
