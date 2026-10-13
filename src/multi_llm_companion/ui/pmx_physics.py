"""第40段階のPMX剛体・衝突・6自由度ジョイント物理。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from multi_llm_companion.ui.models import (
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
)
from multi_llm_companion.ui.pmx_pose import (
    BoneTransform,
    add,
    clamp_vector,
    identity_matrix,
    length,
    matrix_transpose,
    matrix_vector,
    rotation_matrix,
    subtract,
)

type Vector3 = tuple[float, float, float]

_GRAVITY: Vector3 = (0.0, -0.98, 0.0)
_FIXED_STEP_SECONDS = 1.0 / 60.0
_MAX_CATCH_UP_STEPS = 4
_SOLVER_ITERATIONS = 1
_MAX_LINEAR_SPEED = 8.0
_MAX_ANGULAR_SPEED = 12.0
_MAX_CORRECTION_PER_ITERATION = 0.08


@dataclass(frozen=True, slots=True)
class PhysicsPose:
    """物理演算からボーンへ加算する移動量と回転量。"""

    translations: dict[int, Vector3] = field(default_factory=dict)
    rotations: dict[int, Vector3] = field(default_factory=dict)


@dataclass(slots=True)
class _BodyState:
    position: Vector3
    rotation: Vector3
    velocity: Vector3 = (0.0, 0.0, 0.0)
    angular_velocity: Vector3 = (0.0, 0.0, 0.0)
    aabb_rotation: Vector3 | None = None
    aabb_extent: Vector3 | None = None


@dataclass(frozen=True, slots=True)
class _BodyAnchor:
    position: Vector3
    rotation: Vector3
    parent_rotation: tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True, slots=True)
class _Aabb:
    minimum: Vector3
    maximum: Vector3


class PmxPhysicsSimulator:
    """PMX剛体を固定刻みで更新する、発散防止付きの軽量ソルバー。"""

    def __init__(
        self,
        *,
        fixed_step_seconds: float = _FIXED_STEP_SECONDS,
        max_catch_up_steps: int = _MAX_CATCH_UP_STEPS,
    ) -> None:
        self._preview: UiAvatarModelPreview | None = None
        self._states: list[_BodyState] = []
        self._joint_bases: list[
            tuple[
                tuple[Vector3, Vector3, Vector3],
                tuple[Vector3, Vector3, Vector3],
            ]
        ] = []
        self._rest_penetrations: dict[tuple[int, int], float] = {}
        self._accumulator = 0.0
        self._fixed_step_seconds = max(1.0 / 240.0, min(fixed_step_seconds, 1.0 / 20.0))
        self._max_catch_up_steps = max(1, min(max_catch_up_steps, 8))
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

    def reset(self, preview: UiAvatarModelPreview | None) -> None:
        self._preview = preview
        self._states = (
            [_BodyState(body.position, body.rotation) for body in preview.rigid_bodies]
            if preview is not None
            else []
        )
        self._joint_bases = (
            [
                (basis, matrix_transpose(basis))
                for joint in preview.physics_joints
                for basis in (rotation_matrix(*joint.rotation),)
            ]
            if preview is not None
            else []
        )
        self._rest_penetrations = (
            _rest_penetrations(preview, self._states) if preview is not None else {}
        )
        self._accumulator = 0.0
        self._last_pose = PhysicsPose()
        self._last_substeps = 0
        self._dropped_seconds = 0.0

    def step(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
    ) -> dict[int, Vector3]:
        """第27段階との互換用。新しいコードは`step_pose`を使う。"""

        return self.step_pose(preview, delta_seconds, enabled).translations

    def step_pose(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
        bone_transforms: dict[int, BoneTransform] | None = None,
    ) -> PhysicsPose:
        """固定時間刻みで進め、ボーンへ加算する物理姿勢を返す。"""

        if preview is not self._preview:
            self.reset(preview)
        if preview is None or not enabled or not preview.rigid_bodies:
            if self._states and (not enabled or preview is None):
                self.reset(preview)
            return PhysicsPose()

        anchors = _body_anchors(preview, bone_transforms or {})
        safe_delta = max(0.0, min(delta_seconds, 0.25))
        self._accumulator += safe_delta
        available_steps = int(self._accumulator / self._fixed_step_seconds)
        substeps = min(available_steps, self._max_catch_up_steps)
        self._last_substeps = substeps
        for step_index in range(substeps):
            self._simulate_fixed_step(
                preview,
                anchors,
                self._fixed_step_seconds,
                resolve_collisions=step_index == substeps - 1,
            )
            self._accumulator -= self._fixed_step_seconds
        if available_steps > self._max_catch_up_steps:
            dropped = self._accumulator - min(self._accumulator, self._fixed_step_seconds)
            self._dropped_seconds += max(0.0, dropped)
            self._accumulator = min(self._accumulator, self._fixed_step_seconds)
        if substeps:
            self._last_pose = self._bone_pose(preview, anchors)
        return self._last_pose

    def _simulate_fixed_step(
        self,
        preview: UiAvatarModelPreview,
        anchors: tuple[_BodyAnchor, ...],
        delta: float,
        *,
        resolve_collisions: bool,
    ) -> None:
        previous_positions = [state.position for state in self._states]
        previous_rotations = [state.rotation for state in self._states]
        for index, body in enumerate(preview.rigid_bodies):
            state = self._states[index]
            anchor = anchors[index]
            if body.mode == 0:
                state.position = anchor.position
                state.rotation = anchor.rotation
                state.velocity = (0.0, 0.0, 0.0)
                state.angular_velocity = (0.0, 0.0, 0.0)
                continue
            linear_factor = max(0.0, 1.0 - body.linear_damping * delta * 8.0)
            angular_factor = max(0.0, 1.0 - body.angular_damping * delta * 8.0)
            state.velocity = scale(add(state.velocity, scale(_GRAVITY, delta)), linear_factor)
            state.angular_velocity = scale(state.angular_velocity, angular_factor)
            state.position = add(state.position, scale(state.velocity, delta))
            state.rotation = add(state.rotation, scale(state.angular_velocity, delta))
            if body.mode == 2:
                state.position = anchor.position
                state.velocity = (0.0, 0.0, 0.0)

        for _ in range(_SOLVER_ITERATIONS):
            for joint_index, joint in enumerate(preview.physics_joints):
                basis, inverse_basis = self._joint_bases[joint_index]
                self._apply_joint(preview, joint, delta, basis, inverse_basis)
        if resolve_collisions:
            self._resolve_collisions(preview)

        for index, body in enumerate(preview.rigid_bodies):
            if body.mode == 0:
                continue
            state = self._states[index]
            if body.mode == 1:
                state.velocity = clamp_length(
                    scale(subtract(state.position, previous_positions[index]), 1.0 / delta),
                    _MAX_LINEAR_SPEED,
                )
            state.angular_velocity = clamp_length(
                scale(subtract(state.rotation, previous_rotations[index]), 1.0 / delta),
                _MAX_ANGULAR_SPEED,
            )
            self._stabilize_body(body, state, anchors[index])

    def _apply_joint(
        self,
        preview: UiAvatarModelPreview,
        joint: UiAvatarPhysicsJoint,
        delta: float,
        basis: tuple[Vector3, Vector3, Vector3],
        inverse_basis: tuple[Vector3, Vector3, Vector3],
    ) -> None:
        first_index = joint.rigid_body_a
        second_index = joint.rigid_body_b
        if not (0 <= first_index < len(self._states) and 0 <= second_index < len(self._states)):
            return
        first_body = preview.rigid_bodies[first_index]
        second_body = preview.rigid_bodies[second_index]
        first_state = self._states[first_index]
        second_state = self._states[second_index]
        rest_translation = subtract(second_body.position, first_body.position)
        current_translation = subtract(second_state.position, first_state.position)
        local_delta = matrix_vector(
            inverse_basis,
            subtract(current_translation, rest_translation),
        )
        limited_translation = clamp_vector(
            local_delta,
            joint.translation_min,
            joint.translation_max,
        )
        translation_error = subtract(local_delta, limited_translation)
        translation_spring = tuple(
            local_delta[axis] * min(0.25, joint.spring_translation[axis] * delta * delta)
            for axis in range(3)
        )
        correction = clamp_length(
            matrix_vector(basis, add(translation_error, translation_spring)),
            _MAX_CORRECTION_PER_ITERATION,
        )
        self._apply_pair_position(first_body, first_state, second_body, second_state, correction)

        rest_rotation = subtract(second_body.rotation, first_body.rotation)
        current_rotation = subtract(second_state.rotation, first_state.rotation)
        world_rotation_delta = subtract(current_rotation, rest_rotation)
        if length(world_rotation_delta) < 0.000001:
            return
        local_rotation_delta = matrix_vector(inverse_basis, world_rotation_delta)
        limited_rotation = clamp_vector(
            local_rotation_delta,
            joint.rotation_min,
            joint.rotation_max,
        )
        rotation_error = subtract(local_rotation_delta, limited_rotation)
        rotation_spring = tuple(
            local_rotation_delta[axis] * min(0.2, joint.spring_rotation[axis] * delta * delta)
            for axis in range(3)
        )
        angular_correction = clamp_length(
            matrix_vector(basis, add(rotation_error, rotation_spring)),
            0.12,
        )
        self._apply_pair_rotation(
            first_body,
            first_state,
            second_body,
            second_state,
            angular_correction,
        )

    def _apply_pair_position(
        self,
        first_body: UiAvatarRigidBody,
        first_state: _BodyState,
        second_body: UiAvatarRigidBody,
        second_state: _BodyState,
        correction: Vector3,
    ) -> None:
        first_inverse = _inverse_mass(first_body, translation=True)
        second_inverse = _inverse_mass(second_body, translation=True)
        total = first_inverse + second_inverse
        if total <= 0.0:
            return
        if first_inverse > 0.0:
            first_state.position = add(
                first_state.position, scale(correction, first_inverse / total)
            )
        if second_inverse > 0.0:
            second_state.position = subtract(
                second_state.position,
                scale(correction, second_inverse / total),
            )

    def _apply_pair_rotation(
        self,
        first_body: UiAvatarRigidBody,
        first_state: _BodyState,
        second_body: UiAvatarRigidBody,
        second_state: _BodyState,
        correction: Vector3,
    ) -> None:
        first_inverse = _inverse_mass(first_body, translation=False)
        second_inverse = _inverse_mass(second_body, translation=False)
        total = first_inverse + second_inverse
        if total <= 0.0:
            return
        if first_inverse > 0.0:
            first_state.rotation = add(
                first_state.rotation, scale(correction, first_inverse / total)
            )
        if second_inverse > 0.0:
            second_state.rotation = subtract(
                second_state.rotation,
                scale(correction, second_inverse / total),
            )

    def _resolve_collisions(self, preview: UiAvatarModelPreview) -> None:
        aabbs = tuple(
            _body_aabb(body, self._states[index]) for index, body in enumerate(preview.rigid_bodies)
        )
        ordered = sorted(range(len(aabbs)), key=lambda index: aabbs[index].minimum[0])
        active: list[int] = []
        for second_index in ordered:
            second_aabb = aabbs[second_index]
            active = [
                index for index in active if aabbs[index].maximum[0] >= second_aabb.minimum[0]
            ]
            for first_index in active:
                if _can_collide(
                    preview.rigid_bodies[first_index],
                    preview.rigid_bodies[second_index],
                ):
                    self._resolve_collision_pair(
                        preview,
                        first_index,
                        second_index,
                        aabbs[first_index],
                        second_aabb,
                    )
            active.append(second_index)

    def _resolve_collision_pair(
        self,
        preview: UiAvatarModelPreview,
        first_index: int,
        second_index: int,
        first_aabb: _Aabb,
        second_aabb: _Aabb,
    ) -> None:
        overlap_x = min(first_aabb.maximum[0], second_aabb.maximum[0]) - max(
            first_aabb.minimum[0], second_aabb.minimum[0]
        )
        if overlap_x <= 0.0:
            return
        overlap_y = min(first_aabb.maximum[1], second_aabb.maximum[1]) - max(
            first_aabb.minimum[1], second_aabb.minimum[1]
        )
        if overlap_y <= 0.0:
            return
        overlap_z = min(first_aabb.maximum[2], second_aabb.maximum[2]) - max(
            first_aabb.minimum[2], second_aabb.minimum[2]
        )
        if overlap_z <= 0.0:
            return
        first_body = preview.rigid_bodies[first_index]
        second_body = preview.rigid_bodies[second_index]
        first_inverse = _inverse_mass(first_body, translation=True)
        second_inverse = _inverse_mass(second_body, translation=True)
        total = first_inverse + second_inverse
        if total <= 0.0:
            return
        overlaps = (overlap_x, overlap_y, overlap_z)
        axis = (
            0
            if overlap_x <= overlap_y and overlap_x <= overlap_z
            else (1 if overlap_y <= overlap_z else 2)
        )
        direction = (
            1.0
            if self._states[second_index].position[axis] >= self._states[first_index].position[axis]
            else -1.0
        )
        normal = tuple(direction if index == axis else 0.0 for index in range(3))
        pair = (min(first_index, second_index), max(first_index, second_index))
        allowed_penetration = self._rest_penetrations.get(pair, 0.0)
        excess_penetration = overlaps[axis] - allowed_penetration
        if excess_penetration <= 0.00001:
            return
        correction = scale(
            normal,
            min(excess_penetration, _MAX_CORRECTION_PER_ITERATION),
        )
        first_state = self._states[first_index]
        second_state = self._states[second_index]
        if first_inverse > 0.0:
            first_state.position = subtract(
                first_state.position, scale(correction, first_inverse / total)
            )
        if second_inverse > 0.0:
            second_state.position = add(
                second_state.position, scale(correction, second_inverse / total)
            )

        relative_velocity = subtract(second_state.velocity, first_state.velocity)
        normal_speed = sum(relative_velocity[index] * normal[index] for index in range(3))
        if normal_speed >= 0.0:
            return
        restitution = min(first_body.restitution, second_body.restitution)
        impulse = -(1.0 + restitution) * normal_speed / total
        if first_inverse > 0.0:
            first_state.velocity = subtract(
                first_state.velocity,
                scale(normal, impulse * first_inverse),
            )
        if second_inverse > 0.0:
            second_state.velocity = add(
                second_state.velocity,
                scale(normal, impulse * second_inverse),
            )
        friction = min(1.0, (first_body.friction + second_body.friction) * 0.5)
        first_state.velocity = _dampen_tangent(first_state.velocity, normal, friction)
        second_state.velocity = _dampen_tangent(second_state.velocity, normal, friction)

    def _stabilize_body(
        self,
        body: UiAvatarRigidBody,
        state: _BodyState,
        anchor: _BodyAnchor,
    ) -> None:
        radius = _body_radius(body)
        maximum_offset = max(0.25, min(4.0, radius * 4.0))
        values = (*state.position, *state.rotation, *state.velocity, *state.angular_velocity)
        if (
            not all(math.isfinite(value) for value in values)
            or length(subtract(state.position, anchor.position)) > maximum_offset
        ):
            state.position = anchor.position
            state.rotation = anchor.rotation
            state.velocity = (0.0, 0.0, 0.0)
            state.angular_velocity = (0.0, 0.0, 0.0)

    def _bone_pose(
        self,
        preview: UiAvatarModelPreview,
        anchors: tuple[_BodyAnchor, ...],
    ) -> PhysicsPose:
        translations: dict[int, Vector3] = {}
        rotations: dict[int, Vector3] = {}
        translation_counts: dict[int, int] = {}
        rotation_counts: dict[int, int] = {}
        for index, body in enumerate(preview.rigid_bodies):
            bone_index = body.bone_index
            if body.mode == 0 or not 0 <= bone_index < len(preview.bones):
                continue
            if body.mode == 1:
                world_offset = subtract(self._states[index].position, anchors[index].position)
                offset = matrix_vector(
                    matrix_transpose(anchors[index].parent_rotation),
                    world_offset,
                )
                translations[bone_index] = add(
                    translations.get(bone_index, (0.0, 0.0, 0.0)), offset
                )
                translation_counts[bone_index] = translation_counts.get(bone_index, 0) + 1
            world_rotation = subtract(self._states[index].rotation, anchors[index].rotation)
            rotation = matrix_vector(
                matrix_transpose(anchors[index].parent_rotation),
                world_rotation,
            )
            rotations[bone_index] = add(rotations.get(bone_index, (0.0, 0.0, 0.0)), rotation)
            rotation_counts[bone_index] = rotation_counts.get(bone_index, 0) + 1
        return PhysicsPose(
            translations={
                index: scale(value, 1.0 / translation_counts[index])
                for index, value in translations.items()
            },
            rotations={
                index: scale(value, 1.0 / rotation_counts[index])
                for index, value in rotations.items()
            },
        )


def _body_anchors(
    preview: UiAvatarModelPreview,
    transforms: dict[int, BoneTransform],
) -> tuple[_BodyAnchor, ...]:
    anchors: list[_BodyAnchor] = []
    for body in preview.rigid_bodies:
        if 0 <= body.bone_index < len(preview.bones) and body.bone_index in transforms:
            bone_rotation, bone_position = transforms[body.bone_index]
            parent_index = preview.bones[body.bone_index].parent_index
            parent_rotation = (
                transforms[parent_index][0]
                if 0 <= parent_index < len(preview.bones) and parent_index in transforms
                else identity_matrix()
            )
            local_offset = subtract(body.position, preview.bones[body.bone_index].position)
            anchors.append(
                _BodyAnchor(
                    add(bone_position, matrix_vector(bone_rotation, local_offset)),
                    add(body.rotation, _matrix_to_euler(bone_rotation)),
                    parent_rotation,
                )
            )
        else:
            anchors.append(_BodyAnchor(body.position, body.rotation, identity_matrix()))
    return tuple(anchors)


def _rest_penetrations(
    preview: UiAvatarModelPreview,
    states: list[_BodyState],
) -> dict[tuple[int, int], float]:
    """PMX初期姿勢で許容されている剛体の重なり量を記録する。"""

    aabbs = tuple(
        _body_aabb(body, states[index]) for index, body in enumerate(preview.rigid_bodies)
    )
    result: dict[tuple[int, int], float] = {}
    for first_index in range(len(aabbs)):
        for second_index in range(first_index + 1, len(aabbs)):
            if not _can_collide(
                preview.rigid_bodies[first_index],
                preview.rigid_bodies[second_index],
            ):
                continue
            penetration = _aabb_penetration(aabbs[first_index], aabbs[second_index])
            if penetration > 0.0:
                result[(first_index, second_index)] = penetration
    return result


def _aabb_penetration(first: _Aabb, second: _Aabb) -> float:
    overlaps = tuple(
        min(first.maximum[axis], second.maximum[axis])
        - max(first.minimum[axis], second.minimum[axis])
        for axis in range(3)
    )
    return min(overlaps) if all(value > 0.0 for value in overlaps) else 0.0


def _body_aabb(body: UiAvatarRigidBody, state: _BodyState) -> _Aabb:
    extent = state.aabb_extent
    if (
        extent is None
        or state.aabb_rotation is None
        or any(
            abs(state.rotation[index] - state.aabb_rotation[index]) > 0.0001 for index in range(3)
        )
    ):
        matrix = rotation_matrix(*state.rotation)
        if body.shape == 0:
            radius = body.size[0]
            extent = (radius, radius, radius)
        elif body.shape == 1:
            extent = tuple(
                sum(abs(matrix[row][column]) * body.size[column] for column in range(3))
                for row in range(3)
            )
        else:
            radius = body.size[0]
            half_height = body.size[1] * 0.5
            axis = matrix_vector(matrix, (0.0, 1.0, 0.0))
            extent = tuple(abs(axis[index]) * half_height + radius for index in range(3))
        state.aabb_rotation = state.rotation
        state.aabb_extent = extent  # type: ignore[assignment]
    return _Aabb(subtract(state.position, extent), add(state.position, extent))


def _body_radius(body: UiAvatarRigidBody) -> float:
    if body.shape == 0:
        return body.size[0]
    if body.shape == 1:
        return length(body.size)
    return body.size[0] + body.size[1] * 0.5


def _can_collide(first: UiAvatarRigidBody, second: UiAvatarRigidBody) -> bool:
    if first.mode == 0 and second.mode == 0:
        return False
    if first.non_collision_mask & (1 << second.collision_group):
        return False
    return not bool(second.non_collision_mask & (1 << first.collision_group))


def _inverse_mass(body: UiAvatarRigidBody, *, translation: bool) -> float:
    if body.mode == 0 or (translation and body.mode == 2):
        return 0.0
    return 1.0 / max(0.001, body.mass)


def _dampen_tangent(velocity: Vector3, normal: Vector3, friction: float) -> Vector3:
    normal_speed = sum(velocity[index] * normal[index] for index in range(3))
    normal_part = scale(normal, normal_speed)
    tangent = subtract(velocity, normal_part)
    return add(normal_part, scale(tangent, max(0.0, 1.0 - friction)))


def _matrix_to_euler(matrix: tuple[Vector3, Vector3, Vector3]) -> Vector3:
    yaw = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(yaw)) > 0.00001:
        pitch = math.atan2(matrix[2][1], matrix[2][2])
        roll = math.atan2(matrix[1][0], matrix[0][0])
    else:
        pitch = math.atan2(-matrix[1][2], matrix[1][1])
        roll = 0.0
    return pitch, yaw, roll


def scale(value: Vector3, factor: float) -> Vector3:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def clamp_length(value: Vector3, maximum: float) -> Vector3:
    size = length(value)
    return value if size <= maximum or size < 0.000001 else scale(value, maximum / size)
