from __future__ import annotations

import ctypes
import math
from dataclasses import replace

import pytest

from multi_llm_companion.ui.bullet_pmx_physics import _body_anchors
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms
from multi_llm_companion.ui.umt_native_physics import (
    _FIXED_STEP_SECONDS,
    _GRAVITY,
    _MMD_UNIT_SCALE,
    UmtNativePmxPhysicsSimulator,
    _euler_zxy_rotation,
    _native_config,
    _native_joints,
    _native_matrix4,
    _NativeConfig,
    _NativeJointData,
    _prepare_native_bodies,
    _read_matrix4,
    _rigid_body_data,
    _RigidBodySimulationData,
    _to_native_position,
)


def _preview() -> UiAvatarModelPreview:
    return UiAvatarModelPreview(
        model_name="UMT座標対応テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("固定", -1, (1.0, 2.0, 3.0)),
            UiAvatarBone("物理", 0, (4.0, 5.0, 6.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                bone_index=0,
                position=(2.0, 4.0, 6.0),
                mass=0.0,
                linear_damping=0.1,
                mode=0,
                collision_group=3,
                non_collision_mask=0x0005,
                shape=1,
                size=(1.0, 2.0, 3.0),
                rotation=(0.1, 0.2, 0.3),
                angular_damping=0.4,
                restitution=0.5,
                friction=0.6,
            ),
            UiAvatarRigidBody(
                bone_index=1,
                position=(4.0, 6.0, 8.0),
                mass=1.0,
                linear_damping=0.2,
                mode=1,
                collision_group=4,
                non_collision_mask=0x8002,
                shape=0,
                size=(0.5, 0.5, 0.5),
            ),
        ),
        physics_joints=(
            UiAvatarPhysicsJoint(
                rigid_body_a=0,
                rigid_body_b=1,
                position=(3.0, 5.0, 7.0),
                spring_translation=(10.0, 20.0, 30.0),
                rotation=(0.2, 0.3, 0.4),
                translation_min=(-1.0, -2.0, -3.0),
                translation_max=(4.0, 5.0, 6.0),
                rotation_min=(-0.1, -0.2, -0.3),
                rotation_max=(0.4, 0.5, 0.6),
                spring_rotation=(40.0, 50.0, 60.0),
            ),
        ),
    )


@pytest.mark.parametrize("hidden_mode", (1, 2))
def test_unused_readback_keeps_native_bodies_and_mode_two_alignment(monkeypatch, hidden_mode):
    preview = _preview()
    preview = replace(
        preview,
        bones=preview.bones + (UiAvatarBone("独立布で置換する表示", 0, (4, 5, 6)),),
        rigid_bodies=preview.rigid_bodies
        + (replace(preview.rigid_bodies[1], bone_index=2, mode=hidden_mode),),
    )
    simulator = UmtNativePmxPhysicsSimulator()
    simulator._bodies = _prepare_native_bodies(preview)
    simulator._dynamic_body_indexes = (1, 2)
    simulator._physics_bone_order = (1, 2)
    bodies_before = tuple(simulator._bodies)
    reads, shifts = [], []

    def read(index):
        reads.append(index)
        return preview.rigid_bodies[index].position, _euler_zxy_rotation(0, 0, 0)

    monkeypatch.setattr(
        simulator, "_read_transforms", lambda indexes: {i: read(i) for i in indexes}
    )
    monkeypatch.setattr(simulator, "_shift_body_position", lambda *args: shifts.append(args))
    anchors = _body_anchors(preview, bone_transforms(preview, {}, {}), simulator._bodies)
    full = simulator._bone_pose(preview, anchors)
    reads.clear()
    shifts.clear()
    simulator._output_bone_indexes = frozenset((0, 1))
    filtered = simulator._bone_pose(preview, anchors)

    assert filtered.rotations == {1: full.rotations[1]}
    assert filtered.translations == {1: full.translations[1]}
    assert reads == ([1] if hidden_mode == 1 else [1, 2])
    assert len(shifts) == (0 if hidden_mode == 1 else 1)
    assert tuple(simulator._bodies) == bodies_before
    assert simulator._dynamic_body_indexes == (1, 2)
    simulator.reset(None)
    assert simulator._output_bone_indexes is None


def test_native_abi_layout_matches_public_umt_structs() -> None:
    assert ctypes.sizeof(_NativeConfig) == 36
    assert ctypes.sizeof(_RigidBodySimulationData) == 288
    assert ctypes.sizeof(_NativeJointData) == 212
    assert _RigidBodySimulationData.group_index.offset == 8
    assert _RigidBodySimulationData.collision_group_mask.offset == 10
    assert _RigidBodySimulationData.initial_transform.offset == 76
    assert _RigidBodySimulationData.rest_world_transform.offset == 220
    assert _NativeJointData.rigid_body_a_index.offset == 4
    assert _NativeJointData.frame_in_a.offset == 12


def test_native_runtime_config_matches_unity_mmd_tools_reference() -> None:
    config = _native_config()

    assert pytest.approx(0.08) == _MMD_UNIT_SCALE
    assert pytest.approx(1.0 / 120.0) == _FIXED_STEP_SECONDS
    assert pytest.approx((0.0, -98.0 * 0.08, 0.0)) == _GRAVITY
    assert tuple(config.ground_normal) == pytest.approx((0.0, 1.0, 0.0))
    assert config.ground_constant == pytest.approx(0.0)
    assert config.mmd_unit_to_unity_unit == pytest.approx(0.08)
    assert config.convex_margin == pytest.approx(0.04 * 0.08)
    assert config.spring_translation_damping == pytest.approx(1.0)
    assert config.spring_rotation_damping == pytest.approx(1.0 / (0.08 * 0.08))
    assert config.locked_translation_reinforce == 2


def test_pmx_rigid_rotation_matches_unity_euler_zxy_order() -> None:
    rotation = _euler_zxy_rotation(math.pi / 2.0, math.pi / 2.0, math.pi / 2.0)

    assert tuple(value for row in rotation for value in row) == pytest.approx(
        (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
        abs=1e-6,
    )


def test_rigid_body_data_preserves_pmx_bullet_collision_mask() -> None:
    preview = _preview()
    bodies = _prepare_native_bodies(preview)
    data = _rigid_body_data(0, bodies[0], preview)

    assert tuple(data.size) == pytest.approx((0.08, 0.16, 0.24))
    assert tuple(data.position) == pytest.approx((-0.16, 0.32, -0.48))
    assert tuple(data.rotation) == pytest.approx((-0.1, 0.2, -0.3))
    assert data.collision_group_mask == 0x0005
    assert tuple(data.bone_model_position) == pytest.approx((-0.08, 0.16, -0.24))
    assert (
        data.rest_world_transform[12],
        data.rest_world_transform[13],
        data.rest_world_transform[14],
    ) == pytest.approx((-0.16, 0.32, -0.48))


def test_joint_data_converts_frames_limits_and_spring_units() -> None:
    preview = _preview()
    bodies = _prepare_native_bodies(preview)
    joints = _native_joints(preview, bodies)

    assert joints is not None
    joint = joints[0]
    assert tuple(joint.translation_min) == pytest.approx((-0.32, -0.16, -0.48))
    assert tuple(joint.translation_max) == pytest.approx((0.08, 0.4, 0.24))
    assert tuple(joint.rotation_min) == pytest.approx((-0.4, -0.2, -0.6))
    assert tuple(joint.rotation_max) == pytest.approx((0.1, 0.5, 0.3))
    assert tuple(joint.spring_translation) == pytest.approx((10.0, 20.0, 30.0))
    assert tuple(joint.spring_rotation) == pytest.approx(
        tuple(value * 0.08 * 0.08 for value in (40.0, 50.0, 60.0))
    )
    body_data = _rigid_body_data(0, bodies[0], preview)
    body_position, body_rotation = _read_matrix4(body_data.rest_world_transform)
    frame_position, _frame_rotation = _read_matrix4(joint.frame_in_a)
    reconstructed_joint_position = tuple(
        body_position[row]
        + sum(body_rotation[row][column] * frame_position[column] for column in range(3))
        for row in range(3)
    )
    assert reconstructed_joint_position == pytest.approx(
        _to_native_position(preview.physics_joints[0].position)
    )


def test_native_rotation_matrix_round_trip_preserves_app_rotation() -> None:
    preview = _preview()
    body = _prepare_native_bodies(preview)[0]
    data = _rigid_body_data(0, body, preview)

    native_rotation = (
        (data.rest_world_transform[0], data.rest_world_transform[4], data.rest_world_transform[8]),
        (data.rest_world_transform[1], data.rest_world_transform[5], data.rest_world_transform[9]),
        (data.rest_world_transform[2], data.rest_world_transform[6], data.rest_world_transform[10]),
    )
    determinant = (
        native_rotation[0][0]
        * (
            native_rotation[1][1] * native_rotation[2][2]
            - native_rotation[1][2] * native_rotation[2][1]
        )
        - native_rotation[0][1]
        * (
            native_rotation[1][0] * native_rotation[2][2]
            - native_rotation[1][2] * native_rotation[2][0]
        )
        + native_rotation[0][2]
        * (
            native_rotation[1][0] * native_rotation[2][1]
            - native_rotation[1][1] * native_rotation[2][0]
        )
    )
    assert math.isclose(determinant, 1.0, abs_tol=1e-6)


def test_batched_native_readback_matches_single_calls_in_requested_order():
    calls = []

    class Dll:
        def MMDBulletPhysicsGetRigidBodyMotionTransforms(self, context, count, indexes, values):
            calls.append(tuple(indexes))
            for offset, index in enumerate(indexes):
                matrix = _native_matrix4(
                    (index * 0.1, index * 0.2, -index * 0.3),
                    _euler_zxy_rotation(index * 0.1, -0.2, 0.3),
                )
                values[offset * 16 : (offset + 1) * 16] = matrix

    simulator = UmtNativePmxPhysicsSimulator()
    simulator._dll = Dll()
    simulator._context = ctypes.c_void_p(1)
    try:
        indexes = (7, 2, 5)
        batched = simulator._read_transforms(indexes)
        assert calls == [indexes]
        calls.clear()
        singles = {i: simulator._read_transform(i) for i in indexes}
        assert calls == [(7,), (2,), (5,)]
        assert batched == singles
        calls.clear()
        assert simulator._read_transforms(()) == {}
        assert not calls
    finally:
        simulator._dll = None
        simulator._context = None
