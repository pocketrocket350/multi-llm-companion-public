"""UMTNativePlugin.dll の隔離ABI検証。

このスクリプトはアプリ本体を読み込まず、DLLコンテキスト内で固定剛体、
ジョイント付き動的剛体、自由落下剛体を構築する。ABI、固定刻み、6DoF、
0.08単位の読戻しをアプリ外プロセスで確認する。
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


class RigidBodySimulationData(ctypes.Structure):
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


class NativeConfig(ctypes.Structure):
    _fields_ = [
        ("ground_normal", ctypes.c_float * 3),
        ("ground_constant", ctypes.c_float),
        ("mmd_unit_to_unity_unit", ctypes.c_float),
        ("convex_margin", ctypes.c_float),
        ("spring_translation_damping", ctypes.c_float),
        ("spring_rotation_damping", ctypes.c_float),
        ("locked_translation_reinforce", ctypes.c_int),
    ]


class NativeJointData(ctypes.Structure):
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


def identity_at(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> ctypes.Array[ctypes.c_float]:
    values = (ctypes.c_float * 16)()
    for index in (0, 5, 10, 15):
        values[index] = 1.0
    values[12] = x
    values[13] = y
    values[14] = z
    return values


def body_at(index: int, *, x: float, y: float, mode: int) -> RigidBodySimulationData:
    body = RigidBodySimulationData()
    body.rigid_body_index = index
    body.related_bone_index = -1
    body.group_index = 0
    body.collision_group_mask = -1
    body.shape = 0
    body.size[:] = (0.1, 0.0, 0.0)
    body.position[:] = (x, y, 0.0)
    body.mass = 0.0 if mode == 0 else 1.0
    body.linear_damping = 0.05
    body.angular_damping = 0.05
    body.friction = 0.5
    body.mode = mode
    body.initial_transform = identity_at()
    body.no_related_bone = True
    body.bone_local_transform = identity_at()
    body.rest_world_transform = identity_at(x=x, y=y)
    body.bone_transform_level = -1
    return body


def read_y_positions(dll: ctypes.CDLL, context: int, indexes: tuple[int, ...]) -> tuple[float, ...]:
    index_values = (ctypes.c_int * len(indexes))(*indexes)
    transforms = (ctypes.c_float * (16 * len(indexes)))()
    dll.MMDBulletPhysicsGetRigidBodyMotionTransforms(
        context, len(indexes), index_values, transforms
    )
    return tuple(float(transforms[index * 16 + 13]) for index in range(len(indexes)))


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: verify_umt_native_isolated.py UMTNativePlugin.dll [fixed_step_seconds]"
        )
    dll = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
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
        ctypes.POINTER(NativeConfig),
    ]
    dll.MMDBulletPhysicsReset.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    dll.MMDBulletPhysicsBuildRigidBodies.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(RigidBodySimulationData),
        ctypes.c_int,
    ]
    dll.MMDBulletPhysicsBuildJoints.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NativeJointData),
        ctypes.c_int,
    ]
    dll.MMDBulletPhysicsSetRigidBodyTransforms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_bool,
    ]
    dll.MMDBulletPhysicsStepSimulation.argtypes = [ctypes.c_void_p, ctypes.c_float]
    dll.MMDBulletPhysicsGetRigidBodyMotionTransforms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]

    fixed_step = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0 / 120.0
    gravity = (ctypes.c_float * 3)(0.0, -98.0 * 0.08, 0.0)
    context = dll.MMDBulletPhysicsCreate(gravity, 4, 120, fixed_step)
    if not context:
        raise RuntimeError("native context creation failed")
    try:
        config = NativeConfig()
        config.ground_normal[:] = (0.0, 1.0, 0.0)
        config.mmd_unit_to_unity_unit = 0.08
        config.convex_margin = 0.04 * 0.08
        config.spring_translation_damping = 1.0
        config.spring_rotation_damping = 1.0 / (0.08 * 0.08)
        config.locked_translation_reinforce = 2
        dll.MMDBulletPhysicsSetConfig(context, ctypes.byref(config))

        bodies = (RigidBodySimulationData * 3)(
            body_at(0, x=0.0, y=2.0, mode=0),
            body_at(1, x=0.0, y=1.0, mode=1),
            body_at(2, x=5.0, y=2.0, mode=1),
        )
        dll.MMDBulletPhysicsBuildRigidBodies(context, bodies, len(bodies))

        joint = NativeJointData()
        joint.type = 0
        joint.rigid_body_a_index = 0
        joint.rigid_body_b_index = 1
        joint.frame_in_a = identity_at(y=-0.5)
        joint.frame_in_b = identity_at(y=0.5)
        joint.rotation_min[:] = (-0.3, -0.3, -0.3)
        joint.rotation_max[:] = (0.3, 0.3, 0.3)
        joint.spring_rotation[:] = (0.1, 0.1, 0.1)
        dll.MMDBulletPhysicsBuildJoints(context, ctypes.byref(joint), 1)
        before_reset = read_y_positions(dll, context, (1, 2))
        dll.MMDBulletPhysicsReset(context, 0)
        seed_indexes = (ctypes.c_int * 3)(0, 1, 2)
        seed_transforms = (ctypes.c_float * 48)()
        for body_index, body in enumerate(bodies):
            seed_transforms[body_index * 16 : (body_index + 1) * 16] = body.rest_world_transform
        dll.MMDBulletPhysicsSetRigidBodyTransforms(
            context,
            3,
            seed_transforms,
            seed_indexes,
            True,
        )
        after_reset = read_y_positions(dll, context, (1, 2))

        first_elapsed = fixed_step
        dll.MMDBulletPhysicsStepSimulation(context, ctypes.c_float(first_elapsed))
        after_first_step = read_y_positions(dll, context, (1, 2))
        for _ in range(119):
            dll.MMDBulletPhysicsStepSimulation(context, ctypes.c_float(fixed_step))
        indexes = (ctypes.c_int * 2)(1, 2)
        transforms = (ctypes.c_float * 32)()
        dll.MMDBulletPhysicsGetRigidBodyMotionTransforms(context, 2, indexes, transforms)
        constrained_y = float(transforms[13])
        falling_y = float(transforms[29])
        print(f"before_reset_y={before_reset}")
        print(f"after_reset_y={after_reset}")
        print(f"first_elapsed={first_elapsed:.10f}")
        print(f"after_first_step_y={after_first_step}")
        print(f"final_y={(constrained_y, falling_y)}")
        if abs(constrained_y - 1.0) > 0.1:
            raise RuntimeError(f"joint did not preserve rest offset: y={constrained_y}")
        if not falling_y < 1.5:
            raise RuntimeError(f"dynamic body did not fall: y={falling_y}")
        round_trip_mmd_y = constrained_y / 0.08
        if abs(round_trip_mmd_y - 12.5) > 1.25:
            raise RuntimeError(f"0.08 round trip failed: y={round_trip_mmd_y}")
        print(f"abi_size={ctypes.sizeof(RigidBodySimulationData)}")
        print(f"joint_abi_size={ctypes.sizeof(NativeJointData)}")
        print(f"config_abi_size={ctypes.sizeof(NativeConfig)}")
        print("context_created=True")
        print(f"joint_constraint_stable=True final_y={constrained_y:.6f}")
        print(f"unit_round_trip=True mmd_y={round_trip_mmd_y:.6f}")
        print(f"dynamic_body_fell=True initial_y=2.0 final_y={falling_y:.6f}")
    finally:
        dll.MMDBulletPhysicsDestroy(context)
        print("context_destroyed=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
