"""公式PMXと設定済みVMDでUMTネイティブ物理の初期安定性を確認する。"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from multi_llm_companion.integrations.pmx_preview import read_pmx_preview
from multi_llm_companion.integrations.vmd_motion import read_vmd_motion
from multi_llm_companion.ui import umt_native_physics
from multi_llm_companion.ui.models import UiAvatarModelPreview, UiAvatarRigidBody
from multi_llm_companion.ui.physics_baseline import to_physics_ui_preview
from multi_llm_companion.ui.pmx_pose import (
    bone_transforms,
    prepare_ik_solver,
    solve_ik_rotations,
)
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def _body_name(preview: UiAvatarModelPreview, body: UiAvatarRigidBody) -> str:
    name = body.name
    bone_index = body.bone_index
    bones = preview.bones
    if 0 <= bone_index < len(bones):
        name += bones[bone_index].name
    return name.casefold()


def _maximum_displacement(
    simulator: umt_native_physics.UmtNativePmxPhysicsSimulator,
    preview: UiAvatarModelPreview,
    initial_positions: tuple[tuple[float, float, float], ...],
    keywords: tuple[str, ...] = (),
) -> float:
    displacements = []
    for index, body in enumerate(simulator._bodies):
        if keywords and not any(
            word.casefold() in _body_name(preview, body.source) for word in keywords
        ):
            continue
        position, _rotation = simulator._read_transform(index)
        displacements.append(_distance(position, initial_positions[index]))
    return max(displacements, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=Path)
    parser.add_argument("pmx", type=Path)
    parser.add_argument("vmd", type=Path)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    dll_path = args.dll.resolve()
    umt_native_physics._dll_path = lambda: dll_path
    loaded = read_pmx_preview(args.pmx.resolve())
    motion = read_vmd_motion(args.vmd.resolve())
    preview = to_physics_ui_preview(replace(loaded, vmd_motion=motion))
    player = VmdMotionPlayer()
    prepared_ik = prepare_ik_solver(preview)

    rotations, translations, _morphs = player.advance(preview, 0.0, True)
    solved = solve_ik_rotations(preview, rotations, {}, translations, prepared_ik)
    transforms = bone_transforms(preview, solved, translations)
    simulator = umt_native_physics.UmtNativePmxPhysicsSimulator()
    simulator.step_pose(preview, 0.0, True, transforms)
    initial_positions = tuple(
        simulator._read_transform(index)[0] for index in range(len(simulator._bodies))
    )

    simulator.step_pose(preview, simulator.fixed_step_seconds, True, transforms)
    startup_all = _maximum_displacement(simulator, preview, initial_positions)
    startup_skirt = _maximum_displacement(
        simulator, preview, initial_positions, ("スカート", "skirt")
    )
    startup_hair = _maximum_displacement(simulator, preview, initial_positions, ("髪", "hair"))
    startup_ranking = sorted(
        (
            _distance(simulator._read_transform(index)[0], initial_positions[index]),
            body.source.name,
            body.source.collision_group,
            body.source.non_collision_mask,
        )
        for index, body in enumerate(simulator._bodies)
    )

    frame_seconds = 1.0 / 30.0
    frame_count = max(1, math.ceil(max(0.0, args.seconds) / frame_seconds))
    non_finite_count = 0
    maximum_pose_translation = 0.0
    maximum_pose_rotation = 0.0
    for _ in range(frame_count):
        rotations, translations, _morphs = player.advance(preview, frame_seconds, True)
        solved = solve_ik_rotations(preview, rotations, {}, translations, prepared_ik)
        transforms = bone_transforms(preview, solved, translations)
        pose = simulator.step_pose(preview, frame_seconds, True, transforms)
        for value in pose.translations.values():
            if not all(math.isfinite(component) for component in value):
                non_finite_count += 1
            maximum_pose_translation = max(
                maximum_pose_translation, _distance(value, (0.0, 0.0, 0.0))
            )
        for value in pose.rotations.values():
            if not all(math.isfinite(component) for component in value):
                non_finite_count += 1
            maximum_pose_rotation = max(
                maximum_pose_rotation, *(abs(component) for component in value)
            )

    print(f"rigid_bodies={len(preview.rigid_bodies)} joints={len(preview.physics_joints)}")
    print(f"startup_all_mmd={startup_all:.6f}")
    print(f"startup_skirt_mmd={startup_skirt:.6f}")
    print(f"startup_hair_mmd={startup_hair:.6f}")
    for rank, (distance, name, group, mask) in enumerate(reversed(startup_ranking[-5:]), start=1):
        print(
            f"startup_rank_{rank}={distance:.6f} name={name!r} group={group} pmx_mask={mask:#06x}"
        )
    print(f"simulated_seconds={frame_count * frame_seconds:.3f}")
    print(f"maximum_pose_translation_mmd={maximum_pose_translation:.6f}")
    print(f"maximum_pose_rotation_radians={maximum_pose_rotation:.6f}")
    print(f"non_finite_count={non_finite_count}")
    if startup_skirt > 0.1:
        raise RuntimeError(f"初回ステップのスカート変位が大きすぎます: {startup_skirt:.6f}")
    if non_finite_count:
        raise RuntimeError(f"非有限の物理姿勢が発生しました: {non_finite_count}")
    if maximum_pose_translation > 100.0 or maximum_pose_rotation > math.tau:
        raise RuntimeError("3秒検証中に物理姿勢が許容範囲を超えました")
    print("configured_model_stable=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
