from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from panda3d.core import LVector3f

import multi_llm_companion.ui.bullet_pmx_physics as bullet_pmx_physics
from multi_llm_companion.ui.bullet_pmx_physics import (
    BulletPmxPhysicsSimulator,
    _authored_collision,
)
from multi_llm_companion.ui.dense_pmx_widget import (
    _apply_prepared_bone_pose,
    _physics_anchor_motion,
    _prepare_pose_data,
)
from multi_llm_companion.ui.mmd_motion_mixer import MixedMotion
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
    UiAvatarTexturedTriangle,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms


def _collision_preview(
    dynamic_name: str = "Skirt_0_0",
    *,
    static_mask: int = 0xFFFF,
    dynamic_mask: int = 0x0001,
    static_mode: int = 0,
) -> UiAvatarModelPreview:
    return UiAvatarModelPreview(
        model_name="Bullet衝突テスト",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        bones=(
            UiAvatarBone("右手捩", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone(dynamic_name, -1, (0.5, 0.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                static_mode,
                (0.0, 0.0, 0.0),
                1.0,
                0.1,
                0,
                name="右手捩",
                collision_group=2,
                non_collision_mask=static_mask,
                shape=0,
                size=(1.0, 1.0, 1.0),
            ),
            UiAvatarRigidBody(
                1,
                (0.5, 0.0, 0.0),
                1.0,
                0.1,
                1,
                name=dynamic_name,
                collision_group=15,
                non_collision_mask=dynamic_mask,
                shape=0,
                size=(1.0, 1.0, 1.0),
            ),
        ),
    )


def test_bullet_does_not_override_pmx_non_collision_mask_for_cloth() -> None:
    physics = BulletPmxPhysicsSimulator()
    preview = _collision_preview()

    pose = physics.step_pose(
        preview,
        1.0 / 15.0,
        True,
        bone_transforms(preview, {}, {}),
    )

    assert abs(pose.translations[1][0]) < 0.001


def test_bullet_limits_a_short_render_delay_to_keep_the_ui_responsive() -> None:
    physics = BulletPmxPhysicsSimulator()
    preview = _collision_preview()

    physics.step_pose(preview, 0.2, True, bone_transforms(preview, {}, {}))

    assert physics.last_substeps == 2
    assert physics.dropped_seconds > 0.0


def test_bullet_interpolates_kinematic_vmd_targets_across_fixed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _collision_preview()
    physics = BulletPmxPhysicsSimulator()
    physics.step_pose(preview, 0.0, True, bone_transforms(preview, {}, {}))
    calls: list[tuple[float, bool]] = []
    original = physics._update_kinematic_bodies

    def record_targets(
        anchors: tuple[bullet_pmx_physics._BodyAnchor, ...],
        hand_transforms: tuple[
            tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]], ...
        ] = (),
        *,
        clear_kinematic_velocity: bool = False,
    ) -> None:
        calls.append((anchors[0].position[0], clear_kinematic_velocity))
        original(
            anchors,
            hand_transforms,
            clear_kinematic_velocity=clear_kinematic_velocity,
        )

    monkeypatch.setattr(physics, "_update_kinematic_bodies", record_targets)
    moved = bone_transforms(preview, {}, {0: (2.0, 0.0, 0.0)})

    physics.step_pose(preview, 2.0 / 30.0, True, moved)

    assert calls == [(pytest.approx(1.0), False), (pytest.approx(2.0), False)]


def test_initial_settle_updates_joint_springs_from_the_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = BulletPmxPhysicsSimulator()
    calls = 0
    original = physics._update_joint_springs

    def record_spring_update() -> None:
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(physics, "_update_joint_springs", record_spring_update)
    physics.reset(_collision_preview())

    assert calls == bullet_pmx_physics._INITIAL_SETTLE_STEPS


def test_bullet_contact_moves_cloth_instead_of_moving_the_hand_away() -> None:
    preview = _collision_preview(static_mask=0, dynamic_mask=0)
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {}, {})

    pose = physics.step_pose(preview, 1.0 / 15.0, True, transforms)

    assert _authored_collision(preview.rigid_bodies[0], preview.rigid_bodies[1])
    assert physics.last_substeps == 2
    assert 1 in pose.translations
    assert math.dist(pose.translations[1], (0.0, 0.0, 0.0)) > 0.1
    assert tuple(physics._bodies[0].path.get_pos()) == (0.0, 0.0, 0.0)
    assert all(math.isfinite(value) for value in pose.translations[1])


def test_bullet_reuses_invariant_body_data_after_world_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _collision_preview(static_mask=0, dynamic_mask=0)
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {1: (0.1, 0.2, 0.3)}, {})

    physics.step_pose(preview, 0.0, True, transforms)
    uncached = bullet_pmx_physics._body_anchors(preview, transforms)
    cached = bullet_pmx_physics._body_anchors(preview, transforms, physics._bodies)

    assert cached == uncached
    assert physics._kinematic_body_indexes == (0,)
    assert physics._dynamic_body_indexes == (1,)
    assert physics._physics_bone_order == (1,)

    def fail_depth(_preview: UiAvatarModelPreview, _bone_index: int) -> int:
        raise AssertionError("物理ボーンの親階層を再走査している")

    monkeypatch.setattr(bullet_pmx_physics, "_bone_depth", fail_depth)
    pose = physics.step_pose(preview, 1.0 / 30.0, True, transforms)

    assert 1 in pose.translations


def test_missing_pmx_hand_body_gets_a_non_pushing_kinematic_collider() -> None:
    preview = UiAvatarModelPreview(
        model_name="手のひら補助テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("右手首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("右中指１", 0, (0.0, 1.0, 0.0)),
            UiAvatarBone("Skirt_0_0", -1, (0.0, 0.5, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                2,
                (0.0, 0.5, 0.0),
                0.2,
                0.5,
                1,
                name="Skirt_0_0",
                collision_group=15,
                non_collision_mask=0x0001,
                shape=0,
                size=(0.2, 0.2, 0.2),
            ),
        ),
    )
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {}, {})

    physics.step_pose(preview, 1.0 / 60.0, True, transforms)

    assert len(physics._hand_colliders) == 1
    hand = physics._hand_colliders[0]
    assert hand.source.name == "補助右手のひら"
    assert _authored_collision(hand.source, preview.rigid_bodies[0])
    assert not hand.node.get_collision_response()
    assert tuple(hand.path.get_pos()) == (0.0, pytest.approx(0.52), 0.0)

    moved = bone_transforms(preview, {}, {0: (2.0, 0.0, 0.0)})
    physics.step_pose(preview, 1.0 / 60.0, True, moved)

    assert float(hand.path.get_x()) == pytest.approx(2.0)


def test_missing_hand_collider_does_not_push_overlapping_skirt() -> None:
    preview = UiAvatarModelPreview(
        model_name="手のひら接触監視テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("右手首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("右中指１", 0, (0.0, 1.0, 0.0)),
            UiAvatarBone("Skirt_0_0", -1, (0.0, 0.5, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                2,
                (0.0, 0.5, 0.0),
                0.2,
                1.0,
                1,
                name="Skirt_0_0",
                collision_group=15,
                non_collision_mask=0x0001,
                shape=0,
                size=(0.2, 0.2, 0.2),
            ),
        ),
    )
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {}, {})

    physics.step_pose(preview, 0.0, True, transforms)
    assert physics._world is not None
    assert physics.auxiliary_hand_contact_count() >= 1
    physics._world.set_gravity(LVector3f(0.0, 0.0, 0.0))
    start = tuple(physics._bodies[0].path.get_pos())
    moved = bone_transforms(preview, {}, {0: (-0.4, 0.0, 0.0)})
    for _ in range(30):
        physics.step_pose(preview, 1.0 / 30.0, True, moved)

    assert math.dist(tuple(physics._bodies[0].path.get_pos()), start) < 0.001


def test_local_thin_cloth_contact_moves_only_the_configured_skirt() -> None:
    preview = replace(
        _collision_preview(
            static_mask=0x0001,
            dynamic_mask=0x0001,
            static_mode=1,
        ),
        thin_cloth_contact_enabled=True,
        thin_cloth_source_body_patterns=("右手",),
        thin_cloth_target_body_patterns=("skirt_",),
        thin_cloth_maximum_correction_per_step=0.01,
    )
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {}, {})

    physics.step_pose(preview, 0.0, True, transforms)
    assert physics._world is not None
    assert len(physics._thin_cloth_colliders) == 1
    assert physics._thin_cloth_colliders[0].target_body_indexes == (1,)
    surfaces = physics.thin_cloth_source_surfaces()

    assert len(surfaces) == 1
    assert surfaces[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert surfaces[0].radius == pytest.approx(1.0)
    assert surfaces[0].half_length == 0.0


def test_bullet_records_diagnostic_offset_and_stabilization_reset() -> None:
    preview = _collision_preview(static_mask=0, dynamic_mask=0)
    physics = BulletPmxPhysicsSimulator()
    transforms = bone_transforms(preview, {}, {})

    physics.step_pose(preview, 1.0 / 15.0, True, transforms)

    assert physics.maximum_body_offset_observed > 0.0
    before = physics.stabilization_reset_count
    non_finite_before = physics.non_finite_reset_count
    physics._bodies[1].path.set_x(math.inf)
    physics._stabilize_bodies(
        bullet_pmx_physics._body_anchors(preview, transforms, physics._bodies)
    )

    assert physics.stabilization_reset_count == before + 1
    assert physics.non_finite_reset_count == non_finite_before + 1


def test_existing_pmx_hand_body_is_not_duplicated() -> None:
    preview = UiAvatarModelPreview(
        model_name="既存手剛体テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("右手首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("右中指１", 0, (0.0, 1.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 0.0, 0.0),
                0.0,
                1.0,
                0,
                name="右手首",
                collision_group=0,
                non_collision_mask=0xFFFF,
                shape=0,
                size=(0.2, 0.2, 0.2),
            ),
        ),
    )
    physics = BulletPmxPhysicsSimulator()

    physics.step_pose(preview, 0.0, True, bone_transforms(preview, {}, {}))

    assert physics._hand_colliders == []


def test_pmx_joint_translation_spring_returns_body_toward_rest() -> None:
    preview = UiAvatarModelPreview(
        model_name="ばねテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("固定", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("衣装", -1, (1.0, 0.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 0.0, 0.0),
                0.0,
                0.5,
                0,
                name="固定",
                non_collision_mask=0,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
            UiAvatarRigidBody(
                1,
                (1.0, 0.0, 0.0),
                0.2,
                0.5,
                1,
                name="Skirt_0_0",
                non_collision_mask=0,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
        ),
        physics_joints=(
            UiAvatarPhysicsJoint(
                0,
                1,
                (0.5, 0.0, 0.0),
                (30.0, 0.0, 0.0),
                translation_min=(-1.0, 0.0, 0.0),
                translation_max=(1.0, 0.0, 0.0),
            ),
        ),
    )
    physics = BulletPmxPhysicsSimulator()
    physics.reset(preview)
    assert physics._joints[0].translation_springs == ((0, pytest.approx(7.5)),)
    assert physics._joints[0].rotation_springs == ()
    assert physics._joints[0].equilibrium_translation == (0.0, 0.0, 0.0)
    assert physics._joints[0].equilibrium_rotation == (0.0, 0.0, 0.0)
    physics._bodies[1].path.set_x(1.5)
    physics._bodies[1].node.set_active(True)

    for _ in range(300):
        physics.step_pose(
            preview,
            1.0 / 60.0,
            True,
            bone_transforms(preview, {}, {}),
        )

    assert abs(float(physics._bodies[1].path.get_x()) - 1.0) < 0.2


def test_initial_world_uses_current_bone_pose_before_settle() -> None:
    preview = UiAvatarModelPreview(
        model_name="現在姿勢初期化テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("衣装", -1, (0.0, 0.0, 0.0)),),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (1.0, 0.0, 0.0),
                0.1,
                0.5,
                1,
                name="Skirt_0_0",
                non_collision_mask=0,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
        ),
    )
    transforms = bone_transforms(preview, {}, {0: (2.0, 0.0, 0.0)})
    physics = BulletPmxPhysicsSimulator()

    physics.step_pose(preview, 0.0, True, transforms)

    assert float(physics._bodies[0].path.get_x()) == 3.0


def test_joint_frames_are_built_before_current_pose_is_applied() -> None:
    preview = UiAvatarModelPreview(
        model_name="ジョイント構築順序テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("固定", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("衣装", -1, (1.0, 0.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 0.0, 0.0),
                0.0,
                0.5,
                0,
                name="固定",
                non_collision_mask=0,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
            UiAvatarRigidBody(
                1,
                (1.0, 0.0, 0.0),
                0.1,
                0.5,
                1,
                name="Skirt_0_0",
                non_collision_mask=0,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
        ),
        physics_joints=(
            UiAvatarPhysicsJoint(
                0,
                1,
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                translation_min=(0.0, 0.0, 0.0),
                translation_max=(0.0, 0.0, 0.0),
            ),
        ),
    )
    transforms = bone_transforms(
        preview,
        {},
        {0: (2.0, 0.0, 0.0), 1: (2.0, 0.0, 0.0)},
    )
    physics = BulletPmxPhysicsSimulator()

    physics.step_pose(preview, 1.0 / 60.0, True, transforms)

    assert float(physics._bodies[0].path.get_x()) == 2.0
    assert abs(float(physics._bodies[1].path.get_x()) - 3.0) < 0.2


def test_physics_anchor_uses_displayed_hand_but_not_previous_cloth_result() -> None:
    preview = _collision_preview()
    displayed = MixedMotion(
        rotations={0: (0.0, 0.0, 0.5), 1: (0.0, 0.0, 0.9)},
        translations={0: (0.2, 0.0, 0.0), 1: (0.8, 0.0, 0.0)},
    )

    rotations, translations = _physics_anchor_motion(
        preview,
        {},
        {},
        {},
        {},
        displayed,
    )

    assert rotations[0] == (0.0, 0.0, 0.5)
    assert translations[0] == (0.2, 0.0, 0.0)
    assert 1 not in rotations
    assert 1 not in translations


def test_physics_anchor_keeps_current_vmd_hand_over_stale_displayed_hand() -> None:
    preview = _collision_preview()
    rotations, translations = _physics_anchor_motion(
        preview,
        {0: (0.0, 0.0, 0.7)},
        {0: (0.3, 0.0, 0.0)},
        {},
        {},
        MixedMotion(
            rotations={0: (0.0, 0.0, 0.2)},
            translations={0: (0.1, 0.0, 0.0)},
        ),
    )

    assert rotations[0] == (0.0, 0.0, 0.7)
    assert translations[0] == (0.3, 0.0, 0.0)


def test_physics_anchor_uses_the_blended_hand_during_a_transition() -> None:
    preview = _collision_preview()
    rotations, translations = _physics_anchor_motion(
        preview,
        {0: (0.0, 0.0, 0.7)},
        {0: (0.3, 0.0, 0.0)},
        {},
        {},
        MixedMotion(
            rotations={0: (0.0, 0.0, 0.2)},
            translations={0: (0.1, 0.0, 0.0)},
        ),
        prefer_displayed=True,
    )

    assert rotations[0] == (0.0, 0.0, 0.2)
    assert translations[0] == (0.1, 0.0, 0.0)


def test_pmx_mask_excludes_body_from_skirt_but_allows_sleeve() -> None:
    body = UiAvatarRigidBody(
        0,
        (0.0, 0.0, 0.0),
        0.0,
        0.0,
        0,
        name="下半身",
        collision_group=0,
        non_collision_mask=0xFFFF,
    )
    skirt_a = UiAvatarRigidBody(
        1,
        (0.0, 0.0, 0.0),
        0.1,
        0.5,
        1,
        name="Skirt_0_0",
        collision_group=15,
        non_collision_mask=0x0001,
    )
    skirt_b = UiAvatarRigidBody(
        1,
        (0.0, 0.0, 0.0),
        0.1,
        0.5,
        1,
        name="Skirt_1_0",
        collision_group=15,
        non_collision_mask=0x0001,
    )
    sleeve = UiAvatarRigidBody(
        1,
        (0.0, 0.0, 0.0),
        0.1,
        0.5,
        2,
        name="袖",
        collision_group=13,
        non_collision_mask=0,
    )

    assert not _authored_collision(body, skirt_a)
    assert _authored_collision(sleeve, skirt_a)
    assert _authored_collision(skirt_a, skirt_b)


def test_physics_child_translation_is_relative_to_dynamic_parent() -> None:
    preview = UiAvatarModelPreview(
        model_name="物理親子テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("親", -1, (0.0, 2.0, 0.0)),
            UiAvatarBone("子", 0, (0.0, 1.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 2.0, 0.0),
                0.1,
                0.5,
                1,
                name="親",
                collision_group=1,
                non_collision_mask=0xFFFF,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
            UiAvatarRigidBody(
                1,
                (0.0, 1.0, 0.0),
                0.1,
                0.5,
                1,
                name="子",
                collision_group=1,
                non_collision_mask=0xFFFF,
                shape=0,
                size=(0.1, 0.1, 0.1),
            ),
        ),
    )

    physics = BulletPmxPhysicsSimulator()
    pose = physics.step_pose(
        preview,
        1.0 / 15.0,
        True,
        bone_transforms(preview, {}, {}),
    )

    assert pose.translations[0][1] < 0.0
    assert abs(pose.translations[1][1]) < 0.001


def test_prepared_pose_keeps_foot_vertices_on_the_rest_floor() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="接地テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (0.0, 2.0, 0.0)),
        bones=(
            UiAvatarBone("全ての親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左足首", 0, (0.0, 0.0, 0.0)),
        ),
        vertex_weights=(((1, 1.0),), ((1, 1.0),), ((0, 1.0),)),
        dense_textured_triangles=(triangle,),
    )
    prepared = _prepare_pose_data(preview)

    assert prepared is not None
    result = _apply_prepared_bone_pose(
        np.asarray(preview.points, dtype=np.float32),
        preview,
        prepared,
        {},
        {},
        {1: (0.0, 1.0, 0.0)},
    )

    assert float(result[prepared.ground_vertex_indexes, 1].min()) == 0.0
    assert result[2, 1] == 1.0
