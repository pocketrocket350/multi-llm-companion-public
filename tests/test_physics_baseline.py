from __future__ import annotations

import math

from multi_llm_companion.integrations.base import AvatarModelPreview
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
)
from multi_llm_companion.ui.physics_baseline import measure_physics_baseline, to_physics_ui_preview


def _baseline_preview() -> UiAvatarModelPreview:
    return UiAvatarModelPreview(
        model_name="基準測定テスト",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        bones=(
            UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("右腕", 0, (1.0, 7.0, 0.0)),
            UiAvatarBone("右手首", 1, (2.0, 5.0, 0.0)),
            UiAvatarBone("右中指１", 2, (2.0, 4.0, 0.0)),
            UiAvatarBone("Skirt_0_0", 0, (0.0, 4.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 4.5, 0.0),
                0.0,
                1.0,
                0,
                name="下半身",
                collision_group=0,
                non_collision_mask=0xFFFF,
                shape=0,
                size=(0.4, 0.4, 0.4),
            ),
            UiAvatarRigidBody(
                1,
                (0.6, 4.0, 0.0),
                0.0,
                1.0,
                0,
                name="右腕",
                collision_group=2,
                non_collision_mask=0xFFFF,
                shape=2,
                size=(0.2, 1.0, 0.0),
            ),
            UiAvatarRigidBody(
                4,
                (0.0, 4.0, 0.0),
                0.2,
                0.8,
                1,
                name="Skirt_0_0",
                collision_group=15,
                non_collision_mask=1 << 0,
                shape=0,
                size=(0.3, 0.3, 0.3),
            ),
        ),
        physics_joints=(
            UiAvatarPhysicsJoint(
                0,
                2,
                (0.0, 4.25, 0.0),
                (4.0, 4.0, 4.0),
                translation_min=(-0.5, -0.5, -0.5),
                translation_max=(0.5, 0.5, 0.5),
                rotation_min=(-0.3, -0.3, -0.3),
                rotation_max=(0.3, 0.3, 0.3),
                spring_rotation=(2.0, 2.0, 2.0),
            ),
        ),
    )


def test_physics_baseline_measures_repeatable_scenarios_without_asset_names() -> None:
    report = measure_physics_baseline(_baseline_preview(), cadence_seconds=0.25)

    assert report.cadence_seconds == 0.25
    assert report.rigid_body_count == 3
    assert report.dynamic_body_count == 1
    assert report.joint_count == 1
    assert report.vmd_bone_frame_count == 0
    assert report.collision_coverage.hand_arm_sleeve_to_skirt_pairs == 1
    assert report.collision_coverage.authored_colliding_pairs == 0
    assert report.collision_coverage.diagnostic_only_excluded_pairs == 1
    assert {item.name for item in report.scenarios} >= {
        "通常待機",
        "移動開始・停止後の収束",
        "身体の左右回転・停止",
        "腕を身体の横へ下ろす",
        "手が前面・側面を横切る",
        "物理無効時の移動比較",
        "30Hz固定刻み基準",
    }
    assert all(math.isfinite(item.mean_ms) for item in report.scenarios)
    assert all(item.non_finite_count == 0 for item in report.scenarios)
    assert all(item.maximum_consecutive_over_cadence >= 0 for item in report.scenarios)
    assert "基準測定テスト" not in str(report.as_dict())


def test_physics_baseline_keeps_the_local_thin_cloth_profile() -> None:
    converted = to_physics_ui_preview(
        AvatarModelPreview(
            model_name="ローカル確認用",
            vertex_count=0,
            face_count=0,
            points=(),
            thin_cloth_contact_enabled=True,
            thin_cloth_source_body_patterns=("右腕",),
            thin_cloth_target_body_patterns=("Skirt_",),
            thin_cloth_maximum_correction_per_step=0.006,
        )
    )

    assert converted.thin_cloth_contact_enabled is True
    assert converted.thin_cloth_source_body_patterns == ("右腕",)
    assert converted.thin_cloth_target_body_patterns == ("Skirt_",)
    assert converted.thin_cloth_maximum_correction_per_step == 0.006
