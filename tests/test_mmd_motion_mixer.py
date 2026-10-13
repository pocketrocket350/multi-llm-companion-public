from __future__ import annotations

import math

import numpy as np
import pytest

from multi_llm_companion.ui.mmd_motion_mixer import MmdMotionMixer, MotionLayer
from multi_llm_companion.ui.pmx_pose import rotation_matrix
from multi_llm_companion.ui.vmd_player import _quaternion_to_euler


def test_mixer_keeps_gesture_gaze_and_expression_on_separate_channels() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)

    mixed = mixer.mix(
        delta_seconds=0.1,
        gesture=MotionLayer(rotations={10: (0.0, 0.0, 0.4)}),
        gaze=MotionLayer(rotations={20: (0.0, 0.2, 0.0)}),
        expression=MotionLayer(morphs={"笑顔": 0.7}),
        protected_morphs=frozenset({"あ", "い", "う", "え", "お"}),
    )

    assert mixed.rotations[10] == pytest.approx((0.0, 0.0, 0.4))
    assert mixed.rotations[20] == pytest.approx((0.0, 0.2, 0.0))
    assert mixed.morphs["笑顔"] == pytest.approx(0.7)


def test_higher_priority_gaze_wins_when_layers_target_same_bone() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)

    mixed = mixer.mix(
        delta_seconds=0.1,
        vmd=MotionLayer(rotations={3: (0.1, 0.0, 0.0)}),
        gesture=MotionLayer(rotations={3: (0.2, 0.0, 0.0)}),
        gaze=MotionLayer(rotations={3: (0.0, 0.3, 0.0)}),
    )

    assert mixed.rotations[3] == pytest.approx((0.0, 0.3, 0.0))


def test_weighted_gesture_blends_from_vmd_pose_instead_of_pmx_zero_pose() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)

    mixed = mixer.mix(
        delta_seconds=0.1,
        vmd=MotionLayer(rotations={3: (0.0, 0.0, 0.4)}, exact=True),
        gesture=MotionLayer(rotations={3: (1.0, 0.0, 0.0)}, pose_weight=0.5),
    )

    assert mixed.rotations[3] == pytest.approx((0.5, 0.0, 0.2))


def test_exact_vmd_layer_is_not_delayed_but_higher_layer_can_override_it() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.4)

    mixed = mixer.mix(
        delta_seconds=0.01,
        vmd=MotionLayer(
            rotations={3: (0.6, 0.0, 0.0), 4: (0.0, 0.5, 0.0)},
            exact=True,
        ),
        gesture=MotionLayer(rotations={4: (0.0, 0.2, 0.0)}),
    )

    assert mixed.rotations[3] == pytest.approx((0.6, 0.0, 0.0))
    assert mixed.rotations[4] == pytest.approx((0.0, 0.005, 0.0))


def test_lip_morphs_are_reserved_for_fast_lip_sync_path() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)

    mixed = mixer.mix(
        delta_seconds=0.1,
        vmd=MotionLayer(morphs={"あ": 1.0, "笑顔": 0.3}),
        expression=MotionLayer(morphs={"あ": 0.5, "笑顔": 0.8}),
        protected_morphs=frozenset({"あ"}),
    )

    assert "あ" not in mixed.morphs
    assert mixed.morphs["笑顔"] == pytest.approx(0.8)


def test_switching_motion_interpolates_instead_of_jumping() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.4)
    first = mixer.mix(
        delta_seconds=0.4,
        gesture=MotionLayer(rotations={5: (0.0, 0.0, 0.6)}),
    )
    mixer.interrupt()
    second = mixer.mix(
        delta_seconds=0.1,
        gesture=MotionLayer(rotations={5: (0.0, 0.0, -0.6)}),
    )

    assert first.rotations[5][2] == pytest.approx(0.6)
    assert second.rotations[5][2] == pytest.approx(0.3)
    assert second.rotations[5][2] != pytest.approx(-0.6)


def test_emergency_stop_holds_pose_until_released() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)
    before = mixer.mix(
        delta_seconds=0.1,
        gesture=MotionLayer(rotations={4: (0.2, 0.0, 0.0)}),
    )
    mixer.set_emergency_stopped(True)
    stopped = mixer.mix(delta_seconds=1.0)

    assert stopped == before


def test_all_outputs_are_limited_when_commands_overlap() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)
    mixed = mixer.mix(
        delta_seconds=0.1,
        gesture=MotionLayer(
            rotations={1: (50.0, -50.0, 20.0)},
            translations={1: (10.0, -10.0, 4.0)},
        ),
    )

    assert all(abs(value) <= 0.7 for value in mixed.rotations[1])
    assert all(abs(value) <= 0.35 for value in mixed.translations[1])


def test_validated_vmd_layer_keeps_authored_bone_angle_and_translation() -> None:
    mixer = MmdMotionMixer(transition_seconds=0.1)

    mixed = mixer.mix(
        delta_seconds=0.1,
        gesture=MotionLayer(
            rotations={1: (1.9, -1.1, 0.8)},
            translations={1: (0.9, -0.6, 0.4)},
            rotation_limit=2.8,
            translation_limit=2.0,
        ),
    )

    assert mixed.rotations[1] == pytest.approx((1.9, -1.1, 0.8))
    assert mixed.translations[1] == pytest.approx((0.9, -0.6, 0.4))


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((0, 0, math.pi - 0.0001), (0, 0, -math.pi + 0.0001)),
        ((3.12, -1.551, -math.pi), (3.12, -1.551, math.pi)),
        ((0.2, -1.4, 0.3), (0.2 - math.pi, -math.pi + 1.4, 0.3 - math.pi)),
    ],
)
def test_rotation_interpolation_does_not_spin_at_equivalent_euler_angles(before, after):
    mixer = MmdMotionMixer(transition_seconds=0.4)
    mixer.mix(delta_seconds=0.4, gesture=MotionLayer(rotations={0: before}, rotation_limit=math.pi))
    actual = mixer.mix(
        delta_seconds=0.1, gesture=MotionLayer(rotations={0: after}, rotation_limit=math.pi)
    ).rotations[0]
    np.testing.assert_allclose(rotation_matrix(*actual), rotation_matrix(*before), atol=0.0003)


def test_zero_elapsed_preserves_rotation_while_translation_stays_linear():
    mixer = MmdMotionMixer()
    before = mixer.mix(
        delta_seconds=0.35,
        gesture=MotionLayer(rotations={0: (1.0, 0.3, -0.2)}, rotation_limit=math.pi),
    )
    after = mixer.mix(delta_seconds=0, gesture=MotionLayer(rotations={0: (0, 0, 0)}))
    assert after.rotations == before.rotations


def test_rotation_interpolation_preserves_existing_path_without_angle_wrap():
    mixer = MmdMotionMixer(transition_seconds=0.4)
    before, after = (0.3, 0.4, 0.5), (-0.4, 0.2, -0.6)
    mixer.mix(delta_seconds=0.4, gesture=MotionLayer(rotations={0: before}))
    pose = mixer.mix(delta_seconds=0.1, gesture=MotionLayer(rotations={0: after}))
    assert pose.rotations[0] == pytest.approx(
        tuple(before[i] + (after[i] - before[i]) * 0.25 for i in range(3))
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_rotation_interpolation_preserves_orientation_at_gimbal_pole(sign):
    sine, cosine = math.sin(0.2) / math.sqrt(2), math.cos(0.2) / math.sqrt(2)
    converted = _quaternion_to_euler((sine, sign * cosine, -sign * sine, cosine))
    np.testing.assert_allclose(
        rotation_matrix(*converted), rotation_matrix(0.4, sign * math.pi / 2, 0), atol=1e-9
    )
    mixer = MmdMotionMixer(transition_seconds=0.4)
    before = (0.2, sign * math.pi / 2 - 0.01, 0.3)
    after = (0.2, sign * math.pi / 2 + 0.01, 0.3)
    mixer.mix(delta_seconds=0.4, gesture=MotionLayer(rotations={0: before}, rotation_limit=math.pi))
    actual = mixer.mix(
        delta_seconds=0.2, gesture=MotionLayer(rotations={0: after}, rotation_limit=math.pi)
    ).rotations[0]
    np.testing.assert_allclose(
        rotation_matrix(*actual), rotation_matrix(0.2, sign * math.pi / 2, 0.3), atol=1e-9
    )
