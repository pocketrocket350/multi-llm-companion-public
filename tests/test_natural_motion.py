from __future__ import annotations

from itertools import pairwise

from multi_llm_companion.domain import ConversationState
from multi_llm_companion.ui.natural_motion import NaturalMotionController


def test_natural_motion_uses_non_repeating_breath_and_blink_intervals() -> None:
    controller = NaturalMotionController(seed=42)
    breath_peaks: list[float] = []
    blink_starts: list[float] = []
    previous_breath = 0.0
    breath_was_rising = False
    previous_blink = 0.0
    elapsed = 0.0

    # 10分を高速に進め、固定周期へ戻っていないことを確認する。
    for _ in range(12_000):
        elapsed += 0.05
        frame = controller.advance(0.05)
        rising = frame.breath > previous_breath
        if breath_was_rising and not rising:
            breath_peaks.append(elapsed)
        if previous_blink == 0.0 and frame.blink_weight > 0.0:
            blink_starts.append(elapsed)
        breath_was_rising = rising
        previous_breath = frame.breath
        previous_blink = frame.blink_weight

    breath_intervals = [round(after - before, 1) for before, after in pairwise(breath_peaks)]
    blink_intervals = [round(after - before, 1) for before, after in pairwise(blink_starts)]
    assert len(breath_intervals) > 100
    assert len(set(breath_intervals)) >= 10
    assert len(blink_intervals) > 80
    assert len(set(blink_intervals)) >= 20


def test_gaze_looks_at_user_thinks_away_and_down_then_returns_smoothly() -> None:
    controller = NaturalMotionController(seed=7)
    for _ in range(30):
        idle = controller.advance(0.05)
    assert abs(idle.gaze_yaw) < 0.001
    assert abs(idle.gaze_pitch) < 0.001

    controller.set_state("thinking")
    first_thinking = controller.advance(0.05)
    assert abs(first_thinking.gaze_yaw - idle.gaze_yaw) < 0.03
    assert abs(first_thinking.gaze_pitch - idle.gaze_pitch) < 0.01
    for _ in range(11):
        looking_away = controller.advance(0.05)
    assert abs(looking_away.gaze_yaw) > 0.08
    assert abs(looking_away.gaze_pitch) < 0.005

    for _ in range(20):
        looking_down = controller.advance(0.05)
    assert 0.04 < abs(looking_down.gaze_yaw) < abs(looking_away.gaze_yaw)
    assert looking_down.gaze_pitch < -0.045
    assert looking_down.posture_pitch > idle.posture_pitch

    controller.set_state("reacting")
    first_return = controller.advance(0.05)
    assert 0.0 < abs(first_return.gaze_yaw) < abs(looking_down.gaze_yaw)
    assert looking_down.gaze_pitch < first_return.gaze_pitch < 0.0
    for _ in range(59):
        returned = controller.advance(0.05)
    assert abs(returned.gaze_yaw) < 0.001
    assert abs(returned.gaze_pitch) < 0.001

    controller.set_state("speaking")
    for _ in range(4):
        speaking = controller.advance(0.05)
    assert abs(speaking.gaze_yaw) < 0.001
    assert abs(speaking.gaze_pitch) < 0.001
    assert speaking.posture_pitch < looking_down.posture_pitch
    controller.phrase_boundary()
    for _ in range(5):
        nodding = controller.advance(0.05)
    assert nodding.nod_pitch > 0.03


def test_reacting_and_interrupted_are_formal_safe_motion_states() -> None:
    controller = NaturalMotionController(seed=11)

    controller.set_state(ConversationState.REACTING)
    reacting = controller.advance(0.05)
    controller.set_state(ConversationState.INTERRUPTED)
    interrupted = controller.advance(0.05)

    assert controller.state == ConversationState.INTERRUPTED
    assert abs(reacting.posture_pitch) < 0.01
    assert abs(interrupted.posture_pitch) < 0.01


def test_emotion_changes_motion_strength_without_exceeding_small_limits() -> None:
    neutral = NaturalMotionController(seed=19)
    happy = NaturalMotionController(seed=19)
    happy.set_emotion("happy", 1.0)
    neutral_samples = [neutral.advance(0.05) for _ in range(100)]
    happy_samples = [happy.advance(0.05) for _ in range(100)]

    assert max(item.breath for item in happy_samples) > max(item.breath for item in neutral_samples)
    assert max(abs(item.neck_yaw) for item in happy_samples) > max(
        abs(item.neck_yaw) for item in neutral_samples
    )
    assert max(abs(item.neck_yaw) for item in happy_samples) < 0.05
    assert max(abs(item.weight_shift) for item in happy_samples) < 0.01


def test_idle_motion_contains_visible_non_periodic_balance_and_head_tilt() -> None:
    controller = NaturalMotionController(seed=31)
    samples = [controller.advance(0.05) for _ in range(1_200)]

    assert max(abs(item.weight_shift) for item in samples) > 0.004
    assert max(abs(item.head_roll) for item in samples) > 0.007
    rounded_balance = {round(item.weight_shift, 4) for item in samples}
    rounded_head_roll = {round(item.head_roll, 4) for item in samples}
    assert len(rounded_balance) > 40
    assert len(rounded_head_roll) > 40


def test_disabling_natural_motion_returns_a_neutral_frame() -> None:
    controller = NaturalMotionController(seed=3)
    for _ in range(20):
        controller.advance(0.05)
    controller.set_enabled(False)

    assert controller.advance(0.05).breath == 0.0
    assert controller.frame.blink_weight == 0.0
