"""会話状態に応じた、非周期的で軽量な自然動作を作る。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from multi_llm_companion.domain import ConversationState

type ConversationMotionState = ConversationState


@dataclass(frozen=True, slots=True)
class NaturalMotionFrame:
    """GPUと目ボーンへ渡す、一時刻分の小さな動作。"""

    breath: float = 0.0
    neck_yaw: float = 0.0
    neck_pitch: float = 0.0
    head_roll: float = 0.0
    weight_shift: float = 0.0
    posture_pitch: float = 0.0
    nod_pitch: float = 0.0
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0
    blink_weight: float = 0.0


class NaturalMotionController:
    """固定周期を避けながら、状態別の自然動作を連続値で返す。"""

    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)
        self._enabled = True
        self._state = ConversationState.IDLE
        self._emotion = "neutral"
        self._emotion_intensity = 0.0
        self._elapsed = 0.0
        self._state_elapsed = 0.0
        self._breath_phase = 0.0
        self._breath_period = self._random.uniform(3.6, 5.4)
        self._breath_amplitude = self._random.uniform(0.0045, 0.0065)
        self._neck_yaw = _SmoothRandomSignal(self._random, -1.0, 1.0, 3.2, 7.8)
        self._neck_pitch = _SmoothRandomSignal(self._random, -1.0, 1.0, 3.8, 8.6)
        self._head_roll = _SmoothRandomSignal(self._random, -1.0, 1.0, 4.8, 11.5)
        self._weight = _SmoothRandomSignal(self._random, -1.0, 1.0, 4.5, 10.5)
        self._state_neck_yaw = 0.0
        self._state_neck_pitch = 0.0
        self._posture_pitch = 0.0
        self._gaze_yaw = 0.0
        self._gaze_pitch = 0.0
        self._thinking_side = 1.0
        self._speaking_elapsed = 0.0
        self._nod_elapsed = 10.0
        self._blink_elapsed: float | None = None
        self._next_blink_at = self._random.uniform(2.4, 5.8)
        self._double_blink_pending = False
        self._frame = NaturalMotionFrame()

    @property
    def frame(self) -> NaturalMotionFrame:
        return self._frame

    @property
    def state(self) -> ConversationMotionState:
        return self._state

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._blink_elapsed = None
            self._frame = NaturalMotionFrame()

    def set_state(self, state: ConversationMotionState | str) -> None:
        normalized = ConversationState(state)
        if normalized == self._state:
            return
        self._state = normalized
        self._state_elapsed = 0.0
        if normalized == ConversationState.THINKING:
            self._thinking_side = self._random.choice((-1.0, 1.0))
        if normalized == ConversationState.SPEAKING:
            self._speaking_elapsed = 0.0

    def set_emotion(self, emotion: str, intensity: float) -> None:
        self._emotion = emotion.strip().lower() or "neutral"
        self._emotion_intensity = max(0.0, min(1.0, intensity))

    def phrase_boundary(self) -> None:
        """文章中の短い無音を、首だけの小さなうなずきへ変える。"""

        if (
            self._enabled
            and self._state == ConversationState.SPEAKING
            and self._nod_elapsed >= 0.42
        ):
            self._nod_elapsed = 0.0

    def advance(self, delta_seconds: float) -> NaturalMotionFrame:
        safe_delta = max(0.0, min(0.1, delta_seconds))
        if not self._enabled:
            self._frame = NaturalMotionFrame()
            return self._frame
        self._elapsed += safe_delta
        self._state_elapsed += safe_delta
        self._speaking_elapsed += safe_delta
        self._nod_elapsed += safe_delta
        self._advance_breath(safe_delta)
        raw_neck_yaw = self._neck_yaw.advance(safe_delta)
        raw_neck_pitch = self._neck_pitch.advance(safe_delta)
        raw_head_roll = self._head_roll.advance(safe_delta)
        raw_weight = self._weight.advance(safe_delta)
        targets = _state_targets(
            self._state,
            self._thinking_side,
            self._speaking_elapsed,
            self._state_elapsed,
        )
        blend = 1.0 - math.exp(-safe_delta / 0.45) if safe_delta > 0.0 else 0.0
        self._state_neck_yaw = _approach(self._state_neck_yaw, targets.neck_yaw, blend)
        self._state_neck_pitch = _approach(self._state_neck_pitch, targets.neck_pitch, blend)
        self._posture_pitch = _approach(self._posture_pitch, targets.posture_pitch, blend)
        self._gaze_yaw = _approach(self._gaze_yaw, targets.gaze_yaw, blend)
        self._gaze_pitch = _approach(self._gaze_pitch, targets.gaze_pitch, blend)
        strength = _emotion_strength(self._emotion, self._emotion_intensity)
        state_strength = {
            ConversationState.IDLE: 1.0,
            ConversationState.LISTENING: 0.82,
            ConversationState.THINKING: 0.72,
            ConversationState.REACTING: 0.88,
            ConversationState.SPEAKING: 1.05,
            ConversationState.INTERRUPTED: 0.65,
        }[self._state]
        nod_pitch = _nod_value(self._nod_elapsed) * strength
        self._frame = NaturalMotionFrame(
            breath=_breath_value(self._breath_phase, self._breath_amplitude)
            * max(0.72, min(1.25, strength)),
            neck_yaw=(raw_neck_yaw * 0.016 * state_strength + self._state_neck_yaw) * strength,
            neck_pitch=(raw_neck_pitch * 0.010 * state_strength + self._state_neck_pitch)
            * strength,
            head_roll=raw_head_roll * 0.012 * state_strength * strength,
            weight_shift=(raw_weight * 0.0062 + targets.weight_shift) * strength,
            posture_pitch=self._posture_pitch * strength,
            nod_pitch=nod_pitch,
            gaze_yaw=self._gaze_yaw,
            gaze_pitch=self._gaze_pitch,
            blink_weight=self._advance_blink(safe_delta),
        )
        return self._frame

    def _advance_breath(self, delta_seconds: float) -> None:
        self._breath_phase += delta_seconds * math.tau / self._breath_period
        if self._breath_phase < math.tau:
            return
        self._breath_phase %= math.tau
        self._breath_period = self._random.uniform(3.6, 5.4)
        self._breath_amplitude = self._random.uniform(0.0045, 0.0065)

    def _advance_blink(self, delta_seconds: float) -> float:
        if self._blink_elapsed is None:
            if self._elapsed < self._next_blink_at:
                return 0.0
            self._blink_elapsed = 0.0
        self._blink_elapsed += delta_seconds
        duration = 0.19
        progress = self._blink_elapsed / duration
        if progress >= 1.0:
            self._blink_elapsed = None
            if self._double_blink_pending:
                self._next_blink_at = self._elapsed + self._random.uniform(0.16, 0.24)
                self._double_blink_pending = False
            else:
                self._next_blink_at = self._elapsed + self._random.uniform(2.4, 6.8)
                self._double_blink_pending = self._random.random() < 0.14
            return 0.0
        if progress < 0.34:
            return progress / 0.34
        return max(0.0, 1.0 - (progress - 0.34) / 0.66)


@dataclass(frozen=True, slots=True)
class _StateTargets:
    neck_yaw: float
    neck_pitch: float
    weight_shift: float
    posture_pitch: float
    gaze_yaw: float
    gaze_pitch: float


def _state_targets(
    state: ConversationMotionState,
    thinking_side: float,
    speaking_elapsed: float,
    state_elapsed: float,
) -> _StateTargets:
    if state == ConversationState.LISTENING:
        return _StateTargets(0.0, -0.012, 0.0008, -0.008, 0.0, 0.0)
    if state == ConversationState.THINKING:
        down_progress = _smooth_progress(state_elapsed, start=0.65, duration=0.60)
        return _StateTargets(
            thinking_side * (0.018 - down_progress * 0.012),
            0.004 + down_progress * 0.012,
            -thinking_side * 0.0008,
            0.006,
            thinking_side * (0.11 - down_progress * 0.07),
            -down_progress * 0.08,
        )
    if state == ConversationState.REACTING:
        return _StateTargets(0.0, -0.006, 0.0005, -0.010, 0.0, 0.0)
    if state == ConversationState.SPEAKING:
        onset = math.exp(-speaking_elapsed / 0.7)
        return _StateTargets(0.0, -0.004, 0.0005, -0.004 - onset * 0.012, 0.0, 0.0)
    return _StateTargets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _emotion_strength(emotion: str, intensity: float) -> float:
    targets = {
        "happy": 1.0 + intensity * 0.22,
        "surprised": 1.0 + intensity * 0.35,
        "angry": 1.0 + intensity * 0.18,
        "sad": 1.0 - intensity * 0.25,
        "embarrassed": 1.0 - intensity * 0.08,
    }
    return max(0.68, min(1.35, targets.get(emotion, 1.0)))


def _breath_value(phase: float, amplitude: float) -> float:
    return amplitude * (0.5 - math.cos(phase) * 0.5)


def _nod_value(elapsed: float) -> float:
    duration = 0.52
    if not 0.0 <= elapsed <= duration:
        return 0.0
    return math.sin(math.pi * elapsed / duration) * 0.045


def _approach(current: float, target: float, ratio: float) -> float:
    return current + (target - current) * ratio


def _smooth_progress(elapsed: float, *, start: float, duration: float) -> float:
    progress = max(0.0, min(1.0, (elapsed - start) / duration))
    return progress * progress * (3.0 - 2.0 * progress)


class _SmoothRandomSignal:
    """ランダムな目標を滑らかにつなぎ、固定周期を作らない。"""

    def __init__(
        self,
        source: random.Random,
        minimum: float,
        maximum: float,
        minimum_duration: float,
        maximum_duration: float,
    ) -> None:
        self._random = source
        self._minimum = minimum
        self._maximum = maximum
        self._minimum_duration = minimum_duration
        self._maximum_duration = maximum_duration
        self._start = 0.0
        self._target = self._random.uniform(minimum, maximum)
        self._elapsed = 0.0
        self._duration = self._random.uniform(minimum_duration, maximum_duration)

    def advance(self, delta_seconds: float) -> float:
        self._elapsed += delta_seconds
        while self._elapsed >= self._duration:
            self._elapsed -= self._duration
            self._start = self._target
            self._target = self._random.uniform(self._minimum, self._maximum)
            self._duration = self._random.uniform(
                self._minimum_duration,
                self._maximum_duration,
            )
        progress = self._elapsed / self._duration
        smooth = progress * progress * (3.0 - 2.0 * progress)
        return self._start + (self._target - self._start) * smooth
