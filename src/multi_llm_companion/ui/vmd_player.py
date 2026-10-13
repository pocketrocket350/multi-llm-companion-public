"""第28段階のVMDボーンキーフレーム再生。"""

from __future__ import annotations

import math
from collections import defaultdict

from multi_llm_companion.ui.models import (
    UiAvatarModelPreview,
    UiVmdBoneFrame,
    UiVmdCameraFrame,
    UiVmdIkFrame,
    UiVmdLightFrame,
    UiVmdMorphFrame,
    UiVmdMotion,
    UiVmdShadowFrame,
)

type Vector3 = tuple[float, float, float]
_LOOP_TRANSITION_SECONDS = 0.25


class VmdMotionPlayer:
    """VMDのボーンフレームを30fps・ベジェ補間で再生する。"""

    def __init__(self) -> None:
        self._motion: UiVmdMotion | None = None
        self._frames_by_name: dict[str, tuple[UiVmdBoneFrame, ...]] = {}
        self._morph_frames_by_name: dict[str, tuple[UiVmdMorphFrame, ...]] = {}
        self._camera_frames: tuple[UiVmdCameraFrame, ...] = ()
        self._light_frames: tuple[UiVmdLightFrame, ...] = ()
        self._shadow_frames: tuple[UiVmdShadowFrame, ...] = ()
        self._ik_frames: tuple[UiVmdIkFrame, ...] = ()
        self._elapsed_seconds = 0.0
        self._camera_frame: UiVmdCameraFrame | None = None
        self._light_frame: UiVmdLightFrame | None = None
        self._shadow_frame: UiVmdShadowFrame | None = None
        self._ik_states: dict[str, bool] = {}
        self._display_model = True
        self._last_output: (
            tuple[dict[int, Vector3], dict[int, Vector3], dict[str, float]] | None
        ) = None
        self._loop_transition_from: (
            tuple[dict[int, Vector3], dict[int, Vector3], dict[str, float]] | None
        ) = None
        self._loop_transition_remaining = 0.0

    @property
    def status_text(self) -> str:
        motion = self._motion
        if motion is None:
            return "VMD未設定"
        sections = (
            f"{motion.frame_count}フレーム・表情 {len(motion.morph_frames)}・"
            f"IK {len(motion.ik_frames)}・カメラ {len(motion.camera_frames)}・"
            f"照明 {len(motion.light_frames)}・影 {len(motion.shadow_frames)}"
        )
        return f"VMD再生: {motion.name}（{sections}）{_compatibility_text(motion)}"

    @property
    def camera_frame(self) -> UiVmdCameraFrame | None:
        return self._camera_frame

    @property
    def light_frame(self) -> UiVmdLightFrame | None:
        return self._light_frame

    @property
    def shadow_frame(self) -> UiVmdShadowFrame | None:
        return self._shadow_frame

    @property
    def ik_states(self) -> dict[str, bool]:
        return dict(self._ik_states)

    @property
    def display_model(self) -> bool:
        return self._display_model

    @property
    def is_finished(self) -> bool:
        """非ループVMDが終端へ到達した場合だけTrueを返す。"""

        motion = self._motion
        if motion is None or motion.loop:
            return False
        return self._elapsed_seconds >= max(1, motion.frame_count) / 30.0

    def reset(self, motion: UiVmdMotion | None) -> None:
        self._motion = motion
        self._elapsed_seconds = 0.0
        grouped: dict[str, list[UiVmdBoneFrame]] = defaultdict(list)
        grouped_morphs: dict[str, list[UiVmdMorphFrame]] = defaultdict(list)
        if motion is not None:
            for frame in motion.bone_frames:
                grouped[frame.bone_name].append(frame)
            for frame in motion.morph_frames:
                grouped_morphs[frame.morph_name].append(frame)
        self._frames_by_name = {
            name: tuple(sorted(frames, key=lambda frame: frame.frame_number))
            for name, frames in grouped.items()
        }
        self._morph_frames_by_name = {
            name: tuple(sorted(frames, key=lambda frame: frame.frame_number))
            for name, frames in grouped_morphs.items()
        }
        self._camera_frames = (
            tuple(sorted(motion.camera_frames, key=lambda frame: frame.frame_number))
            if motion is not None
            else ()
        )
        self._light_frames = (
            tuple(sorted(motion.light_frames, key=lambda frame: frame.frame_number))
            if motion is not None
            else ()
        )
        self._shadow_frames = (
            tuple(sorted(motion.shadow_frames, key=lambda frame: frame.frame_number))
            if motion is not None
            else ()
        )
        self._ik_frames = (
            tuple(sorted(motion.ik_frames, key=lambda frame: frame.frame_number))
            if motion is not None
            else ()
        )
        self._camera_frame = None
        self._light_frame = None
        self._shadow_frame = None
        self._ik_states = {}
        self._display_model = True
        self._last_output = None
        self._loop_transition_from = None
        self._loop_transition_remaining = 0.0

    def advance(
        self,
        preview: UiAvatarModelPreview | None,
        delta_seconds: float,
        enabled: bool,
    ) -> tuple[dict[int, Vector3], dict[int, Vector3], dict[str, float]]:
        """現在時刻のボーン変形と表情モーフの強さを返す。"""

        motion = preview.vmd_motion if preview is not None else None
        return self.advance_motion(preview, motion, delta_seconds, enabled)

    def advance_motion(
        self,
        preview: UiAvatarModelPreview | None,
        motion: UiVmdMotion | None,
        delta_seconds: float,
        enabled: bool,
    ) -> tuple[dict[int, Vector3], dict[int, Vector3], dict[str, float]]:
        """PMXを再設定せず、指定した会話VMDだけを進める。"""

        if motion != self._motion:
            self.reset(motion)
        motion = self._motion
        if motion is None or not enabled or not _has_playable_frames(motion):
            self._camera_frame = None
            self._light_frame = None
            self._shadow_frame = None
            self._ik_states = {}
            self._display_model = True
            return {}, {}, {}
        # 重いメッシュ更新を低頻度にしても、VMDの再生時間は実時間へ合わせる。
        previous_elapsed = self._elapsed_seconds
        self._elapsed_seconds += max(0.0, min(delta_seconds, 1.0))
        duration = max(1, motion.frame_count) / 30.0
        if motion.loop:
            self._elapsed_seconds %= duration
        else:
            self._elapsed_seconds = min(self._elapsed_seconds, duration)
        looped = motion.loop and self._elapsed_seconds < previous_elapsed
        frame_position = self._elapsed_seconds * 30.0
        indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
        rotations: dict[int, Vector3] = {}
        translations: dict[int, Vector3] = {}
        for name, frames in self._frames_by_name.items():
            bone_index = indexes.get(name)
            if bone_index is None:
                continue
            lower, upper = _surrounding_frames(frames, frame_position)
            ratio = _ratio(lower.frame_number, upper.frame_number, frame_position)
            translations[bone_index] = tuple(
                lower.translation[index]
                + (upper.translation[index] - lower.translation[index])
                * _bezier_ratio(ratio, upper.interpolation[index])
                for index in range(3)
            )  # type: ignore[assignment]
            quaternion = _slerp_quaternion(
                lower.rotation,
                upper.rotation,
                _bezier_ratio(ratio, upper.interpolation[3]),
            )
            rotations[bone_index] = _quaternion_to_euler(quaternion)
        morph_weights: dict[str, float] = {}
        for name, frames in self._morph_frames_by_name.items():
            lower, upper = _surrounding_morph_frames(frames, frame_position)
            ratio = _ratio(lower.frame_number, upper.frame_number, frame_position)
            morph_weights[name] = max(
                0.0, min(1.0, lower.weight + (upper.weight - lower.weight) * ratio)
            )
        intensity = max(0.0, min(2.0, motion.intensity_scale))
        rotations = {
            index: tuple(value * intensity for value in rotation)
            for index, rotation in rotations.items()
        }
        translations = {
            index: tuple(value * intensity for value in translation)
            for index, translation in translations.items()
        }
        morph_weights = {
            name: max(0.0, min(1.0, weight * intensity)) for name, weight in morph_weights.items()
        }
        self._camera_frame = _sample_camera(self._camera_frames, frame_position)
        self._light_frame = _sample_light(self._light_frames, frame_position)
        self._shadow_frame = _latest_frame(self._shadow_frames, frame_position)
        self._ik_states, self._display_model = _sample_ik(self._ik_frames, frame_position)
        if looped and self._last_output is not None:
            self._loop_transition_from = self._last_output
            self._loop_transition_remaining = min(_LOOP_TRANSITION_SECONDS, duration)
        if self._loop_transition_from is not None and self._loop_transition_remaining > 0.0:
            window = min(_LOOP_TRANSITION_SECONDS, duration)
            elapsed = window - self._loop_transition_remaining + max(0.0, delta_seconds)
            ratio = max(0.0, min(1.0, elapsed / max(window, 0.000001)))
            rotations = _interpolate_vector_maps(self._loop_transition_from[0], rotations, ratio)
            translations = _interpolate_vector_maps(
                self._loop_transition_from[1], translations, ratio
            )
            morph_weights = _interpolate_scalar_maps(
                self._loop_transition_from[2], morph_weights, ratio
            )
            self._loop_transition_remaining = max(
                0.0, self._loop_transition_remaining - max(0.0, delta_seconds)
            )
            if self._loop_transition_remaining <= 0.0:
                self._loop_transition_from = None
        self._last_output = (dict(rotations), dict(translations), dict(morph_weights))
        return rotations, translations, morph_weights


def _has_playable_frames(motion: UiVmdMotion) -> bool:
    return bool(
        motion.bone_frames
        or motion.morph_frames
        or motion.camera_frames
        or motion.light_frames
        or motion.shadow_frames
        or motion.ik_frames
    )


def _sample_camera(
    frames: tuple[UiVmdCameraFrame, ...],
    position: float,
) -> UiVmdCameraFrame | None:
    if not frames:
        return None
    lower, upper = _surrounding_camera_frames(frames, position)
    ratio = _ratio(lower.frame_number, upper.frame_number, position)
    channel_ratios = tuple(_bezier_ratio(ratio, upper.interpolation[index]) for index in range(6))
    return UiVmdCameraFrame(
        frame_number=round(position),
        distance=lower.distance + (upper.distance - lower.distance) * channel_ratios[4],
        position=tuple(
            lower.position[index]
            + (upper.position[index] - lower.position[index]) * channel_ratios[index]
            for index in range(3)
        ),  # type: ignore[arg-type]
        rotation=tuple(
            lower.rotation[index]
            + (upper.rotation[index] - lower.rotation[index]) * channel_ratios[3]
            for index in range(3)
        ),  # type: ignore[arg-type]
        interpolation=upper.interpolation,
        view_angle=round(
            lower.view_angle + (upper.view_angle - lower.view_angle) * channel_ratios[5]
        ),
        perspective=lower.perspective if ratio < 1.0 else upper.perspective,
    )


def _sample_light(
    frames: tuple[UiVmdLightFrame, ...],
    position: float,
) -> UiVmdLightFrame | None:
    if not frames:
        return None
    lower, upper = _surrounding_light_frames(frames, position)
    ratio = _ratio(lower.frame_number, upper.frame_number, position)
    return UiVmdLightFrame(
        frame_number=round(position),
        color=tuple(
            lower.color[index] + (upper.color[index] - lower.color[index]) * ratio
            for index in range(3)
        ),  # type: ignore[arg-type]
        position=tuple(
            lower.position[index] + (upper.position[index] - lower.position[index]) * ratio
            for index in range(3)
        ),  # type: ignore[arg-type]
    )


def _latest_frame[T](frames: tuple[T, ...], position: float) -> T | None:
    latest: T | None = None
    for frame in frames:
        if frame.frame_number > position:  # type: ignore[attr-defined]
            break
        latest = frame
    return latest or (frames[0] if frames else None)


def _sample_ik(
    frames: tuple[UiVmdIkFrame, ...],
    position: float,
) -> tuple[dict[str, bool], bool]:
    states = {state.ik_name: True for frame in frames for state in frame.states}
    display_model = True
    for frame in frames:
        if frame.frame_number > position:
            break
        display_model = frame.display_model
        states.update({state.ik_name: state.enabled for state in frame.states})
    return states, display_model


def _surrounding_camera_frames(
    frames: tuple[UiVmdCameraFrame, ...],
    position: float,
) -> tuple[UiVmdCameraFrame, UiVmdCameraFrame]:
    lower = frames[0]
    for upper in frames[1:]:
        if position <= upper.frame_number:
            return lower, upper
        lower = upper
    return lower, lower


def _surrounding_light_frames(
    frames: tuple[UiVmdLightFrame, ...],
    position: float,
) -> tuple[UiVmdLightFrame, UiVmdLightFrame]:
    lower = frames[0]
    for upper in frames[1:]:
        if position <= upper.frame_number:
            return lower, upper
        lower = upper
    return lower, lower


def _interpolate_vector_maps(
    first: dict[int, Vector3],
    second: dict[int, Vector3],
    ratio: float,
) -> dict[int, Vector3]:
    result: dict[int, Vector3] = {}
    for index in first.keys() | second.keys():
        before = first.get(index, (0.0, 0.0, 0.0))
        after = second.get(index, (0.0, 0.0, 0.0))
        value = tuple(before[axis] + (after[axis] - before[axis]) * ratio for axis in range(3))
        if any(abs(component) > 0.00001 for component in value):
            result[index] = value  # type: ignore[assignment]
    return result


def _interpolate_scalar_maps(
    first: dict[str, float],
    second: dict[str, float],
    ratio: float,
) -> dict[str, float]:
    return {
        name: value
        for name in first.keys() | second.keys()
        if (value := first.get(name, 0.0) + (second.get(name, 0.0) - first.get(name, 0.0)) * ratio)
        > 0.00001
    }


def _compatibility_text(motion: UiVmdMotion) -> str:
    report = motion.compatibility
    if report is None:
        return ""
    bone_total = len(report.supported_bones) + len(report.missing_bones)
    morph_total = len(report.supported_morphs) + len(report.missing_morphs)
    ik_total = len(report.supported_iks) + len(report.missing_iks)
    parts = [
        f"対応 ボーン{len(report.supported_bones)}/{bone_total}",
        f"モーフ{len(report.supported_morphs)}/{morph_total}",
        f"IK{len(report.supported_iks)}/{ik_total}",
        f"最大移動/身長{report.translation_height_ratio:.0%}",
    ]
    if report.target_model_matches is False:
        parts.append("対象モデル名不一致")
    missing = (*report.missing_bones, *report.missing_morphs, *report.missing_iks)
    if missing:
        parts.append("未検出 " + "、".join(missing[:3]))
    if report.unsupported_sections:
        parts.append("未対応 " + "、".join(report.unsupported_sections))
    return " / " + "・".join(parts)


def _surrounding_frames(
    frames: tuple[UiVmdBoneFrame, ...],
    position: float,
) -> tuple[UiVmdBoneFrame, UiVmdBoneFrame]:
    lower = frames[0]
    for upper in frames[1:]:
        if position <= upper.frame_number:
            return lower, upper
        lower = upper
    return lower, lower


def _ratio(lower: int, upper: int, position: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (position - lower) / (upper - lower)))


def _surrounding_morph_frames(
    frames: tuple[UiVmdMorphFrame, ...],
    position: float,
) -> tuple[UiVmdMorphFrame, UiVmdMorphFrame]:
    lower = frames[0]
    for upper in frames[1:]:
        if position <= upper.frame_number:
            return lower, upper
        lower = upper
    return lower, lower


def _quaternion_to_euler(quaternion: tuple[float, float, float, float]) -> Vector3:
    """VMDのx,y,z,wクォータニオンを既存ボーン計算用のオイラー角へ変換する。"""

    x, y, z, w = quaternion
    size = math.sqrt(x * x + y * y + z * z + w * w)
    if size < 0.00001:
        return (0.0, 0.0, 0.0)
    x, y, z, w = (value / size for value in (x, y, z, w))
    yaw_value = 2.0 * (w * y - z * x)
    if abs(yaw_value) >= 1.0 - 1e-12:
        # 真上・真下では独立したX/Z角を求めず、同じ向きの解へまとめる。
        pitch = math.atan2(2.0 * (w * x - y * z), 1.0 - 2.0 * (x * x + z * z))
        return (pitch, math.copysign(math.pi / 2.0, yaw_value), 0.0)
    pitch = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = math.asin(yaw_value)
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (pitch, yaw, roll)


def _bezier_ratio(ratio: float, curve: tuple[int, int, int, int]) -> float:
    """VMDの制御点から、指定時刻に対応する変化量を求める。"""

    ratio = max(0.0, min(1.0, ratio))
    if ratio in {0.0, 1.0}:
        return ratio
    x1, y1, x2, y2 = (max(0, min(127, value)) / 127.0 for value in curve)
    if abs(x1 - y1) < 1e-7 and abs(x2 - y2) < 1e-7:
        return ratio
    lower = 0.0
    upper = 1.0
    parameter = ratio
    for _ in range(15):
        parameter = (lower + upper) * 0.5
        x = _cubic_bezier(parameter, x1, x2)
        if x < ratio:
            lower = parameter
        else:
            upper = parameter
    return _cubic_bezier(parameter, y1, y2)


def _cubic_bezier(parameter: float, first: float, second: float) -> float:
    inverse = 1.0 - parameter
    return (
        3.0 * inverse * inverse * parameter * first
        + 3.0 * inverse * parameter * parameter * second
        + parameter * parameter * parameter
    )


def _slerp_quaternion(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    """回転が遠回りしないよう、単位クォータニオンを球面補間する。"""

    start = _normalize_quaternion(first)
    end = _normalize_quaternion(second)
    dot = sum(left * right for left, right in zip(start, end, strict=True))
    if dot < 0.0:
        end = tuple(-value for value in end)  # type: ignore[assignment]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(left + (right - left) * ratio for left, right in zip(start, end, strict=True))
        )
    angle = math.acos(dot)
    denominator = math.sin(angle)
    start_scale = math.sin((1.0 - ratio) * angle) / denominator
    end_scale = math.sin(ratio * angle) / denominator
    return tuple(
        start_scale * left + end_scale * right for left, right in zip(start, end, strict=True)
    )  # type: ignore[return-value]


def _normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    size = math.sqrt(sum(value * value for value in quaternion))
    if size < 0.00001:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / size for value in quaternion)  # type: ignore[return-value]
