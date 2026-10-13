"""複数のMMD動作を優先順位に従って安全に合成する。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

type Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MotionLayer:
    """一つの動作源が要求するボーン・モーフ変形。"""

    rotations: dict[int, Vector3] = field(default_factory=dict)
    translations: dict[int, Vector3] = field(default_factory=dict)
    morphs: dict[str, float] = field(default_factory=dict)
    rotation_limit: float = 0.7
    translation_limit: float = 0.35
    pose_weight: float = 1.0
    exact: bool = False


@dataclass(frozen=True, slots=True)
class MixedMotion:
    """優先順位の解決と補間を終えた一つの姿勢。"""

    rotations: dict[int, Vector3] = field(default_factory=dict)
    translations: dict[int, Vector3] = field(default_factory=dict)
    morphs: dict[str, float] = field(default_factory=dict)


class MmdMotionMixer:
    """口・表情・視線・動作・VMD・物理の競合を一か所で解決する。"""

    def __init__(self, *, transition_seconds: float = 0.35) -> None:
        self._transition_seconds = max(0.05, transition_seconds)
        self._current = MixedMotion()
        self._emergency_stopped = False
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def current(self) -> MixedMotion:
        return self._current

    def interrupt(self) -> int:
        """古い会話動作の結果を無効化する番号を進める。"""

        self._revision += 1
        return self._revision

    def set_emergency_stopped(self, stopped: bool) -> None:
        """停止中は現在姿勢を保持し、新しい動作を混ぜない。"""

        if stopped != self._emergency_stopped:
            self._emergency_stopped = stopped
            self._revision += 1

    def mix(
        self,
        *,
        delta_seconds: float,
        physics: MotionLayer | None = None,
        vmd: MotionLayer | None = None,
        idle: MotionLayer | None = None,
        gesture: MotionLayer | None = None,
        gaze: MotionLayer | None = None,
        expression: MotionLayer | None = None,
        protected_morphs: frozenset[str] = frozenset(),
    ) -> MixedMotion:
        """低い優先度から上書きし、急な姿勢飛びを補間で抑える。"""

        if self._emergency_stopped:
            return self._current

        # 後から適用する層ほど優先度が高い。口モーフは高速経路が所有する。
        ordered = tuple(
            layer or MotionLayer() for layer in (physics, vmd, idle, gesture, gaze, expression)
        )
        target_rotations: dict[int, Vector3] = {}
        target_translations: dict[int, Vector3] = {}
        target_morphs: dict[str, float] = {}
        exact_rotation_indexes: set[int] = set()
        exact_translation_indexes: set[int] = set()
        for layer in ordered:
            rotation_limit = max(0.0, min(3.2, layer.rotation_limit))
            translation_limit = max(0.0, min(2.0, layer.translation_limit))
            pose_weight = max(0.0, min(2.0, layer.pose_weight))
            if pose_weight > 0.0:
                target_rotations.update(
                    {
                        index: _weighted_vector(
                            target_rotations.get(index, (0.0, 0.0, 0.0)),
                            value,
                            pose_weight,
                            rotation_limit,
                        )
                        for index, value in layer.rotations.items()
                    }
                )
                target_translations.update(
                    {
                        index: _weighted_vector(
                            target_translations.get(index, (0.0, 0.0, 0.0)),
                            value,
                            pose_weight,
                            translation_limit,
                        )
                        for index, value in layer.translations.items()
                    }
                )
            if layer.exact:
                exact_rotation_indexes.update(layer.rotations)
                exact_translation_indexes.update(layer.translations)
            elif pose_weight > 0.0:
                exact_rotation_indexes.difference_update(layer.rotations)
                exact_translation_indexes.difference_update(layer.translations)
            target_morphs.update(
                {
                    name: max(0.0, min(1.0, weight))
                    for name, weight in layer.morphs.items()
                    if name not in protected_morphs
                }
            )

        ratio = max(0.0, min(1.0, delta_seconds / self._transition_seconds))
        rotations = _interpolate_rotations(
            self._current.rotations, target_rotations, ratio, exact_rotation_indexes
        )
        translations = _interpolate_vectors(
            self._current.translations,
            target_translations,
            ratio,
        )
        rotations.update({index: target_rotations[index] for index in exact_rotation_indexes})
        translations.update(
            {index: target_translations[index] for index in exact_translation_indexes}
        )
        self._current = MixedMotion(
            rotations=rotations,
            translations=translations,
            morphs=_interpolate_scalars(self._current.morphs, target_morphs, ratio),
        )
        return self._current


def _safe_vector(value: Vector3, limit: float) -> Vector3:
    return tuple(max(-limit, min(limit, component)) for component in value)  # type: ignore[return-value]


def _weighted_vector(before: Vector3, after: Vector3, weight: float, limit: float) -> Vector3:
    blended = tuple(before[axis] + (after[axis] - before[axis]) * weight for axis in range(3))
    return _safe_vector(blended, limit)  # type: ignore[arg-type]


def _interpolate_vectors(
    current: dict[int, Vector3],
    target: dict[int, Vector3],
    ratio: float,
) -> dict[int, Vector3]:
    result: dict[int, Vector3] = {}
    for index in current.keys() | target.keys():
        before = current.get(index, (0.0, 0.0, 0.0))
        after = target.get(index, (0.0, 0.0, 0.0))
        value = tuple(before[axis] + (after[axis] - before[axis]) * ratio for axis in range(3))
        if any(abs(component) > 0.00001 for component in value):
            result[index] = value  # type: ignore[assignment]
    return result


def _nearest_euler(before: Vector3, after: Vector3) -> Vector3:
    """同じ向きの二つのZ-Y-X解から、現在の角度表現に近い方を選ぶ。"""

    x, y, z = after
    candidates = (
        tuple(
            before[axis] + math.remainder(value[axis] - before[axis], math.tau) for axis in range(3)
        )
        for value in (after, (x + math.pi, math.pi - y, z + math.pi))
    )
    return min(candidates, key=lambda value: sum((value[i] - before[i]) ** 2 for i in range(3)))  # type: ignore[return-value]


def _interpolate_rotations(
    current: dict[int, Vector3],
    target: dict[int, Vector3],
    ratio: float,
    exact_indexes: set[int],
) -> dict[int, Vector3]:
    result: dict[int, Vector3] = {}
    for index in current.keys() | target.keys():
        if index in exact_indexes:
            continue
        before = current.get(index, (0.0, 0.0, 0.0))
        after = target.get(index, (0.0, 0.0, 0.0))
        if ratio == 0 or before == after:
            value = before
        elif ratio == 1:
            value = after
        else:
            # 通常の軌道は変えず、角度表現の折り返し・別解だけを連続化する。
            after = _nearest_euler(before, after)
            value = tuple(before[i] + (after[i] - before[i]) * ratio for i in range(3))
        if any(abs(component) > 0.00001 for component in value):
            result[index] = value
    return result


def _interpolate_scalars(
    current: dict[str, float],
    target: dict[str, float],
    ratio: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in current.keys() | target.keys():
        value = current.get(name, 0.0) + (target.get(name, 0.0) - current.get(name, 0.0)) * ratio
        if value > 0.00001:
            result[name] = value
    return result
