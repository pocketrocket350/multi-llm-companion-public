"""会話ジェスチャーを選び、同じ動作の連続を抑える。"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from multi_llm_companion.domain import normalize_gesture_intent

from .base import AvatarVmdMotion


@dataclass(frozen=True, slots=True)
class GestureCandidate:
    """登録済みの会話動作候補。"""

    gesture_id: str
    intent: str
    category: str
    body_mask: tuple[str, ...] = ()
    cooldown_seconds: float = 0.0
    intensity_scale: float = 1.0
    motion: AvatarVmdMotion | None = None
    motion_type: str = "talk"
    moods: tuple[str, ...] = ()
    energy: str = "medium"
    weight: float = 1.0
    audio_required: bool = False

    @property
    def family(self) -> str:
        return normalize_gesture_intent(self.intent)

    @property
    def is_large(self) -> bool:
        large_parts = {"arms", "both_arms", "upper_body", "full_body"}
        return bool(large_parts.intersection(self.body_mask))


@dataclass(frozen=True, slots=True)
class GestureSelection:
    """選択を終えた一回分の会話ジェスチャー。"""

    gesture: str = "none"
    candidate: GestureCandidate | None = None
    reason: str = "動作なし"


class GestureSelector:
    """意味タグ、間隔、履歴から一件だけを決める軽量な選択器。"""

    def __init__(
        self,
        candidates: Iterable[GestureCandidate] = (),
        *,
        enabled: bool = True,
        min_interval_seconds: float = 3.5,
        same_gesture_cooldown_seconds: float = 20.0,
        same_category_cooldown_seconds: float = 6.0,
        max_large_per_minute: int = 2,
        recent_motion_count: int = 5,
        clock: Callable[[], float] = time.monotonic,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._enabled = enabled
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        self._same_gesture_cooldown_seconds = max(0.0, same_gesture_cooldown_seconds)
        self._same_category_cooldown_seconds = max(0.0, same_category_cooldown_seconds)
        self._max_large_per_minute = max(0, max_large_per_minute)
        self._recent_motion_count = max(0, recent_motion_count)
        self._clock = clock
        self._random_source = random_source
        # 採用された候補だけを使い、組み込み動作では代用しない。
        self._candidates = tuple(candidates)
        self._candidate_order = {
            candidate.gesture_id: index for index, candidate in enumerate(self._candidates)
        }
        self._last_selected_at: float | None = None
        self._last_family_at: dict[str, float] = {}
        self._last_category_at: dict[str, float] = {}
        self._last_candidate_at: dict[str, float] = {}
        self._large_history: deque[float] = deque()
        self._recent_motion_ids: deque[str] = deque(maxlen=self._recent_motion_count)

    def select(
        self,
        requested_intent: str,
        *,
        mood: str = "",
        energy: float | None = None,
        audio_available: bool = False,
    ) -> GestureSelection:
        """LLMの意味タグを正規化し、抑制条件を通った一件を返す。"""

        family = normalize_gesture_intent(requested_intent)
        if family == "none":
            return GestureSelection(reason="LLMが動作なしを指定")
        matches = tuple(
            candidate
            for candidate in self._candidates
            if candidate.family == family and (audio_available or not candidate.audio_required)
        )
        if not matches:
            return GestureSelection(reason=f"未対応の動作: {requested_intent.strip()}")
        matches = self._prefer_metadata_matches(matches, mood=mood, energy=energy)

        now = self._clock()
        if self._enabled and self._is_global_interval_active(now):
            return GestureSelection(reason="動作間隔を確保中")

        eligible = tuple(
            candidate
            for candidate in matches
            if not self._enabled or self._candidate_is_available(candidate, now)
        )
        if not eligible:
            return GestureSelection(reason="同じ動作の連続を抑制中")

        without_recent = tuple(
            candidate
            for candidate in eligible
            if candidate.gesture_id not in self._recent_motion_ids
        )
        selected = self._weighted_choice(without_recent or eligible)
        self._record(selected, now)
        return GestureSelection(gesture=selected.family, candidate=selected, reason="選択済み")

    def reset(self) -> None:
        """アプリ終了時などに選択履歴を破棄する。"""

        self._last_selected_at = None
        self._last_family_at.clear()
        self._last_category_at.clear()
        self._last_candidate_at.clear()
        self._large_history.clear()
        self._recent_motion_ids.clear()

    def _prefer_metadata_matches(
        self,
        candidates: tuple[GestureCandidate, ...],
        *,
        mood: str,
        energy: float | None,
    ) -> tuple[GestureCandidate, ...]:
        normalized_mood = mood.strip().lower()
        if normalized_mood:
            mood_matches = tuple(
                candidate for candidate in candidates if normalized_mood in candidate.moods
            )
            generic = tuple(candidate for candidate in candidates if not candidate.moods)
            candidates = mood_matches or generic or candidates
        if energy is not None:
            energy_name = _energy_name(energy)
            energy_matches = tuple(
                candidate for candidate in candidates if candidate.energy == energy_name
            )
            if energy_matches:
                candidates = energy_matches
        return candidates

    def _weighted_choice(
        self,
        candidates: tuple[GestureCandidate, ...],
    ) -> GestureCandidate:
        total = sum(max(0.0, candidate.weight) for candidate in candidates)
        if total <= 0.0:
            return min(
                candidates, key=lambda candidate: self._candidate_order[candidate.gesture_id]
            )
        ratio = min(max(self._random_source(), 0.0), 0.999999999999)
        threshold = ratio * total
        cumulative = 0.0
        for candidate in candidates:
            cumulative += max(0.0, candidate.weight)
            if threshold < cumulative:
                return candidate
        return candidates[-1]

    def _is_global_interval_active(self, now: float) -> bool:
        return (
            self._last_selected_at is not None
            and now - self._last_selected_at < self._min_interval_seconds
        )

    def _candidate_is_available(self, candidate: GestureCandidate, now: float) -> bool:
        family_last = self._last_family_at.get(candidate.family)
        family_cooldown = max(
            self._same_gesture_cooldown_seconds,
            candidate.cooldown_seconds,
        )
        if family_last is not None and now - family_last < family_cooldown:
            return False
        category_last = self._last_category_at.get(candidate.category)
        if category_last is not None and now - category_last < self._same_category_cooldown_seconds:
            return False
        self._remove_old_large_history(now)
        return not (
            candidate.is_large
            and self._max_large_per_minute > 0
            and len(self._large_history) >= self._max_large_per_minute
        )

    def _record(self, candidate: GestureCandidate, now: float) -> None:
        self._last_selected_at = now
        self._last_family_at[candidate.family] = now
        self._last_category_at[candidate.category] = now
        self._last_candidate_at[candidate.gesture_id] = now
        self._recent_motion_ids.append(candidate.gesture_id)
        self._remove_old_large_history(now)
        if candidate.is_large:
            self._large_history.append(now)

    def _remove_old_large_history(self, now: float) -> None:
        while self._large_history and now - self._large_history[0] >= 60.0:
            self._large_history.popleft()


def _energy_name(value: float) -> str:
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"
