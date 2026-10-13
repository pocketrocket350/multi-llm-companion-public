"""利用者が許可したときだけ、会話の間に能動発話を始める。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class ProactiveSettings:
    """利用者が画面で選ぶ能動会話の範囲。"""

    enabled: bool = False
    read_aloud: bool = False
    quiet_start_hour: int = 22
    quiet_end_hour: int = 8
    minimum_idle_minutes: int = 30
    minimum_interval_minutes: int = 120


@dataclass(frozen=True, slots=True)
class ProactiveState:
    """連続した能動発話を防ぐ、画面設定へ保存する状態。"""

    last_delivered_at: datetime | None = None
    last_user_activity_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProactiveTurn:
    """会話LLMへ渡す、利用者が許可した一回分の開始要求。"""

    reason: str = "待機時間と最短間隔を満たしたため"


class ProactiveConversationService:
    """発火条件だけを判定し、発話本文は会話サービスへ委ねる。"""

    def __init__(self, state: ProactiveState | None = None) -> None:
        self._state = state or ProactiveState()

    @property
    def state(self) -> ProactiveState:
        return self._state

    def note_user_activity(self, now: datetime) -> None:
        """利用者が操作した直後は、能動発話の対象から外す。"""

        self._state = ProactiveState(
            last_delivered_at=self._state.last_delivered_at,
            last_user_activity_at=now,
        )

    def disable(self) -> None:
        """無効化時に保留対象を残さず、次回は新しい利用者操作を待つ。"""

        self._state = ProactiveState(last_delivered_at=self._state.last_delivered_at)

    def consider(
        self,
        settings: ProactiveSettings,
        *,
        now: datetime,
        is_busy: bool,
        is_listening: bool,
        is_voice_speaking: bool,
        has_draft: bool,
    ) -> ProactiveTurn | None:
        """許可・静かな時間・会話状態・頻度を満たすときだけ開始する。"""

        if (
            not settings.enabled
            or _is_quiet_hour(now.hour, settings)
            or is_busy
            or is_listening
            or is_voice_speaking
            or has_draft
            or not self._is_idle_long_enough(settings, now)
            or not self._is_interval_elapsed(settings, now)
        ):
            return None
        # 生成失敗や取消も含めて間隔を消費し、連続発話を防ぐ。
        self._state = ProactiveState(
            last_delivered_at=now,
            last_user_activity_at=self._state.last_user_activity_at,
        )
        return ProactiveTurn()

    def _is_idle_long_enough(self, settings: ProactiveSettings, now: datetime) -> bool:
        last_activity = self._state.last_user_activity_at
        if last_activity is None:
            return True
        return now - last_activity >= timedelta(minutes=settings.minimum_idle_minutes)

    def _is_interval_elapsed(self, settings: ProactiveSettings, now: datetime) -> bool:
        last_delivery = self._state.last_delivered_at
        if last_delivery is None:
            return True
        return now - last_delivery >= timedelta(minutes=settings.minimum_interval_minutes)


def _is_quiet_hour(hour: int, settings: ProactiveSettings) -> bool:
    start = settings.quiet_start_hour % 24
    end = settings.quiet_end_hour % 24
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


__all__ = [
    "ProactiveConversationService",
    "ProactiveSettings",
    "ProactiveState",
    "ProactiveTurn",
]
