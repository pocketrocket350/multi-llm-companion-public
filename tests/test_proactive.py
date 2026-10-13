from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from multi_llm_companion.services.proactive import (
    ProactiveConversationService,
    ProactiveSettings,
    ProactiveState,
)

NOW = datetime(2026, 9, 22, 12, 0)
SETTINGS = ProactiveSettings(
    enabled=True,
    quiet_start_hour=22,
    quiet_end_hour=8,
    minimum_idle_minutes=30,
    minimum_interval_minutes=120,
)


def _consider(
    service: ProactiveConversationService,
    *,
    now: datetime = NOW,
    settings: ProactiveSettings = SETTINGS,
    is_busy: bool = False,
    is_listening: bool = False,
    is_voice_speaking: bool = False,
    has_draft: bool = False,
) -> object | None:
    return service.consider(
        settings,
        now=now,
        is_busy=is_busy,
        is_listening=is_listening,
        is_voice_speaking=is_voice_speaking,
        has_draft=has_draft,
    )


def test_proactive_is_disabled_by_default() -> None:
    assert _consider(ProactiveConversationService(), settings=ProactiveSettings()) is None


@pytest.mark.parametrize("hour", [22, 23, 0, 7])
def test_proactive_respects_quiet_hours(hour: int) -> None:
    assert _consider(ProactiveConversationService(), now=NOW.replace(hour=hour)) is None


@pytest.mark.parametrize(
    ("is_busy", "is_listening", "is_voice_speaking", "has_draft"),
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_proactive_never_competes_with_an_active_conversation(
    is_busy: bool,
    is_listening: bool,
    is_voice_speaking: bool,
    has_draft: bool,
) -> None:
    assert (
        _consider(
            ProactiveConversationService(),
            is_busy=is_busy,
            is_listening=is_listening,
            is_voice_speaking=is_voice_speaking,
            has_draft=has_draft,
        )
        is None
    )


def test_proactive_waits_after_user_activity_and_rate_limits_each_attempt() -> None:
    service = ProactiveConversationService()
    service.note_user_activity(NOW)

    assert _consider(service, now=NOW + timedelta(minutes=29)) is None
    turn = _consider(service, now=NOW + timedelta(minutes=30))

    assert turn is not None
    assert service.state == ProactiveState(
        last_delivered_at=NOW + timedelta(minutes=30),
        last_user_activity_at=NOW,
    )
    assert _consider(service, now=NOW + timedelta(minutes=149)) is None
    assert _consider(service, now=NOW + timedelta(minutes=150)) is not None


def test_disable_discards_the_idle_anchor_and_requires_reenable() -> None:
    service = ProactiveConversationService()
    service.note_user_activity(NOW)
    service.disable()

    assert service.state.last_user_activity_at is None
    assert _consider(service, settings=ProactiveSettings()) is None
