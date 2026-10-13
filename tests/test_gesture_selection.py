from __future__ import annotations

from dataclasses import dataclass

import pytest

from multi_llm_companion.integrations.gesture_selection import (
    GestureCandidate,
    GestureSelector,
)


@dataclass
class FakeClock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_selector_suppresses_same_gesture_until_cooldown_ends() -> None:
    clock = FakeClock()
    selector = GestureSelector(
        (GestureCandidate("registered_wave", "wave", "arms", ("arms",)),),
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=20.0,
        same_category_cooldown_seconds=0.0,
        clock=clock,
    )

    first = selector.select("wave")
    clock.advance(19.9)
    blocked = selector.select("wave")
    clock.advance(0.1)
    allowed = selector.select("wave")

    assert first.gesture == "wave"
    assert blocked.gesture == "none"
    assert blocked.reason == "同じ動作の連続を抑制中"
    assert allowed.gesture == "wave"


def test_selector_applies_global_and_category_intervals() -> None:
    clock = FakeClock()
    selector = GestureSelector(
        (
            GestureCandidate("nod_a", "small_nod", "upper", ("head",)),
            GestureCandidate("wave_a", "wave", "upper", ("hand",)),
        ),
        min_interval_seconds=3.5,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=6.0,
        clock=clock,
    )

    assert selector.select("nod").candidate is not None
    clock.advance(3.4)
    assert selector.select("wave").reason == "動作間隔を確保中"
    clock.advance(0.1)
    assert selector.select("wave").reason == "同じ動作の連続を抑制中"
    clock.advance(2.5)
    assert selector.select("wave").gesture == "wave"


def test_selector_keeps_registered_nod_and_its_alias_available() -> None:
    clock = FakeClock()
    candidate = GestureCandidate(
        "licensed_nod",
        "small_nod",
        "head",
        ("head",),
    )
    selector = GestureSelector(
        (candidate,),
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
        clock=clock,
    )

    selection = selector.select("nod")

    assert selection.candidate == candidate


def test_selector_rejects_unknown_intent_without_guessing() -> None:
    selector = GestureSelector()

    selection = selector.select("dance_unknown")

    assert selection.gesture == "none"
    assert selection.reason == "未対応の動作: dance_unknown"


def test_selector_limits_large_gestures_per_minute() -> None:
    clock = FakeClock()
    selector = GestureSelector(
        (GestureCandidate("registered_wave", "wave", "arms", ("arms",)),),
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
        max_large_per_minute=1,
        clock=clock,
    )

    assert selector.select("wave").gesture == "wave"
    clock.advance(59.9)
    assert selector.select("wave").gesture == "none"
    clock.advance(0.1)
    assert selector.select("wave").gesture == "wave"


def test_disabled_policy_keeps_supported_gesture_available() -> None:
    selector = GestureSelector(
        (GestureCandidate("registered_wave", "wave", "arms", ("arms",)),), enabled=False
    )

    assert selector.select("wave").gesture == "wave"
    assert selector.select("wave").gesture == "wave"


def test_selector_excludes_the_five_most_recent_motions_when_possible() -> None:
    candidates = tuple(
        GestureCandidate(
            f"reaction_{index}",
            "react",
            f"category_{index}",
        )
        for index in range(6)
    )
    selector = GestureSelector(
        candidates,
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
        max_large_per_minute=0,
        recent_motion_count=5,
        random_source=lambda: 0.0,
    )

    selected_ids = [selector.select("react").candidate.gesture_id for _ in range(7)]

    assert selected_ids == [
        "reaction_0",
        "reaction_1",
        "reaction_2",
        "reaction_3",
        "reaction_4",
        "reaction_5",
        "reaction_0",
    ]


def test_selector_uses_mood_energy_weight_and_audio_availability() -> None:
    candidates = (
        GestureCandidate(
            "happy_low",
            "react",
            "face",
            moods=("happy",),
            energy="low",
            weight=1.0,
        ),
        GestureCandidate(
            "happy_high",
            "react",
            "arms",
            moods=("happy",),
            energy="high",
            weight=9.0,
        ),
        GestureCandidate(
            "happy_high_audio",
            "react",
            "full_body",
            moods=("happy",),
            energy="high",
            weight=100.0,
            audio_required=True,
        ),
    )
    selector = GestureSelector(
        candidates,
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
        max_large_per_minute=0,
        recent_motion_count=0,
        random_source=lambda: 0.95,
    )

    selection = selector.select("react", mood="happy", energy=0.9)

    assert selection.candidate is not None
    assert selection.candidate.gesture_id == "happy_high"


def test_selector_handles_three_hundred_metadata_entries() -> None:
    selector = GestureSelector(
        tuple(
            GestureCandidate(
                f"motion_{index}",
                "react",
                f"category_{index}",
                weight=float((index % 5) + 1),
            )
            for index in range(300)
        ),
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
        max_large_per_minute=0,
        random_source=lambda: 0.5,
    )

    selection = selector.select("react")

    assert selection.candidate is not None
    assert selection.candidate.gesture_id.startswith("motion_")


def test_unavailable_registered_motion_does_not_fall_back_to_builtin() -> None:
    selector = GestureSelector(
        (
            GestureCandidate(
                "licensed_wave_with_audio",
                "wave",
                "arms",
                audio_required=True,
            ),
        ),
        min_interval_seconds=0.0,
        same_gesture_cooldown_seconds=0.0,
        same_category_cooldown_seconds=0.0,
    )

    selection = selector.select("wave", audio_available=False)

    assert selection.gesture == "none"
    assert selection.candidate is None

    available = selector.select("wave", audio_available=True)
    assert available.candidate is not None
    assert available.candidate.gesture_id == "licensed_wave_with_audio"


@pytest.mark.parametrize("gesture", ["wave", "nod", "small_nod"])
@pytest.mark.parametrize("policy_enabled", [True, False])
def test_unregistered_wave_and_nod_are_not_implicitly_added(
    gesture: str, policy_enabled: bool
) -> None:
    selector = GestureSelector(enabled=policy_enabled)

    for _ in range(2):
        selection = selector.select(gesture)
        assert selection.gesture == "none"
        assert selection.candidate is None
        selector.reset()
