"""会話から実再生までの診断記録を、外部通信なしで集計する。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

_LATENCY_FIELD_NAMES = (
    "ptt_to_recognition_started_ms",
    "ptt_to_recognition_finished_ms",
    "speech_end_to_recognition_finished_ms",
    "recognition_stop_to_finished_ms",
    "voice_stop_to_stopped_ms",
    "voice_stop_to_playback_stop_completed_ms",
    "input_to_llm_first_chunk_ms",
    "input_to_sentence_ready_ms",
    "input_to_voice_request_ms",
    "input_to_synthesis_requested_ms",
    "synthesis_request_to_audio_received_ms",
    "audio_received_to_playback_started_ms",
    "input_to_playback_started_ms",
    "input_to_playback_position_observed_ms",
    "input_to_lip_sync_applied_ms",
    "playback_started_to_lip_sync_applied_ms",
    "input_to_playback_completed_ms",
    "playback_completed_to_continuous_recognition_started_ms",
)


@dataclass(frozen=True, slots=True)
class InteractionTraceSession:
    """一回の入力から後続イベントまでの遅延。"""

    input_finished_ms: int
    provider_name: str | None = None
    model_name: str | None = None
    ptt_pressed_ms: int | None = None
    recognition_started_ms: int | None = None
    speech_end_detected_ms: int | None = None
    recognition_stop_requested_ms: int | None = None
    recognition_finished_ms: int | None = None
    voice_stop_requested_ms: int | None = None
    voice_stopped_ms: int | None = None
    playback_stop_completed_ms: int | None = None
    first_llm_chunk_ms: int | None = None
    first_sentence_ready_ms: int | None = None
    voice_request_ms: int | None = None
    synthesis_requested_ms: int | None = None
    synthesis_completed_ms: int | None = None
    playback_started_ms: int | None = None
    playback_position_observed_ms: int | None = None
    lip_sync_applied_ms: int | None = None
    playback_completed_ms: int | None = None
    continuous_recognition_started_ms: int | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "input_finished_ms": self.input_finished_ms,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "ptt_to_recognition_started_ms": _elapsed(
                self.ptt_pressed_ms, self.recognition_started_ms
            ),
            "ptt_to_recognition_finished_ms": _elapsed(
                self.ptt_pressed_ms, self.recognition_finished_ms
            ),
            "speech_end_to_recognition_finished_ms": _elapsed(
                self.speech_end_detected_ms, self.recognition_finished_ms
            ),
            "recognition_stop_to_finished_ms": _elapsed(
                self.recognition_stop_requested_ms, self.recognition_finished_ms
            ),
            "voice_stop_to_stopped_ms": _elapsed(
                self.voice_stop_requested_ms, self.voice_stopped_ms
            ),
            "voice_stop_to_playback_stop_completed_ms": _elapsed(
                self.voice_stop_requested_ms, self.playback_stop_completed_ms
            ),
            "input_to_llm_first_chunk_ms": _elapsed(
                self.input_finished_ms, self.first_llm_chunk_ms
            ),
            "input_to_sentence_ready_ms": _elapsed(
                self.input_finished_ms, self.first_sentence_ready_ms
            ),
            "input_to_voice_request_ms": _elapsed(self.input_finished_ms, self.voice_request_ms),
            "input_to_synthesis_requested_ms": _elapsed(
                self.input_finished_ms, self.synthesis_requested_ms
            ),
            "synthesis_request_to_audio_received_ms": _elapsed(
                self.synthesis_requested_ms, self.synthesis_completed_ms
            ),
            "audio_received_to_playback_started_ms": _elapsed(
                self.synthesis_completed_ms, self.playback_started_ms
            ),
            "input_to_playback_started_ms": _elapsed(
                self.input_finished_ms, self.playback_started_ms
            ),
            "input_to_playback_position_observed_ms": _elapsed(
                self.input_finished_ms, self.playback_position_observed_ms
            ),
            "input_to_lip_sync_applied_ms": _elapsed(
                self.input_finished_ms, self.lip_sync_applied_ms
            ),
            "playback_started_to_lip_sync_applied_ms": _elapsed(
                self.playback_started_ms, self.lip_sync_applied_ms
            ),
            "input_to_playback_completed_ms": _elapsed(
                self.input_finished_ms, self.playback_completed_ms
            ),
            "playback_completed_to_continuous_recognition_started_ms": _elapsed(
                self.playback_completed_ms, self.continuous_recognition_started_ms
            ),
        }


def summarize_interaction_trace(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, int | str | None]]:
    """入力確定ごとに、最初のLLM・音声イベントだけを集計する。"""

    sessions: list[InteractionTraceSession] = []
    current: InteractionTraceSession | None = None
    pending: dict[str, int] = {}
    previous_timestamp: int | None = None
    for record in records:
        timestamp = record.get("monotonic_ms")
        event = record.get("event")
        if not isinstance(timestamp, int) or timestamp < 0 or not isinstance(event, str):
            continue
        if previous_timestamp is not None and timestamp < previous_timestamp:
            continue
        previous_timestamp = timestamp
        if event in {
            "ptt_pressed",
            "recognition_started",
            "speech_end_detected",
            "recognition_stop_requested",
            "recognition_finished",
            "voice_stop_requested",
            "voice_stopped",
            "playback_stop_completed",
        }:
            pending[event] = timestamp
            if (
                event == "recognition_started"
                and current is not None
                and current.playback_completed_ms is not None
                and current.continuous_recognition_started_ms is None
            ):
                current = _replace(current, continuous_recognition_started_ms=timestamp)
            continue
        if event == "input_finished":
            if current is not None:
                sessions.append(current)
            current = InteractionTraceSession(
                input_finished_ms=timestamp,
                provider_name=_text_value(record.get("provider_name")),
                model_name=_text_value(record.get("model_name")),
                ptt_pressed_ms=pending.pop("ptt_pressed", None),
                recognition_started_ms=pending.pop("recognition_started", None),
                speech_end_detected_ms=pending.pop("speech_end_detected", None),
                recognition_stop_requested_ms=pending.pop("recognition_stop_requested", None),
                recognition_finished_ms=pending.pop("recognition_finished", None),
                voice_stop_requested_ms=pending.pop("voice_stop_requested", None),
                voice_stopped_ms=pending.pop("voice_stopped", None),
                playback_stop_completed_ms=pending.pop("playback_stop_completed", None),
            )
            continue
        if current is None:
            continue
        if event == "llm_first_chunk" and current.first_llm_chunk_ms is None:
            current = _replace(current, first_llm_chunk_ms=timestamp)
        elif event == "llm_sentence_ready" and current.first_sentence_ready_ms is None:
            current = _replace(current, first_sentence_ready_ms=timestamp)
        elif event == "voisona_stream_started" and current.voice_request_ms is None:
            current = _replace(current, voice_request_ms=timestamp)
        elif event == "stream_synthesis_requested" and _is_first_sentence(record):
            if current.synthesis_requested_ms is None:
                current = _replace(current, synthesis_requested_ms=timestamp)
        elif event == "stream_synthesis_completed" and _is_first_sentence(record):
            if current.synthesis_completed_ms is None:
                current = _replace(current, synthesis_completed_ms=timestamp)
        elif event in {"stream_managed_wav_playback_started", "managed_wav_playback_started"} and (
            current.playback_started_ms is None
        ):
            current = _replace(current, playback_started_ms=timestamp)
        elif event == "playback_position_started" and (
            current.playback_position_observed_ms is None
        ):
            current = _replace(current, playback_position_observed_ms=timestamp)
        elif event == "lip_sync_state_applied" and record.get("active") is True:
            if current.lip_sync_applied_ms is None:
                current = _replace(current, lip_sync_applied_ms=timestamp)
        elif (
            event in {"stream_managed_wav_playback_completed", "managed_wav_playback_completed"}
            and current.playback_completed_ms is None
        ):
            current = _replace(current, playback_completed_ms=timestamp)
    if current is not None:
        sessions.append(current)
    return [session.as_dict() for session in sessions]


def summarize_interaction_trace_statistics(
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, int | dict[str, dict[str, int | None]]]:
    """複数ターンの区間別中央値・95%点・最大値を安全に集計する。"""

    recorded_sessions = list(sessions)
    metrics: dict[str, dict[str, int | None]] = {}
    for field_name in _LATENCY_FIELD_NAMES:
        values = sorted(
            value
            for session in recorded_sessions
            if isinstance((value := session.get(field_name)), int) and value >= 0
        )
        metrics[field_name] = _latency_statistics(values)
    playback_started_count = sum(
        session.get("input_to_playback_started_ms") is not None for session in recorded_sessions
    )
    playback_position_observed_count = sum(
        session.get("input_to_playback_position_observed_ms") is not None
        for session in recorded_sessions
    )
    return {
        "turn_count": len(recorded_sessions),
        "playback_started_count": playback_started_count,
        "playback_not_observed_count": len(recorded_sessions) - playback_started_count,
        "playback_position_observed_count": playback_position_observed_count,
        "playback_position_not_observed_count": len(recorded_sessions)
        - playback_position_observed_count,
        "metrics": metrics,
    }


def _latency_statistics(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"sample_count": 0, "median_ms": None, "p95_ms": None, "maximum_ms": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else round((values[middle - 1] + values[middle]) / 2)
    return {
        "sample_count": len(values),
        "median_ms": median,
        "p95_ms": values[ceil(len(values) * 0.95) - 1],
        "maximum_ms": values[-1],
    }


def _replace(session: InteractionTraceSession, **changes: int) -> InteractionTraceSession:
    return InteractionTraceSession(
        input_finished_ms=session.input_finished_ms,
        provider_name=session.provider_name,
        model_name=session.model_name,
        ptt_pressed_ms=session.ptt_pressed_ms,
        recognition_started_ms=session.recognition_started_ms,
        speech_end_detected_ms=session.speech_end_detected_ms,
        recognition_stop_requested_ms=session.recognition_stop_requested_ms,
        recognition_finished_ms=session.recognition_finished_ms,
        voice_stop_requested_ms=session.voice_stop_requested_ms,
        voice_stopped_ms=session.voice_stopped_ms,
        playback_stop_completed_ms=session.playback_stop_completed_ms,
        first_llm_chunk_ms=changes.get("first_llm_chunk_ms", session.first_llm_chunk_ms),
        first_sentence_ready_ms=changes.get(
            "first_sentence_ready_ms", session.first_sentence_ready_ms
        ),
        voice_request_ms=changes.get("voice_request_ms", session.voice_request_ms),
        synthesis_requested_ms=changes.get(
            "synthesis_requested_ms", session.synthesis_requested_ms
        ),
        synthesis_completed_ms=changes.get(
            "synthesis_completed_ms", session.synthesis_completed_ms
        ),
        playback_started_ms=changes.get("playback_started_ms", session.playback_started_ms),
        playback_position_observed_ms=changes.get(
            "playback_position_observed_ms", session.playback_position_observed_ms
        ),
        lip_sync_applied_ms=changes.get("lip_sync_applied_ms", session.lip_sync_applied_ms),
        playback_completed_ms=changes.get("playback_completed_ms", session.playback_completed_ms),
        continuous_recognition_started_ms=changes.get(
            "continuous_recognition_started_ms", session.continuous_recognition_started_ms
        ),
    )


def _is_first_sentence(record: Mapping[str, Any]) -> bool:
    return record.get("sentence_index") == 0


def _text_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _elapsed(start: int | None, end: int | None) -> int | None:
    return end - start if start is not None and end is not None else None


def summarize_cycle_timing(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int | None]]:
    """診断中のUI・PMX周期処理を、イベント種別ごとに集計する。"""

    event_names = ("ui_voice_sync_tick", "pmx_lightweight_tick", "pmx_heavy_tick")
    summary: dict[str, dict[str, int | None]] = {}
    for event_name in event_names:
        matching = [record for record in records if record.get("event") == event_name]
        interval_ms = _nonnegative_integer_values(matching, "interval_ms")
        duration_ms = _nonnegative_integer_values(matching, "callback_duration_ms")
        overrun_ms = _nonnegative_integer_values(matching, "overrun_ms")
        summary[event_name] = {
            "sample_count": len(matching),
            "interval_median_ms": _latency_statistics(interval_ms)["median_ms"],
            "interval_p95_ms": _latency_statistics(interval_ms)["p95_ms"],
            "interval_maximum_ms": _latency_statistics(interval_ms)["maximum_ms"],
            "duration_p95_ms": _latency_statistics(duration_ms)["p95_ms"],
            "duration_maximum_ms": _latency_statistics(duration_ms)["maximum_ms"],
            "overrun_count": sum(value > 0 for value in overrun_ms),
            "overrun_maximum_ms": max(overrun_ms, default=0),
        }
    return summary


def _nonnegative_integer_values(records: Iterable[Mapping[str, Any]], field_name: str) -> list[int]:
    return sorted(
        value
        for record in records
        if isinstance((value := record.get(field_name)), int) and value >= 0
    )


__all__ = [
    "InteractionTraceSession",
    "summarize_interaction_trace",
    "summarize_interaction_trace_statistics",
    "summarize_cycle_timing",
]
