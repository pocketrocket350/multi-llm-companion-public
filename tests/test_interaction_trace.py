from __future__ import annotations

from pathlib import Path

import pytest

from multi_llm_companion.integrations.interaction_trace import (
    summarize_cycle_timing,
    summarize_interaction_trace,
    summarize_interaction_trace_statistics,
)
from scripts.analyze_lip_sync_trace import load_trace_records, main


def test_trace_summary_measures_each_stage_from_input() -> None:
    sessions = summarize_interaction_trace(
        [
            {
                "monotonic_ms": 100,
                "event": "input_finished",
                "provider_name": "openai",
                "model_name": "gpt-5.6-luna",
            },
            {"monotonic_ms": 180, "event": "llm_first_chunk"},
            {"monotonic_ms": 240, "event": "llm_sentence_ready"},
            {"monotonic_ms": 260, "event": "voisona_stream_started"},
            {"monotonic_ms": 280, "event": "stream_synthesis_requested", "sentence_index": 0},
            {"monotonic_ms": 640, "event": "stream_synthesis_completed", "sentence_index": 0},
            {"monotonic_ms": 900, "event": "stream_managed_wav_playback_started"},
            {"monotonic_ms": 940, "event": "playback_position_started"},
            {"monotonic_ms": 960, "event": "lip_sync_state_applied", "active": True},
            {"monotonic_ms": 1_400, "event": "stream_managed_wav_playback_completed"},
            {"monotonic_ms": 1_460, "event": "recognition_started"},
        ]
    )

    assert sessions[0]["provider_name"] == "openai"
    assert sessions[0]["model_name"] == "gpt-5.6-luna"
    assert sessions[0]["input_to_llm_first_chunk_ms"] == 80
    assert sessions[0]["input_to_sentence_ready_ms"] == 140
    assert sessions[0]["input_to_voice_request_ms"] == 160
    assert sessions[0]["input_to_synthesis_requested_ms"] == 180
    assert sessions[0]["synthesis_request_to_audio_received_ms"] == 360
    assert sessions[0]["audio_received_to_playback_started_ms"] == 260
    assert sessions[0]["input_to_playback_started_ms"] == 800
    assert sessions[0]["input_to_playback_position_observed_ms"] == 840
    assert sessions[0]["input_to_lip_sync_applied_ms"] == 860
    assert sessions[0]["playback_started_to_lip_sync_applied_ms"] == 60
    assert sessions[0]["input_to_playback_completed_ms"] == 1_300
    assert sessions[0]["playback_completed_to_continuous_recognition_started_ms"] == 60


def test_trace_summary_keeps_sessions_separate_and_ignores_invalid_records() -> None:
    sessions = summarize_interaction_trace(
        [
            {"monotonic_ms": 100, "event": "input_finished"},
            {"monotonic_ms": "invalid", "event": "llm_first_chunk"},
            {"monotonic_ms": 200, "event": "input_finished"},
            {"monotonic_ms": 190, "event": "llm_first_chunk"},
            {"monotonic_ms": 260, "event": "llm_first_chunk"},
        ]
    )

    assert len(sessions) == 2
    assert sessions[0]["input_to_llm_first_chunk_ms"] is None
    assert sessions[0]["input_to_playback_position_observed_ms"] is None
    assert sessions[1]["input_to_llm_first_chunk_ms"] == 60
    assert sessions[1]["input_to_synthesis_requested_ms"] is None


def test_trace_summary_measures_ptt_and_voice_stop_without_content() -> None:
    sessions = summarize_interaction_trace(
        [
            {"monotonic_ms": 100, "event": "ptt_pressed"},
            {"monotonic_ms": 120, "event": "recognition_started"},
            {"monotonic_ms": 280, "event": "recognition_stop_requested"},
            {"monotonic_ms": 300, "event": "speech_end_detected"},
            {"monotonic_ms": 320, "event": "recognition_finished"},
            {"monotonic_ms": 350, "event": "voice_stop_requested"},
            {"monotonic_ms": 390, "event": "voice_stopped"},
            {"monotonic_ms": 395, "event": "playback_stop_completed"},
            {"monotonic_ms": 400, "event": "input_finished"},
        ]
    )

    assert sessions[0]["ptt_to_recognition_started_ms"] == 20
    assert sessions[0]["ptt_to_recognition_finished_ms"] == 220
    assert sessions[0]["speech_end_to_recognition_finished_ms"] == 20
    assert sessions[0]["recognition_stop_to_finished_ms"] == 40
    assert sessions[0]["voice_stop_to_stopped_ms"] == 40
    assert sessions[0]["voice_stop_to_playback_stop_completed_ms"] == 45


def test_trace_statistics_uses_nearest_rank_p95_and_counts_missing_playback() -> None:
    summary = summarize_interaction_trace_statistics(
        [
            {"input_to_llm_first_chunk_ms": value, "input_to_playback_started_ms": value + 100}
            for value in range(100, 2_100, 100)
        ]
    )

    assert summary["turn_count"] == 20
    assert summary["playback_started_count"] == 20
    assert summary["playback_not_observed_count"] == 0
    assert summary["metrics"]["input_to_llm_first_chunk_ms"] == {
        "sample_count": 20,
        "median_ms": 1050,
        "p95_ms": 1900,
        "maximum_ms": 2000,
    }

    incomplete = summarize_interaction_trace_statistics([{"input_to_llm_first_chunk_ms": 100}])

    assert incomplete["playback_not_observed_count"] == 1


def test_cycle_timing_counts_interval_and_callback_overruns() -> None:
    summary = summarize_cycle_timing(
        [
            {
                "event": "ui_voice_sync_tick",
                "interval_ms": 60,
                "callback_duration_ms": 2,
                "overrun_ms": 0,
            },
            {
                "event": "ui_voice_sync_tick",
                "interval_ms": 67,
                "callback_duration_ms": 4,
                "overrun_ms": 7,
            },
            {
                "event": "pmx_heavy_tick",
                "interval_ms": 38,
                "callback_duration_ms": 40,
                "overrun_ms": 5,
            },
        ]
    )

    assert summary["ui_voice_sync_tick"] == {
        "sample_count": 2,
        "interval_median_ms": 64,
        "interval_p95_ms": 67,
        "interval_maximum_ms": 67,
        "duration_p95_ms": 4,
        "duration_maximum_ms": 4,
        "overrun_count": 1,
        "overrun_maximum_ms": 7,
    }
    assert summary["pmx_lightweight_tick"]["sample_count"] == 0
    assert summary["pmx_heavy_tick"]["interval_p95_ms"] == 38
    assert summary["pmx_heavy_tick"]["duration_maximum_ms"] == 40


def test_trace_loader_ignores_malformed_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "lip_sync_trace.jsonl"
    path.write_text(
        '{"monotonic_ms":100,"event":"input_finished"}\nnot-json\n["not a record"]\n',
        encoding="utf-8",
    )

    assert load_trace_records(path) == [{"monotonic_ms": 100, "event": "input_finished"}]


def test_trace_cli_can_write_only_the_latest_session(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "lip_sync_trace.jsonl"
    output = tmp_path / "summary.json"
    trace.write_text(
        "\n".join(
            (
                '{"monotonic_ms":100,"event":"input_finished"}',
                '{"monotonic_ms":200,"event":"input_finished"}',
                '{"monotonic_ms":250,"event":"llm_first_chunk"}',
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_lip_sync_trace.py", "--input", str(trace), "--output", str(output), "--latest"],
    )

    assert main() == 0
    assert '"input_finished_ms": 200' in output.read_text(encoding="utf-8")
    assert '"input_finished_ms": 100' not in output.read_text(encoding="utf-8")


def test_trace_cli_help_supports_percent_sign(capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["analyze_lip_sync_trace.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "95%点" in capsys.readouterr().out


def test_trace_cli_adds_aggregate_summary(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "lip_sync_trace.jsonl"
    output = tmp_path / "summary.json"
    trace.write_text(
        "\n".join(
            (
                '{"monotonic_ms":100,"event":"input_finished"}',
                '{"monotonic_ms":200,"event":"stream_managed_wav_playback_started"}',
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_lip_sync_trace.py",
            "--input",
            str(trace),
            "--output",
            str(output),
            "--aggregate",
        ],
    )

    assert main() == 0
    report = output.read_text(encoding="utf-8")
    assert '"turn_count": 1' in report
    assert '"input_to_playback_started_ms"' in report
    assert '"cycle_timing"' in report
