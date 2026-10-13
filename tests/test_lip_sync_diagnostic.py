"""口パク診断記録が通常動作へ影響しないことを確認する。"""

from __future__ import annotations

import json
from pathlib import Path

from multi_llm_companion.integrations.lip_sync_diagnostic import write_lip_sync_trace
from multi_llm_companion.integrations.voisona_talk import VoiSonaTalkVoiceOutput


def test_lip_sync_trace_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MLC_LIP_SYNC_DIAGNOSTIC", raising=False)
    target = tmp_path / "trace.jsonl"

    write_lip_sync_trace("provider", "api_poll", path=target, state="running")

    assert not target.exists()


def test_lip_sync_trace_records_only_given_safe_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLC_LIP_SYNC_DIAGNOSTIC", "1")
    target = tmp_path / "trace.jsonl"

    write_lip_sync_trace("renderer", "mouth_mesh_updated", path=target, mouth="a", duration_ms=4.5)

    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["source"] == "renderer"
    assert record["event"] == "mouth_mesh_updated"
    assert record["mouth"] == "a"
    assert record["duration_ms"] == 4.5
    assert "text" not in record
    assert "uuid" not in record


def test_voisona_diagnostic_summary_hides_unrecognized_response_values() -> None:
    summary = VoiSonaTalkVoiceOutput._diagnostic_response_summary(
        {
            "state": "private-status-value",
            "status": "running",
            "progress_percentage": 42.2,
            "text": "記録してはいけない本文",
        }
    )

    assert summary["state"] == "unrecognized_text"
    assert summary["status"] == "running"
    assert summary["progress_percent"] == 42
    assert "text" not in summary
