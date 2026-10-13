"""口パクの時刻ずれを調べるための、安全な一時診断記録。"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_TRACE_ENVIRONMENT_NAME = "MLC_LIP_SYNC_DIAGNOSTIC"
_TRACE_PATH = Path("local") / "diagnostics" / "lip_sync_trace.jsonl"
_TRACE_LOCK = threading.Lock()

type TraceValue = bool | int | float | str | None


def is_lip_sync_diagnostic_enabled() -> bool:
    """明示的に有効化された場合だけ診断記録を出す。"""

    return os.getenv(_TRACE_ENVIRONMENT_NAME, "").strip() == "1"


def write_lip_sync_trace(
    source: str,
    event: str,
    *,
    path: Path | None = None,
    **fields: TraceValue,
) -> None:
    """本文・UUID・認証情報を含めず、時刻と状態だけを記録する。"""

    if not is_lip_sync_diagnostic_enabled():
        return
    record: dict[str, TraceValue] = {
        "monotonic_ms": time.monotonic_ns() // 1_000_000,
        "source": source,
        "event": event,
    }
    record.update({key: value for key, value in fields.items() if value is not None})
    target = path or _TRACE_PATH
    try:
        with _TRACE_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as trace_file:
                trace_file.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    except OSError:
        # 診断記録の失敗で、通常の音声出力を妨げない。
        return
