"""音声診断JSONLを集計し、会話から実再生までの遅延を表示する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multi_llm_companion.integrations.interaction_trace import (
    summarize_cycle_timing,
    summarize_interaction_trace,
    summarize_interaction_trace_statistics,
)


def load_trace_records(path: Path) -> list[dict[str, Any]]:
    """壊れた行を無視して、安全な診断レコードだけを読む。"""

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="音声診断ログを外部通信なしで集計します。")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("local/diagnostics/lip_sync_trace.jsonl"),
        help="MLC_LIP_SYNC_DIAGNOSTIC=1で作成したJSONLファイル",
    )
    parser.add_argument("--output", type=Path, help="集計JSONの保存先")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="最後に記録された会話だけを出力する",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="全ターンの区間別中央値、95%%点、最大値を追加する",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"[エラー] 診断ログが見つかりません: {args.input}")
        return 1
    records = load_trace_records(args.input)
    sessions = summarize_interaction_trace(records)
    if args.latest:
        sessions = sessions[-1:]
    report: dict[str, object] = {"sessions": sessions}
    if args.aggregate:
        report["summary"] = summarize_interaction_trace_statistics(sessions)
        report["cycle_timing"] = summarize_cycle_timing(records)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
