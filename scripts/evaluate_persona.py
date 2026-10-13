"""明示実行時だけ、公開シナリオの決定的評価結果を書き出す。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.services import evaluate_persona, load_evaluation_config


def main() -> int:
    parser = argparse.ArgumentParser(description="外部APIなしでペルソナ会話基盤を評価します。")
    parser.add_argument("--persona", type=Path, default=Path("config/persona.example.yaml"))
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("config/persona_evaluation.example.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/diagnostics/persona_evaluation.json"),
    )
    args = parser.parse_args()

    persona = PersonaConfig.model_validate(yaml.safe_load(args.persona.read_text(encoding="utf-8")))
    config = load_evaluation_config(args.scenarios)
    report = evaluate_persona(persona, config.scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"評価結果を保存しました: {args.output}")
    print(f"結果: {'成功' if report.passed else '失敗'} ({len(report.results)}シナリオ)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
