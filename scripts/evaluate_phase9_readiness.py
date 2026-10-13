"""承認した範囲だけで、隔離した50ターンの修正前後を検証する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from multi_llm_companion.config import PersonaConfig, load_runtime_config
from multi_llm_companion.services.dialogue import DialogueDirector
from scripts import evaluate_natural_dialogue_live as evaluation

_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _ROOT / "local/diagnostics/phase9-readiness-20260922"
_ACCEPT = [
    "今回の入力と期待する反応に沿い、設定された親しみある口調を保つ",
    "求められた長さで返し、質問・助言・名前・同じ励ましを機械的に繰り返さない",
    "利用可能な文脈と承認済み記憶だけを使い、訂正・境界・削除・無効化を尊重する",
]
_FAIL = [
    "入力を取り違える、訂正前の事実や架空設定を現実の事実として使う",
    "根拠のない記憶・日時・実体験・親密さを断言する",
    "質問や提案を断られたのに続ける、話題の終了を無視する",
    "短答依頼に不要な長文、詳説依頼に説明不足、同じ呼びかけや励ましを反復する",
    "JSON断片・内部指示・不必要な英語や重複を会話本文として出す",
]


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Budget:
    """各回100要求・0.50米ドルの枠を事前予約し、再実行を防ぐ。"""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise ValueError("この評価は開始済みです。再実行には追加の承認が必要です。")
        self.path = path
        self.records: list[dict[str, Any]] = []
        self.spent = 0.0
        _write(path, {"records": [], "charged_upper_usd": 0.0})

    def wrap(self, provider: Any, config: Any) -> Any:
        original = provider.chat

        def chat(messages, *, system_prompt=None):
            # UTF-8バイト数に通信上の余裕を加え、入力トークン数を保守的に見積もる。
            text = (system_prompt or "") + "".join(item.content for item in messages)
            input_bound = len(text.encode("utf-8")) + 64 * (len(messages) + 1) + 512
            reservation = (input_bound * 0.20 + config.max_output_tokens * 1.20) / 1_000_000
            if len(self.records) >= 100 or self.spent + reservation > 0.50:
                raise RuntimeError("承認された要求数または予算の上限に達しました。")
            record: dict[str, Any] = {
                "request": len(self.records) + 1,
                "reserved_usd": reservation,
                "status": "started",
            }
            self.records.append(record)
            self.spent += reservation
            self.save()
            try:
                response = original(messages, system_prompt=system_prompt)
                usage = provider.get_usage()
                if usage is not None:
                    charge = (usage.input_tokens * 0.20 + usage.output_tokens * 1.20) / 1_000_000
                    self.spent += charge - reservation
                    record["usage"] = usage.model_dump()
                    record["charged_usd"] = charge
                record.update(status="completed", raw_response=provider.last_raw_response)
                return response
            finally:
                self.save()
                print(
                    f"request {len(self.records)}/100, charged upper ${self.spent:.5f}", flush=True
                )

        provider.chat = chat
        return provider

    def save(self) -> None:
        _write(self.path, {"records": self.records, "charged_upper_usd": self.spent})


def structural_checks(turn: dict[str, Any], trace: dict[str, Any]) -> dict[str, bool]:
    """自然さの採点と別に、実際の保存・利用・出力を検査する。"""

    checks = turn.get("checks", {})
    speech = trace["speech"]
    memory = "\n".join(item["content"] for item in trace["memories"])
    candidates = "\n".join(
        item["content"] for item in trace["memories"] if item["status"] == "candidate"
    )
    used = "\n".join(trace["used_memories"])
    results = {
        "response_available": bool(speech) and not trace["error"],
        "actions_succeeded": not trace["action_errors"],
        "no_structure_leak": not any(
            item in speech for item in ('"speech"', 'speech":', '"memory_candidates"')
        ),
    }
    for key, content, positive in (
        ("speech_contains", speech, True),
        ("speech_absent", speech, False),
        ("memory_contains", memory, True),
        ("memory_absent", memory, False),
        ("candidate_contains", candidates, True),
        ("used_contains", used, True),
    ):
        for value in checks.get(key, []):
            results[f"{key}:{value}"] = (value in content) == positive
    if "max_chars" in checks:
        results["max_chars"] = len(speech) <= checks["max_chars"]
    if "min_chars" in checks:
        results["min_chars"] = len(speech) >= checks["min_chars"]
    if checks.get("no_question"):
        results["no_question_mark"] = "?" not in speech and "？" not in speech
    if checks.get("used_empty"):
        results["used_empty"] = not trace["used_memories"]
    return results


def _judge_case(scenario, speeches, traces, number):
    # 将来ターンを見せず、履歴分割と管理操作を採点者に明示する。
    turns = []
    for turn in scenario["turns"][:number]:
        actions = json.dumps(turn.get("actions_before", []), ensure_ascii=False)
        turns.append({"user": f"[事前の記憶管理操作: {actions}]\n{turn['user']}"})
    target = scenario["turns"][number - 1]
    trace = traces[number - 1]
    expectation = f"対象ターンの期待: {target['expected']}"
    turns[-1]["user"] += (
        f"\n[実際に取得した記憶: {json.dumps(trace['used_memories'], ensure_ascii=False)}]"
        f"\n[管理操作の失敗: {trace['action_errors']}]"
    )
    return {
        "id": scenario["id"],
        "turns": turns,
        "evaluation": {
            "evaluation_scope": "session",
            "acceptance_criteria": [*_ACCEPT, expectation],
            "failure_criteria": _FAIL,
            "exclusions": ["人間本人による自然さの最終受入、実際の音声とアバターの品質"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stage", choices=["before", "after"])
    args = parser.parse_args()
    runtime = load_runtime_config(_ROOT)
    snapshot_path = _OUTPUT / "fixed-input.json"
    if args.prepare:
        if snapshot_path.exists():
            raise ValueError("固定条件はすでに保存されています。上書きしません。")
        scenario = yaml.safe_load((_OUTPUT / "scenario.yaml").read_text(encoding="utf-8"))
        assert len(scenario["turns"]) == 50
        _write(
            snapshot_path,
            {
                "scenario": scenario,
                "persona": runtime.persona.model_dump(mode="json"),
                "model": "gpt-5.6-luna",
                "temperature": 0.0,
                "max_output_tokens": 1024,
                "initial_memory": [],
                "initial_conversation": [],
                "price_source": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
                "rates_per_million": {"input_usd": 0.20, "output_usd": 1.20},
            },
        )
        print("固定した50ターン・人格・空の初期記憶を保存。API呼び出しは0件。")
        return 0
    if not args.stage:
        parser.error("--prepare または --stage が必要です")
    fixed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    active = runtime.active_provider
    if active.model != fixed["model"] or active.base_url.rstrip("/") != "https://api.openai.com/v1":
        raise ValueError("承認されたモデル・公式接続先と一致しません。")
    config = replace(active, temperature=fixed["temperature"], max_output_tokens=1024)
    budget = Budget(_OUTPUT / f"{args.stage}-requests.json")
    factory = evaluation.build_provider
    evaluation.build_provider = lambda selected: budget.wrap(factory(selected), selected)
    traces: list[dict[str, Any]] = []
    result = {
        "stage": args.stage,
        "started_at": datetime.now(UTC).isoformat(),
        "fixed_input_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "application_source": str(Path(evaluation.ChatService.__module__.replace(".", "/"))),
        "trace": traces,
        "judgments": [],
        "complete": False,
    }
    import inspect

    source_root = Path(inspect.getfile(evaluation.ChatService)).parents[2]
    result["source_root"] = str(source_root)
    result["source_hashes"] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_root.rglob("*.py")
    }
    if args.stage == "before" and source_root != _OUTPUT / "before/src":
        raise ValueError("修正前評価は保存済みの修正前コードを指定してください。")
    scenario = fixed["scenario"]
    result_path = _OUTPUT / f"{args.stage}-result.json"
    try:
        speeches = evaluation.run_conversation(
            runtime,
            PersonaConfig.model_validate(fixed["persona"]),
            scenario,
            DialogueDirector,
            provider_config=config,
            trace=traces,
        )
        for turn, trace in zip(scenario["turns"], traces, strict=True):
            trace["checks"] = structural_checks(turn, trace)
        _write(result_path, result)
        for number in range(1, 51):
            judgment = evaluation.judge_response(
                config,
                _judge_case(scenario, speeches, traces, number),
                1,
                args.stage,
                speeches[:number],
                0,
                number,
            )
            judgment["raw_judge"] = budget.records[-1].get("raw_response", "")
            result["judgments"].append(judgment)
            _write(result_path, result)
        result["complete"] = True
    finally:
        result["finished_at"] = datetime.now(UTC).isoformat()
        result["requests"] = len(budget.records)
        result["charged_upper_usd"] = budget.spent
        _write(result_path, result)
    print(f"{args.stage}: 50返答＋50採点を保存。利用者受入ではありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
