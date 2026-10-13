"""公開NAT会話を実LLMとJudgeで条件適合として評価するスクリプト。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from multi_llm_companion.application import _engine_contract
from multi_llm_companion.config import (
    MemoryConfig,
    PersonaConfig,
    ResolvedProviderConfig,
    RuntimeConfig,
    load_runtime_config,
)
from multi_llm_companion.domain import ChatMessage, MemoryCategory, MessageRole
from multi_llm_companion.providers import build_provider
from multi_llm_companion.services.chat import ChatService, ChatServiceError
from multi_llm_companion.services.conversation_context import ConversationContextService
from multi_llm_companion.services.dialogue import DialogueDirector
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.services.persona import build_persona_prompt
from multi_llm_companion.storage import SQLiteConversationRepository

_CONDITION_FEATURES: dict[str, frozenset[str] | None] = {
    "baseline": None,
    "performance": None,
    "motif": frozenset({"motif"}),
    "small_opinion": frozenset({"small_opinion"}),
    "unearned_agreement": frozenset({"unearned_agreement"}),
}


class BaselineDialogueDirector(DialogueDirector):
    """会話演技プラン追加前と同じ実行時指示で比較する。"""

    def build_prompt(self, user_text: str, messages: list[Any]) -> str:
        return self.build_runtime_prompt(user_text, self.rebuild_state(messages))


@dataclass(frozen=True, slots=True)
class ResponseJudgment:
    """合格条件と失格条件に対する一つの返答の判定。"""

    verdict: str
    satisfied_criteria: tuple[str, ...]
    violations: tuple[tuple[str, str], ...]
    reason: str


def run_conversation(
    runtime: RuntimeConfig,
    persona: PersonaConfig,
    conversation: dict[str, Any],
    director_type: type[DialogueDirector],
    *,
    provider_config: ResolvedProviderConfig,
    performance_features: frozenset[str] | None = None,
    turn_limit: int | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> list[str]:
    """1会話だけを通常DBと音声を使わずに実行する。"""

    provider = build_provider(provider_config)
    repository = SQLiteConversationRepository(":memory:")
    memory_config = MemoryConfig.model_validate(conversation.get("memory_config", {}))
    memories = (
        MemoryService(repository, memory_config) if conversation.get("memory_config") else None
    )
    sent_prompt = ""
    original_chat = provider.chat

    def capture_chat(messages, *, system_prompt=None):
        nonlocal sent_prompt
        sent_prompt = system_prompt or ""
        return original_chat(messages, system_prompt=system_prompt)

    if trace is not None:
        provider.chat = capture_chat
    service = ChatService(
        provider=provider,
        repository=repository,
        system_prompt=build_persona_prompt(persona),
        engine_contract=_engine_contract(runtime),
        dialogue_director=director_type(persona, performance_features=performance_features),
        memory_service=memories,
        conversation_context=ConversationContextService(
            max_recent_messages=memory_config.max_recent_messages,
            summary_max_characters=memory_config.summary_max_characters,
        ),
    )
    try:
        service.start_new_conversation(str(conversation["id"]))
        turns = conversation["turns"] if turn_limit is None else conversation["turns"][:turn_limit]
        speeches: list[str] = []
        for number, turn in enumerate(turns, 1):
            action_errors = []
            for action in turn.get("actions_before", []):
                try:
                    _apply_memory_evaluation_action(service, action)
                except (ValueError, LookupError) as error:
                    action_errors.append(str(error))
            failure = ""
            try:
                speech = service.send_message(str(turn["user"])).speech
            except ChatServiceError as error:
                if trace is None:
                    raise
                failure = str(error)
                speech = ""
            speeches.append(speech)
            if trace is not None:
                usage = provider.get_usage()
                trace.append(
                    {
                        "turn": number,
                        "user": turn["user"],
                        "speech": speech,
                        "raw_response": provider.last_raw_response,
                        "system_prompt": sent_prompt,
                        "context_status": service.context_status,
                        "usage": usage.model_dump() if usage else None,
                        "error": failure,
                        "action_errors": action_errors,
                        "used_memories": [item.content for item in memories.last_context_memories]
                        if memories
                        else [],
                        "memories": [
                            {
                                "content": item.content,
                                "category": item.category.value,
                                "status": item.status.value,
                            }
                            for item in memories.list_memories()
                        ]
                        if memories
                        else [],
                    }
                )
        return speeches
    finally:
        service.close()


def _apply_memory_evaluation_action(service: ChatService, action: dict[str, Any]) -> None:
    """隔離DB内だけで、利用者画面と同じ記憶操作を行う。"""

    if action["type"] == "new_conversation":
        service.start_new_conversation("隔離評価")
        return
    memories = service.memory_service
    if memories is None:
        raise ValueError("記憶操作には隔離MemoryServiceが必要です。")
    if action["type"] == "enabled":
        memories.set_enabled(bool(action["value"]))
        return
    if action["type"] == "auto_approve":
        memories.set_auto_approve_candidates(bool(action["value"]))
        return
    matches = [item for item in memories.list_memories() if action["match"] in item.content]
    if len(matches) != 1:
        raise LookupError("検証対象の記憶を一意に特定できません。")
    item = matches[0]
    if action["type"] == "delete":
        memories.delete(item.id)
    elif action["type"] in {"approve", "edit"}:
        method = memories.approve if action["type"] == "approve" else memories.update
        method(
            item.id,
            action.get("content", item.content),
            MemoryCategory(action.get("category", item.category)),
        )
    else:
        raise ValueError("未対応の記憶評価操作です。")


def _generation_turn_limit(conversation: dict[str, Any]) -> int:
    """採点対象までの会話だけを生成する。"""

    evaluation = conversation["evaluation"]
    if evaluation.get("evaluation_scope", "target_turn") == "session":
        return len(conversation["turns"])
    return max(evaluation["target_turns"])


def _transcript(conversation: dict[str, Any], speeches: list[str]) -> str:
    lines: list[str] = []
    for turn, speech in zip(conversation["turns"], speeches, strict=True):
        lines.extend((f"利用者: {turn['user']}", f"相手: {speech}"))
    return "\n".join(lines)


def _target_material(conversation: dict[str, Any], speeches: list[str], target_turn: int) -> str:
    evaluation = conversation["evaluation"]
    evidence_turns = set(evaluation["evidence_turns"])
    challenge_turns = set(evaluation["challenge_turns"])
    lines = [
        "[合格条件]",
        *[f"- {item}" for item in evaluation["acceptance_criteria"]],
        "[今回の対象外]",
        *[f"- {item}" for item in evaluation["exclusions"]],
        "[失格条件]",
        *[f"- {item}" for item in evaluation["failure_criteria"]],
        "[根拠となる利用者発話]",
    ]
    for turn_number, turn in enumerate(conversation["turns"], start=1):
        if turn_number in evidence_turns or (
            turn_number in challenge_turns and turn_number != target_turn
        ):
            lines.append(f"利用者（{turn_number}）: {turn['user']}")
    target = conversation["turns"][target_turn - 1]
    lines.extend(
        (
            "[採点対象]",
            f"利用者（{target_turn}）: {target['user']}",
            f"相手（{target_turn}）: {speeches[target_turn - 1]}",
        )
    )
    return "\n".join(lines)


def _session_material(conversation: dict[str, Any], speeches: list[str], target_turn: int) -> str:
    """会話全体の根拠と、個別に採点する返答を一緒に渡す。"""

    evaluation = conversation["evaluation"]
    target = conversation["turns"][target_turn - 1]
    return "\n".join(
        (
            "[合格条件]",
            *[f"- {item}" for item in evaluation["acceptance_criteria"]],
            "[今回の対象外]",
            *[f"- {item}" for item in evaluation["exclusions"]],
            "[失格条件]",
            *[f"- {item}" for item in evaluation["failure_criteria"]],
            "[会話全文]",
            _transcript(conversation, speeches),
            "[今回の採点対象]",
            f"利用者（{target_turn}）: {target['user']}",
            f"相手（{target_turn}）: {speeches[target_turn - 1]}",
        )
    )


def _judge_material(conversation: dict[str, Any], speeches: list[str], target_turn: int) -> str:
    if conversation["evaluation"].get("evaluation_scope", "target_turn") == "session":
        return _session_material(conversation, speeches, target_turn)
    return _target_material(conversation, speeches, target_turn)


def parse_response_judgment(
    raw_text: str,
    acceptance_criteria: list[str],
    failure_criteria: list[str],
) -> ResponseJudgment | None:
    """根拠のない合否や、設計外の条件を安全に無効扱いにする。"""

    candidate = raw_text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    for data in _trailing_json_objects(candidate):
        judgment = _validated_response_judgment(data, acceptance_criteria, failure_criteria)
        if judgment is not None:
            return judgment
    return None


def _trailing_json_objects(raw_text: str) -> list[object]:
    """前置きがあっても、末尾の完全なJSONだけを採点候補にする。"""

    values: list[object] = []
    with suppress(json.JSONDecodeError):
        values.append(json.loads(raw_text))
    decoder = json.JSONDecoder()
    for start in range(len(raw_text) - 1, -1, -1):
        if raw_text[start] != "{":
            continue
        try:
            data, end = decoder.raw_decode(raw_text, start)
        except json.JSONDecodeError:
            continue
        if not raw_text[end:].strip():
            values.append(data)
            break
    return values


def _validated_response_judgment(
    data: object,
    acceptance_criteria: list[str],
    failure_criteria: list[str],
) -> ResponseJudgment | None:
    if not isinstance(data, dict) or data.get("verdict") not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        return None
    satisfied = data.get("satisfied_criteria")
    violations = data.get("violations")
    reason = data.get("reason")
    if (
        not isinstance(satisfied, list)
        or not all(isinstance(item, str) and item in acceptance_criteria for item in satisfied)
        or not isinstance(violations, list)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None
    parsed_violations: list[tuple[str, str]] = []
    for violation in violations:
        if not isinstance(violation, dict):
            return None
        criterion = violation.get("criterion")
        evidence = violation.get("evidence")
        if not isinstance(criterion, str) or criterion not in failure_criteria:
            return None
        if not isinstance(evidence, str) or not evidence.strip():
            return None
        parsed_violations.append((criterion, " ".join(evidence.split())[:160]))
    if data["verdict"] == "FAIL" and not parsed_violations:
        return None
    if data["verdict"] == "PASS" and (not satisfied or parsed_violations):
        return None
    return ResponseJudgment(
        verdict=data["verdict"],
        satisfied_criteria=tuple(satisfied),
        violations=tuple(parsed_violations),
        reason=" ".join(reason.split())[:240],
    )


def judge_response(
    provider_config: ResolvedProviderConfig,
    conversation: dict[str, Any],
    run_number: int,
    condition: str,
    speeches: list[str],
    retries: int,
    target_turn: int,
) -> dict[str, object]:
    """一つの返答を、合格・失格条件と根拠文で判定する。"""

    evaluation = conversation["evaluation"]
    prompt = f"""以下の日本語のAI会話相手の返答を、指定された合格条件と失格条件だけで判定してください。対象外の項目や、文体の好みで採点してはいけません。失格なら、返答内の短い引用を必ず示してください。根拠が足りなければINCONCLUSIVEにしてください。\n\n次のJSONオブジェクトだけを返してください。Markdownは禁止です。\n{{\"verdict\":\"PASS または FAIL または INCONCLUSIVE\",\"satisfied_criteria\":[\"合格条件から選ぶ\"],\"violations\":[{{\"criterion\":\"失格条件から選ぶ\",\"evidence\":\"返答からの短い引用\"}}],\"reason\":\"40文字以内の日本語\"}}\n\n{_judge_material(conversation, speeches, target_turn)}\n"""
    provider = build_provider(provider_config)
    for attempt in range(1, retries + 2):
        raw_text = provider.chat([ChatMessage(role=MessageRole.USER, content=prompt)]).speech
        judgment = parse_response_judgment(
            raw_text,
            evaluation["acceptance_criteria"],
            evaluation["failure_criteria"],
        )
        if judgment is not None and all(
            evidence in " ".join(speeches[target_turn - 1].split())
            for _criterion, evidence in judgment.violations
        ):
            return {
                "conversation_id": conversation["id"],
                "run": run_number,
                "condition": condition,
                "target_turn": target_turn,
                "attempts": attempt,
                "verdict": judgment.verdict.lower(),
                "satisfied_criteria": list(judgment.satisfied_criteria),
                "violations": [
                    {"criterion": criterion, "evidence": evidence}
                    for criterion, evidence in judgment.violations
                ],
                "reason": judgment.reason,
            }
    return {
        "conversation_id": conversation["id"],
        "run": run_number,
        "condition": condition,
        "target_turn": target_turn,
        "attempts": retries + 1,
        "verdict": "unparsed",
        "satisfied_criteria": [],
        "violations": [],
        "reason": "Judgeが指定JSONを返しませんでした。",
    }


def _select_conversations(raw_config: dict[str, Any], prefixes: list[str]) -> list[dict[str, Any]]:
    selected = [
        item
        for item in raw_config["conversations"]
        if any(str(item["id"]).startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise ValueError("指定されたNATシナリオが見つかりません。")
    return selected


def _select_target_turns(
    conversations: list[dict[str, Any]], requested_turns: list[int] | None
) -> dict[str, list[int]]:
    """指定があれば、設計済みの採点対象ターンだけに絞る。"""

    selected: dict[str, list[int]] = {}
    for conversation in conversations:
        available = list(conversation["evaluation"]["target_turns"])
        if requested_turns is None:
            selected[str(conversation["id"])] = available
            continue
        unknown = sorted(set(requested_turns) - set(available))
        if unknown:
            raise ValueError(
                f"{conversation['id']} では採点対象ではないターンです: {', '.join(map(str, unknown))}"
            )
        selected[str(conversation["id"])] = list(requested_turns)
    return selected


def _load_persona(path: Path) -> PersonaConfig:
    return PersonaConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="公開NAT会話を実LLMとJudgeで条件適合として評価します。"
    )
    parser.add_argument("--persona", type=Path, default=Path("local/persona.yaml"))
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("config/natural_dialogue_evaluation.example.yaml"),
    )
    parser.add_argument("--scenario-prefix", action="append", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--target-turn", action="append", type=int)
    parser.add_argument("--judge-retries", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--condition",
        action="append",
        choices=tuple(_CONDITION_FEATURES),
        help="評価する条件。省略時は baseline と performance。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="この評価だけに適用する生成・Judge の温度。通常設定は変更しない。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/diagnostics/natural_dialogue_live_evaluation.json"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.judge_retries < 0 or args.max_workers < 1:
        raise ValueError("runs、judge-retries、max-workersは指定範囲を確認してください。")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        raise ValueError("temperature は 0 から 2 の範囲で指定してください。")
    conditions = args.condition or ["baseline", "performance"]
    if len(set(conditions)) != len(conditions):
        raise ValueError("condition は重複なしで指定してください。")

    root = Path.cwd()
    runtime = load_runtime_config(root)
    provider_config = (
        replace(runtime.active_provider, temperature=args.temperature)
        if args.temperature is not None
        else runtime.active_provider
    )
    persona = _load_persona(args.persona)
    raw_config = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    conversations = _select_conversations(raw_config, args.scenario_prefix)
    selected_target_turns = _select_target_turns(conversations, args.target_turn)
    generation_turn_limits = {
        str(conversation["id"]): _generation_turn_limit(conversation)
        for conversation in conversations
    }
    generated: dict[tuple[str, int, str], list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        jobs = [
            (
                kind,
                run_number,
                conversation,
                executor.submit(
                    run_conversation,
                    runtime,
                    persona,
                    conversation,
                    director_type,
                    provider_config=provider_config,
                    performance_features=features,
                    turn_limit=generation_turn_limits[str(conversation["id"])],
                ),
            )
            for run_number in range(1, args.runs + 1)
            for conversation in conversations
            for kind, director_type, features in (
                (kind, BaselineDialogueDirector, None)
                if kind == "baseline"
                else (kind, DialogueDirector, _CONDITION_FEATURES[kind])
                for kind in conditions
            )
        ]
        for kind, run_number, conversation, future in jobs:
            generated[(kind, run_number, str(conversation["id"]))] = future.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, args.max_workers)) as executor:
        judge_jobs = [
            executor.submit(
                judge_response,
                provider_config,
                conversation,
                run_number,
                condition,
                generated[(condition, run_number, str(conversation["id"]))],
                args.judge_retries,
                target_turn,
            )
            for run_number in range(1, args.runs + 1)
            for conversation in conversations
            for condition in conditions
            for target_turn in selected_target_turns[str(conversation["id"])]
        ]
        judgments = [future.result() for future in judge_jobs]

    verdicts = {
        condition: {"pass": 0, "fail": 0, "inconclusive": 0, "unparsed": 0}
        for condition in conditions
    }
    for judgment in judgments:
        verdicts[str(judgment["condition"])][str(judgment["verdict"])] += 1
    report = {
        "model": runtime.active_provider.model,
        "temperature": provider_config.temperature,
        "max_output_tokens": provider_config.max_output_tokens,
        "executed_at": datetime.now(UTC).isoformat(),
        "scenario_ids": [conversation["id"] for conversation in conversations],
        "scenario_evaluations": {
            conversation["id"]: conversation["evaluation"] for conversation in conversations
        },
        "runs": args.runs,
        "judge_retries": args.judge_retries,
        "conditions": conditions,
        "generation_turn_limits": generation_turn_limits,
        "selected_target_turns": selected_target_turns,
        "verdicts": verdicts,
        "judgments": judgments,
        "generated": [
            {
                "conversation_id": conversation["id"],
                "run": run_number,
                "conditions": {
                    condition: generated[(condition, run_number, str(conversation["id"]))]
                    for condition in conditions
                },
            }
            for run_number in range(1, args.runs + 1)
            for conversation in conversations
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"評価結果を保存しました: {args.output}")
    for condition, condition_verdicts in verdicts.items():
        print(
            f"{condition}: pass={condition_verdicts['pass']} "
            f"fail={condition_verdicts['fail']} "
            f"inconclusive={condition_verdicts['inconclusive']} "
            f"unparsed={condition_verdicts['unparsed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
