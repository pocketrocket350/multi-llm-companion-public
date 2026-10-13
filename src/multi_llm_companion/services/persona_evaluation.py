"""外部APIを使わない、決定的な会話運営回帰評価。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.domain import ChatMessage, MessageRole
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services.chat import ChatService, ChatServiceError
from multi_llm_companion.services.dialogue import DialogueDirector
from multi_llm_companion.services.persona import build_persona_prompt
from multi_llm_companion.storage import SQLiteConversationRepository


class PersonaEvaluationScenario(BaseModel):
    """1ターン分の固定入力・固定回答・決定的期待値。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    conversation: str = "default"
    user: str = Field(min_length=1)
    mock_response: str
    expected_mode: str | None = None
    expected_speech: str | None = None
    prompt_contains: list[str] = Field(default_factory=list)
    prompt_not_contains: list[str] = Field(default_factory=list)


class PersonaEvaluationConfig(BaseModel):
    """公開可能な固定シナリオ一覧。"""

    model_config = ConfigDict(extra="forbid")

    scenarios: list[PersonaEvaluationScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_scenario_ids(self) -> PersonaEvaluationConfig:
        ids = [scenario.id for scenario in self.scenarios]
        duplicates = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
        if duplicates:
            raise ValueError(f"評価シナリオIDが重複しています: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True, slots=True)
class PersonaEvaluationResult:
    scenario_id: str
    passed: bool
    failures: tuple[str, ...]
    selected_mode: str
    speech: str


@dataclass(frozen=True, slots=True)
class PersonaEvaluationReport:
    results: tuple[PersonaEvaluationResult, ...]
    limitation: str = (
        "固定Mock回答によるdirector・prompt順序・保存復元・安全規則・speech復旧と安全停止の"
        "同条件比較であり、実LLMの人格品質を証明するものではありません。"
    )

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "limitation": self.limitation,
            "results": [
                {
                    "scenario_id": result.scenario_id,
                    "passed": result.passed,
                    "failures": list(result.failures),
                    "selected_mode": result.selected_mode,
                    "speech": result.speech,
                }
                for result in self.results
            ],
        }


def load_evaluation_config(path: Path) -> PersonaEvaluationConfig:
    """秘密値を参照せず、評価用YAMLだけを読み込む。"""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PersonaEvaluationConfig.model_validate(raw)


def evaluate_persona(
    persona: PersonaConfig,
    scenarios: Sequence[PersonaEvaluationScenario],
) -> PersonaEvaluationReport:
    """in-memory SQLiteとMockProviderだけで全シナリオを実行する。"""

    active_response = ""
    active_prompt = ""

    def responder(messages: Sequence[ChatMessage]) -> str:
        nonlocal active_prompt
        system = next((item.content for item in messages if item.role is MessageRole.SYSTEM), "")
        active_prompt = system
        return active_response

    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(responder=responder, chunk_size=3)
    director = DialogueDirector(persona)
    service = ChatService(
        provider=provider,
        repository=repository,
        system_prompt=build_persona_prompt(persona),
        engine_contract="ENGINE CONTRACT",
        dialogue_director=director,
    )
    conversation_ids: dict[str, str] = {}
    results: list[PersonaEvaluationResult] = []
    try:
        for scenario in scenarios:
            conversation_id = conversation_ids.get(scenario.conversation)
            if conversation_id is None:
                conversation_id = service.start_new_conversation(scenario.conversation).id
                conversation_ids[scenario.conversation] = conversation_id
            elif service.current_conversation_id != conversation_id:
                service.load_conversation(conversation_id)

            active_response = scenario.mock_response
            active_prompt = ""
            selected_mode = director.select_mode(scenario.user).value
            failures: list[str] = []
            previous_count = len(service.get_current_messages())
            try:
                speech = service.send_message(scenario.user).speech
            except ChatServiceError as error:
                speech = ""
                # 空の期待値は、指定Mock応答を受信した後の安全停止だけに使う。
                if scenario.expected_speech != "" or provider.last_raw_response != active_response:
                    failures.append(f"chat error: {error}")
                if any(
                    item.role is MessageRole.ASSISTANT
                    for item in service.get_current_messages()[previous_count:]
                ):
                    failures.append("安全停止した応答が履歴に保存されています")
            prompt = active_prompt
            if scenario.expected_mode is not None and selected_mode != scenario.expected_mode:
                failures.append(f"mode: expected={scenario.expected_mode}, actual={selected_mode}")
            if scenario.expected_speech is not None and speech != scenario.expected_speech:
                failures.append(f"speech: expected={scenario.expected_speech!r}, actual={speech!r}")
            for expected in scenario.prompt_contains:
                if expected not in prompt:
                    failures.append(f"prompt missing: {expected}")
            for forbidden in scenario.prompt_not_contains:
                if forbidden in prompt:
                    failures.append(f"prompt unexpectedly contains: {forbidden}")
            results.append(
                PersonaEvaluationResult(
                    scenario_id=scenario.id,
                    passed=not failures,
                    failures=tuple(failures),
                    selected_mode=selected_mode,
                    speech=speech,
                )
            )
    finally:
        service.close()
    return PersonaEvaluationReport(tuple(results))


__all__ = [
    "PersonaEvaluationConfig",
    "PersonaEvaluationReport",
    "PersonaEvaluationResult",
    "PersonaEvaluationScenario",
    "evaluate_persona",
    "load_evaluation_config",
]
