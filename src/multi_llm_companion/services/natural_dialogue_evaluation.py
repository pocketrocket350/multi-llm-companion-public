"""外部APIを使わない日本語自然会話評価セットの構造確認。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.domain import ChatMessage, MessageRole
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services.chat import ChatService
from multi_llm_companion.services.dialogue import DialogueDirector
from multi_llm_companion.services.persona import build_persona_prompt
from multi_llm_companion.storage import SQLiteConversationRepository


class NaturalDialogueTurn(BaseModel):
    """公開可能な自然会話評価の一発話。"""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1)
    mock_response: str = Field(min_length=1)
    prompt_contains: list[str] = Field(default_factory=list)
    prompt_not_contains: list[str] = Field(default_factory=list)


class NaturalDialogueEvaluationDesign(BaseModel):
    """シナリオが検証する失敗モードと判定範囲。"""

    model_config = ConfigDict(extra="forbid")

    failure_mode: str = Field(min_length=1)
    evidence_turns: list[int] = Field(default_factory=list)
    challenge_turns: list[int] = Field(min_length=1)
    target_turns: list[int] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    failure_criteria: list[str] = Field(min_length=1)
    exclusions: list[str] = Field(min_length=1)
    evaluation_scope: Literal["target_turn", "session"] = "target_turn"


class NaturalDialogueConversation(BaseModel):
    """同じ会話内で連続実行する評価ケース。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    turns: list[NaturalDialogueTurn] = Field(min_length=10, max_length=20)
    evaluation: NaturalDialogueEvaluationDesign | None = None

    @model_validator(mode="after")
    def validate_evaluation_design(self) -> NaturalDialogueConversation:
        if self.evaluation is None:
            return self
        design = self.evaluation
        referenced_turns = design.evidence_turns + design.challenge_turns + design.target_turns
        invalid_turns = sorted(
            turn for turn in referenced_turns if turn < 1 or turn > len(self.turns)
        )
        if invalid_turns:
            raise ValueError(f"対象ターンが会話の範囲外です: {invalid_turns}")
        for label, turns in (
            ("根拠ターン", design.evidence_turns),
            ("挑戦ターン", design.challenge_turns),
            ("採点対象ターン", design.target_turns),
        ):
            if len(set(turns)) != len(turns):
                raise ValueError(f"{label}は重複なしで指定してください。")
        return self


class NaturalDialogueEvaluationConfig(BaseModel):
    """公開用の日本語自然会話評価セット。"""

    model_config = ConfigDict(extra="forbid")

    conversations: list[NaturalDialogueConversation] = Field(min_length=8)

    @model_validator(mode="after")
    def reject_duplicate_conversation_ids(self) -> NaturalDialogueEvaluationConfig:
        ids = [conversation.id for conversation in self.conversations]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"自然会話評価の会話IDが重複しています: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True, slots=True)
class NaturalDialogueEvaluationResult:
    conversation_id: str
    turn_number: int
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NaturalDialogueEvaluationReport:
    results: tuple[NaturalDialogueEvaluationResult, ...]
    limitation: str = (
        "固定Mock回答による評価セットの構造・会話継続・実行時プロンプトの回帰確認であり、"
        "実LLMの自然さや人格品質を採点するものではありません。"
    )

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


def load_natural_dialogue_evaluation_config(path: Path) -> NaturalDialogueEvaluationConfig:
    """秘密値を参照せず、公開用YAMLだけを読み込む。"""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return NaturalDialogueEvaluationConfig.model_validate(raw)


def evaluate_natural_dialogue(
    persona: PersonaConfig,
    conversations: Sequence[NaturalDialogueConversation],
) -> NaturalDialogueEvaluationReport:
    """in-memory SQLiteとMockProviderだけで各会話を連続実行する。"""

    active_response = ""
    active_prompt = ""

    def responder(messages: Sequence[ChatMessage]) -> str:
        nonlocal active_prompt
        active_prompt = next(
            (item.content for item in messages if item.role is MessageRole.SYSTEM),
            "",
        )
        return active_response

    repository = SQLiteConversationRepository(":memory:")
    service = ChatService(
        provider=MockProvider(responder=responder, chunk_size=3),
        repository=repository,
        system_prompt=build_persona_prompt(persona),
        engine_contract="ENGINE CONTRACT",
        dialogue_director=DialogueDirector(persona),
    )
    results: list[NaturalDialogueEvaluationResult] = []
    try:
        for conversation in conversations:
            service.start_new_conversation(conversation.id)
            for turn_number, turn in enumerate(conversation.turns, start=1):
                active_response = turn.mock_response
                active_prompt = ""
                service.send_message(turn.user)
                failures = [
                    f"prompt missing: {expected}"
                    for expected in turn.prompt_contains
                    if expected not in active_prompt
                ]
                failures.extend(
                    f"prompt unexpectedly contains: {forbidden}"
                    for forbidden in turn.prompt_not_contains
                    if forbidden in active_prompt
                )
                results.append(
                    NaturalDialogueEvaluationResult(
                        conversation_id=conversation.id,
                        turn_number=turn_number,
                        passed=not failures,
                        failures=tuple(failures),
                    )
                )
    finally:
        service.close()
    return NaturalDialogueEvaluationReport(tuple(results))


__all__ = [
    "NaturalDialogueConversation",
    "NaturalDialogueEvaluationDesign",
    "NaturalDialogueEvaluationConfig",
    "NaturalDialogueEvaluationReport",
    "NaturalDialogueEvaluationResult",
    "NaturalDialogueTurn",
    "evaluate_natural_dialogue",
    "load_natural_dialogue_evaluation_config",
]
