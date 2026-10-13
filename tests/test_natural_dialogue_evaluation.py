from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.services import (
    NaturalDialogueConversation,
    NaturalDialogueEvaluationConfig,
    evaluate_natural_dialogue,
    load_natural_dialogue_evaluation_config,
)

_SHIPPED_CONVERSATION_IDS = {
    "NAT-01-stock-acknowledgement",
    "NAT-02-plain-language",
    "NAT-03-question-and-suggestion",
    "NAT-04-unearned-agreement",
    "NAT-05-small-opinion",
    "NAT-06-user-autonomy",
    "NAT-07-natural-memory",
    "NAT-08-long-conversation",
}


def test_shipped_natural_dialogue_evaluation_runs_without_api_or_regular_database() -> None:
    root = Path(__file__).resolve().parents[1]
    persona = PersonaConfig.model_validate(
        yaml.safe_load((root / "config/persona.example.yaml").read_text(encoding="utf-8"))
    )
    config = load_natural_dialogue_evaluation_config(
        root / "config/natural_dialogue_evaluation.example.yaml"
    )

    report = evaluate_natural_dialogue(persona, config.conversations)

    assert {conversation.id for conversation in config.conversations} == _SHIPPED_CONVERSATION_IDS
    assert len(config.conversations) == 8
    assert all(10 <= len(conversation.turns) <= 20 for conversation in config.conversations)
    assert all(conversation.evaluation is not None for conversation in config.conversations)
    assert all(
        all(1 <= turn <= len(conversation.turns) for turn in conversation.evaluation.target_turns)
        for conversation in config.conversations
    )
    assert all(
        conversation.evaluation.failure_mode
        and conversation.evaluation.challenge_turns
        and conversation.evaluation.acceptance_criteria
        and conversation.evaluation.failure_criteria
        and conversation.evaluation.exclusions
        for conversation in config.conversations
    )
    assert len(report.results) == 80
    assert report.passed
    assert "実LLMの自然さや人格品質を採点するものではありません" in report.limitation


def test_natural_dialogue_evaluation_rejects_duplicate_conversation_ids() -> None:
    turns = [{"user": f"発話{index}", "mock_response": "固定回答"} for index in range(10)]

    with pytest.raises(ValidationError, match="自然会話評価の会話IDが重複しています: duplicate"):
        NaturalDialogueEvaluationConfig.model_validate(
            {
                "conversations": [
                    {"id": "duplicate", "turns": turns},
                    {"id": "duplicate", "turns": turns},
                    {"id": "third", "turns": turns},
                    {"id": "fourth", "turns": turns},
                    {"id": "fifth", "turns": turns},
                    {"id": "sixth", "turns": turns},
                    {"id": "seventh", "turns": turns},
                    {"id": "eighth", "turns": turns},
                ]
            }
        )


def test_natural_dialogue_evaluation_requires_ten_to_twenty_turns() -> None:
    short_turns = [{"user": f"発話{index}", "mock_response": "固定回答"} for index in range(9)]

    with pytest.raises(ValidationError):
        NaturalDialogueEvaluationConfig.model_validate(
            {"conversations": [{"id": f"case-{index}", "turns": short_turns} for index in range(8)]}
        )


def test_natural_dialogue_evaluation_rejects_incomplete_or_invalid_evaluation_design() -> None:
    turns = [{"user": f"発話{index}", "mock_response": "固定応答"} for index in range(10)]
    incomplete = {
        "id": "case",
        "turns": turns,
        "evaluation": {"failure_mode": "failure"},
    }
    invalid_turn = {
        "id": "case",
        "turns": turns,
        "evaluation": {
            "failure_mode": "failure",
            "evidence_turns": [],
            "challenge_turns": [1],
            "target_turns": [11],
            "acceptance_criteria": ["条件"],
            "failure_criteria": ["失格条件"],
            "exclusions": ["対象外"],
        },
    }

    with pytest.raises(ValidationError, match="challenge_turns"):
        NaturalDialogueConversation.model_validate(incomplete)
    with pytest.raises(ValidationError, match="範囲外"):
        NaturalDialogueConversation.model_validate(invalid_turn)
