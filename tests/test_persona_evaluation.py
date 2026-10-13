from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.services import (
    PersonaEvaluationConfig,
    evaluate_persona,
    load_evaluation_config,
)
from multi_llm_companion.services.chat import ChatService, ChatServiceError

_SHIPPED_SCENARIO_IDS = {
    "greeting",
    "tired_without_advice",
    "technical_question",
    "playful",
    "correction",
    "continuation",
    "past_open_loop",
    "short_input",
    "question_one",
    "repeated_question_cooldown",
    "question_cooldown",
    "suggestion_one",
    "suggestion_cooldown",
    "motif_one",
    "repeated_motif_cooldown",
    "no_invented_facts",
    "role_not_identity",
    "dependency_boundary",
    "memory_correction_policy",
    "memory_delete_policy",
    "malformed_json_fallback",
    "missing_metadata_fallback",
}


def test_shipped_evaluation_runs_without_api_or_regular_database() -> None:
    root = Path(__file__).resolve().parents[1]
    persona = PersonaConfig.model_validate(
        yaml.safe_load((root / "config/persona.example.yaml").read_text(encoding="utf-8"))
    )
    config = load_evaluation_config(root / "config/persona_evaluation.example.yaml")

    report = evaluate_persona(persona, config.scenarios)

    scenario_ids = [scenario.id for scenario in config.scenarios]
    assert len(scenario_ids) == 22
    assert len(set(scenario_ids)) == 22
    assert set(scenario_ids) == _SHIPPED_SCENARIO_IDS
    assert len(report.results) == 22
    assert report.passed
    assert "実LLMの人格品質を証明するものではありません" in report.limitation
    assert all(result.passed for result in report.results)
    malformed = next(
        result for result in report.results if result.scenario_id == "malformed_json_fallback"
    )
    assert malformed.speech == ""


def test_evaluation_reports_deterministic_expectation_failure(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        "scenarios:\n"
        "  - id: mismatch\n"
        "    user: こんにちは\n"
        "    mock_response: 回答\n"
        "    expected_mode: planning\n",
        encoding="utf-8",
    )
    config = load_evaluation_config(path)
    persona = PersonaConfig(name="旧案内役", system_prompt="短く答える")

    report = evaluate_persona(persona, config.scenarios)

    assert report.passed is False
    assert report.results[0].failures == ("mode: expected=planning, actual=casual",)


def test_same_user_text_uses_each_scenario_response_in_order(tmp_path: Path) -> None:
    path = tmp_path / "same-user.yaml"
    path.write_text(
        "scenarios:\n"
        "  - id: first\n"
        "    conversation: one\n"
        "    user: 同じ質問\n"
        "    mock_response: 一つ目の固定回答\n"
        "    expected_speech: 一つ目の固定回答\n"
        "  - id: second\n"
        "    conversation: one\n"
        "    user: 同じ質問\n"
        "    mock_response: 二つ目の固定回答\n"
        "    expected_speech: 二つ目の固定回答\n"
        "  - id: another-conversation\n"
        "    conversation: two\n"
        "    user: 同じ質問\n"
        "    mock_response: 別会話の固定回答\n"
        "    expected_speech: 別会話の固定回答\n",
        encoding="utf-8",
    )
    config = load_evaluation_config(path)
    persona = PersonaConfig(name="旧案内役", system_prompt="短く答える")

    report = evaluate_persona(persona, config.scenarios)

    assert report.passed
    assert [result.speech for result in report.results] == [
        "一つ目の固定回答",
        "二つ目の固定回答",
        "別会話の固定回答",
    ]


def test_shipped_scenarios_cover_repeated_question_and_motif() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_evaluation_config(root / "config/persona_evaluation.example.yaml")
    scenarios = {scenario.id: scenario for scenario in config.scenarios}

    first_question = scenarios["question_one"]
    repeated_question = scenarios["repeated_question_cooldown"]
    assert first_question.conversation == repeated_question.conversation
    assert first_question.user == repeated_question.user
    assert repeated_question.expected_speech == "同じ質問は重ねず、最初の手順をまとめます。"
    assert "質問直後なので" in repeated_question.prompt_contains

    repeated_motif = scenarios["repeated_motif_cooldown"]
    assert repeated_motif.expected_speech == "今度は、静かな風が通るような気持ちだね。"
    assert "同じ比喩やモチーフを連続して使わない" in repeated_motif.prompt_contains


def test_evaluation_config_rejects_duplicate_scenario_ids_in_japanese() -> None:
    with pytest.raises(ValidationError, match="評価シナリオIDが重複しています: duplicate"):
        PersonaEvaluationConfig.model_validate(
            {
                "scenarios": [
                    {"id": "duplicate", "user": "一つ目", "mock_response": "回答1"},
                    {"id": "duplicate", "user": "二つ目", "mock_response": "回答2"},
                ]
            }
        )


def test_unexpected_chat_error_is_a_failure_even_with_empty_speech_expected(monkeypatch) -> None:
    config = PersonaEvaluationConfig.model_validate(
        {
            "scenarios": [
                {
                    "id": "unexpected-error",
                    "user": "入力",
                    "mock_response": "回答",
                    "expected_speech": "",
                }
            ]
        }
    )

    def fail(_service, _text):
        raise ChatServiceError("予期しないエラー")

    monkeypatch.setattr(ChatService, "send_message", fail)
    report = evaluate_persona(PersonaConfig(name="確認", system_prompt="短く"), config.scenarios)
    assert not report.passed
    assert report.results[0].failures == ("chat error: 予期しないエラー",)


def test_safety_stop_does_not_satisfy_an_expected_spoken_response() -> None:
    config = PersonaEvaluationConfig.model_validate(
        {
            "scenarios": [
                {
                    "id": "unexpected-stop",
                    "user": "入力",
                    "mock_response": '{"speech":"途中"',
                    "expected_speech": "返答が必要",
                }
            ]
        }
    )
    report = evaluate_persona(PersonaConfig(name="確認", system_prompt="短く"), config.scenarios)
    assert not report.passed
    assert any(failure.startswith("chat error:") for failure in report.results[0].failures)
