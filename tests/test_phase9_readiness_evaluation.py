from types import SimpleNamespace

import pytest

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.domain import ChatMessage, TokenUsage
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services.dialogue import DialogueDirector
from scripts import evaluate_natural_dialogue_live as evaluation
from scripts.evaluate_phase9_readiness import Budget, structural_checks


def test_evaluator_connects_memory_and_actions_without_any_disk_database(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation,
        "build_provider",
        lambda _: MockProvider(responder=lambda _: '{"speech":"回答"}'),
    )
    monkeypatch.setattr(evaluation, "_engine_contract", lambda _: "形式")
    trace = []
    evaluation.run_conversation(
        SimpleNamespace(),
        PersonaConfig(name="テスト", system_prompt="短く"),
        {
            "id": "isolation",
            "memory_config": {"enabled": True},
            "turns": [
                {"user": "私をナツと呼んで"},
                {"user": "私の名前は？", "actions_before": [{"type": "new_conversation"}]},
                {
                    "user": "私の名前は？",
                    "actions_before": [
                        {"type": "enabled", "value": False},
                        {"type": "new_conversation"},
                    ],
                },
                {
                    "user": "私の名前は？",
                    "actions_before": [
                        {"type": "delete", "match": "ナツ"},
                        {"type": "new_conversation"},
                    ],
                },
            ],
        },
        DialogueDirector,
        provider_config=SimpleNamespace(),
        trace=trace,
    )
    assert len(trace) == 4
    assert "ナツ" in trace[1]["used_memories"][0]
    assert trace[2]["used_memories"] == []
    assert trace[3]["memories"] == []
    assert all(not item["action_errors"] for item in trace)


def test_budget_charges_usage_and_refuses_reuse_or_more_than_100_requests(tmp_path) -> None:
    path = tmp_path / "requests.json"
    budget = Budget(path)
    provider = SimpleNamespace(
        chat=lambda messages, **kwargs: "回答",
        get_usage=lambda: TokenUsage(prompt_tokens=100, completion_tokens=10),
        last_raw_response="回答",
    )
    wrapped = budget.wrap(provider, SimpleNamespace(max_output_tokens=1024))
    assert wrapped.chat([ChatMessage(role="user", content="入力")]) == "回答"
    assert budget.spent == pytest.approx(0.000032)
    with pytest.raises(ValueError, match="開始済み"):
        Budget(path)
    budget.records *= 100
    with pytest.raises(RuntimeError, match="上限"):
        wrapped.chat([])


def test_budget_reserves_cost_before_sending_and_does_not_send_on_excess(tmp_path) -> None:
    budget = Budget(tmp_path / "requests.json")
    budget.spent = 0.4999
    calls = []
    provider = SimpleNamespace(chat=lambda *a, **k: calls.append(1))
    budget.wrap(provider, SimpleNamespace(max_output_tokens=1024))
    with pytest.raises(RuntimeError, match="上限"):
        provider.chat([])
    assert calls == []


def test_structural_checks_include_failed_actions_and_memory_not_just_speech() -> None:
    result = structural_checks(
        {"checks": {"memory_absent": ["旧名"], "used_empty": True}},
        {
            "speech": "回答",
            "error": "",
            "action_errors": ["対象不明"],
            "used_memories": ["旧名"],
            "memories": [{"content": "旧名", "status": "approved"}],
        },
    )
    assert result["response_available"]
    assert not result["actions_succeeded"]
    assert not result["memory_absent:旧名"]
    assert not result["used_empty"]
