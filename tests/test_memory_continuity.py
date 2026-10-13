from __future__ import annotations

import json

import httpx
import pytest

from multi_llm_companion.domain import ChatMessage
from multi_llm_companion.services.conversation_context import ConversationContextService
from scripts.evaluate_memory_continuity import MIDDLE, TAIL, ApiBudget, run, scenario


def test_sixty_connected_turns_retain_middle_and_tail_with_125_memories(tmp_path):
    assert run(tmp_path) == 0
    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert result["turns"] == 60
    assert result["initial_memory_count"] == 125
    assert result["candidate_success"] == result["candidate_total"] == 5
    assert result["answer_token_success"] == result["answer_total"] == 12
    assert result["forbidden_reuse_count"] == 0
    restart = json.loads((tmp_path / "restart.json").read_text(encoding="utf-8"))
    assert restart["deleted_absent"]
    assert restart["pending_not_approved"] == "candidate"
    assert "ご飯" in restart["corrected_value"]


def test_middle_recall_keeps_correction_and_boundary_priority():
    context = ConversationContextService(max_recent_messages=2, summary_max_characters=400)
    messages = [
        ChatMessage(role="user", content="初期の普通の話。" * 100),
        ChatMessage(role="user", content=MIDDLE),
        ChatMessage(role="user", content="訂正: 約束の日は木曜です。"),
        ChatMessage(role="user", content="この話を記憶に残さないで。覚えないで。"),
        ChatMessage(role="assistant", content="了解。"),
        ChatMessage(role="user", content="カナさんへのしおりの約束は？"),
    ]
    summary = context.build(messages).summary
    assert "銀色" in summary
    assert "木曜" in summary
    assert "覚えないで" in summary


def test_long_user_message_keeps_relevant_clause_after_160_characters():
    context = ConversationContextService(max_recent_messages=2, summary_max_characters=300)
    messages = [
        ChatMessage(role="user", content="展示の感想だよ。" * 50 + TAIL),
        ChatMessage(role="assistant", content="了解。"),
        ChatMessage(role="user", content="石の展示の合言葉は？"),
    ]
    assert "琥珀灯" in context.build(messages).summary


def test_proactive_context_does_not_use_assistant_as_retrieval_query():
    context = ConversationContextService(max_recent_messages=2, summary_max_characters=100)
    messages = [
        ChatMessage(role="user", content="最初の話。" * 50),
        ChatMessage(role="user", content=MIDDLE),
        ChatMessage(role="user", content="ひと息つこう。"),
        ChatMessage(role="assistant", content="カナさんへのしおりの約束は？"),
    ]
    assert "銀色" not in context.build(messages).summary


def test_api_budget_rejects_unexpected_destination_without_charging(tmp_path):
    budget = ApiBudget(tmp_path)
    request = httpx.Request(
        "POST",
        "https://example.invalid",
        json={"model": "gpt-6-luna", "max_completion_tokens": 1536},
    )
    with pytest.raises(RuntimeError):
        budget.check(request)
    assert budget.requests == 0


def test_api_budget_keeps_spent_requests_across_separate_test_processes(tmp_path):
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-6-luna", "max_completion_tokens": 1536},
    )
    ApiBudget(tmp_path, limit=1).check(request)
    resumed = ApiBudget(tmp_path, limit=1)
    with pytest.raises(RuntimeError):
        resumed.check(request)
    assert resumed.requests == 1


def test_api_budget_failure_blocks_additional_paid_requests(tmp_path):
    (tmp_path / "api-budget.json").write_text(
        json.dumps({"requests": 1, "failed": True}), encoding="utf-8"
    )
    budget = ApiBudget(tmp_path)
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-6-luna", "max_completion_tokens": 1536},
    )
    with pytest.raises(RuntimeError):
        budget.check(request)
    assert budget.requests == 1


def test_api_budget_preserves_a_smaller_preapproved_limit(tmp_path):
    (tmp_path / "api-budget.json").write_text(
        json.dumps({"requests": 1, "max_requests": 1, "failed": False}), encoding="utf-8"
    )
    budget = ApiBudget(tmp_path)
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-6-luna", "max_completion_tokens": 1536},
    )
    with pytest.raises(RuntimeError):
        budget.check(request)
    assert budget.summary()["max_requests"] == 1
    assert budget.summary()["budget_usd"] == 0.02


def test_scenario_is_one_60_turn_conversation_with_distinct_evidence_checks():
    turns = scenario()
    assert len(turns) == 60
    assert sum(bool(turn.get("probe")) for turn in turns) == 12
    assert {turn.get("action") for turn in turns} >= {"delete", "correct", "disable", "enable"}
