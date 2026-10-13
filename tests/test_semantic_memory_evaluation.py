import httpx
import pytest

from scripts.evaluate_semantic_memory import SemanticEvaluationBudget


def request(**changes):
    data = {
        "model": "gpt-6-luna",
        "stream": False,
        "max_completion_tokens": 1536,
        "messages": [{"role": "user", "content": "検査用"}],
    }
    data.update(changes)
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions", json=data)


def test_budget_limits_every_request_and_stops_after_api_failure():
    budget = SemanticEvaluationBudget()
    for _ in range(16):
        budget.check(request())
    assert budget.summary()["reserved_usd"] <= 0.50
    with pytest.raises(RuntimeError):
        budget.check(request())
    budget = SemanticEvaluationBudget()
    budget.failed = True
    with pytest.raises(RuntimeError):
        budget.check(request())
    assert budget.requests == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "gpt-6-sol"},
        {"max_completion_tokens": 1537},
        {"max_completion_tokens": 0},
        {"stream": True},
        {"tools": []},
        {"service_tier": "priority"},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"messages": [{"role": "user", "content": "x" * 80_001}]},
    ],
)
def test_invalid_payload_never_uses_budget(changes):
    budget = SemanticEvaluationBudget()
    with pytest.raises(RuntimeError):
        budget.check(request(**changes))
    assert budget.requests == 0


def test_other_endpoints_are_not_allowed():
    budget = SemanticEvaluationBudget()
    for url in [
        "http://api.openai.com/v1/chat/completions",
        "https://example.com/v1/chat/completions",
        "https://api.openai.com/v1/responses",
    ]:
        with pytest.raises(RuntimeError):
            budget.check(httpx.Request("POST", url, content=request().content))


def test_prior_attempts_cannot_reset_approval_budget():
    budget = SemanticEvaluationBudget(15)
    budget.check(request())
    with pytest.raises(RuntimeError):
        budget.check(request())
    assert budget.requests == 16


def test_streaming_app_request_has_same_model_output_and_request_limit():
    budget = SemanticEvaluationBudget(15)
    budget.check(request(stream=True, stream_options={"include_usage": True}))
    with pytest.raises(RuntimeError):
        budget.check(request(stream=True, stream_options={"include_usage": True}))


def test_smaller_evaluation_scope_cannot_send_a_third_request():
    budget = SemanticEvaluationBudget(request_limit=2)
    budget.check(request())
    budget.check(request())
    with pytest.raises(RuntimeError):
        budget.check(request())
    assert budget.summary()["reserved_usd"] < 0.10
