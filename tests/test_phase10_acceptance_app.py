import httpx
import pytest

from scripts.phase10_acceptance_app import EvaluationBudget


def request(search=False, **changes):
    payload = {"model": "gpt-5.6-luna"}
    if search:
        payload.update(
            max_output_tokens=1024,
            max_tool_calls=1,
            store=False,
            tools=[
                {"type": "web_search", "search_context_size": "low", "external_web_access": True}
            ],
        )
    else:
        payload["max_completion_tokens"] = 1024
    payload.update(changes)
    path = "responses" if search else "chat/completions"
    return httpx.Request("POST", f"https://api.openai.com/v1/{path}", json=payload)


def test_budget_stops_before_ninth_request_and_fourth_search():
    budget = EvaluationBudget()
    for _ in range(3):
        budget.check(request(True))
    with pytest.raises(RuntimeError, match="上限"):
        budget.check(request(True))
    for _ in range(5):
        budget.check(request())
    with pytest.raises(RuntimeError, match="上限"):
        budget.check(request())
    assert budget.summary()["api_requests"] == 8
    assert budget.summary()["reserved_usd"] == 0.4


def test_failed_request_prevents_retry():
    budget = EvaluationBudget()
    budget.check(request())
    budget.failed = True
    with pytest.raises(RuntimeError, match="失敗"):
        budget.check(request())
    assert budget.requests == 1


@pytest.mark.parametrize(
    "unapproved",
    [
        request(model="another-model"),
        request(max_completion_tokens=1025),
        request(True, max_tool_calls=2),
        request(True, store=True),
        request(messages="x" * 40_001),
        httpx.Request("POST", "https://another.example/v1/responses", json={}),
    ],
)
def test_budget_rejects_unapproved_payload_before_counting(unapproved):
    budget = EvaluationBudget()
    with pytest.raises(RuntimeError):
        budget.check(unapproved)
    assert budget.requests == 0
