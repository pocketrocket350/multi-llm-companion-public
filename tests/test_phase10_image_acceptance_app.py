import base64

import httpx
import pytest
from test_phase10_acceptance_app import request

from scripts.phase10_image_acceptance_app import ImageEvaluationBudget


def image_request(**changes):
    data = {
        "model": "gpt-5.6-luna",
        "store": False,
        "max_output_tokens": 1024,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "何色？"},
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(b"\xff\xd8test").decode(),
                    },
                ],
            }
        ],
    }
    data.update(changes)
    return httpx.Request("POST", "https://api.openai.com/v1/responses", json=data)


def test_budget_counts_images_search_and_every_chat_separately():
    budget = ImageEvaluationBudget()
    for _ in range(2):
        assert budget.check(image_request()) == "image"
    with pytest.raises(RuntimeError, match="上限"):
        budget.check(image_request())
    assert budget.check(request(True)) == "search"
    with pytest.raises(RuntimeError, match="上限"):
        budget.check(request(True))
    for _ in range(3):
        assert budget.check(request()) == "chat"
    with pytest.raises(RuntimeError, match="上限"):
        budget.check(request())
    assert budget.summary() == {
        "requests": 6,
        "images": 2,
        "searches": 1,
        "reserved_usd": 0.2,
        "failed": False,
    }


@pytest.mark.parametrize(
    "bad",
    [
        image_request(model="other"),
        image_request(max_output_tokens=1025),
        image_request(store=True),
        image_request(input="history"),
        image_request(tools=[{"type": "code_interpreter"}]),
        httpx.Request("POST", "https://another.example/v1/responses", json={}),
    ],
)
def test_budget_rejects_unapproved_content(bad):
    budget = ImageEvaluationBudget()
    with pytest.raises(RuntimeError):
        budget.check(bad)
    assert budget.requests == 0


def test_failure_blocks_further_requests():
    budget = ImageEvaluationBudget()
    budget.check(image_request())
    budget.failed = True
    with pytest.raises(RuntimeError, match="失敗"):
        budget.check(request())
