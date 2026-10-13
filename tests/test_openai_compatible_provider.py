from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest

from multi_llm_companion.domain import ChatMessage
from multi_llm_companion.providers import (
    OpenAICompatibleProvider,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderResponseError,
)
from multi_llm_companion.providers.errors import _EmptyProviderResponse


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("content", ["", None, "   "])
def test_empty_answer_keeps_usage_without_exposing_reasoning(streaming, content):
    usage = {"prompt_tokens": 5, "completion_tokens": 1024, "total_tokens": 1029}

    def handle(_):
        choice = {
            "delta" if streaming else "message": {
                "content": content,
                "reasoning_content": "非公開の内部推論",
            },
            "finish_reason": "length",
        }
        data = {"choices": [choice], "usage": usage}
        return (
            httpx.Response(200, text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n")
            if streaming
            else httpx.Response(200, json=data)
        )

    provider = _provider(httpx.MockTransport(handle))
    with pytest.raises(_EmptyProviderResponse) as error:
        if streaming:
            list(provider.stream_chat([ChatMessage(role="user", content="質問")]))
        else:
            provider.chat([ChatMessage(role="user", content="質問")])
    assert "内部推論" not in str(error.value)
    assert provider.get_usage().total_tokens == 1029
    assert provider.last_response is None


def _provider(
    transport: httpx.BaseTransport, api_key: str = "test-secret"
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="test",
        base_url="https://example.invalid/v1",
        api_key=api_key,
        model="test-model",
        transport=transport,
    )


def test_non_streaming_response_and_usage_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        assert payload["messages"][-1] == {"role": "user", "content": "質問"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "通常回答"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    provider = _provider(httpx.MockTransport(handler))

    response = provider.chat([ChatMessage(role="user", content="質問")])

    assert response.speech == "通常回答"
    assert provider.get_usage() is not None
    assert provider.get_usage().total_tokens == 14  # type: ignore[union-attr]
    assert "test-secret" not in repr(provider)


def test_streaming_response_is_joined_and_usage_is_read() -> None:
    body = "\n".join(
        (
            'data: {"choices":[{"delta":{"content":"こん"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"にちは"}}]}',
            "",
            (
                'data: {"choices":[],"usage":{"prompt_tokens":3,'
                '"completion_tokens":2,"total_tokens":5}}'
            ),
            "",
            "data: [DONE]",
            "",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    provider = _provider(httpx.MockTransport(handler))

    chunks = list(provider.stream_chat([ChatMessage(role="user", content="質問")]))

    assert chunks == ["こん", "にちは"]
    assert provider.last_response is not None
    assert provider.last_response.speech == "こんにちは"
    assert provider.get_usage() is not None
    assert provider.get_usage().input_tokens == 3  # type: ignore[union-attr]


def test_authentication_error_is_japanese_and_does_not_expose_key() -> None:
    secret = "never-show-this"
    provider = _provider(
        httpx.MockTransport(lambda _request: httpx.Response(401)),
        api_key=secret,
    )

    with pytest.raises(ProviderAuthenticationError) as error:
        provider.chat([ChatMessage(role="user", content="質問")])

    assert "APIキー" in str(error.value)
    assert secret not in str(error.value)


@pytest.mark.parametrize("streaming", [False, True])
def test_invalid_emotion_preserves_generated_gesture(streaming: bool) -> None:
    raw = json.dumps({"speech": "バキュン。", "emotion": "playful", "gesture": "playful_shoot"})

    def handler(_request: httpx.Request) -> httpx.Response:
        if not streaming:
            return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})
        chunks = (raw[:25], raw[25:])
        body = (
            "".join(
                "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]}) + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    provider = _provider(httpx.MockTransport(handler))
    messages = [ChatMessage(role="user", content="バキュンして")]
    if streaming:
        assert "".join(provider.stream_chat(messages)) == raw
        response = provider.last_response
    else:
        response = provider.chat(messages)
    assert response is not None
    assert response.gesture == "playful_shoot"
    assert response.emotion.value == "neutral"


def test_stream_retries_without_optional_usage_setting() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        payloads.append(payload)
        if "stream_options" in payload:
            return httpx.Response(400)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n',
        )

    provider = _provider(httpx.MockTransport(handler))

    assert list(provider.stream_chat([ChatMessage(role="user", content="質問")])) == ["OK"]
    assert len(payloads) == 2
    assert "stream_options" in payloads[0]
    assert "stream_options" not in payloads[1]


@pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5.6-luna", "gpt-6-luna"])
@pytest.mark.parametrize("streaming", [False, True])
def test_openai_reasoning_models_use_compatible_chat_completion_parameters(
    model: str, streaming: bool
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["max_completion_tokens"] == 1024
        assert "max_tokens" not in payload
        assert "temperature" not in payload
        if model == "gpt-6-luna":
            assert payload["reasoning_effort"] == "none"
        else:
            assert "reasoning_effort" not in payload
        if streaming:
            assert payload["stream_options"] == {"include_usage": True}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"choices":[{"delta":{"content":"対応済み"}}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "対応済み"}}]},
        )

    provider = OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-secret",
        model=model,
        transport=httpx.MockTransport(handler),
    )

    messages = [ChatMessage(role="user", content="質問")]
    if streaming:
        assert "".join(provider.stream_chat(messages)) == "対応済み"
        assert provider.last_response.speech == "対応済み"
    else:
        assert provider.chat(messages).speech == "対応済み"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "base_url,model",
    [
        ("https://api.openai.com/v1", "gpt-4.1"),
        ("https://example.invalid/v1", "gpt-6-luna"),
        ("https://api.openai.com.example.invalid/v1", "gpt-6-luna"),
        ("https://example.invalid/api.openai.com/v1", "gpt-5.6-luna"),
    ],
)
def test_other_models_and_compatible_endpoints_keep_existing_parameters(base_url, model):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.4
        assert "max_completion_tokens" not in payload
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    provider = OpenAICompatibleProvider(
        name="compatible",
        base_url=base_url,
        api_key="test-only",
        model=model,
        temperature=0.4,
        max_output_tokens=512,
        transport=httpx.MockTransport(handler),
    )
    assert provider.chat([ChatMessage(role="user", content="質問")]).speech == "OK"


def test_bad_request_for_unsupported_parameter_is_explained_safely() -> None:
    provider = _provider(
        httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "code": "unsupported_parameter",
                        "message": "Unsupported parameter: temperature",
                    }
                },
            )
        )
    )

    with pytest.raises(ProviderResponseError) as error:
        provider.chat([ChatMessage(role="user", content="質問")])

    assert "受け付けない設定" in str(error.value)
    assert "temperature" not in str(error.value)


def test_network_error_is_converted_to_safe_japanese() -> None:
    secret = "do-not-display"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    provider = _provider(httpx.MockTransport(handler), api_key=secret)

    with pytest.raises(ProviderConnectionError) as error:
        provider.chat([ChatMessage(role="user", content="質問")])

    assert "接続できません" in str(error.value)
    assert secret not in str(error.value)


def test_client_is_reused_and_done_does_not_wait_for_stream_eof():
    bodies = []

    class Body(httpx.SyncByteStream):
        closed = False

        def __init__(self, event):
            self.event = event

        def __iter__(self):
            yield ("data: " + json.dumps(self.event) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
            raise AssertionError("完了後の待機や通信エラーを読み込んではならない")

        def close(self):
            self.closed = True

    class Transport(httpx.MockTransport):
        close_count = 0

        def close(self):
            self.close_count += 1

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        payload = json.loads(request.content)
        if "tools" in payload:
            message = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "reply",
                        "type": "function",
                        "function": {"name": "reply_to_user", "arguments": '{"speech":"OK"}'},
                    }
                ]
            }
        else:
            message = {"content": '{"speech":"OK"}'}
        finish_reason = "tool_calls" if "tools" in payload else "stop"
        if not payload["stream"]:
            return httpx.Response(
                200, json={"choices": [{"message": message, "finish_reason": finish_reason}]}
            )
        body = Body({"choices": [{"delta": message, "finish_reason": finish_reason}]})
        bodies.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    transport = Transport(handler)
    provider = _provider(transport)
    messages = [ChatMessage(role="user", content="質問")]
    try:
        assert provider.chat(messages).speech == "OK"
        assert "".join(provider.stream_chat(messages)) == '{"speech":"OK"}'
        assert provider._research_reply(messages).speech == "OK"
        assert list(provider._stream_research_reply(messages))[-1].speech == "OK"
        assert provider.health_check()
        assert transport.close_count == 0
        assert len(bodies) == 2 and all(body.closed for body in bodies)
    finally:
        provider._close()
    assert transport.close_count == 1
    provider._close()
    assert transport.close_count == 1


def test_provider_close_waits_for_active_fork_without_blocking_new_request():
    entered, release = Event(), Event()

    class Transport(httpx.MockTransport):
        close_count = 0

        def close(self):
            self.close_count += 1

    def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"]
        if text == "旧応答":
            entered.set()
            assert release.wait(3)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    transport = Transport(handler)
    provider = _provider(transport)
    fork = provider._fork_request()
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(fork.chat, [ChatMessage(role="user", content="旧応答")])
        try:
            assert entered.wait(2)
            new = pool.submit(provider.chat, [ChatMessage(role="user", content="新応答")])
            assert new.result(timeout=2).speech == "新応答"
            provider._close()
            assert transport.close_count == 0
        finally:
            release.set()
        assert old.result(timeout=2).speech == "旧応答"
    assert provider.last_response.speech == "新応答"
    assert fork.last_response.speech == "旧応答"
    assert transport.close_count == 1


def test_service_releases_connections_on_switch_close_and_allows_rollback():
    from multi_llm_companion.providers import MockProvider
    from multi_llm_companion.services import ChatService
    from multi_llm_companion.storage import SQLiteConversationRepository

    provider = _provider(
        httpx.MockTransport(
            lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        )
    )
    service = ChatService(provider=provider, repository=SQLiteConversationRepository(":memory:"))
    try:
        provider.chat([ChatMessage(role="user", content="最初")])
        with provider._client() as first:
            assert not first.is_closed
        service.set_provider(MockProvider())
        assert first.is_closed
        service.set_provider(provider)
        assert provider.chat([ChatMessage(role="user", content="設定を戻す")]).speech == "OK"
        with provider._client() as restored:
            assert not restored.is_closed
        assert restored is not first
    finally:
        service.close()
    assert restored.is_closed
