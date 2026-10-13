from multi_llm_companion.domain import ChatMessage
from multi_llm_companion.providers import MockProvider


def test_mock_supports_all_provider_operations() -> None:
    provider = MockProvider(chunk_size=3)
    messages = [ChatMessage(role="user", content="テスト")]

    chunks = list(provider.stream_chat(messages))

    assert "".join(chunks) == "モック応答です。『テスト』を受け取りました。"
    assert len(chunks) > 1
    assert provider.last_response is not None
    assert provider.last_response.speech == "".join(chunks)
    assert provider.get_usage() is not None
    assert provider.health_check() is True


def test_mock_parses_structured_response() -> None:
    provider = MockProvider(
        response_text=(
            '{"speech":"構造化された回答","emotion":"neutral",'
            '"emotion_intensity":0,"gesture":"","gaze":"",'
            '"memory_candidates":[]}'
        )
    )

    response = provider.chat([ChatMessage(role="user", content="質問")])

    assert response.speech == "構造化された回答"
