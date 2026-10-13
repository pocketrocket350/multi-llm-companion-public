"""APIキーなしで画面と保存を確認するためのモック。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from multi_llm_companion.domain import (
    AssistantResponse,
    ChatMessage,
    MessageRole,
    TokenUsage,
    parse_assistant_response,
)

from .base import ChatProvider

type MockResponder = Callable[[Sequence[ChatMessage]], str]


class MockProvider(ChatProvider):
    """入力内容に対して決まった形式で返答する確認用プロバイダー。"""

    def __init__(
        self,
        *,
        name: str = "mock",
        model: str = "mock-model",
        response_text: str | None = None,
        responder: MockResponder | None = None,
        chunk_size: int = 4,
    ) -> None:
        super().__init__(name=name, model=model)
        if chunk_size <= 0:
            raise ValueError("chunk_size は1以上にしてください")
        self._response_text = response_text
        self._responder = responder
        self._chunk_size = chunk_size

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AssistantResponse:
        self._begin_request()
        prepared = self.prepare_messages(messages, system_prompt)
        raw_text = self._make_response(prepared)
        response = parse_assistant_response(raw_text)
        self._record_result(prepared, raw_text, response)
        return response

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        self._begin_request()
        prepared = self.prepare_messages(messages, system_prompt)
        raw_text = self._make_response(prepared)
        for start in range(0, len(raw_text), self._chunk_size):
            yield raw_text[start : start + self._chunk_size]
        response = parse_assistant_response(raw_text)
        self._record_result(prepared, raw_text, response)

    def health_check(self) -> bool:
        self._last_error = None
        return True

    def _make_response(self, messages: Sequence[ChatMessage]) -> str:
        if self._responder is not None:
            return self._responder(messages)
        if self._response_text is not None:
            return self._response_text

        user_text = next(
            (message.content for message in reversed(messages) if message.role is MessageRole.USER),
            "",
        )
        if user_text:
            return f"モック応答です。『{user_text}』を受け取りました。"
        return "モック応答です。メッセージを入力してください。"

    def _record_result(
        self,
        messages: Sequence[ChatMessage],
        raw_text: str,
        response: AssistantResponse,
    ) -> None:
        input_length = sum(len(message.content) for message in messages)
        self._last_usage = TokenUsage(
            prompt_tokens=_estimated_tokens(input_length),
            completion_tokens=_estimated_tokens(len(raw_text)),
            total_tokens=(_estimated_tokens(input_length) + _estimated_tokens(len(raw_text))),
        )
        self._last_raw_response = raw_text
        self._last_response = response


def _estimated_tokens(character_count: int) -> int:
    """モック専用の概算。実APIの計測値とは区別する。"""

    if character_count <= 0:
        return 0
    return max(1, (character_count + 3) // 4)
