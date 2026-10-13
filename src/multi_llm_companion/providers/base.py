"""すべてのLLMプロバイダーが守る共通インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy

from multi_llm_companion.domain import AssistantResponse, ChatMessage, TokenUsage


class ChatProvider(ABC):
    """LLM固有実装を画面や保存処理から切り離す。"""

    def __init__(self, *, name: str, model: str) -> None:
        self._name = name
        self._model = model
        self._last_usage: TokenUsage | None = None
        self._last_raw_response = ""
        self._last_response: AssistantResponse | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        """設定画面で表示するプロバイダー名。"""

        return self._name

    @property
    def model(self) -> str:
        """設定画面で表示するモデル名。"""

        return self._model

    @property
    def last_raw_response(self) -> str:
        """直前のLLM回答。ストリーム完了後の解析に利用する。"""

        return self._last_raw_response

    @property
    def last_response(self) -> AssistantResponse | None:
        """直前の回答を共通形式で返す。"""

        return self._last_response

    @property
    def last_error(self) -> str | None:
        """health_check失敗理由など、安全に表示できる日本語。"""

        return self._last_error

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AssistantResponse:
        """回答を最後まで取得する。"""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """回答本文を受信できた断片ごとに返す。"""

    @abstractmethod
    def health_check(self) -> bool:
        """現在の設定で接続できるか確認する。"""

    def get_usage(self) -> TokenUsage | None:
        """直前の回答で取得できたトークン数を返す。"""

        return self._last_usage

    @staticmethod
    def prepare_messages(
        messages: Sequence[ChatMessage],
        system_prompt: str | None,
    ) -> list[ChatMessage]:
        """人格プロンプトを先頭に追加したコピーを作る。"""

        prepared = list(messages)
        if system_prompt:
            prepared.insert(0, ChatMessage(role="system", content=system_prompt))
        return prepared

    def _begin_request(self) -> None:
        """前回値が今回の結果として誤表示されないよう消去する。"""

        self._last_usage = None
        self._last_raw_response = ""
        self._last_response = None
        self._last_error = None

    def _close(self) -> None:
        """保持している通信資源を解放する。接続を持つ実装だけが上書きする。"""

        return None

    def _fork_request(self) -> ChatProvider:
        """接続条件を保ち、並行する能動要求の結果状態だけを分離する。"""
        provider = copy(self)
        provider._begin_request()
        return provider

    def _adopt_request_result(self, provider: ChatProvider) -> None:
        """取消されていない分離要求の結果を、従来の状態表示へ戻す。"""
        self._last_usage = provider.get_usage()
        self._last_raw_response = provider.last_raw_response
        self._last_response = provider.last_response
        self._last_error = provider.last_error
