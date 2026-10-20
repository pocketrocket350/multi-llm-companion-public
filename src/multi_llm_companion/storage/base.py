"""保存先を将来変更できるようにする共通インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager

from multi_llm_companion.domain import (
    AssistantResponse,
    Conversation,
    EmotionState,
    MemoryCategory,
    MemoryEntry,
    MemoryStatus,
    MessageRole,
    StoredMessage,
    TokenUsage,
    TokenUsageSummary,
)


class StorageError(RuntimeError):
    """会話履歴の保存や読込に失敗したときの例外。"""


class ConversationRepository(ABC):
    """会話履歴を扱う保存先の共通インターフェース。"""

    def _transaction(self) -> AbstractContextManager[None]:
        """内部の複数書き込みを一括確定し、失敗時は元に戻す。"""

        raise NotImplementedError("保存先が一括保存に対応していません。")

    @abstractmethod
    def create_conversation(self, title: str = "新しい会話") -> Conversation:
        """空の会話を1件作る。"""

    @abstractmethod
    def list_conversations(self, *, limit: int | None = None) -> list[Conversation]:
        """更新が新しい順に会話一覧を返す。"""

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """IDで会話を1件取得する。"""

    @abstractmethod
    def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> Conversation:
        """会話一覧に表示する題名を更新する。"""

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """会話と配下のメッセージを削除する。"""

    @abstractmethod
    def delete_all_conversations(self) -> int:
        """全会話を削除し、削除件数を返す。"""

    @abstractmethod
    def add_message(
        self,
        conversation_id: str,
        role: MessageRole | str,
        content: str,
        *,
        response: AssistantResponse | None = None,
        usage: TokenUsage | None = None,
    ) -> StoredMessage:
        """会話へメッセージを追加する。"""

    @abstractmethod
    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        """1会話のメッセージを古い順に返す。"""

    @abstractmethod
    def get_token_usage_summary(
        self,
        conversation_id: str | None = None,
    ) -> TokenUsageSummary:
        """指定会話、または全会話の保存済みトークン使用量を集計する。"""

    @abstractmethod
    def create_memory(
        self,
        content: str,
        *,
        category: MemoryCategory = MemoryCategory.OTHER,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
        source_conversation_id: str | None = None,
    ) -> MemoryEntry:
        """承認前または承認済みの記憶を保存する。"""

    @abstractmethod
    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
    ) -> list[MemoryEntry]:
        """記憶を更新が新しい順に返す。"""

    @abstractmethod
    def update_memory(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
    ) -> MemoryEntry:
        """記憶の文面・分類・承認状態を更新する。"""

    @abstractmethod
    def delete_memory(self, memory_id: int) -> bool:
        """記憶を完全に削除する。"""

    @abstractmethod
    def delete_all_memories(self) -> int:
        """すべての記憶を完全に削除する。"""

    @abstractmethod
    def find_relevant_memories(
        self,
        text: str,
        *,
        limit: int,
        min_score: int = 1,
    ) -> list[MemoryEntry]:
        """承認済み記憶から、文章に関係しそうなものを返す。"""

    @abstractmethod
    def get_emotion_state(self, conversation_id: str) -> EmotionState | None:
        """会話に保存された現在の感情状態を返す。"""

    @abstractmethod
    def save_emotion_state(self, state: EmotionState) -> EmotionState:
        """会話ごとの現在感情を保存または更新する。"""

    def get_messages(self, conversation_id: str) -> Sequence[StoredMessage]:
        """list_messagesの読みやすい別名。"""

        return self.list_messages(conversation_id)

    @abstractmethod
    def close(self) -> None:
        """保存先との接続を閉じる。"""
