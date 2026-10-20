"""標準ライブラリSQLiteを使った会話履歴保存。"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from uuid import uuid4

from pydantic import ValidationError

from multi_llm_companion.domain import (
    AssistantResponse,
    Conversation,
    Emotion,
    EmotionState,
    MemoryCategory,
    MemoryEntry,
    MemoryStatus,
    MessageRole,
    StoredMessage,
    TokenUsage,
    TokenUsageSummary,
)

from .base import ConversationRepository, StorageError


class SQLiteConversationRepository(ConversationRepository):
    """1つのSQLiteファイルへ会話とメッセージを保存する。"""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._lock = threading.RLock()
        self._closed = False
        if self._database_path != ":memory:":
            try:
                Path(self._database_path).expanduser().resolve().parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            except OSError as exc:
                raise StorageError(
                    "会話履歴フォルダーを作成できませんでした。保存先を確認してください。"
                ) from exc

        try:
            self._connection = sqlite3.connect(
                self._database_path,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self._database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
        except sqlite3.Error as exc:
            raise StorageError(
                "会話履歴データベースを開けませんでした。保存先を確認してください。"
            ) from exc

    @property
    def database_path(self) -> str:
        """現在利用中のデータベース場所。"""

        return self._database_path

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """入れ子の保存も、外側の確定まで同じロックとトランザクションに含める。"""

        with self._lock:
            self._ensure_open()
            savepoint = f"write_{uuid4().hex}"
            self._connection.execute(f"SAVEPOINT {savepoint}")
            try:
                yield
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise

    def create_conversation(self, title: str = "新しい会話") -> Conversation:
        normalized_title = title.strip() or "新しい会話"
        conversation_id = str(uuid4())
        now = _now_text()
        try:
            with self._transaction():
                self._ensure_open()
                self._connection.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, normalized_title, now, now),
                )
        except sqlite3.Error as exc:
            raise StorageError("新しい会話を保存できませんでした。") from exc
        return Conversation(
            id=conversation_id,
            title=normalized_title,
            created_at=_parse_datetime(now),
            updated_at=_parse_datetime(now),
        )

    def list_conversations(self, *, limit: int | None = None) -> list[Conversation]:
        if limit is not None and limit <= 0:
            raise ValueError("limit は1以上にしてください")
        query = (
            "SELECT id, title, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC, rowid DESC"
        )
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        try:
            with self._lock:
                self._ensure_open()
                rows = self._connection.execute(query, parameters).fetchall()
            return [_conversation_from_row(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError("会話一覧を読み込めませんでした。") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(
                "会話履歴データが破損しているため、一覧を読み込めませんでした。"
            ) from exc

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        try:
            with self._lock:
                self._ensure_open()
                row = self._connection.execute(
                    """
                    SELECT id, title, created_at, updated_at
                    FROM conversations WHERE id = ?
                    """,
                    (conversation_id,),
                ).fetchone()
            return _conversation_from_row(row) if row is not None else None
        except sqlite3.Error as exc:
            raise StorageError("会話を読み込めませんでした。") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(
                "会話履歴データが破損しているため、会話を読み込めませんでした。"
            ) from exc

    def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> Conversation:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("会話の題名を入力してください")
        now = _now_text()
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute(
                    """
                    UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?
                    """,
                    (normalized_title, now, conversation_id),
                )
                if cursor.rowcount == 0:
                    raise StorageError("指定された会話が見つかりません。")
        except sqlite3.Error as exc:
            raise StorageError("会話の題名を更新できませんでした。") from exc
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise StorageError("更新した会話を読み込めませんでした。")
        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError("会話履歴を削除できませんでした。") from exc

    def delete_all_conversations(self) -> int:
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute("DELETE FROM conversations")
                return max(cursor.rowcount, 0)
        except sqlite3.Error as exc:
            raise StorageError("会話履歴をすべて削除できませんでした。") from exc

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole | str,
        content: str,
        *,
        response: AssistantResponse | None = None,
        usage: TokenUsage | None = None,
    ) -> StoredMessage:
        try:
            normalized_role = role if isinstance(role, MessageRole) else MessageRole(role)
        except ValueError as exc:
            raise ValueError("メッセージのroleが正しくありません") from exc

        now = _now_text()
        response_json = response.model_dump_json() if response is not None else None
        prompt_tokens = usage.prompt_tokens if usage is not None else None
        completion_tokens = usage.completion_tokens if usage is not None else None
        total_tokens = usage.total_tokens if usage is not None else None
        try:
            with self._transaction():
                self._ensure_open()
                exists = self._connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if exists is None:
                    raise StorageError("メッセージの保存先となる会話が見つかりません。")
                cursor = self._connection.execute(
                    """
                    INSERT INTO messages (
                        conversation_id,
                        role,
                        content,
                        response_json,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        normalized_role.value,
                        content,
                        response_json,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        now,
                    ),
                )
                self._connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
                message_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StorageError("メッセージを保存できませんでした。") from exc
        return StoredMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=normalized_role,
            content=content,
            response=response,
            usage=usage,
            created_at=_parse_datetime(now),
        )

    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        try:
            with self._lock:
                self._ensure_open()
                rows = self._connection.execute(
                    """
                    SELECT
                        id,
                        conversation_id,
                        role,
                        content,
                        response_json,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (conversation_id,),
                ).fetchall()
            return [_message_from_row(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError("メッセージ履歴を読み込めませんでした。") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError(
                "会話履歴データが破損しているため、メッセージを読み込めませんでした。"
            ) from exc

    def get_token_usage_summary(
        self,
        conversation_id: str | None = None,
    ) -> TokenUsageSummary:
        """APIから取得して保存した使用量だけを合計する。"""

        query = """
            SELECT
                COALESCE(SUM(prompt_tokens), 0) AS input_tokens,
                COALESCE(SUM(completion_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COUNT(*) AS response_count
            FROM messages
            WHERE role = 'assistant'
              AND prompt_tokens IS NOT NULL
              AND completion_tokens IS NOT NULL
              AND total_tokens IS NOT NULL
        """
        parameters: tuple[str, ...] = ()
        if conversation_id is not None:
            query += " AND conversation_id = ?"
            parameters = (conversation_id,)
        try:
            with self._lock:
                self._ensure_open()
                row = self._connection.execute(query, parameters).fetchone()
            if row is None:
                return TokenUsageSummary()
            return TokenUsageSummary(
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                total_tokens=int(row["total_tokens"]),
                response_count=int(row["response_count"]),
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise StorageError("トークン使用量を集計できませんでした。") from exc

    def create_memory(
        self,
        content: str,
        *,
        category: MemoryCategory = MemoryCategory.OTHER,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
        source_conversation_id: str | None = None,
    ) -> MemoryEntry:
        normalized = _memory_content(content)
        now = _now_text()
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute(
                    """
                    INSERT INTO memories (
                        content, category, status, source_conversation_id,
                        created_at, updated_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (normalized, category.value, status.value, source_conversation_id, now, now),
                )
                memory_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StorageError("記憶候補を保存できませんでした。") from exc
        memory = self._get_memory(memory_id)
        if memory is None:
            raise StorageError("保存した記憶候補を読み込めませんでした。")
        return memory

    def list_memories(self, *, status: MemoryStatus | None = None) -> list[MemoryEntry]:
        query = "SELECT * FROM memories"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status.value,)
        query += " ORDER BY updated_at DESC, id DESC"
        try:
            with self._lock:
                self._ensure_open()
                rows = self._connection.execute(query, parameters).fetchall()
            return [_memory_from_row(row) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError("記憶一覧を読み込めませんでした。") from exc

    def update_memory(
        self,
        memory_id: int,
        *,
        content: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
    ) -> MemoryEntry:
        current = self._get_memory(memory_id)
        if current is None:
            raise StorageError("指定された記憶が見つかりません。")
        new_content = _memory_content(content) if content is not None else current.content
        new_category = category or current.category
        new_status = status or current.status
        now = _now_text()
        try:
            with self._transaction():
                self._ensure_open()
                self._connection.execute(
                    """
                    UPDATE memories
                    SET content = ?, category = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_content, new_category.value, new_status.value, now, memory_id),
                )
        except sqlite3.Error as exc:
            raise StorageError("記憶を更新できませんでした。") from exc
        updated = self._get_memory(memory_id)
        if updated is None:
            raise StorageError("更新した記憶を読み込めませんでした。")
        return updated

    def delete_memory(self, memory_id: int) -> bool:
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError("記憶を削除できませんでした。") from exc

    def delete_all_memories(self) -> int:
        try:
            with self._transaction():
                self._ensure_open()
                cursor = self._connection.execute("DELETE FROM memories")
                return max(cursor.rowcount, 0)
        except sqlite3.Error as exc:
            raise StorageError("記憶をすべて削除できませんでした。") from exc

    def find_relevant_memories(
        self,
        text: str,
        *,
        limit: int,
        min_score: int = 1,
    ) -> list[MemoryEntry]:
        if limit <= 0 or min_score <= 0:
            return []
        normalized = _searchable_text(text)
        if not normalized:
            return []
        candidates = self.list_memories(status=MemoryStatus.APPROVED)
        ranked = sorted(
            (
                (memory, _memory_score(normalized, _searchable_text(memory.content)))
                for memory in candidates
            ),
            key=lambda item: (item[1], item[0].updated_at),
            reverse=True,
        )
        selected = [memory for memory, score in ranked if score >= min_score][:limit]
        if selected:
            now = _now_text()
            try:
                with self._transaction():
                    self._ensure_open()
                    self._connection.executemany(
                        "UPDATE memories SET last_used_at = ? WHERE id = ?",
                        [(now, memory.id) for memory in selected],
                    )
            except sqlite3.Error as exc:
                raise StorageError("利用した記憶を更新できませんでした。") from exc
        return selected

    def get_emotion_state(self, conversation_id: str) -> EmotionState | None:
        try:
            with self._lock:
                self._ensure_open()
                row = self._connection.execute(
                    "SELECT * FROM emotion_states WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            return _emotion_state_from_row(row) if row is not None else None
        except sqlite3.Error as exc:
            raise StorageError("感情状態を読み込めませんでした。") from exc

    def save_emotion_state(self, state: EmotionState) -> EmotionState:
        now = _now_text()
        try:
            with self._transaction():
                self._ensure_open()
                exists = self._connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (state.conversation_id,),
                ).fetchone()
                if exists is None:
                    raise StorageError("感情状態の保存先となる会話が見つかりません。")
                self._connection.execute(
                    """
                    INSERT INTO emotion_states (conversation_id, emotion, intensity, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        emotion = excluded.emotion,
                        intensity = excluded.intensity,
                        updated_at = excluded.updated_at
                    """,
                    (state.conversation_id, state.emotion.value, state.intensity, now),
                )
        except sqlite3.Error as exc:
            raise StorageError("感情状態を保存できませんでした。") from exc
        saved = self.get_emotion_state(state.conversation_id)
        if saved is None:
            raise StorageError("保存した感情状態を読み込めませんでした。")
        return saved

    def _get_memory(self, memory_id: int) -> MemoryEntry | None:
        try:
            with self._lock:
                self._ensure_open()
                row = self._connection.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
            return _memory_from_row(row) if row is not None else None
        except sqlite3.Error as exc:
            raise StorageError("記憶を読み込めませんでした。") from exc

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteConversationRepository:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("会話履歴データベースは既に閉じられています。")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    response_json TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id, id);

                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                ON conversations(updated_at DESC);

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL CHECK (category IN (
                        'preference', 'profile', 'plan', 'relationship', 'other'
                    )),
                    status TEXT NOT NULL CHECK (status IN ('candidate', 'approved')),
                    source_conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memories_status_updated_at
                ON memories(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS emotion_states (
                    conversation_id TEXT PRIMARY KEY,
                    emotion TEXT NOT NULL CHECK (emotion IN (
                        'neutral', 'happy', 'sad', 'angry', 'embarrassed', 'surprised'
                    )),
                    intensity REAL NOT NULL CHECK (intensity >= 0 AND intensity <= 1),
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                );
                """
            )


def _now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=str(row["id"]),
        title=str(row["title"]),
        created_at=_parse_datetime(str(row["created_at"])),
        updated_at=_parse_datetime(str(row["updated_at"])),
    )


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    response: AssistantResponse | None = None
    response_json = row["response_json"]
    if isinstance(response_json, str):
        try:
            response = AssistantResponse.model_validate_json(response_json)
        except ValidationError:
            response = None

    usage: TokenUsage | None = None
    usage_values = (
        row["prompt_tokens"],
        row["completion_tokens"],
        row["total_tokens"],
    )
    if any(value is not None for value in usage_values):
        usage = TokenUsage(
            prompt_tokens=int(row["prompt_tokens"] or 0),
            completion_tokens=int(row["completion_tokens"] or 0),
            total_tokens=int(row["total_tokens"] or 0),
        )

    return StoredMessage(
        id=int(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=MessageRole(str(row["role"])),
        content=str(row["content"]),
        response=response,
        usage=usage,
        created_at=_parse_datetime(str(row["created_at"])),
    )


def _memory_from_row(row: sqlite3.Row) -> MemoryEntry:
    last_used = row["last_used_at"]
    return MemoryEntry(
        id=int(row["id"]),
        content=str(row["content"]),
        category=MemoryCategory(str(row["category"])),
        status=MemoryStatus(str(row["status"])),
        source_conversation_id=(
            str(row["source_conversation_id"])
            if row["source_conversation_id"] is not None
            else None
        ),
        created_at=_parse_datetime(str(row["created_at"])),
        updated_at=_parse_datetime(str(row["updated_at"])),
        last_used_at=_parse_datetime(str(last_used)) if last_used is not None else None,
    )


def _emotion_state_from_row(row: sqlite3.Row) -> EmotionState:
    return EmotionState(
        conversation_id=str(row["conversation_id"]),
        emotion=Emotion(str(row["emotion"])),
        intensity=float(row["intensity"]),
        updated_at=_parse_datetime(str(row["updated_at"])),
    )


def _memory_content(content: str) -> str:
    normalized = " ".join(content.split())
    if not normalized:
        raise ValueError("記憶の内容を入力してください。")
    if len(normalized) > 500:
        raise ValueError("記憶は500文字以内にしてください。")
    return normalized


def _searchable_text(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _memory_score(query: str, content: str) -> int:
    if not query or not content:
        return 0
    if query in content or content in query:
        return min(len(query), len(content)) + 10
    query_bigrams = _meaningful_bigrams(query)
    content_bigrams = _meaningful_bigrams(content)
    bigram_score = len(query_bigrams & content_bigrams)
    character_score = len(_meaningful_characters(query) & _meaningful_characters(content))
    return bigram_score * 2 + character_score * 2


def _meaningful_bigrams(text: str) -> set[str]:
    return {
        text[index : index + 2]
        for index in range(len(text) - 1)
        if not any(character in _MEMORY_STOP_CHARACTERS for character in text[index : index + 2])
    }


def _meaningful_characters(text: str) -> set[str]:
    return {character for character in text if character not in _MEMORY_STOP_CHARACTERS}


_MEMORY_STOP_CHARACTERS = frozenset("のはがをにでとへもや話")
