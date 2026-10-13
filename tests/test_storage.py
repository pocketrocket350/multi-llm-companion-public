from __future__ import annotations

from pathlib import Path

from multi_llm_companion.domain import (
    AssistantResponse,
    Emotion,
    EmotionState,
    MemoryCategory,
    MemoryStatus,
    TokenUsage,
)
from multi_llm_companion.storage import SQLiteConversationRepository


def test_sqlite_saves_response_and_usage_then_deletes_cascade(tmp_path: Path) -> None:
    repository = SQLiteConversationRepository(tmp_path / "history" / "chat.db")
    conversation = repository.create_conversation("保存テスト")
    repository.add_message(conversation.id, "user", "こんにちは")
    response = AssistantResponse(
        speech="ようこそ",
        emotion=Emotion.HAPPY,
        emotion_intensity=0.5,
        gesture="wave",
        gaze="user",
        memory_candidates=[],
        dialogue_mode="supporting",
        dialogue_act="comfort",
        current_topic="仕事の疲れ",
        open_loop="必要なら続きを聞く",
    )
    usage = TokenUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19)
    repository.add_message(
        conversation.id,
        "assistant",
        response.speech,
        response=response,
        usage=usage,
    )

    messages = repository.list_messages(conversation.id)

    assert [message.content for message in messages] == ["こんにちは", "ようこそ"]
    assert messages[1].response == response
    assert messages[1].response is not None
    assert messages[1].response.open_loop == "必要なら続きを聞く"
    assert messages[1].usage == usage
    assert repository.delete_conversation(conversation.id) is True
    assert repository.list_messages(conversation.id) == []
    assert repository.get_conversation(conversation.id) is None
    repository.close()


def test_conversations_are_listed_newest_first() -> None:
    repository = SQLiteConversationRepository(":memory:")
    first = repository.create_conversation("1件目")
    second = repository.create_conversation("2件目")

    assert [item.id for item in repository.list_conversations()] == [second.id, first.id]
    repository.close()


def test_old_response_json_loads_with_dialogue_defaults() -> None:
    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation("旧JSON")
    repository._connection.execute(  # type: ignore[attr-defined]
        """
        INSERT INTO messages (conversation_id, role, content, response_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            conversation.id,
            "assistant",
            "旧回答",
            '{"speech":"旧回答","emotion":"neutral"}',
            conversation.created_at.isoformat(),
        ),
    )

    response = repository.list_messages(conversation.id)[0].response

    assert response is not None
    assert response.speech == "旧回答"
    assert response.dialogue_mode.value == "casual"
    assert response.dialogue_act.value == "answer"
    assert response.current_topic == ""
    assert response.open_loop == ""
    repository.close()


def test_token_usage_summary_separates_one_conversation_and_all_history() -> None:
    repository = SQLiteConversationRepository(":memory:")
    first = repository.create_conversation("1件目")
    second = repository.create_conversation("2件目")
    repository.add_message(
        first.id,
        "assistant",
        "1回目",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    repository.add_message(second.id, "assistant", "取得できなかった回答")
    repository.add_message(
        second.id,
        "assistant",
        "2回目",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
    )

    assert repository.get_token_usage_summary(first.id).model_dump() == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "response_count": 1,
    }
    assert repository.get_token_usage_summary().model_dump() == {
        "input_tokens": 30,
        "output_tokens": 13,
        "total_tokens": 43,
        "response_count": 2,
    }
    repository.close()


def test_memory_is_separate_from_conversation_and_searches_approved_items() -> None:
    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation("記憶元")
    candidate = repository.create_memory(
        "ユーザーは辛くないカレーが好きです",
        category=MemoryCategory.PREFERENCE,
        source_conversation_id=conversation.id,
    )

    assert repository.find_relevant_memories("カレーを食べたい", limit=5) == []
    approved = repository.update_memory(candidate.id, status=MemoryStatus.APPROVED)
    matches = repository.find_relevant_memories("カレーを食べたい", limit=5)

    assert matches[0].id == approved.id
    assert repository.delete_conversation(conversation.id) is True
    saved = repository.list_memories()
    assert [item.id for item in saved] == [approved.id]
    assert saved[0].last_used_at is not None
    assert repository.delete_all_memories() == 1
    repository.close()


def test_emotion_state_is_saved_per_conversation_and_deleted_with_it() -> None:
    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation("感情テスト")
    state = repository.save_emotion_state(
        EmotionState(
            conversation_id=conversation.id,
            emotion=Emotion.HAPPY,
            intensity=0.6,
            updated_at=conversation.updated_at,
        )
    )

    assert state.emotion is Emotion.HAPPY
    assert state.intensity == 0.6
    assert repository.get_emotion_state(conversation.id) is not None
    assert repository.delete_conversation(conversation.id) is True
    assert repository.get_emotion_state(conversation.id) is None
    repository.close()
