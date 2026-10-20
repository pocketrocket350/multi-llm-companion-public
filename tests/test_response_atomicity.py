from __future__ import annotations

import json

import pytest

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import AssistantResponse, MemoryStatus
from multi_llm_companion.integrations.mock_avatar import Mock3DAvatarController
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService, MemoryService
from multi_llm_companion.services.emotion import EmotionService
from multi_llm_companion.storage import SQLiteConversationRepository, StorageError


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("first_action", ["add", "update", "delete"])
def test_failed_memory_batch_rolls_back_response_emotion_and_every_change(
    tmp_path, monkeypatch, streaming, first_action
):
    with SQLiteConversationRepository(tmp_path / "test.db") as repository:
        memory = MemoryService(repository, MemoryConfig())
        avatar = Mock3DAvatarController()
        old = repository.create_memory("ユーザーの趣味は釣り", status=MemoryStatus.APPROVED)
        text = "趣味は釣りから陶芸に変えたよ。仕事は大工です。"
        if first_action == "delete":
            text = "趣味のことは忘れて"
        updates = [
            {
                "action": first_action,
                "content": "" if first_action == "delete" else "ユーザーの趣味は陶芸",
                "evidence": text.split("。")[0],
                "target_id": None if first_action == "add" else old.id,
                "target_content": "" if first_action == "add" else old.content,
            },
            {"action": "add", "content": "ユーザーの仕事は大工", "evidence": "仕事は大工です"},
        ]
        provider = MockProvider(
            responder=lambda messages: json.dumps(
                {"speech": "そうなんだね。", "emotion": "happy", "memory_updates": updates},
                ensure_ascii=False,
            )
        )
        service = ChatService(
            provider=provider,
            repository=repository,
            memory_service=memory,
            emotion_service=EmotionService(repository),
            avatar_controller=avatar,
        )
        original = repository.create_memory

        def fail_second(content, **kwargs):
            if content == "ユーザーの仕事は大工":
                raise StorageError("人工的な保存失敗")
            return original(content, **kwargs)

        monkeypatch.setattr(repository, "create_memory", fail_second)
        original_delete = repository.delete_memory
        if first_action == "delete":

            def fail_after_delete(memory_id):
                original_delete(memory_id)
                raise StorageError("人工的な削除失敗")

            monkeypatch.setattr(repository, "delete_memory", fail_after_delete)
        with pytest.raises(StorageError):
            if streaming:
                list(service.stream_message(text))
            else:
                service.send_message(text)
        conversation_id = service.current_conversation_id
        assert [item.role.value for item in repository.list_messages(conversation_id)] == ["user"]
        assert [
            item.model_dump(exclude={"last_used_at"}) for item in repository.list_memories()
        ] == [old.model_dump(exclude={"last_used_at"})]
        assert repository.get_emotion_state(conversation_id) is None
        assert memory.last_candidates == []
        assert service.last_response is None
        assert service._published_response is None
        assert avatar.last_response is None

        monkeypatch.setattr(repository, "create_memory", original)
        monkeypatch.setattr(repository, "delete_memory", original_delete)
        service.send_message(text)
        assert [item.role.value for item in repository.list_messages(conversation_id)] == [
            "user",
            "user",
            "assistant",
        ]
        assert service.last_response is not None
        assert repository.get_emotion_state(conversation_id) is not None


@pytest.mark.parametrize("failure", ["assistant", "emotion"])
def test_storage_failure_before_memory_never_publishes_reply(tmp_path, monkeypatch, failure):
    with SQLiteConversationRepository(tmp_path / "test.db") as repository:
        service = ChatService(
            provider=MockProvider(responder=lambda messages: '{"speech":"返答","emotion":"happy"}'),
            repository=repository,
            emotion_service=EmotionService(repository),
        )
        original = repository.add_message

        def fail(*args, **kwargs):
            raise StorageError("人工的な保存失敗")

        if failure == "assistant":
            monkeypatch.setattr(
                repository,
                "add_message",
                lambda conversation_id, role, *args, **kwargs: (
                    fail()
                    if role == "assistant"
                    else original(conversation_id, role, *args, **kwargs)
                ),
            )
        else:
            monkeypatch.setattr(repository, "save_emotion_state", fail)
        with pytest.raises(StorageError):
            service.send_message("こんにちは")
        assert [item.role.value for item in service.get_current_messages()] == ["user"]
        assert repository.get_emotion_state(service.current_conversation_id) is None
        assert service.last_response is service._published_response is None


def test_direct_memory_batch_is_atomic(tmp_path, monkeypatch):
    with SQLiteConversationRepository(tmp_path / "test.db") as repository:
        memory = MemoryService(repository, MemoryConfig())
        text = "趣味は陶芸です。仕事は大工です。"
        response = AssistantResponse.model_validate(
            {
                "speech": "そうなんだね。",
                "memory_updates": [
                    {"action": "add", "content": "趣味は陶芸", "evidence": "趣味は陶芸です"},
                    {"action": "add", "content": "仕事は大工", "evidence": "仕事は大工です"},
                ],
            }
        )
        original = repository.create_memory

        def fail_second(content, **kwargs):
            if content == "仕事は大工":
                raise StorageError("人工的な保存失敗")
            return original(content, **kwargs)

        monkeypatch.setattr(repository, "create_memory", fail_second)
        with pytest.raises(StorageError):
            memory.propose(text, response, "test", turn=memory.prepare_turn(text))
        assert repository.list_memories() == memory.last_candidates == []


def test_nested_storage_rollback_preserves_outer_transaction(tmp_path):
    with SQLiteConversationRepository(tmp_path / "test.db") as repository:
        with repository._transaction():
            repository.create_memory("残す")
            with pytest.raises(StorageError), repository._transaction():
                repository.create_memory("戻す")
                raise StorageError("内側だけ失敗")
            repository.create_memory("継続")
        assert {item.content for item in repository.list_memories()} == {"残す", "継続"}
