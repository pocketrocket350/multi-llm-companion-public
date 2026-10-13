from __future__ import annotations

from collections.abc import Sequence

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import ChatMessage, MessageRole
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService, MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository


def test_approved_memory_crosses_conversations_but_not_disable_or_delete() -> None:
    prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage]) -> str:
        prompts.append(
            next(message.content for message in messages if message.role is MessageRole.SYSTEM)
        )
        return '{"speech":"了解","memory_candidates":[]}'

    repository = SQLiteConversationRepository(":memory:")
    memories = MemoryService(repository, MemoryConfig())
    service = ChatService(
        provider=MockProvider(responder=responder),
        repository=repository,
        system_prompt="人格",
        memory_service=memories,
    )
    try:
        service.send_message("私をナツと呼んで")
        saved = memories.list_memories()
        assert len(saved) == 1

        service.start_new_conversation("記憶を使う")
        service.send_message("私の名前は？")
        assert "ナツ" in prompts[-1]

        memories.set_enabled(False)
        service.start_new_conversation("記憶を無効にする")
        service.send_message("私の名前は？")
        assert "ナツ" not in prompts[-1]

        memories.set_enabled(True)
        assert memories.delete(saved[0].id) is True
        service.start_new_conversation("記憶を削除する")
        service.send_message("私の名前は？")
        assert "ナツ" not in prompts[-1]
    finally:
        service.close()


def test_natural_hobby_is_persistent_and_reusable_without_old_conversation(tmp_path):
    path = tmp_path / "memory.db"
    prompts = []

    def responder(messages):
        prompts.append(messages[0].content)
        return '{"speech":"うん。","memory_candidates":[]}'

    def open_service():
        repository = SQLiteConversationRepository(path)
        memory = MemoryService(repository, MemoryConfig())
        return ChatService(
            provider=MockProvider(responder=responder),
            repository=repository,
            system_prompt="人格",
            memory_service=memory,
        ), memory

    service, memory = open_service()
    try:
        memory.set_enabled(False)
        service.send_message("うん。モトジムカーナが趣味です")
        assert memory.list_memories() == []
        memory.set_enabled(True)
        service.send_message("うん。モトジムカーナが趣味です")
        assert len(memory.list_memories()) == 1
    finally:
        service.close()
    service, memory = open_service()
    try:
        service.start_new_conversation()
        service.send_message("私の趣味を覚えてる？")
        assert "ユーザーの趣味はモトジムカーナ" in prompts[-1]
        assert len(service.get_current_messages()) == 2
        service.send_message("私の趣味はモトジムカーナです")
        assert len(memory.list_memories()) == 1
        memory.delete(memory.list_memories()[0].id)
        service.start_new_conversation()
        service.send_message("私の趣味を覚えてる？")
        assert "モトジムカーナ" not in prompts[-1]
    finally:
        service.close()


def test_cancelled_hobby_reply_does_not_save_local_memory():
    repository = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repository, MemoryConfig())
    service = ChatService(
        provider=MockProvider(response_text='{"speech":"そうなんだね。"}', chunk_size=1),
        repository=repository,
        memory_service=memory,
    )
    try:
        stream = service.stream_message("うん。モトジムカーナが趣味です")
        next(stream)
        service._invalidate_active_stream()
        list(stream)
        assert memory.last_candidates == []
        assert memory.list_memories() == []
    finally:
        service.close()
