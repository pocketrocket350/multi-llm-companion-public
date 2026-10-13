from __future__ import annotations

from collections.abc import Sequence

import pytest

from multi_llm_companion.config import (
    BehaviorPolicyConfig,
    CharacterConfig,
    MemoryConfig,
    PersonaConfig,
)
from multi_llm_companion.domain import (
    AssistantResponse,
    ChatMessage,
    DialogueAct,
    DialogueMode,
    MemoryCategory,
    MemoryStatus,
)
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import (
    ChatService,
    DialogueDirector,
    MemoryService,
    ProactiveTurnInvalidatedError,
    ProactiveTurnSuppressedError,
)
from multi_llm_companion.storage import SQLiteConversationRepository


def _persona() -> PersonaConfig:
    return PersonaConfig(
        character=CharacterConfig(name="案内役"),
        behavior_policy=BehaviorPolicyConfig(),
    )


def test_proactive_turn_uses_persona_and_conversation_not_saved_memory() -> None:
    prompts: list[str] = []

    def responder(messages: Sequence[ChatMessage]) -> str:
        prompts.append(next(message.content for message in messages if message.role == "system"))
        return (
            '{"speech":"少し静かな時間が続いてるね。",'
            '"memory_candidates":["利用者は青い表紙が好き"],'
            '"dialogue_act":"acknowledge"}'
        )

    repository = SQLiteConversationRepository(":memory:")
    memories = MemoryService(repository, MemoryConfig())
    provider = MockProvider(responder=responder)
    service = ChatService(
        provider=provider,
        repository=repository,
        system_prompt="人格: 穏やかな案内役",
        memory_service=memories,
        dialogue_director=DialogueDirector(_persona()),
    )
    conversation = service.start_new_conversation()
    repository.add_message(conversation.id, "user", "青い表紙の本の話をしたい")
    repository.add_message(
        conversation.id,
        "assistant",
        "青い表紙は、棚でも少し目に留まるね。",
        response=AssistantResponse(
            speech="青い表紙は、棚でも少し目に留まるね。",
            dialogue_act="acknowledge",
            current_topic="本",
        ),
    )
    repository.create_memory(
        "利用者は青い表紙が好き",
        category=MemoryCategory.PREFERENCE,
        status=MemoryStatus.APPROVED,
        source_conversation_id=conversation.id,
    )

    response = service.generate_proactive_turn()
    stored = service.get_current_messages()

    assert response.memory_candidates == []
    assert [(message.role.value, message.content) for message in stored] == [
        ("user", "青い表紙の本の話をしたい"),
        ("assistant", "青い表紙は、棚でも少し目に留まるね。"),
        ("assistant", "少し静かな時間が続いてるね。"),
    ]
    assert len(repository.list_memories()) == 1
    assert memories.last_candidates == []
    assert memories.last_context_memories == []
    assert "【能動的な話しかけ】" in prompts[-1]
    assert "予定やリマインダーのような発言はしない" in prompts[-1]
    assert "保存済みの利用者記憶は参照しない" in prompts[-1]
    assert "【利用者が承認した参考情報】" not in prompts[-1]
    assert "【利用者の現在の承認済み記憶】" not in prompts[-1]
    service.close()


@pytest.mark.parametrize("act", [DialogueAct.CLOSE, DialogueAct.BOUNDARY])
def test_proactive_turn_does_not_restart_a_closed_topic_or_boundary(act: DialogueAct) -> None:
    repository = SQLiteConversationRepository(":memory:")

    def responder(_messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("終話・境界の直後にはLLMを呼ばない必要があります。")

    service = ChatService(
        provider=MockProvider(responder=responder),
        repository=repository,
        system_prompt="人格: 穏やかな案内役",
        dialogue_director=DialogueDirector(_persona()),
    )
    conversation = service.start_new_conversation()
    user_text = "ここで話を終える" if act is DialogueAct.CLOSE else "触れないで"
    repository.add_message(conversation.id, "user", user_text)
    repository.add_message(
        conversation.id,
        "assistant",
        "わかった。",
        response=AssistantResponse(
            speech="わかった。",
            dialogue_mode=(
                DialogueMode.CASUAL if act is DialogueAct.CLOSE else DialogueMode.BOUNDARY
            ),
            dialogue_act=act,
        ),
    )

    with pytest.raises(ProactiveTurnSuppressedError):
        service.generate_proactive_turn()

    assert [(item.role.value, item.content) for item in service.get_current_messages()] == [
        ("user", user_text),
        ("assistant", "わかった。"),
    ]
    service.close()


def test_cancelled_proactive_turn_is_not_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service: ChatService

    def responder(_messages: Sequence[ChatMessage]) -> str:
        service.invalidate_proactive_turn()
        return '{"speech":"これは保存されない。"}'

    service = ChatService(
        provider=MockProvider(responder=responder),
        repository=repository,
        system_prompt="人格: 穏やかな案内役",
    )
    conversation = service.start_new_conversation()

    with pytest.raises(ProactiveTurnInvalidatedError):
        service.generate_proactive_turn()

    assert service.get_current_messages() == []
    assert repository.get_conversation(conversation.id) is not None
    service.close()
