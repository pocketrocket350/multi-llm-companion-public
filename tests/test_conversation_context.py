from __future__ import annotations

from multi_llm_companion.config import (
    BehaviorPolicyConfig,
    CharacterConfig,
    MemoryConfig,
    PersonaConfig,
)
from multi_llm_companion.domain import AssistantResponse, ChatMessage, MemoryStatus, MessageRole
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService, DialogueDirector, MemoryService
from multi_llm_companion.services.conversation_context import ConversationContextService
from multi_llm_companion.storage import SQLiteConversationRepository


def test_context_keeps_recent_messages_and_summarizes_older_ones() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=300)
    messages = [
        ChatMessage(role=MessageRole.USER, content="昔の質問"),
        ChatMessage(role=MessageRole.ASSISTANT, content="昔の回答"),
        ChatMessage(role=MessageRole.USER, content="最近の質問"),
        ChatMessage(role=MessageRole.ASSISTANT, content="最近の回答"),
    ]

    context = service.build(messages)

    assert [message.content for message in context.messages] == ["最近の質問", "最近の回答"]
    assert "昔の質問" in context.summary
    assert context.status_text == "要点 + 直近 2 件（全 4 件）"


def test_context_keeps_a_user_correction_when_old_history_is_truncated() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=100)
    messages = [
        ChatMessage(role=MessageRole.USER, content="以前の長い話題 " * 20),
        ChatMessage(role=MessageRole.USER, content="実はコーヒーは飲める"),
        ChatMessage(role=MessageRole.ASSISTANT, content="了解"),
        ChatMessage(role=MessageRole.USER, content="紅茶が一番好き"),
    ]

    context = service.build(messages)

    assert "【この会話で利用者が訂正した情報】" in context.summary
    assert "実はコーヒーは飲める" in context.summary


def test_context_keeps_a_long_yappari_correction_within_summary_limit() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=100)
    messages = [
        ChatMessage(role=MessageRole.USER, content="前の話題 " * 30),
        ChatMessage(role=MessageRole.USER, content="やっぱりコーヒーも飲める。" + "補足 " * 50),
        ChatMessage(role=MessageRole.ASSISTANT, content="了解"),
        ChatMessage(role=MessageRole.USER, content="紅茶が一番好き"),
    ]

    context = service.build(messages)

    assert "やっぱりコーヒーも飲める" in context.summary


def test_context_keeps_a_do_not_remember_boundary_when_old_history_is_truncated() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=100)
    messages = [
        ChatMessage(role=MessageRole.USER, content="前の話題 " * 30),
        ChatMessage(role=MessageRole.USER, content="覚えないで、今日は眠いだけ"),
        ChatMessage(role=MessageRole.ASSISTANT, content="了解"),
        ChatMessage(role=MessageRole.USER, content="別の話をしよう"),
    ]

    context = service.build(messages)

    assert "【利用者が指定した記憶の境界】" in context.summary
    assert "覚えないで、今日は眠いだけ" in context.summary


def test_context_keeps_correction_and_memory_boundary_together_when_both_are_long() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=100)
    messages = [
        ChatMessage(role=MessageRole.USER, content="前の話題 " * 30),
        ChatMessage(
            role=MessageRole.USER,
            content="訂正: 会議は金曜ではなく木曜です。" + "補足 " * 40,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="覚えないで、今日は眠いだけ。" + "補足 " * 40,
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="了解"),
        ChatMessage(role=MessageRole.USER, content="別の話をしよう"),
    ]

    context = service.build(messages)

    assert "会議は金曜ではなく木曜" in context.summary
    assert "覚えないで、今日は眠いだけ" in context.summary


def test_new_prompt_order_uses_state_before_history_truncation() -> None:
    captured: list[str] = []

    def responder(messages: list[ChatMessage]) -> str:
        captured.append(messages[0].content)
        return '{"speech":"回答","dialogue_mode":"casual","dialogue_act":"answer"}'

    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation("順序")
    repository.add_message(conversation.id, "user", "昔の猫の話")
    repository.add_message(
        conversation.id,
        "assistant",
        "続きを聞きました。どう思う？",
        response=AssistantResponse(
            speech="続きを聞きました。どう思う？",
            dialogue_mode="supporting",
            dialogue_act="ask",
            current_topic="猫の話",
            open_loop="写真の話を聞く",
        ),
    )
    repository.add_message(conversation.id, "user", "途中")
    repository.add_message(conversation.id, "assistant", "了解")
    memory = repository.create_memory("ユーザーは猫が好き")
    repository.update_memory(memory.id, status=MemoryStatus.APPROVED)
    persona = PersonaConfig(
        character=CharacterConfig(name="案内役"),
        behavior_policy=BehaviorPolicyConfig(),
    )
    service = ChatService(
        provider=MockProvider(responder=responder),
        repository=repository,
        system_prompt="STATIC",
        engine_contract="ENGINE",
        dialogue_director=DialogueDirector(persona),
        memory_service=MemoryService(repository, MemoryConfig()),
        conversation_context=ConversationContextService(
            max_recent_messages=2,
            summary_max_characters=300,
        ),
    )
    service.load_conversation(conversation.id)

    service.send_message("猫の話の続き")

    prompt = captured[0]
    positions = [
        prompt.index("STATIC"),
        prompt.index("【今回の会話モード"),
        prompt.index("【直前のDialogueState】"),
        prompt.index("【この会話の過去の要点】"),
        prompt.index("【利用者の現在の承認済み記憶】"),
        prompt.index("ENGINE"),
    ]
    assert positions == sorted(positions)
    assert "open_loop: なし" in prompt  # 直近のresponse_json欠損は既定値として扱う。
    service.close()


def test_long_prefix_does_not_remove_correction_or_boundary() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=160)
    messages = [
        ChatMessage(role="user", content="前置き " * 80 + "訂正: 会議は木曜です。"),
        ChatMessage(role="user", content="前置き " * 80 + "覚えないで、今日は眠いだけ。"),
        ChatMessage(role="assistant", content="了解"),
        ChatMessage(role="user", content="別の話"),
    ]
    summary = service.build(messages).summary
    assert "訂正: 会議は木曜" in summary
    assert "覚えないで、今日は眠いだけ" in summary


def test_general_boundaries_remain_after_compression() -> None:
    service = ConversationContextService(max_recent_messages=2, summary_max_characters=350)
    messages = [
        *[ChatMessage(role="user", content="以前の普通の話 " * 40) for _ in range(6)],
        ChatMessage(role="user", content="質問しないで。提案も今はいらない。"),
        ChatMessage(role="user", content="恋人のように扱わないで。仕事の話には触れないで。"),
        *[ChatMessage(role="user", content="普通の話 " * 40) for _ in range(8)],
        ChatMessage(role="assistant", content="了解"),
        ChatMessage(role="user", content="紅茶を飲む"),
    ]
    summary = service.build(messages).summary
    for boundary in (
        "質問しないで",
        "提案も今はいらない",
        "恋人のように扱わないで",
        "仕事の話には触れないで",
    ):
        assert boundary in summary
