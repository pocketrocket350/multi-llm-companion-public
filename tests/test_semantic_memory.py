from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import AssistantResponse, MemoryCategory, MemoryStatus
from multi_llm_companion.domain.models import WebSource, parse_assistant_response
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService, MemoryService
from multi_llm_companion.services.chat import ChatServiceError
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.streaming import SpeechStreamFilter


def reply(evidence, content="", **values):
    return AssistantResponse.model_validate(
        {
            "speech": "そうなんだね。",
            "memory_updates": [
                {"action": "add", "evidence": evidence, "content": content, **values}
            ],
        }
    )


@pytest.fixture
def memory():
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    yield service
    repository.close()


@pytest.mark.parametrize(
    ("text", "fact"),
    [
        ("休日は古いラジオを直して過ごしてる", "ユーザーは休日に古いラジオを修理して過ごす"),
        ("朝食はトーストとヨーグルトにしてる", "ユーザーの朝食はトーストとヨーグルト"),
        ("走る方だよ", "ユーザーはモトジムカーナを自分で走ることを楽しむ"),
        ("春の初完走は大切な思い出なんだ", "ユーザーにとって春の初完走は大切な経験"),
        ("うん。モトジムカーナが趣味です", "ユーザーの趣味はモトジムカーナ"),
    ],
)
def test_grounded_semantic_fact_does_not_require_fixed_keywords(memory, text, fact):
    saved = memory.propose(text, reply(text, fact), "one", turn=memory.prepare_turn(text))
    assert [item.content for item in saved] == [fact]
    assert saved[0].status is MemoryStatus.APPROVED


def test_empty_semantic_decision_is_not_overridden_by_legacy_rules(memory):
    text = "私は紅茶が好きです"
    response = AssistantResponse(speech="うん。", memory_updates=[], memory_candidates=[text])
    memory.propose(text, response, "one", turn=memory.prepare_turn(text))
    assert memory.list_memories() == []


@pytest.mark.parametrize(
    ("text", "evidence", "fact"),
    [
        ("そうなんだね", "架空の根拠", "ユーザーは海で泳ぐ"),
        ("うん", "前の会話の言葉", "ユーザーは登山家"),
        ("パスワード: abc123secret", "パスワード: abc123secret", "ユーザーの秘密"),
        ("青色が好み", "青色が好み", "APIキー: sk-this-must-not-be-saved"),
        ("紅茶が好きだけど覚えないで", "紅茶が好き", "ユーザーは紅茶が好き"),
    ],
)
def test_unmatched_sensitive_or_refused_evidence_is_rejected(memory, text, evidence, fact):
    memory.propose(text, reply(evidence, fact), "one", turn=memory.prepare_turn(text))
    assert memory.list_memories() == []


def seed(memory, content="ユーザーの朝食はトースト"):
    return memory._repository.create_memory(content, status=MemoryStatus.APPROVED)


@pytest.mark.parametrize("auto_approve", [True, False])
def test_semantic_correction_replaces_same_id_without_stale_fact(memory, auto_approve):
    old = seed(memory)
    memory.set_auto_approve_candidates(auto_approve)
    text = "最近はご飯に変えたよ"
    response = reply(
        text,
        "ユーザーの朝食はご飯",
        action="update",
        target_id=old.id,
        target_content=old.content,
        category="preference",
    )
    memory.propose(text, response, "one", turn=memory.prepare_turn(text))
    saved = memory.list_memories()
    assert len(saved) == 1
    assert saved[0].id == old.id
    assert saved[0].content == "ユーザーの朝食はご飯"
    assert saved[0].status is (MemoryStatus.APPROVED if auto_approve else MemoryStatus.CANDIDATE)
    context = memory.prepare_turn("朝はどうしよう").prompt
    assert old.content not in context
    assert (saved[0].content in context) is auto_approve


@pytest.mark.parametrize("operation", ["edit", "delete", "disable_enable", "approval"])
def test_memory_management_invalidates_pending_updates(memory, operation):
    old = seed(memory)
    text = "朝はご飯に変えたよ"
    turn = memory.prepare_turn(text)
    if operation == "edit":
        memory.update(old.id, "ユーザーの朝食はおかゆ", MemoryCategory.PREFERENCE)
    elif operation == "delete":
        memory.delete(old.id)
    elif operation == "disable_enable":
        memory.set_enabled(False)
        memory.set_enabled(True)
    else:
        memory.set_auto_approve_candidates(False)
    before = memory.list_memories()
    memory.propose(
        text,
        reply(
            text,
            "ユーザーの朝食はご飯",
            action="update",
            target_id=old.id,
            target_content=old.content,
        ),
        "one",
        turn=turn,
    )
    assert memory.list_memories() == before


@pytest.mark.parametrize(("target_id", "target_content"), [(9999, "other"), (1, "違う旧内容")])
def test_stale_target_cannot_modify_memory(memory, target_id, target_content):
    seed(memory)
    text = "朝食はご飯に変えたよ"
    turn = memory.prepare_turn(text)
    before = memory.list_memories()
    memory.propose(
        text,
        reply(
            text, "朝食はご飯", action="update", target_id=target_id, target_content=target_content
        ),
        "one",
        turn=turn,
    )
    assert memory.list_memories() == before


@pytest.mark.parametrize("text", ["朝食のことは忘れて", "朝食の話もしたね"])
def test_delete_requires_current_forget_request_and_matching_target(memory, text):
    old = seed(memory)
    memory.propose(
        text,
        reply(text, action="delete", target_id=old.id, target_content=old.content),
        "one",
        turn=memory.prepare_turn(text),
    )
    assert (memory.list_memories() == []) is ("忘れて" in text)


def test_semantic_context_supports_unrelated_wording_and_is_bounded(memory):
    for index in range(45):
        seed(memory, f"ユーザーは収集品{index:02}を整理する")
    seed(memory, "ユーザーは休日に古いラジオを修理して過ごす")
    seed(memory, "未承認の事実")
    memory._repository.update_memory(memory.list_memories()[0].id, status=MemoryStatus.CANDIDATE)
    turn = memory.prepare_turn("週末は何をしようかな")
    assert "古いラジオ" in turn.prompt
    assert "未承認の事実" not in turn.prompt
    assert len(turn.memories) == 40
    assert len(memory.last_context_memories) == 40
    memory.set_enabled(False)
    assert memory.prepare_turn("週末は？").memories == ()
    assert "古いラジオ" not in memory.prepare_turn("週末は？").prompt


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        [{"action": "erase", "evidence": "うん"}],
        [{"action": "add", "evidence": ""}],
        [{"action": "add", "evidence": "うん", "target_id": True}],
    ],
)
def test_malformed_update_keeps_speech_without_enabling_legacy_fallback(value):
    response = parse_assistant_response(json.dumps({"speech": "返答", "memory_updates": value}))
    assert response.speech == "返答"
    assert response.memory_updates == []


def test_response_storage_compatibility_and_streaming_never_speak_metadata(tmp_path):
    path = tmp_path / "isolated.db"
    repo = SQLiteConversationRepository(path)
    conversation = repo.create_conversation("隔離")
    response = reply("朝食はご飯", "ユーザーの朝食はご飯")
    repo.add_message(conversation.id, "assistant", response.speech, response=response)
    repo.close()
    repo = SQLiteConversationRepository(path)
    try:
        restored = repo.list_messages(conversation.id)[0].response
        assert restored == response
        assert (
            parse_assistant_response('{"speech":"旧回答","memory_candidates":[]}').memory_updates
            is None
        )
        stream = SpeechStreamFilter()
        spoken = "".join(stream.feed(part) for part in response.model_dump_json())
        spoken += stream.finish(response.speech)
        assert spoken == response.speech
    finally:
        repo.close()


def test_search_history_does_not_block_new_personal_facts_or_cross_conversation_recall(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "memory.db")
    memory = MemoryService(repo, MemoryConfig())
    prompts = []
    text = "この記事も面白いね。私は休日に古いラジオを直して過ごしてる"

    def responder(messages):
        prompts.append(messages[0].content)
        if messages[-1].content == text:
            return reply(
                "私は休日に古いラジオを直して過ごしてる", "ユーザーは休日に古いラジオを修理する"
            ).model_dump_json()
        return '{"speech":"ラジオを直す時間をとるのもよさそうだね。","memory_updates":[]}'

    service = ChatService(
        provider=MockProvider(responder=responder), repository=repo, memory_service=memory
    )
    try:
        conversation = service.start_new_conversation()
        repo.add_message(
            conversation.id,
            "assistant",
            "外部の記事です",
            response=AssistantResponse(
                speech="外部の記事です",
                sources=[
                    WebSource(
                        title="記事", url="https://example.com/", retrieved_at=datetime.now(UTC)
                    )
                ],
            ),
        )
        service.send_message(text)
        assert len(memory.list_memories()) == 1
        service.start_new_conversation()
        service.send_message("週末は何をしようかな")
        assert "古いラジオ" in prompts[-1]
        assert "外部の記事です" not in prompts[-1]
        assert len(service.get_current_messages()) == 2
    finally:
        service.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_cancelled_response_never_applies_semantic_updates(streaming):
    repo = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repo, MemoryConfig())
    text = "朝食はトーストにしてる"
    service = None

    def responder(messages):
        if not streaming:
            service._invalidate_active_stream()
        return reply(text, "ユーザーの朝食はトースト").model_dump_json()

    service = ChatService(
        provider=MockProvider(responder=responder, chunk_size=1),
        repository=repo,
        memory_service=memory,
    )
    try:
        if streaming:
            stream = service.stream_message(text)
            next(stream)
            service._invalidate_active_stream()
            list(stream)
        else:
            with pytest.raises(ChatServiceError):
                service.send_message(text)
        assert memory.list_memories() == []
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()


def test_external_reply_drops_all_memory_metadata(memory):
    repo = memory._repository
    service = ChatService(provider=MockProvider(), repository=repo, memory_service=memory)
    try:
        conversation = service.start_new_conversation()
        response = service._save_assistant_response(
            conversation.id,
            "この記事について話そう",
            reply("この記事", "ユーザーは記者"),
            propose_memory=False,
        )
        assert response.memory_updates == []
        assert repo.list_messages(conversation.id)[0].response.memory_updates == []
        assert memory.list_memories() == []
    finally:
        service.close()


@pytest.mark.parametrize(
    ("text", "evidence", "fact"),
    [
        (
            "弟はモトジムカーナが趣味です",
            "モトジムカーナが趣味です",
            "ユーザーはモトジムカーナが趣味",
        ),
        ("仮に私は猫が好きです", "私は猫が好きです", "ユーザーは猫が好き"),
        ("『私は猫が好きです』と弟が言った", "私は猫が好きです", "ユーザーは猫が好き"),
        ("この記事では私は猫が好きと書かれています", "私は猫が好き", "ユーザーは猫が好き"),
        ("画像には私は猫が好きと書いてある", "私は猫が好き", "ユーザーは猫が好き"),
        ("私は猫が好きではありません", "私は猫が好き", "ユーザーは猫が好き"),
        ("今日は疲れた", "疲れた", "ユーザーは疲れている"),
        ("一緒に話そう", "一緒に話そう", "ユーザーとAIは恋人の関係"),
        ("私は本を読んでいる", "は", "ユーザーは作家"),
    ],
)
def test_obvious_source_confusion_is_rejected_even_if_model_proposes_it(
    memory, text, evidence, fact
):
    memory.propose(text, reply(evidence, fact), "one", turn=memory.prepare_turn(text))
    assert memory.list_memories() == []


def test_quoted_book_title_is_not_treated_as_external_personal_claim(memory):
    text = "『銀河鉄道の夜』を繰り返し読んでる"
    fact = "ユーザーは『銀河鉄道の夜』を繰り返し読む"
    memory.propose(text, reply(text, fact), "one", turn=memory.prepare_turn(text))
    assert memory.list_memories()[0].content == fact


def test_family_information_keeps_the_correct_subject(memory):
    text = "弟は陶芸をしてるんだ"
    memory.propose(
        text, reply(text, "ユーザーの弟は陶芸をしている"), "one", turn=memory.prepare_turn(text)
    )
    assert memory.list_memories()[0].content == "ユーザーの弟は陶芸をしている"


def test_update_without_category_preserves_existing_category(memory):
    old = memory._repository.create_memory(
        "ユーザーの朝食はトースト", category=MemoryCategory.PREFERENCE, status=MemoryStatus.APPROVED
    )
    text = "朝食はご飯に変えたよ"
    memory.propose(
        text,
        reply(
            text,
            "ユーザーの朝食はご飯",
            action="update",
            target_id=old.id,
            target_content=old.content,
        ),
        "one",
        turn=memory.prepare_turn(text),
    )
    assert memory.list_memories()[0].category is MemoryCategory.PREFERENCE


@pytest.mark.parametrize("enabled", [True, False])
def test_explicit_forget_still_works_without_model_update_and_for_pending_memory(memory, enabled):
    memory._repository.create_memory("ユーザーの朝食はトースト", status=MemoryStatus.CANDIDATE)
    memory.set_enabled(enabled)
    text = "朝食のことは忘れて"
    memory.propose(
        text,
        AssistantResponse(speech="うん。", memory_updates=[]),
        "one",
        turn=memory.prepare_turn(text),
    )
    assert memory.list_memories() == []


def test_rejected_proposal_is_not_retained_in_conversation_history():
    repo = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repo, MemoryConfig())
    text = "今日は疲れた"
    provider = MockProvider(response_text=reply(text, "ユーザーは疲れている").model_dump_json())
    service = ChatService(provider=provider, repository=repo, memory_service=memory)
    try:
        service.send_message(text)
        assert service.get_current_messages()[-1].response.memory_updates == []
        assert memory.list_memories() == []
    finally:
        service.close()


@pytest.mark.parametrize(
    ("text", "evidence", "fact"),
    [
        ("走る方だよ。", "走る方", "ユーザーはモトジムカーナを自分で走る"),
        ("赤。", "赤", "ユーザーの好きな色は赤"),
    ],
)
def test_short_japanese_answers_can_ground_semantic_memory(memory, text, evidence, fact):
    memory.propose(text, reply(evidence, fact), "one", turn=memory.prepare_turn(text))
    assert memory.list_memories()[0].content == fact
