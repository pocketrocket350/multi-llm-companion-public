from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import AssistantResponse, MemoryCategory, MemoryStatus
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository


@pytest.mark.parametrize("auto_approve", [True, False])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("うん。モトジムカーナが趣味です", "ユーザーの趣味はモトジムカーナ"),
        ("はい、私の趣味は読書です。おすすめある？", "ユーザーの趣味は読書"),
        ("写真を撮ることが趣味なんだよ。", "ユーザーの趣味は写真を撮ること"),
        ("趣味は模型作り", "ユーザーの趣味は模型作り"),
        ("僕は登山を趣味にしています。", "ユーザーの趣味は登山"),
        ("モトジムカーナは趣味です", "ユーザーの趣味はモトジムカーナ"),
        ("登山を趣味にしてます", "ユーザーの趣味は登山"),
        ("うん、紅茶が好きなんだ。", "ユーザーは紅茶が好き"),
        ("私はたらこ料理が好きです", "ユーザーはたらこ料理が好き"),
        ("私はさかなが好きです", "ユーザーはさかなが好き"),
        ("私はおはぎが好きです", "ユーザーはおはぎが好き"),
        ("おはぎが好きです", "ユーザーはおはぎが好き"),
        ("はがき集めを趣味にしています", "ユーザーの趣味ははがき集め"),
        ("新しいものを作ることが趣味です", "ユーザーの趣味は新しいものを作ること"),
        ("翻訳小説が好きです", "ユーザーは翻訳小説が好き"),
        ("はちみつ系ののど飴が好きです。", "ユーザーははちみつ系ののど飴が好き"),
        ("ホラー映画が苦手なんです", "ユーザーはホラー映画が苦手"),
    ],
)
def test_natural_personal_declaration_without_model_candidate(text, expected, auto_approve):
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig(auto_approve_candidates=auto_approve))
        service.propose(text, AssistantResponse(speech="そうなんだね。"), "one")
        saved = service.list_memories()
        assert [item.content for item in saved] == [expected]
        assert saved[0].category is MemoryCategory.PREFERENCE
        assert saved[0].status is (
            MemoryStatus.APPROVED if auto_approve else MemoryStatus.CANDIDATE
        )
        service.propose(text, AssistantResponse(speech="うん。"), "one")
        assert len(service.list_memories()) == 1
    finally:
        repository.close()


@pytest.mark.parametrize(
    "text",
    [
        "モトジムカーナが趣味ですか？",
        "趣味はモトジムカーナですか",
        "趣味は読書かな",
        "趣味は読書らしい",
        "モトジムカーナが趣味かもしれません。",
        "モトジムカーナが趣味なら楽しそう。",
        "もしモトジムカーナが趣味です。",
        "弟はモトジムカーナが趣味です。",
        "太郎の趣味はモトジムカーナです。",
        "太郎はモトジムカーナが趣味です。",
        "モトジムカーナが弟の趣味です。",
        "上司がモトジムカーナを趣味にしている",
        "会社の上司がモトジムカーナを趣味にしている",
        "友人の発言です。モトジムカーナが趣味です。",
        "次の例文を読んで。趣味はモトジムカーナです。",
        "記事によると、モトジムカーナが趣味です。",
        "「モトジムカーナが趣味です」と言っていた。",
        "うん。モトジムカーナが趣味ではありません。",
        "趣味は読書です。覚えないで。",
        "今だけ、趣味は読書です。",
        "趣味はAPIキー: sk-abcdefghijklmnop です。",
    ],
)
def test_natural_declaration_does_not_infer_a_personal_fact(text):
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig())
        assert service.propose(text, AssistantResponse(speech="うん。"), "one") == []
        assert service.list_memories() == []
    finally:
        repository.close()


def test_local_declaration_survives_rejected_model_candidate_and_word_order_changes():
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig())
        service.propose(
            "うん。モトジムカーナが趣味です",
            AssistantResponse(speech="うん。", memory_candidates=["今日は眠い"]),
            "one",
        )
        assert len(service.list_memories()) == 1
        service.propose(
            "私の趣味はモトジムカーナです",
            AssistantResponse(speech="うん。", memory_candidates=["利用者はモトジムカーナが趣味"]),
            "one",
        )
        assert len(service.list_memories()) == 1
    finally:
        repository.close()


@pytest.mark.parametrize("auto_approve", [True, False])
@pytest.mark.parametrize("subject", ["趣味", "私の趣味"])
def test_hobby_correction_and_forgetting(auto_approve, subject):
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig())
        service.propose("うん。読書が趣味です", AssistantResponse(speech="うん。"), "one")
        service.set_auto_approve_candidates(auto_approve)
        service.propose(
            f"訂正。{subject}は読書ではなく登山です。", AssistantResponse(speech="うん。"), "one"
        )
        saved = service.list_memories()
        assert [item.content for item in saved] == ["ユーザーの趣味は登山"]
        assert saved[0].status is (
            MemoryStatus.APPROVED if auto_approve else MemoryStatus.CANDIDATE
        )
        assert "読書" not in service.context_for("私の趣味は？")
        service.propose("登山のことを忘れて", AssistantResponse(speech="うん。"), "one")
        assert service.list_memories() == []
    finally:
        repository.close()


@pytest.mark.parametrize("auto_approve", [True, False])
def test_personal_preference_before_question_is_saved_without_model_candidate(auto_approve):
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=auto_approve))
    text = "私はモトジムカーナが大好きです。\nモトジムカーナって知っていますか？"
    try:
        service.propose(text, AssistantResponse(speech="知ってるよ。"), "one")
        saved = service.list_memories()
        assert len(saved) == 1
        assert saved[0].content == "ユーザーはモトジムカーナが大好き"
        assert saved[0].category is MemoryCategory.PREFERENCE
        assert saved[0].status is (
            MemoryStatus.APPROVED if auto_approve else MemoryStatus.CANDIDATE
        )
        service.propose(text, AssistantResponse(speech="うん。"), "one")
        assert len(service.list_memories()) == 1
    finally:
        repository.close()


@pytest.mark.parametrize(
    "text",
    [
        "私はモトジムカーナが好きですか？",
        "私はモトジムカーナが好きかもしれません。",
        "弟はモトジムカーナが大好きです。",
        "友人の発言です。\n私はモトジムカーナが大好きです。",
        "次の例文を読んで。\n私はモトジムカーナが大好きです。",
        "「私はモトジムカーナが大好きです」と弟が言った。",
        "記事によると、私はモトジムカーナが大好きです。",
        "仮に私はモトジムカーナが大好きです。",
        "私はモトジムカーナが好きではありません。",
        "私はモトジムカーナが大好きです。覚えないで。",
        "私はAPIキー: sk-abcdefghijklmnop が好きです。",
    ],
)
def test_local_preference_fallback_does_not_infer_or_save_unsafe_statements(text):
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig())
        assert service.propose(text, AssistantResponse(speech="うん。"), "one") == []
        assert service.list_memories() == []
    finally:
        repository.close()


def test_local_preference_respects_memory_disabled_and_survives_reopen(tmp_path):
    path = tmp_path / "memory.db"
    repository = SQLiteConversationRepository(path)
    service = MemoryService(repository, MemoryConfig(enabled=False))
    text = "私はモトジムカーナが大好きです。"
    assert service.propose(text, AssistantResponse(speech="うん。"), "one") == []
    service.set_enabled(True)
    service.propose(text, AssistantResponse(speech="うん。"), "one")
    repository.close()
    repository = SQLiteConversationRepository(path)
    try:
        service = MemoryService(repository, MemoryConfig())
        assert "モトジムカーナが大好き" in service.context_for("モトジムカーナの話をしよう")
        service.propose("モトジムカーナの好みは忘れて", AssistantResponse(speech="うん。"), "one")
        assert service.list_memories() == []
    finally:
        repository.close()


@pytest.mark.parametrize("auto_approve", [True, False])
def test_local_preference_correction_replaces_old_fact_without_model_candidate(auto_approve):
    repository = SQLiteConversationRepository(":memory:")
    try:
        service = MemoryService(repository, MemoryConfig())
        service.propose("私は猫が好きです。", AssistantResponse(speech="うん。"), "one")
        service.set_auto_approve_candidates(auto_approve)
        service.propose(
            "訂正。私は猫が好きではなく犬が好きです。", AssistantResponse(speech="うん。"), "one"
        )
        memories = service.list_memories()
        assert len(memories) == 1
        assert memories[0].content == "ユーザーは犬が好き"
        assert memories[0].status is (
            MemoryStatus.APPROVED if auto_approve else MemoryStatus.CANDIDATE
        )
        assert "猫が好き" not in service.context_for("猫の好み")
    finally:
        repository.close()


def test_explicit_calling_correction_replaces_the_old_value_even_with_model_candidate() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose("私をミナと呼んで", AssistantResponse(speech="了解"), "one")
    service.propose(
        "訂正。私をナツと呼んで。ミナという呼び名は使わないで。",
        AssistantResponse(
            speech="了解", memory_candidates=["利用者はナツという名前で呼んでほしい"]
        ),
        "one",
    )
    assert len(service.list_memories()) == 1
    context = service.context_for("私を何と呼ぶ？")
    assert "ナツ" in context
    assert "ミナ" not in context
    repository.close()


def test_explicit_plan_correction_updates_only_the_identified_plan() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose("会議は金曜です。覚えておいて", AssistantResponse(speech="了解"), "one")
    service.propose("通院は金曜です。覚えておいて", AssistantResponse(speech="了解"), "one")
    service.propose(
        "訂正。会議は金曜ではなく木曜です。",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーの予定: 会議は木曜です"]),
        "one",
    )
    contents = [item.content for item in service.list_memories()]
    assert len(contents) == 2
    assert any("会議は木曜" in item for item in contents)
    assert not any("会議は金曜" in item for item in contents)
    assert any("通院は金曜" in item for item in contents)
    repository.close()


@pytest.mark.parametrize(
    ("user", "candidate"),
    [
        ("私は猫が好きではありません", "ユーザーは猫が好き"),
        ("私は猫が好きではありません", "ユーザーは猫が好きではない"),
        ("架空の話で、仕事は宇宙飛行士という設定です", "ユーザーの仕事は宇宙飛行士"),
        ("今だけコーヒーが好きという役を演じます", "ユーザーはコーヒーが好き"),
    ],
)
def test_negative_fictional_and_temporary_user_facts_are_not_saved(user, candidate) -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    assert (
        service.propose(
            user, AssistantResponse(speech="了解", memory_candidates=[candidate]), "one"
        )
        == []
    )
    assert service.list_memories() == []
    repository.close()


def test_forget_removes_a_pending_candidate_including_single_character_preference() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))
    service.propose(
        "好きな色は青",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは青が好き"]),
        "one",
    )
    service.propose("青のことを忘れて", AssistantResponse(speech="了解"), "one")
    assert service.list_memories() == []
    repository.close()


def test_disable_immediately_clears_previously_used_memory_and_general_preference_query() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose(
        "紅茶が好き",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは紅茶が好き"]),
        "one",
    )
    assert "紅茶" in service.context_for("私の飲み物の好みは？")
    assert service.last_context_memories
    service.set_enabled(False)
    assert service.last_context_memories == []
    repository.close()


def test_name_in_another_memory_does_not_restore_an_obsolete_calling() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose(
        "私をミナと呼んで",
        AssistantResponse(speech="了解", memory_candidates=["利用者の呼び名は「ミナ」"]),
        "one",
    )
    service.propose(
        "紅茶が好き",
        AssistantResponse(speech="了解", memory_candidates=["ミナは紅茶が好き"]),
        "one",
    )
    service.propose("私をナツと呼んで", AssistantResponse(speech="了解"), "one")
    assert all("ミナ" not in item.content for item in service.list_memories())
    assert "紅茶" in service.context_for("紅茶の好みは？")
    repository.close()


@pytest.mark.parametrize(
    "user", ["会議の予定を覚えている？", "会議の予定を覚えてる？", "会議の予定は保存してある？"]
)
def test_question_about_memory_does_not_create_a_plan(user) -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    assert service.propose(user, AssistantResponse(speech="分からない"), "one") == []
    repository.close()


def test_correction_awaits_approval_without_reusing_the_old_value() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose("私をミナと呼んで", AssistantResponse(speech="了解"), "one")
    service.set_auto_approve_candidates(False)
    changed = service.propose("私をナツと呼んで", AssistantResponse(speech="了解"), "one")
    assert len(changed) == 1
    assert changed[0].status is MemoryStatus.CANDIDATE
    assert service.context_for("私の呼び名は？") == ""
    service.approve(changed[0].id, changed[0].content, changed[0].category)
    assert "ナツ" in service.context_for("私の呼び名は？")
    repository.close()


def test_forget_confirmation_includes_pending_candidate_but_not_fictional_deletion() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))
    service.propose(
        "紅茶が好き", AssistantResponse(speech="了解", memory_candidates=["紅茶が好き"]), "one"
    )
    assert "対象を1件特定" in service.context_for("紅茶のことを忘れて")
    assert service.forget_from("架空の話です。紅茶を忘れて", "one") == []
    assert service.forget_from("忘れてと言った内容は残っていない？", "one") == []
    assert len(service.list_memories()) == 1
    repository.close()


def test_preference_correction_uses_the_explicit_old_and_new_values() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    service.propose(
        "好きな色は青です",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは青が好き"]),
        "one",
    )
    service.propose(
        "訂正。好きな色は青ではなく緑です。",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは緑が好き"]),
        "one",
    )
    assert [item.content for item in service.list_memories()] == ["ユーザーは緑が好き"]
    repository.close()


@pytest.mark.parametrize(
    ("user", "candidate"),
    [
        ("私をミナと呼んでいない", "ユーザーの呼び方はミナと呼んでほしい"),
        ("毎週映画を見ることはありません", "ユーザーとは毎週映画の話をする"),
        ("私は毎日紅茶を飲んでいない", "ユーザーは毎日紅茶を飲む習慣がある"),
    ],
)
def test_negative_facts_cannot_bypass_guard_via_special_memory_categories(user, candidate) -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    assert (
        service.propose(
            user, AssistantResponse(speech="了解", memory_candidates=[candidate]), "one"
        )
        == []
    )
    repository.close()


def test_ambiguous_calling_records_do_not_accumulate_or_get_reused() -> None:
    repository = SQLiteConversationRepository(":memory:")
    for name in ("ミナ", "アキラ"):
        repository.create_memory(
            f"ユーザーの呼び方は{name}と呼んでほしい",
            category=MemoryCategory.PROFILE,
            status=MemoryStatus.APPROVED,
        )
    service = MemoryService(repository, MemoryConfig())
    assert service.propose("私をナツと呼んで", AssistantResponse(speech="了解"), "one") == []
    assert len(service.list_memories()) == 2
    context = service.context_for("私の呼び名は？")
    assert "ミナ" not in context and "アキラ" not in context
    repository.close()


def test_candidate_requires_approval_when_auto_approval_is_disabled() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))
    candidates = service.propose(
        "私は辛い食べ物が苦手です",
        AssistantResponse(speech="了解です", memory_candidates=["ユーザーは辛い食べ物が苦手"]),
        "conversation-id",
    )

    assert len(candidates) == 1
    assert candidates[0].status is MemoryStatus.CANDIDATE
    assert service.context_for("辛くない料理を教えて") == ""
    service.approve(candidates[0].id, candidates[0].content, candidates[0].category)
    assert "辛い食べ物が苦手" in service.context_for("辛くない料理を教えて")
    repository.close()


def test_explicit_calling_is_candidate_when_auto_approval_is_disabled() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))

    saved = service.propose(
        "私をタカシと呼んで",
        AssistantResponse(speech="わかった、タカシ。これからそう呼ぶね。"),
        "conversation-id",
    )

    assert [memory.category for memory in saved] == [MemoryCategory.PROFILE]
    assert [memory.status for memory in saved] == [MemoryStatus.CANDIDATE]
    service.approve(saved[0].id, saved[0].content, saved[0].category)
    assert "タカシ" in service.context_for("私の名前は？")
    repository.close()


def test_explicit_plan_reminder_is_saved_when_model_omits_candidate() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))

    saved = service.propose(
        "午前5時に起床します。声をかけてください。",
        AssistantResponse(speech="正確な時刻のアラームはできません。"),
        "conversation-id",
    )

    assert [memory.content for memory in saved] == ["ユーザーの予定: 午前5時に起床します"]
    assert [memory.category for memory in saved] == [MemoryCategory.PLAN]
    assert [memory.status for memory in saved] == [MemoryStatus.CANDIDATE]
    repository.close()


def test_unrequested_time_is_not_saved_as_a_plan() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    saved = service.propose(
        "午前5時に起床します。",
        AssistantResponse(speech="早起きですね。"),
        "conversation-id",
    )

    assert saved == []
    repository.close()


def test_existing_candidate_stays_out_of_context_until_approved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    legacy = repository.create_memory("ユーザーは紅茶が好き", category=MemoryCategory.PREFERENCE)

    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))

    assert repository.list_memories()[0].id == legacy.id
    assert repository.list_memories()[0].status is MemoryStatus.CANDIDATE
    assert service.context_for("紅茶を淹れたい") == ""
    repository.close()


def test_forget_deletes_only_one_identified_memory() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    tea = repository.create_memory("ユーザーは紅茶が好き", status=MemoryStatus.APPROVED)
    repository.create_memory("ユーザーは猫が好き", status=MemoryStatus.APPROVED)

    deleted = service.forget_from("紅茶のことを忘れて", "conversation-id")

    assert [memory.id for memory in deleted] == [tea.id]
    assert [memory.content for memory in service.list_memories()] == ["ユーザーは猫が好き"]
    assert service.forget_from("忘れて", "conversation-id") == []
    assert len(service.list_memories()) == 1
    repository.close()


def test_recent_forget_deletes_only_the_single_memory_from_this_conversation() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    recent = repository.create_memory(
        "ユーザーは紅茶が好き",
        status=MemoryStatus.APPROVED,
        source_conversation_id="conversation-id",
    )
    repository.create_memory("ユーザーは猫が好き", status=MemoryStatus.APPROVED)

    deleted = service.forget_from("さっきのことを忘れて", "conversation-id")

    assert [memory.id for memory in deleted] == [recent.id]
    assert [memory.content for memory in service.list_memories()] == ["ユーザーは猫が好き"]
    repository.close()


def test_calling_boundary_habit_and_preference_are_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))
    cases = (
        ("私をタカシと呼んで", "ユーザーの呼び方はタカシと呼んでほしい"),
        ("恋人の話は本人から話すまで触れないで", "恋人の話は触れないでほしい"),
        ("毎週日曜に映画の話をする", "ユーザーとは毎週日曜に映画の話をする"),
        ("好きな色は青", "ユーザーは青が好き"),
    )

    for user_text, candidate in cases:
        saved = service.propose(
            user_text,
            AssistantResponse(speech="了解です", memory_candidates=[candidate]),
            "conversation-id",
        )
        assert len(saved) == 1
        assert saved[0].status is MemoryStatus.CANDIDATE
    repository.close()


def test_one_off_chat_and_ungrounded_guess_are_not_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    assert (
        service.propose(
            "今日は晴れですね",
            AssistantResponse(
                speech="そうですね", memory_candidates=["ユーザーは今日は晴れが好き"]
            ),
            "conversation-id",
        )
        == []
    )
    assert (
        service.propose(
            "雑談をしよう",
            AssistantResponse(speech="いいですね", memory_candidates=["ユーザーは青が好き"]),
            "conversation-id",
        )
        == []
    )
    assert service.list_memories() == []
    repository.close()


def test_secret_like_candidate_and_do_not_remember_request_are_not_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    response = AssistantResponse(
        speech="了解です",
        memory_candidates=["APIキー: sk-this-must-not-be-saved", "ユーザーは青が好き"],
    )

    assert service.propose("これは覚えないで", response, "conversation-id") == []
    assert service.list_memories() == []
    assert len(service.propose("好きな色は青", response, "conversation-id")) == 1
    repository.close()


def test_memory_service_avoids_near_duplicate_candidates() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    response = AssistantResponse(
        speech="了解です",
        memory_candidates=["ユーザーは青色が好きです", "ユーザーは青色が好き"],
    )

    candidates = service.propose("好きな色は青です", response, "conversation-id")

    assert len(candidates) == 1
    repository.close()


def test_candidate_lifecycle_supports_approval_correction_discard_and_delete() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(auto_approve_candidates=False))
    candidate = service.propose(
        "ユーザーは紅茶が好き",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは紅茶が好き"]),
        "conversation-id",
    )[0]

    approved = service.approve(candidate.id, "ユーザーは紅茶が好き", MemoryCategory.PREFERENCE)
    corrected = service.update(approved.id, "ユーザーは緑茶が好き", MemoryCategory.PREFERENCE)

    assert corrected.status is MemoryStatus.APPROVED
    context = service.context_for("緑茶について話したい")
    assert "緑茶が好き" in context
    assert "紅茶が好き" not in context

    discarded = service.propose(
        "ユーザーは猫が好き",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは猫が好き"]),
        "conversation-id",
    )[0]
    assert service.delete(discarded.id) is True
    assert service.delete(corrected.id) is True
    assert service.list_memories() == []
    repository.close()


def test_memory_is_auto_approved_by_default() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    saved = service.propose(
        "ユーザーは紅茶が好き",
        AssistantResponse(speech="了解", memory_candidates=["ユーザーは紅茶が好き"]),
        "conversation-id",
    )

    assert saved[0].status is MemoryStatus.APPROVED
    assert "紅茶が好き" in service.context_for("紅茶について話したい")
    repository.close()


def test_transient_emotion_and_inferred_relationship_are_not_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    response = AssistantResponse(
        speech="了解です",
        memory_candidates=[
            "ユーザーは今日は悲しい",
            "ユーザーは疲れている",
            "ユーザーとの信頼が深まった",
            "ユーザーの好感度が高い",
            "ユーザーはアシスタントへ依存している",
        ],
    )

    assert service.propose("今日は悲しいし疲れている", response, "conversation-id") == []
    assert service.list_memories() == []
    repository.close()


def test_transient_state_without_user_subject_is_not_saved() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    transient_states = [
        "今日は悲しい",
        "疲れている",
        "不安",
        "落ち込んでいる",
        "寂しい",
        "つらい",
        "しんどい",
        "怒っている",
        "嬉しい",
        "眠い",
        "メンタルが不調",
    ]

    candidates = service.propose(
        "今日は調子について話した",
        AssistantResponse(speech="了解です", memory_candidates=transient_states),
        "conversation-id",
    )

    assert candidates == []
    assert service.list_memories() == []
    repository.close()


def test_persistent_preference_with_emotional_description_is_not_overfiltered() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    candidates = service.propose(
        "悲しい曲が好きです",
        AssistantResponse(speech="了解です", memory_candidates=["ユーザーは悲しい曲が好き"]),
        "conversation-id",
    )

    assert [item.content for item in candidates] == ["ユーザーは悲しい曲が好き"]
    repository.close()


def test_unapproved_relationship_labels_are_denied_by_default() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    relationship_labels = [
        "ユーザーとアシスタントは恋人同士",
        "ユーザーとアシスタントは親友",
        "ユーザーはアシスタントのパートナー",
        "ユーザーとアシスタントは相棒",
        "ユーザーとアシスタントは特別な関係",
        "ユーザーとの絆がある",
        "ユーザーとの信頼がある",
        "ユーザーとの親密さが増した",
        "ユーザーへの愛情がある",
        "ユーザーはアシスタントへ依存している",
        "ユーザーの好感度が高い",
    ]

    candidates = service.propose(
        "今日は雑談した",
        AssistantResponse(speech="了解です", memory_candidates=relationship_labels),
        "conversation-id",
    )

    assert candidates == []
    assert service.list_memories() == []
    repository.close()


def test_broad_user_assistant_relationship_claims_are_denied() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    relationship_claims = [
        "ユーザーとアシスタントは友達",
        "利用者とAIは家族",
        "ユーザーとアシスタントは仲間",
        "ユーザーとAIは師弟",
        "利用者とアシスタントは知り合い",
        "ユーザーとAIは同居人",
        "ユーザーとアシスタントは相方",
        "ユーザーとAIは相互に大切",
        "利用者とアシスタントは仲が良い",
        "ユーザーはAIの友達",
    ]

    candidates = service.propose(
        "今日は雑談した",
        AssistantResponse(speech="了解です", memory_candidates=relationship_claims),
        "conversation-id",
    )

    assert candidates == []
    assert service.list_memories() == []
    repository.close()


def test_user_family_profile_is_not_mistaken_for_assistant_relationship() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    candidates = service.propose(
        "家族は3人です",
        AssistantResponse(speech="了解です", memory_candidates=["ユーザーの家族は3人"]),
        "conversation-id",
    )

    assert [item.content for item in candidates] == ["ユーザーの家族は3人"]
    repository.close()


def test_explicit_calling_boundary_and_shared_habit_can_be_candidates() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    calling = service.propose(
        "私をミナと呼んでください",
        AssistantResponse(
            speech="了解です",
            memory_candidates=["ユーザーの呼び方はミナと呼んでほしい"],
        ),
        "conversation-id",
    )
    boundary = service.propose(
        "家族の話は聞かないで",
        AssistantResponse(
            speech="了解です",
            memory_candidates=["ユーザーは家族の話を聞かないでほしい"],
        ),
        "conversation-id",
    )
    habit = service.propose(
        "毎週日曜に一緒に映画の話をする",
        AssistantResponse(
            speech="楽しみです",
            memory_candidates=["ユーザーとは毎週日曜に映画の話をする"],
        ),
        "conversation-id",
    )

    assert [item.content for item in (*calling, *boundary, *habit)] == [
        "ユーザーの呼び方はミナと呼んでほしい",
        "ユーザーは家族の話を聞かないでほしい",
        "ユーザーとは毎週日曜に映画の話をする",
    ]
    assert len(service.list_memories()) == 3
    repository.close()


def test_relationship_candidate_without_matching_user_evidence_is_rejected() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    response = AssistantResponse(
        speech="了解です",
        memory_candidates=[
            "ユーザーの呼び方はミナと呼んでほしい",
            "ユーザーは家族の話を聞かないでほしい",
            "ユーザーとは毎週日曜に映画の話をする",
        ],
    )

    assert service.propose("今日は映画の話をした", response, "conversation-id") == []
    repository.close()


def test_relationship_candidate_must_match_explicit_user_details() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    mismatches = (
        (
            "私をミナと呼んでください",
            "ユーザーの呼び方はアキラと呼んでほしい",
        ),
        (
            "猫の話は聞かないで",
            "ユーザーは家族の話を聞かないでほしい",
        ),
        (
            "毎日Pythonを勉強する",
            "ユーザーとは毎週日曜に映画の話をする",
        ),
        (
            "毎週Pythonを勉強する",
            "ユーザーとは毎週映画の話をする",
        ),
    )

    for user_text, candidate in mismatches:
        response = AssistantResponse(speech="了解です", memory_candidates=[candidate])
        assert service.propose(user_text, response, "conversation-id") == []

    assert service.list_memories() == []
    repository.close()


def test_grounded_relationship_fact_with_forbidden_claim_mixed_in_is_rejected() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    mixed_candidates = (
        (
            "私をミナと呼んでください",
            "ユーザーの呼び方はミナ。ユーザーとアシスタントは恋人同士",
        ),
        (
            "家族の話は聞かないで",
            "ユーザーは家族の話を聞かないでほしい。ユーザーとアシスタントは恋人同士",
        ),
        (
            "毎週日曜に一緒に映画の話をする",
            "ユーザーとは毎週日曜に映画の話をする。ユーザーとアシスタントは親友",
        ),
    )

    for user_text, candidate in mixed_candidates:
        response = AssistantResponse(speech="了解です", memory_candidates=[candidate])
        assert service.propose(user_text, response, "conversation-id") == []

    assert service.list_memories() == []
    repository.close()


def test_strict_allowlist_rejects_unrelated_or_mixed_second_clause() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    mixed_candidates = (
        (
            "私をミナと呼んでください",
            "ユーザーの呼び方はミナ。ユーザーと私は恋人",
        ),
        (
            "私をミナと呼んでください",
            "ユーザーの呼び方はミナ。今日は晴れ",
        ),
        (
            "家族の話は聞かないで",
            "ユーザーは家族の話を聞かないでほしい。猫が好き",
        ),
        (
            "毎週日曜に一緒に映画の話をする",
            "ユーザーとは毎週日曜に映画の話をする。青が好き",
        ),
        (
            "私をミナと呼んでください。家族の話は聞かないで",
            "ユーザーの呼び方はミナ。ユーザーは家族の話を聞かないでほしい",
        ),
        (
            "私をミナと呼んで",
            "ユーザーの呼び方はミナ、猫が好き",
        ),
    )

    for user_text, candidate in mixed_candidates:
        response = AssistantResponse(speech="了解です", memory_candidates=[candidate])
        assert service.propose(user_text, response, "conversation-id") == []

    assert service.list_memories() == []
    repository.close()


def test_strict_allowlist_rejects_appended_relationship_without_clear_separator() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())
    mixed_candidates = (
        (
            "私をミナと呼んで",
            "ユーザーの呼び方はミナ.ユーザーと私は恋人",
        ),
        (
            "私をミナと呼んで",
            "ユーザーの呼び方はミナでユーザーと私は恋人",
        ),
        (
            "私をミナと呼んで",
            "ユーザーの呼び方はミナで恋人は私",
        ),
        (
            "私をミナと呼んで",
            "ユーザーの呼び方はミナで特別な関係なのは私",
        ),
        (
            "家族の話は聞かないで",
            "ユーザーは家族の話を聞かないでほしいそしてユーザーと私は恋人",
        ),
        (
            "毎週日曜に一緒に映画の話をする",
            "ユーザーとは毎週日曜に映画の話をするそしてユーザーと私は恋人",
        ),
    )

    for user_text, candidate in mixed_candidates:
        response = AssistantResponse(speech="了解です", memory_candidates=[candidate])
        assert service.propose(user_text, response, "conversation-id") == []

    assert service.list_memories() == []
    repository.close()


def test_grounded_boundary_can_contain_relationship_or_emotion_words() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    relationship_boundary = service.propose(
        "恋人の話は聞かないで",
        AssistantResponse(
            speech="了解です",
            memory_candidates=["ユーザーは恋人の話を聞かないでほしい"],
        ),
        "conversation-id",
    )
    emotion_boundary = service.propose(
        "悲しい話は聞かないで",
        AssistantResponse(
            speech="了解です",
            memory_candidates=["ユーザーは悲しい話を聞かないでほしい"],
        ),
        "conversation-id",
    )

    assert [item.content for item in (*relationship_boundary, *emotion_boundary)] == [
        "ユーザーは恋人の話を聞かないでほしい",
        "ユーザーは悲しい話を聞かないでほしい",
    ]
    repository.close()


def test_persistent_preference_for_sad_stories_is_not_transient_state() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig())

    candidates = service.propose(
        "悲しい話が好き",
        AssistantResponse(speech="了解です", memory_candidates=["ユーザーは悲しい話が好き"]),
        "conversation-id",
    )

    assert [item.content for item in candidates] == ["ユーザーは悲しい話が好き"]
    repository.close()


def test_memory_context_excludes_only_weakly_related_memories() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(min_relevance_score=2))
    cat = repository.create_memory("ユーザーは猫の写真が好き", category=MemoryCategory.PREFERENCE)
    dog = repository.create_memory("ユーザーは犬の話が好き", category=MemoryCategory.PREFERENCE)
    repository.update_memory(cat.id, status=MemoryStatus.APPROVED)
    repository.update_memory(dog.id, status=MemoryStatus.APPROVED)

    context = service.context_for("猫の話を聞かせて")

    assert "猫の写真" in context
    assert "犬の話" not in context
    assert "毎回説明しない" in context
    repository.close()


def test_memory_context_excludes_expired_memories_without_deleting_them() -> None:
    repository = SQLiteConversationRepository(":memory:")
    service = MemoryService(repository, MemoryConfig(max_memory_age_days=1))
    memory = repository.create_memory("ユーザーは猫が好き", category=MemoryCategory.PREFERENCE)
    repository.update_memory(memory.id, status=MemoryStatus.APPROVED)
    repository._connection.execute(  # type: ignore[attr-defined]
        "UPDATE memories SET updated_at = ? WHERE id = ?",
        ((datetime.now(UTC) - timedelta(days=2)).isoformat(), memory.id),
    )

    assert service.context_for("猫について話したい") == ""
    assert len(repository.list_memories(status=MemoryStatus.APPROVED)) == 1
    repository.close()
