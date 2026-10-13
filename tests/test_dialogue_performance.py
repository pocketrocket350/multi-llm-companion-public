from __future__ import annotations

from datetime import UTC, datetime

from multi_llm_companion.domain import DialogueMode, StoredMessage
from multi_llm_companion.services.dialogue_performance import (
    build_conversation_performance_plan,
    build_performance_prompt,
)


def _assistant(message_id: int, speech: str) -> StoredMessage:
    return StoredMessage(
        id=message_id,
        conversation_id="one",
        role="assistant",
        content=speech,
        created_at=datetime.now(UTC),
    )


def _user(message_id: int, content: str) -> StoredMessage:
    return StoredMessage(
        id=message_id,
        conversation_id="one",
        role="user",
        content=content,
        created_at=datetime.now(UTC),
    )


def test_plan_avoids_stock_acknowledgement_after_recent_acknowledgement() -> None:
    plan = build_conversation_performance_plan(
        "まだ会議が終わらない",
        [_assistant(1, "そうなんだ。長いね。")],
        mode=DialogueMode.CASUAL,
    )

    assert plan.avoid_stock_acknowledgement
    assert "定型相づちだけで始めず" in build_performance_prompt(plan)


def test_plan_avoids_reused_motif_in_recent_three_assistant_messages() -> None:
    plan = build_conversation_performance_plan(
        "帰り道も長い",
        [
            _assistant(1, "雨の音がするね。"),
            _assistant(2, "雨の日は足元に気をつけたい。"),
            _assistant(3, "駅まで歩いたんだね。"),
        ],
        mode=DialogueMode.CASUAL,
    )

    assert plan.avoid_reused_motif
    assert "平易な言葉で答える" in build_performance_prompt(plan)


def test_plan_prefers_plain_language_for_a_short_status_message() -> None:
    plan = build_conversation_performance_plan(
        "靴下が濡れた",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert plan.prefer_plain_language
    assert "新しい比喩、情景説明、頼まれていない助言を足さない" in build_performance_prompt(plan)


def test_plan_does_not_prefer_plain_language_for_a_long_message() -> None:
    plan = build_conversation_performance_plan(
        "会議が終わらず、帰宅後の予定も変えなければならなくなって困っている",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert plan.prefer_plain_language is False


def test_plan_respects_a_user_decision_without_adding_advice() -> None:
    plan = build_conversation_performance_plan(
        "靴下だけ洗うことにする",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert plan.respect_user_decision
    assert "次の行動や別案を提案しない" in build_performance_prompt(plan)


def test_plan_does_not_treat_a_report_as_a_user_decision() -> None:
    plan = build_conversation_performance_plan(
        "靴下が濡れた",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert plan.respect_user_decision is False


def test_plan_respects_a_recent_user_correction() -> None:
    plan = build_conversation_performance_plan(
        "いちばん好きなのは紅茶",
        [_user(1, "実はコーヒーは飲める"), _assistant(2, "そうなんだ")],
        mode=DialogueMode.CASUAL,
    )

    assert plan.respect_correction
    assert "以前の情報を言い切らず" in build_performance_prompt(plan)


def test_plan_respects_a_recent_yappari_correction() -> None:
    plan = build_conversation_performance_plan(
        "一番好きなのは紅茶",
        [_user(1, "やっぱりコーヒーも飲める"), _assistant(2, "そうなんだ")],
        mode=DialogueMode.CASUAL,
    )

    assert plan.respect_correction


def test_plan_does_not_keep_correction_rule_after_three_new_user_turns() -> None:
    plan = build_conversation_performance_plan(
        "今日は本を読んだ",
        [
            _user(1, "実はコーヒーは飲める"),
            _user(2, "昨日は散歩した"),
            _user(3, "今日は雨だった"),
            _user(4, "帰りに買い物した"),
        ],
        mode=DialogueMode.CASUAL,
    )

    assert plan.respect_correction is False


def test_plan_suppresses_unearned_agreement_for_harmful_or_extreme_claim() -> None:
    plan = build_conversation_performance_plan(
        "徹夜すれば全部終わる",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert plan.avoid_unearned_agreement
    assert "無条件に同意せず" in build_performance_prompt(plan)


def test_plan_allows_small_opinion_for_light_chat_only() -> None:
    casual_plan = build_conversation_performance_plan(
        "好きな映画はある？",
        [],
        mode=DialogueMode.CASUAL,
    )
    listening_plan = build_conversation_performance_plan(
        "好きな映画はある？",
        [],
        mode=DialogueMode.LISTENING,
    )

    assert casual_plan.allow_small_opinion
    assert "小さな所感を一つだけ" in build_performance_prompt(casual_plan)
    assert listening_plan.allow_small_opinion is False


def test_plan_does_not_allow_small_opinion_for_a_user_preference_or_correction() -> None:
    user_preference = build_conversation_performance_plan(
        "休日はカフェで読書したい",
        [],
        mode=DialogueMode.CASUAL,
    )
    correction = build_conversation_performance_plan(
        "前に好きって言ったけど、やっぱりコーヒーは苦手",
        [],
        mode=DialogueMode.CASUAL,
    )

    assert user_preference.allow_small_opinion is False
    assert correction.allow_small_opinion is False


def test_plan_can_enable_only_one_performance_feature() -> None:
    plan = build_conversation_performance_plan(
        "毎晩やれば全員すぐに上達するよ。映画は好き？",
        [
            _assistant(1, "雨の音がいいね。"),
            _assistant(2, "雨の日は落ち着く。"),
        ],
        mode=DialogueMode.CASUAL,
        enabled_features=frozenset({"motif"}),
    )

    assert plan.avoid_reused_motif
    assert plan.avoid_stock_acknowledgement is False
    assert plan.avoid_unearned_agreement is False
    assert plan.allow_small_opinion is False


def test_plan_rejects_unknown_performance_feature() -> None:
    try:
        build_conversation_performance_plan(
            "こんにちは",
            [],
            mode=DialogueMode.CASUAL,
            enabled_features=frozenset({"unknown"}),
        )
    except ValueError as error:
        assert "未対応" in str(error)
    else:
        raise AssertionError("未対応の機能名は拒否する必要があります。")
