from __future__ import annotations

from datetime import UTC, datetime

import pytest

from multi_llm_companion.config import BehaviorPolicyConfig, CharacterConfig, PersonaConfig
from multi_llm_companion.domain import (
    AssistantResponse,
    DialogueAct,
    DialogueMode,
    StoredMessage,
)
from multi_llm_companion.services.dialogue import DialogueDirector, DialogueState
from multi_llm_companion.storage import SQLiteConversationRepository


def _director() -> DialogueDirector:
    return DialogueDirector(
        PersonaConfig(
            character=CharacterConfig(name="案内役"),
            behavior_policy=BehaviorPolicyConfig(),
            dialogue_modes={
                "listening": {
                    "applies_when": "疲れを受け止めるとき",
                    "do": "短く受け止める",
                    "avoid": "行動を迫る",
                }
            },
        )
    )


def _assistant(message_id: int, response: AssistantResponse | None) -> StoredMessage:
    return StoredMessage(
        id=message_id,
        conversation_id="one",
        role="assistant",
        content=response.speech if response else "旧回答",
        response=response,
        created_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今日は疲れた", DialogueMode.LISTENING),
        ("Pythonのエラー原因を教えて？", DialogueMode.EXPLAINING),
        ("来週の計画を立てたい", DialogueMode.PLANNING),
        ("冗談を言って", DialogueMode.PLAYFUL),
        ("どうしたらいいか相談したい", DialogueMode.SUPPORTING),
        ("その話はやめて", DialogueMode.BOUNDARY),
        ("こんにちは", DialogueMode.CASUAL),
    ],
)
def test_director_selects_one_mode_without_llm(text: str, expected: DialogueMode) -> None:
    assert _director().select_mode(text) is expected


def test_state_rebuilds_only_from_assistant_response_metadata() -> None:
    messages = [
        _assistant(
            1,
            AssistantResponse(
                speech="どう思う？",
                dialogue_mode="supporting",
                dialogue_act="answer",
                current_topic=" 旅行  の  計画 ",
                open_loop="行き先を決める",
            ),
        ),
        _assistant(
            2,
            AssistantResponse(
                speech="候補を三つ挙げます。",
                dialogue_mode="planning",
                dialogue_act="suggest",
                current_topic="旅行",
                open_loop="日程を確認する",
            ),
        ),
        _assistant(3, None),
    ]

    state = _director().rebuild_state(messages)

    assert state.last_mode is DialogueMode.CASUAL
    assert state.last_act is DialogueAct.ANSWER
    assert state.current_topic == ""
    assert state.open_loop == ""
    assert state.recent_question_count == 0
    assert state.recent_suggestion_count == 1


def test_runtime_prompt_prioritizes_current_topic_and_applies_cooldowns() -> None:
    state = DialogueState(
        last_mode=DialogueMode.PLANNING,
        last_act=DialogueAct.ASK,
        current_topic="旅行",
        open_loop="日程を確認する",
        recent_question_count=2,
        recent_suggestion_count=2,
    )

    prompt = _director().build_runtime_prompt("別の話にしよう。今日は疲れた", state)

    assert "今回の会話モード: listening" in prompt
    assert "current_topic: なし" in prompt
    assert "open_loop: なし" in prompt
    assert "今回は明確な必要性がない限り質問しない" in prompt
    assert "同じ行動を短期間に繰り返さない" in prompt
    assert "閉じた話題を蒸し返さず" in prompt
    assert "行動要求や助言を押しつけない" in prompt


def test_continuation_can_reference_open_loop_without_treating_it_as_fact() -> None:
    prompt = _director().build_runtime_prompt(
        "さっきの話の続き",
        DialogueState(current_topic="本の話", open_loop="感想を聞く"),
    )

    assert "参照条件: 継続表現あり" in prompt
    assert "open_loop: 感想を聞く" in prompt
    assert "確定した利用者情報として扱わない" in prompt


def test_reply_length_and_grounding_rules_do_not_force_the_same_response() -> None:
    brief = _director().build_prompt("一文だけ返して", [])
    detail = _director().build_prompt("詳しく説明して", [])
    assert "挨拶を含めて一文だけ" in brief
    assert "理由や具体例まで説明" in detail
    assert "毎回名前を添えたり" in brief
    assert "曜日だけから明日などの日時を補わない" in brief


def test_continuation_keeps_a_topic_named_between_sakki_and_hanashi() -> None:
    prompt = _director().build_runtime_prompt(
        "さっきのカレーの話に戻る",
        DialogueState(current_topic="料理", open_loop="次の具材"),
    )

    assert "参照条件: 継続表現あり" in prompt
    assert "current_topic: 料理" in prompt


def test_runtime_prompt_includes_performance_plan_without_changing_cooldowns() -> None:
    messages = [_assistant(1, AssistantResponse(speech="そうなんだ。長いね。"))]

    prompt = _director().build_prompt("まだ会議が終わらない", messages)

    assert "毎回を質問終端にせず" in prompt
    assert "【今回の会話演技】" in prompt
    assert "定型相づちだけで始めず" in prompt


def test_proactive_prompt_keeps_closed_topics_and_repeated_questions_out() -> None:
    messages = [
        _assistant(
            1,
            AssistantResponse(
                speech="どう思う？",
                dialogue_act="ask",
            ),
        ),
        _assistant(
            2,
            AssistantResponse(
                speech="今日はここまでにしよう。",
                dialogue_act="close",
                current_topic="会議",
            ),
        ),
    ]

    prompt = _director().build_proactive_prompt(messages)

    assert "【今回の会話モード: casual】" in prompt
    assert "利用者からの新しい発言がない場面" in prompt
    assert "確認、言い換え、要約をせず、能動発話を始めない" in prompt
    assert "直近に質問が続いているため、新しい質問を重ねない" in prompt


def test_state_is_isolated_by_conversation_and_zero_after_deletion() -> None:
    repository = SQLiteConversationRepository(":memory:")
    first = repository.create_conversation("一つ目")
    second = repository.create_conversation("二つ目")
    repository.add_message(
        first.id,
        "assistant",
        "続きを話します。",
        response=AssistantResponse(
            speech="続きを話します。",
            dialogue_mode="planning",
            dialogue_act="plan",
            current_topic="一つ目の話題",
            open_loop="一つ目の続き",
        ),
    )

    first_state = _director().rebuild_state(repository.list_messages(first.id))
    second_state = _director().rebuild_state(repository.list_messages(second.id))

    assert first_state.current_topic == "一つ目の話題"
    assert second_state == DialogueState()
    assert repository.delete_conversation(first.id)
    assert _director().rebuild_state(repository.list_messages(first.id)) == DialogueState()
    repository.close()
