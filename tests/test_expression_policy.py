from __future__ import annotations

import pytest

from multi_llm_companion.domain import (
    AssistantResponse,
    ConversationState,
    Emotion,
    ExpressionChannel,
    ExpressionOutputStatus,
    ExpressionPolicy,
    SpeechAct,
    diagnose_expression_channels,
)


def test_expression_policy_normalizes_aliases_without_changing_common_response() -> None:
    response = AssistantResponse(
        speech="こんにちは？",
        emotion="happy",
        emotion_intensity=0.7,
        gesture=" Greeting ",
        gaze="Viewer",
    )
    policy = ExpressionPolicy()

    intent = policy.normalize(response)

    assert response.gesture == " Greeting "
    assert response.gaze == "Viewer"
    assert intent.gesture_intent == "wave"
    assert intent.gaze_intent == "user"
    assert intent.speech_act == SpeechAct.QUESTION
    assert intent.conversation_state == ConversationState.REACTING


def test_expression_policy_preserves_registered_external_intent() -> None:
    policy = ExpressionPolicy(supported_gesture_intents=("head_scratch",))

    intent = policy.normalize(
        AssistantResponse(speech="考えます", gesture="Head-Scratch", gaze="left")
    )

    assert intent.gesture_intent == "head_scratch"
    assert intent.gaze_intent == "left"


def test_expression_policy_safely_rejects_unknown_gesture_and_gaze() -> None:
    policy = ExpressionPolicy()

    intent = policy.normalize(
        AssistantResponse(speech="未対応です", gesture="dangerous_dance", gaze="behind")
    )

    assert intent.gesture_intent == "none"
    assert intent.gaze_intent == "forward"


def test_conversation_state_has_one_japanese_label_for_each_state() -> None:
    assert {state.display_name for state in ConversationState} == {
        "待機中",
        "聞き取り中",
        "回答を作成中",
        "姿勢を準備中",
        "読み上げ中",
        "音声を停止しました",
    }


@pytest.mark.parametrize("emotion", list(Emotion))
@pytest.mark.parametrize("intensity", [0.0, 0.15, 0.5, 1.0])
def test_expression_policy_builds_bounded_outputs_for_representative_states(
    emotion: Emotion,
    intensity: float,
) -> None:
    policy = ExpressionPolicy(voice_expression_enabled=True)

    intent = policy.normalize(
        AssistantResponse(
            speech="表現テスト",
            emotion=emotion,
            emotion_intensity=intensity,
            gesture="wave",
            gaze="left",
        )
    )

    assert intent.outputs is not None
    outputs = intent.outputs
    assert 0.0 <= outputs.voice.intensity <= 0.55
    assert 0.0 <= outputs.expression.intensity <= 0.82
    assert 0.0 <= outputs.gesture.intensity <= 0.68
    assert 0.0 <= outputs.gaze.intensity <= 0.50
    assert 0.0 <= outputs.natural_motion.intensity <= 0.32
    if emotion == Emotion.NEUTRAL or intensity == 0.0:
        assert {
            outputs.voice.intensity,
            outputs.expression.intensity,
            outputs.gaze.intensity,
            outputs.natural_motion.intensity,
        } == {0.0}
        assert outputs.gesture.intensity == pytest.approx(0.68)
    else:
        assert (
            len(
                {
                    outputs.voice.intensity,
                    outputs.expression.intensity,
                    outputs.gesture.intensity,
                    outputs.gaze.intensity,
                    outputs.natural_motion.intensity,
                }
            )
            >= 3
        )


def test_voice_expression_is_disabled_by_default_without_disabling_visual_outputs() -> None:
    intent = ExpressionPolicy().normalize(
        AssistantResponse(
            speech="うれしいです",
            emotion="happy",
            emotion_intensity=1.0,
            gesture="wave",
            gaze="user",
        )
    )

    assert intent.outputs is not None
    assert intent.outputs.voice.status == ExpressionOutputStatus.DISABLED
    assert intent.outputs.voice.intensity == 0.0
    assert intent.outputs.expression.intensity > 0.0
    assert "音声 無効" in intent.outputs.status_text
    assert "ボイス固有スタイル未対応" in intent.outputs.status_text
    assert "音声" not in intent.outputs.visual_status_text
    assert "ボイス固有スタイル" not in intent.outputs.visual_status_text


def test_unknown_outputs_are_safely_corrected_once_and_reported() -> None:
    intent = ExpressionPolicy().normalize(
        AssistantResponse(
            speech="安全な補正",
            emotion="surprised",
            emotion_intensity=0.8,
            gesture="unknown_motion",
            gaze="behind",
        )
    )

    assert intent.outputs is not None
    assert intent.gesture_intent == "none"
    assert intent.gaze_intent == "forward"
    assert len(intent.outputs.corrections) == 2
    assert intent.outputs.diagnostics == ()
    assert "安全補正済み" in intent.outputs.status_text


def test_contradiction_diagnostic_ignores_weak_expression_but_detects_strong_conflict() -> None:
    weak = {
        "voice": ExpressionChannel(Emotion.HAPPY, 0.2),
        "expression": ExpressionChannel(Emotion.SAD, 0.2),
    }
    strong = {
        "voice": ExpressionChannel(Emotion.HAPPY, 0.5),
        "expression": ExpressionChannel(Emotion.SAD, 0.5),
    }

    assert diagnose_expression_channels(weak) == ()
    assert "強い感情方向" in diagnose_expression_channels(strong)[0]


def test_existing_expression_intent_constructor_remains_compatible() -> None:
    intent = ExpressionPolicy().normalize(AssistantResponse(speech="互換テスト"))
    legacy = type(intent)(
        intent.emotion,
        intent.intensity,
        intent.gesture_intent,
        intent.gaze_intent,
        intent.speech_act,
        intent.conversation_state,
    )

    assert legacy.outputs is None
