from datetime import UTC, datetime

from multi_llm_companion.domain import AssistantResponse, EmotionState
from multi_llm_companion.integrations import (
    Mock3DAvatarController,
    NullAvatarController,
    NullSpeechRecognizer,
    NullVoiceOutput,
)


def test_phase_one_integrations_do_nothing() -> None:
    recognizer = NullSpeechRecognizer()
    voice = NullVoiceOutput()
    avatar = NullAvatarController()

    assert recognizer.is_available is False
    assert voice.is_available is False
    assert avatar.is_available is False
    recognizer.start(lambda _text: None)
    recognizer.stop()
    voice.speak("音声にはしない")
    voice.stop()
    avatar.apply_response(AssistantResponse(speech="表示だけ"))
    avatar.reset()


def test_mock_3d_avatar_receives_common_avatar_commands_without_external_io() -> None:
    avatar = Mock3DAvatarController()
    response = AssistantResponse(speech="3Dへ渡す文章", gesture="wave", gaze="left")
    state = EmotionState(
        conversation_id="test-conversation",
        emotion="happy",
        intensity=0.6,
        updated_at=datetime.now(UTC),
    )

    avatar.apply_response(response)
    avatar.apply_emotion_state(state)

    assert avatar.is_available is True
    assert avatar.status_text == "接続済み（モデル未読み込み）"
    assert avatar.last_response == response
    assert avatar.last_emotion_state == state
    avatar.reset()
    assert avatar.last_response is None
    assert avatar.reset_count == 1
