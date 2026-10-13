from __future__ import annotations

from multi_llm_companion.domain import AssistantResponse, Emotion
from multi_llm_companion.services.emotion import EmotionService
from multi_llm_companion.storage import SQLiteConversationRepository


def test_emotion_transition_is_smoothed_and_persisted() -> None:
    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation("感情の会話")
    service = EmotionService(repository)

    happy = service.apply_response(
        conversation.id,
        AssistantResponse(speech="うれしいです", emotion=Emotion.HAPPY, emotion_intensity=0.8),
    )
    still_happy = service.apply_response(
        conversation.id,
        AssistantResponse(speech="少し悲しいです", emotion=Emotion.SAD, emotion_intensity=0.2),
    )
    sad = service.apply_response(
        conversation.id,
        AssistantResponse(speech="とても悲しいです", emotion=Emotion.SAD, emotion_intensity=0.9),
    )

    assert (happy.emotion, happy.intensity) == (Emotion.HAPPY, 0.8)
    assert (still_happy.emotion, still_happy.intensity) == (Emotion.HAPPY, 0.34)
    assert (sad.emotion, sad.intensity) == (Emotion.SAD, 0.595)
    assert service.current(conversation.id) == sad
    repository.close()


def test_emotion_service_gradually_decays_previous_intensity() -> None:
    repository = SQLiteConversationRepository(":memory:")
    conversation = repository.create_conversation()
    service = EmotionService(repository, previous_weight=0.5, decay_factor=0.8)
    service.apply_response(
        conversation.id,
        AssistantResponse(speech="うれしい", emotion=Emotion.HAPPY, emotion_intensity=0.8),
    )

    state = service.apply_response(
        conversation.id,
        AssistantResponse(speech="続けます", emotion=Emotion.HAPPY, emotion_intensity=0.8),
    )

    assert state.intensity == 0.72
    repository.close()
