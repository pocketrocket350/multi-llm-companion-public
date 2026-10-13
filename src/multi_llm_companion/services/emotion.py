"""会話の感情状態を保存し、急な変化をなだらかにする。"""

from __future__ import annotations

from datetime import UTC, datetime

from multi_llm_companion.domain import AssistantResponse, Emotion, EmotionState
from multi_llm_companion.storage import ConversationRepository


class EmotionService:
    """LLMの回答から、画面用の会話感情を作る。"""

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        previous_weight: float = 0.5,
        decay_factor: float = 0.85,
    ) -> None:
        if not 0.0 <= previous_weight < 1.0:
            raise ValueError("previous_weight は0以上1未満にしてください。")
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor は0より大きく1以下にしてください。")
        self._repository = repository
        self._previous_weight = previous_weight
        self._decay_factor = decay_factor

    def current(self, conversation_id: str) -> EmotionState | None:
        return self._repository.get_emotion_state(conversation_id)

    def apply_response(
        self,
        conversation_id: str,
        response: AssistantResponse,
    ) -> EmotionState:
        """回答感情を反映し、違う感情へは段階的に移行する。"""

        previous = _decay(self.current(conversation_id), self._decay_factor)
        state = _next_state(conversation_id, previous, response, self._previous_weight)
        return self._repository.save_emotion_state(state)


def _next_state(
    conversation_id: str,
    previous: EmotionState | None,
    response: AssistantResponse,
    previous_weight: float,
) -> EmotionState:
    target_intensity = response.emotion_intensity
    if previous is None:
        return _state(conversation_id, response.emotion, target_intensity)

    if response.emotion == previous.emotion:
        intensity = _blend(previous.intensity, target_intensity, previous_weight)
        return _state(conversation_id, response.emotion, intensity)

    if response.emotion == Emotion.NEUTRAL and previous.intensity > 0.25:
        return _state(conversation_id, previous.emotion, previous.intensity * previous_weight)

    if previous.emotion != Emotion.NEUTRAL and target_intensity < previous.intensity:
        return _state(conversation_id, previous.emotion, previous.intensity * previous_weight)

    intensity = _blend(previous.intensity, target_intensity, previous_weight)
    return _state(conversation_id, response.emotion, intensity)


def _blend(previous: float, target: float, previous_weight: float) -> float:
    return round(previous * previous_weight + target * (1.0 - previous_weight), 3)


def _decay(previous: EmotionState | None, factor: float) -> EmotionState | None:
    """会話が続くほど感情の強さを少しずつ通常へ戻す。"""

    if previous is None or previous.intensity <= 0.0:
        return previous
    intensity = round(previous.intensity * factor, 3)
    emotion = previous.emotion if intensity >= 0.03 else Emotion.NEUTRAL
    return previous.model_copy(update={"emotion": emotion, "intensity": intensity})


def _state(conversation_id: str, emotion: Emotion, intensity: float) -> EmotionState:
    return EmotionState(
        conversation_id=conversation_id,
        emotion=emotion,
        intensity=round(max(0.0, min(1.0, intensity)), 3),
        updated_at=datetime.now(UTC),
    )
