"""第1段階で安全に利用する、何もしない仮実装。"""

from __future__ import annotations

from multi_llm_companion.domain import AssistantResponse, EmotionState

from .base import (
    AudioInputDevice,
    AvatarControlPort,
    AvatarModelPreview,
    IntegrationErrorCallback,
    LevelCallback,
    LipSyncState,
    MmdAvatarCommand,
    RecognitionCallback,
    SpeechRecognitionPort,
    VoiceOutputPort,
)


class NullSpeechRecognizer(SpeechRecognitionPort):
    """マイクを開かず、認識処理もしない。"""

    def __init__(self, reason: str = "無効") -> None:
        self._reason = reason

    @property
    def is_available(self) -> bool:
        return False

    @property
    def is_listening(self) -> bool:
        return False

    @property
    def status_text(self) -> str:
        return self._reason

    @property
    def engine_name(self) -> str:
        return "無効"

    @property
    def input_device(self) -> int | str | None:
        return None

    def list_input_devices(self) -> list[AudioInputDevice]:
        return []

    def start(
        self,
        on_recognized: RecognitionCallback,
        on_error: IntegrationErrorCallback | None = None,
        on_partial: RecognitionCallback | None = None,
        on_level: LevelCallback | None = None,
    ) -> None:
        del on_recognized
        del on_partial
        del on_level
        if on_error is not None and self._reason != "無効":
            on_error(self._reason)

    def stop(self) -> None:
        return None

    def finish(self) -> None:
        return None


class NullVoiceOutput(VoiceOutputPort):
    """VoiSona Talkへ接続せず、音声を再生しない。"""

    def __init__(self, reason: str = "無効") -> None:
        self._reason = reason

    @property
    def is_available(self) -> bool:
        return False

    @property
    def status_text(self) -> str:
        return self._reason

    @property
    def last_error(self) -> str | None:
        return None

    @property
    def is_speaking(self) -> bool:
        return False

    @property
    def playback_status_text(self) -> str:
        return "停止中"

    @property
    def lip_sync_state(self) -> LipSyncState:
        return LipSyncState()

    def speak(self, text: str) -> None:
        del text

    def stop(self) -> None:
        return None


class NullAvatarController(AvatarControlPort):
    """画像や3Dモデルを読み込まず、表示制御もしない。"""

    def __init__(self, reason: str = "無効（2Dアバターのみ）") -> None:
        self._reason = reason

    @property
    def is_available(self) -> bool:
        return False

    @property
    def controller_name(self) -> str:
        return "無効"

    @property
    def status_text(self) -> str:
        return self._reason

    @property
    def model_preview(self) -> AvatarModelPreview | None:
        return None

    @property
    def mmd_command(self) -> MmdAvatarCommand | None:
        return None

    def apply_response(self, response: AssistantResponse) -> None:
        del response

    def apply_emotion_state(self, state: EmotionState) -> None:
        del state

    def reset(self) -> None:
        return None

    def generate_confirmation_vmd(self) -> str:
        raise RuntimeError("PMXモデルを読み込んでから確認用VMDを生成してください。")
