"""第8段階で3Dアバター連携を安全に確認するMock。"""

from __future__ import annotations

from multi_llm_companion.domain import AssistantResponse, EmotionState

from .base import AvatarControlPort, AvatarModelPreview, MmdAvatarCommand


class Mock3DAvatarController(AvatarControlPort):
    """外部通信やモデル読込をせず、3D向け命令だけを受け取る。"""

    def __init__(self) -> None:
        self.last_response: AssistantResponse | None = None
        self.last_emotion_state: EmotionState | None = None
        self.reset_count = 0

    @property
    def is_available(self) -> bool:
        return True

    @property
    def controller_name(self) -> str:
        return "3D連携Mock"

    @property
    def status_text(self) -> str:
        return "接続済み（モデル未読み込み）"

    @property
    def model_preview(self) -> AvatarModelPreview | None:
        return None

    @property
    def mmd_command(self) -> MmdAvatarCommand | None:
        return None

    def apply_response(self, response: AssistantResponse) -> None:
        self.last_response = response

    def apply_emotion_state(self, state: EmotionState) -> None:
        self.last_emotion_state = state

    def reset(self) -> None:
        self.last_response = None
        self.last_emotion_state = None
        self.reset_count += 1

    def generate_confirmation_vmd(self) -> str:
        raise RuntimeError("確認用VMDはPMXモデルを読み込んだ場合だけ生成できます。")
