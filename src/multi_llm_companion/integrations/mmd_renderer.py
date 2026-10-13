"""第24段階のMMD描画方式を交換する基盤。"""

from __future__ import annotations

from multi_llm_companion.domain import AssistantResponse, EmotionState

from .base import (
    AvatarControlPort,
    AvatarModelPreview,
    MmdAvatarCommand,
    MmdRendererPort,
    MmdRendererStatus,
)


class LegacyPmxPreviewRenderer(MmdRendererPort):
    """既存のPMX簡易表示を、将来交換できる描画方式として扱う。"""

    def __init__(self) -> None:
        self.preview: AvatarModelPreview | None = None
        self.command: MmdAvatarCommand | None = None

    @property
    def status(self) -> MmdRendererStatus:
        model_text = (
            f"{self.preview.model_name}（{self.preview.vertex_count:,}頂点）"
            if self.preview is not None
            else "モデル未読み込み"
        )
        physics_text = (
            f"剛体 {len(self.preview.rigid_bodies)} / ジョイント {len(self.preview.physics_joints)}"
            if self.preview is not None and self.preview.physics_joints
            else "物理情報なし"
        )
        vmd_text = (
            f"VMD: {self.preview.vmd_motion.name}"
            if self.preview is not None and self.preview.vmd_motion is not None
            else "VMD未設定"
        )
        return MmdRendererStatus(
            renderer_id="legacy_pmx_preview",
            renderer_name="PMX材質・ボーン表示",
            status_text=(
                f"{model_text}を材質・PMXボーン変形付きで表示中"
                f"（{physics_text}、{vmd_text}、付与変形・軸制約・IKリンク制限・"
                "固定刻み衝突物理・VMDボーン再生対応）"
            ),
            is_full_mmd=False,
            is_fallback=False,
        )

    def load_model(self, preview: AvatarModelPreview | None) -> None:
        self.preview = preview

    def apply_command(self, command: MmdAvatarCommand | None) -> None:
        self.command = command


class MmdRuntimeRenderer(MmdRendererPort):
    """将来の本格MMDランタイム用の入口。現段階では安全に予備表示へ戻す。"""

    def __init__(self, fallback: MmdRendererPort) -> None:
        self._fallback = fallback
        self.preview: AvatarModelPreview | None = None
        self.command: MmdAvatarCommand | None = None

    @property
    def status(self) -> MmdRendererStatus:
        model_text = self.preview.model_name if self.preview is not None else "モデル未読み込み"
        physics_text = (
            f"剛体 {len(self.preview.rigid_bodies)} / ジョイント {len(self.preview.physics_joints)}"
            if self.preview is not None and self.preview.physics_joints
            else "物理情報なし"
        )
        vmd_text = (
            f"VMD: {self.preview.vmd_motion.name}"
            if self.preview is not None and self.preview.vmd_motion is not None
            else "VMD未設定"
        )
        return MmdRendererStatus(
            renderer_id="mmd_runtime",
            renderer_name="MMDランタイム基盤",
            status_text=(
                f"{model_text}: 本格MMD描画は未実装のため、"
                f"PMX材質・PMXボーン変形表示で表示中です（{physics_text}、{vmd_text}、"
                "付与変形・軸制約・IKリンク制限・固定刻み衝突物理対応）。"
            ),
            is_full_mmd=False,
            is_fallback=True,
        )

    def load_model(self, preview: AvatarModelPreview | None) -> None:
        self.preview = preview
        self._fallback.load_model(preview)

    def apply_command(self, command: MmdAvatarCommand | None) -> None:
        self.command = command
        self._fallback.apply_command(command)


class RendererBackedAvatarController(AvatarControlPort):
    """モデル制御と描画方式を別部品として接続するアダプター。"""

    def __init__(self, controller: AvatarControlPort, renderer: MmdRendererPort) -> None:
        self._controller = controller
        self._renderer = renderer
        self._renderer.load_model(controller.model_preview)
        self._renderer.apply_command(controller.mmd_command)

    @property
    def is_available(self) -> bool:
        return self._controller.is_available

    @property
    def controller_name(self) -> str:
        return self._renderer.status.renderer_name

    @property
    def status_text(self) -> str:
        return f"{self._renderer.status.status_text} / 制御状態: {self._controller.status_text}"

    @property
    def model_preview(self) -> AvatarModelPreview | None:
        return self._controller.model_preview

    @property
    def mmd_command(self) -> MmdAvatarCommand | None:
        return self._controller.mmd_command

    def expression_preview_command(self, emotion: str) -> MmdAvatarCommand | None:
        """内側のPMX制御器が持つ表情確認命令だけを透過する。"""

        create = getattr(self._controller, "expression_preview_command", None)
        return create(emotion) if callable(create) else None

    def apply_response(self, response: AssistantResponse) -> None:
        self._controller.apply_response(response)
        self._renderer.apply_command(self._controller.mmd_command)

    def apply_emotion_state(self, state: EmotionState) -> None:
        self._controller.apply_emotion_state(state)
        self._renderer.apply_command(self._controller.mmd_command)

    def reset(self) -> None:
        self._controller.reset()
        self._renderer.apply_command(self._controller.mmd_command)

    def generate_confirmation_vmd(self) -> str:
        path = self._controller.generate_confirmation_vmd()
        self._renderer.load_model(self._controller.model_preview)
        return path
