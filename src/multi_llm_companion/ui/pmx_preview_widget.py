"""PMXモデルの形状と口モーフを確認するための軽量なプレビュー。"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygon,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from multi_llm_companion.integrations.lip_sync_diagnostic import write_lip_sync_trace
from multi_llm_companion.ui.companion_background import (
    _cover_rect,
    _local_background,
    _standing_background_rect,
)
from multi_llm_companion.ui.dense_pmx_widget import DensePmxWidget
from multi_llm_companion.ui.models import (
    UiAvatarModelPreview,
    UiAvatarTexturedTriangle,
    UiMmdAvatarCommand,
    UiVmdMotion,
)
from multi_llm_companion.ui.natural_motion import (
    ConversationMotionState,
    NaturalMotionController,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms as _accurate_bone_transforms
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer

# VMD・物理の固定刻みに合わせ、姿勢も30fpsで更新する。
_HEAVY_UPDATE_INTERVAL_MS = 33


class PmxPreviewWidget(QWidget):
    """PMXの材質テクスチャを使い、口元も確認できる軽量プレビュー。"""

    renderer_status_changed = Signal()
    conversation_pose_ready = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        render_quality: str = "standard",
    ) -> None:
        super().__init__(parent)
        self._preview: UiAvatarModelPreview | None = None
        self._command: UiMmdAvatarCommand | None = None
        self._mouth_shape = "neutral"
        self._mouth_weights: dict[str, float] = {}
        self._last_mouth_at = time.perf_counter()
        self._emotion_weights: dict[str, float] = {}
        self._target_emotion = "neutral"
        self._target_emotion_intensity = 0.0
        self._expression_weights: dict[str, float] = {}
        self._target_expression_weights: dict[str, float] = {}
        self._rest_expression_weights: dict[str, float] = {}
        self._yaw = math.pi
        self._pitch = 0.0
        self._rotation_360 = False
        self._zoom = 1.0
        self._drag_start: QPoint | None = None
        self._texture_cache: dict[str, QImage] = {}
        self._background_image = _local_background()
        self._dense_view = DensePmxWidget(self, render_quality=render_quality)
        self._dense_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._dense_view.renderer_ready.connect(self._dense_renderer_ready)
        self._dense_view.quality_status_changed.connect(self.renderer_status_changed.emit)
        self._dense_view.conversation_pose_ready.connect(self.conversation_pose_ready.emit)
        self._dense_renderer_available = False
        self._dense_renderer_attempted = False
        self._motion_enabled = True
        self._voice_active = False
        self._speech_mouth_active = False
        self._speech_timeline_active = False
        self._expression_preview_active = False
        self._gaze_preview_active = False
        self._natural_motion = NaturalMotionController()
        self._fallback_vmd_player = VmdMotionPlayer()
        self._fallback_vmd_rotations: dict[int, tuple[float, float, float]] = {}
        self._fallback_vmd_translations: dict[int, tuple[float, float, float]] = {}
        self._idle_phase = 0.0
        self._last_lightweight_at = time.perf_counter()
        self._last_heavy_at = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_lightweight_motion)
        self._heavy_timer = QTimer(self)
        self._heavy_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._heavy_timer.setInterval(_HEAVY_UPDATE_INTERVAL_MS)
        self._heavy_timer.timeout.connect(self._advance_heavy_motion)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self._dense_view.setGeometry(self.rect())
        self._dense_view.hide()
        self._sync_lightweight_animation()
        self._timer.start()
        self._heavy_timer.start()

    def set_preview(self, preview: UiAvatarModelPreview | None) -> None:
        if preview is self._preview:
            return
        self._texture_cache.clear()
        self._preview = preview
        self._fallback_vmd_player.reset(preview.vmd_motion if preview is not None else None)
        (
            self._fallback_vmd_rotations,
            self._fallback_vmd_translations,
            _morph_weights,
        ) = self._fallback_vmd_player.advance(preview, 0.0, self._motion_enabled)
        self._dense_view.set_preview(preview)
        self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
        self._sync_lightweight_animation()
        self._dense_view.setVisible(bool(preview and preview.dense_textured_triangles))
        self.renderer_status_changed.emit()
        self.update()

    @property
    def renderer_status_text(self) -> str:
        preview = self._preview
        if preview is None:
            return "PMXテクスチャ: 未使用"
        if self._dense_renderer_available:
            return (
                f"PMXテクスチャ: GPU描画（全 {len(preview.dense_textured_triangles):,}面） / "
                f"{self._dense_view.quality_status_text} / "
                f"{self._dense_view.physics_status_text} / "
                f"{self._dense_view.vmd_status_text}"
            )
        if preview.dense_textured_triangles:
            if not self._dense_renderer_attempted:
                return "PMXテクスチャ: GPU準備中"
            return f"{_texture_status_text(preview)} / {self._fallback_vmd_player.status_text}"
        return f"{_texture_status_text(preview)} / {self._fallback_vmd_player.status_text}"

    def reset_camera(self) -> None:
        """正面・標準倍率へ戻す。"""

        self._yaw = math.pi
        self._pitch = 0.0
        self._zoom = 1.0
        self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
        self.update()

    def set_rotation_360(self, enabled: bool) -> None:
        """手動回転の範囲だけを変更し、姿勢・背景・モーションは変えない。"""

        self._rotation_360 = enabled
        self._yaw = self._yaw % math.tau if enabled else max(0.0, min(math.tau, self._yaw))
        self._pitch = 0.0
        self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
        self.update()

    def set_view_mode(self, mode: str) -> None:
        """通常画角または明示的なVMD演出画角を選ぶ。"""

        self._dense_view.set_view_mode(mode)
        if self._dense_view._view_mode == "full" and self._zoom < 1.0:
            self._zoom = 1.0
            self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
        self.renderer_status_changed.emit()
        self.update()

    @contextmanager
    def _temporary_preview_camera(self, mode: str):
        """管理中だけ正面の標準画角にし、閉じたら利用者のカメラへ戻す。"""
        camera = (self._yaw, self._pitch, self._zoom)
        previous_mode = self._dense_view._view_mode
        self.reset_camera()
        self.set_view_mode(mode)
        try:
            yield
        finally:
            self._yaw, self._pitch, self._zoom = camera
            self._dense_view.set_camera(*camera)
            self.set_view_mode(previous_mode)

    def _set_rest_expression(self, weights: dict[str, float]) -> None:
        """待機へ戻るときの基礎表情だけを保持する。"""

        if weights == self._rest_expression_weights:
            return
        self._rest_expression_weights = dict(weights)
        self._dense_view._set_rest_expression(weights)
        if self._command is None or (
            self._command.emotion == "neutral"
            and not self._command.expression_morphs
            and not self._command.emotion_morph
        ):
            self._target_expression_weights = dict(weights)
            if not self._motion_enabled:
                self._expression_weights = dict(weights)
        self.update()

    def set_mmd_command(
        self,
        command: UiMmdAvatarCommand | None,
        *,
        smooth_gaze_transition: bool = False,
    ) -> None:
        if command == self._command and not smooth_gaze_transition:
            return
        self._command = command
        if smooth_gaze_transition:
            self._dense_view.set_mmd_command(command, smooth_gaze_transition=True)
        else:
            self._dense_view.set_mmd_command(command)
        self._natural_motion.set_emotion(
            command.emotion if command is not None else "neutral",
            (
                command.natural_motion_intensity
                if command is not None and command.natural_motion_intensity is not None
                else (command.emotion_intensity if command is not None else 0.0)
            ),
        )
        self._target_expression_weights = (
            {name: max(0.0, min(1.0, weight)) for name, weight in command.expression_morphs if name}
            if command is not None
            else {}
        )
        if command is None or (
            command.emotion == "neutral"
            and not command.expression_morphs
            and not command.emotion_morph
        ):
            self._target_expression_weights = dict(self._rest_expression_weights)
        if command is None or command.expression_morphs or not command.emotion_morph:
            self._target_emotion = "neutral"
            self._target_emotion_intensity = 0.0
        else:
            self._target_emotion = command.emotion
            self._target_emotion_intensity = max(
                0.0,
                min(1.0, command.emotion_intensity),
            )
            self._emotion_weights.setdefault(command.emotion, 0.0)
        if not self._motion_enabled:
            self._emotion_weights = (
                {self._target_emotion: self._target_emotion_intensity}
                if self._target_emotion != "neutral"
                else {}
            )
            self._expression_weights = dict(self._target_expression_weights)
        self.update()

    def set_lip_sync(self, mouth_shape: str) -> None:
        """現在の口形を、簡易描画とGPU描画の両方へ反映する。"""

        normalized = mouth_shape if mouth_shape in {"a", "i", "u", "e", "o"} else "neutral"
        if normalized == self._mouth_shape:
            return
        if self._voice_active:
            self._advance_mouth_transition()
        self._mouth_shape = normalized
        if not self._voice_active:
            # 手動のモーフ確認では選んだ形をそのまま表示する。
            self._mouth_weights = {} if normalized == "neutral" else {normalized: 1.0}
            self._dense_view.set_mouth_shape(normalized)
        self.update()

    def _advance_mouth_transition(self) -> None:
        """短い閉口や母音の切替をなじませ、全開・全閉の往復を抑える。"""

        if not self._voice_active:
            return
        now = time.perf_counter()
        elapsed = max(0.0, now - self._last_mouth_at)
        self._last_mouth_at = now
        target = {} if self._mouth_shape == "neutral" else {self._mouth_shape: 1.0}
        blend = 1.0 - math.exp(-elapsed / (0.060 if not target else 0.050))
        weights = {}
        for shape in self._mouth_weights.keys() | target.keys():
            current = self._mouth_weights.get(shape, 0.0)
            value = current + (target.get(shape, 0.0) - current) * blend
            if abs(value - target.get(shape, 0.0)) < 0.01:
                value = target.get(shape, 0.0)
            if value > 0.0:
                weights[shape] = value
        self._mouth_weights = weights
        self._dense_view._set_mouth_weights(self._mouth_shape, weights)
        self.update()

    def set_voice_active(self, active: bool) -> None:
        """発話中も待機動作を継続し、会話ジェスチャーだけを重ねる。"""

        if active == self._voice_active:
            return
        was_active = self._voice_active
        self._voice_active = active
        if active:
            self._last_mouth_at = time.perf_counter()
            self._timer.start()
        else:
            # 停止・割り込みでは補間を残さず、その場で閉じる。
            if not self._motion_enabled:
                self._timer.stop()
            self._mouth_shape = "neutral"
            self._mouth_weights = {}
            self._dense_view.set_mouth_shape("neutral")
            self.update()
        if was_active and not active:
            self._dense_view.end_conversation_motion()
        self._sync_lightweight_animation()

    def set_conversation_state(self, state: ConversationMotionState) -> None:
        """待機・聞き取り・思考・発話を自然動作へ渡す。"""

        self._natural_motion.set_state(state)

    def set_speech_activity(self, mouth_active: bool, timeline_active: bool) -> None:
        """文章中の無音区間を検出し、首だけのうなずきを予約する。"""

        if (
            self._speech_mouth_active
            and not mouth_active
            and timeline_active
            and self._speech_timeline_active
        ):
            self._natural_motion.phrase_boundary()
        self._speech_mouth_active = mouth_active
        self._speech_timeline_active = timeline_active

    def begin_conversation_motion(self) -> int | None:
        """音声より先に、会話の視線・姿勢を準備する。"""

        return self._dense_view.begin_conversation_motion()

    def end_conversation_motion(self) -> None:
        """音声がない場合も会話ジェスチャーを残さない。"""

        self._dense_view.end_conversation_motion()

    def interrupt_motion(self) -> None:
        """新しい利用者入力を優先し、古い会話動作を解除する。"""

        self._expression_preview_active = False
        self._gaze_preview_active = False
        self._command = None
        self._target_emotion = "neutral"
        self._target_emotion_intensity = 0.0
        self._target_expression_weights = dict(self._rest_expression_weights)
        if not self._motion_enabled:
            self._expression_weights = dict(self._rest_expression_weights)
        self._natural_motion.set_emotion("neutral", 0.0)
        self._dense_view.interrupt_motion()
        self.update()

    @property
    def is_motion_preview_active(self) -> bool:
        return self._dense_view.is_motion_preview_active

    @property
    def is_expression_preview_active(self) -> bool:
        return self._expression_preview_active

    @property
    def is_gaze_preview_active(self) -> bool:
        return self._gaze_preview_active

    def start_expression_preview(self, command: UiMmdAvatarCommand) -> None:
        """会話へ保存せず、指定された表情モーフだけを確認する。"""

        if self._voice_active:
            raise RuntimeError("音声再生中は表情を確認できません。")
        self._expression_preview_active = True
        self.set_mmd_command(command)

    def stop_expression_preview(self) -> None:
        """表情確認を終え、通常の待機状態へ戻す。"""

        if not self._expression_preview_active:
            return
        self._expression_preview_active = False
        self.set_mmd_command(None)

    def start_gaze_preview(self, command: UiMmdAvatarCommand) -> None:
        """会話へ保存せず、既存の視線層と補間だけを確認する。"""

        if self._voice_active:
            raise RuntimeError("音声再生中は視線を確認できません。")
        self._gaze_preview_active = True
        self.set_mmd_command(command, smooth_gaze_transition=True)

    def stop_gaze_preview(self) -> None:
        """視線確認を終え、既存の待機復帰補間へ渡す。"""

        if not self._gaze_preview_active:
            return
        self._gaze_preview_active = False
        self.end_conversation_motion()

    @property
    def is_conversation_motion_active(self) -> bool:
        return self._dense_view.is_conversation_motion_active

    def start_motion_preview(self, motion: UiVmdMotion) -> None:
        """現在のPMXを維持したまま、管理画面のVMDだけを再生する。"""

        if self._voice_active:
            raise RuntimeError("音声再生中はモーションをプレビューできません。")
        self._command = None
        self._target_emotion = "neutral"
        self._target_emotion_intensity = 0.0
        self._target_expression_weights = dict(self._rest_expression_weights)
        self._natural_motion.set_emotion("neutral", 0.0)
        self._dense_view.start_motion_preview(motion)
        self.update()

    def stop_motion_preview(self) -> None:
        self._dense_view.stop_motion_preview()
        self.update()

    def set_motion_enabled(self, enabled: bool) -> None:
        """簡易プレビューの待機動作を開始または停止する。"""

        self._motion_enabled = enabled
        self._natural_motion.set_enabled(enabled)
        if enabled:
            self._last_lightweight_at = time.perf_counter()
            self._last_heavy_at = time.perf_counter()
            self._timer.start()
            self._heavy_timer.start()
        else:
            if not self._voice_active:
                self._timer.stop()
            self._heavy_timer.stop()
            self.interrupt_motion()
        self._sync_lightweight_animation()
        self._dense_view.advance_heavy_animation(0.0, enabled)
        self.update()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._timer.stop()
        self._heavy_timer.stop()
        self._dense_view.shutdown()
        super().closeEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#E8F5F9"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        preview = self._preview
        base_scale = (
            _scale_for(preview.points, min(self.width(), self.height()) * 0.52) * self._zoom
            if preview is not None and preview.points and not self._dense_renderer_available
            else 1.0
        )
        if not self._background_image.isNull():
            image = self._background_image
            if self._dense_renderer_available:
                painter.drawImage(QRectF(self.rect()), image, self._dense_view._background_crop())
            elif preview is not None and preview.points and self._dense_view._view_mode != "vmd":
                center = self._dense_view._model_center
                height = self._dense_view._model_height
                target = _standing_background_rect(
                    image.width(),
                    image.height(),
                    QPointF(
                        self.width() / 2 + center[0] * base_scale,
                        self.height() / 2 - (center[1] - height / 2) * base_scale,
                    ),
                    height * base_scale,
                )
                painter.drawImage(target, image)
            else:
                painter.drawImage(
                    QRectF(self.rect()),
                    image,
                    _cover_rect(image.width(), image.height(), self.width(), self.height()),
                )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if preview is None or not preview.points:
            self._draw_message(painter, "PMXモデルは未読み込みです")
            return

        command = self._command
        if not self._dense_renderer_available:
            half_width = self.width() / 2
            half_height = self.height() / 2
            breathing = 1.0 + (
                math.sin(self._idle_phase * 0.55) * 0.012 if self._motion_enabled else 0.0
            )
            scale = base_scale * breathing
            gaze_yaw = (
                0.0
                if self._has_gaze_bone(preview)
                else _gaze_yaw(command.gaze if command is not None else "forward")
            )
            idle_yaw = math.sin(self._idle_phase * 0.28) * 0.018 if self._motion_enabled else 0.0
            rotation_yaw = self._yaw + gaze_yaw + idle_yaw
            if preview.triangles:
                self._draw_mesh(painter, preview, half_width, half_height, scale, rotation_yaw)
                self._draw_face_closeup(painter, preview, rotation_yaw)
            else:
                self._draw_points(painter, preview, half_width, half_height, scale, rotation_yaw)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        current = event.position().toPoint()
        delta = current - self._drag_start
        yaw = self._yaw + delta.x() * 0.01
        self._yaw = yaw % math.tau if self._rotation_360 else max(0.0, min(math.tau, yaw))
        self._pitch = 0.0
        self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
        self._drag_start = current
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120
        if steps:
            minimum = 1.0 if self._dense_view._view_mode == "full" else 0.4
            self._zoom = max(minimum, min(3.0, self._zoom * (1.12**steps)))
            self._dense_view.set_camera(self._yaw, self._pitch, self._zoom)
            self.update()
        event.accept()

    def _advance_lightweight_motion(self) -> None:
        """GPU行列だけで動く呼吸・揺れを、約30fpsで進める。"""

        if not self._motion_enabled and not self._voice_active:
            return
        now = time.perf_counter()
        raw_elapsed = max(0.0, now - self._last_lightweight_at)
        elapsed = min(0.1, raw_elapsed)
        self._last_lightweight_at = now
        self._advance_mouth_transition()
        if self._motion_enabled:
            self._idle_phase += elapsed * 0.92
            self._natural_motion.advance(elapsed)
            self._advance_emotion_transition()
            self._sync_lightweight_animation()
        self.update()
        self._write_timer_trace(
            "pmx_lightweight_tick",
            raw_elapsed,
            started_at=now,
            period_ms=self._timer.interval(),
        )

    def _advance_heavy_motion(self) -> None:
        """ボーン・VMD・物理を低頻度で更新し、発話中は追いつき計算をしない。"""

        if not self._motion_enabled:
            return
        now = time.perf_counter()
        raw_elapsed = max(0.0, now - self._last_heavy_at)
        elapsed = min(1.0, raw_elapsed)
        self._last_heavy_at = now
        try:
            if self._dense_view.is_conversation_motion_active:
                self._dense_view.advance_conversation_animation(elapsed)
                if not self._dense_renderer_available:
                    (
                        self._fallback_vmd_rotations,
                        self._fallback_vmd_translations,
                        _morph_weights,
                    ) = self._fallback_vmd_player.advance(self._preview, elapsed, True)
                    self.update()
                return
            if self._dense_renderer_available:
                self._dense_view.advance_heavy_animation(elapsed, True)
                return
            self._dense_view.advance_heavy_animation(elapsed, True)
            (
                self._fallback_vmd_rotations,
                self._fallback_vmd_translations,
                _morph_weights,
            ) = self._fallback_vmd_player.advance(self._preview, elapsed, True)
            self.update()
        finally:
            self._write_timer_trace(
                "pmx_heavy_tick",
                raw_elapsed,
                started_at=now,
                period_ms=self._heavy_timer.interval(),
            )

    @staticmethod
    def _write_timer_trace(
        event: str,
        elapsed_seconds: float,
        *,
        started_at: float,
        period_ms: int,
    ) -> None:
        interval_ms = round(elapsed_seconds * 1000)
        write_lip_sync_trace(
            "pmx",
            event,
            period_ms=period_ms,
            interval_ms=interval_ms,
            overrun_ms=max(0, interval_ms - period_ms),
            callback_duration_ms=round((time.perf_counter() - started_at) * 1000),
        )

    def _sync_lightweight_animation(self) -> None:
        self._dense_view.set_lightweight_animation(
            self._idle_phase,
            self._motion_enabled,
            self._voice_active,
        )
        self._dense_view.set_natural_motion(self._natural_motion.frame)

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        self._dense_view.setGeometry(self.rect())
        super().resizeEvent(event)  # type: ignore[arg-type]

    def _dense_renderer_ready(self, available: bool) -> None:
        self._dense_renderer_available = available
        self._dense_renderer_attempted = True
        if not available:
            self._dense_view.hide()
        self.renderer_status_changed.emit()
        self.update()

    def _advance_emotion_transition(self) -> None:
        """表情の強さを少しずつ目標値へ近づける。"""

        step = 0.18
        emotions = set(self._emotion_weights)
        if self._target_emotion != "neutral":
            emotions.add(self._target_emotion)
        updated: dict[str, float] = {}
        for emotion in emotions:
            current = self._emotion_weights.get(emotion, 0.0)
            target = self._target_emotion_intensity if emotion == self._target_emotion else 0.0
            if current < target:
                current = min(target, current + step)
            elif current > target:
                current = max(target, current - step)
            if current > 0.001:
                updated[emotion] = current
        self._emotion_weights = updated
        expression_names = self._expression_weights.keys() | self._target_expression_weights.keys()
        expression_updated: dict[str, float] = {}
        for name in expression_names:
            current = self._expression_weights.get(name, 0.0)
            target = self._target_expression_weights.get(name, 0.0)
            if current < target:
                current = min(target, current + step)
            elif current > target:
                current = max(target, current - step)
            if current > 0.001:
                expression_updated[name] = current
        self._expression_weights = expression_updated

    def _draw_points(
        self,
        painter: QPainter,
        preview: UiAvatarModelPreview,
        half_width: float,
        half_height: float,
        scale: float,
        yaw: float,
    ) -> None:
        painter.setPen(QPen(QColor("#f1c7e0"), 2))
        for x, y, depth in self._transformed_points(preview, yaw, camera=True):
            perspective = 1.0 / max(0.35, 1.0 + depth * 0.12)
            painter.drawPoint(
                int(half_width + x * scale * perspective),
                int(half_height - y * scale * perspective),
            )

    def _draw_mesh(
        self,
        painter: QPainter,
        preview: UiAvatarModelPreview,
        half_width: float,
        half_height: float,
        scale: float,
        yaw: float,
    ) -> None:
        """テクスチャ付き面を優先し、利用できない場合は単色面を描く。"""

        transformed = self._transformed_points(preview, yaw, camera=True)
        if preview.textured_triangles:
            self._draw_textured_mesh(
                painter,
                preview,
                transformed,
                half_width,
                half_height,
                scale,
            )
            return
        triangles: list[tuple[float, QPolygon]] = []
        for indexes in preview.triangles:
            try:
                first, second, third = (transformed[index] for index in indexes)
            except IndexError:
                continue
            vertices = (first, second, third)
            polygon = QPolygon()
            for x, y, depth in vertices:
                perspective = 1.0 / max(0.35, 1.0 + depth * 0.12)
                polygon.append(
                    QPoint(
                        int(half_width + x * scale * perspective),
                        int(half_height - y * scale * perspective),
                    )
                )
            triangles.append((sum(vertex[2] for vertex in vertices) / 3, polygon))

        base_color = QColor("#dda1bf")
        painter.setPen(QPen(QColor("#754157"), 1))
        for depth, polygon in sorted(triangles, key=lambda item: item[0], reverse=True):
            brightness = max(78, min(138, int(110 - depth * 12)))
            painter.setBrush(base_color.lighter(brightness))
            painter.drawPolygon(polygon)

    def _draw_textured_mesh(
        self,
        painter: QPainter,
        preview: UiAvatarModelPreview,
        transformed: list[tuple[float, float, float]],
        half_width: float,
        half_height: float,
        scale: float,
    ) -> None:
        """UV座標で画像を三角形に貼り付け、画像が読めない面は単色へ戻す。"""

        triangles: list[
            tuple[float, UiAvatarTexturedTriangle, tuple[QPointF, QPointF, QPointF]]
        ] = []
        for triangle in preview.textured_triangles:
            try:
                vertices = tuple(transformed[index] for index in triangle.indexes)
            except IndexError:
                continue
            projected = tuple(
                QPointF(
                    half_width + x * scale / max(0.35, 1.0 + depth * 0.12),
                    half_height - y * scale / max(0.35, 1.0 + depth * 0.12),
                )
                for x, y, depth in vertices
            )
            triangles.append((sum(vertex[2] for vertex in vertices) / 3, triangle, projected))

        for _depth, triangle, points in sorted(triangles, key=lambda item: item[0], reverse=True):
            image = self._texture(triangle.texture_path)
            if image.isNull() or not self._draw_textured_triangle(
                painter,
                image,
                triangle.uvs,
                points,
            ):
                color = _diffuse_color(triangle.diffuse)
                painter.setPen(QPen(color.darker(145), 1))
                painter.setBrush(color)
                painter.drawPolygon(QPolygon([point.toPoint() for point in points]))

    def _texture(self, texture_path: str) -> QImage:
        image = self._texture_cache.get(texture_path)
        if image is None:
            image = QImage(texture_path)
            self._texture_cache[texture_path] = image
        return image

    def _draw_textured_triangle(
        self,
        painter: QPainter,
        image: QImage,
        uvs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        points: tuple[QPointF, QPointF, QPointF],
    ) -> bool:
        source = tuple(QPointF(u * image.width(), (1.0 - v) * image.height()) for u, v in uvs)
        transform = _triangle_transform(source, points)
        if transform is None:
            return False
        path = QPainterPath()
        path.moveTo(points[0])
        path.lineTo(points[1])
        path.lineTo(points[2])
        path.closeSubpath()
        painter.save()
        painter.setClipPath(path)
        painter.setWorldTransform(transform)
        painter.drawImage(QPointF(0.0, 0.0), image)
        painter.restore()
        painter.setPen(QPen(QColor(45, 30, 40, 90), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(QPolygon([point.toPoint() for point in points]))
        return True

    def _draw_face_closeup(
        self,
        painter: QPainter,
        preview: UiAvatarModelPreview,
        yaw: float,
    ) -> None:
        """口と表情のモーフに関係する面を切り出して大きく描く。"""

        target_indexes = self._face_target_indexes(preview)
        triangles = [
            triangle
            for triangle in preview.triangles
            if any(index in target_indexes for index in triangle)
        ]
        if not target_indexes or not triangles:
            return
        transformed = self._transformed_points(preview, yaw, camera=True)
        target_points = [transformed[index] for index in target_indexes if index < len(transformed)]
        if not target_points:
            return
        rect = QRect(7, 47, min(190, self.width() - 14), 120)
        painter.fillRect(rect, QColor("#273544"))
        painter.setPen(QPen(QColor("#f4d5e4"), 1))
        painter.drawRect(rect)
        painter.drawText(
            rect.adjusted(6, 4, -6, -4),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            "顔モーフ拡大: "
            f"{(self._command.expression_preset or self._command.emotion_morph) if self._command else '表情なし'}"
            f" / {_mouth_label(self._mouth_shape) if self._mouth_shape != 'neutral' else '口形なし'}",
        )

        minimum_x = min(point[0] for point in target_points)
        maximum_x = max(point[0] for point in target_points)
        minimum_y = min(point[1] for point in target_points)
        maximum_y = max(point[1] for point in target_points)
        span = max(maximum_x - minimum_x, maximum_y - minimum_y, 0.001)
        scale = min(rect.width() - 16, rect.height() - 30) / span
        center_x = (minimum_x + maximum_x) / 2
        center_y = (minimum_y + maximum_y) / 2
        draw_x = rect.center().x()
        draw_y = rect.center().y() + 10
        painter.setPen(QPen(QColor("#7a4059"), 1))
        painter.setBrush(QColor("#e8aac7"))
        for triangle in triangles:
            try:
                points = [transformed[index] for index in triangle]
            except IndexError:
                continue
            polygon = QPolygon(
                [
                    QPoint(
                        int(draw_x + (point[0] - center_x) * scale),
                        int(draw_y - (point[1] - center_y) * scale),
                    )
                    for point in points
                ]
            )
            painter.drawPolygon(polygon)

    def _face_target_indexes(self, preview: UiAvatarModelPreview) -> set[int]:
        """現在動いている口・表情・姿勢だけを顔の拡大対象にする。"""

        target_indexes = {
            index
            for shape in self._mouth_weights
            for index, _offset_x, _offset_y, _offset_z in preview.mouth_morphs.get(shape, ())
        }
        for emotion, weight in self._emotion_weights.items():
            if weight <= 0.001:
                continue
            target_indexes.update(
                index
                for index, _offset_x, _offset_y, _offset_z in preview.emotion_morphs.get(
                    emotion,
                    (),
                )
            )
        for morph_name, weight in self._expression_weights.items():
            if weight <= 0.001:
                continue
            target_indexes.update(
                index
                for index, _offset_x, _offset_y, _offset_z in preview.named_vertex_morphs.get(
                    morph_name,
                    (),
                )
            )
        target_indexes.update(self._pose_target_indexes(preview))
        return target_indexes

    def _transformed_points(
        self,
        preview: UiAvatarModelPreview,
        yaw: float,
        *,
        camera: bool = False,
    ) -> list[tuple[float, float, float]]:
        offsets: dict[int, tuple[float, float, float]] = {}
        for shape, weight in self._mouth_weights.items():
            _add_weighted_offsets(offsets, preview.mouth_morphs.get(shape, ()), weight)
        for emotion, weight in self._emotion_weights.items():
            _add_weighted_offsets(
                offsets,
                preview.emotion_morphs.get(emotion, ()),
                weight,
            )
        for morph_name, weight in self._expression_weights.items():
            _add_weighted_offsets(
                offsets,
                preview.named_vertex_morphs.get(morph_name, ()),
                weight,
            )
        morphed_points = [
            (
                point[0] + offsets.get(index, (0.0, 0.0, 0.0))[0],
                point[1] + offsets.get(index, (0.0, 0.0, 0.0))[1],
                point[2] + offsets.get(index, (0.0, 0.0, 0.0))[2],
            )
            for index, point in enumerate(preview.points)
        ]
        posed_points = _apply_bone_pose(
            morphed_points,
            preview,
            self._pose_rotations(preview),
            self._fallback_vmd_translations,
        )
        rotate = _camera_rotate if camera else _rotate
        return [rotate(point, yaw, 0.1) for point in posed_points]

    def _has_gaze_bone(self, preview: UiAvatarModelPreview) -> bool:
        command = self._command
        return bool(
            command
            and command.gaze_bone
            and any(bone.name == command.gaze_bone for bone in preview.bones)
        )

    def _pose_rotations(
        self,
        preview: UiAvatarModelPreview,
    ) -> dict[int, tuple[float, float, float]]:
        """設定済みの目・頭・上半身ボーンへ小さな回転を割り当てる。"""

        command = self._command
        if command is None or not preview.bones:
            return {}
        indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
        rotations = dict(self._fallback_vmd_rotations)
        gaze_index = indexes.get(command.gaze_bone)
        if gaze_index is not None:
            gaze_pitch = {"up": 0.12, "down": -0.12}.get(command.gaze, 0.0) * command.gaze_intensity
            gaze_yaw = _gaze_yaw(command.gaze) * command.gaze_intensity
            rotations[gaze_index] = (gaze_pitch, gaze_yaw, 0.0)
        head_index = indexes.get(command.head_bone)
        if head_index is not None:
            nod_pitch = (
                math.sin(self._idle_phase * 1.8) * 0.14 * command.gesture_intensity
                if command.gesture == "nod" and self._motion_enabled
                else 0.0
            )
            rotations[head_index] = (nod_pitch, 0.0, 0.0)
            upper_body_index = indexes.get(command.upper_body_bone)
            if upper_body_index is not None and nod_pitch:
                rotations[upper_body_index] = (nod_pitch * 0.25, 0.0, 0.0)
        return rotations

    def _pose_target_indexes(self, preview: UiAvatarModelPreview) -> set[int]:
        command = self._command
        if command is None or not preview.vertex_weights:
            return set()
        names = {command.gaze_bone, command.head_bone}
        indexes = {
            index for index, bone in enumerate(preview.bones) if bone.name in names and bone.name
        }
        return {
            vertex_index
            for vertex_index, weights in enumerate(preview.vertex_weights)
            if any(bone_index in indexes for bone_index, _weight in weights)
        }

    def _draw_message(self, painter: QPainter, message: str) -> None:
        painter.setPen(QColor("#1A267C"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, message)


def _scale_for(points: tuple[tuple[float, float, float], ...], target_size: float) -> float:
    maximum = max((max(abs(value) for value in point) for point in points), default=1.0)
    return target_size / max(maximum, 0.001)


def _add_weighted_offsets(
    combined: dict[int, tuple[float, float, float]],
    offsets: tuple[tuple[int, float, float, float], ...],
    weight: float,
) -> None:
    """同じ頂点へ作用する複数のモーフ差分を加算する。"""

    if weight <= 0.0:
        return
    for index, offset_x, offset_y, offset_z in offsets:
        current_x, current_y, current_z = combined.get(index, (0.0, 0.0, 0.0))
        combined[index] = (
            current_x + offset_x * weight,
            current_y + offset_y * weight,
            current_z + offset_z * weight,
        )


def _apply_bone_pose(
    points: list[tuple[float, float, float]],
    preview: UiAvatarModelPreview,
    rotations: dict[int, tuple[float, float, float]],
    translations: dict[int, tuple[float, float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    """PMXの基本BDEF/SDEFウェイトで、指定したボーン姿勢を頂点へ反映する。"""

    if not (rotations or translations) or len(preview.vertex_weights) != len(points):
        return points
    transforms = _bone_transforms(preview, rotations, translations)
    result: list[tuple[float, float, float]] = []
    for point, weights in zip(points, preview.vertex_weights, strict=True):
        valid_weights = [
            (bone_index, weight)
            for bone_index, weight in weights
            if bone_index in transforms and weight > 0.0
        ]
        total_weight = sum(weight for _bone_index, weight in valid_weights)
        if total_weight <= 0.0:
            result.append(point)
            continue
        x = y = z = 0.0
        for bone_index, weight in valid_weights:
            rotation, translation = transforms[bone_index]
            rest_position = preview.bones[bone_index].position
            transformed = _add(
                _matrix_vector(rotation, _subtract(point, rest_position)),
                translation,
            )
            normalized_weight = weight / total_weight
            x += transformed[0] * normalized_weight
            y += transformed[1] * normalized_weight
            z += transformed[2] * normalized_weight
        result.append((x, y, z))
    return result


def _bone_transforms(
    preview: UiAvatarModelPreview,
    rotations: dict[int, tuple[float, float, float]],
    translations: dict[int, tuple[float, float, float]] | None = None,
) -> dict[int, tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]]:
    """簡易表示も本格表示と同じPMX変形規則を使う。"""

    return _accurate_bone_transforms(preview, rotations, translations)


def _matrix_vector(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _add(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _subtract(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _rotate(
    point: tuple[float, float, float],
    yaw: float,
    pitch: float,
) -> tuple[float, float, float]:
    x, y, z = point
    cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)
    rotated_x = x * cosine_yaw - z * sine_yaw
    rotated_z = x * sine_yaw + z * cosine_yaw
    cosine_pitch, sine_pitch = math.cos(pitch), math.sin(pitch)
    rotated_y = y * cosine_pitch - rotated_z * sine_pitch
    depth = y * sine_pitch + rotated_z * cosine_pitch
    return rotated_x, rotated_y, depth


def _camera_rotate(
    point: tuple[float, float, float],
    yaw: float,
    pitch: float,
) -> tuple[float, float, float]:
    """PMXのZだけを反転し、正面の左右を保って投影する。"""

    return _rotate((point[0], point[1], -point[2]), yaw - math.pi, pitch)


def _emotion_color(emotion: str) -> QColor:
    colors = {
        "happy": "#f7d774",
        "sad": "#8fb8e8",
        "angry": "#ed8a8a",
        "embarrassed": "#f1a4bd",
        "surprised": "#d8b4f8",
    }
    return QColor(colors.get(emotion, "#f1c7e0"))


def _gaze_yaw(gaze: str) -> float:
    return {"left": 0.22, "right": -0.22}.get(gaze, 0.0)


def _mouth_label(shape: str) -> str:
    return {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(shape, "閉じる")


def _texture_status_text(preview: UiAvatarModelPreview) -> str:
    if preview.textured_triangles:
        return f"PMXテクスチャ: 簡易表示 {len(preview.textured_triangles):,}面"
    if preview.triangles:
        return "PMXテクスチャ: なし（単色メッシュ表示）"
    return "PMXテクスチャ: 大きなモデルのため簡易表示"


def _diffuse_color(diffuse: tuple[float, float, float, float]) -> QColor:
    """PMXの材質色を、壊れた値を避けてQtの色へ変換する。"""

    red, green, blue, alpha = (max(0.0, min(1.0, value)) for value in diffuse)
    return QColor.fromRgbF(red, green, blue, alpha)


def _triangle_transform(
    source: tuple[QPointF, QPointF, QPointF],
    target: tuple[QPointF, QPointF, QPointF],
) -> QTransform | None:
    """画像上のUV三角形から画面上の三角形へのアフィン変換を作る。"""

    first, second, third = source
    target_first, target_second, target_third = target
    source_delta_x1 = second.x() - first.x()
    source_delta_y1 = second.y() - first.y()
    source_delta_x2 = third.x() - first.x()
    source_delta_y2 = third.y() - first.y()
    determinant = source_delta_x1 * source_delta_y2 - source_delta_x2 * source_delta_y1
    if abs(determinant) < 0.0001:
        return None

    def coefficients(
        first_value: float, second_value: float, third_value: float
    ) -> tuple[float, float, float]:
        delta_one = second_value - first_value
        delta_two = third_value - first_value
        horizontal = (delta_one * source_delta_y2 - delta_two * source_delta_y1) / determinant
        vertical = (source_delta_x1 * delta_two - source_delta_x2 * delta_one) / determinant
        return horizontal, vertical, first_value - horizontal * first.x() - vertical * first.y()

    horizontal_x, vertical_x, offset_x = coefficients(
        target_first.x(), target_second.x(), target_third.x()
    )
    horizontal_y, vertical_y, offset_y = coefficients(
        target_first.y(), target_second.y(), target_third.y()
    )
    return QTransform(
        horizontal_x,
        horizontal_y,
        0.0,
        vertical_x,
        vertical_y,
        0.0,
        offset_x,
        offset_y,
        1.0,
    )
