"""外部素材を使わずに描く、動きのある汎用2Dアバター表示。"""

from __future__ import annotations

from math import sin

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class AvatarWidget(QWidget):
    """感情・視線・簡単な動きを表示するコード描画アバター。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._emotion = "neutral"
        self._intensity = 0.0
        self._gesture = ""
        self._gaze = ""
        self._gesture_intensity = 1.0
        self._gaze_intensity = 1.0
        self._motion_enabled = True
        self._speaking = False
        self._mouth_open = False
        self._mouth_shape = "neutral"
        self._lip_sync_controlled = False
        self._gesture_phase = 0.0
        self._idle_phase = 0.0
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._target_gaze_x = 0.0
        self._target_gaze_y = 0.0
        self._hair_color = QColor("#684765")
        self._outfit_color = QColor("#b87aa4")
        self._accent_color = QColor("#d99ac1")
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._advance_frame)
        self.setMinimumSize(170, 230)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setToolTip("外部画像を使わない、アプリ内で描画する汎用アバターです。")
        self._timer.start()

    @property
    def emotion(self) -> str:
        return self._emotion

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def gesture(self) -> str:
        return self._gesture

    @property
    def gaze(self) -> str:
        return self._gaze

    @property
    def gesture_intensity(self) -> float:
        return self._gesture_intensity

    @property
    def gaze_intensity(self) -> float:
        return self._gaze_intensity

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def motion_enabled(self) -> bool:
        return self._motion_enabled

    def set_appearance(self, hair_color: str, outfit_color: str, accent_color: str) -> None:
        """ローカル設定の色だけを、外部素材なしで表示へ反映する。"""

        self._hair_color = QColor(hair_color)
        self._outfit_color = QColor(outfit_color)
        self._accent_color = QColor(accent_color)
        self.update()

    def set_avatar_state(
        self,
        emotion: str = "neutral",
        intensity: float = 0.0,
        gesture: str = "",
        gaze: str = "",
        gesture_intensity: float = 1.0,
        gaze_intensity: float = 1.0,
    ) -> None:
        """回答の共通形式を、表示用の安全な状態へ反映する。"""

        allowed = {"neutral", "happy", "sad", "angry", "embarrassed", "surprised"}
        self._emotion = emotion if emotion in allowed else "neutral"
        self._intensity = max(0.0, min(1.0, float(intensity)))
        self._gesture = gesture.strip()
        self._gaze = gaze.strip()
        self._gesture_intensity = max(0.0, min(1.0, float(gesture_intensity)))
        self._gaze_intensity = max(0.0, min(1.0, float(gaze_intensity)))
        gaze_x, gaze_y = _gaze_target(self._gaze)
        self._target_gaze_x = gaze_x * self._gaze_intensity
        self._target_gaze_y = gaze_y * self._gaze_intensity
        self._start_animation_if_needed()
        self.update()

    def set_speaking(self, speaking: bool) -> None:
        """互換用の単純な口パク表示を開始または停止する。"""

        self._speaking = speaking
        self._lip_sync_controlled = False
        if not speaking:
            self._mouth_open = False
            self._mouth_shape = "neutral"
        self._start_animation_if_needed()
        self.update()

    def set_lip_sync(self, is_active: bool, mouth_shape: str) -> None:
        """推定済みの母音口形を表示する。"""

        self._speaking = is_active
        self._lip_sync_controlled = True
        self._mouth_shape = mouth_shape if mouth_shape in {"a", "i", "u", "e", "o"} else "neutral"
        self._mouth_open = self._mouth_shape != "neutral"
        self._start_animation_if_needed()
        self.update()

    def set_motion_enabled(self, enabled: bool) -> None:
        """利用者が動きの表示を停止できるようにする。"""

        self._motion_enabled = enabled
        if enabled:
            self._start_animation_if_needed()
        else:
            self._timer.stop()
            self._mouth_open = False
            self.update()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#f5f8fc"))

        size = min(rect.width(), rect.height())
        scale = size / 240.0
        center_x = rect.center().x()
        base_head_y = rect.top() + int(82 * scale)
        breath_offset = self._breath_offset(scale)
        body_head_y = base_head_y + breath_offset
        head_y = body_head_y + self._nod_offset(scale)
        head_x = center_x + self._head_sway_offset(scale)

        self._draw_body(painter, center_x, body_head_y, scale)
        self._draw_head(painter, head_x, head_y, scale)
        painter.end()

    def _advance_frame(self) -> None:
        if not self._motion_enabled:
            self._timer.stop()
            return
        self._gesture_phase += 0.35
        self._idle_phase += 0.11
        if not self._lip_sync_controlled:
            self._mouth_open = self._speaking and int(self._gesture_phase * 2) % 2 == 0
        self._gaze_x += (self._target_gaze_x - self._gaze_x) * 0.3
        self._gaze_y += (self._target_gaze_y - self._gaze_y) * 0.3
        self.update()

    def _start_animation_if_needed(self) -> None:
        if self._motion_enabled:
            self._timer.start()

    def _has_continuous_motion(self) -> bool:
        return self._motion_enabled

    def _breath_offset(self, scale: float) -> int:
        """待機中もわずかに上下する、呼吸風の動き。"""

        if not self._motion_enabled:
            return 0
        strength = 2.0 if not self._speaking else 3.0
        return int(sin(self._idle_phase * 0.55) * strength * scale)

    def _head_sway_offset(self, scale: float) -> int:
        """正面を保ったまま、首が固まって見えない程度に揺らす。"""

        if not self._motion_enabled:
            return 0
        strength = 1.6 if not self._speaking else 2.4
        return int(sin(self._idle_phase * 0.34) * strength * scale)

    def _eye_open_ratio(self) -> float:
        """数秒に一度の短い瞬きを、外部素材なしで表す。"""

        if not self._motion_enabled:
            return 1.0
        distance = abs((self._idle_phase % 10.0) - 8.0)
        return max(0.12, min(1.0, distance / 0.18))

    def _nod_offset(self, scale: float) -> int:
        if self._motion_enabled and _gesture_kind(self._gesture) == "nod":
            return int(sin(self._gesture_phase) * 7 * scale * self._gesture_intensity)
        return 0

    def _draw_body(self, painter: QPainter, center_x: int, head_y: int, scale: float) -> None:
        shoulder_y = head_y + int(73 * scale)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._outfit_color)
        painter.drawRoundedRect(
            center_x - int(70 * scale),
            shoulder_y,
            int(140 * scale),
            int(110 * scale),
            int(30 * scale),
            int(30 * scale),
        )

        painter.setPen(
            QPen(self._outfit_color.darker(125), max(2, int(7 * scale)), Qt.PenStyle.SolidLine)
        )
        if _gesture_kind(self._gesture) == "wave":
            wave_x = (
                int(sin(self._gesture_phase) * 12 * scale * self._gesture_intensity)
                if self._motion_enabled
                else 0
            )
            wave_y = (
                int(sin(self._gesture_phase * 2) * 5 * scale * self._gesture_intensity)
                if self._motion_enabled
                else 0
            )
            painter.drawLine(
                center_x + int(48 * scale),
                shoulder_y + int(26 * scale),
                center_x + int(82 * scale) + wave_x,
                shoulder_y - int(35 * scale) + wave_y,
            )
            painter.setBrush(QColor("#f1c7a7"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                center_x + int(73 * scale) + wave_x,
                shoulder_y - int(53 * scale) + wave_y,
                int(22 * scale),
                int(28 * scale),
            )
            return

        painter.drawLine(
            center_x - int(48 * scale),
            shoulder_y + int(22 * scale),
            center_x - int(76 * scale),
            shoulder_y + int(70 * scale),
        )
        painter.drawLine(
            center_x + int(48 * scale),
            shoulder_y + int(22 * scale),
            center_x + int(76 * scale),
            shoulder_y + int(70 * scale),
        )

    def _draw_head(self, painter: QPainter, center_x: int, head_y: int, scale: float) -> None:
        head_width = int(112 * scale)
        head_height = int(132 * scale)
        head_left = center_x - head_width // 2
        head_top = head_y - head_height // 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._hair_color)
        painter.drawRoundedRect(
            head_left - int(12 * scale),
            head_top - int(6 * scale),
            head_width + int(24 * scale),
            head_height + int(78 * scale),
            int(32 * scale),
            int(32 * scale),
        )
        painter.setBrush(QColor("#f1c7a7"))
        painter.drawEllipse(head_left, head_top, head_width, head_height)
        painter.setBrush(self._hair_color)
        painter.drawPie(
            head_left - int(3 * scale),
            head_top - int(9 * scale),
            head_width + int(6 * scale),
            int(63 * scale),
            0,
            180 * 16,
        )
        painter.setBrush(self._accent_color)
        bow_y = head_top + int(13 * scale)
        painter.drawEllipse(head_left - int(14 * scale), bow_y, int(23 * scale), int(17 * scale))
        painter.drawEllipse(head_left - int(1 * scale), bow_y, int(23 * scale), int(17 * scale))
        painter.setBrush(self._accent_color.lighter(125))
        painter.drawEllipse(
            head_left + int(4 * scale), bow_y + int(4 * scale), int(9 * scale), int(9 * scale)
        )

        eye_y = head_y - int(3 * scale)
        gaze_x = int(self._gaze_x * 4 * scale)
        gaze_y = int(self._gaze_y * 4 * scale)
        eye_open_ratio = self._eye_open_ratio()
        self._draw_eye(
            painter,
            center_x - int(25 * scale),
            eye_y,
            gaze_x,
            gaze_y,
            scale,
            eye_open_ratio,
        )
        self._draw_eye(
            painter,
            center_x + int(25 * scale),
            eye_y,
            gaze_x,
            gaze_y,
            scale,
            eye_open_ratio,
        )
        self._draw_eyebrows(painter, center_x, eye_y, scale)
        self._draw_mouth(painter, center_x, head_y + int(28 * scale), scale)

        if self._emotion == "embarrassed" and self._intensity > 0:
            painter.setBrush(QColor(230, 119, 131, int(70 + self._intensity * 100)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                center_x - int(50 * scale),
                head_y + int(16 * scale),
                int(20 * scale),
                int(9 * scale),
            )
            painter.drawEllipse(
                center_x + int(30 * scale),
                head_y + int(16 * scale),
                int(20 * scale),
                int(9 * scale),
            )

    def _draw_eye(
        self,
        painter: QPainter,
        center_x: int,
        center_y: int,
        gaze_x: int,
        gaze_y: int,
        scale: float,
        eye_open_ratio: float,
    ) -> None:
        if eye_open_ratio < 0.3:
            painter.setPen(QPen(QColor("#26364e"), max(1, int(2 * scale))))
            painter.drawLine(
                center_x - int(12 * scale),
                center_y,
                center_x + int(12 * scale),
                center_y,
            )
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        eye_height = max(2, int(20 * scale * eye_open_ratio))
        painter.drawEllipse(
            center_x - int(13 * scale),
            center_y - eye_height // 2,
            int(26 * scale),
            eye_height,
        )
        painter.setBrush(QColor("#26364e"))
        pupil_size = max(2, int(10 * scale * eye_open_ratio))
        painter.drawEllipse(
            center_x - pupil_size // 2 + gaze_x,
            center_y - pupil_size // 2 + gaze_y,
            pupil_size,
            pupil_size,
        )

    def _draw_eyebrows(self, painter: QPainter, center_x: int, eye_y: int, scale: float) -> None:
        painter.setPen(QPen(QColor("#3d4d6a"), max(1, int(3 * scale))))
        offset = int(17 * scale)
        if self._emotion == "angry":
            painter.drawLine(
                center_x - int(39 * scale), eye_y - offset, center_x - int(12 * scale), eye_y
            )
            painter.drawLine(
                center_x + int(39 * scale), eye_y - offset, center_x + int(12 * scale), eye_y
            )
        elif self._emotion == "sad":
            painter.drawLine(
                center_x - int(39 * scale), eye_y, center_x - int(12 * scale), eye_y - offset
            )
            painter.drawLine(
                center_x + int(39 * scale), eye_y, center_x + int(12 * scale), eye_y - offset
            )
        else:
            painter.drawLine(
                center_x - int(39 * scale),
                eye_y - offset,
                center_x - int(12 * scale),
                eye_y - offset,
            )
            painter.drawLine(
                center_x + int(12 * scale),
                eye_y - offset,
                center_x + int(39 * scale),
                eye_y - offset,
            )

    def _draw_mouth(self, painter: QPainter, center_x: int, center_y: int, scale: float) -> None:
        painter.setPen(QPen(QColor("#9b4f62"), max(1, int(3 * scale))))
        if self._speaking and self._mouth_open and self._motion_enabled:
            width, height = {
                "a": (22, 18),
                "i": (28, 7),
                "u": (12, 19),
                "e": (22, 11),
                "o": (17, 21),
            }.get(self._mouth_shape, (20, 16))
            painter.setBrush(QColor("#9b4f62"))
            painter.drawEllipse(
                center_x - int(width * scale / 2),
                center_y - int(height * scale / 2),
                int(width * scale),
                int(height * scale),
            )
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._emotion == "surprised":
            painter.setBrush(QColor("#9b4f62"))
            painter.drawEllipse(
                center_x - int(8 * scale),
                center_y - int(7 * scale),
                int(16 * scale),
                int(18 * scale),
            )
            return

        path = QPainterPath(QPointF(center_x - int(19 * scale), center_y))
        curve = int(13 * scale * max(0.35, self._intensity))
        if self._emotion == "happy":
            path.quadTo(
                QPointF(center_x, center_y + curve), QPointF(center_x + int(19 * scale), center_y)
            )
        elif self._emotion in {"sad", "angry"}:
            path.quadTo(
                QPointF(center_x, center_y - curve), QPointF(center_x + int(19 * scale), center_y)
            )
        else:
            path.lineTo(QPointF(center_x + int(16 * scale), center_y))
        painter.drawPath(path)


def _gesture_kind(gesture: str) -> str:
    normalized = gesture.lower()
    if "wave" in normalized or "手を振" in gesture:
        return "wave"
    if "nod" in normalized or "うなず" in gesture:
        return "nod"
    return "none"


def _gaze_target(gaze: str) -> tuple[float, float]:
    """自由文でも安全に扱える、少数の視線方向への変換。"""

    normalized = gaze.lower()
    if "left" in normalized or "左" in gaze:
        return -1.0, 0.0
    if "right" in normalized or "右" in gaze:
        return 1.0, 0.0
    if "down" in normalized or "下" in gaze:
        return 0.0, 1.0
    if "up" in normalized or "上" in gaze:
        return 0.0, -1.0
    return 0.0, 0.0


def _close_to_target(current_x: float, current_y: float, target_x: float, target_y: float) -> bool:
    return abs(current_x - target_x) < 0.01 and abs(current_y - target_y) < 0.01
