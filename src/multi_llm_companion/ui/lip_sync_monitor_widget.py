"""推定した口形を大きく確認するための、素材を使わない表示部品。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class LipSyncMonitorWidget(QWidget):
    """PMX簡易プレビューでも口形を確認できる母音モニター。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mouth_shape = "neutral"
        self.setMinimumHeight(96)
        self.setMaximumHeight(112)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setToolTip("文章から推定した口形の確認用表示です。PMXのモーフはまだ変形しません。")

    @property
    def mouth_shape(self) -> str:
        return self._mouth_shape

    def set_lip_sync(self, mouth_shape: str) -> None:
        """あ・い・う・え・おの口形だけを安全に表示する。"""

        self._mouth_shape = mouth_shape if mouth_shape in {"a", "i", "u", "e", "o"} else "neutral"
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7f1f5"))
        painter.setPen(QPen(QColor("#d9bfd0"), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 8, 8)

        painter.setPen(QColor("#583c50"))
        painter.drawText(8, 20, "口形モニター（推定）")
        label = _mouth_label(self._mouth_shape)
        painter.setPen(QColor("#9b4f62"))
        painter.drawText(8, 43, f"現在: {label}")

        width, height = _mouth_size(self._mouth_shape)
        center_x = self.width() - 49
        center_y = 51
        painter.setPen(QPen(QColor("#9b4f62"), 3))
        painter.setBrush(
            QColor("#b95d74") if self._mouth_shape != "neutral" else Qt.BrushStyle.NoBrush
        )
        if self._mouth_shape == "neutral":
            painter.drawLine(center_x - 20, center_y, center_x + 20, center_y)
        else:
            painter.drawEllipse(center_x - width // 2, center_y - height // 2, width, height)

        for index, vowel in enumerate(("a", "i", "u", "e", "o")):
            x = 12 + index * 34
            active = vowel == self._mouth_shape
            painter.setPen(QPen(QColor("#9b4f62") if active else QColor("#9b8791"), 1))
            painter.setBrush(QColor("#f2c8d7") if active else QColor("#ffffff"))
            painter.drawEllipse(x, 67, 22, 20)
            painter.setPen(QColor("#583c50"))
            painter.drawText(
                x,
                67,
                22,
                20,
                Qt.AlignmentFlag.AlignCenter,
                _mouth_label(vowel),
            )
        painter.end()


def _mouth_size(shape: str) -> tuple[int, int]:
    return {
        "a": (34, 28),
        "i": (42, 10),
        "u": (16, 31),
        "e": (34, 17),
        "o": (25, 34),
    }.get(shape, (0, 0))


def _mouth_label(shape: str) -> str:
    return {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(shape, "閉じる")
