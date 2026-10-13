"""管理画面のサイズとプレビューを、既存の描画に合わせる。"""

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QWidget


def _fit_dialog(dialog: QDialog, width: int, height: int) -> None:
    """タスクバーとタイトルバーの余白を残し、最大化も可能にする。"""
    dialog.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
    dialog.setSizeGripEnabled(True)
    area = dialog.screen().availableGeometry()
    dialog.resize(min(width, area.width() - 32), min(height, area.height() - 64))


class _AvatarPreview(QLabel):
    """現在の描画を共有し、別のPMX読込や描画器を作らない。"""

    def __init__(self, source: QWidget, parent: QWidget) -> None:
        super().__init__(parent)
        self._source = source
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #D8EEF6; border-radius: 8px;")
        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start()
        self._refresh()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _refresh(self) -> None:
        if self.isVisible() and self._source.isVisible():
            ratio = self.devicePixelRatioF()
            pixmap = self._source.grab().scaled(
                QSize(round(self.width() * ratio), round(self.height() * ratio)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pixmap.setDevicePixelRatio(ratio)
            self.setPixmap(pixmap)
