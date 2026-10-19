"""会話の配置や入力フォーカスを変えずに、一時的な案内を表示する。"""

from PySide6.QtCore import QEvent, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

NOTICE_DEFAULT_SECONDS = {"memory": 6.0, "info": 3.5}


class FloatingNotice(QFrame):
    action_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.kind = ""
        self._remaining_ms = 0
        self._preferences = {
            kind: (True, round(seconds * 1000)) for kind, seconds in NOTICE_DEFAULT_SECONDS.items()
        }
        self.setObjectName("floatingNotice")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(
            "QFrame#floatingNotice {background: #F6FCFD; border: 1px solid #B6D4DF;"
            " border-radius: 10px;}"
            "QLabel {color: #40586B; font-size: 13px; background: transparent;}"
            "QPushButton {color: #40586B; background: transparent; border: none;"
            " padding: 3px 6px; font-size: 12px;}"
            "QPushButton:hover {background: #E4F2F5; border-radius: 5px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 8, 9)
        layout.setSpacing(3)
        row = QHBoxLayout()
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        row.addWidget(self.label, 1)
        self.close_button = QPushButton("×")
        self.close_button.setAccessibleName("通知を閉じる")
        self.close_button.setFixedSize(24, 24)
        row.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)
        self.action_button = QPushButton("記憶を管理")
        self.action_button.setAccessibleName("通知から記憶を管理")
        layout.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignLeft)
        for button in (self.close_button, self.action_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.dismiss)
        self.action_button.clicked.connect(self.action_requested)
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._animation = QPropertyAnimation(self._opacity, b"opacity", self)
        self._animation.finished.connect(self._animation_finished)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        parent.installEventFilter(self)
        self.hide()

    def configure(self, kind: str, *, enabled: bool, seconds: float) -> None:
        self._preferences[kind] = (enabled, round(seconds * 1000))
        if not enabled and self.kind == kind:
            self.dismiss()

    def show_notice(self, message: str, *, kind: str = "info") -> None:
        enabled, duration_ms = self._preferences[kind]
        if not enabled:
            return
        # 記憶の案内を短い処理通知で上書きせず、通知を積み上げない。
        if not message or (kind == "info" and self.kind == "memory" and not self.isHidden()):
            return
        self._timer.stop()
        self._animation.stop()
        self.kind = kind
        self.label.setText(message)
        self.setAccessibleName(message)
        self.action_button.setVisible(kind == "memory")
        self._place()
        self._opacity.setOpacity(0.0)
        self.show()
        self.raise_()
        self._animate_to(1.0, 180)
        self._remaining_ms = duration_ms
        if not self.underMouse():
            self._timer.start(self._remaining_ms)

    def dismiss(self) -> None:
        self._timer.stop()
        self._animation.stop()
        self.hide()
        self.label.clear()
        self.setAccessibleName("")
        self.kind = ""

    def _place(self) -> None:
        available = max(1, self.parentWidget().width() - 24)
        self.setFixedWidth(min(380, available))
        self.layout().activate()
        self.adjustSize()
        self.move(max(0, self.parentWidget().width() - self.width() - 12), 12)

    def _animate_to(self, opacity: float, duration: int) -> None:
        self._animation.stop()
        self._animation.setDuration(duration)
        self._animation.setStartValue(self._opacity.opacity())
        self._animation.setEndValue(opacity)
        self._animation.start()

    def _fade_out(self) -> None:
        self._animate_to(0.0, 450)

    def _animation_finished(self) -> None:
        if self._opacity.opacity() == 0.0:
            self.dismiss()

    def enterEvent(self, event) -> None:  # noqa: N802
        if self._timer.isActive():
            self._remaining_ms = max(1500, self._timer.remainingTime())
        self._timer.stop()
        self._animation.stop()
        self._opacity.setOpacity(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if not self.isHidden():
            self._timer.start(self._remaining_ms)
        super().leaveEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._place()
        return super().eventFilter(watched, event)
