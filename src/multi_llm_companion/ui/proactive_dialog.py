"""能動的な話しかけを利用者が明示的に管理する画面。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.services.proactive import ProactiveSettings
from multi_llm_companion.ui._dialog_layout import _fit_dialog


class ProactiveSettingsDialog(QDialog):
    """能動会話の既定オフと、音声許可を明示する。"""

    def __init__(self, settings: ProactiveSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("能動的な話しかけ")
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        scroll.setWidget(page)
        root.addWidget(scroll, 1)
        layout = QVBoxLayout(page)
        self.enabled_check = QCheckBox("能動的な話しかけを有効にする")
        self.enabled_check.setChecked(settings.enabled)
        layout.addWidget(self.enabled_check)
        self.read_aloud_check = QCheckBox("能動発話を読み上げる（現在の音声設定を使用）")
        self.read_aloud_check.setChecked(settings.read_aloud)
        layout.addWidget(self.read_aloud_check)
        description = QLabel(
            "会話がしばらく止まったとき、雨衣が会話欄から短く話しかけます。\n"
            "予定・タスクの登録やリマインダーではありません。保存済みの利用者記憶も自動では使いません。\n"
            "有効中は、現在選んでいるLLMへ人格設定と開いている会話の文脈を送って一回生成します。"
            "読み上げは上の項目をオンにした場合だけです。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.quiet_start = _hour_spin(settings.quiet_start_hour)
        self.quiet_end = _hour_spin(settings.quiet_end_hour)
        self.minimum_idle = _minute_spin(settings.minimum_idle_minutes, 5, 720)
        self.minimum_interval = _minute_spin(settings.minimum_interval_minutes, 15, 1440)
        form.addRow("静かな時間帯の開始", self.quiet_start)
        form.addRow("静かな時間帯の終了", self.quiet_end)
        form.addRow("利用者の発言後の待機（分）", self.minimum_idle)
        form.addRow("話しかける最短間隔（分）", self.minimum_interval)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        _fit_dialog(self, 550, 570)

    def selected_settings(self) -> ProactiveSettings:
        return ProactiveSettings(
            enabled=self.enabled_check.isChecked(),
            read_aloud=self.read_aloud_check.isChecked(),
            quiet_start_hour=self.quiet_start.value(),
            quiet_end_hour=self.quiet_end.value(),
            minimum_idle_minutes=self.minimum_idle.value(),
            minimum_interval_minutes=self.minimum_interval.value(),
        )


def _hour_spin(value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, 23)
    spin.setSuffix(" 時")
    spin.setValue(value)
    return spin


def _minute_spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSuffix(" 分")
    spin.setValue(value)
    return spin
