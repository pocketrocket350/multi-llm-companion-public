"""マイクと音声認識の設定画面。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.ui._dialog_layout import _fit_dialog
from multi_llm_companion.ui.models import SpeechSettings


class SettingsDialog(QDialog):
    def __init__(
        self,
        speech_settings: SpeechSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("マイクの設定")

        explanation = QLabel(
            "音声を文字にする方式とマイクを選びます。保存するとすぐ反映され、次の起動時も引き継ぎます。"
            "AIや読み上げ音声の接続先は、設定の「環境設定」で変更できます。"
        )
        explanation.setWordWrap(True)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Whisper（高精度・標準）", "whisper")
        self.engine_combo.addItem("Vosk（軽量）", "vosk")
        engine_index = self.engine_combo.findData(speech_settings.engine)
        if engine_index >= 0:
            self.engine_combo.setCurrentIndex(engine_index)
        form.addRow("音声認識:", self.engine_combo)

        self.microphone_combo = QComboBox()
        self.microphone_combo.addItem("Windowsの既定マイク", None)
        for device in speech_settings.devices:
            self.microphone_combo.addItem(device.name, device.id)
        device_index = self.microphone_combo.findData(speech_settings.input_device)
        if device_index >= 0:
            self.microphone_combo.setCurrentIndex(device_index)
        form.addRow("使用するマイク:", self.microphone_combo)
        for combo in (self.engine_combo, self.microphone_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(12)
            combo.currentTextChanged.connect(combo.setToolTip)
            combo.setToolTip(combo.currentText())

        self.silence_threshold_spin = QDoubleSpinBox()
        self.silence_threshold_spin.setRange(0.001, 1.0)
        self.silence_threshold_spin.setDecimals(3)
        self.silence_threshold_spin.setSingleStep(0.005)
        self.silence_threshold_spin.setValue(speech_settings.silence_threshold)
        form.addRow("発話の音量しきい値:", self.silence_threshold_spin)

        self.silence_duration_spin = QDoubleSpinBox()
        self.silence_duration_spin.setRange(0.3, 10.0)
        self.silence_duration_spin.setDecimals(1)
        self.silence_duration_spin.setSingleStep(0.1)
        self.silence_duration_spin.setSuffix(" 秒")
        self.silence_duration_spin.setValue(speech_settings.silence_duration)
        form.addRow("発話終了までの無音時間:", self.silence_duration_spin)

        self.max_recording_duration_spin = QDoubleSpinBox()
        self.max_recording_duration_spin.setRange(3.0, 120.0)
        self.max_recording_duration_spin.setDecimals(0)
        self.max_recording_duration_spin.setSuffix(" 秒")
        self.max_recording_duration_spin.setValue(speech_settings.max_recording_duration)
        form.addRow("最長録音時間:", self.max_recording_duration_spin)

        self.partial_interval_spin = QDoubleSpinBox()
        self.partial_interval_spin.setRange(1.0, 15.0)
        self.partial_interval_spin.setDecimals(1)
        self.partial_interval_spin.setSingleStep(0.5)
        self.partial_interval_spin.setSuffix(" 秒")
        self.partial_interval_spin.setValue(speech_settings.partial_interval)
        form.addRow("途中認識の更新間隔:", self.partial_interval_spin)

        audio_note = QLabel(
            "Whisperは精度重視、Voskは軽さ重視です。Whisperの初回利用時はモデルを取得します。"
        )
        audio_note.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        scroll.setWidget(page)
        root.addWidget(scroll, 1)
        layout = QVBoxLayout(page)
        layout.addWidget(explanation)
        layout.addLayout(form)
        layout.addWidget(audio_note)
        layout.addStretch()
        root.addWidget(buttons)
        _fit_dialog(self, 550, 570)

    def selected_speech_engine(self) -> str:
        return str(self.engine_combo.currentData())

    def selected_microphone(self) -> int | None:
        value = self.microphone_combo.currentData()
        return value if isinstance(value, int) else None

    def selected_speech_turn_settings(self) -> tuple[float, float, float, float]:
        return (
            self.silence_threshold_spin.value(),
            self.silence_duration_spin.value(),
            self.max_recording_duration_spin.value(),
            self.partial_interval_spin.value(),
        )
