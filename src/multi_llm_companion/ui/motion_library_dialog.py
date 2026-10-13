"""ローカルVMDを検索、試写、有効化する管理画面。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.ui._dialog_layout import _fit_dialog
from multi_llm_companion.ui.models import (
    ChatBackend,
    UiMotionLibraryItem,
    UiVmdMotion,
)

_TYPE_LABELS = {
    "idle": "待機",
    "talk": "会話",
    "reaction": "反応",
    "dance": "ダンス",
}
_ENERGY_LABELS = {"low": "低", "medium": "中", "high": "高"}


class MotionLibraryDialog(QDialog):
    """本番の会話状態と分離して、カタログ内のVMDを管理する。"""

    def __init__(
        self,
        backend: ChatBackend,
        start_preview: Callable[[UiVmdMotion], None],
        stop_preview: Callable[[], None],
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._start_preview = start_preview
        self._stop_preview = stop_preview
        self._motions: list[UiMotionLibraryItem] = []
        self._current: UiMotionLibraryItem | None = None
        self._initial_preview_pending = True
        self.catalog_changed = False
        self.setWindowTitle("モーションの管理")
        self._build_ui()
        self._reload()
        _fit_dialog(self, 680, 640)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        self._body_layout = QHBoxLayout()
        outer.addLayout(self._body_layout, 1)
        panel = QWidget()
        self._body_layout.addWidget(panel, 1)
        root = QVBoxLayout(panel)
        explanation = QLabel(
            "選ぶと全身で一度再生します。再生だけでは設定は変わりません。"
            "「会話・待機で使用する」のチェックはすぐ保存されます。"
        )
        explanation.setWordWrap(True)
        root.addWidget(explanation)

        filters = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("動きの名前を検索")
        root.addWidget(self.search_edit)
        self.type_combo = QComboBox()
        self.type_combo.addItem("すべての分類", "")
        for value, label in _TYPE_LABELS.items():
            self.type_combo.addItem(label, value)
        self.mood_combo = QComboBox()
        self.mood_combo.addItem("すべての雰囲気", "")
        self.energy_combo = QComboBox()
        self.energy_combo.addItem("すべての強さ", "")
        for value, label in _ENERGY_LABELS.items():
            self.energy_combo.addItem(label, value)
        filters.addWidget(self.type_combo)
        filters.addWidget(self.mood_combo)
        filters.addWidget(self.energy_combo)
        root.addLayout(filters)

        self.motion_list = QListWidget()
        self.motion_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.motion_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        root.addWidget(self.motion_list, 1)

        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setMaximumHeight(170)
        details = QWidget()
        details_scroll.setWidget(details)
        form = QFormLayout(details)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.id_label = QLabel("-")
        self.source_name_label = QLabel("-")
        self.source_name_label.setWordWrap(True)
        self.action_label = QLabel("-")
        self.type_label = QLabel("-")
        self.moods_label = QLabel("-")
        self.energy_label = QLabel("-")
        self.frames_label = QLabel("-")
        self.audio_label = QLabel("-")
        self.status_label = QLabel("-")
        self.status_label.setWordWrap(True)
        self.enabled_check = QCheckBox("会話・待機で使用する")
        form.addRow("元の名称", self.source_name_label)
        form.addRow("内部識別名", self.id_label)
        form.addRow("動作の用途", self.action_label)
        form.addRow("分類", self.type_label)
        form.addRow("雰囲気", self.moods_label)
        form.addRow("強さ", self.energy_label)
        form.addRow("長さ", self.frames_label)
        form.addRow("音源", self.audio_label)
        root.addWidget(self.enabled_check)
        root.addWidget(self.status_label)
        details_button = QPushButton("選んだ動きの詳細")
        details_button.setCheckable(True)
        details_button.toggled.connect(details_scroll.setVisible)
        root.addWidget(details_button)
        root.addWidget(details_scroll)
        details_scroll.hide()

        actions = QHBoxLayout()
        self.preview_button = QPushButton("プレビュー再生")
        self.stop_button = QPushButton("プレビュー停止")
        close_button = QPushButton("閉じる")
        actions.addWidget(self.preview_button)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        actions.addWidget(close_button)
        outer.addLayout(actions)

        self.search_edit.textChanged.connect(self._apply_filters)
        self.type_combo.currentIndexChanged.connect(self._apply_filters)
        self.mood_combo.currentIndexChanged.connect(self._apply_filters)
        self.energy_combo.currentIndexChanged.connect(self._apply_filters)
        self.motion_list.currentItemChanged.connect(self._motion_changed)
        self.enabled_check.toggled.connect(self._enabled_changed)
        self.preview_button.clicked.connect(self._preview)
        self.stop_button.clicked.connect(self._stop_preview)
        close_button.clicked.connect(self.accept)

    def _reload(self, select_id: str = "") -> None:
        try:
            self._motions = self._backend.list_motion_library()
        except Exception as exc:
            self._motions = []
            self._show_error(str(exc) or "モーション一覧を読み込めませんでした。")
        current_mood = str(self.mood_combo.currentData())
        moods = sorted({mood for motion in self._motions for mood in motion.moods})
        self.mood_combo.blockSignals(True)
        self.mood_combo.clear()
        self.mood_combo.addItem("すべての雰囲気", "")
        for mood in moods:
            self.mood_combo.addItem(mood, mood)
        mood_index = self.mood_combo.findData(current_mood)
        self.mood_combo.setCurrentIndex(max(0, mood_index))
        self.mood_combo.blockSignals(False)
        self._apply_filters(select_id)

    def _apply_filters(self, select_id: str = "") -> None:
        query = self.search_edit.text().strip().casefold()
        motion_type = str(self.type_combo.currentData())
        mood = str(self.mood_combo.currentData())
        energy = str(self.energy_combo.currentData())
        previous_id = select_id or (self._current.motion_id if self._current else "")
        self.motion_list.blockSignals(True)
        self.motion_list.clear()
        selected: QListWidgetItem | None = None
        for motion in self._motions:
            searchable = " ".join(
                (motion.source_name, motion.motion_id, motion.action, *motion.moods)
            ).casefold()
            if query and query not in searchable:
                continue
            if motion_type and motion.motion_type != motion_type:
                continue
            if mood and mood not in motion.moods:
                continue
            if energy and motion.energy != energy:
                continue
            enabled = "使用する" if motion.enabled else "使用しない"
            available = "利用可能" if motion.available else "利用不可"
            label = (
                f"{motion.source_name or motion.motion_id}"
                f"  /  {_TYPE_LABELS.get(motion.motion_type, motion.motion_type)}"
                f"  /  {motion.action}  /  {enabled}・{available}"
            )
            item = QListWidgetItem(label)
            if not motion.enabled or not motion.available:
                item.setForeground(QColor("#718396"))
            item.setToolTip(label + "\n" + motion.status_text)
            item.setData(Qt.ItemDataRole.UserRole, motion)
            self.motion_list.addItem(item)
            if motion.motion_id == previous_id:
                selected = item
        self.motion_list.blockSignals(False)
        if selected is not None:
            self.motion_list.setCurrentItem(selected)
        elif self.motion_list.count():
            self.motion_list.setCurrentRow(0)
        else:
            self._select_motion(None)

    def _motion_changed(
        self,
        item: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        motion = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        self._select_motion(motion if isinstance(motion, UiMotionLibraryItem) else None)

    def _select_motion(self, motion: UiMotionLibraryItem | None) -> None:
        previous_id = self._current.motion_id if self._current else None
        self._current = motion
        if motion is None:
            for label in (
                self.id_label,
                self.source_name_label,
                self.action_label,
                self.type_label,
                self.moods_label,
                self.energy_label,
                self.frames_label,
                self.audio_label,
                self.status_label,
            ):
                label.setText("-")
            self.enabled_check.blockSignals(True)
            self.enabled_check.setChecked(False)
            self.enabled_check.blockSignals(False)
            self.enabled_check.setEnabled(False)
            self.preview_button.setEnabled(False)
            self._stop_preview()
            return
        self.id_label.setText(motion.motion_id)
        self.source_name_label.setText(motion.source_name or "記録なし")
        self.action_label.setText(motion.action)
        self.type_label.setText(_TYPE_LABELS.get(motion.motion_type, motion.motion_type))
        self.moods_label.setText("、".join(motion.moods) or "指定なし")
        self.energy_label.setText(_ENERGY_LABELS.get(motion.energy, motion.energy))
        self.frames_label.setText(f"{motion.frame_count}フレーム")
        self.audio_label.setText("必要" if motion.audio_required else "不要")
        self.status_label.setText(motion.status_text)
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(motion.enabled)
        self.enabled_check.blockSignals(False)
        self.enabled_check.setEnabled(True)
        self.preview_button.setEnabled(motion.available)
        if self.isVisible() and motion.motion_id != previous_id:
            if motion.available:
                self._preview()
            else:
                self._stop_preview()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._initial_preview_pending:
            self._initial_preview_pending = False
            self._preview()

    def _enabled_changed(self, enabled: bool) -> None:
        if self._current is None:
            return
        motion_id = self._current.motion_id
        self._stop_preview()
        try:
            self._backend.set_motion_enabled(motion_id, enabled)
            self.catalog_changed = True
        except Exception as exc:
            self._show_error(str(exc) or "モーション設定を変更できませんでした。")
        self._reload(motion_id)

    def _preview(self) -> None:
        if self._current is None or not self._current.available:
            return
        try:
            motion = self._backend.load_motion_preview(self._current.motion_id)
            self._start_preview(replace(motion, loop=False))
        except Exception as exc:
            self._show_error(str(exc) or "モーションをプレビューできませんでした。")

    def done(self, result: int) -> None:
        self._stop_preview()
        super().done(result)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "モーションの管理", message)
