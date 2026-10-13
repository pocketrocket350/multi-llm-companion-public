"""雨衣PMXに実在するモーフを確認し、会話用の組合せとして保存する画面。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.ui._dialog_layout import _fit_dialog
from multi_llm_companion.ui.models import UiAvatarModelPreview, UiExpressionPreset

_EMOTION_LABELS = {
    "neutral": "通常",
    "happy": "うれしい",
    "sad": "悲しい",
    "angry": "怒り",
    "embarrassed": "照れ",
    "surprised": "驚き",
}


def mouth_shape_for_morph(preview: UiAvatarModelPreview, morph_name: str) -> str:
    """実際に読んだ頂点データから、対応する口形を逆引きする。"""

    offsets = preview.named_vertex_morphs.get(morph_name)
    if offsets is None:
        return ""
    return next(
        (
            shape
            for shape, mouth_offsets in preview.mouth_morphs.items()
            if mouth_offsets == offsets
        ),
        "",
    )


class ExpressionLibraryDialog(QDialog):
    """実在モーフを名前・形式で探し、組合せを確認・保存する。"""

    def __init__(
        self,
        preview: UiAvatarModelPreview,
        start_preview: Callable[[str], None],
        stop_preview: Callable[[], None],
        parent: object | None = None,
        save_preset: Callable[[str, str, dict[str, float]], None] | None = None,
        start_combination_preview: Callable[[dict[str, float]], None] | None = None,
        load_presets: Callable[[], list[UiExpressionPreset]] | None = None,
        delete_preset: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_preview = preview
        self._start_preview = start_preview
        self._stop_preview = stop_preview
        self._save_preset = save_preset
        self._start_combination_preview = start_combination_preview
        self._load_presets = load_presets
        self._delete_preset = delete_preset
        self._presets: dict[str, UiExpressionPreset] = {}
        self._selected_name = ""
        self._selected_kind = ""
        self._combination: dict[str, float] = {}
        self._requires_emotion_choice = False
        self.setWindowTitle("表情の管理")
        self._build_ui()
        self._refresh_saved_presets()
        self._apply_filters()
        _fit_dialog(self, 620, 640)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self._body_layout = QHBoxLayout()
        root.addLayout(self._body_layout, 1)
        self.tabs = QTabWidget()
        self._body_layout.addWidget(self.tabs, 1)
        browse = QWidget()
        browse_layout = QVBoxLayout(browse)
        note = QLabel("一覧で選ぶと雨衣の表情が変わります。見るだけでは設定は変わりません。")
        note.setWordWrap(True)
        browse_layout.addWidget(note)
        self.saved_preset_list = QListWidget()
        browse_layout.addWidget(self.saved_preset_list, 1)
        self.saved_preset_detail_label = QLabel("保存済みの表情を選んでください。")
        self.saved_preset_detail_label.setWordWrap(True)
        browse_layout.addWidget(self.saved_preset_detail_label)
        saved_actions = QHBoxLayout()
        self.load_preset_button = QPushButton("編集")
        self.duplicate_preset_button = QPushButton("複製")
        self.new_preset_button = QPushButton("新しく作る")
        self.delete_preset_button = QPushButton("削除")
        for button in (
            self.load_preset_button,
            self.duplicate_preset_button,
            self.new_preset_button,
            self.delete_preset_button,
        ):
            saved_actions.addWidget(button)
        browse_layout.addLayout(saved_actions)
        self.tabs.addTab(browse, "表情を選ぶ")

        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        editor = QWidget()
        self.editor_scroll.setWidget(editor)
        edit = QVBoxLayout(editor)
        self.tabs.addTab(self.editor_scroll, "表情を編集")
        explanation = QLabel(
            "目や口などの変形（モーフ）を組み合わせます。保存するまで会話用の表情は変わりません。"
        )
        explanation.setWordWrap(True)
        edit.addWidget(explanation)
        filters = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("目・眉・口などの名前を検索")
        self.kind_combo = QComboBox()
        for label, kind in (
            ("すべて", ""),
            ("口形", "mouth"),
            ("形の変化", "vertex"),
            ("色・材質", "material"),
            ("描画未対応", "unsupported"),
        ):
            self.kind_combo.addItem(label, kind)
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(self.kind_combo)
        edit.addLayout(filters)
        self.morph_list = QListWidget()
        self.morph_list.setFixedHeight(130)
        edit.addWidget(self.morph_list)
        self.detail_label = QLabel("モーフを選んでください。")
        self.detail_label.setWordWrap(True)
        edit.addWidget(self.detail_label)
        morph_actions = QHBoxLayout()
        self.preview_button = QPushButton("このパーツを試す")
        self.add_button = QPushButton("組合せに追加")
        morph_actions.addWidget(self.preview_button)
        morph_actions.addWidget(self.add_button)
        edit.addLayout(morph_actions)
        edit.addWidget(QLabel("編集中の組合せ"))
        self.combination_list = QListWidget()
        self.combination_list.setFixedHeight(110)
        edit.addWidget(self.combination_list)
        combination_actions = QHBoxLayout()
        self.remove_button = QPushButton("外す")
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.01, 1.0)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setValue(1.0)
        self.strength_spin.setPrefix("強さ ")
        self.combination_preview_button = QPushButton("組合せを試す")
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.strength_spin.setEnabled(False)
        self.combination_preview_button.setEnabled(False)
        combination_actions.addWidget(self.remove_button)
        combination_actions.addWidget(self.strength_spin)
        combination_actions.addWidget(self.combination_preview_button)
        edit.addLayout(combination_actions)
        save_form = QFormLayout()
        save_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("例: gentle_smile")
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItem("会話には使わない（保存のみ）", "")
        for emotion, label in _EMOTION_LABELS.items():
            self.emotion_combo.addItem(label, emotion)
        save_form.addRow("保存名", self.preset_name_edit)
        save_form.addRow("会話で使う場面", self.emotion_combo)
        edit.addLayout(save_form)
        self.save_button = QPushButton("新しいプリセットとして保存")
        self.save_button.setEnabled(self._save_preset is not None)
        edit.addWidget(self.save_button)
        actions = QHBoxLayout()
        self.stop_button = QPushButton("プレビューを終了")
        close_button = QPushButton("閉じる")
        self.preview_button.setEnabled(False)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        actions.addWidget(close_button)
        root.addLayout(actions)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.kind_combo.currentIndexChanged.connect(self._apply_filters)
        self.saved_preset_list.currentItemChanged.connect(self._saved_preset_selection_changed)
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.duplicate_preset_button.clicked.connect(self._duplicate_selected_preset)
        self.new_preset_button.clicked.connect(self._new_preset)
        self.delete_preset_button.clicked.connect(self._delete_selected_preset)
        self.morph_list.currentItemChanged.connect(self._selection_changed)
        self.combination_list.currentItemChanged.connect(self._combination_selection_changed)
        self.preview_button.clicked.connect(self._preview)
        self.stop_button.clicked.connect(self._finish_preview)
        self.add_button.clicked.connect(self._add_selected_morph)
        self.remove_button.clicked.connect(self._remove_selected_morph)
        self.strength_spin.valueChanged.connect(self._change_selected_strength)
        self.combination_preview_button.clicked.connect(self._preview_combination)
        self.save_button.clicked.connect(self._save_combination)
        self.preset_name_edit.textChanged.connect(self._update_save_action)
        self.emotion_combo.activated.connect(self._emotion_assignment_chosen)
        close_button.clicked.connect(self.accept)

    def _tab_changed(self, index: int) -> None:
        if index == 0:
            self._preview_saved_preset()
        else:
            self._stop_preview()

    def _preview_saved_preset(self) -> None:
        preset = self._current_saved_preset()
        if preset is not None and preset.morph_weights and self._start_combination_preview:
            self._start_combination_preview(dict(preset.morph_weights))
        else:
            self._stop_preview()

    def _finish_preview(self) -> None:
        self._stop_preview()
        self.saved_preset_list.setCurrentRow(-1)

    def _refresh_saved_presets(self, selected_name: str = "") -> None:
        presets = self._load_presets() if self._load_presets is not None else []
        if not selected_name:
            selected_name = next((p.name for p in presets if "neutral" in p.emotions), "")
        self._presets = {preset.name: preset for preset in presets}
        self.saved_preset_list.blockSignals(True)
        self.saved_preset_list.clear()
        selected: QListWidgetItem | None = None
        for preset in presets:
            meanings = "・".join(_EMOTION_LABELS.get(name, name) for name in preset.emotions)
            item = QListWidgetItem(f"{meanings or '自由な表情'}  —  {preset.name}")
            if not preset.emotions or not preset.morph_weights or preset.unresolved_roles:
                state = "使用しない" if not preset.emotions else "対応するモーフが不足"
                item.setText(item.text() + f"  /  {state}")
                item.setForeground(QColor("#718396"))
            item.setToolTip(item.text())
            item.setData(Qt.ItemDataRole.UserRole, preset.name)
            self.saved_preset_list.addItem(item)
            if preset.name == selected_name:
                selected = item
        self.saved_preset_list.blockSignals(False)
        if selected is not None:
            self.saved_preset_list.setCurrentItem(selected)
        elif self.saved_preset_list.count():
            self.saved_preset_list.setCurrentRow(0)
        else:
            self._saved_preset_selection_changed(None, None)
        self._update_save_action()

    def _saved_preset_selection_changed(
        self,
        item: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        name = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        preset = self._presets.get(name)
        enabled = preset is not None
        self.load_preset_button.setEnabled(enabled)
        self.duplicate_preset_button.setEnabled(enabled)
        self.delete_preset_button.setEnabled(enabled and self._delete_preset is not None)
        if preset is None:
            self.saved_preset_detail_label.setText("保存済みプリセットを選んでください。")
            return
        weights = "、".join(f"{morph} {weight:.2f}" for morph, weight in preset.morph_weights)
        meanings = "・".join(_EMOTION_LABELS.get(name, name) for name in preset.emotions)
        details = [
            f"会話で使う場面: {meanings or 'なし'}",
            f"モーフ: {weights or '解決できるモーフなし'}",
        ]
        if preset.unresolved_roles:
            details.append(f"対応先がない役割: {'、'.join(preset.unresolved_roles)}")
        self.saved_preset_detail_label.setText("\n".join(details))
        if self.tabs.currentIndex() == 0:
            self._preview_saved_preset()

    def _current_saved_preset(self) -> UiExpressionPreset | None:
        item = self.saved_preset_list.currentItem()
        name = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        return self._presets.get(name)

    def _load_selected_preset(self) -> None:
        self._load_preset(copy=False)

    def _duplicate_selected_preset(self) -> None:
        self._load_preset(copy=True)

    def _load_preset(self, *, copy: bool) -> None:
        preset = self._current_saved_preset()
        if preset is None:
            return
        self.tabs.setCurrentIndex(1)
        name = preset.name
        if copy:
            base = f"{name}_copy"
            name = base
            suffix = 2
            while name in self._presets:
                name = f"{base}_{suffix}"
                suffix += 1
        self.preset_name_edit.setText(name)
        emotion = preset.emotions[0] if len(preset.emotions) == 1 else ""
        index = self.emotion_combo.findData(emotion)
        self.emotion_combo.setCurrentIndex(max(0, index))
        self._requires_emotion_choice = len(preset.emotions) > 1
        self._combination = dict(preset.morph_weights)
        self._refresh_combination()
        if len(preset.emotions) > 1:
            self.saved_preset_detail_label.setText(
                self.saved_preset_detail_label.text()
                + " / 複数の大分類に登録されています。保存前に一つ選び直してください。"
            )

    def _new_preset(self) -> None:
        self.tabs.setCurrentIndex(1)
        self.saved_preset_list.clearSelection()
        self.preset_name_edit.clear()
        self.emotion_combo.setCurrentIndex(0)
        self._requires_emotion_choice = False
        self._combination.clear()
        self._refresh_combination()

    def _emotion_assignment_chosen(self, _index: int) -> None:
        self._requires_emotion_choice = False

    def _delete_selected_preset(self) -> None:
        preset = self._current_saved_preset()
        if preset is None or self._delete_preset is None:
            return
        answer = QMessageBox.question(
            self,
            "表情プリセットを削除しますか",
            f"「{preset.name}」を削除します。元に戻すには作り直す必要があります。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._delete_preset(preset.name)
        except RuntimeError as exc:
            QMessageBox.warning(self, "表情を削除できません", str(exc))
            return
        self._new_preset()
        self.tabs.setCurrentIndex(0)
        self._refresh_saved_presets()

    def _update_save_action(self) -> None:
        name = self.preset_name_edit.text().strip().lower()
        self.save_button.setText(
            "同名のプリセットを上書き" if name in self._presets else "新しいプリセットとして保存"
        )

    def _items(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for name in self._model_preview.morph_names:
            if mouth_shape_for_morph(self._model_preview, name):
                kind = "mouth"
            elif name in self._model_preview.named_vertex_morphs:
                kind = "vertex"
            elif name in self._model_preview.named_material_morphs:
                kind = "material"
            else:
                kind = "unsupported"
            result.append((name, kind))
        return result

    def _apply_filters(self) -> None:
        query = self.search_edit.text().strip().casefold()
        kind = str(self.kind_combo.currentData())
        previous = self._selected_name
        self.morph_list.blockSignals(True)
        self.morph_list.clear()
        selected: QListWidgetItem | None = None
        labels = {
            "mouth": "口形",
            "vertex": "頂点",
            "material": "材質・GPUのみ",
            "unsupported": "描画未対応",
        }
        for name, item_kind in self._items():
            if query and query not in name.casefold():
                continue
            if kind and kind != item_kind:
                continue
            item = QListWidgetItem(f"{name}  /  {labels[item_kind]}")
            if item_kind == "unsupported":
                item.setForeground(QColor("#718396"))
            item.setData(Qt.ItemDataRole.UserRole, (name, item_kind))
            self.morph_list.addItem(item)
            if name == previous:
                selected = item
        self.morph_list.blockSignals(False)
        if selected is not None:
            self.morph_list.setCurrentItem(selected)
        elif self.morph_list.count():
            self.morph_list.setCurrentRow(0)
        else:
            self._selection_changed(None, None)

    def _selection_changed(
        self,
        item: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(value, tuple) or len(value) != 2:
            self._selected_name = ""
            self._selected_kind = ""
            self.detail_label.setText("モーフを選んでください。")
            self.preview_button.setEnabled(False)
            self.add_button.setEnabled(False)
            return
        name, kind = value
        self._selected_name = str(name)
        self._selected_kind = str(kind)
        if kind == "unsupported":
            self.detail_label.setText(f"{name}: この描画では確認できません。")
            self.preview_button.setEnabled(False)
            self.add_button.setEnabled(False)
            return
        if kind == "mouth":
            self.detail_label.setText(
                f"{name}: 口形を確認できますが、会話用組合せには追加できません。"
            )
            self.preview_button.setEnabled(True)
            self.add_button.setEnabled(False)
            return
        text = "GPU描画で確認できます。" if kind == "material" else "頂点を直接変形して確認します。"
        self.detail_label.setText(f"{name}: {text}")
        self.preview_button.setEnabled(True)
        self.add_button.setEnabled(True)

    def _preview(self) -> None:
        if self._selected_name:
            self._start_preview(self._selected_name)

    def _add_selected_morph(self) -> None:
        if not self._selected_name or self._selected_kind in {"mouth", "unsupported"}:
            return
        self._combination.setdefault(self._selected_name, 1.0)
        self._refresh_combination(self._selected_name)

    def _remove_selected_morph(self) -> None:
        item = self.combination_list.currentItem()
        name = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        if name:
            self._combination.pop(name, None)
            self._refresh_combination()

    def _preview_combination(self) -> None:
        if self._combination and self._start_combination_preview is not None:
            self._start_combination_preview(dict(self._combination))

    def _combination_selection_changed(
        self,
        item: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        name = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        weight = self._combination.get(name)
        self.remove_button.setEnabled(weight is not None)
        self.strength_spin.setEnabled(weight is not None)
        if weight is not None:
            self.strength_spin.blockSignals(True)
            self.strength_spin.setValue(weight)
            self.strength_spin.blockSignals(False)

    def _change_selected_strength(self, weight: float) -> None:
        item = self.combination_list.currentItem()
        name = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        if name in self._combination:
            self._combination[name] = weight
            self._refresh_combination(name)

    def _refresh_combination(self, selected_name: str = "") -> None:
        self.combination_list.blockSignals(True)
        self.combination_list.clear()
        selected: QListWidgetItem | None = None
        for name, weight in self._combination.items():
            item = QListWidgetItem(f"{name}  /  強さ {weight:.2f}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.combination_list.addItem(item)
            if name == selected_name:
                selected = item
        self.combination_list.blockSignals(False)
        if selected is not None:
            self.combination_list.setCurrentItem(selected)
        elif self.combination_list.count():
            self.combination_list.setCurrentRow(0)
        else:
            self._combination_selection_changed(None, None)
        self.combination_preview_button.setEnabled(
            bool(self._combination) and self._start_combination_preview is not None
        )

    def _save_combination(self) -> None:
        if self._save_preset is None:
            return
        if self._requires_emotion_choice:
            QMessageBox.warning(
                self,
                "会話で使う大分類を選んでください",
                "このプリセットは複数の大分類に登録されていました。"
                "保存前に大分類を一つ選ぶか、保存のみを選んでください。",
            )
            return
        normalized_name = self.preset_name_edit.text().strip().lower()
        if normalized_name in self._presets:
            answer = QMessageBox.question(
                self,
                "同名の表情を上書きしますか",
                f"「{normalized_name}」の組合せと会話で使う大分類を上書きします。",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._save_preset(
                self.preset_name_edit.text(),
                str(self.emotion_combo.currentData()),
                dict(self._combination),
            )
        except RuntimeError as exc:
            QMessageBox.warning(self, "表情を保存できません", str(exc))
            return
        QMessageBox.information(self, "表情を保存しました", "会話用の表情設定へ保存しました。")
        self._refresh_saved_presets(normalized_name)

    def done(self, result: int) -> None:
        self._stop_preview()
        super().done(result)
