"""利用者が保存済みの記憶を後から管理する画面。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.ui._dialog_layout import _fit_dialog
from multi_llm_companion.ui.models import ChatBackend, UiMemory

_CATEGORY_ITEMS = [
    ("好み・習慣", "preference"),
    ("プロフィール", "profile"),
    ("予定・目標", "plan"),
    ("会話からの記憶", "relationship"),
    ("その他の記憶", "other"),
]


class MemoryDialog(QDialog):
    """保存済みの記憶を確認、編集、削除する。"""

    def __init__(self, backend: ChatBackend, parent: object | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._current: UiMemory | None = None
        self.setWindowTitle("記憶の管理")
        self._build_ui()
        self._reload()
        _fit_dialog(self, 960, 650)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.enabled_check = QCheckBox("今後の会話で保存した記憶を使う")
        self.auto_approve_check = QCheckBox("候補を自動承認する")
        root.addWidget(self.enabled_check)
        root.addWidget(self.auto_approve_check)
        note = QLabel(
            "チェックと承認・削除はすぐ反映されます。内容の編集は「修正を保存」で反映します。"
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("記憶の内容を検索")
        root.addWidget(self.search_edit)
        self.selection_label = QLabel()
        root.addWidget(self.selection_label)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.splitter, 1)
        self.memory_list = QListWidget()
        self.memory_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.memory_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.memory_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.memory_list.setMinimumWidth(180)
        self.splitter.addWidget(self.memory_list)
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        detail = QWidget()
        self.detail_scroll.setWidget(detail)
        self.splitter.addWidget(self.detail_scroll)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([520, 360])
        form = QFormLayout(detail)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.content_edit = QTextEdit()
        self.content_edit.setFixedHeight(140)
        self.category_combo = QComboBox()
        for label, value in _CATEGORY_ITEMS:
            self.category_combo.addItem(label, value)
        self.updated_label = QLabel("-")
        self.created_label = QLabel("-")
        self.source_label = QLabel("-")
        self.source_label.setWordWrap(True)
        self.status_label = QLabel("-")
        form.addRow("内容", self.content_edit)
        form.addRow("分類", self.category_combo)
        form.addRow("状態", self.status_label)
        form.addRow("作成日時", self.created_label)
        form.addRow("最終更新", self.updated_label)
        form.addRow("根拠の会話", self.source_label)
        self.save_button = QPushButton("修正を保存")
        form.addRow(self.save_button)

        actions = QHBoxLayout()
        self.approve_button = QPushButton("選択した候補を承認")
        self.delete_button = QPushButton("選んだ記憶を削除")
        self.delete_all_button = QPushButton("すべての記憶を削除")
        close_button = QPushButton("閉じる")
        actions.addWidget(self.approve_button)
        actions.addWidget(self.delete_button)
        actions.addStretch()
        actions.addWidget(self.delete_all_button)
        actions.addWidget(close_button)
        root.addLayout(actions)

        self.enabled_check.toggled.connect(self._enabled_changed)
        self.auto_approve_check.toggled.connect(self._auto_approve_changed)
        self.memory_list.itemSelectionChanged.connect(self._memory_changed)
        self.search_edit.textChanged.connect(self._filter_memories)
        self.save_button.clicked.connect(self._save)
        self.approve_button.clicked.connect(self._approve)
        self.delete_button.clicked.connect(self._delete)
        self.delete_all_button.clicked.connect(self._delete_all)
        close_button.clicked.connect(self.accept)

    def _reload(self, select_id: int | None = None) -> None:
        settings = self._backend.memory_settings()
        self.enabled_check.blockSignals(True)
        self.enabled_check.setChecked(settings.enabled)
        self.enabled_check.blockSignals(False)
        self.auto_approve_check.blockSignals(True)
        self.auto_approve_check.setChecked(settings.auto_approve_candidates)
        self.auto_approve_check.blockSignals(False)
        self.memory_list.blockSignals(True)
        self.memory_list.clear()
        selected: QListWidgetItem | None = None
        for memory in self._backend.list_memories():
            state = "確認待ち" if memory.status != "approved" else "承認済み"
            item = QListWidgetItem(f"［{state}］ {memory.content}")
            item.setToolTip(memory.content)
            item.setData(Qt.ItemDataRole.UserRole, memory)
            self.memory_list.addItem(item)
            if memory.id == select_id:
                selected = item
        self.memory_list.blockSignals(False)
        if selected is not None:
            self.memory_list.setCurrentItem(selected)
        self._filter_memories()

    def _selected_memories(self) -> list[UiMemory]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.memory_list.selectedItems()
            if not item.isHidden()
        ]

    def _filter_memories(self, _text: str = "") -> None:
        query = self.search_edit.text().strip().casefold()
        for row in range(self.memory_list.count()):
            item = self.memory_list.item(row)
            hidden = query not in item.text().casefold()
            if hidden:
                item.setSelected(False)
            item.setHidden(hidden)
        self._memory_changed()

    def _memory_changed(self) -> None:
        selected = self._selected_memories()
        self._select_memory(selected[0] if len(selected) == 1 else None)
        self.selection_label.setText(
            f"{len(selected)}件を選択 / 全{self.memory_list.count()}件　Ctrlで追加・Shiftで範囲選択"
        )
        self.approve_button.setEnabled(any(m.status != "approved" for m in selected))
        self.delete_button.setEnabled(bool(selected))
        self.delete_button.setText(
            f"選択した{len(selected)}件を削除" if selected else "選んだ記憶を削除"
        )
        if len(selected) > 1:
            self.content_edit.setPlaceholderText("内容を編集するときは1件だけ選んでください。")
        else:
            self.content_edit.setPlaceholderText("一覧から記憶を選んでください。")

    def _select_memory(self, memory: object) -> None:
        self._current = memory if isinstance(memory, UiMemory) else None
        if self._current is None:
            self.content_edit.clear()
            self.category_combo.setCurrentIndex(4)
            self.updated_label.setText("-")
            self.created_label.setText("-")
            self.source_label.setText("-")
            self.status_label.setText("-")
        else:
            self.content_edit.setPlainText(self._current.content)
            index = self.category_combo.findData(self._current.category)
            self.category_combo.setCurrentIndex(index if index >= 0 else 4)
            self.updated_label.setText(self._current.updated_at or "-")
            self.created_label.setText(self._current.created_at or "-")
            self.source_label.setText(
                self._conversation_label(self._current.source_conversation_id)
            )
            self.status_label.setText(
                "承認済み" if self._current.status == "approved" else "確認待ち"
            )
        self.save_button.setEnabled(self._current is not None)
        self.content_edit.setEnabled(self._current is not None)
        self.category_combo.setEnabled(self._current is not None)
        self.approve_button.setEnabled(
            self._current is not None and self._current.status != "approved"
        )
        self.delete_button.setEnabled(self._current is not None)

    def _enabled_changed(self, enabled: bool) -> None:
        try:
            self._backend.set_memory_enabled(enabled)
        except Exception as exc:
            self._show_error(str(exc) or "記憶の設定を変更できませんでした。")
            self._reload()

    def _auto_approve_changed(self, enabled: bool) -> None:
        try:
            self._backend.set_memory_auto_approve_candidates(enabled)
        except Exception as exc:
            self._show_error(str(exc) or "自動承認の設定を変更できませんでした。")
            self._reload()

    def _save(self) -> None:
        if self._current is None:
            return
        try:
            updated = self._backend.update_memory(
                self._current.id,
                self.content_edit.toPlainText(),
                str(self.category_combo.currentData()),
            )
        except Exception as exc:
            self._show_error(str(exc) or "記憶を変更できませんでした。")
            return
        self._reload(updated.id)

    def _approve(self) -> None:
        selected = self._selected_memories()
        if not selected:
            return
        completed = 0
        try:
            for memory in selected:
                if memory.status == "approved":
                    continue
                self._backend.approve_memory(
                    memory.id,
                    self.content_edit.toPlainText() if len(selected) == 1 else memory.content,
                    str(self.category_combo.currentData())
                    if len(selected) == 1
                    else memory.category,
                )
                completed += 1
        except Exception as exc:
            self._reload()
            self._show_error(f"{completed}件を承認しました。残りを承認できませんでした。\n{exc}")
            return
        self._reload(selected[0].id if len(selected) == 1 else None)

    def _delete(self) -> None:
        selected = self._selected_memories()
        if not selected:
            return
        if len(selected) > 1:
            answer = QMessageBox.question(
                self,
                "選択した記憶を削除",
                f"選択した{len(selected)}件を削除します。この操作は元に戻せません。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        completed = 0
        try:
            for memory in selected:
                self._backend.delete_memory(memory.id)
                completed += 1
        except Exception as exc:
            self._reload()
            self._show_error(f"{completed}件を削除しました。残りを削除できませんでした。\n{exc}")
            return
        self._reload()

    def _delete_all(self) -> None:
        answer = QMessageBox.question(
            self,
            "すべての記憶を削除",
            "保存済みの記憶をすべて削除します。この操作は元に戻せません。よろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._backend.delete_all_memories()
        except Exception as exc:
            self._show_error(str(exc) or "記憶を削除できませんでした。")
            return
        self._reload()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "記憶の管理", message)

    def _conversation_label(self, conversation_id: str) -> str:
        if not conversation_id:
            return "記録なし"
        for conversation in self._backend.list_conversations():
            if conversation.id == conversation_id:
                return conversation.title
        return "削除済みの会話"
