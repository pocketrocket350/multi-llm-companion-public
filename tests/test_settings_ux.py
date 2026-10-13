from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea
from test_ui import FakeBackend
from test_ui import isolate_window_settings as isolate_window_settings

from multi_llm_companion.ui._dialog_layout import _AvatarPreview
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.expression_library_dialog import ExpressionLibraryDialog
from multi_llm_companion.ui.memory_dialog import MemoryDialog
from multi_llm_companion.ui.models import (
    SpeechSettings,
    UiAudioDevice,
    UiAvatarModelPreview,
    UiExpressionPreset,
    UiMemory,
)
from multi_llm_companion.ui.settings_dialog import SettingsDialog


@pytest.fixture
def memory_dialog():
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.memories = [
        UiMemory(i, f"検証用の記憶 {i}", "profile", "candidate" if i % 2 else "approved")
        for i in range(30)
    ]
    dialog = MemoryDialog(backend)
    dialog.resize(760, 530)
    dialog.show()
    app.processEvents()
    yield dialog, backend
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def select_rows(dialog, *rows):
    dialog.memory_list.clearSelection()
    for row in rows:
        dialog.memory_list.item(row).setSelected(True)


def test_memory_category_choices_are_distinct_and_keep_stored_values(memory_dialog):
    dialog, backend = memory_dialog
    combo = dialog.category_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert len(labels) == len(set(labels))
    select_rows(dialog, 0)
    for category in ("relationship", "other"):
        combo.setCurrentIndex(combo.findData(category))
        dialog.save_button.click()
        assert backend.memories[0].category == category
        assert combo.currentData() == category


def test_memory_has_large_resizable_list_and_safe_batch_approval(memory_dialog):
    dialog, backend = memory_dialog
    assert dialog.memory_list.height() >= 250
    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    select_rows(dialog, 1)
    dialog.content_edit.setPlainText("他の記憶へ使ってはいけない編集文")
    dialog.memory_list.item(3).setSelected(True)
    assert not dialog.content_edit.isEnabled()
    assert not dialog.save_button.isEnabled()
    assert "2件を選択" in dialog.selection_label.text()
    dialog._save()
    dialog.approve_button.click()
    assert backend.memories[1].status == backend.memories[3].status == "approved"
    assert backend.memories[1].content == "検証用の記憶 1"
    assert backend.memories[3].content == "検証用の記憶 3"
    assert backend.memories[5].status == "candidate"


def test_memory_batch_delete_names_count_and_cancel_keeps_data(memory_dialog, monkeypatch):
    dialog, backend = memory_dialog
    select_rows(dialog, 1, 2, 4)
    questions = []

    def decline(_parent, _title, message, *_args):
        questions.append(message)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", decline)
    dialog.delete_button.click()
    assert len(backend.memories) == 30
    assert "3件" in questions[0]
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    dialog.delete_button.click()
    assert {m.id for m in backend.memories} == set(range(30)) - {1, 2, 4}


def test_memory_hidden_selection_is_excluded_from_batch(memory_dialog):
    dialog, backend = memory_dialog
    select_rows(dialog, 1, 3)
    dialog.search_edit.setText("検証用の記憶 3")
    assert [m.id for m in dialog._selected_memories()] == [3]
    dialog.approve_button.click()
    assert backend.memories[3].status == "approved"
    assert backend.memories[1].status == "candidate"


def test_memory_batch_error_reports_partial_progress_and_reloads(memory_dialog, monkeypatch):
    dialog, backend = memory_dialog
    original = backend.delete_memory
    errors = []

    def fail_second(memory_id):
        if memory_id == 2:
            raise RuntimeError("試験用の保存エラー")
        original(memory_id)

    monkeypatch.setattr(backend, "delete_memory", fail_second)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    select_rows(dialog, 1, 2, 3)
    dialog.delete_button.click()
    assert "1件を削除" in errors[0] and "試験用の保存エラー" in errors[0]
    assert {m.id for m in backend.memories} == set(range(30)) - {1}
    assert dialog.memory_list.count() == 29


def test_memory_ctrl_and_shift_selection_and_maximize(memory_dialog):
    dialog, _backend = memory_dialog
    widget = dialog.memory_list
    QTest.mouseClick(
        widget.viewport(),
        Qt.MouseButton.LeftButton,
        pos=widget.visualItemRect(widget.item(1)).center(),
    )
    QTest.mouseClick(
        widget.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        widget.visualItemRect(widget.item(3)).center(),
    )
    assert len(widget.selectedItems()) == 2
    widget.setCurrentRow(2, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    QTest.keyClick(widget, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert len(widget.selectedItems()) == 2
    dialog.showMaximized()
    QTest.qWait(10)
    assert dialog.isMaximized()


def test_expression_selection_previews_without_editing_or_saving_and_close_stops():
    QApplication.instance() or QApplication([])
    calls, saved, stopped = [], [], []
    model = UiAvatarModelPreview("試験", 1, 0, ((0, 0, 0),))
    presets = [
        UiExpressionPreset("one", (("笑い", 0.2),), ("happy",)),
        UiExpressionPreset("two", (("じと目", 0.3),), ("neutral",)),
    ]
    dialog = ExpressionLibraryDialog(
        model,
        lambda _: None,
        lambda: stopped.append(True),
        start_combination_preview=calls.append,
        load_presets=lambda: presets,
        save_preset=lambda *args: saved.append(args),
    )
    dialog.resize(540, 480)
    dialog.show()
    QTest.qWait(10)
    dialog.saved_preset_list.setCurrentRow(1)
    assert calls[-1] == {"じと目": 0.3}
    assert saved == [] and dialog._combination == {}
    assert dialog.tabs.currentIndex() == 0
    dialog.load_preset_button.click()
    assert dialog.tabs.currentIndex() == 1
    assert dialog._combination == {"じと目": 0.3}
    dialog.editor_scroll.ensureWidgetVisible(dialog.save_button)
    assert dialog.editor_scroll.verticalScrollBar().value() > 0
    dialog.tabs.setCurrentIndex(0)
    assert calls[-1] == {"じと目": 0.3}
    dialog.close()
    assert stopped and saved == []
    dialog.deleteLater()


def test_legacy_face_width_and_hidden_status_still_open_usable_settings():
    QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window._settings.setValue("companion_view", "face")
    window._settings.setValue("status_visible", False)
    window._restore_window_preferences()
    window.show()
    window.options_button.click()
    window.options_tabs.setCurrentIndex(2)
    assert window.content_splitter.sizes()[0] > window.content_splitter.sizes()[1]
    assert window._camera_view == "bust"
    assert window.provider_label.isVisible()
    assert not hasattr(window, "status_toggle_button")
    window.close()


def test_unresolved_expression_clears_previous_preview_and_can_restart_after_stop():
    QApplication.instance() or QApplication([])
    calls, stops = [], []
    presets = [
        UiExpressionPreset("usable", (("笑い", 0.3),), ("happy",)),
        UiExpressionPreset("missing", (), ("sad",), ("missing_role",)),
    ]
    dialog = ExpressionLibraryDialog(
        UiAvatarModelPreview("試験", 1, 0, ((0, 0, 0),)),
        lambda _: None,
        lambda: stops.append(True),
        start_combination_preview=calls.append,
        load_presets=lambda: presets,
    )
    assert calls == [{"笑い": 0.3}]
    dialog.saved_preset_list.setCurrentRow(1)
    assert stops == [True]
    assert "解決できるモーフなし" in dialog.saved_preset_detail_label.text()
    dialog.saved_preset_list.setCurrentRow(0)
    dialog.stop_button.click()
    assert dialog.saved_preset_list.currentRow() == -1
    dialog.saved_preset_list.setCurrentRow(0)
    assert calls == [{"笑い": 0.3}] * 3
    dialog.close()


def test_manager_reuses_current_avatar_and_stops_preview_on_close(monkeypatch):
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.avatar_model_preview = lambda: UiAvatarModelPreview("試験", 1, 0, ((0, 0, 0),))
    backend.list_expression_presets = lambda: [
        UiExpressionPreset("one", (("笑い", 0.2),), ("happy",))
    ]
    window = ChatWindow(backend)
    window.show()
    checked = []

    def inspect_and_close():
        dialog = app.activeModalWidget()
        assert isinstance(dialog, ExpressionLibraryDialog)
        preview = dialog.findChild(_AvatarPreview)
        assert preview._source is window.pmx_preview_widget
        assert preview._timer.isActive()
        assert window._expression_preview_active
        assert not window.options_dialog.isVisible()
        checked.append(True)
        dialog.close()
        assert not preview._timer.isActive()

    QTimer.singleShot(50, inspect_and_close)
    window._open_expression_manager()
    assert checked
    assert not window._expression_preview_active
    assert window.pmx_preview_widget._command is None
    assert backend.messages == [] and backend.spoken_texts == []
    window.close()


@pytest.mark.parametrize("width", [420, 550])
def test_connection_settings_long_device_name_does_not_expand_past_viewport(width):
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    name = "マイクの長い製品名" * 12
    dialog = SettingsDialog(
        ["openai"], "openai", SpeechSettings("whisper", 1, [UiAudioDevice(1, name)]), window
    )
    dialog.resize(width, 500)
    dialog.show()
    app.processEvents()
    scroll = dialog.findChild(QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0
    assert scroll.widget().width() == scroll.viewport().width()
    assert dialog.microphone_combo.toolTip() == name
    assert dialog.selected_microphone() == 1
    dialog.close()
    window.close()


def test_unused_and_unresolved_expressions_are_muted_but_remain_selectable():
    QApplication.instance() or QApplication([])
    calls = []
    presets = [
        UiExpressionPreset("active", (("笑い", 0.3),), ("happy",)),
        UiExpressionPreset("unused", (("笑い", 0.5),)),
        UiExpressionPreset("missing", (), ("sad",), ("missing_role",)),
    ]
    dialog = ExpressionLibraryDialog(
        UiAvatarModelPreview("試験", 1, 0, ((0, 0, 0),)),
        lambda _: None,
        lambda: None,
        start_combination_preview=calls.append,
        load_presets=lambda: presets,
    )
    for row in (1, 2):
        item = dialog.saved_preset_list.item(row)
        assert item.foreground().color().name() == "#718396"
        assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    dialog.saved_preset_list.setCurrentRow(1)
    assert calls[-1] == {"笑い": 0.5}
    assert "使用しない" in dialog.saved_preset_list.item(1).text()
    dialog.close()


def test_motion_manager_uses_full_body_and_restores_user_camera(monkeypatch):
    from test_motion_library_dialog import _MotionBackend

    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    motions = _MotionBackend()
    backend.list_motion_library = motions.list_motion_library
    backend.load_motion_preview = motions.load_motion_preview
    window = ChatWindow(backend)
    source = window.pmx_preview_widget
    source._zoom = 1.7
    source._pitch = 0.2
    before = (source._yaw, source._pitch, source._zoom)

    def inspect(dialog):
        assert source._dense_view._view_mode == "full"
        assert source._zoom == 1.0 and source._pitch == 0.0
        assert window._camera_view == "bust"
        dialog.close()

    monkeypatch.setattr(window, "_exec_avatar_manager", inspect)
    window._open_motion_manager()
    assert source._dense_view._view_mode == "bust"
    assert (source._yaw, source._pitch, source._zoom) == before
    assert not source._dense_view.is_motion_preview_active
    window.close()
