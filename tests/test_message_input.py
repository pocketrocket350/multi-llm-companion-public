"""フォーカス時だけのIME操作と、変換確定・送信の境界。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from multi_llm_companion.ui.chat_window import MessageInput


@pytest.fixture
def inputs():
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    layout = QVBoxLayout(window)
    other = QPushButton("別の操作")
    edit = MessageInput()
    layout.addWidget(other)
    layout.addWidget(edit)
    window.show()
    window.activateWindow()
    other.setFocus()
    app.processEvents()
    yield edit, other
    window.close()
    window.deleteLater()
    app.processEvents()


def test_focus_opens_ime_once_and_leaves_manual_off_during_typing(inputs, monkeypatch):
    edit, other = inputs
    calls = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window._enable_windows_ime", lambda: calls.append(1)
    )
    edit.setFocus()
    QApplication.processEvents()
    assert calls == [1]
    QTest.keyClicks(edit, "hello")
    QApplication.processEvents()
    assert calls == [1]
    other.setFocus()
    edit.setFocus()
    QApplication.processEvents()
    assert calls == [1, 1]


@pytest.mark.parametrize("reason", ["focus_moved", "read_only", "disabled"])
def test_deferred_ime_does_not_touch_another_or_disabled_input(inputs, monkeypatch, reason):
    edit, other = inputs
    calls = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window._enable_windows_ime", lambda: calls.append(1)
    )
    edit.setFocus()
    if reason == "focus_moved":
        other.setFocus()
    elif reason == "read_only":
        edit.setReadOnly(True)
    else:
        edit.setEnabled(False)
    QApplication.processEvents()
    assert not calls


def test_enter_during_ime_composition_does_not_send_and_committed_text_can_send(inputs):
    edit, _ = inputs
    sent = []
    edit.send_requested.connect(lambda: sent.append(edit.toPlainText()))
    edit.setFocus()
    QApplication.sendEvent(edit, QInputMethodEvent("にほんご", []))
    QTest.keyClick(edit, Qt.Key.Key_Return)
    assert not sent
    committed = QInputMethodEvent()
    committed.setCommitString("日本語")
    QApplication.sendEvent(edit, committed)
    assert edit.toPlainText() == "日本語"
    QTest.keyClick(edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert edit.toPlainText() == "日本語\n"
    assert not sent
    QTest.keyClick(edit, Qt.Key.Key_Return)
    assert sent == ["日本語\n"]
