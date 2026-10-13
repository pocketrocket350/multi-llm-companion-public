"""アイコン化しても操作名と状態、標準ウィンドウの挙動を維持する。"""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog
from test_ui import FakeBackend
from test_ui import isolate_window_settings as isolate_window_settings

from multi_llm_companion.ui.chat_window import ChatWindow


def test_icon_controls_keep_accessible_names_and_keyboard_actions():
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window.show()
    app.processEvents()
    for button, label in (
        (window.history_button, "履歴"),
        (window.options_button, "設定"),
        (window.microphone_button, "マイク"),
        (window.send_button, "送信"),
        (window.attach_image_button, "画像を添付"),
    ):
        assert button.text() == ""
        assert not button.icon().isNull()
        assert button.accessibleName() == label
        assert button.toolTip() == label
        assert button.width() >= 40 and button.height() >= 40
    assert not window.send_button.isEnabled()
    window.input_edit.setPlainText("こんにちは")
    assert window.send_button.isEnabled()
    QTest.keyClick(window.history_button, Qt.Key.Key_Space)
    assert window.history_panel.isVisible()
    QTest.keyClick(window.history_button, Qt.Key.Key_Space)
    assert not window.history_panel.isVisible()
    QTest.keyClick(window.options_button, Qt.Key.Key_Space)
    assert window.options_dialog.isVisible()
    assert not window.windowFlags() & Qt.WindowType.FramelessWindowHint
    window.close()


def test_microphone_icon_distinguishes_recording_recognizing_and_failure(monkeypatch):
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    callbacks = []

    def start(recognized, failed, _partial, _level):
        backend.listening = True
        callbacks.append((recognized, failed))

    monkeypatch.setattr(backend, "start_speech_recognition", start)
    window = ChatWindow(backend)
    window.show()
    microphone_icon = window.microphone_button.icon().cacheKey()
    window.microphone_button.click()
    assert window.microphone_button.accessibleName() == "録音停止"
    assert window.microphone_button.icon().cacheKey() != microphone_icon
    assert window.recognition_panel.isVisible()
    assert window.ptt_phase_label.text() == "マイク: 録音中"
    window.microphone_button.click()
    assert not backend.is_listening()
    assert window.microphone_button.icon().cacheKey() == microphone_icon
    assert window.microphone_button.accessibleName() == "マイク"

    window._start_hold_to_talk()
    window._finish_hold_to_talk()
    assert not window.microphone_button.isEnabled()
    assert window.microphone_button.accessibleName() == "認識中"
    callbacks[-1][0]("アイコンからの音声入力")
    app.processEvents()
    assert window.microphone_button.isEnabled()
    assert window.microphone_button.accessibleName() == "マイク"
    assert window.input_edit.toPlainText() == "アイコンからの音声入力"

    window.microphone_button.click()
    backend.listening = False
    callbacks[-1][1]("検査用の認識エラー")
    app.processEvents()
    assert window.error_label.isVisible()
    assert window.microphone_button.icon().cacheKey() == microphone_icon
    assert window.microphone_button.accessibleName() == "マイク"
    window.close()


def test_title_palette_is_reapplied_only_to_own_windows(monkeypatch):
    app = QApplication.instance() or QApplication([])
    applied = []
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window._apply_title_bar_palette", applied.append
    )
    window = ChatWindow(FakeBackend())
    assert not applied
    window.show()
    window.options_button.click()
    unrelated = QDialog()
    unrelated.show()
    app.processEvents()
    assert window in applied and window.options_dialog in applied
    assert unrelated not in applied
    applied.clear()
    app.sendEvent(window, QEvent(QEvent.Type.WinIdChange))
    assert applied == [window]
    unrelated.close()
    window.close()
