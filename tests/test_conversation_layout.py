import threading

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication
from test_ui import FakeBackend
from test_ui import isolate_window_settings as isolate_window_settings
from test_web_search_ui import screen as screen
from test_web_search_ui import wait_until

from multi_llm_companion.providers import MockProvider
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.companion_background import _cover_rect
from multi_llm_companion.ui.conversation_view import ConversationView


def test_adopted_layout_has_left_avatar_hidden_history_and_separate_settings():
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window.show()
    app.processEvents()
    assert window.content_splitter.widget(0) is window.avatar_panel
    assert window._camera_view == "bust"
    assert window.history_panel.isHidden()
    assert window.activity_panel.isHidden()
    assert not window.provider_label.isVisible()
    window.history_button.click()
    window.resize(800, 600)
    app.processEvents()
    assert window.conversation_list.isVisible()
    assert window.width() == 800 and window.height() == 600
    assert window.send_button.isVisible() and window.input_edit.width() > 50
    window.history_button.click()
    assert window.history_panel.isHidden()
    window.options_button.click()
    assert window.settings_button.isVisible()
    window.options_tabs.setCurrentIndex(1)
    window.camera_view_button.click()
    assert window.pmx_preview_widget._dense_view._view_mode == "full"
    window.options_tabs.setCurrentIndex(2)
    assert window.provider_label.isVisible()
    window.close()
    assert not window.options_dialog.isVisible()


def test_recording_and_error_are_visible_without_opening_diagnostics():
    QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.show()
    backend.listening = True
    window._set_ptt_phase("録音中")
    assert window.recognition_panel.isVisible()
    assert window.activity_panel.isVisible()
    backend.listening = False
    window._set_ptt_phase("待機")
    window._update_conversation_state()
    assert window.activity_panel.isHidden()
    window._show_error("確認用エラー")
    assert window.error_label.isVisible()
    assert not window.options_dialog.isVisible()
    window._clear_error()
    assert window.error_label.isHidden()
    window.close()


def test_existing_window_preferences_round_trip_and_keyboard_input():
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window.show()
    window.resize(780, 600)
    window.content_splitter.setSizes([600, 400])
    window.always_on_top_checkbox.setChecked(True)
    window._settings.setValue("status_visible", False)
    app.processEvents()
    window.activateWindow()
    window.input_edit.setFocus()
    app.processEvents()
    assert window.input_edit.hasFocus()
    QTest.keyClicks(window.input_edit, "hello")
    QTest.keyClick(window.input_edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert window.input_edit.toPlainText() == "hello\n"
    QTest.keyClick(window.input_edit, Qt.Key.Key_Tab)
    assert window.send_button.hasFocus()
    window.close()
    restored = ChatWindow(FakeBackend())
    restored.show()
    app.processEvents()
    assert restored.content_splitter.sizes()[0] > restored.content_splitter.sizes()[1]
    assert restored.always_on_top_checkbox.isChecked()
    assert not restored.status_panel.isHidden()
    assert restored._settings.value("status_visible", type=bool) is False
    assert restored._camera_view == "bust"
    assert restored.width() == 780 and restored.height() == 600
    assert set(restored._settings.allKeys()) == {
        "window_geometry",
        "layout/main_splitter",
        "layout/content_splitter",
        "status_visible",
        "always_on_top",
    }
    restored.close()


def test_generating_stop_button_discards_late_normal_response(screen):
    window, backend = screen
    started, release = threading.Event(), threading.Event()

    def respond(_messages):
        started.set()
        assert release.wait(5)
        return '{"speech":"表示してはいけない遅着結果"}'

    backend._service.set_provider(MockProvider(responder=respond))
    spoken = []
    backend._service.voice_output.start_stream = spoken.append
    window.input_edit.setPlainText("お茶を飲もう")
    window.send_button.click()
    try:
        wait_until(started.is_set)
        assert window.activity_panel.isVisible()
        assert window.stop_voice_button.isVisible() and window.stop_voice_button.isEnabled()
        window.stop_voice_button.click()
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "遅着結果" not in window.chat_view.toPlainText()
    assert not spoken
    assert not any(
        m.role == "assistant" for m in backend.load_messages(window._current_conversation_id)
    )
    assert window.activity_panel.isHidden()


@pytest.mark.parametrize("width", [300, 550])
def test_bubbles_fit_text_and_reuse_widgets_while_streaming(width):
    QApplication.instance() or QApplication([])
    view = ConversationView()
    view.resize(width, 400)
    view.show()
    messages = [("user", "あなた", "こんにちは。"), ("assistant", "雨衣", "一緒に話そう。")]
    view.set_messages(messages)
    QTest.qWait(20)
    user, assistant = (row[1] for row in view._rows)
    assert "#8DE055" in user.frame.styleSheet()
    assert "#FFFFFF" in assistant.frame.styleSheet()
    assert user.x() > assistant.x()
    assert user.text.viewport().height() >= user.text.document().size().height()
    assert assistant.text.viewport().height() >= assistant.text.document().size().height()
    assert view._rows[0][0].height() < 100
    messages[1] = (
        "assistant",
        "雨衣",
        "長い本文を折り返す確認です。" * 40 + '<a href="https://example.org/a">出典</a>',
    )
    view.set_messages(messages)
    QTest.qWait(20)
    assert view._rows[0][1] is user and view._rows[1][1] is assistant
    assert assistant.text.viewport().height() >= assistant.text.document().size().height()
    assert assistant.width() <= view.viewport().width()
    assert assistant.text.openExternalLinks()
    assert assistant.text.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert 'href="https://example.org/a"' in view.toHtml()
    assert view.verticalScrollBar().maximum() > 0
    assert view.verticalScrollBar().value() == view.verticalScrollBar().maximum()
    view.set_messages([])
    assert not view._rows
    view.close()
    view.deleteLater()
    QTest.qWait(10)


@pytest.mark.parametrize("following", [True, False])
def test_late_scroll_range_keeps_tail_following_without_waiting_for_another_timer(following):
    QApplication.instance() or QApplication([])
    view = ConversationView()
    view.resize(300, 400)
    view.show()
    view.set_messages([("assistant", "雨衣", "長い読み返しの文章。" * 100)])
    QTest.qWait(20)
    bar = view.verticalScrollBar()
    view._follow_latest()
    QTest.qWait(20)
    try:
        if not following:
            bar.setValue(0)
        assert bar.maximum() > 0
        # 遅れて確定する文書寸法の増加を、イベントの追加待機なしで確認する。
        bar.setRange(0, bar.maximum() + 95)
        assert bar.value() == (bar.maximum() if following else 0)
        assert view._follow_tail is following
    finally:
        view.close()
        view.deleteLater()
        QTest.qWait(10)


@pytest.mark.parametrize("width,height", [(526, 722), (264, 542), (800, 400)])
def test_background_cover_preserves_provided_image_aspect(width, height):
    crop = _cover_rect(853, 1844, width, height)
    assert crop.width() / crop.height() == pytest.approx(width / height)
    assert 0 <= crop.left() <= crop.right() <= 853
    assert 0 <= crop.top() <= crop.bottom() <= 1844
    assert crop.center().x() == pytest.approx(853 / 2)
    assert crop.center().y() == pytest.approx(1844 / 2)


def test_reading_position_and_selection_survive_reply_updates_and_resize():
    QApplication.instance() or QApplication([])
    view = ConversationView()
    view.resize(380, 400)
    view.show()
    messages = [("assistant", "雨衣", "読み返す文章です。<br>" * 30)]
    messages.append(("assistant", "雨衣", "新しい返答です。"))
    view.set_messages(messages)
    QTest.qWait(30)
    bar = view.verticalScrollBar()
    bar.setValue(120)
    text = view._rows[0][1].text
    cursor = text.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(8, cursor.MoveMode.KeepAnchor)
    text.setTextCursor(cursor)
    selected = cursor.selectedText()
    messages[-1] = ("assistant", "雨衣", "返答を更新します。<br>" * 10)
    view.set_messages(messages)
    QTest.qWait(30)
    assert bar.value() == 120
    assert text.textCursor().selectedText() == selected
    view.resize(360, 380)
    QTest.qWait(30)
    assert bar.value() == 120
    bar.setValue(bar.maximum())
    messages[-1] = ("assistant", "雨衣", "末尾では追随します。<br>" * 20)
    view.set_messages(messages)
    QTest.qWait(30)
    assert bar.value() == bar.maximum()
    view.close()
    view.deleteLater()


def test_streaming_reuses_unchanged_history_layout_but_resizes_and_updates_text():
    QApplication.instance() or QApplication([])
    view = ConversationView()
    view.resize(420, 300)
    view.show()
    messages = [("assistant", "雨衣", "読み返す長い履歴です。" * 60)]
    messages.append(("assistant", "雨衣", "新しい返答"))
    view.set_messages(messages)
    QTest.qWait(30)
    history = view._rows[0][1]
    size_changes = QSignalSpy(history.text.document().documentLayout().documentSizeChanged)
    try:
        for index in range(20):
            messages[-1] = ("assistant", "雨衣", "新しい返答" + "続き" * index)
            view.set_messages(messages)
        QTest.qWait(30)
        assert size_changes.count() == 0
        old_height = history.height()
        view.resize(550, 300)
        QTest.qWait(30)
        assert size_changes.count() > 0
        assert history.height() < old_height
        messages[0] = ("assistant", "雨衣", "訂正した短い履歴")
        view.set_messages(messages)
        QTest.qWait(30)
        assert history.text.toPlainText() == "訂正した短い履歴"
        assert history.height() < old_height
        assert history.text.viewport().height() >= history.text.document().size().height()
    finally:
        view.close()
        view.deleteLater()
        QTest.qWait(10)


def test_chat_stream_does_not_pull_reader_to_bottom():
    QApplication.instance() or QApplication([])
    from multi_llm_companion.ui.models import UiMessage

    window = ChatWindow(FakeBackend())
    window.show()
    window._messages = [UiMessage("assistant", "読み返す文章です。\n" * 50)]
    window._messages.append(UiMessage("assistant", "新しい返答"))
    window._render_messages()
    QTest.qWait(30)
    bar = window.chat_view.verticalScrollBar()
    bar.setValue(100)
    window._append_assistant_chunk(window._stream_generation, "が届いています。")
    QTest.qWait(30)
    assert bar.value() == 100
    window.close()


def test_explicit_send_and_history_switch_show_latest_message(screen):
    window, backend = screen
    first = backend.create_conversation("長い会話")
    second = backend.create_conversation("別の会話")
    for conversation in (first, second):
        backend._repository.add_message(conversation, "assistant", "過去の文章です。\n" * 60)
    window._reload_conversations(first)
    QTest.qWait(30)
    bar = window.chat_view.verticalScrollBar()
    bar.setValue(100)
    window._reload_conversations(second)
    QTest.qWait(30)
    assert bar.value() == bar.maximum()
    bar.setValue(100)
    window.input_edit.setPlainText("ここから話を続けよう")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    QTest.qWait(30)
    assert bar.value() == bar.maximum()


def test_long_partial_recognition_is_readable_without_growing_small_window():
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    window = ChatWindow(backend)
    window.resize(800, 600)
    window.show()
    window.history_button.click()
    backend.listening = True
    window._set_ptt_phase("録音中")
    partial = "長い発話の途中認識です。" * 30
    window._speech_partial_updated(partial)
    app.processEvents()
    assert window.width() == 800 and window.height() == 600
    assert window.partial_label.width() >= window.input_edit.width()
    assert window.partial_label.isReadOnly()
    assert window.partial_label.text() == f"認識途中: {partial}"
    assert window.partial_label.height() >= window.partial_label.sizeHint().height()
    assert window.send_button.isVisible() and window.microphone_button.isVisible()
    backend.listening = False
    window.close()
