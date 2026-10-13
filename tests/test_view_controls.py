"""モデル操作、分割幅、会話日時を通常設定・会話から分離して確認する。"""

import math
from datetime import UTC, datetime, timedelta

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_application import _runtime
from test_ui import FakeBackend
from test_ui import isolate_window_settings as isolate_window_settings

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.conversation_view import ConversationView
from multi_llm_companion.ui.models import UiMessage
from multi_llm_companion.ui.pmx_preview_widget import PmxPreviewWidget


def drag_model(widget, dx, dy):
    for kind, point, button, buttons in (
        (QEvent.Type.MouseButtonPress, QPointF(100, 100), Qt.LeftButton, Qt.LeftButton),
        (QEvent.Type.MouseMove, QPointF(100 + dx, 100 + dy), Qt.NoButton, Qt.LeftButton),
        (QEvent.Type.MouseButtonRelease, QPointF(100 + dx, 100 + dy), Qt.LeftButton, Qt.NoButton),
    ):
        QApplication.sendEvent(
            widget, QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier)
        )


def test_rotation_is_horizontal_and_limited_until_enabled():
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    try:
        drag_model(widget, 1000, 500)
        assert widget._yaw == pytest.approx(math.tau)
        assert widget._pitch == widget._dense_view._pitch == 0
        drag_model(widget, -2000, -500)
        assert widget._yaw == pytest.approx(0)
        widget.set_rotation_360(True)
        widget.reset_camera()
        drag_model(widget, 250, 500)
        assert widget._yaw > 3 * math.pi / 2
        assert widget._pitch == 0
        drag_model(widget, 400, -500)
        assert widget._yaw == pytest.approx((math.pi + 6.5) % math.tau)
        drag_model(widget, -300, 0)
        widget.set_rotation_360(False)
        assert 0 <= widget._yaw <= math.tau
        assert widget._dense_view._yaw == widget._yaw
    finally:
        widget.close()
        widget.deleteLater()


def test_corner_button_and_rotation_setting_survive_restart():
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window.show()
    app.processEvents()
    assert window.camera_view_button.parentWidget().parentWidget() is window.avatar_panel
    assert window.camera_view_button.accessibleName() == "全身表示に切り替える"
    assert not window.camera_view_button.icon().isNull()
    assert not hasattr(window, "camera_view_combo")
    window.camera_view_button.click()
    assert window._camera_view == "full"
    assert window.camera_view_button.accessibleName() == "バストアップに切り替える"
    window.rotation_360_checkbox.setChecked(True)
    window.close()
    restored = ChatWindow(FakeBackend())
    assert restored._camera_view == "full"
    assert restored.rotation_360_checkbox.isChecked()
    assert restored.pmx_preview_widget._rotation_360
    restored.camera_view_button.click()
    restored.rotation_360_checkbox.setChecked(False)
    restored.close()
    again = ChatWindow(FakeBackend())
    assert again._camera_view == "bust"
    assert not again.pmx_preview_widget._rotation_360
    again.close()


def wheel_model(widget, steps):
    point = QPointF(100, 100)
    QApplication.sendEvent(
        widget,
        QWheelEvent(
            point,
            point,
            QPoint(),
            QPoint(0, 120 * steps),
            Qt.NoButton,
            Qt.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )


def test_full_view_zoom_stops_at_height_fit_and_clamps_after_bust_switch():
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    try:
        widget.set_view_mode("bust")
        wheel_model(widget, -30)
        assert widget._zoom == pytest.approx(0.4)
        widget.set_view_mode("full")
        assert widget._zoom == widget._dense_view._zoom == 1.0
        wheel_model(widget, -30)
        assert widget._zoom == 1.0
        widget.resize(1000, 700)
        wheel_model(widget, 2)
        assert widget._zoom > 1
        wheel_model(widget, -30)
        assert widget._zoom == widget._dense_view._zoom == 1.0
        widget.set_view_mode("bust")
        wheel_model(widget, -30)
        assert widget._zoom == pytest.approx(0.4)
        with widget._temporary_preview_camera("full"):
            assert widget._zoom == 1.0
        assert widget._dense_view._view_mode == "bust"
        assert widget._zoom == widget._dense_view._zoom == pytest.approx(0.4)
    finally:
        widget.close()
        widget.deleteLater()


def drag_splitter(splitter, dx):
    handle = splitter.handle(1)
    center = handle.rect().center()
    QTest.mousePress(handle, Qt.LeftButton, pos=center)
    QTest.mouseMove(handle, center + QPoint(dx, 0), delay=20)
    QTest.mouseRelease(handle, Qt.LeftButton, pos=center + QPoint(dx, 0))
    QApplication.processEvents()


def test_three_pane_drag_widths_restore_after_history_is_hidden():
    app = QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    window.resize(1280, 800)
    window.show()
    window.history_button.click()
    app.processEvents()
    before = window.main_splitter.sizes()
    drag_splitter(window.main_splitter, 65)
    history_size = window.main_splitter.sizes()[0]
    assert history_size > before[0] + 30
    before = window.content_splitter.sizes()
    drag_splitter(window.content_splitter, -75)
    panes = window.content_splitter.sizes()
    assert panes[0] < before[0] - 30
    ratio = panes[0] / sum(panes)
    window.history_button.click()
    window.close()
    restored = ChatWindow(FakeBackend())
    # offscreenの仮想画面は800pxのため、幅の復元は変更前と同じ表示領域で比べる。
    restored.resize(1280, 800)
    restored.show()
    assert restored.history_panel.isHidden()
    restored.history_button.click()
    app.processEvents()
    assert restored.main_splitter.sizes()[0] == pytest.approx(history_size, abs=2)
    panes = restored.content_splitter.sizes()
    assert panes[0] / sum(panes) == pytest.approx(ratio, abs=0.015)
    restored.resize(800, 600)
    app.processEvents()
    assert restored.width() == 800
    assert restored.input_edit.width() > 50
    assert restored.camera_view_button.geometry().right() < restored.avatar_panel.width()
    restored.close()


def test_history_uses_stored_timestamps_without_modifying_them(tmp_path):
    backend = ApplicationBackend(_runtime(tmp_path))
    try:
        conv = backend.create_conversation("日時の検査")
        backend._repository.add_message(conv, "user", "保存済みの発言")
        stored = backend._repository.list_messages(conv)[0]
        loaded = backend.load_messages(conv)[0]
        assert loaded.created_at == stored.created_at
        assert loaded.created_at.tzinfo is not None
    finally:
        backend.close()


def test_local_dates_and_times_are_separate_from_speech_and_keep_bubbles():
    app = QApplication.instance() or QApplication([])
    zone = datetime.now().astimezone().tzinfo
    first = datetime(2026, 9, 24, 23, 59, tzinfo=zone).astimezone(UTC)
    second = first + timedelta(minutes=2)
    view = ConversationView()
    view.resize(360, 550)
    view.show()
    messages = [("user", "あなた", "こんばんは"), ("assistant", "雨衣", "お話ししよう")]
    view.set_messages(messages, timestamps=[first, second])
    app.processEvents()
    assert [label.text() for label in view._date_labels] == ["2026/09/24（木）", "2026/09/25（金）"]
    assert all(not label.isHidden() for label in view._date_labels)
    assert [row[1].time_label.text() for row in view._rows] == ["23:59", "00:01"]
    assert "23:59" not in view.toHtml()
    bubble = view._rows[-1][1]
    messages[-1] = ("assistant", "雨衣", "お話ししよう。" * 12)
    view.set_messages(messages, timestamps=[first, second])
    app.processEvents()
    assert view._rows[-1][1] is bubble
    assert bubble.time_label.text() == "00:01"
    assert bubble.text.viewport().height() >= bubble.text.document().size().height()
    view.set_messages(messages, timestamps=[second, second])
    assert view._date_labels[1].isHidden()
    view.set_messages(messages)
    assert all(label.isHidden() for label in view._date_labels)
    assert all(row[1].time_label.isHidden() for row in view._rows)
    view.close()
    view.deleteLater()


def test_stream_chunks_preserve_message_time_and_ignore_stale_chunks():
    QApplication.instance() or QApplication([])
    window = ChatWindow(FakeBackend())
    timestamp = datetime(2026, 9, 24, 23, 59, tzinfo=UTC)
    window._messages = [UiMessage("assistant", "途中", created_at=timestamp)]
    window._append_assistant_chunk(window._stream_generation, "の文章")
    assert window._messages[-1].created_at == timestamp
    assert window._messages[-1].content == "途中の文章"
    window._append_assistant_chunk(window._stream_generation - 1, "古い文章")
    assert window._messages[-1].content == "途中の文章"
    window.close()
