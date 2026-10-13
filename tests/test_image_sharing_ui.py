import json
import threading
from datetime import datetime

import httpx
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox
from test_image_sharing import analysis_payload, image_client
from test_image_sharing import picture as picture
from test_web_search_ui import screen as screen
from test_web_search_ui import wait_until

from multi_llm_companion.domain.models import ImageAttachmentInfo, WebSource
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services.image_sharing import ImageSharingSession
from multi_llm_companion.ui.chat_window import QFileDialog
from multi_llm_companion.ui.image_attachment import prepare_image
from multi_llm_companion.ui.models import UiMessage


def attach(window, backend, picture, monkeypatch, handler=None):
    backend._service.image_sharing = ImageSharingSession(
        image_client(handler or (lambda _: httpx.Response(200, json=analysis_payload())))
    )
    window._refresh_web_permission()
    backend._service.set_provider(MockProvider(response_text='{"speech":"赤い四角が見えるね。"}'))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(picture), ""))
    window.attach_image_button.click()


def test_select_preview_remove_does_not_send(screen, picture, monkeypatch):
    window, backend = screen
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    assert backend.image_sharing_enabled()
    assert window.attachment_panel.isVisible()
    assert not window.attachment_preview.pixmap().isNull()
    QApplication.processEvents()
    assert window.attachment_panel.y() > window.input_edit.y()
    assert window.attach_image_button.geometry().center().y() == (
        window.attachment_panel.geometry().center().y()
    )
    assert window.attachment_panel.x() > window.attach_image_button.x()
    assert not any(label.text() for label in window.attachment_panel.findChildren(QLabel))
    assert window.remove_attachment_button.text() == ""
    assert not window.remove_attachment_button.icon().isNull()
    assert window.remove_attachment_button.accessibleName() == "画像を外す"
    assert window.remove_attachment_button.parent() is window.attachment_panel
    assert window.attachment_preview.geometry().intersects(
        window.remove_attachment_button.geometry()
    )
    window.remove_attachment_button.click()
    assert window._attached_image is None
    assert window.attachment_panel.isHidden()


def test_image_send_without_permission_dialog_stores_user_preview(screen, picture, monkeypatch):
    window, backend = screen
    attach(window, backend, picture, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *_: pytest.fail("追加の確認は禁止"))
    assert backend.image_sharing_enabled()
    assert not hasattr(window, "attachment_description")
    window.input_edit.setPlainText("この画像を調べて、一緒に話そう")
    window.send_button.click()
    assert window._web_turn_active and window._image_turn_active
    assert window.web_cancel_button.text() == "画像・返答を取消"
    wait_until(lambda: window._thread is None and not window._busy)
    assert "赤い四角" in window.chat_view.toPlainText()
    assert "添付画像: fixture.png" not in window.chat_view.toPlainText()
    assert "原本は保存していません" not in window.chat_view.toPlainText()
    assert not window.chat_view._rows[0][1].image_label.pixmap().isNull()
    assert backend.load_messages(window._current_conversation_id)[0].image_attachment.preview_jpeg
    assert backend.image_sharing_enabled()
    assert backend.web_search_enabled()
    assert not backend.list_memories()
    stored = backend.load_messages(window._current_conversation_id)[-1]
    assert stored.image_attachment.name == "fixture.png"
    assert stored.sources == ()
    assert window._attached_image is None and window.attachment_panel.isHidden()
    assert not window._settings.contains("image_sharing/enabled")


@pytest.mark.parametrize("action", ["cancel", "disable", "input", "microphone", "stop"])
def test_image_late_results_are_discarded_at_every_user_boundary(
    screen, picture, monkeypatch, action
):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    calls = []

    def delayed(_):
        started.set()
        assert release.wait(5)
        return httpx.Response(200, json=analysis_payload())

    attach(window, backend, picture, monkeypatch, delayed)
    backend.set_image_sharing_enabled(True)
    backend._service.set_provider(
        MockProvider(responder=lambda _: calls.append(True) or '{"speech":"遅着"}')
    )
    window.input_edit.setPlainText("画像を見て")
    window.send_button.click()
    try:
        wait_until(started.is_set)
        if action == "cancel":
            window.web_cancel_button.click()
        elif action == "disable":
            window._toggle_image_permission(False)
        elif action == "input":
            assert not window.input_edit.isReadOnly()
            QTest.keyClicks(window.input_edit, "next")
        elif action == "microphone":
            window._toggle_microphone()
        else:
            window._stop_voice_output()
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not calls
    assert not [
        m for m in backend.load_messages(window._current_conversation_id) if m.role == "assistant"
    ]
    assert not backend.list_memories()
    window.input_edit.setPlainText("ひと息つこう")
    backend._service.set_provider(MockProvider(response_text='{"speech":"うん、ひと息つこう。"}'))
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "ひと息つこう" in window.chat_view.toPlainText()


def test_disabling_image_sharing_removes_attachment_without_a_dialog(screen, picture, monkeypatch):
    window, backend = screen
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: pytest.fail("追加の確認は禁止"))
    assert backend.image_sharing_enabled()
    window.image_permission_checkbox.click()
    assert window._thread is None and not backend.image_sharing_enabled()
    assert window._attached_image is None and window.attachment_panel.isHidden()
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    backend.set_image_sharing_enabled(False)
    window.input_edit.setPlainText("この画像を見て")
    window.send_button.click()
    assert "オフ" in window.error_label.text()
    assert window._thread is None and not backend.image_sharing_enabled()


def test_changing_conversation_clears_unsent_image(screen, picture, monkeypatch):
    window, backend = screen
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    window.new_button.click()
    assert window._attached_image is None and window.attachment_panel.isHidden()


def test_image_error_returns_to_normal_chat_without_automatic_resend(screen, picture, monkeypatch):
    window, backend = screen
    wire = []

    def fail(request):
        wire.append(json.loads(request.content))
        return httpx.Response(500, text="private-error")

    attach(window, backend, picture, monkeypatch, fail)
    backend.set_image_sharing_enabled(True)
    window.input_edit.setPlainText("見て")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "画像解析に失敗" in window.error_label.text()
    assert "private-error" not in window.error_label.text()
    assert window._attached_image is None
    window.input_edit.setPlainText("普通に話そう")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(wire) == 1


def test_delete_last_conversation_clears_unsent_image(screen, picture, monkeypatch):
    window, backend = screen
    window.new_button.click()
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    window._delete_current_conversation()
    assert window._attached_image is None


def test_cancel_before_worker_enters_backend_never_sends(screen, picture, monkeypatch):
    window, backend = screen
    attach(window, backend, picture, monkeypatch, lambda _: pytest.fail("送信禁止"))
    backend.set_image_sharing_enabled(True)
    release = threading.Event()
    original = backend.stream_message

    def paused(*args, **kwargs):
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "stream_message", paused)
    window.input_edit.setPlainText("見て")
    window.send_button.click()
    window.web_cancel_button.click()
    release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not backend.load_messages(window._current_conversation_id)


def drop_urls(target, urls, proposed=Qt.DropAction.CopyAction):
    mime = QMimeData()
    mime.setUrls(urls)
    buttons, modifiers = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    actions = Qt.DropAction.CopyAction | proposed
    enter = QDragEnterEvent(QPoint(10, 10), actions, mime, buttons, modifiers)
    enter.setDropAction(proposed)
    QApplication.sendEvent(target, enter)
    if not enter.isAccepted():
        return False
    move = QDragMoveEvent(QPoint(10, 10), actions, mime, buttons, modifiers)
    move.setDropAction(proposed)
    QApplication.sendEvent(target, move)
    assert move.isAccepted()
    assert move.dropAction() == Qt.DropAction.CopyAction
    event = QDropEvent(QPointF(10, 10), actions, mime, buttons, modifiers)
    event.setDropAction(proposed)
    QApplication.sendEvent(target, event)
    if event.isAccepted():
        assert event.dropAction() == Qt.DropAction.CopyAction
    return event.isAccepted()


@pytest.mark.parametrize("surface", ["window", "input"])
@pytest.mark.parametrize("proposed", [Qt.DropAction.CopyAction, Qt.DropAction.MoveAction])
def test_drop_attaches_without_sending_or_inserting_path(
    screen, picture, monkeypatch, surface, proposed
):
    window, backend = screen
    monkeypatch.setattr(backend, "stream_message", lambda *_: pytest.fail("送信禁止"))
    window.input_edit.setPlainText("この写真を見て")
    target = window if surface == "window" else window.input_edit.viewport()
    assert drop_urls(target, [QUrl.fromLocalFile(str(picture))], proposed)
    assert picture.is_file()
    assert window._attached_image.name == picture.name
    assert window.attachment_panel.isVisible()
    assert window.input_edit.toPlainText() == "この写真を見て"
    assert window._thread is None
    window.remove_attachment_button.click()
    assert window._attached_image is None


@pytest.mark.parametrize("surface", ["window", "input"])
@pytest.mark.parametrize("kind", ["private_url", "multiple", "non_image", "busy"])
def test_invalid_or_busy_drops_leave_draft_unchanged(screen, picture, surface, kind):
    window, _ = screen
    window.input_edit.setPlainText("下書き")
    urls = [QUrl.fromLocalFile(str(picture))]
    if kind == "private_url":
        urls = [QUrl("https://127.0.0.1/image.png")]
    elif kind == "multiple":
        urls *= 2
    elif kind == "non_image":
        urls = [QUrl.fromLocalFile(str(picture.with_suffix(".txt")))]
    else:
        window._web_turn_active = True
        window._set_busy(True)
    try:
        target = window if surface == "window" else window.input_edit.viewport()
        assert not drop_urls(target, urls)
        assert window._attached_image is None
        assert window.input_edit.toPlainText() == "下書き"
    finally:
        window._web_turn_active = False
        window._set_busy(False)


def test_drop_uses_existing_image_validation_and_keeps_previous_image(screen, picture, tmp_path):
    window, _ = screen
    assert drop_urls(window.input_edit.viewport(), [QUrl.fromLocalFile(str(picture))])
    previous = window._attached_image
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"invalid image")
    assert drop_urls(window.input_edit.viewport(), [QUrl.fromLocalFile(str(broken))])
    assert window._attached_image is previous
    assert window.error_label.isVisible()
    assert "画像の形式" in window.error_label.text()
    assert window._thread is None


def test_image_history_reload_preserves_sources_text_and_old_metadata(screen, picture):
    window, _ = screen
    info = prepare_image(str(picture)).history_info()
    now = datetime.now().astimezone()
    sources = tuple(
        WebSource(title=f"資料{i}", url=f"https://example.com/{i}", retrieved_at=now)
        for i in range(3)
    )
    window._messages = [
        UiMessage(role="user", content="この画像について", image_attachment=info, created_at=now),
        UiMessage(
            role="assistant",
            content="資料を確認したよ。[S1] [S2] [S3]",
            sources=sources,
            created_at=now,
        ),
        UiMessage(
            role="assistant",
            content="以前の画像だね。",
            image_attachment=ImageAttachmentInfo(
                name="old.png", width=80, height=60, analyzed_at=now
            ),
        ),
    ]
    window._render_messages()
    first = window.chat_view._rows[0][1]
    image_key = first._image.cacheKey()
    window._render_messages()
    assert first._image.cacheKey() == image_key
    assert "old.png" in window.chat_view.toPlainText()
    assert "原本は保存していません" in window.chat_view.toPlainText()
    assert all(f'href="https://example.com/{i}"' in window.chat_view.toHtml() for i in range(3))
    assert info.preview_jpeg not in window.chat_view.toPlainText()
    window._messages = [UiMessage(role="user", content="別の会話")]
    window._render_messages()
    assert window.chat_view._rows[0][1].image_label.isHidden()
