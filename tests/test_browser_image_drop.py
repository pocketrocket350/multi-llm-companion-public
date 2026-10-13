import base64
import threading

import httpx
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import QApplication
from test_image_sharing import picture as picture
from test_web_search_ui import screen as screen
from test_web_search_ui import wait_until

from multi_llm_companion.services.image_sharing import ImageSharingError
from multi_llm_companion.ui import image_attachment
from multi_llm_companion.ui.image_attachment import (
    _image_drop_source,
    _ImageDrop,
    _prepare_drop,
    prepare_image,
)


def drop_image(target, mime):
    enter = QDragEnterEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(target, event)
    assert event.isAccepted()


@pytest.mark.parametrize("kind", ["url", "html", "bytes", "pixels", "data"])
@pytest.mark.parametrize("surface", ["window", "input"])
def test_browser_drop_previews_without_saving_or_sending(
    screen, picture, monkeypatch, kind, surface
):
    window, backend = screen
    raw = picture.read_bytes()
    mime = QMimeData()
    if kind == "url":
        mime.setUrls([QUrl("https://images.example.com/photo")])
    elif kind == "html":
        mime.setUrls([QUrl("https://example.com/page")])
        mime.setHtml(
            '<a href="https://example.com/page"><img src="https://images.example.com/photo"></a>'
        )
    elif kind == "bytes":
        mime.setData("image/png", raw)
    elif kind == "pixels":
        mime.setImageData(QImage(str(picture)))
    else:
        mime.setHtml('<img src="data:image/png;base64,' + base64.b64encode(raw).decode() + '">')
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://images.example.com/photo"
        assert request.method == "GET"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        return httpx.Response(200, content=raw)

    original = image_attachment._prepare_drop
    monkeypatch.setattr(
        image_attachment,
        "_prepare_drop",
        lambda source, cancelled: original(
            source, cancelled, transport=httpx.MockTransport(handler)
        ),
    )
    monkeypatch.setattr(backend, "stream_message", lambda *_: pytest.fail("送信禁止"))
    window.input_edit.setPlainText("この写真を見て")
    drop_image(window if surface == "window" else window.input_edit.viewport(), mime)
    wait_until(lambda: not window._image_fetches)
    assert not window.attachment_preview.pixmap().isNull()
    assert window._attached_image.preview_jpeg
    assert window.send_button.isEnabled()
    assert window.input_edit.toPlainText() == "この写真を見て"
    assert all(not backend.load_messages(c.id) for c in backend.list_conversations())
    assert len(requests) == (1 if kind in {"url", "html"} else 0)
    window.remove_attachment_button.click()
    assert window._attached_image is None and window.attachment_panel.isHidden()


@pytest.mark.parametrize(
    "action", ["remove", "replace", "new", "stop", "microphone", "disable", "close"]
)
def test_pending_browser_fetch_can_be_cancelled_without_late_preview(
    screen, picture, monkeypatch, action
):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    image = prepare_image(str(picture))

    def delayed(_source, _cancelled):
        started.set()
        assert release.wait(5)
        return image

    monkeypatch.setattr(image_attachment, "_prepare_drop", delayed)
    window.input_edit.setPlainText("質問")
    window._attach_image(_ImageDrop("https://images.example.com/photo"))
    try:
        wait_until(started.is_set)
        assert window._image_loading and not window.send_button.isEnabled()
        assert not window.input_edit.isReadOnly()
        window.input_edit.insertPlainText("の続き")
        assert window._image_loading
        if action == "remove":
            window.remove_attachment_button.click()
        elif action == "replace":
            window._attach_image(str(picture))
        elif action == "new":
            window.new_button.click()
        elif action == "stop":
            window._stop_voice_output()
        elif action == "microphone":
            window._toggle_microphone()
        elif action == "disable":
            window._toggle_image_permission(False)
        else:
            assert not window.close()
    finally:
        release.set()
        wait_until(lambda: not window._image_fetches)
    assert not window._image_loading
    assert (
        window._attached_image is None
        if action != "replace"
        else window._attached_image is not image
    )
    assert all(not backend.load_messages(c.id) for c in backend.list_conversations())
    if action == "close":
        wait_until(lambda: not window.isVisible())


def test_failed_browser_fetch_retains_previous_image_and_recovers(screen, picture, monkeypatch):
    window, _ = screen
    window._attach_image(str(picture))
    previous = window._attached_image

    def failed(*_):
        raise ImageSharingError("画像を取得できませんでした。")

    monkeypatch.setattr(image_attachment, "_prepare_drop", failed)
    window._attach_image(_ImageDrop("https://images.example.com/photo"))
    wait_until(lambda: not window._image_fetches)
    assert window._attached_image is previous and not window._image_loading
    assert "取得できません" in window.error_label.text()
    window.remove_attachment_button.click()
    window.input_edit.setPlainText("普通に話そう")
    window.send_button.click()
    wait_until(lambda: not window._busy and window._thread is None)
    assert window.error_label.isHidden()


@pytest.mark.parametrize(
    "case", ["html", "oversize", "redirect_private", "redirect_loop", "timeout", "status", "cancel"]
)
def test_browser_fetch_limits_and_failures(picture, case):
    calls = []

    def handler(request):
        calls.append(request)
        if case == "timeout":
            raise httpx.ReadTimeout("private-token", request=request)
        if case == "status":
            return httpx.Response(403, text="private-token")
        if case.startswith("redirect"):
            return httpx.Response(
                302,
                headers={
                    "location": "http://127.0.0.1/x" if case == "redirect_private" else "/again"
                },
            )
        return httpx.Response(
            200, content=b"x" * 15_000_001 if case == "oversize" else b"<html>not an image</html>"
        )

    with pytest.raises(ImageSharingError) as error:
        _prepare_drop(
            _ImageDrop("https://images.example.com/photo"),
            lambda: case == "cancel",
            transport=httpx.MockTransport(handler),
        )
    assert "private-token" not in str(error.value)
    assert len(calls) == (0 if case == "cancel" else 4 if case == "redirect_loop" else 1)


def test_image_url_redirect_and_data_validation(picture):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return (
            httpx.Response(302, headers={"location": "/final"})
            if len(calls) == 1
            else httpx.Response(200, content=picture.read_bytes())
        )

    image = _prepare_drop(
        _ImageDrop("https://images.example.com/start?private-token=x"),
        lambda: False,
        transport=httpx.MockTransport(handler),
    )
    assert image.name == "final" and image.width == 80
    for data in (
        "data:image/png;base64,not-valid!",
        "data:image/gif;base64,R0lG",
        "data:image/png;base64," + "a" * 20_000_001,
    ):
        with pytest.raises(ImageSharingError):
            _prepare_drop(_ImageDrop(data), lambda: False)


def test_multiple_html_images_are_not_silently_selected():
    mime = QMimeData()
    mime.setHtml('<img src="https://example.com/a"><img src="https://example.com/b">')
    assert _image_drop_source(mime) is None
