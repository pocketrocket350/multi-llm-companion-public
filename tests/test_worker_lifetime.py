"""終了通知とnativeスレッドの後始末完了を区別する。"""

import threading
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from shiboken6 import isValid
from test_web_search_ui import screen as screen
from test_web_search_ui import wait_until

from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services.proactive import ProactiveSettings
from multi_llm_companion.ui import chat_window, image_attachment
from multi_llm_companion.ui.models import SendOutcome


@pytest.mark.parametrize("kind", ["stream", "voice", "proactive", "image"])
def test_worker_stays_alive_until_thread_cleanup_finishes(screen, monkeypatch, kind):
    window, backend = screen
    responding, reply_allowed = threading.Event(), threading.Event()
    finishing, finish_allowed = threading.Event(), threading.Event()
    retained = []

    def hold(value):
        responding.set()
        assert reply_allowed.wait(5)
        return value

    if kind == "stream":
        backend._service.set_provider(
            MockProvider(responder=lambda _: hold('{"speech":"ひと息つこう。"}'))
        )
        window.input_edit.setPlainText("ひと息つこう")
        window.send_button.click()
        fields = ("_thread", "_worker")
    elif kind == "voice":
        monkeypatch.setattr(backend, "start_voice_output", lambda _: hold(None))
        window._prepare_voice_output("ひと息つこう。")
        fields = ("_voice_thread", "_voice_worker")
    elif kind == "proactive":
        window.new_button.click()
        monkeypatch.setattr(window, "_proactive_settings", lambda: ProactiveSettings(enabled=True))
        monkeypatch.setattr(window._proactive, "consider", lambda *_, **__: object())
        monkeypatch.setattr(
            backend, "_generate_proactive_turn", lambda _, **__: hold(SendOutcome("ひと息つこう。"))
        )
        window._on_proactive_timer()
        fields = ("_proactive_thread", "_proactive_worker")
    else:
        prepare = image_attachment._prepare_drop

        def delayed_image(source, cancelled):
            hold(None)
            return prepare(source, cancelled)

        monkeypatch.setattr(image_attachment, "_prepare_drop", delayed_image)
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.red)
        window._attach_image(image_attachment._ImageDrop(image))

    def references():
        if kind == "image":
            return next(iter(window._image_fetches.values()), (None, None))
        return tuple(getattr(window, field) for field in fields)

    wait_until(responding.is_set)
    thread, worker = references()
    gui_thread = threading.get_native_id()
    destroyed_on = []
    worker.destroyed.connect(
        lambda: destroyed_on.append(threading.get_native_id()), Qt.ConnectionType.DirectConnection
    )

    def finish_tail():
        # finished配送後も残るnative終了処理を、時間差付きで再現する。
        finishing.set()
        assert finish_allowed.wait(5)

    thread.finished.connect(finish_tail, Qt.ConnectionType.DirectConnection)

    def observe():
        try:
            assert finishing.wait(5)
            time.sleep(0.15)
            # Python参照のみ確認し、別スレッドからQtの画面を操作しない。
            retained.append(references() == (thread, worker))
        finally:
            finish_allowed.set()

    observer = threading.Thread(target=observe)
    observer.start()
    reply_allowed.set()
    try:
        wait_until(lambda: references() == (None, None) and not window._busy)
    finally:
        finish_allowed.set()
        observer.join(5)
    assert not observer.is_alive()
    assert retained == [True], "native終了処理中にworker参照が解放されています"
    assert not isValid(worker)
    assert destroyed_on == [gui_thread], "Qtの接続解除をGUI以外で実行しています"
    if kind in {"stream", "proactive"}:
        assert "ひと息つこう。" in window.chat_view.toPlainText()
    elif kind == "image":
        assert not window.attachment_preview.pixmap().isNull()


def test_voice_close_waits_for_start_worker_and_discards_late_completion(screen, monkeypatch):
    window, backend = screen
    started, release = threading.Event(), threading.Event()

    def start(_):
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(backend, "start_proactive_voice_output", start)
    window._prepare_voice_output("古い能動発話", proactive=True)
    wait_until(started.is_set)
    try:
        assert not window._busy
        assert not window.close(), "音声開始スレッドが生きたまま閉じています"
        assert window._pending_voice_text is None
    finally:
        release.set()
        wait_until(lambda: window._voice_thread is None)
    wait_until(lambda: not window.isVisible())
    assert window._continuous_voice_waiting_generation is None


@pytest.mark.parametrize("cancel_pending", [False, True])
def test_newer_voice_waits_for_previous_start_worker(screen, monkeypatch, cancel_pending):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    spoken = []

    def start(text):
        spoken.append(text)
        if text == "古い発話":
            started.set()
            assert release.wait(5)

    monkeypatch.setattr(backend, "start_voice_output", start)
    window._prepare_voice_output("古い発話")
    wait_until(started.is_set)
    previous = window._voice_thread

    def overlapping_worker(*_, **__):
        pytest.fail("終了していない音声開始workerを新しいworkerで上書きしています")

    try:
        window._interrupt_voice_output()
        with monkeypatch.context() as pending:
            pending.setattr(chat_window, "VoiceStartWorker", overlapping_worker)
            window._prepare_voice_output("新しい発話")
        assert window._voice_thread is previous
        assert window._pending_voice_text == "新しい発話"
        if cancel_pending:
            window._stop_voice_output()
    finally:
        release.set()
        wait_until(lambda: window._voice_thread is None)
    assert spoken == (["古い発話"] if cancel_pending else ["古い発話", "新しい発話"])
