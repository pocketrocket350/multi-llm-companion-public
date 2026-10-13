"""旧能動通信を残した新規送信・取消・結果分離・スレッド回収を確認する。"""

import json
import threading

import httpx
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
from shiboken6 import isValid
from test_application import _runtime
from test_web_search_ui import wait_until

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.providers import OpenAICompatibleProvider
from multi_llm_companion.services import ChatBusyError, ChatService, ProactiveTurnInvalidatedError
from multi_llm_companion.services.proactive import ProactiveSettings
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.workers import ProactiveWorker

OLD = "旧能動回答。"
NEW = "新しい回答。"


def reply(speech, tokens=8):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": json.dumps({"speech": speech}, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": tokens - 3, "completion_tokens": 3, "total_tokens": tokens},
        },
    )


def provider(handler):
    def guarded(request):
        data = json.loads(request.content)
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert data["model"] == "gpt-6-luna"
        return handler(data)

    return OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="artificial",
        model="gpt-6-luna",
        transport=httpx.MockTransport(guarded),
    )


def launch(call):
    results = []

    def run():
        try:
            results.append(call())
        except Exception as exc:
            results.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results


@pytest.mark.parametrize("old_result", ["reply", "http_error", "invalid_json"])
def test_new_request_completes_while_old_http_still_waits(old_result):
    old_started, release = threading.Event(), threading.Event()
    requests = []

    def handle(data):
        requests.append(data)
        if "【能動的な話しかけ】" in str(data["messages"]):
            assert data["stream"] is False
            old_started.set()
            assert release.wait(5)
            if old_result == "http_error":
                raise httpx.ReadError("artificial-old-error")
            if old_result == "invalid_json":
                return httpx.Response(200, json={"choices": []})
            return reply(OLD, 204)
        assert data["stream"] is True and data["tool_choice"] == "required"
        return reply(NEW)

    p = provider(handle)
    service = ChatService(provider=p, repository=SQLiteConversationRepository(":memory:"))
    service.start_new_conversation("人工割り込み")
    old, outcomes = launch(service.generate_proactive_turn)
    try:
        assert old_started.wait(5)
        assert service.is_busy
        service.invalidate_proactive_turn()
        assert not service.is_busy
        assert "".join(service.stream_speech("新入力")) == NEW
        assert old.is_alive()
        assert service.last_response.speech == NEW and service.last_usage.total_tokens == 8
        release.set()
        old.join(5)
        assert not old.is_alive() and isinstance(outcomes[0], ProactiveTurnInvalidatedError)
        assert [m.content for m in service.get_current_messages()] == ["新入力", NEW]
        assert service.get_current_messages()[-1].usage.total_tokens == 8
        assert p.last_response.speech == NEW and p.get_usage().total_tokens == 8
        assert p.last_error is None and len(requests) == 2
    finally:
        release.set()
        old.join(5)
        service.close()


def test_old_finally_and_repeated_cancel_cannot_unlock_active_new_request():
    old_started, new_started = threading.Event(), threading.Event()
    release_old, release_new = threading.Event(), threading.Event()

    def handle(data):
        if "【能動的な話しかけ】" in str(data["messages"]):
            old_started.set()
            assert release_old.wait(5)
            return reply(OLD, 204)
        new_started.set()
        assert release_new.wait(5)
        return reply(NEW)

    service = ChatService(
        provider=provider(handle), repository=SQLiteConversationRepository(":memory:")
    )
    service.start_new_conversation()
    old, outcomes = launch(service.generate_proactive_turn)
    current = None
    try:
        assert old_started.wait(5)
        service.invalidate_proactive_turn()
        current, new_outcomes = launch(lambda: service.send_message("新入力"))
        assert new_started.wait(5)
        service.invalidate_proactive_turn()
        release_old.set()
        old.join(5)
        assert isinstance(outcomes[0], ProactiveTurnInvalidatedError)
        assert service.is_busy
        with pytest.raises(ChatBusyError):
            service.send_message("重複入力")
        assert service.is_busy
        release_new.set()
        current.join(5)
        assert not current.is_alive() and new_outcomes[0].speech == NEW
        assert not service.is_busy
        assert [m.content for m in service.get_current_messages()] == ["新入力", NEW]
    finally:
        release_old.set()
        release_new.set()
        old.join(5)
        if current is not None:
            current.join(5)
        service.close()


def test_current_proactive_result_is_adopted_once_and_keeps_usage():
    p = provider(lambda _: reply(OLD, 204))
    service = ChatService(provider=p, repository=SQLiteConversationRepository(":memory:"))
    service.start_new_conversation()
    try:
        response = service.generate_proactive_turn()
        assert response.speech == p.last_response.speech == OLD
        assert service.last_usage.total_tokens == p.get_usage().total_tokens == 204
        assert [m.content for m in service.get_current_messages()] == [OLD]
        assert not service.is_busy
    finally:
        service.close()


@pytest.mark.parametrize("stage", ["before_run", "before_setup", "before_register"])
def test_worker_cancel_before_start_never_sends_or_saves(tmp_path, monkeypatch, stage):
    QApplication.instance() or QApplication([])
    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(provider(lambda _: pytest.fail("取消済み要求を送信しました")))
    conversation = backend.create_conversation("人工取消")
    worker = ProactiveWorker(backend, conversation, 1)
    cancelled, succeeded = [], []
    worker.cancelled.connect(cancelled.append)
    worker.succeeded.connect(lambda *args: succeeded.append(args))
    if stage == "before_run":
        worker.cancel()
    elif stage == "before_setup":
        original = backend._generate_proactive_turn

        def setup(*args, **kwargs):
            worker.cancel()
            backend.cancel_proactive_turn()
            return original(*args, **kwargs)

        monkeypatch.setattr(backend, "_generate_proactive_turn", setup)
    else:
        original = backend._service._generate_proactive_turn

        def register(*args, **kwargs):
            backend.cancel_proactive_turn()
            return original(*args, **kwargs)

        monkeypatch.setattr(backend._service, "_generate_proactive_turn", register)
    try:
        worker.run()
        assert cancelled == [1] and succeeded == []
        assert backend._service.get_current_messages() == [] and not backend._service.is_busy
    finally:
        backend.close()


def test_ui_sends_before_old_http_finishes_and_keeps_worker_until_cleanup(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        lambda *_: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat),
    )
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[1]))
    old_started, release = threading.Event(), threading.Event()
    requests = []

    def handle(data):
        requests.append(data)
        if "【能動的な話しかけ】" in str(data["messages"]):
            old_started.set()
            assert release.wait(5)
            return reply(OLD, 204)
        return reply(NEW)

    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(provider(handle))
    spoken = []
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window = ChatWindow(backend)
    window._proactive_timer.stop()
    window._current_conversation_id = backend.create_conversation("人工割り込み")
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(
            enabled=True, read_aloud=True, quiet_start_hour=0, quiet_end_hour=0
        ),
    )
    monkeypatch.setattr(window._proactive, "consider", lambda *_, **__: object())
    try:
        window.show()
        app.processEvents()
        window._on_proactive_timer()
        wait_until(old_started.is_set)
        old_thread, old_worker = window._proactive_thread, window._proactive_worker
        window.input_edit.setPlainText("新入力")
        window.send_button.click()
        wait_until(
            lambda: (
                window._thread is None
                and not window._busy
                and NEW in window.chat_view.toPlainText()
            )
        )
        assert not release.is_set() and old_thread.isRunning()
        assert window._proactive_thread is old_thread and window._proactive_worker is old_worker
        assert isValid(old_worker) and spoken == [NEW]
        assert backend._service.last_usage.total_tokens == 8
        window._on_proactive_timer()
        assert len(requests) == 2
        assert not window.close() and notices == ["能動発話を停止しました"]
        assert isValid(old_worker)
        release.set()
        wait_until(lambda: window._proactive_thread is None)
        assert not isValid(old_worker)
        assert OLD not in window.chat_view.toPlainText() and spoken == [NEW]
        assert [m.content for m in backend._service.get_current_messages()] == ["新入力", NEW]
        assert len(requests) == 2
    finally:
        release.set()
        wait_until(
            lambda: (
                window._thread is None
                and window._proactive_thread is None
                and window._voice_thread is None
                and not window._busy
            )
        )
        assert window.close()
        backend.close()


def test_ui_old_cleanup_does_not_disrupt_new_worker_still_running(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        lambda *_: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat),
    )
    old_started, new_started = threading.Event(), threading.Event()
    release_old, release_new = threading.Event(), threading.Event()

    def handle(data):
        if "【能動的な話しかけ】" in str(data["messages"]):
            old_started.set()
            assert release_old.wait(5)
            return reply(OLD, 204)
        new_started.set()
        assert release_new.wait(5)
        return reply(NEW)

    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(provider(handle))
    window = ChatWindow(backend)
    window._proactive_timer.stop()
    window._current_conversation_id = backend.create_conversation("人工終了順序")
    monkeypatch.setattr(
        window,
        "_proactive_settings",
        lambda: ProactiveSettings(enabled=True, quiet_start_hour=0, quiet_end_hour=0),
    )
    monkeypatch.setattr(window._proactive, "consider", lambda *_, **__: object())
    try:
        window.show()
        app.processEvents()
        window._on_proactive_timer()
        wait_until(old_started.is_set)
        window.input_edit.setPlainText("新入力")
        window.send_button.click()
        wait_until(new_started.is_set)
        current_thread, current_worker = window._thread, window._worker
        release_old.set()
        wait_until(lambda: window._proactive_thread is None)
        assert window._thread is current_thread and isValid(current_worker)
        assert current_thread.isRunning() and backend._service.is_busy
        release_new.set()
        wait_until(lambda: window._thread is None and not window._busy)
        assert NEW in window.chat_view.toPlainText() and OLD not in window.chat_view.toPlainText()
        assert window.error_label.isHidden()
    finally:
        release_old.set()
        release_new.set()
        wait_until(
            lambda: (
                window._thread is None
                and window._proactive_thread is None
                and window._voice_thread is None
                and not window._busy
            )
        )
        assert window.close()
        backend.close()


def test_backend_outcome_is_built_before_request_state_is_released(tmp_path, monkeypatch):
    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(provider(lambda _: reply(OLD, 204)))
    conversation = backend.create_conversation("人工状態組立")
    original = backend._build_send_outcome
    observed = []

    def build(response, **kwargs):
        observed.append(backend._service._state_lock._is_owned() and backend._service.is_busy)
        assert backend._service.last_usage.total_tokens == 204
        return original(response, **kwargs)

    monkeypatch.setattr(backend, "_build_send_outcome", build)
    try:
        outcome = backend.generate_proactive_turn(conversation)
        assert observed == [True] and outcome.input_tokens == 201 and outcome.output_tokens == 3
        assert not backend._service.is_busy
    finally:
        backend.close()
