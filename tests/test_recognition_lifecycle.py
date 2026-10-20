from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from multi_llm_companion.integrations import VoskSpeechRecognizer, WhisperSpeechRecognizer
from multi_llm_companion.integrations.noop import NullSpeechRecognizer
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService
from multi_llm_companion.storage import SQLiteConversationRepository


@pytest.mark.parametrize("engine", ["whisper", "vosk"])
@pytest.mark.parametrize("operation", ["restart", "replace"])
def test_restart_is_rejected_until_stopped_worker_has_exited(
    tmp_path, monkeypatch, request, engine, operation
):
    streams = []
    entered, release = threading.Event(), threading.Event()
    recognized, errors = [], []

    class Stream:
        def __init__(self, **kwargs):
            self.closed = False
            streams.append(self)

        def abort(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawInputStream=Stream))
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=lambda *args, **kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "vosk",
        SimpleNamespace(Model=lambda path: object(), KaldiRecognizer=lambda *args: object()),
    )
    recognizer = (
        WhisperSpeechRecognizer()
        if engine == "whisper"
        else VoskSpeechRecognizer(model_path=tmp_path)
    )
    service = ChatService(
        provider=MockProvider(),
        repository=SQLiteConversationRepository(":memory:"),
        speech_recognizer=recognizer,
    )
    request.addfinalizer(service.close)
    replacement = NullSpeechRecognizer()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(8)
        return "古い結果"

    if engine == "whisper":
        monkeypatch.setattr(recognizer, "_capture", lambda callback: b"\x00\x01")
        monkeypatch.setattr(recognizer, "_transcribe", blocked)
    else:
        monkeypatch.setattr(recognizer, "_capture", blocked)
    recognizer.start(recognized.append, errors.append)
    worker = recognizer._thread
    try:
        assert entered.wait(3)
        if operation == "restart":
            recognizer.stop()
            recognizer.start(recognized.append, errors.append)
            assert errors and "停止処理中" in errors[-1]
        else:
            with pytest.raises(RuntimeError, match="停止処理中"):
                service.set_speech_recognizer(replacement)
            assert service.speech_recognizer is recognizer
        assert worker.is_alive()
        assert recognizer._thread is worker
        assert recognizer._stop_event.is_set()
        assert len(streams) == 1
    finally:
        release.set()
        worker.join(5)
        recognizer.stop()
    assert not worker.is_alive()
    assert recognized == []
    assert all(stream.closed for stream in streams)
    monkeypatch.setattr(recognizer, "_capture", lambda *args: b"")
    errors.clear()
    recognizer.start(recognized.append, errors.append)
    restarted = recognizer._thread
    restarted.join(3)
    assert not restarted.is_alive()
    assert len(streams) == 2
    assert not any("停止処理中" in message for message in errors)
    assert not recognizer.is_listening
    if operation == "replace":
        service.set_speech_recognizer(replacement)
        assert service.speech_recognizer is replacement
