from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
import pytest

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.integrations.voisona_talk import VoiSonaTalkVoiceOutput


@pytest.mark.parametrize("failure", ["runtime", "value", "timeout", 401, 404, 500])
def test_delete_failure_is_reported_after_other_requests_and_client_close(tmp_path, failure):
    deleted = []

    def handler(request):
        deleted.append(request.url.path)
        if request.url.path.endswith("/first"):
            if failure == "runtime":
                raise RuntimeError("人工的な削除失敗")
            if failure == "value":
                raise ValueError("人工的な削除失敗")
            if failure == "timeout":
                raise httpx.ReadTimeout("人工的な通信失敗")
            return httpx.Response(failure)
        return httpx.Response(204)

    output = VoiSonaTalkVoiceOutput(
        base_url="https://voice.invalid",
        username="test",
        password="test",
        transport=httpx.MockTransport(handler),
        wav_player=SimpleNamespace(stop=lambda: None),
        temp_root=tmp_path,
    )
    with output._lock:
        output._last_uuid = "first"
        output._stream_active_requests.add("second")
        output.close()
    assert not output._wait_for_cleanup(2)
    assert set(deleted) == {"/speech-syntheses/first", "/speech-syntheses/second"}
    assert output._client.is_closed
    assert output._cleanup_thread is None


@pytest.mark.parametrize("failure", ["runtime", "value", "timeout", 500])
def test_backend_close_reports_failure_only_after_late_post_is_reclaimed(tmp_path, failure):
    posted, release, failed, returned = (threading.Event() for _ in range(4))
    deleted, results = [], []

    def handler(request):
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "test"}])
        if request.method == "POST":
            posted.set()
            assert release.wait(5)
            return httpx.Response(200, json={"uuid": "late"})
        assert request.method == "DELETE"
        deleted.append(request.url.path)
        if request.url.path.endswith("/first"):
            failed.set()
            if failure == "runtime":
                raise RuntimeError("人工的な削除失敗")
            if failure == "value":
                raise ValueError("人工的な削除失敗")
            if failure == "timeout":
                raise httpx.ReadTimeout("人工的な通信失敗")
            return httpx.Response(failure)
        return httpx.Response(204)

    output = VoiSonaTalkVoiceOutput(
        base_url="https://voice.invalid",
        username="test",
        password="test",
        voice_name="test",
        transport=httpx.MockTransport(handler),
        wav_player=SimpleNamespace(stop=lambda: None),
        temp_root=tmp_path,
    )
    backend = object.__new__(ApplicationBackend)
    backend._service = SimpleNamespace(close=output.stop, voice_output=output)

    def close():
        try:
            backend.close()
        except RuntimeError:
            results.append("failed")
        else:
            results.append("success")
        finally:
            returned.set()

    poster = threading.Thread(target=output.speak, args=("遅れて受け付けられる音声",))
    closer = threading.Thread(target=close)
    poster.start()
    try:
        assert posted.wait(2)
        output._queue_cleanup(("first",))
        assert failed.wait(2)
        closer.start()
        assert not returned.wait(0.1)
        assert not output._client.is_closed
    finally:
        release.set()
        poster.join(3)
        if closer.ident is not None:
            closer.join(3)
        output.close()
        output._wait_for_cleanup(2)
    assert not poster.is_alive() and not closer.is_alive()
    assert results == ["failed"]
    assert set(deleted) == {"/speech-syntheses/first", "/speech-syntheses/late"}
    assert output._client.is_closed
    assert output._client_users == 0
    assert output._cleanup_thread is None
    assert not tuple(tmp_path.iterdir())
