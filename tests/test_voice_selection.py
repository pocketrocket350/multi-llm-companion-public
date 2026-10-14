from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from multi_llm_companion.integrations.voisona_talk import VoiSonaTalkVoiceOutput
from scripts import list_voisona_voices


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("requested", ["", "   ", "missing-private-value"])
def test_invalid_voice_stops_before_synthesis(
    requested: str, streaming: bool, tmp_path: Path
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"voice_name": "other"}])

    output = VoiSonaTalkVoiceOutput(
        base_url="http://localhost/api",
        username="artificial-user",
        password="artificial-password",
        voice_name=requested,
        transport=httpx.MockTransport(handler),
        temp_root=tmp_path,
    )
    try:
        if streaming:
            output.start_stream("人工テスト")
            output.finish_stream()
            deadline = time.monotonic() + 2
            while output.is_speaking and time.monotonic() < deadline:
                time.sleep(0.01)
        else:
            output.speak("人工テスト")
        assert output.last_error and "ボイス名" in output.last_error
        assert "missing-private-value" not in output.last_error
        assert "artificial-password" not in output.last_error
        assert not output.is_speaking
        assert not output.lip_sync_state.is_active
        assert all(request.method == "GET" for request in requests)
        assert len(requests) == (1 if requested.strip() else 0)
        assert not list(tmp_path.iterdir())
    finally:
        output.close()


@pytest.mark.parametrize("wrapper", ["list", "voices", "items"])
def test_exact_voice_selection_ignores_first_voice(wrapper: str) -> None:
    voices = [
        {"voice_name": "other", "voice_version": "1"},
        {"voice_name": "adopted", "voice_version": "2"},
    ]
    payload = voices if wrapper == "list" else {wrapper: voices}
    output = VoiSonaTalkVoiceOutput(
        base_url="http://localhost/api",
        username="artificial-user",
        password="artificial-password",
        voice_name="adopted",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    try:
        assert output._resolve_voice() == ("adopted", "2")
    finally:
        output.close()


@pytest.mark.parametrize("failure", [False, True])
def test_voice_listing_only_reads_names_and_hides_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: bool,
) -> None:
    config = tmp_path / "audio.yaml"
    config.write_text("voice_output: {}", encoding="utf-8")
    monkeypatch.setenv("VOISONA_TALK_USERNAME", "artificial-user")
    monkeypatch.setenv("VOISONA_TALK_PASSWORD", "hidden-credential")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure:
            raise httpx.ConnectError("hidden-credential")
        return httpx.Response(200, json={"voices": [{"voice_name": "test-voice"}]})

    client_type = httpx.Client
    monkeypatch.setattr(
        list_voisona_voices.httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(handler)),
    )
    before = config.read_bytes()
    assert list_voisona_voices.main(config) == int(failure)
    printed = capsys.readouterr().out
    assert "hidden-credential" not in printed
    assert "artificial-user" not in printed
    if not failure:
        assert json.dumps("test-voice") in printed
    assert len(requests) == 1 and requests[0].method == "GET"
    assert requests[0].url.path.endswith("/voices")
    assert config.read_bytes() == before
