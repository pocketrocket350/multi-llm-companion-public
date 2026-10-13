from __future__ import annotations

import json
import sys
import threading
import time
import wave
from array import array
from ctypes import POINTER, c_void_p, cast, sizeof
from io import BytesIO
from pathlib import Path
from queue import Queue

import httpx
import pytest
import sounddevice

from multi_llm_companion.config import AudioConfig, SpeechRecognitionConfig
from multi_llm_companion.domain import Emotion, ExpressionChannel, ExpressionOutputStatus
from multi_llm_companion.integrations import (
    VoiSonaTalkVoiceOutput,
    VoskSpeechRecognizer,
    WhisperSpeechRecognizer,
    build_speech_recognizer,
)
from multi_llm_companion.integrations.audio_capture import meter_value, normalized_level
from multi_llm_companion.integrations.lip_sync import JapaneseLipSyncSchedule
from multi_llm_companion.integrations.voisona_talk import _PreparedStreamSegment
from multi_llm_companion.integrations.wav_playback import (
    SoundDeviceWavPlayer,
    WinMmWavPlayer,
    _MmTime,
    create_wav_player,
)


class FakeWavPlayer:
    def __init__(
        self,
        *,
        finish_on_play: bool = False,
        lead_in_seconds: float = 0.0,
    ) -> None:
        self.is_playing = False
        self.position_seconds = 0.0
        self.duration_seconds = 0.0
        self.amplitude = 0.5
        self.finish_on_play = finish_on_play
        self.lead_in_seconds = lead_in_seconds
        self.loaded = False

    def load(self, wav_data: bytes) -> None:
        with wave.open(BytesIO(wav_data), "rb") as wav_file:
            self.duration_seconds = (
                wav_file.getnframes() / wav_file.getframerate() + self.lead_in_seconds
            )
        self.position_seconds = 0.0
        self.loaded = True

    def play(self) -> None:
        self.is_playing = not self.finish_on_play

    def stop(self) -> None:
        self.is_playing = False
        self.position_seconds = 0.0


def _wav_bytes(duration: float, sample_rate: int = 48_000) -> bytes:
    output = BytesIO()
    frame_count = round(duration * sample_rate)
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(array("h", [3000] * frame_count).tobytes())
    return output.getvalue()


def test_vosk_reports_missing_model_in_japanese(tmp_path: Path) -> None:
    recognizer = VoskSpeechRecognizer(tmp_path / "missing-model")
    errors: list[str] = []

    recognizer.start(lambda _text: None, errors.append)

    assert recognizer.is_available is False
    assert errors and "モデルが見つかりません" in errors[0]


def test_audio_level_distinguishes_silence_and_voice() -> None:
    silence = array("h", [0] * 100).tobytes()
    voice = array("h", [5000, -5000] * 50).tobytes()

    assert normalized_level(silence) == 0
    assert meter_value(normalized_level(voice)) > 50


def test_factory_switches_between_whisper_and_vosk(tmp_path: Path) -> None:
    whisper_config = AudioConfig(
        speech_recognition=SpeechRecognitionConfig(enabled=True, provider="whisper")
    )
    vosk_config = AudioConfig(
        speech_recognition=SpeechRecognitionConfig(enabled=True, provider="vosk")
    )

    assert isinstance(build_speech_recognizer(whisper_config, tmp_path), WhisperSpeechRecognizer)
    assert isinstance(build_speech_recognizer(vosk_config, tmp_path), VoskSpeechRecognizer)


def test_whisper_converts_audio_with_injected_model() -> None:
    class Segment:
        text = "こんにちは"

    class FakeModel:
        def transcribe(self, _audio: object, **_options: object) -> tuple[list[Segment], None]:
            return [Segment()], None

    recognizer = WhisperSpeechRecognizer()
    recognizer._model = FakeModel()  # type: ignore[attr-defined]

    assert recognizer._transcribe(array("h", [1000] * 1600).tobytes(), beam_size=1) == "こんにちは"


def test_voisona_requests_file_synthesis_without_exposing_password(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    status_checks = [
        {"state": "running", "progress_percentage": 50},
        {"state": "succeeded", "progress_percentage": 100},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(
                200,
                json=[{"voice_name": "original-voice", "voice_version": "1.0"}],
            )
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.1))
            return httpx.Response(200, json={"uuid": "test-uuid"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json=status_checks.pop(0))

    secret = "do-not-print-this-password"
    player = FakeWavPlayer(finish_on_play=True)
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user@example.invalid",
        password=secret,
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )

    output.speak("こんにちは")

    assert output.last_error is None
    # 合成受付直後に最初の完了照会を行う。
    assert len(requests) == 3
    payload = json.loads(requests[1].content)
    assert payload["text"] == "こんにちは"
    assert payload["voice_name"] == "original-voice"
    assert payload["destination"] == "file"
    assert payload["can_overwrite_file"] is True
    assert Path(payload["output_file_path"]).is_relative_to(tmp_path)
    assert secret not in repr(output)
    assert output.is_speaking is True
    generation = output._monitor_generation  # type: ignore[attr-defined]
    output._monitor_request("test-uuid", generation)
    assert player.loaded is True
    assert output.is_speaking is False
    assert output.playback_status_text == "停止中"
    assert not tuple(tmp_path.iterdir())


def test_voisona_stream_synthesizes_sentences_in_order_and_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    played_texts: list[str] = []
    post_lock = threading.Lock()
    active_posts = 0
    max_active_posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            nonlocal active_posts, max_active_posts
            with post_lock:
                active_posts += 1
                max_active_posts = max(max_active_posts, active_posts)
            try:
                if payload["text"] == "一文目です。":
                    time.sleep(0.05)
                Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.01))
                return httpx.Response(200, json={"uuid": f"stream-{len(requests)}"})
            finally:
                with post_lock:
                    active_posts -= 1
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"state": "succeeded"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=0.01,
        transport=httpx.MockTransport(handler),
        wav_player=FakeWavPlayer(finish_on_play=True),
        temp_root=tmp_path,
    )

    original_play = output._play_stream_segment  # type: ignore[attr-defined]

    def record_play(
        generation: int,
        prepared: object,
        *,
        playback_gap_ms: int | None = None,
    ) -> bool:
        played_texts.append(prepared.text)  # type: ignore[attr-defined]
        return original_play(
            generation,
            prepared,  # type: ignore[arg-type]
            playback_gap_ms=playback_gap_ms,
        )

    monkeypatch.setattr(output, "_play_stream_segment", record_play)

    output.start_stream("一文目です。")
    output.enqueue_stream("二文目です！")
    output.finish_stream()

    deadline = time.monotonic() + 3.0
    while output.is_speaking and time.monotonic() < deadline:
        time.sleep(0.01)

    post_texts = [
        json.loads(request.content)["text"] for request in requests if request.method == "POST"
    ]
    assert sorted(post_texts) == sorted(["一文目です。", "二文目です！"])
    assert played_texts == ["一文目です。", "二文目です！"]
    assert max_active_posts >= 2
    assert output.is_speaking is False
    assert output.last_error is None
    assert not tuple(tmp_path.iterdir())
    output.close()


def test_voisona_starts_first_wav_without_waiting_for_next_sentence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_played_at: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            if payload["text"] == "二文目。":
                time.sleep(0.25)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.4))
            return httpx.Response(200, json={"uuid": payload["text"]})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"state": "succeeded"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=0.01,
        transport=httpx.MockTransport(handler),
        wav_player=FakeWavPlayer(finish_on_play=True),
        temp_root=tmp_path,
    )
    started_at = time.monotonic()

    original_play = output._play_stream_segment  # type: ignore[attr-defined]

    def record_first_playback(
        generation: int,
        prepared: _PreparedStreamSegment,
        *,
        playback_gap_ms: int | None = None,
    ) -> bool:
        if prepared.index == 0:
            first_played_at.append(time.monotonic())
        return original_play(generation, prepared, playback_gap_ms=playback_gap_ms)

    monkeypatch.setattr(output, "_play_stream_segment", record_first_playback)
    output.start_stream("一文目。")
    output.enqueue_stream("二文目。")
    output.finish_stream()

    deadline = started_at + 2.0
    while not first_played_at and time.monotonic() < deadline:
        time.sleep(0.01)

    assert first_played_at
    assert first_played_at[0] - started_at < 0.15
    output.close()


def test_voisona_stream_drains_completed_sentences_during_playback(
    tmp_path: Path,
) -> None:
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(204)),
        wav_player=FakeWavPlayer(finish_on_play=True),
        temp_root=tmp_path,
    )
    pending: dict[int, _PreparedStreamSegment] = {}
    pending_queue: Queue[_PreparedStreamSegment | None] = Queue()
    pending_queue.put(_PreparedStreamSegment(1, "二文目。", b"", {}, "second"))
    pending_queue.put(None)

    finished = output._drain_stream_output_queue(  # type: ignore[attr-defined]
        pending_queue,
        pending,
    )

    assert finished is True
    assert tuple(pending) == (1,)
    output.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "field,state",
    [("state", "failed"), ("status", "error"), ("state", "canceled"), ("state", "cancelled")],
)
def test_voisona_terminal_failure_stops_without_waiting_for_timeout(
    tmp_path: Path, stream: bool, field: str, state: str
) -> None:
    failed_poll = threading.Event()
    posts = []

    def handler(request):
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            posts.append(payload["text"])
            if payload["text"] == "復帰。":
                Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.01))
            return httpx.Response(
                200, json={"uuid": "recovery" if payload["text"] == "復帰。" else "failed"}
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith("/recovery"):
            return httpx.Response(200, json={"state": "succeeded"})
        failed_poll.set()
        return httpx.Response(200, json={field: state, "completed": True})

    player = FakeWavPlayer(finish_on_play=True)
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    try:
        if stream:
            output.start_stream("合成に失敗する文。")
            output.finish_stream()
        else:
            output.speak("合成に失敗する文。")
        assert failed_poll.wait(1)
        deadline = time.monotonic() + 1
        while output.is_speaking and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not output.is_speaking
        assert output.last_error and "音声生成に失敗" in output.last_error
        assert not player.is_playing and not output.lip_sync_state.is_active
        output.enqueue_stream("失敗した発話の遅着。")
        assert posts == ["合成に失敗する文。"]
        output.start_stream("復帰。")
        output.finish_stream()
        deadline = time.monotonic() + 1
        while output.is_speaking and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not output.is_speaking and output.last_error is None
        assert posts == ["合成に失敗する文。", "復帰。"]
    finally:
        output.close()


def test_voisona_failed_prefetch_stops_current_wav(tmp_path: Path) -> None:
    playing = threading.Event()
    player = FakeWavPlayer()
    original_play = player.play

    def play():
        original_play()
        playing.set()

    player.play = play

    def handler(request):
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(10))
            return httpx.Response(
                200, json={"uuid": "first" if payload["text"] == "一文目。" else "failed"}
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith("/failed"):
            assert playing.wait(1)
            return httpx.Response(200, json={"state": "failed"})
        return httpx.Response(200, json={"state": "succeeded"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    try:
        output.start_stream("一文目。")
        output.enqueue_stream("失敗する二文目。")
        output.finish_stream()
        assert playing.wait(1)
        deadline = time.monotonic() + 1
        while output.is_speaking and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not output.is_speaking and not player.is_playing
        assert not output.lip_sync_state.is_active
        assert output.last_error and "音声生成に失敗" in output.last_error
    finally:
        output.close()


def test_voisona_expression_disabled_keeps_legacy_request_payload(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"uuid": "test-uuid"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        expression_enabled=False,
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=FakeWavPlayer(),
        temp_root=tmp_path,
    )

    output.speak_with_expression(
        "従来互換",
        ExpressionChannel(Emotion.HAPPY, 0.55),
    )

    payload = json.loads(requests[1].content)
    assert "global_parameters" not in payload
    assert output.expression_status_text == "無効（従来動作）"
    output.stop()


def test_legacy_speak_never_adds_expression_parameters_when_feature_is_enabled(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"uuid": "test-uuid"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        expression_enabled=True,
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=FakeWavPlayer(),
        temp_root=tmp_path,
    )

    output.speak("従来呼び出し")

    assert "global_parameters" not in json.loads(requests[1].content)
    output.stop()


@pytest.mark.parametrize(
    ("emotion", "speed_direction", "intonation_direction"),
    [
        (Emotion.HAPPY, 1, 1),
        (Emotion.SAD, -1, -1),
        (Emotion.ANGRY, 1, 1),
        (Emotion.EMBARRASSED, -1, -1),
        (Emotion.SURPRISED, 1, 1),
    ],
)
def test_voisona_expression_uses_only_bounded_voice_independent_parameters(
    tmp_path: Path,
    emotion: Emotion,
    speed_direction: int,
    intonation_direction: int,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"uuid": "test-uuid"})

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        expression_enabled=True,
        max_speed_change=0.25,
        max_intonation_change=0.50,
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=FakeWavPlayer(),
        temp_root=tmp_path,
    )

    output.speak_with_expression(
        "感情表現",
        ExpressionChannel(emotion, 1.0, ExpressionOutputStatus.APPLIED),
    )

    payload = json.loads(requests[1].content)
    assert set(payload["global_parameters"]) == {"speed", "intonation"}
    speed = payload["global_parameters"]["speed"]
    intonation = payload["global_parameters"]["intonation"]
    assert 0.85 <= speed <= 1.15
    assert 0.80 <= intonation <= 1.20
    assert (speed - 1.0) * speed_direction > 0.0
    assert (intonation - 1.0) * intonation_direction > 0.0
    assert "style_weights" not in payload
    output.stop()


def test_voisona_neutral_or_unsupported_expression_adds_no_parameters(tmp_path: Path) -> None:
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        expression_enabled=True,
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        wav_player=FakeWavPlayer(),
        temp_root=tmp_path,
    )

    assert output._expression_parameters(ExpressionChannel(Emotion.NEUTRAL, 0.55)) is None
    assert (
        output._expression_parameters(
            ExpressionChannel(
                Emotion.HAPPY,
                0.55,
                ExpressionOutputStatus.UNSUPPORTED,
            )
        )
        is None
    )


def test_voisona_stop_rejects_wav_from_an_old_synthesis_generation(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"uuid": "old-uuid"})

    player = FakeWavPlayer()
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    output.speak("停止される音声")
    old_generation = output._monitor_generation  # type: ignore[attr-defined]

    output.stop()
    started = output._start_downloaded_audio(  # type: ignore[attr-defined]
        "old-uuid",
        old_generation,
        {"state": "succeeded"},
    )

    assert started is False
    assert player.loaded is False
    assert output.is_speaking is False
    assert output.lip_sync_state.is_active is False


def test_japanese_lip_sync_schedule_follows_vowels_after_start_time() -> None:
    schedule = JapaneseLipSyncSchedule.from_text(
        "あいうえお。",
        start_at=1.0,
        character_duration=0.1,
    )

    assert schedule.state_at(0.99).is_active is False
    assert schedule.state_at(1.05).mouth_shape == "a"
    assert schedule.state_at(1.15).mouth_shape == "i"
    assert schedule.state_at(1.25).mouth_shape == "u"
    assert schedule.state_at(1.35).mouth_shape == "e"
    assert schedule.state_at(1.45).mouth_shape == "o"
    assert schedule.state_at(1.55).is_active is False
    assert schedule.state_at(1.55).is_timeline_active is True
    assert schedule.state_at_progress(0.05).mouth_shape == "a"
    assert schedule.state_at_progress(0.20).mouth_shape == "i"
    assert schedule.state_at_progress(1.0).is_active is False
    assert abs(schedule.end_at - 1.72) < 0.0001
    rebased = schedule.rebased(5.0)
    assert rebased.state_at(5.05).mouth_shape == "a"
    assert abs(rebased.end_at - 5.72) < 0.0001


def test_phoneme_timing_schedule_uses_voisona_times_without_estimation() -> None:
    schedule = JapaneseLipSyncSchedule.from_phoneme_timings(
        ((0.0, 0.12, "k"), (0.12, 0.24, "a"), (0.24, 0.32, "sil")),
        start_at=5.0,
    )

    assert schedule is not None
    assert schedule.state_at(5.05).mouth_shape == "neutral"
    state = schedule.state_at(5.15)
    assert state.mouth_shape == "a"
    assert state.is_estimated is False
    assert state.is_timeline_active is True


def test_sounddevice_player_reads_wav_duration_without_playing_audio() -> None:
    player = SoundDeviceWavPlayer()

    player.load(_wav_bytes(0.25))

    assert abs(player.duration_seconds - 0.25) < 0.0001
    assert player.position_seconds == 0.0
    assert player.is_playing is False


def test_sounddevice_player_callback_advances_real_wav_position() -> None:
    class FakeStream:
        time = 10.05

    class FakeTimeInfo:
        outputBufferDacTime = 10.0

    player = SoundDeviceWavPlayer()
    player.load(_wav_bytes(0.1, sample_rate=1_000))
    player._stream = FakeStream()  # type: ignore[attr-defined]

    player._audio_callback(  # type: ignore[attr-defined]
        bytearray(50 * 2),
        50,
        FakeTimeInfo(),
        None,
    )

    assert abs(player.position_seconds - 0.05) < 0.0001
    assert player.amplitude > 0.0
    with pytest.raises(sounddevice.CallbackStop):
        player._audio_callback(  # type: ignore[attr-defined]
            bytearray(50 * 2),
            50,
            FakeTimeInfo(),
            None,
        )
    # 全データを機器へ先送りしても、再生位置は実出力時計の50ミリ秒のまま。
    assert abs(player.position_seconds - 0.05) < 0.0001


def test_windows_native_player_uses_device_sample_position_without_callback() -> None:
    class FakeWinMm:
        def __init__(self) -> None:
            self.samples = 0
            self.calls: list[str] = []

        def waveOutOpen(self, handle_pointer: object, *_args: object) -> int:  # noqa: N802
            cast(handle_pointer, POINTER(c_void_p)).contents.value = 123
            self.calls.append("open")
            return 0

        def waveOutPrepareHeader(self, *_args: object) -> int:  # noqa: N802
            self.calls.append("prepare")
            return 0

        def waveOutWrite(self, *_args: object) -> int:  # noqa: N802
            self.calls.append("write")
            return 0

        def waveOutGetPosition(self, _handle: object, time_pointer: object, *_args: object) -> int:  # noqa: N802
            cast(time_pointer, POINTER(_MmTime)).contents.sample = self.samples
            self.calls.append("position")
            return 0

        def waveOutReset(self, *_args: object) -> int:  # noqa: N802
            self.calls.append("reset")
            return 0

        def waveOutUnprepareHeader(self, *_args: object) -> int:  # noqa: N802
            self.calls.append("unprepare")
            return 0

        def waveOutClose(self, *_args: object) -> int:  # noqa: N802
            self.calls.append("close")
            return 0

    api = FakeWinMm()
    player = WinMmWavPlayer(api=api, lead_in_seconds=0.0)
    player.load(_wav_bytes(0.1, sample_rate=1_000))
    player.play()
    api.samples = 50

    assert player.is_playing is True
    assert abs(player.position_seconds - 0.05) < 0.0001
    assert player.amplitude > 0.0
    assert api.calls[:3] == ["open", "prepare", "write"]
    # 音声コールバックを通らず、Windowsの再生サンプル位置だけを読む。
    assert not hasattr(player, "_audio_callback")

    header = player._header  # type: ignore[attr-defined]
    assert header is not None
    header.dwFlags = player._WHDR_DONE  # type: ignore[attr-defined]
    assert player.is_playing is False
    player.stop()
    assert "close" in api.calls


def test_windows_native_player_prepends_preroll_without_removing_speech() -> None:
    player = WinMmWavPlayer(api=object(), lead_in_seconds=0.12)  # type: ignore[arg-type]

    player.load(_wav_bytes(0.1, sample_rate=1_000))

    assert abs(player.lead_in_seconds - 0.12) < 0.0001
    assert abs(player.duration_seconds - 0.22) < 0.0001
    assert player._pcm[: 120 * 2] == bytes(120 * 2)  # type: ignore[attr-defined]
    assert (
        player._pcm[120 * 2 :]
        == array(  # type: ignore[attr-defined]
            "h", [3000] * 100
        ).tobytes()
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows専用の既定再生器")
def test_windows_uses_native_wav_player_and_correct_mmtime_size() -> None:
    assert isinstance(create_wav_player(), WinMmWavPlayer)
    # MMTIMEは共用領域を含む12バイト。8バイトではWindows APIが位置取得を拒否する。
    assert sizeof(_MmTime) == 12


def test_voisona_prefers_returned_phoneme_timings_during_managed_playback(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.4))
            return httpx.Response(200, json={"uuid": "test-uuid"})
        return httpx.Response(200, json={"state": "running", "progress_percentage": 25})

    player = FakeWavPlayer()
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )

    output.speak("あい")
    result = {
        "state": "succeeded",
        "duration": 0.4,
        "phoneme_timings": [
            {"phoneme": "a", "start": 0.0, "end": 0.2},
            {"phoneme": "i", "start": 0.2, "end": 0.4},
        ],
    }
    generation = output._monitor_generation  # type: ignore[attr-defined]
    started = output._start_downloaded_audio(  # type: ignore[attr-defined]
        "test-uuid",
        generation,
        result,
    )
    player.position_seconds = 0.05
    state = output.lip_sync_state

    assert started is True
    assert state.mouth_shape == "a"
    assert state.is_estimated is False
    output.stop()


def test_voisona_lip_sync_uses_managed_wav_position_and_closes_at_end(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(1.0))
        return httpx.Response(200, json={"uuid": "test-uuid"})

    player = FakeWavPlayer()
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        lip_sync_character_duration=0.1,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )

    output.speak("あいうえお")
    assert output.lip_sync_state.is_active is False
    generation = output._monitor_generation  # type: ignore[attr-defined]
    started = output._start_downloaded_audio(  # type: ignore[attr-defined]
        "test-uuid",
        generation,
        {"state": "succeeded", "progress_percentage": 100},
    )
    for elapsed, expected in ((0.05, "a"), (0.25, "i"), (0.45, "u"), (0.65, "e"), (0.85, "o")):
        player.position_seconds = elapsed
        assert output.lip_sync_state.mouth_shape == expected
    assert started is True
    output._finish_request("test-uuid", generation)  # type: ignore[attr-defined]
    assert output.lip_sync_state.is_active is False
    output.stop()
    assert output.lip_sync_state.is_active is False


def test_voisona_stop_records_when_an_active_wav_player_has_stopped(monkeypatch) -> None:
    records: list[str] = []
    monkeypatch.setattr(
        "multi_llm_companion.integrations.voisona_talk.write_lip_sync_trace",
        lambda _source, event, **_fields: records.append(event),
    )
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        wav_player=FakeWavPlayer(),
    )
    output._playback_started = True  # type: ignore[attr-defined]

    output.stop()

    assert records == ["playback_stop_completed"]


def test_voisona_lip_sync_uses_wav_duration_when_api_omits_duration(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.2))
        return httpx.Response(200, json={"uuid": "test-uuid"})

    player = FakeWavPlayer()
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        lip_sync_character_duration=0.1,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    output.speak("あい")
    generation = output._monitor_generation  # type: ignore[attr-defined]
    started = output._start_downloaded_audio(  # type: ignore[attr-defined]
        "test-uuid",
        generation,
        {"state": "succeeded"},
    )

    player.position_seconds = 0.05
    assert output.lip_sync_state.mouth_shape == "a"
    player.position_seconds = 0.15
    assert output.lip_sync_state.mouth_shape == "i"
    assert started is True
    output.stop()


def test_voisona_preroll_does_not_advance_lip_sync_or_shorten_schedule(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "POST":
            payload = json.loads(request.content)
            Path(payload["output_file_path"]).write_bytes(_wav_bytes(0.2))
        return httpx.Response(200, json={"uuid": "test-uuid"})

    player = FakeWavPlayer(lead_in_seconds=0.1)
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        lip_sync_character_duration=0.1,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    output.speak("あい")
    generation = output._monitor_generation  # type: ignore[attr-defined]
    assert output._start_downloaded_audio(  # type: ignore[attr-defined]
        "test-uuid",
        generation,
        {"state": "succeeded"},
    )

    player.position_seconds = 0.05
    assert output.lip_sync_state.is_active is False
    player.position_seconds = 0.15
    assert output.lip_sync_state.mouth_shape == "a"
    player.position_seconds = 0.25
    assert output.lip_sync_state.mouth_shape == "i"
    output.stop()


def test_voisona_missing_wav_stops_safely_and_reports_japanese(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "original-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"uuid": "test-uuid"})

    player = FakeWavPlayer()
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        monitor_interval=999,
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
    )
    output.speak("あ")
    generation = output._monitor_generation  # type: ignore[attr-defined]

    started = output._start_downloaded_audio(  # type: ignore[attr-defined]
        "test-uuid",
        generation,
        {"state": "succeeded"},
    )

    assert started is False
    assert output.is_speaking is False
    assert output.lip_sync_state.is_active is False
    assert output.last_error == "VoiSona TalkのWAV音声を再生できませんでした。"
    assert not tuple(tmp_path.iterdir())


def test_voisona_connection_error_is_safe_japanese() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-connection-details")

    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        transport=httpx.MockTransport(handler),
    )

    output.speak("テスト")

    assert output.last_error is not None
    assert "secret" not in output.last_error


def test_voisona_cleanup_reraises_unexpected_closed_client_error(tmp_path: Path) -> None:
    output = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="user",
        password="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(204)),
        temp_root=tmp_path,
    )
    output._client.close()  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError):
        output._delete_synthesis("request")  # type: ignore[attr-defined]

    output._is_closed = True  # type: ignore[attr-defined]
    output._delete_synthesis("request")  # type: ignore[attr-defined]
