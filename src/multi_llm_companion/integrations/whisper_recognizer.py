"""faster-whisperを使う高精度なオフライン音声認識。"""

from __future__ import annotations

import os
import queue
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from .audio_capture import input_devices, meter_value, normalized_level
from .base import (
    AudioInputDevice,
    IntegrationErrorCallback,
    LevelCallback,
    RecognitionCallback,
    SpeechRecognitionPort,
)
from .lip_sync_diagnostic import write_lip_sync_trace


class WhisperSpeechRecognizer(SpeechRecognitionPort):
    """発話を録音し、Whisperで日本語へ変換する。"""

    def __init__(
        self,
        *,
        model_name: str = "small",
        input_device: int | str | None = None,
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 1.2,
        max_recording_duration: float = 30.0,
        partial_interval: float = 2.5,
        device: str = "cpu",
        compute_type: str = "int8",
        model_root: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._input_device = input_device
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._max_recording_duration = max_recording_duration
        self._partial_interval = partial_interval
        self._device = device
        self._compute_type = compute_type
        self._model_root = model_root or Path("local/models/faster-whisper-small")
        self._listening = False
        self._state = "待機中 (Whisper)"
        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._audio_queue: queue.Queue[tuple[bytes, float]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._listening

    @property
    def status_text(self) -> str:
        if not self.is_available:
            return "Whisper未インストール"
        if self._model is None and not any(self._model_root.rglob("model.bin")):
            return "Whisperモデル未取得"
        return self._state

    @property
    def engine_name(self) -> str:
        return "whisper"

    @property
    def input_device(self) -> int | str | None:
        return self._input_device

    def list_input_devices(self) -> list[AudioInputDevice]:
        return input_devices()

    def start(
        self,
        on_recognized: RecognitionCallback,
        on_error: IntegrationErrorCallback | None = None,
        on_partial: RecognitionCallback | None = None,
        on_level: LevelCallback | None = None,
    ) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop_event.is_set():
                    self._report(
                        on_error, "前の音声認識の停止処理中です。少し待って再開してください。"
                    )
                return
            if self._listening:
                return
            if not self.is_available:
                self._report(on_error, "Whisperが未導入です。setup_windows.batを実行してください。")
                return
            self._stop_event.clear()
            self._finish_event.clear()
            self._clear_queue()
            self._listening = True
            self._state = "モデル準備中 (Whisper)"
            self._thread = threading.Thread(
                target=self._recognize,
                args=(on_recognized, on_error, on_partial, on_level),
                name="whisper-speech-recognition",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            stream = self._stream
            thread = self._thread
            self._listening = False
            self._state = "停止処理中 (Whisper)"
        if stream is not None:
            with suppress(Exception):
                stream.abort()
                stream.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._stream = None
                self._thread = None
                self._state = "待機中 (Whisper)"

    def finish(self) -> None:
        """録音だけを終え、取得済み音声の認識は継続する。"""

        self._finish_event.set()

    def _recognize(
        self,
        on_recognized: RecognitionCallback,
        on_error: IntegrationErrorCallback | None,
        on_partial: RecognitionCallback | None,
        on_level: LevelCallback | None,
    ) -> None:
        try:
            import sounddevice

            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from faster_whisper import WhisperModel

            if self._model is None:
                model_source = (
                    str(self._model_root)
                    if (self._model_root / "model.bin").is_file()
                    else self._model_name
                )
                self._model = WhisperModel(
                    model_source,
                    device=self._device,
                    compute_type=self._compute_type,
                    download_root=str(self._model_root),
                )
            if self._stop_event.is_set():
                return
            self._state = "聞き取り中 (Whisper)"

            def audio_callback(
                indata: bytes,
                _frames: int,
                _time: Any,
                status: Any,
            ) -> None:
                if status:
                    return
                data = bytes(indata)
                level = normalized_level(data)
                self._audio_queue.put((data, level))
                if on_level is not None:
                    on_level(meter_value(level))

            self._stream = sounddevice.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=2000,
                device=self._input_device,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            )
            audio = self._capture(on_partial)
            if audio and not self._stop_event.is_set():
                self._state = "認識処理中 (Whisper)"
                text = self._transcribe(audio, beam_size=5)
                if self._stop_event.is_set():
                    return
                if text:
                    on_recognized(text)
                else:
                    self._report(
                        on_error, "音声を認識できませんでした。マイク音量を確認してください。"
                    )
            elif not self._stop_event.is_set():
                self._report(
                    on_error, "音声を検出できませんでした。マイク選択と音量を確認してください。"
                )
        except Exception as exc:
            if not self._stop_event.is_set():
                self._report(on_error, self._safe_error(exc))
        finally:
            if on_level is not None:
                on_level(0)
            stream = self._stream
            if stream is not None:
                with suppress(Exception):
                    stream.close()
            with self._lock:
                self._listening = False
                self._stream = None
                self._state = "待機中 (Whisper)"

    def _capture(self, on_partial: RecognitionCallback | None) -> bytes:
        chunks: list[bytes] = []
        spoken = False
        silence = 0.0
        elapsed = 0.0
        next_partial = self._partial_interval
        block_seconds = 2000 / self._sample_rate
        with self._stream:
            while (
                not self._stop_event.is_set()
                and not self._finish_event.is_set()
                and elapsed < self._max_recording_duration
            ):
                try:
                    data, level = self._audio_queue.get(timeout=0.3)
                except queue.Empty:
                    continue
                chunks.append(data)
                elapsed += len(data) / 2 / self._sample_rate
                if level >= self._silence_threshold:
                    spoken = True
                    silence = 0.0
                elif spoken:
                    silence += block_seconds

                if spoken and on_partial is not None and elapsed >= next_partial:
                    partial = self._transcribe(b"".join(chunks), beam_size=1)
                    if partial and not self._stop_event.is_set():
                        on_partial(partial)
                    next_partial = elapsed + self._partial_interval
                if spoken and silence >= self._silence_duration:
                    write_lip_sync_trace("recognition", "speech_end_detected")
                    break
        return b"".join(chunks) if spoken else b""

    def _transcribe(self, raw_audio: bytes, *, beam_size: int) -> str:
        import numpy as np

        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language="ja",
            task="transcribe",
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def _clear_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _report(callback: IntegrationErrorCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if type(exc).__name__ in {"PortAudioError", "OSError"}:
            return "マイクを開始できません。Windowsのマイク権限と入力機器を確認してください。"
        return (
            "Whisperを準備または実行できませんでした。初回はインターネット接続、"
            "空き容量、モデル設定を確認してください。"
        )
