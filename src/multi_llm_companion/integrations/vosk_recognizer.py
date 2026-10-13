"""VoskとWindowsのマイクを使う軽量なオフライン音声認識。"""

from __future__ import annotations

import json
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


class VoskSpeechRecognizer(SpeechRecognitionPort):
    """小さなモデルで途中結果を素早く返す認識方式。"""

    def __init__(
        self,
        model_path: Path,
        *,
        input_device: int | str | None = None,
        sample_rate: int = 16000,
        silence_threshold: float = 0.015,
        silence_duration: float = 1.2,
        max_recording_duration: float = 30.0,
    ) -> None:
        self._model_path = model_path
        self._input_device = input_device
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._max_recording_duration = max_recording_duration
        self._listening = False
        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._audio_queue: queue.Queue[tuple[bytes, float]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._lock = threading.RLock()

    @property
    def is_available(self) -> bool:
        return self._model_path.is_dir()

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._listening

    @property
    def status_text(self) -> str:
        if self.is_listening:
            return "聞き取り中 (Vosk)"
        if not self._model_path.is_dir():
            return "Voskモデル未配置"
        return "待機中 (Vosk)"

    @property
    def engine_name(self) -> str:
        return "vosk"

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
            if self._listening:
                return
            if not self._model_path.is_dir():
                self._report(on_error, f"音声認識モデルが見つかりません: {self._model_path}")
                return
            self._stop_event.clear()
            self._finish_event.clear()
            self._clear_queue()
            self._listening = True
            self._thread = threading.Thread(
                target=self._recognize,
                args=(on_recognized, on_error, on_partial, on_level),
                name="vosk-speech-recognition",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        stream = self._stream
        if stream is not None:
            with suppress(Exception):
                stream.abort()
                stream.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            self._listening = False
            self._stream = None
            self._thread = None

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
            from vosk import KaldiRecognizer, Model

            recognizer = KaldiRecognizer(Model(str(self._model_path)), self._sample_rate)

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
            self._capture(recognizer, on_recognized, on_error, on_partial)
        except Exception as exc:
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

    def _capture(
        self,
        recognizer: Any,
        on_recognized: RecognitionCallback,
        on_error: IntegrationErrorCallback | None,
        on_partial: RecognitionCallback | None,
    ) -> None:
        spoken = False
        silence = 0.0
        elapsed = 0.0
        final_parts: list[str] = []
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
                elapsed += len(data) / 2 / self._sample_rate
                if level >= self._silence_threshold:
                    spoken = True
                    silence = 0.0
                elif spoken:
                    silence += block_seconds

                if recognizer.AcceptWaveform(data):
                    found = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    if found:
                        final_parts.append(found)
                elif on_partial is not None:
                    partial = str(json.loads(recognizer.PartialResult()).get("partial", "")).strip()
                    on_partial(" ".join([*final_parts, partial]).strip())

                if spoken and silence >= self._silence_duration:
                    write_lip_sync_trace("recognition", "speech_end_detected")
                    break

        if self._stop_event.is_set():
            return
        tail = str(json.loads(recognizer.FinalResult()).get("text", "")).strip()
        if tail:
            final_parts.append(tail)
        text = " ".join(final_parts).strip()
        if text:
            on_recognized(text)
        elif not self._stop_event.is_set():
            self._report(on_error, "音声を認識できませんでした。マイク音量を確認してください。")

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
        return "Vosk音声認識を開始できませんでした。モデル設定を確認してください。"
