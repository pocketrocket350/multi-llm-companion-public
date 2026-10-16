"""VoiSona Talk REST APIによる音声出力。"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import httpx

from multi_llm_companion.domain import Emotion, ExpressionChannel, ExpressionOutputStatus

from .base import VoiceOutputPort
from .lip_sync import JapaneseLipSyncSchedule, LipSyncState
from .lip_sync_diagnostic import write_lip_sync_trace
from .wav_playback import WavPlaybackPort, create_wav_player


@dataclass(frozen=True, slots=True)
class _StreamSegment:
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class _PreparedStreamSegment:
    index: int
    text: str
    wav_data: bytes
    result: Any
    request_uuid: str


class _VoiceSelectionError(ValueError):
    """ボイス選択の失敗を、固定の案内だけで画面へ渡す。"""


class VoiSonaTalkVoiceOutput(VoiceOutputPort):
    """VoiSona TalkのWAVを取得し、アプリ管理の時計で再生する。"""

    _STREAM_WORKER_COUNT = 2

    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        voice_name: str = "",
        language: str = "ja_JP",
        timeout: float = 15.0,
        monitor_interval: float = 0.05,
        monitor_timeout: float = 180.0,
        playback_preroll: float = 0.12,
        lip_sync_start_delay: float = 0.0,
        lip_sync_character_duration: float = 0.22,
        lip_sync_timing_mode: str = "auto",
        expression_enabled: bool = False,
        max_speed_change: float = 0.12,
        max_intonation_change: float = 0.30,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.BaseTransport | None = None,
        wav_player: WavPlaybackPort | None = None,
        temp_root: Path | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._voice_name = voice_name.strip()
        self._language = language
        self._client = httpx.Client(
            auth=httpx.BasicAuth(username, password),
            timeout=timeout,
            transport=transport,
        )
        self._last_error: str | None = None
        self._last_uuid: str | None = None
        self._is_speaking = False
        self._monitor_generation = 0
        self._monitor_interval = max(0.05, monitor_interval)
        self._monitor_timeout = max(1.0, monitor_timeout)
        self._lip_sync_start_delay = max(0.0, lip_sync_start_delay)
        self._lip_sync_character_duration = max(0.04, lip_sync_character_duration)
        self._lip_sync_timing_mode = lip_sync_timing_mode.lower()
        self._expression_enabled = expression_enabled
        self._max_speed_change = max(0.0, min(0.25, max_speed_change))
        self._max_intonation_change = max(0.0, min(0.50, max_intonation_change))
        self._last_expression_text = "無効（従来動作）" if not expression_enabled else "待機中"
        self._clock = clock
        self._wav_player = wav_player or create_wav_player(lead_in_seconds=playback_preroll)
        self._temp_root = temp_root
        self._pending_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._pending_wav_path: Path | None = None
        self._lip_sync_schedule: JapaneseLipSyncSchedule | None = None
        self._playback_started = False
        self._pending_text = ""
        self._lock = threading.RLock()
        # 文単位先読み用の経路。従来の一括発話状態とは分離する。
        self._stream_generation = 0
        self._stream_expression: ExpressionChannel | None = None
        self._stream_input: Queue[_StreamSegment | None] | None = None
        self._stream_output: Queue[_PreparedStreamSegment | None] | None = None
        self._stream_stop_event: threading.Event | None = None
        self._stream_closed = False
        self._stream_active = False
        self._stream_producer: threading.Thread | None = None
        self._stream_consumer: threading.Thread | None = None
        self._stream_active_requests: set[str] = set()
        self._stream_lip_sync_schedule: JapaneseLipSyncSchedule | None = None
        self._stream_playback_started = False
        self._stream_next_index = 0
        self._is_closed = False
        self._client_users = 0
        self._cleanup_queue: Queue[
            tuple[tuple[str, ...], tempfile.TemporaryDirectory[str] | None]
        ] = Queue()
        self._cleanup_thread: threading.Thread | None = None

    @property
    def is_available(self) -> bool:
        return bool(self._username and self._password)

    @property
    def status_text(self) -> str:
        if not self.is_available:
            return "環境変数未設定"
        if not self._voice_name:
            return "ボイス名未設定（音声ガイドで設定してください）"
        if self._last_error:
            return "エラー"
        expression = (
            f"話速・抑揚 {self._last_expression_text}"
            if self._expression_enabled
            else "無効（従来動作）"
        )
        return f"設定済み / 音声表現: {expression}"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def playback_status_text(self) -> str:
        if not self.is_available:
            return "停止中"
        if self._last_error:
            return "エラー"
        return "再生・処理中" if self.is_speaking else "停止中"

    @property
    def lip_sync_state(self) -> LipSyncState:
        with self._lock:
            if self._stream_active:
                schedule = self._stream_lip_sync_schedule if self._stream_playback_started else None
            else:
                schedule = (
                    self._lip_sync_schedule
                    if self._is_speaking and self._playback_started
                    else None
                )
        if schedule is None:
            return LipSyncState()
        if not self._wav_player.is_playing:
            return LipSyncState()
        position = (
            self._wav_player.position_seconds
            - self._wav_player.lead_in_seconds
            - self._lip_sync_start_delay
        )
        if position < 0.0:
            return LipSyncState()
        return schedule.state_at(position)

    @property
    def expression_status_text(self) -> str:
        """直近の音声表現の適用状態。ボイス名や秘密値は含めない。"""

        return self._last_expression_text

    def speak(self, text: str) -> None:
        """従来互換の音声合成。表現設定を有効にしていても値を追加しない。"""

        self._speak(text, None)

    def speak_with_expression(self, text: str, expression: ExpressionChannel) -> None:
        """発話意図から作成済みの音声専用強度をVoiSonaへ渡す。"""

        self._speak(text, expression)

    def start_stream(self, text: str) -> None:
        """一文目を即時合成し、後続文を並行して先読みする。"""

        self._start_stream_with_expression(text, None)

    def _set_stream_expression(self, expression: ExpressionChannel | None) -> None:
        """確定した今回の感情を、まだ合成していない文へ適用する。"""

        with self._lock:
            if self._stream_active:
                self._stream_expression = expression

    def _start_stream_with_expression(
        self, text: str, expression: ExpressionChannel | None
    ) -> None:
        with self._lock:
            if self._is_closed:
                return
            self.stop()
            if not text.strip() or not self.is_available:
                return
            self._stream_generation += 1
            generation = self._stream_generation
            self._stream_expression = expression
            stop_event = threading.Event()
            input_queue: Queue[_StreamSegment | None] = Queue()
            output_queue: Queue[_PreparedStreamSegment | None] = Queue(maxsize=2)
            self._stream_input = input_queue
            self._stream_output = output_queue
            self._stream_stop_event = stop_event
            self._stream_closed = False
            self._stream_active = True
            self._stream_lip_sync_schedule = None
            self._stream_playback_started = False
            self._is_speaking = True
            self._last_error = None
            self._stream_next_index = 1
            input_queue.put(_StreamSegment(0, text))
        write_lip_sync_trace("companion", "stream_voice_started")
        producer = threading.Thread(
            target=self._produce_stream_segments,
            args=(generation, input_queue, output_queue, stop_event),
            daemon=True,
            name="voisona-stream-synthesis",
        )
        consumer = threading.Thread(
            target=self._consume_stream_segments,
            args=(generation, output_queue, stop_event),
            daemon=True,
            name="voisona-stream-playback",
        )
        with self._lock:
            if generation != self._stream_generation or stop_event.is_set():
                return
            self._stream_producer = producer
            self._stream_consumer = consumer
        producer.start()
        consumer.start()

    def enqueue_stream(self, text: str) -> None:
        """現在の音声再生と並行して、次の文を合成キューへ追加する。"""

        if not text.strip():
            return
        with self._lock:
            input_queue = self._stream_input
            active = self._stream_active and not self._stream_closed
            if self._last_error:
                return
        if input_queue is None or not active:
            self.start_stream(text)
            return
        with self._lock:
            if (
                input_queue is not self._stream_input
                or not self._stream_active
                or self._stream_closed
            ):
                return
            segment = _StreamSegment(self._stream_next_index, text)
            self._stream_next_index += 1
        input_queue.put(segment)
        write_lip_sync_trace(
            "companion",
            "stream_sentence_enqueued",
            sentence_index=segment.index,
        )

    def finish_stream(self) -> None:
        """ストリーム末尾を通知し、合成済み文を最後まで再生する。"""

        with self._lock:
            input_queue = self._stream_input
            if input_queue is None or self._stream_closed:
                return
            self._stream_closed = True
        for _ in range(self._STREAM_WORKER_COUNT):
            input_queue.put(None)
        write_lip_sync_trace("companion", "stream_voice_finished")

    def _produce_stream_segments(
        self,
        generation: int,
        input_queue: Queue[_StreamSegment | None],
        output_queue: Queue[_PreparedStreamSegment | None],
        stop_event: threading.Event,
    ) -> None:
        """VoiSonaの合成を先行させ、再生側へ順序どおり渡す。"""

        completed_workers: Queue[None] = Queue()
        workers: list[threading.Thread] = []
        try:
            with self._client_operation() as active:
                if not active or stop_event.is_set():
                    return
                voice_name, voice_version = self._resolve_voice()
            if stop_event.is_set():
                return
            for worker_index in range(self._STREAM_WORKER_COUNT):
                worker = threading.Thread(
                    target=self._stream_synthesis_worker,
                    args=(
                        generation,
                        input_queue,
                        output_queue,
                        stop_event,
                        completed_workers,
                        voice_name,
                        voice_version,
                    ),
                    daemon=True,
                    name=f"voisona-stream-synthesis-{worker_index + 1}",
                )
                workers.append(worker)
                worker.start()
            for _ in workers:
                completed_workers.get()
            if not stop_event.is_set():
                self._put_stream_output(output_queue, None, stop_event)
        except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
            with self._lock:
                if generation == self._stream_generation:
                    self._last_error = (
                        str(exc)
                        if isinstance(exc, _VoiceSelectionError)
                        else "VoiSona Talkへ先読み音声を依頼できませんでした。設定を確認してください。"
                    )
            stop_event.set()

    def _stream_synthesis_worker(
        self,
        generation: int,
        input_queue: Queue[_StreamSegment | None],
        output_queue: Queue[_PreparedStreamSegment | None],
        stop_event: threading.Event,
        completed_workers: Queue[None],
        voice_name: str,
        voice_version: str,
    ) -> None:
        """2件までの合成を並列実行し、完了順で出力キューへ渡す。"""

        try:
            while not stop_event.is_set():
                try:
                    segment = input_queue.get(timeout=0.1)
                except Empty:
                    continue
                if segment is None:
                    return
                prepared = self._synthesize_stream_segment(
                    generation,
                    segment,
                    voice_name,
                    voice_version,
                    stop_event,
                )
                if prepared is None:
                    if not stop_event.is_set():
                        stop_event.set()
                    return
                if not self._put_stream_output(output_queue, prepared, stop_event):
                    return
        except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError):
            with self._lock:
                if generation == self._stream_generation:
                    self._last_error = "VoiSona Talkへの読み込み音声を合成できませんでした。設定を確認してください。"
            stop_event.set()
        finally:
            completed_workers.put(None)

    @staticmethod
    def _put_stream_output(
        output_queue: Queue[_PreparedStreamSegment | None],
        value: _PreparedStreamSegment | None,
        stop_event: threading.Event,
    ) -> bool:
        """停止時に、満杯のキューで合成スレッドを取り残さない。"""

        while not stop_event.is_set():
            try:
                output_queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _synthesize_stream_segment(
        self,
        generation: int,
        segment: _StreamSegment,
        voice_name: str,
        voice_version: str,
        stop_event: threading.Event,
    ) -> _PreparedStreamSegment | None:
        with self._client_operation() as active:
            if not active:
                return None
            temp_dir = tempfile.TemporaryDirectory(
                prefix="multi-llm-companion-stream-", dir=self._temp_root
            )
            try:
                return self._synthesize_stream_request(
                    generation,
                    segment,
                    voice_name,
                    voice_version,
                    stop_event,
                    Path(temp_dir.name) / "voisona-output.wav",
                )
            finally:
                self._queue_cleanup((), temp_dir)

    def _synthesize_stream_request(
        self,
        generation: int,
        segment: _StreamSegment,
        voice_name: str,
        voice_version: str,
        stop_event: threading.Event,
        wav_path: Path,
    ) -> _PreparedStreamSegment | None:
        """一文をWAVへ変換し、再生前のメモリへ読み込む。"""

        with self._lock:
            if generation != self._stream_generation or stop_event.is_set():
                return None
            global_parameters = self._expression_parameters(self._stream_expression)
        payload: dict[str, Any] = {
            "text": segment.text,
            "language": self._language,
            "voice_name": voice_name,
            "force_enqueue": True,
            "destination": "file",
            "output_file_path": str(wav_path.resolve()),
            "can_overwrite_file": True,
        }
        if voice_version:
            payload["voice_version"] = voice_version
        if global_parameters is not None:
            payload["global_parameters"] = global_parameters
        write_lip_sync_trace(
            "voice",
            "stream_synthesis_requested",
            sentence_index=segment.index,
        )
        response = self._client.post(f"{self._base_url}/speech-syntheses", json=payload)
        response.raise_for_status()
        result = response.json()
        request_uuid = self._extract_uuid(result)
        if request_uuid is None:
            raise ValueError("VoiSona Talkの受付番号がありません。")
        with self._lock:
            stale = generation != self._stream_generation or stop_event.is_set()
            if not stale:
                self._stream_active_requests.add(request_uuid)
        if stale:
            self._queue_cleanup((request_uuid,))
            return None
        try:
            deadline = time.monotonic() + self._monitor_timeout
            while time.monotonic() < deadline and not stop_event.is_set():
                status_response = self._client.get(
                    f"{self._base_url}/speech-syntheses/{request_uuid}"
                )
                status_response.raise_for_status()
                status = status_response.json()
                completed = self._is_request_completed(status)
                write_lip_sync_trace(
                    "voice",
                    "stream_synthesis_poll",
                    completed=completed,
                    running=self._is_request_running(status),
                    **self._diagnostic_response_summary(status),
                )
                if self._is_request_failed(status):
                    with self._lock:
                        if generation == self._stream_generation:
                            self._last_error = (
                                "VoiSona Talkの音声生成に失敗したため、読み上げを停止しました。"
                            )
                    write_lip_sync_trace(
                        "voice", "stream_synthesis_failed", sentence_index=segment.index
                    )
                    return None
                if completed:
                    wav_data = wav_path.read_bytes()
                    write_lip_sync_trace(
                        "voice",
                        "stream_synthesis_completed",
                        sentence_index=segment.index,
                    )
                    return _PreparedStreamSegment(
                        segment.index,
                        segment.text,
                        wav_data,
                        status,
                        request_uuid,
                    )
                # 初回は即時照会し、未完了時だけ次の照会まで待つ。
                stop_event.wait(self._monitor_interval)
            if not stop_event.is_set():
                with self._lock:
                    if generation == self._stream_generation:
                        self._last_error = "VoiSona Talkの先読み音声が時間内に完了しませんでした。"
            return None
        finally:
            with self._lock:
                # 停止処理へ渡した要求を、合成ワーカーから重ねて削除しない。
                if request_uuid in self._stream_active_requests:
                    self._stream_active_requests.remove(request_uuid)
                    self._queue_cleanup((request_uuid,))

    def _consume_stream_segments(
        self,
        generation: int,
        output_queue: Queue[_PreparedStreamSegment | None],
        stop_event: threading.Event,
    ) -> None:
        """合成済みWAVを連続再生し、文ごとの口パク時計を切り替える。"""

        pending: dict[int, _PreparedStreamSegment] = {}
        next_index = 0
        last_playback_end_at: float | None = None
        stream_finished = False
        try:
            while not stop_event.is_set():
                if next_index not in pending and stream_finished:
                    break
                try:
                    prepared = output_queue.get(timeout=0.1)
                except Empty:
                    continue
                if prepared is None:
                    stream_finished = True
                    continue
                pending[prepared.index] = prepared
                while next_index in pending and not stop_event.is_set():
                    segment = pending.pop(next_index)
                    while self._wav_player.is_playing and not stop_event.is_set():
                        stream_finished = (
                            self._drain_stream_output_queue(output_queue, pending)
                            or stream_finished
                        )
                        time.sleep(min(self._monitor_interval, 0.05))
                    if stop_event.is_set():
                        return
                    playback_gap_ms = None
                    if last_playback_end_at is not None:
                        playback_gap_ms = max(
                            0,
                            round((self._clock() - last_playback_end_at) * 1000),
                        )
                    if not self._play_stream_segment(
                        generation,
                        segment,
                        playback_gap_ms=playback_gap_ms,
                    ):
                        return
                    while self._wav_player.is_playing and not stop_event.is_set():
                        stream_finished = (
                            self._drain_stream_output_queue(output_queue, pending)
                            or stream_finished
                        )
                        time.sleep(min(self._monitor_interval, 0.05))
                    last_playback_end_at = self._clock()
                    next_index += 1
        finally:
            self._finish_stream_sequence(generation, stop_event)

    @staticmethod
    def _drain_stream_output_queue(
        output_queue: Queue[_PreparedStreamSegment | None],
        pending: dict[int, _PreparedStreamSegment],
    ) -> bool:
        """再生中に完了済み文を取り込み、次文開始時の待ち時間を抑える。"""

        stream_finished = False
        while True:
            try:
                prepared = output_queue.get_nowait()
            except Empty:
                break
            if prepared is None:
                stream_finished = True
                continue
            pending[prepared.index] = prepared
        return stream_finished

    def _play_stream_segment(
        self,
        generation: int,
        prepared: _PreparedStreamSegment,
        *,
        playback_gap_ms: int | None = None,
    ) -> bool:
        """一文を現在のWAV再生器へ渡し、音声時計を開始する。"""

        try:
            with self._lock:
                if generation != self._stream_generation or not self._stream_active:
                    return False
                self._wav_player.load(prepared.wav_data)
                content_duration = max(
                    0.04,
                    self._wav_player.duration_seconds - self._wav_player.lead_in_seconds,
                )
                schedule = self._create_lip_sync_schedule(
                    prepared.text,
                    prepared.result,
                    0.0,
                    audio_duration=content_duration,
                )
                self._stream_lip_sync_schedule = schedule
                self._stream_playback_started = True
                self._wav_player.play()
            write_lip_sync_trace(
                "voice",
                "stream_managed_wav_playback_started",
                sentence_index=prepared.index,
                audio_duration_ms=round(content_duration * 1000),
                playback_preroll_ms=round(self._wav_player.lead_in_seconds * 1000),
                playback_gap_ms=playback_gap_ms,
            )
            return True
        except (OSError, ValueError, RuntimeError):
            with self._lock:
                if generation == self._stream_generation:
                    self._last_error = "VoiSona Talkの先読みWAVを再生できませんでした。"
            return False

    def _finish_stream_sequence(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        """先読みキューの終了後に、口形と音声状態を安全に戻す。"""

        with self._lock:
            if generation != self._stream_generation:
                return
            self._stream_active = False
            self._stream_input = None
            self._stream_output = None
            self._stream_stop_event = None
            self._stream_producer = None
            self._stream_consumer = None
            self._stream_lip_sync_schedule = None
            self._stream_playback_started = False
            self._stream_next_index = 0
            self._is_speaking = False
            stop_event.set()
            self._wav_player.stop()
        write_lip_sync_trace("voice", "stream_managed_wav_playback_completed")

    def _speak(self, text: str, expression: ExpressionChannel | None) -> None:
        self._speak_unless_cancelled(text, expression, None)

    def _speak_unless_cancelled(
        self,
        text: str,
        expression: ExpressionChannel | None,
        cancel_event: threading.Event | None,
    ) -> None:
        with self._client_operation() as active:
            if active:
                self._speak_request(text, expression, cancel_event)

    def _speak_request(
        self,
        text: str,
        expression: ExpressionChannel | None,
        cancel_event: threading.Event | None,
    ) -> None:
        if not text.strip() or not self.is_available:
            return
        with self._lock:
            if self._is_closed or (cancel_event is not None and cancel_event.is_set()):
                return
            self.stop()
            generation = self._monitor_generation
            global_parameters = self._expression_parameters(expression)
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            voice_name, voice_version = self._resolve_voice()
            temp_dir = tempfile.TemporaryDirectory(
                prefix="multi-llm-companion-voice-",
                dir=self._temp_root,
            )
            wav_path = Path(temp_dir.name) / "voisona-output.wav"
            payload: dict[str, Any] = {
                "text": text,
                "language": self._language,
                "voice_name": voice_name,
                "force_enqueue": True,
                "destination": "file",
                "output_file_path": str(wav_path.resolve()),
                "can_overwrite_file": True,
            }
            if voice_version:
                payload["voice_version"] = voice_version
            if global_parameters is not None:
                payload["global_parameters"] = global_parameters
            with self._lock:
                if (
                    generation != self._monitor_generation
                    or self._is_closed
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    self._queue_cleanup((), temp_dir)
                    return
            response = self._client.post(f"{self._base_url}/speech-syntheses", json=payload)
            response.raise_for_status()
            result = response.json()
            request_uuid = self._extract_uuid(result)
            if request_uuid is None:
                raise ValueError("VoiSona Talkの受付番号がありません。")
            with self._lock:
                if (
                    generation != self._monitor_generation
                    or self._is_closed
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    self._queue_cleanup((request_uuid,), temp_dir)
                    return
                self._last_uuid = request_uuid
                self._last_error = None
                self._is_speaking = True
                self._pending_temp_dir = temp_dir
                self._pending_wav_path = wav_path
                self._lip_sync_schedule = None
                self._playback_started = False
                self._pending_text = text
            write_lip_sync_trace(
                "provider",
                "request_accepted",
                schedule_estimated=self._lip_sync_schedule_is_estimated(),
                schedule_duration_ms=self._schedule_duration_ms(),
                **self._diagnostic_response_summary(result),
            )
            threading.Thread(
                target=self._monitor_request,
                args=(request_uuid, generation),
                daemon=True,
                name="voisona-playback-monitor",
            ).start()
        except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError) as exc:
            if temp_dir is not None:
                temp_dir.cleanup()
            with self._lock:
                if (
                    generation != self._monitor_generation
                    or self._is_closed
                    or (cancel_event is not None and cancel_event.is_set())
                ):
                    return
                self._is_speaking = False
                self._pending_temp_dir = None
                self._pending_wav_path = None
                self._lip_sync_schedule = None
                self._playback_started = False
                self._pending_text = ""
                self._last_error = (
                    str(exc)
                    if isinstance(exc, _VoiceSelectionError)
                    else "VoiSona Talkへ音声を依頼できませんでした。起動状態、REST API設定、"
                    "ユーザー名、API用パスワードを確認してください。"
                )

    def _expression_parameters(
        self,
        expression: ExpressionChannel | None,
    ) -> dict[str, float] | None:
        """ボイス固有名を使わず、公式REST APIの話速と抑揚だけを作る。"""

        if (
            not self._expression_enabled
            or expression is None
            or expression.status != ExpressionOutputStatus.APPLIED
            or expression.emotion == Emotion.NEUTRAL
            or expression.intensity <= 0.0
        ):
            self._last_expression_text = (
                "無効（従来動作）" if not self._expression_enabled else "通常 0%"
            )
            return None
        speed_direction = {
            Emotion.HAPPY: 1.0,
            Emotion.SAD: -1.0,
            Emotion.ANGRY: 0.35,
            Emotion.EMBARRASSED: -0.20,
            Emotion.SURPRISED: 0.80,
        }.get(expression.emotion, 0.0)
        intonation_direction = {
            Emotion.HAPPY: 1.0,
            Emotion.SAD: -0.70,
            Emotion.ANGRY: 0.55,
            Emotion.EMBARRASSED: -0.25,
            Emotion.SURPRISED: 1.0,
        }.get(expression.emotion, 0.0)
        strength = max(0.0, min(1.0, expression.intensity))
        speed = max(0.85, min(1.15, 1.0 + speed_direction * strength * self._max_speed_change))
        intonation = max(
            0.80,
            min(
                1.20,
                1.0 + intonation_direction * strength * self._max_intonation_change,
            ),
        )
        self._last_expression_text = (
            f"{expression.emotion.value} {round(strength * 100)}%"
            f"（話速 {speed:.3f}・抑揚 {intonation:.3f}）"
        )
        return {"speed": round(speed, 4), "intonation": round(intonation, 4)}

    def stop(self) -> None:
        with self._lock:
            request_uuid = self._last_uuid
            temp_dir = self._pending_temp_dir
            stream_stop_event = self._stream_stop_event
            stream_request_uuids = tuple(self._stream_active_requests)
            had_active_playback = self._playback_started or self._stream_playback_started
            self._last_uuid = None
            self._is_speaking = False
            self._pending_temp_dir = None
            self._pending_wav_path = None
            self._lip_sync_schedule = None
            self._playback_started = False
            self._pending_text = ""
            self._monitor_generation += 1
            self._stream_generation += 1
            self._stream_expression = None
            self._stream_active = False
            self._stream_closed = True
            self._stream_input = None
            self._stream_output = None
            self._stream_stop_event = None
            self._stream_lip_sync_schedule = None
            self._stream_playback_started = False
            self._stream_next_index = 0
            self._stream_active_requests.clear()
            if stream_stop_event is not None:
                stream_stop_event.set()
            self._wav_player.stop()
            request_uuids = tuple(
                uuid for uuid in (request_uuid, *stream_request_uuids) if uuid is not None
            )
            self._queue_cleanup(request_uuids, temp_dir)
        if had_active_playback:
            write_lip_sync_trace("wav_player", "playback_stop_completed")

    def close(self) -> None:
        with self._lock:
            self._is_closed = True
            self.stop()

    @contextmanager
    def _client_operation(self) -> Iterator[bool]:
        """遅着受付の後始末を登録するまで、使用中のHTTP接続を閉じない。"""

        with self._lock:
            active = not self._is_closed
            if active:
                self._client_users += 1
        try:
            yield active
        finally:
            if active:
                with self._lock:
                    self._client_users -= 1
                    self._ensure_cleanup_worker()

    def _queue_cleanup(
        self,
        request_uuids: tuple[str, ...],
        temp_dir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        with self._lock:
            if request_uuids or temp_dir is not None:
                self._cleanup_queue.put((request_uuids, temp_dir))
            self._ensure_cleanup_worker()

    def _ensure_cleanup_worker(self) -> None:
        """ロック保持中に、後始末用の作業を一つだけ起動する。"""

        if self._cleanup_thread is not None:
            return
        if self._cleanup_queue.empty() and not (self._is_closed and self._client_users == 0):
            return
        if self._cleanup_queue.empty() and self._client.is_closed:
            return
        self._cleanup_thread = threading.Thread(
            target=self._run_cleanup, daemon=True, name="voisona-cleanup"
        )
        self._cleanup_thread.start()

    def _run_cleanup(self) -> None:
        completed = False
        try:
            self._drain_cleanup_queue()
            completed = True
        finally:
            with self._lock:
                self._cleanup_thread = None
                if completed or not self._cleanup_queue.empty():
                    self._ensure_cleanup_worker()

    def _drain_cleanup_queue(self) -> None:
        while True:
            with self._lock:
                if self._cleanup_queue.empty():
                    close_client = self._is_closed and self._client_users == 0
                    work = None
                else:
                    work = self._cleanup_queue.get_nowait()
            if work is None:
                if close_client:
                    self._client.close()
                return
            request_uuids, temp_dir = work
            try:
                for request_uuid in request_uuids:
                    try:
                        self._delete_synthesis(request_uuid)
                    except (OSError, RuntimeError):
                        write_lip_sync_trace("provider", "cleanup_error")
            finally:
                if temp_dir is not None:
                    with suppress(OSError):
                        temp_dir.cleanup()
                self._cleanup_queue.task_done()

    def _delete_synthesis(self, request_uuid: str) -> None:
        """終了済みクライアントだけは、削除待ちの非同期処理を安全に終える。"""

        try:
            self._client.delete(f"{self._base_url}/speech-syntheses/{request_uuid}")
        except httpx.HTTPError:
            return
        except RuntimeError:
            with self._lock:
                if self._is_closed:
                    return
            raise

    def _monitor_request(self, request_uuid: str, generation: int) -> None:
        """WAV生成完了を待ち、アプリ管理再生の終了まで監視する。"""

        deadline = time.monotonic() + self._monitor_timeout
        while time.monotonic() < deadline:
            with self._lock:
                if generation != self._monitor_generation or self._last_uuid != request_uuid:
                    return
            try:
                with self._client_operation() as active:
                    if not active:
                        return
                    response = self._client.get(f"{self._base_url}/speech-syntheses/{request_uuid}")
                response.raise_for_status()
                result = response.json()
                completed = self._is_request_completed(result)
            except (httpx.HTTPError, TypeError, ValueError):
                # 一時的な状態取得失敗では、再生中表示を急に止めない。
                write_lip_sync_trace("provider", "poll_error")
            else:
                write_lip_sync_trace(
                    "provider",
                    "api_poll",
                    completed=completed,
                    running=self._is_request_running(result),
                    **self._diagnostic_response_summary(result),
                )
                if self._is_request_failed(result):
                    with self._lock:
                        if generation != self._monitor_generation:
                            return
                        self._last_error = (
                            "VoiSona Talkの音声生成に失敗したため、読み上げを停止しました。"
                        )
                    self._finish_request(request_uuid, generation)
                    return
                if completed:
                    if not self._start_downloaded_audio(request_uuid, generation, result):
                        return
                    while self._wav_player.is_playing:
                        time.sleep(min(self._monitor_interval, 0.05))
                        with self._lock:
                            if generation != self._monitor_generation:
                                return
                    self._finish_request(request_uuid, generation)
                    return
            # 初回は即時照会し、未完了または一時エラー時だけ待つ。
            time.sleep(self._monitor_interval)
        with self._lock:
            if generation != self._monitor_generation or self._last_uuid != request_uuid:
                return
            temp_dir = self._pending_temp_dir
            self._last_uuid = None
            self._is_speaking = False
            self._pending_temp_dir = None
            self._pending_wav_path = None
            self._lip_sync_schedule = None
            self._playback_started = False
            self._pending_text = ""
            self._last_error = "VoiSona Talkの音声生成が時間内に完了しませんでした。"
            self._wav_player.stop()
            self._queue_cleanup((request_uuid,), temp_dir)
        write_lip_sync_trace("provider", "monitor_timeout")

    def _start_downloaded_audio(
        self,
        request_uuid: str,
        generation: int,
        result: Any,
    ) -> bool:
        """生成済みWAVを読み込み、再生位置0秒から口パクを開始する。"""

        with self._lock:
            if generation != self._monitor_generation or self._last_uuid != request_uuid:
                return False
            wav_path = self._pending_wav_path
            temp_dir = self._pending_temp_dir
            # 読み込み中は、この処理が一時ファイルの後始末を引き受ける。
            self._pending_temp_dir = None
            text = self._pending_text
        try:
            if wav_path is None:
                raise OSError("WAV path is missing")
            wav_data = wav_path.read_bytes()
            with self._lock:
                if generation != self._monitor_generation or self._last_uuid != request_uuid:
                    return False
                self._wav_player.load(wav_data)
                content_duration = max(
                    0.04,
                    self._wav_player.duration_seconds - self._wav_player.lead_in_seconds,
                )
                schedule = self._create_lip_sync_schedule(
                    text,
                    result,
                    0.0,
                    audio_duration=content_duration,
                )
                self._lip_sync_schedule = schedule
                self._playback_started = True
                self._pending_temp_dir = None
                self._pending_wav_path = None
                self._wav_player.play()
            write_lip_sync_trace(
                "provider",
                "managed_wav_playback_started",
                schedule_estimated=self._lip_sync_schedule_is_estimated(),
                schedule_duration_ms=self._schedule_duration_ms(),
                audio_duration_ms=round(content_duration * 1000),
                playback_preroll_ms=round(self._wav_player.lead_in_seconds * 1000),
            )
            return True
        except (OSError, ValueError, RuntimeError):
            self._fail_playback(request_uuid, generation)
            return False
        finally:
            if temp_dir is not None:
                with suppress(OSError):
                    temp_dir.cleanup()

    def _finish_request(self, request_uuid: str, generation: int) -> None:
        """アプリ管理再生の終了と同時に口を閉じる。"""

        with self._lock:
            if generation != self._monitor_generation or self._last_uuid != request_uuid:
                return
            temp_dir = self._pending_temp_dir
            self._last_uuid = None
            self._is_speaking = False
            self._pending_temp_dir = None
            self._pending_wav_path = None
            self._lip_sync_schedule = None
            self._playback_started = False
            self._pending_text = ""
            self._wav_player.stop()
            self._queue_cleanup((request_uuid,), temp_dir)
        write_lip_sync_trace("provider", "managed_wav_playback_completed")

    def _fail_playback(self, request_uuid: str, generation: int) -> None:
        """WAV取得または再生失敗を安全な日本語で画面へ渡す。"""

        with self._lock:
            if generation != self._monitor_generation or self._last_uuid != request_uuid:
                return
            self._last_uuid = None
            self._is_speaking = False
            self._pending_temp_dir = None
            self._pending_wav_path = None
            self._lip_sync_schedule = None
            self._playback_started = False
            self._pending_text = ""
            self._last_error = "VoiSona TalkのWAV音声を再生できませんでした。"
            self._wav_player.stop()
            self._queue_cleanup((request_uuid,))

    def _resolve_voice(self) -> tuple[str, str]:
        if not self._voice_name:
            raise _VoiceSelectionError(
                "VoiSona Talkのボイス名が未設定です。音声ガイドに従ってボイス名を設定し、"
                "アプリを再起動してください。"
            )
        response = self._client.get(f"{self._base_url}/voices")
        response.raise_for_status()
        voices = self._voice_items(response.json())
        if not voices:
            raise _VoiceSelectionError(
                "VoiSona Talkで利用できるボイスがありません。ボイスの導入とAPI設定を確認してください。"
            )
        selected = next(
            (item for item in voices if str(item.get("voice_name", "")) == self._voice_name),
            None,
        )
        if selected is None:
            raise _VoiceSelectionError(
                "VoiSona Talkに指定したボイス名が見つかりません。音声ガイドで利用可能な名前を"
                "確認し、設定後にアプリを再起動してください。別のボイスは使用しません。"
            )
        name = str(selected.get("voice_name", "")).strip()
        if not name:
            raise ValueError("invalid voice")
        return name, str(selected.get("voice_version", "")).strip()

    def _create_lip_sync_schedule(
        self,
        text: str,
        result: Any,
        started_at: float,
        *,
        audio_duration: float | None = None,
    ) -> JapaneseLipSyncSchedule:
        """APIが返す発音時刻を優先し、なければ従来の安全な推定へ戻す。"""

        if self._lip_sync_timing_mode == "auto":
            schedule = JapaneseLipSyncSchedule.from_phoneme_timings(
                self._phoneme_timings(result),
                start_at=started_at,
            )
            if schedule is not None:
                return schedule
        return JapaneseLipSyncSchedule.from_text(
            text,
            start_at=started_at,
            character_duration=self._lip_sync_character_duration,
            total_duration=audio_duration or self._audio_duration(result),
        )

    def _update_lip_sync_timing(
        self,
        request_uuid: str,
        generation: int,
        result: Any,
    ) -> None:
        """処理中に届いた時刻情報で、開始済みの予定表を更新する。"""

        if self._lip_sync_timing_mode != "auto":
            return
        timings = self._phoneme_timings(result)
        if not timings:
            return
        with self._lock:
            if generation != self._monitor_generation or self._last_uuid != request_uuid:
                return
            schedule = self._lip_sync_schedule
            start_at = schedule.start_at if schedule is not None else self._clock()
            accurate = JapaneseLipSyncSchedule.from_phoneme_timings(timings, start_at=start_at)
            if accurate is not None:
                self._lip_sync_schedule = accurate
                write_lip_sync_trace(
                    "provider",
                    "phoneme_schedule_applied",
                    phoneme_count=len(timings),
                    schedule_duration_ms=self._schedule_duration_ms(),
                )

    def _lip_sync_schedule_is_estimated(self) -> bool | None:
        """予定表の由来だけを返す。本文や音素値は記録しない。"""

        with self._lock:
            schedule = self._lip_sync_schedule
            if schedule is None:
                return None
            return schedule.state_at(schedule.start_at).is_estimated

    def _schedule_duration_ms(self) -> int | None:
        """予定表の総時間だけを、整数ミリ秒で返す。"""

        with self._lock:
            schedule = self._lip_sync_schedule
            if schedule is None:
                return None
            return round((schedule.end_at - schedule.start_at) * 1000)

    @classmethod
    def _diagnostic_response_summary(cls, value: Any) -> dict[str, bool | int | str | None]:
        """API応答の秘密値を出さず、同期判定に必要な項目だけを分類する。"""

        if not isinstance(value, dict):
            return {"response_kind": "non_mapping"}
        timings = cls._phoneme_timings(value)
        return {
            "response_kind": "mapping",
            "state": _safe_api_state(value.get("state")),
            "status": _safe_api_state(value.get("status")),
            "has_progress": isinstance(value.get("progress_percentage"), (int, float)),
            "progress_percent": _safe_progress(value.get("progress_percentage")),
            "phoneme_count": len(timings),
            "has_duration": cls._audio_duration(value) is not None,
        }

    @staticmethod
    def _voice_items(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("voices", "items"):
                items = value.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_uuid(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("uuid", "id"):
                found = value.get(key)
                if isinstance(found, str):
                    return found
        return None

    @staticmethod
    def _audio_duration(value: Any) -> float | None:
        """APIごとの差を吸収して、音声全体の秒数を取り出す。"""

        if not isinstance(value, dict):
            return None
        for item in _nested_mappings(value):
            for key in ("duration", "duration_seconds", "audio_duration", "length_seconds"):
                raw = item.get(key)
                if isinstance(raw, (int, float)) and 0.04 <= float(raw) <= 600.0:
                    return float(raw)
        return None

    @staticmethod
    def _phoneme_timings(value: Any) -> tuple[tuple[float, float, str], ...]:
        """複数のAPI表記から、相対秒の音素区間だけを安全に取り出す。"""

        for item in _nested_mappings(value):
            for key in ("phoneme_timings", "phonemes", "phoneme_timing", "timings"):
                raw_timings = item.get(key)
                if not isinstance(raw_timings, list):
                    continue
                parsed = tuple(
                    timing
                    for raw in raw_timings
                    if (timing := _parse_phoneme_timing(raw)) is not None
                )
                if parsed:
                    return parsed
        return ()

    @staticmethod
    def _is_request_failed(value: Any) -> bool:
        """明示された失敗・取消は、合成中として待ち続けない。"""

        return isinstance(value, dict) and any(
            isinstance(value.get(key), str)
            and value[key].lower() in {"failed", "error", "cancelled", "canceled"}
            for key in ("state", "status")
        )

    @staticmethod
    def _is_request_completed(value: Any) -> bool:
        """APIバージョン差を吸収して、完了状態だけを読み取る。"""

        if not isinstance(value, dict):
            return False
        for key in ("completed", "is_completed", "finished", "done"):
            if value.get(key) is True:
                return True
        status = value.get("status")
        if isinstance(status, bool):
            return status
        if isinstance(status, str):
            return status.lower() in {"completed", "succeeded", "success", "done", "finished"}
        state = value.get("state")
        return isinstance(state, str) and state.lower() in {
            "completed",
            "succeeded",
            "success",
            "done",
            "finished",
        }

    @staticmethod
    def _is_request_running(value: Any) -> bool:
        """進捗率がないAPI応答で、処理・再生中だけを識別する。"""

        if not isinstance(value, dict):
            return False
        for key in ("state", "status"):
            state = value.get(key)
            if isinstance(state, str) and state.lower() in {"running", "processing", "playing"}:
                return True
        return False


def _nested_mappings(value: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """API応答のよくある入れ子を、秘密値を出さずに走査する。"""

    mappings = [value]
    for key in ("result", "data", "speech", "audio"):
        nested = value.get(key)
        if isinstance(nested, dict):
            mappings.append(nested)
    return tuple(mappings)


def _parse_phoneme_timing(value: Any) -> tuple[float, float, str] | None:
    """1音素の開始・終了時刻を読み取る。ミリ秒表記も秒へ直す。"""

    if not isinstance(value, dict):
        return None
    phoneme = next(
        (
            value.get(key)
            for key in ("phoneme", "phone", "symbol", "label")
            if isinstance(value.get(key), str)
        ),
        None,
    )
    start = next(
        (
            value.get(key)
            for key in ("start", "start_time", "offset", "begin")
            if isinstance(value.get(key), (int, float))
        ),
        None,
    )
    end = next(
        (
            value.get(key)
            for key in ("end", "end_time", "finish")
            if isinstance(value.get(key), (int, float))
        ),
        None,
    )
    duration = next(
        (
            value.get(key)
            for key in ("duration", "length")
            if isinstance(value.get(key), (int, float))
        ),
        None,
    )
    if not isinstance(phoneme, str) or not isinstance(start, (int, float)):
        return None
    start_seconds = float(start)
    if start_seconds > 600.0:
        start_seconds /= 1000.0
    if isinstance(end, (int, float)):
        end_seconds = float(end)
        if end_seconds > 600.0:
            end_seconds /= 1000.0
    elif isinstance(duration, (int, float)):
        duration_seconds = float(duration)
        if duration_seconds > 600.0:
            duration_seconds /= 1000.0
        end_seconds = start_seconds + duration_seconds
    else:
        return None
    return (start_seconds, end_seconds, phoneme)


def _safe_api_state(value: Any) -> str:
    """状態欄に使う短い定型値だけを診断記録へ残す。"""

    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, str):
        return "non_text"
    normalized = value.strip().lower()
    safe_states = {
        "accepted",
        "cancelled",
        "canceled",
        "completed",
        "done",
        "error",
        "failed",
        "finished",
        "pending",
        "playing",
        "processing",
        "queued",
        "running",
        "started",
        "succeeded",
        "success",
        "waiting",
    }
    return normalized if normalized in safe_states else "unrecognized_text"


def _safe_progress(value: Any) -> int | None:
    """進捗が数値なら0から100の範囲へ丸めて記録する。"""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0, min(100, round(float(value))))
