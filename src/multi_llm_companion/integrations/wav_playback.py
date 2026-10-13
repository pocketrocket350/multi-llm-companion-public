"""WAV音声をメモリから再生し、現在位置を口パクへ渡す。"""

from __future__ import annotations

import math
import sys
import threading
import wave
from array import array
from contextlib import suppress
from ctypes import (
    Structure,
    Union,
    byref,
    c_char_p,
    c_uint,
    c_ushort,
    c_void_p,
    cast,
    create_string_buffer,
    sizeof,
)
from io import BytesIO
from typing import Any, Protocol
from typing import cast as type_cast

from .lip_sync_diagnostic import is_lip_sync_diagnostic_enabled, write_lip_sync_trace


class WavPlaybackPort(Protocol):
    """VoiSona出力が利用する、再生位置を取得可能なWAV再生口。"""

    @property
    def is_playing(self) -> bool: ...

    @property
    def position_seconds(self) -> float: ...

    @property
    def duration_seconds(self) -> float: ...

    @property
    def lead_in_seconds(self) -> float: ...

    @property
    def amplitude(self) -> float: ...

    def load(self, wav_data: bytes) -> None: ...

    def play(self) -> None: ...

    def stop(self) -> None: ...


class SoundDeviceWavPlayer:
    """sounddeviceでPCM WAVを再生し、実際の出力機器時刻を保持する。"""

    def __init__(self) -> None:
        self._pcm = b""
        self._channels = 0
        self._sample_width = 0
        self._sample_rate = 0
        self._frame_count = 0
        self._current_frame = 0
        self._output_start_time: float | None = None
        self._is_playing = False
        self._stream: Any | None = None
        self._lock = threading.RLock()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    @property
    def position_seconds(self) -> float:
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            stream = self._stream
            output_start_time = self._output_start_time
            duration = self._frame_count / self._sample_rate
        if stream is None or output_start_time is None:
            return 0.0
        try:
            stream_time = float(stream.time)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        return min(duration, max(0.0, stream_time - output_start_time))

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return self._frame_count / self._sample_rate

    @property
    def lead_in_seconds(self) -> float:
        """sounddevice再生では先頭無音を追加しない。"""

        return 0.0

    @property
    def amplitude(self) -> float:
        """現在位置付近20ミリ秒の音量を0から1で返す。"""

        with self._lock:
            if self._sample_rate <= 0 or self._sample_width <= 0 or not self._pcm:
                return 0.0
            frames = max(1, round(self._sample_rate * 0.02))
            played_frame = round(self.position_seconds * self._sample_rate)
            start_frame = max(0, played_frame - frames)
            bytes_per_frame = self._channels * self._sample_width
            data = self._pcm[start_frame * bytes_per_frame : played_frame * bytes_per_frame]
            sample_width = self._sample_width
        return _normalized_pcm_level(data, sample_width)

    def load(self, wav_data: bytes) -> None:
        """PCM WAVを検証してメモリへ読み込む。"""

        self.stop()
        with wave.open(BytesIO(wav_data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("圧縮WAVには対応していません。")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            pcm = wav_file.readframes(frame_count)
        if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
            raise ValueError("WAV音声が空です。")
        if sample_width not in {1, 2, 3, 4}:
            raise ValueError("未対応のWAVビット数です。")
        with self._lock:
            self._pcm = pcm
            self._channels = channels
            self._sample_width = sample_width
            self._sample_rate = sample_rate
            self._frame_count = frame_count
            self._current_frame = 0
            self._output_start_time = None

    def play(self) -> None:
        """既定のWindows出力機器で、読み込んだWAVを再生する。"""

        with self._lock:
            if not self._pcm or self._sample_rate <= 0:
                raise ValueError("再生するWAV音声がありません。")
            channels = self._channels
            sample_rate = self._sample_rate
            dtype = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}[self._sample_width]
            self._current_frame = 0
            self._output_start_time = None
            self._is_playing = True
        try:
            import sounddevice

            stream = sounddevice.RawOutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                latency="low",
                callback=self._audio_callback,
                finished_callback=self._playback_finished,
            )
            with self._lock:
                self._stream = stream
            stream.start()
        except Exception as error:
            with self._lock:
                self._is_playing = False
                self._stream = None
            raise RuntimeError("音声出力を開始できません。") from error

    def stop(self) -> None:
        """再生を停止し、口パク位置を先頭へ戻す。"""

        with self._lock:
            stream = self._stream
            self._stream = None
            self._is_playing = False
            self._current_frame = 0
            self._output_start_time = None
        if stream is not None:
            with suppress(Exception):
                stream.abort()
            with suppress(Exception):
                stream.close()

    def _audio_callback(
        self,
        outdata: Any,
        frames: int,
        time_info: Any,
        _status: Any,
    ) -> None:
        import sounddevice

        with self._lock:
            bytes_per_frame = self._channels * self._sample_width
            start = self._current_frame
            end = min(self._frame_count, start + frames)
            chunk = self._pcm[start * bytes_per_frame : end * bytes_per_frame]
            self._current_frame = end
            if self._output_start_time is None:
                self._output_start_time = _output_dac_time(time_info)
        outdata[:] = bytes(len(outdata))
        outdata[: len(chunk)] = chunk
        if end >= self._frame_count:
            raise sounddevice.CallbackStop

    def _playback_finished(self) -> None:
        with self._lock:
            self._is_playing = False


class _WaveFormatEx(Structure):
    """Windows waveOut API用のPCM形式。"""

    _fields_ = [
        ("wFormatTag", c_ushort),
        ("nChannels", c_ushort),
        ("nSamplesPerSec", c_uint),
        ("nAvgBytesPerSec", c_uint),
        ("nBlockAlign", c_ushort),
        ("wBitsPerSample", c_ushort),
        ("cbSize", c_ushort),
    ]


class _WaveHeader(Structure):
    """Windows waveOut APIへ渡す、再生バッファの情報。"""

    _fields_ = [
        ("lpData", c_char_p),
        ("dwBufferLength", c_uint),
        ("dwBytesRecorded", c_uint),
        ("dwUser", c_void_p),
        ("dwFlags", c_uint),
        ("dwLoops", c_uint),
        ("lpNext", c_void_p),
        ("reserved", c_void_p),
    ]


class _MmTimeValue(Union):
    """MMTIMEの共用領域。SMPTETIMEを含めたWindowsのサイズを保つ。"""

    _fields_ = [("sample", c_uint), ("_reserved", c_uint * 2)]


class _MmTime(Structure):
    """waveOutGetPositionのサンプル位置。"""

    _fields_ = [("wType", c_uint), ("_value", _MmTimeValue)]

    @property
    def sample(self) -> int:
        return int(self._value.sample)

    @sample.setter
    def sample(self, value: int) -> None:
        self._value.sample = value


class _WinMmApi(Protocol):
    """テストでWindows APIを置き換えるための小さな窓口。"""

    def waveOutOpen(self, *args: object) -> int: ...  # noqa: N802

    def waveOutPrepareHeader(self, *args: object) -> int: ...  # noqa: N802

    def waveOutWrite(self, *args: object) -> int: ...  # noqa: N802

    def waveOutGetPosition(self, *args: object) -> int: ...  # noqa: N802

    def waveOutReset(self, *args: object) -> int: ...  # noqa: N802

    def waveOutUnprepareHeader(self, *args: object) -> int: ...  # noqa: N802

    def waveOutClose(self, *args: object) -> int: ...  # noqa: N802


class WinMmWavPlayer:
    """Windows標準のwaveOutで再生し、実再生サンプル位置を返す。

    音声スレッドでPythonコールバックを実行しないため、3D描画とGIL
    （Pythonの同時実行を管理する仕組み）を奪い合わない。
    """

    _WAVE_MAPPER = 0xFFFFFFFF
    _WAVE_FORMAT_PCM = 1
    _MMSYSERR_NOERROR = 0
    _TIME_SAMPLES = 2
    _WHDR_DONE = 0x00000001

    def __init__(
        self,
        *,
        api: _WinMmApi | None = None,
        lead_in_seconds: float = 0.12,
    ) -> None:
        self._pcm = b""
        self._channels = 0
        self._sample_width = 0
        self._sample_rate = 0
        self._frame_count = 0
        self._handle = c_void_p()
        self._header: _WaveHeader | None = None
        self._buffer: Any | None = None
        self._prepared = False
        self._is_playing = False
        self._lock = threading.RLock()
        self._api = api
        self._configured_lead_in_seconds = max(0.0, lead_in_seconds)
        self._lead_in_frames = 0
        self._position_started_traced = False

    @property
    def is_playing(self) -> bool:
        with self._lock:
            header = self._header
            if not self._is_playing or header is None:
                return False
            if int(header.dwFlags) & self._WHDR_DONE:
                self._is_playing = False
                return False
            return True

    @property
    def position_seconds(self) -> float:
        with self._lock:
            if not self._is_playing or not self._handle.value or self._sample_rate <= 0:
                return 0.0
            handle = self._handle
            sample_rate = self._sample_rate
            duration = self.duration_seconds
        position = _MmTime()
        position.wType = self._TIME_SAMPLES
        try:
            result = self._winmm().waveOutGetPosition(handle, byref(position), sizeof(position))
        except (AttributeError, OSError):
            return 0.0
        if result != self._MMSYSERR_NOERROR:
            return 0.0
        seconds = min(duration, max(0.0, int(position.sample) / sample_rate))
        should_trace = False
        with self._lock:
            if seconds > 0.0 and not self._position_started_traced:
                self._position_started_traced = True
                should_trace = True
        if should_trace:
            write_lip_sync_trace(
                "wav_player",
                "playback_position_started",
                position_ms=round(seconds * 1000),
            )
        return seconds

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return self._frame_count / self._sample_rate

    @property
    def lead_in_seconds(self) -> float:
        """Windows音声機器の起動待ちとして追加した無音を返す。"""

        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return self._lead_in_frames / self._sample_rate

    @property
    def amplitude(self) -> float:
        with self._lock:
            if self._sample_rate <= 0 or self._sample_width <= 0 or not self._pcm:
                return 0.0
            frames = max(1, round(self._sample_rate * 0.02))
            played_frame = round(self.position_seconds * self._sample_rate)
            start_frame = max(0, played_frame - frames)
            bytes_per_frame = self._channels * self._sample_width
            data = self._pcm[start_frame * bytes_per_frame : played_frame * bytes_per_frame]
            sample_width = self._sample_width
        return _normalized_pcm_level(data, sample_width)

    def load(self, wav_data: bytes) -> None:
        """PCM WAVを読み込み、Windows APIへ渡す前に検証する。"""

        self.stop()
        with wave.open(BytesIO(wav_data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("圧縮WAVには対応していません。")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            pcm = wav_file.readframes(frame_count)
        if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
            raise ValueError("WAV音声が空です。")
        if sample_width not in {1, 2, 3, 4}:
            raise ValueError("未対応のWAVビット数です。")
        lead_in_frames = round(sample_rate * self._configured_lead_in_seconds)
        silence_sample = b"\x80" if sample_width == 1 else bytes(sample_width)
        silence_frame = silence_sample * channels
        playback_pcm = silence_frame * lead_in_frames + pcm
        if is_lip_sync_diagnostic_enabled():
            first_non_silent_frame = _first_non_silent_frame(
                pcm,
                channels=channels,
                sample_width=sample_width,
                sample_rate=sample_rate,
            )
            write_lip_sync_trace(
                "wav_player",
                "wav_loaded",
                content_duration_ms=round(frame_count / sample_rate * 1000),
                first_non_silent_ms=round(first_non_silent_frame / sample_rate * 1000),
                preroll_ms=round(lead_in_frames / sample_rate * 1000),
            )
        with self._lock:
            self._pcm = playback_pcm
            self._channels = channels
            self._sample_width = sample_width
            self._sample_rate = sample_rate
            self._frame_count = frame_count + lead_in_frames
            self._lead_in_frames = lead_in_frames
            self._position_started_traced = False

    def play(self) -> None:
        """Windows標準の音声出力へWAVを登録し、再生サンプル時計を開始する。"""

        with self._lock:
            if not self._pcm or self._sample_rate <= 0:
                raise ValueError("再生するWAV音声がありません。")
            self._close_device_locked()
            wave_format = _WaveFormatEx(
                self._WAVE_FORMAT_PCM,
                self._channels,
                self._sample_rate,
                self._sample_rate * self._channels * self._sample_width,
                self._channels * self._sample_width,
                self._sample_width * 8,
                0,
            )
            handle = c_void_p()
            api = self._winmm()
            if api.waveOutOpen(byref(handle), self._WAVE_MAPPER, byref(wave_format), 0, 0, 0) != 0:
                raise RuntimeError("音声出力を開始できません。")
            write_lip_sync_trace(
                "wav_player",
                "playback_device_opened",
                preroll_ms=round(self.lead_in_seconds * 1000),
            )
            buffer = create_string_buffer(self._pcm)
            header = _WaveHeader(
                cast(buffer, c_char_p),
                len(self._pcm),
                0,
                None,
                0,
                0,
                None,
                None,
            )
            if api.waveOutPrepareHeader(handle, byref(header), sizeof(header)) != 0:
                api.waveOutClose(handle)
                raise RuntimeError("音声出力を準備できません。")
            if api.waveOutWrite(handle, byref(header), sizeof(header)) != 0:
                api.waveOutUnprepareHeader(handle, byref(header), sizeof(header))
                api.waveOutClose(handle)
                raise RuntimeError("音声出力を開始できません。")
            write_lip_sync_trace(
                "wav_player",
                "playback_buffer_submitted",
                preroll_ms=round(self.lead_in_seconds * 1000),
                playback_duration_ms=round(self.duration_seconds * 1000),
            )
            self._handle = handle
            self._buffer = buffer
            self._header = header
            self._prepared = True
            self._is_playing = True
            self._position_started_traced = False

    def stop(self) -> None:
        """再生を止めてWindowsの音声資源を解放する。"""

        with self._lock:
            self._close_device_locked()

    def _close_device_locked(self) -> None:
        handle = self._handle
        header = self._header
        if handle.value:
            api = self._winmm()
            with suppress(AttributeError, OSError):
                api.waveOutReset(handle)
            if self._prepared and header is not None:
                with suppress(AttributeError, OSError):
                    api.waveOutUnprepareHeader(handle, byref(header), sizeof(header))
            with suppress(AttributeError, OSError):
                api.waveOutClose(handle)
        self._handle = c_void_p()
        self._header = None
        self._buffer = None
        self._prepared = False
        self._is_playing = False

    def _winmm(self) -> _WinMmApi:
        if self._api is not None:
            return self._api
        if sys.platform != "win32":
            raise RuntimeError("Windowsの音声出力に対応していません。")
        import ctypes

        raw_api = ctypes.WinDLL("winmm")
        # ctypesの既定値は64ビットのハンドルを正しく渡せない場合がある。
        # Windows APIの引数型を明示し、再生位置の取得を安定させる。
        raw_api.waveOutOpen.argtypes = [
            ctypes.POINTER(c_void_p),
            c_uint,
            ctypes.POINTER(_WaveFormatEx),
            c_void_p,
            c_void_p,
            c_uint,
        ]
        raw_api.waveOutOpen.restype = c_uint
        raw_api.waveOutPrepareHeader.argtypes = [c_void_p, ctypes.POINTER(_WaveHeader), c_uint]
        raw_api.waveOutPrepareHeader.restype = c_uint
        raw_api.waveOutWrite.argtypes = [c_void_p, ctypes.POINTER(_WaveHeader), c_uint]
        raw_api.waveOutWrite.restype = c_uint
        raw_api.waveOutGetPosition.argtypes = [c_void_p, ctypes.POINTER(_MmTime), c_uint]
        raw_api.waveOutGetPosition.restype = c_uint
        raw_api.waveOutReset.argtypes = [c_void_p]
        raw_api.waveOutReset.restype = c_uint
        raw_api.waveOutUnprepareHeader.argtypes = [c_void_p, ctypes.POINTER(_WaveHeader), c_uint]
        raw_api.waveOutUnprepareHeader.restype = c_uint
        raw_api.waveOutClose.argtypes = [c_void_p]
        raw_api.waveOutClose.restype = c_uint
        api = type_cast(_WinMmApi, raw_api)
        self._api = api
        return api


def create_wav_player(*, lead_in_seconds: float = 0.12) -> WavPlaybackPort:
    """Windowsではネイティブ再生を使い、それ以外は従来実装へ戻す。"""

    if sys.platform == "win32":
        return WinMmWavPlayer(lead_in_seconds=lead_in_seconds)
    return SoundDeviceWavPlayer()


def _normalized_pcm_level(data: bytes, sample_width: int) -> float:
    """PCMのRMS音量を0から1へ正規化する。"""

    if not data:
        return 0.0
    if sample_width == 1:
        samples = (value - 128 for value in data)
        maximum = 128.0
    elif sample_width == 2:
        values = array("h")
        values.frombytes(data[: len(data) - len(data) % 2])
        if sys.byteorder != "little":
            values.byteswap()
        samples = iter(values)
        maximum = 32768.0
    else:
        return 1.0
    squares = [float(value) * float(value) for value in samples]
    if not squares:
        return 0.0
    return min(1.0, math.sqrt(sum(squares) / len(squares)) / maximum)


def _first_non_silent_frame(
    pcm: bytes,
    *,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> int:
    """診断用に、音量が現れる最初の短い区間を探す。"""

    bytes_per_frame = channels * sample_width
    chunk_frames = max(1, round(sample_rate * 0.005))
    frame_count = len(pcm) // bytes_per_frame
    for start_frame in range(0, frame_count, chunk_frames):
        end_frame = min(frame_count, start_frame + chunk_frames)
        chunk = pcm[start_frame * bytes_per_frame : end_frame * bytes_per_frame]
        if _normalized_pcm_level(chunk, sample_width) >= 0.002:
            return start_frame
    return frame_count


def _output_dac_time(time_info: Any) -> float | None:
    """最初のサンプルが実際に出力される機器時刻を取得する。"""

    try:
        return float(time_info.outputBufferDacTime)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return float(time_info["outputBufferDacTime"])
    except (KeyError, TypeError, ValueError):
        return None
