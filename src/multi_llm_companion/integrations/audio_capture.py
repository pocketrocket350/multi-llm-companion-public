"""マイク入力で共通利用する小さな処理。"""

from __future__ import annotations

import math
from array import array

from .base import AudioInputDevice


def input_devices() -> list[AudioInputDevice]:
    """sounddeviceから入力可能な機器だけを取得する。"""

    try:
        import sounddevice

        devices = sounddevice.query_devices()
    except Exception:
        return []
    return [
        AudioInputDevice(index, str(device["name"]))
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


def normalized_level(data: bytes) -> float:
    """16ビット音声の大きさを0から1へ変換する。"""

    samples = array("h")
    samples.frombytes(data)
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return min(1.0, math.sqrt(mean_square) / 32768.0)


def meter_value(level: float) -> int:
    """小さな声も見えるよう、音量を0から100へ拡大する。"""

    return min(100, round(level * 500))
