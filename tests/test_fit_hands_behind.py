"""専用VMD補正が左右の腕以外を変更しないことを確認する。"""

import importlib.util
import math
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fit_hands_behind",
    Path(__file__).resolve().parents[1] / "tools/motion-converter/fit_hands_behind.py",
)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def test_only_arm_quaternions_change_and_tail_is_preserved() -> None:
    original = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"model".ljust(20, b"\0"))
    original.extend(struct.pack("<I", 3))
    for name in ("左腕", "右腕", "左手首"):
        original.extend(name.encode("cp932").ljust(15, b"\0"))
        original.extend(struct.pack("<I3f4f", 123, 1, 2, 3, 0, 0, 0, 1))
        original.extend(bytes(range(64)))
    original.extend(b"preserve morph camera light shadow ik bytes")
    result, offsets = fit.fitted_bytes(bytes(original))
    assert offsets == [85, 196]
    assert len(result) == len(original)
    restored = bytearray(result)
    for offset in offsets:
        q = struct.unpack_from("<4f", result, offset)
        assert q == pytest.approx((math.sin(math.radians(-4)), 0, 0, math.cos(math.radians(-4))))
        restored[offset : offset + 16] = original[offset : offset + 16]
    assert restored == original


def test_invalid_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit.fitted_bytes(b"not vmd")
