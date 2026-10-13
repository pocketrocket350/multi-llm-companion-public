"""猫背補正で、明示した回転以外の原本データを保持する。"""

import importlib.util
import math
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fit_catback", Path(__file__).resolve().parents[1] / "tools/motion-converter/fit_catback.py"
)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


@pytest.mark.parametrize("arm_degrees", [5.0, 15.0])
def test_only_arm_rotations_change(arm_degrees: float) -> None:
    original = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"model".ljust(20, b"\0"))
    original.extend(struct.pack("<I", 3))
    for name in ("左腕", "右腕", "左手首"):
        original.extend(name.encode("cp932").ljust(15, b"\0"))
        original.extend(struct.pack("<I3f4f", 123, 1, 2, 3, 0, 0, 0, 1))
        original.extend(bytes(range(64)))
    original.extend(b"preserve morph camera light shadow ik bytes")
    result, offsets = fit.fitted_bytes(bytes(original), arm_degrees=arm_degrees)
    assert offsets == [85, 196]
    restored = bytearray(result)
    for offset, angle in zip(offsets, (arm_degrees / 2, -arm_degrees / 2), strict=True):
        assert struct.unpack_from("<4f", result, offset) == pytest.approx(
            (0, 0, math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        )
        restored[offset : offset + 16] = original[offset : offset + 16]
    assert restored == original


@pytest.mark.parametrize("source", [b"not vmd", b"Vocaloid Motion Data 0002".ljust(30, b"\0")])
def test_invalid_source_is_rejected(source: bytes) -> None:
    with pytest.raises(ValueError):
        fit.fitted_bytes(source)


def test_waist_clearance_changes_only_right_arm_quaternions() -> None:
    original = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"model".ljust(20, b"\0"))
    original.extend(struct.pack("<I", 4))
    for name in ("左腕", "右腕", "右腕", "右ひじ"):
        original.extend(name.encode("cp932").ljust(15, b"\0"))
        original.extend(struct.pack("<I3f4f", 480, 1, 2, 3, 0, 0, 0, 1))
        original.extend(bytes(range(64)))
    original.extend(b"preserve original morphs and trailing bytes")
    fitted, offsets = fit.right_arm_clearance_bytes(bytes(original))
    assert offsets == [196, 307]
    restored = bytearray(fitted)
    for offset in offsets:
        assert struct.unpack_from("<4f", fitted, offset) == pytest.approx(
            (0, 0, math.sin(math.radians(-4)), math.cos(math.radians(-4)))
        )
        restored[offset : offset + 16] = original[offset : offset + 16]
    assert restored == original


def test_reference_pose_changes_only_approved_rotations_without_extra_holds() -> None:
    original = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"model".ljust(20, b"\0"))
    names = ("左腕", "右腕", "上半身2", "頭", "首", "上半身")
    original.extend(struct.pack("<I", len(names) * 2))
    for name in names:
        for frame in (0, 600):
            original.extend(name.encode("cp932").ljust(15, b"\0"))
            original.extend(struct.pack("<I3f4f", frame, 1, 2, 3, 0, 0, 0, 1))
            original.extend(bytes(range(64)))
    original.extend(b"preserve original morphs and trailing bytes")
    result = fit.reference_pose_bytes(bytes(original))
    restored = bytearray(result)
    for i in range(len(names) * 2):
        name = names[i // 2]
        offset = 54 + i * 111 + 31
        x_angle, z_angle = {
            "左腕": (0, 15),
            "右腕": (0, -15),
            "上半身2": (-10, 0),
            "頭": (10, 0),
        }.get(name, (0, 0))
        assert struct.unpack_from("<4f", result, offset) == pytest.approx(
            (
                math.sin(math.radians(x_angle / 2)),
                0,
                math.sin(math.radians(z_angle / 2)),
                math.cos(math.radians((x_angle + z_angle) / 2)),
            )
        )
        restored[offset : offset + 16] = original[offset : offset + 16]
    assert restored == original


def test_entry_exit_preserves_original_motion_and_morphs(tmp_path: Path) -> None:
    from multi_llm_companion.integrations.vmd_motion import read_vmd_motion

    original = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0") + b"model".ljust(20, b"\0"))
    original.extend(struct.pack("<I", 6))
    for name in ("左腕", "右腕", "上半身"):
        for frame in (0, 30):
            original.extend(name.encode("cp932").ljust(15, b"\0"))
            original.extend(struct.pack("<I3f4f", frame, 1, 2, 3, 0, 0, 0, 1))
            original.extend(bytes(range(64)))
    original.extend(struct.pack("<I", 2))
    for frame, weight in ((10, 0.0), (15, 1.0)):
        original.extend("まばたき".encode("cp932").ljust(15, b"\0"))
        original.extend(struct.pack("<If", frame, weight))
    original.extend(bytes(16))
    before = bytes(original)
    baseline, _ = fit.fitted_bytes(before)
    result = fit.entry_exit_bytes(before)
    baseline_path, result_path = tmp_path / "baseline.vmd", tmp_path / "result.vmd"
    baseline_path.write_bytes(baseline)
    result_path.write_bytes(result)
    old, new = read_vmd_motion(baseline_path), read_vmd_motion(result_path)
    assert bytes(original) == before
    assert new.frame_count == old.frame_count + 120
    indexed = {(f.bone_name, f.frame_number): f for f in new.bone_frames}
    assert len(indexed) == len(new.bone_frames)
    for frame in old.bone_frames:
        shifted = indexed[frame.bone_name, frame.frame_number + 60]
        assert (shifted.translation, shifted.rotation, shifted.interpolation) == (
            frame.translation,
            frame.rotation,
            frame.interpolation,
        )
    for name, sign in (("左腕", 1), ("右腕", -1)):
        for frame in (0, 18, 132, 150):
            assert indexed[name, frame].rotation == pytest.approx(
                (0, 0, math.sin(math.radians(sign * 55 / 2)), math.cos(math.radians(55 / 2)))
            )
    assert indexed["上半身", 0].rotation == indexed["上半身", 150].rotation == (0, 0, 0, 1)
    morphs = {(f.morph_name, f.frame_number): f.weight for f in new.morph_frames}
    for frame in old.morph_frames:
        assert morphs[frame.morph_name, frame.frame_number + 60] == frame.weight
    assert not new.camera_frames and not new.ik_frames
