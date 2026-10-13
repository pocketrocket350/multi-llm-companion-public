"""第28段階のVMDボーンモーション読み取り。"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from .base import (
    AvatarVmdBoneFrame,
    AvatarVmdCameraFrame,
    AvatarVmdIkFrame,
    AvatarVmdIkState,
    AvatarVmdLightFrame,
    AvatarVmdMorphFrame,
    AvatarVmdMotion,
    AvatarVmdShadowFrame,
)

_MAX_FILE_SIZE = 200 * 1024 * 1024
_MAX_BONE_FRAMES = 300_000
_MAX_MORPH_FRAMES = 300_000
_MAX_CAMERA_FRAMES = 100_000
_MAX_LIGHT_FRAMES = 100_000
_MAX_SHADOW_FRAMES = 100_000
_MAX_IK_FRAMES = 100_000
_MAX_IK_STATES = 500_000
_VMD_HEADER = b"Vocaloid Motion Data 0002"
_LINEAR_CURVE = (20, 20, 107, 107)
_LINEAR_INTERPOLATION = (_LINEAR_CURVE,) * 4
_GENTLE_CURVE = (30, 10, 97, 117)
_GENTLE_INTERPOLATION = (_GENTLE_CURVE,) * 4
_IDLE_FRAME_NUMBERS = (0, 45, 90, 135, 180)


class VmdMotionError(RuntimeError):
    """利用者へ安全に示せるVMD読み込みエラー。"""


def read_vmd_motion(path: Path, *, loop: bool = True) -> AvatarVmdMotion:
    """VMDの各キーフレームを上限付きで読み取る。"""

    if path.suffix.lower() != ".vmd":
        raise VmdMotionError("モーションは .vmd ファイルを選んでください。")
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise VmdMotionError("指定されたVMDモーションが見つかりません。") from exc
    except OSError as exc:
        raise VmdMotionError("VMDモーションの情報を読み取れません。") from exc
    if size <= 0 or size > _MAX_FILE_SIZE:
        raise VmdMotionError("VMDモーションのサイズが正しくありません（上限は200MBです）。")
    try:
        with path.open("rb") as stream:
            header = _read_exact(stream, 30)
            if not header.rstrip(b"\0").startswith(_VMD_HEADER):
                raise VmdMotionError("VMD形式ではないため読み込めません。")
            target_model_name = _decode_shift_jis(_read_exact(stream, 20))
            frame_count = _read_uint(stream)
            if frame_count > _MAX_BONE_FRAMES:
                raise VmdMotionError("VMDのボーンフレーム数が大きすぎます。")
            frames: list[AvatarVmdBoneFrame] = []
            for _ in range(frame_count):
                name = _decode_shift_jis(_read_exact(stream, 15))
                frame_number = _read_uint(stream)
                translation = struct.unpack("<3f", _read_exact(stream, 12))
                rotation = struct.unpack("<4f", _read_exact(stream, 16))
                interpolation = _decode_bone_interpolation(_read_exact(stream, 64))
                if not name or not all(math.isfinite(value) for value in (*translation, *rotation)):
                    continue
                frames.append(
                    AvatarVmdBoneFrame(
                        name,
                        frame_number,
                        translation,
                        rotation,
                        interpolation,
                    )
                )
            morph_count = _read_uint(stream)
            if morph_count > _MAX_MORPH_FRAMES:
                raise VmdMotionError("VMDの表情フレーム数が大きすぎます。")
            morph_frames: list[AvatarVmdMorphFrame] = []
            for _ in range(morph_count):
                name = _decode_shift_jis(_read_exact(stream, 15))
                frame_number = _read_uint(stream)
                weight = struct.unpack("<f", _read_exact(stream, 4))[0]
                if name and math.isfinite(weight):
                    morph_frames.append(
                        AvatarVmdMorphFrame(name, frame_number, max(0.0, min(1.0, weight)))
                    )
            camera_frames: list[AvatarVmdCameraFrame] = []
            light_frames: list[AvatarVmdLightFrame] = []
            shadow_frames: list[AvatarVmdShadowFrame] = []
            ik_frames: list[AvatarVmdIkFrame] = []
            camera_count = _read_optional_uint(stream)
            if camera_count is not None:
                if camera_count > _MAX_CAMERA_FRAMES:
                    raise VmdMotionError("VMDのカメラフレーム数が大きすぎます。")
                for _ in range(camera_count):
                    frame_number = _read_uint(stream)
                    distance = struct.unpack("<f", _read_exact(stream, 4))[0]
                    position = struct.unpack("<3f", _read_exact(stream, 12))
                    rotation = struct.unpack("<3f", _read_exact(stream, 12))
                    interpolation = _decode_camera_interpolation(_read_exact(stream, 24))
                    view_angle = _read_uint(stream)
                    perspective = _read_exact(stream, 1)[0] == 0
                    if all(math.isfinite(value) for value in (distance, *position, *rotation)):
                        camera_frames.append(
                            AvatarVmdCameraFrame(
                                frame_number,
                                distance,
                                position,
                                rotation,
                                interpolation,
                                max(1, min(180, view_angle)),
                                perspective,
                            )
                        )
                light_count = _read_uint(stream)
                if light_count > _MAX_LIGHT_FRAMES:
                    raise VmdMotionError("VMDの照明フレーム数が大きすぎます。")
                for _ in range(light_count):
                    frame_number = _read_uint(stream)
                    color = struct.unpack("<3f", _read_exact(stream, 12))
                    position = struct.unpack("<3f", _read_exact(stream, 12))
                    if all(math.isfinite(value) for value in (*color, *position)):
                        light_frames.append(
                            AvatarVmdLightFrame(
                                frame_number,
                                tuple(max(0.0, min(1.0, value)) for value in color),
                                position,
                            )
                        )
                shadow_count = _read_uint(stream)
                if shadow_count > _MAX_SHADOW_FRAMES:
                    raise VmdMotionError("VMDの影フレーム数が大きすぎます。")
                for _ in range(shadow_count):
                    frame_number = _read_uint(stream)
                    mode = _read_exact(stream, 1)[0]
                    distance = struct.unpack("<f", _read_exact(stream, 4))[0]
                    if math.isfinite(distance):
                        shadow_frames.append(
                            AvatarVmdShadowFrame(frame_number, max(0, min(2, mode)), distance)
                        )
                # 古いVMDは影セクションの直後で終わり、IK切替欄を持たない。
                ik_count = _read_optional_uint(stream) or 0
                if ik_count > _MAX_IK_FRAMES:
                    raise VmdMotionError("VMDのIKフレーム数が大きすぎます。")
                ik_state_count = 0
                for _ in range(ik_count):
                    frame_number = _read_uint(stream)
                    display_model = bool(_read_exact(stream, 1)[0])
                    state_count = _read_uint(stream)
                    ik_state_count += state_count
                    if ik_state_count > _MAX_IK_STATES:
                        raise VmdMotionError("VMDのIK切替数が大きすぎます。")
                    states: list[AvatarVmdIkState] = []
                    for _ in range(state_count):
                        name = _decode_shift_jis(_read_exact(stream, 20))
                        enabled = bool(_read_exact(stream, 1)[0])
                        if name:
                            states.append(AvatarVmdIkState(name, enabled))
                    ik_frames.append(AvatarVmdIkFrame(frame_number, display_model, tuple(states)))
    except VmdMotionError:
        raise
    except (OSError, struct.error) as exc:
        raise VmdMotionError("VMDモーションの読み込み中にエラーが発生しました。") from exc
    maximum_frame = max(
        max((frame.frame_number for frame in frames), default=0),
        max((frame.frame_number for frame in morph_frames), default=0),
        max((frame.frame_number for frame in camera_frames), default=0),
        max((frame.frame_number for frame in light_frames), default=0),
        max((frame.frame_number for frame in shadow_frames), default=0),
        max((frame.frame_number for frame in ik_frames), default=0),
    )
    return AvatarVmdMotion(
        path.stem,
        maximum_frame,
        tuple(frames),
        loop,
        tuple(morph_frames),
        target_model_name=target_model_name,
        camera_frames=tuple(camera_frames),
        light_frames=tuple(light_frames),
        shadow_frames=tuple(shadow_frames),
        ik_frames=tuple(ik_frames),
    )


def write_confirmation_vmd(
    path: Path,
    bone_names: tuple[str, ...],
    morph_names: tuple[str, ...] = (),
    target_model_name: str = "multi-llm-companion",
) -> None:
    """モデル内に実在するボーンだけで、短い自然待機VMDを作る。"""

    selected = set(bone_names)
    frames: list[AvatarVmdBoneFrame] = []
    for name in ("センター", "グルーブ", "全ての親"):
        if name in selected:
            frames.extend(_idle_translation_frames(name, (0.0, 0.018, 0.006, 0.024, 0.0)))
            break
    for name in ("上半身", "上半身2"):
        if name in selected:
            frames.extend(_idle_rotation_frames(name, 0, (0.0, -0.045, -0.016, -0.052, 0.0)))
            break
    for name in ("首", "頭"):
        if name in selected:
            frames.extend(_idle_rotation_frames(name, 2, (0.0, 0.035, 0.0, -0.040, 0.0)))
            break
    if not frames:
        raise VmdMotionError(
            "自然待機VMDを作るための「センター」「上半身」「首」などのボーンが見つかりません。"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(_VMD_HEADER.ljust(30, b"\0"))
            stream.write(_encode_name(target_model_name, 20))
            stream.write(struct.pack("<I", len(frames)))
            for frame in sorted(frames, key=lambda item: (item.frame_number, item.bone_name)):
                stream.write(_encode_name(frame.bone_name, 15))
                stream.write(
                    struct.pack(
                        "<I3f4f",
                        frame.frame_number,
                        *frame.translation,
                        *frame.rotation,
                    )
                )
                stream.write(_encode_bone_interpolation(frame.interpolation))
            # まばたき、表情、口形は既存の常時動作とリップシンクへ任せる。
            stream.write(struct.pack("<I", 0))
            stream.write(struct.pack("<I", 0) * 4)  # カメラ、照明、影、IK
    except OSError as exc:
        raise VmdMotionError("確認用VMDを書き出せませんでした。") from exc


def write_listening_idle_vmd(
    path: Path,
    bone_names: tuple[str, ...],
    target_model_name: str = "multi-llm-companion",
) -> None:
    """足を動かさず、聞く姿勢の小さな整え直しだけを持つ待機VMDを作る。"""

    selected = set(bone_names)
    profiles = (
        ("腰", 0, (0.0, 0.003, -0.004, 0.006, 0.002, 0.0)),
        ("上半身", 0, (0.0, -0.012, -0.005, -0.019, -0.009, 0.0)),
        ("上半身2", 0, (0.0, -0.006, 0.002, -0.009, -0.004, 0.0)),
        ("首", 0, (0.0, -0.004, -0.006, -0.008, -0.003, 0.0)),
        ("首", 2, (0.0, 0.008, 0.014, -0.005, -0.006, 0.0)),
        ("頭", 2, (0.0, 0.004, 0.008, -0.003, -0.003, 0.0)),
    )
    frame_numbers = (0, 45, 105, 180, 270, 360)
    frames = [
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, 0.0, 0.0),
            _axis_rotation(axis, angle),
            _GENTLE_INTERPOLATION,
        )
        for name, axis, angles in profiles
        if name in selected
        for frame_number, angle in zip(frame_numbers, angles, strict=True)
    ]
    # 雨衣の基準待機から、腕を下げた穏やかな初期姿勢だけを引き継ぐ。
    # 腕を未所有にするとPMXの基準T字姿勢が見えてしまうため、手首まで固定する。
    arm_posture = {
        "左腕": (-0.00874, 0.07854, -0.28744, 0.95453),
        "左ひじ": (0.03236, 0.11009, 0.0, 0.99339),
        "左手首": (-0.04182, -0.01275, -0.06492, 0.99693),
        "右腕": (-0.00837, -0.07905, 0.29390, 0.95253),
        "右ひじ": (0.03236, -0.11009, 0.0, 0.99339),
        "右手首": (-0.04182, 0.01275, 0.06492, 0.99693),
    }
    frames.extend(
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, 0.0, 0.0),
            rotation,
            _GENTLE_INTERPOLATION,
        )
        for name, rotation in arm_posture.items()
        if name in selected
        for frame_number in frame_numbers
    )
    if not frames:
        raise VmdMotionError("聞く姿勢VMDに必要な上半身・首・頭のボーンが見つかりません。")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(_VMD_HEADER.ljust(30, b"\0"))
            stream.write(_encode_name(target_model_name, 20))
            stream.write(struct.pack("<I", len(frames)))
            for frame in sorted(frames, key=lambda item: (item.frame_number, item.bone_name)):
                stream.write(_encode_name(frame.bone_name, 15))
                stream.write(
                    struct.pack(
                        "<I3f4f",
                        frame.frame_number,
                        *frame.translation,
                        *frame.rotation,
                    )
                )
                stream.write(_encode_bone_interpolation(frame.interpolation))
            # 表情、口形、まばたき、足IKは既存の専用経路へ任せる。
            stream.write(struct.pack("<I", 0))
            stream.write(struct.pack("<I", 0) * 4)
    except OSError as exc:
        raise VmdMotionError("聞く姿勢VMDを書き出せませんでした。") from exc


def write_gentle_sway_idle_vmd(
    path: Path,
    bone_names: tuple[str, ...],
    target_model_name: str = "multi-llm-companion",
) -> None:
    """足を固定し、呼吸に続く小さな重心の整え直しを持つ待機VMDを作る。"""

    selected = set(bone_names)
    frame_numbers = (0, 75, 150, 240, 330, 450)
    profiles = (
        ("腰", 2, (0.0, -0.006, -0.011, -0.004, 0.006, 0.0)),
        ("上半身", 2, (0.0, 0.012, 0.023, 0.010, -0.010, 0.0)),
        ("上半身2", 2, (0.0, 0.009, 0.016, 0.007, -0.008, 0.0)),
        ("上半身2", 0, (0.0, -0.006, -0.012, -0.005, 0.005, 0.0)),
        ("首", 2, (0.0, 0.001, 0.006, 0.003, -0.004, 0.0)),
        ("頭", 2, (0.0, -0.004, -0.008, -0.004, 0.006, 0.0)),
        ("頭", 0, (0.0, 0.006, 0.012, 0.005, -0.005, 0.0)),
    )
    frames = [
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, 0.0, 0.0),
            _axis_rotation(axis, angle),
            _GENTLE_INTERPOLATION,
        )
        for name, axis, angles in profiles
        if name in selected
        for frame_number, angle in zip(frame_numbers, angles, strict=True)
    ]
    arm_posture = {
        "左腕": (-0.00874, 0.07854, -0.28744, 0.95453),
        "左ひじ": (0.03236, 0.11009, 0.0, 0.99339),
        "左手首": (-0.04182, -0.01275, -0.06492, 0.99693),
        "右腕": (-0.00837, -0.07905, 0.29390, 0.95253),
        "右ひじ": (0.03236, -0.11009, 0.0, 0.99339),
        "右手首": (-0.04182, 0.01275, 0.06492, 0.99693),
    }
    frames.extend(
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, 0.0, 0.0),
            rotation,
            _GENTLE_INTERPOLATION,
        )
        for name, rotation in arm_posture.items()
        if name in selected
        for frame_number in frame_numbers
    )
    if not frames:
        raise VmdMotionError("ゆるやかな待機VMDに必要な上半身・首・頭のボーンが見つかりません。")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(_VMD_HEADER.ljust(30, b"\0"))
            stream.write(_encode_name(target_model_name, 20))
            stream.write(struct.pack("<I", len(frames)))
            for frame in sorted(frames, key=lambda item: (item.frame_number, item.bone_name)):
                stream.write(_encode_name(frame.bone_name, 15))
                stream.write(
                    struct.pack(
                        "<I3f4f",
                        frame.frame_number,
                        *frame.translation,
                        *frame.rotation,
                    )
                )
                stream.write(_encode_bone_interpolation(frame.interpolation))
            # 表情、口形、まばたき、足IKは既存の専用経路へ任せる。
            stream.write(struct.pack("<I", 0))
            stream.write(struct.pack("<I", 0) * 4)
    except OSError as exc:
        raise VmdMotionError("ゆるやかな待機VMDを書き出せませんでした。") from exc


def _idle_translation_frames(
    name: str,
    heights: tuple[float, ...],
) -> tuple[AvatarVmdBoneFrame, ...]:
    return tuple(
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, height, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            _GENTLE_INTERPOLATION,
        )
        for frame_number, height in zip(_IDLE_FRAME_NUMBERS, heights, strict=True)
    )


def _idle_rotation_frames(
    name: str,
    axis: int,
    angles: tuple[float, ...],
) -> tuple[AvatarVmdBoneFrame, ...]:
    return tuple(
        AvatarVmdBoneFrame(
            name,
            frame_number,
            (0.0, 0.0, 0.0),
            _axis_rotation(axis, angle),
            _GENTLE_INTERPOLATION,
        )
        for frame_number, angle in zip(_IDLE_FRAME_NUMBERS, angles, strict=True)
    )


def _axis_rotation(axis: int, angle: float) -> tuple[float, float, float, float]:
    values = [0.0, 0.0, 0.0]
    values[axis] = math.sin(angle / 2.0)
    return (*values, math.cos(angle / 2.0))


def _read_exact(stream: object, size: int) -> bytes:
    data = stream.read(size)  # type: ignore[attr-defined]
    if len(data) != size:
        raise VmdMotionError("VMDモーションの途中でデータが終わっています。")
    return data


def _read_uint(stream: object) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_optional_uint(stream: object) -> int | None:
    data = stream.read(4)  # type: ignore[attr-defined]
    if not data:
        return None
    if len(data) != 4:
        raise VmdMotionError("VMDモーションの途中でデータが終わっています。")
    return struct.unpack("<I", data)[0]


def _decode_shift_jis(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("shift_jis", errors="replace").strip()


def _encode_name(value: str, size: int) -> bytes:
    encoded = value.encode("shift_jis", errors="replace")[:size]
    return encoded.ljust(size, b"\0")


def _decode_bone_interpolation(
    value: bytes,
) -> tuple[tuple[int, int, int, int], ...]:
    """VMD先頭16バイトからXYZ移動・回転のベジェ曲線を取り出す。"""

    if not any(value[:16]):
        # 旧版が生成したゼロ埋めVMDは、従来どおり線形として扱う。
        return _LINEAR_INTERPOLATION
    curves: list[tuple[int, int, int, int]] = []
    for channel in range(4):
        curve = tuple(value[channel + offset] for offset in (0, 4, 8, 12))
        if curve[0] > curve[2]:
            curves.append(_LINEAR_CURVE)
        else:
            curves.append(curve)  # type: ignore[arg-type]
    return tuple(curves)


def _decode_camera_interpolation(value: bytes) -> tuple[tuple[int, int, int, int], ...]:
    """VMDカメラの位置・回転・距離・画角のベジェ曲線を取り出す。"""

    curves: list[tuple[int, int, int, int]] = []
    for channel in range(6):
        curve = tuple(value[channel * 4 + point] for point in range(4))
        if curve[0] > curve[2]:
            curves.append(_LINEAR_CURVE)
        else:
            curves.append(curve)  # type: ignore[arg-type]
    return tuple(curves)


def _encode_bone_interpolation(
    curves: tuple[tuple[int, int, int, int], ...],
) -> bytes:
    """確認用VMDへ標準的な補間値を書き込む。"""

    encoded = bytearray(64)
    selected = curves if len(curves) == 4 else _LINEAR_INTERPOLATION
    for channel, curve in enumerate(selected):
        for point, offset in zip(curve, (0, 4, 8, 12), strict=True):
            encoded[channel + offset] = max(0, min(127, point))
    return bytes(encoded)
