"""猫背の姿勢と腕を雨衣へ合わせ、原本を保持した複製VMDを生成する。"""

import hashlib
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "local/motions/distributed_samples/_従業員オトカム氏/オトカム式_待機モーションパック/ot0510_待機_猫背_まばたき付き.vmd"
)
TARGET = Path("local/motions/ui_fitted/ot0510_待機_猫背_まばたき付き_猫背姿勢調整_袖余裕.vmd")


def fitted_bytes(original: bytes, *, arm_degrees: float = 15.0) -> tuple[bytes, list[int]]:
    """既存の両手後ろ補正と同じ形式で、腕の回転以外の全バイトを保持する。"""
    if len(original) < 54 or not original[:30].rstrip(b"\0").startswith(
        b"Vocaloid Motion Data 0002"
    ):
        raise ValueError("VMD 0002ではありません")
    count = struct.unpack_from("<I", original, 50)[0]
    if len(original) < 54 + count * 111:
        raise ValueError("ボーンフレームが途中で切れています")
    result, offsets, names = bytearray(original), [], set()
    for index in range(count):
        start = 54 + index * 111
        name = original[start : start + 15].split(b"\0", 1)[0].decode("cp932")
        if name not in ("左腕", "右腕"):
            continue
        half = math.radians(arm_degrees if name == "左腕" else -arm_degrees) / 2
        sine, cosine = math.sin(half), math.cos(half)
        offset = start + 31
        x, y, z, w = struct.unpack_from("<4f", original, offset)
        quaternion = (
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
            cosine * w - sine * z,
        )
        struct.pack_into("<4f", result, offset, *quaternion)
        offsets.append(offset)
        names.add(name)
    if names != {"左腕", "右腕"}:
        raise ValueError("左右の腕トラックがありません")
    return bytes(result), offsets


def right_arm_clearance_bytes(original: bytes) -> tuple[bytes, list[int]]:
    """既存の回転合成を使い、下ろした右腕だけを外へ8度離す。"""
    fitted, arm_offsets = fitted_bytes(original, arm_degrees=8.0)
    result, offsets = bytearray(original), []
    for offset in arm_offsets:
        name = original[offset - 31 : offset - 16].split(b"\0", 1)[0].decode("cp932")
        if name == "右腕":
            result[offset : offset + 16] = fitted[offset : offset + 16]
            offsets.append(offset)
    return bytes(result), offsets


def reference_pose_bytes(original: bytes) -> bytes:
    """元の20秒を保ち、胸を前へ丸めて顔の向きを補い、腕を体側へ戻す。"""
    fitted, _ = fitted_bytes(original, arm_degrees=15.0)
    result = bytearray(fitted)
    count = struct.unpack_from("<I", fitted, 50)[0]
    adjusted = set()
    for index in range(count):
        start = 54 + index * 111
        name = fitted[start : start + 15].split(b"\0", 1)[0].decode("cp932")
        angle = {"上半身2": -10.0, "頭": 10.0}.get(name)
        if angle is None:
            continue
        half = math.radians(angle) / 2
        sine, cosine = math.sin(half), math.cos(half)
        x, y, z, w = struct.unpack_from("<4f", fitted, start + 31)
        struct.pack_into(
            "<4f",
            result,
            start + 31,
            cosine * x + sine * w,
            cosine * y - sine * z,
            cosine * z + sine * y,
            cosine * w - sine * x,
        )
        adjusted.add(name)
    if adjusted != {"上半身2", "頭"}:
        raise ValueError("胸と頭のトラックがありません")
    return bytes(result)


def entry_exit_bytes(original: bytes) -> bytes:
    """15度版の前後に各2秒の腕開閉を加え、元の20秒間と瞬きを保持する。"""
    fitted, _ = fitted_bytes(original)
    bone_count = struct.unpack_from("<I", fitted, 50)[0]
    morph_offset = 54 + bone_count * 111
    morph_count = struct.unpack_from("<I", fitted, morph_offset)[0]
    tail_offset = morph_offset + 4 + morph_count * 23
    if len(fitted) < tail_offset or fitted[tail_offset:] not in (
        b"",
        bytes(12),
        bytes(16),
    ):
        raise ValueError("カメラ・照明・影・IKを含むVMDはこの加工の対象外です")
    bones = [fitted[54 + i * 111 : 54 + (i + 1) * 111] for i in range(bone_count)]
    morphs = [
        fitted[morph_offset + 4 + i * 23 : morph_offset + 4 + (i + 1) * 23]
        for i in range(morph_count)
    ]
    end = max(struct.unpack_from("<I", record, 15)[0] for record in bones + morphs)

    def at(record: bytes, frame: int, *, open_arm: bool = False) -> bytes:
        result = bytearray(record)
        struct.pack_into("<I", result, 15, frame)
        name = record[:15].split(b"\0", 1)[0].decode("cp932")
        if open_arm and name in ("左腕", "右腕"):
            half = math.radians(40 if name == "左腕" else -40) / 2
            sine, cosine = math.sin(half), math.cos(half)
            x, y, z, w = struct.unpack_from("<4f", record, 31)
            struct.pack_into(
                "<4f",
                result,
                31,
                cosine * x - sine * y,
                cosine * y + sine * x,
                cosine * z + sine * w,
                cosine * w - sine * z,
            )
        return bytes(result)

    def extended(records: list[bytes], *, arms: bool) -> list[bytes]:
        tracks: dict[bytes, list[bytes]] = {}
        for record in records:
            tracks.setdefault(record[:15], []).append(record)
        output = []
        for track in tracks.values():
            track.sort(key=lambda record: struct.unpack_from("<I", record, 15)[0])
            if arms and struct.unpack_from("<I", track[0], 15)[0] != 0:
                raise ValueError("開始フレーム0のないトラックは加工しません")
            output.extend(at(track[0], frame, open_arm=arms) for frame in (0, 18))
            output.extend(
                at(record, struct.unpack_from("<I", record, 15)[0] + 60) for record in track
            )
            if struct.unpack_from("<I", track[-1], 15)[0] < end:
                output.append(at(track[-1], end + 60))
            output.extend(at(track[-1], frame, open_arm=arms) for frame in (end + 102, end + 120))
        return output

    extended_bones = extended(bones, arms=True)
    extended_morphs = extended(morphs, arms=False)
    return (
        fitted[:50]
        + struct.pack("<I", len(extended_bones))
        + b"".join(extended_bones)
        + struct.pack("<I", len(extended_morphs))
        + b"".join(extended_morphs)
        + fitted[tail_offset:]
    )


if __name__ == "__main__":
    source, target = ROOT / SOURCE, ROOT / TARGET
    original = source.read_bytes()
    fitted = reference_pose_bytes(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != fitted:
        raise FileExistsError("既存の異なる補正版は上書きしません")
    target.write_bytes(fitted)
    assert source.read_bytes() == original
    print("added_entry_exit_seconds", 0)
    print("source_sha256", hashlib.sha256(original).hexdigest())
    print("fitted_sha256", hashlib.sha256(fitted).hexdigest())
