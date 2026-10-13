"""両手後ろだけを雨衣へ合わせ、原本を保持した複製VMDを生成する。"""

import hashlib
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    "local/motions/distributed_samples/_従業員オトカム氏/オトカム式_待機モーションパック/ot0510_待機_両手後ろ_まばたき付き.vmd"
)
TARGET = Path("local/motions/ui_fitted/ot0510_待機_両手後ろ_まばたき付き.vmd")
ANGLE_DEGREES = -8.0


def fitted_bytes(original: bytes) -> tuple[bytes, list[int]]:
    """左右の腕の回転16バイトだけを変更し、補間・瞬き・時刻を保持する。"""
    if not original[:30].rstrip(b"\0").startswith(b"Vocaloid Motion Data 0002"):
        raise ValueError("VMD 0002ではありません")
    count = struct.unpack_from("<I", original, 50)[0]
    if len(original) < 54 + count * 111:
        raise ValueError("ボーンフレームが途中で切れています")
    result = bytearray(original)
    offsets = []
    names = set()
    sine = math.sin(math.radians(ANGLE_DEGREES) / 2)
    cosine = math.cos(math.radians(ANGLE_DEGREES) / 2)
    for index in range(count):
        start = 54 + index * 111
        name = original[start : start + 15].split(b"\0", 1)[0].decode("cp932")
        if name not in ("左腕", "右腕"):
            continue
        offset = start + 31
        x, y, z, w = struct.unpack_from("<4f", original, offset)
        quaternion = (
            cosine * x + sine * w,
            cosine * y - sine * z,
            cosine * z + sine * y,
            cosine * w - sine * x,
        )
        struct.pack_into("<4f", result, offset, *quaternion)
        offsets.append(offset)
        names.add(name)
    if names != {"左腕", "右腕"}:
        raise ValueError("左右の腕トラックがありません")
    return bytes(result), offsets


if __name__ == "__main__":
    source = ROOT / SOURCE
    target = ROOT / TARGET
    original = source.read_bytes()
    fitted, offsets = fitted_bytes(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != fitted:
        raise FileExistsError("既存の異なる補正版は上書きしません")
    target.write_bytes(fitted)
    assert source.read_bytes() == original
    print("changed_arm_keys", len(offsets))
    print("source_sha256", hashlib.sha256(original).hexdigest())
    print("fitted_sha256", hashlib.sha256(fitted).hexdigest())
