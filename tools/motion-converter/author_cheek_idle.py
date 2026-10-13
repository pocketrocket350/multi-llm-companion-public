"""公式の静止姿勢を参考に、口元へ手を添える発話用候補を生成する。"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from multi_llm_companion.integrations.pmx_preview import read_pmx_preview
from multi_llm_companion.integrations.vmd_motion import (
    _GENTLE_INTERPOLATION,
    _VMD_HEADER,
    _encode_bone_interpolation,
    _encode_name,
    read_vmd_motion,
)

ROOT = Path(__file__).resolve().parents[2]


def multiply(a, b):
    av, bv = np.array(a[:3]), np.array(b[:3])
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def aim(source, target):
    a, b = source / np.linalg.norm(source), target / np.linalg.norm(target)
    q = np.r_[np.cross(a, b), 1 + a @ b]
    return q / np.linalg.norm(q)


def blend(a, b, weight):
    a, b = np.array(a), np.array(b)
    if a @ b < 0:
        b = -b
    q = (1 - weight) * a + weight * b
    return q / np.linalg.norm(q)


def rotate(q, vector):
    return multiply(multiply(q, np.r_[vector, 0.0]), np.r_[-q[:3], q[3]])[:3]


def orient_hand(finger, palm, target_finger, target_palm):
    """指方向と掌の法線を別々に合わせ、手首の裏返りを防ぐ。"""
    q = aim(finger, target_finger)
    axis = target_finger / np.linalg.norm(target_finger)
    source = rotate(q, palm)
    target = target_palm - axis * (target_palm @ axis)
    target /= np.linalg.norm(target)
    angle = math.atan2(axis @ np.cross(source, target), source @ target)
    return multiply(np.r_[axis * math.sin(angle / 2), math.cos(angle / 2)], q)


def main():
    model = read_pmx_preview(ROOT / "local/models/mmd/Ui_MMD_ver1/雨衣_2025_0516.pmx")
    bones = {b.name: np.array(b.position) for b in model.bones}
    baseline = read_vmd_motion(ROOT / "local/motions/phase8/listening_idle_01.vmd")
    rest = {f.bone_name: f.rotation for f in baseline.bone_frames if f.frame_number == 0}
    shoulder, elbow, wrist = (bones[n] for n in ("左腕", "左ひじ", "左手首"))
    # 実ボーン長で肘を外下方に置き、指先は頬の手前へ向ける。
    target = np.array((1.10, 14.70, -2.3))
    direction = target - shoulder
    distance = np.linalg.norm(direction)
    direction /= distance
    upper, lower = np.linalg.norm(elbow - shoulder), np.linalg.norm(wrist - elbow)
    along = (upper**2 - lower**2 + distance**2) / (2 * distance)
    side = np.array((1.0, -1.0, 0.0))
    side -= direction * (side @ direction)
    side /= np.linalg.norm(side)
    elbow_target = shoulder + along * direction + math.sqrt(upper**2 - along**2) * side
    arm_q = aim(elbow - shoulder, elbow_target - shoulder)
    forearm_q = aim(wrist - elbow, target - elbow_target)
    # 掌の回転を前腕にも分配し、アプリの手首回転上限を超えない。
    forearm_axis = target - elbow_target
    forearm_axis /= np.linalg.norm(forearm_axis)
    forearm_q = multiply(np.r_[forearm_axis * math.sin(0.25), math.cos(0.25)], forearm_q)
    finger = bones["左人指１"] - wrist
    palm = np.cross(finger, bones["左小指１"] - wrist)
    palm /= np.linalg.norm(palm)
    target_finger = np.array((-0.45, 1.0, 0.10))
    target_palm = np.array((-0.6, 0.0, 1.0))
    hand_q = orient_hand(finger, palm, target_finger, target_palm)
    assert rotate(hand_q, palm) @ target_palm > 0.9
    inverse_arm = np.r_[-arm_q[:3], arm_q[3]]
    inverse_forearm = np.r_[-forearm_q[:3], forearm_q[3]]
    pose = dict(rest)
    pose.update(
        {
            "左腕": arm_q,
            "左ひじ": multiply(inverse_arm, forearm_q),
            "左手首": multiply(inverse_forearm, hand_q),
            "首": (0, 0, math.sin(-0.025), math.cos(-0.025)),
            "頭": (0, 0, math.sin(-0.025), math.cos(-0.025)),
        }
    )
    # 指は軽く曲げる。表情・目・口形・脚・IKにはキーを作らない。
    for name in bones:
        if name.startswith(("左中指", "左薬指", "左小指")) and name[-1] in "１２３":
            rest[name] = (0, 0, 0, 1)
            axis = np.cross(finger, palm)
            axis /= np.linalg.norm(axis)
            pose[name] = tuple(np.r_[axis * math.sin(0.10), math.cos(0.10)])
    # 復帰キーは作らず、発話終了時の既存ミキサー補間へ任せる。
    keys = ((0, 0), (45, 1), (60, 1))
    output = ROOT / "local/motions/phase8/cheek_idle_02.vmd"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(_VMD_HEADER.ljust(30, b"\0"))
        stream.write(_encode_name("雨衣", 20))
        stream.write(struct.pack("<I", len(rest) * len(keys)))
        for frame, weight in keys:
            for name, rotation in rest.items():
                q = blend(rotation, pose[name], weight)
                stream.write(_encode_name(name, 15))
                stream.write(struct.pack("<I3f4f", frame, 0, 0, 0, *q))
                stream.write(_encode_bone_interpolation(_GENTLE_INTERPOLATION))
        stream.write(struct.pack("<5I", 0, 0, 0, 0, 0))
    motion = read_vmd_motion(output)
    assert len({(f.bone_name, f.frame_number) for f in motion.bone_frames}) == len(
        motion.bone_frames
    )
    assert not motion.morph_frames and not motion.ik_frames
    assert motion.frame_count == 60
    assert all(np.isfinite(f.rotation).all() for f in motion.bone_frames)
    assert all(abs(np.linalg.norm(f.rotation) - 1) < 1e-6 for f in motion.bone_frames)
    assert not any("足" in f.bone_name or "目" in f.bone_name for f in motion.bone_frames)
    for name in rest:
        track = [f for f in motion.bone_frames if f.bone_name == name]
        assert track[-2].rotation == track[-1].rotation
    # 手首単体でなく、腕・肘を合成した手の向きで動作量を検査する。
    hand_world = []
    for time in (0, 60):
        q = np.array((0.0, 0.0, 0.0, 1.0))
        for name in ("左腕", "左ひじ", "左手首"):
            key = next(
                f for f in motion.bone_frames if f.bone_name == name and f.frame_number == time
            )
            q = multiply(q, key.rotation)
        hand_world.append(q)
    assert abs(np.dot(*hand_world)) < 0.9
    print(output)
    print("Shoulder, elbow, wrist targets:", shoulder, elbow_target, target)


if __name__ == "__main__":
    main()
