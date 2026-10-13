from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from multi_llm_companion.integrations.vmd_motion import (
    VmdMotionError,
    read_vmd_motion,
    write_confirmation_vmd,
    write_gentle_sway_idle_vmd,
    write_listening_idle_vmd,
)
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiVmdBoneFrame,
    UiVmdCameraFrame,
    UiVmdCompatibility,
    UiVmdIkFrame,
    UiVmdIkState,
    UiVmdLightFrame,
    UiVmdMorphFrame,
    UiVmdMotion,
    UiVmdShadowFrame,
)
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer


def _write_vmd(path: Path, interpolation: bytes = b"\0" * 64) -> None:
    data = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0"))
    data.extend(b"test model".ljust(20, b"\0"))
    data.extend(struct.pack("<I", 2))
    for frame_number, translation in ((0, (0.0, 0.0, 0.0)), (3, (0.3, 0.0, 0.0))):
        data.extend("センター".encode("shift_jis").ljust(15, b"\0"))
        data.extend(struct.pack("<I3f4f", frame_number, *translation, 0.0, 0.0, 0.0, 1.0))
        data.extend(interpolation)
    data.extend(struct.pack("<I", 2))
    for frame_number, weight in ((0, 0.0), (3, 0.8)):
        data.extend("笑顔".encode("shift_jis").ljust(15, b"\0"))
        data.extend(struct.pack("<If", frame_number, weight))
    path.write_bytes(data)


def _write_extended_vmd(path: Path) -> None:
    data = bytearray(b"Vocaloid Motion Data 0002".ljust(30, b"\0"))
    data.extend("人工モデル".encode("shift_jis").ljust(20, b"\0"))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 2))
    camera_curve = bytes((20, 20, 107, 107)) * 6
    for frame_number, distance in ((0, -45.0), (30, -30.0)):
        data.extend(struct.pack("<If3f3f", frame_number, distance, 0.0, 10.0, 0.0, 0.1, 0.2, 0.0))
        data.extend(camera_curve)
        data.extend(struct.pack("<IB", 35, 0))
    data.extend(struct.pack("<I", 2))
    for frame_number, red in ((0, 0.8), (30, 0.4)):
        data.extend(struct.pack("<I3f3f", frame_number, red, 0.7, 0.6, -0.5, 1.0, 0.5))
    data.extend(struct.pack("<I", 1))
    data.extend(struct.pack("<IBf", 15, 1, 12.5))
    data.extend(struct.pack("<I", 2))
    for frame_number, enabled in ((0, 1), (15, 0)):
        data.extend(struct.pack("<IBI", frame_number, 1, 1))
        data.extend("左足ＩＫ".encode("shift_jis").ljust(20, b"\0"))
        data.extend(struct.pack("<B", enabled))
    path.write_bytes(data)


def test_vmd_motion_reads_bone_frames(tmp_path: Path) -> None:
    path = tmp_path / "motion.vmd"
    _write_vmd(path)

    motion = read_vmd_motion(path)

    assert motion.name == "motion"
    assert motion.frame_count == 3
    assert [frame.bone_name for frame in motion.bone_frames] == ["センター", "センター"]
    assert [frame.weight for frame in motion.morph_frames] == [0.0, pytest.approx(0.8)]
    assert motion.bone_frames[0].interpolation == ((20, 20, 107, 107),) * 4


def test_vmd_motion_reads_separate_bezier_curves(tmp_path: Path) -> None:
    curves = (
        (12, 24, 100, 110),
        (20, 30, 90, 107),
        (4, 50, 80, 120),
        (16, 8, 112, 124),
    )
    raw = bytearray(64)
    for channel, curve in enumerate(curves):
        for value, offset in zip(curve, (0, 4, 8, 12), strict=True):
            raw[channel + offset] = value
    path = tmp_path / "interpolation.vmd"
    _write_vmd(path, bytes(raw))
    motion = read_vmd_motion(path)

    assert motion.bone_frames[0].interpolation == curves


def test_vmd_motion_reads_camera_light_shadow_and_ik_frames(tmp_path: Path) -> None:
    path = tmp_path / "extended.vmd"
    _write_extended_vmd(path)

    motion = read_vmd_motion(path)

    assert motion.target_model_name == "人工モデル"
    assert motion.frame_count == 30
    assert [frame.distance for frame in motion.camera_frames] == [-45.0, -30.0]
    assert motion.camera_frames[0].view_angle == 35
    assert motion.camera_frames[0].perspective is True
    assert motion.light_frames[1].color == pytest.approx((0.4, 0.7, 0.6))
    assert motion.shadow_frames[0].mode == 1
    assert motion.shadow_frames[0].distance == pytest.approx(12.5)
    assert [frame.states[0].enabled for frame in motion.ik_frames] == [True, False]


def test_vmd_motion_rejects_truncated_optional_section(tmp_path: Path) -> None:
    path = tmp_path / "truncated.vmd"
    _write_extended_vmd(path)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(VmdMotionError, match="途中"):
        read_vmd_motion(path)


def test_vmd_motion_accepts_legacy_file_without_ik_section(tmp_path: Path) -> None:
    path = tmp_path / "legacy.vmd"
    _write_vmd(path)
    path.write_bytes(path.read_bytes() + struct.pack("<3I", 0, 0, 0))

    motion = read_vmd_motion(path)

    assert len(motion.bone_frames) == 2
    assert len(motion.morph_frames) == 2
    assert motion.ik_frames == ()


@pytest.mark.parametrize("tail", [b"\0", b"\0\0", b"\0\0\0", struct.pack("<I", 1)])
def test_vmd_motion_rejects_partial_legacy_ik_section(tmp_path: Path, tail: bytes) -> None:
    path = tmp_path / "partial_ik.vmd"
    _write_vmd(path)
    path.write_bytes(path.read_bytes() + struct.pack("<3I", 0, 0, 0) + tail)

    with pytest.raises(VmdMotionError, match="途中"):
        read_vmd_motion(path)


def test_vmd_player_interpolates_bone_translation() -> None:
    motion = UiVmdMotion(
        "テスト",
        3,
        (
            UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame("センター", 3, (0.3, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
        morph_frames=(
            UiVmdMorphFrame("笑顔", 0, 0.0),
            UiVmdMorphFrame("笑顔", 3, 0.8),
        ),
    )
    preview = UiAvatarModelPreview(
        model_name="VMDテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=motion,
    )
    player = VmdMotionPlayer()

    rotations, translations, morph_weights = player.advance(preview, 0.05, True)

    assert rotations[0] == pytest.approx((0.0, 0.0, 0.0))
    assert translations[0] == pytest.approx((0.15, 0.0, 0.0))
    assert morph_weights["笑顔"] == pytest.approx(0.4)
    assert (
        player.status_text == "VMD再生: テスト（3フレーム・表情 2・IK 0・カメラ 0・照明 0・影 0）"
    )


def test_vmd_player_applies_catalog_intensity_to_idle_motion() -> None:
    half = 2**-0.5
    motion = UiVmdMotion(
        "強度補正テスト",
        30,
        (UiVmdBoneFrame("右腕", 0, (0.4, 0.0, 0.0), (half, 0.0, 0.0, half)),),
        loop=False,
        morph_frames=(UiVmdMorphFrame("笑顔", 0, 0.8),),
        intensity_scale=0.75,
    )
    preview = UiAvatarModelPreview(
        "強度補正テスト",
        0,
        0,
        (),
        bones=(UiAvatarBone("右腕", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=motion,
    )

    rotations, translations, morphs = VmdMotionPlayer().advance(preview, 0.1, True)

    assert translations[0] == pytest.approx((0.3, 0.0, 0.0))
    assert rotations[0][0] == pytest.approx(math.pi * 0.375)
    assert morphs["笑顔"] == pytest.approx(0.6)


def test_vmd_player_applies_separate_translation_curves() -> None:
    motion = UiVmdMotion(
        "補間テスト",
        30,
        (
            UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame(
                "センター",
                30,
                (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
                (
                    (0, 0, 127, 0),
                    (20, 20, 107, 107),
                    (0, 127, 127, 127),
                    (20, 20, 107, 107),
                ),
            ),
        ),
        loop=False,
    )
    preview = UiAvatarModelPreview(
        "補間テスト",
        0,
        0,
        (),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=motion,
    )
    _rotations, translations, _morphs = VmdMotionPlayer().advance(preview, 0.5, True)

    assert translations[0][0] < translations[0][1]
    assert translations[0][1] == pytest.approx(0.5)
    assert translations[0][2] > translations[0][1]


def test_vmd_player_uses_shortest_quaternion_rotation() -> None:
    half = 2**-0.5
    motion = UiVmdMotion(
        "回転補間テスト",
        30,
        (
            UiVmdBoneFrame("頭", 0, (0.0, 0.0, 0.0), (0.0, 0.0, half, half)),
            UiVmdBoneFrame("頭", 30, (0.0, 0.0, 0.0), (0.0, 0.0, -half, -half)),
        ),
        loop=False,
    )
    preview = UiAvatarModelPreview(
        "回転補間テスト",
        0,
        0,
        (),
        bones=(UiAvatarBone("頭", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=motion,
    )
    rotations, _translations, _morphs = VmdMotionPlayer().advance(preview, 0.5, True)

    assert rotations[0][2] == pytest.approx(math.pi / 2.0)


def test_vmd_player_replays_camera_light_shadow_and_ik_state() -> None:
    motion = UiVmdMotion(
        "演出テスト",
        30,
        (),
        loop=False,
        camera_frames=(
            UiVmdCameraFrame(0, -45.0, (0.0, 10.0, 0.0), (0.0, 0.0, 0.0)),
            UiVmdCameraFrame(30, -30.0, (3.0, 10.0, 0.0), (0.0, 0.2, 0.0)),
        ),
        light_frames=(
            UiVmdLightFrame(0, (1.0, 0.8, 0.6), (-0.5, 1.0, 0.5)),
            UiVmdLightFrame(30, (0.5, 0.4, 0.3), (0.5, 1.0, 0.5)),
        ),
        shadow_frames=(UiVmdShadowFrame(0, 1, 10.0),),
        ik_frames=(
            UiVmdIkFrame(0, True, (UiVmdIkState("左足ＩＫ", True),)),
            UiVmdIkFrame(15, True, (UiVmdIkState("左足ＩＫ", False),)),
        ),
    )
    preview = UiAvatarModelPreview("演出テスト", 0, 0, (), vmd_motion=motion)
    player = VmdMotionPlayer()

    player.advance(preview, 0.5, True)

    assert player.camera_frame is not None
    assert player.camera_frame.distance == pytest.approx(-37.5)
    assert player.camera_frame.position[0] == pytest.approx(1.5)
    assert player.light_frame is not None
    assert player.light_frame.color == pytest.approx((0.75, 0.6, 0.45))
    assert player.shadow_frame == motion.shadow_frames[0]
    assert player.ik_states == {"左足ＩＫ": False}
    assert player.display_model is True


def test_vmd_player_smooths_loop_boundary() -> None:
    motion = UiVmdMotion(
        "ループ境界",
        30,
        (
            UiVmdBoneFrame("センター", 0, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame("センター", 30, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
    )
    preview = UiAvatarModelPreview(
        "ループ境界",
        0,
        0,
        (),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=motion,
    )
    player = VmdMotionPlayer()
    _rotations, before, _morphs = player.advance(preview, 0.95, True)
    _rotations, after, _morphs = player.advance(preview, 0.10, True)

    assert before[0][0] > 0.8
    assert after[0][0] > 0.0
    assert after[0][0] < before[0][0]


def test_vmd_player_status_distinguishes_missing_and_unsupported_items() -> None:
    motion = UiVmdMotion(
        "診断テスト",
        1,
        (),
        compatibility=UiVmdCompatibility(
            supported_bones=("センター",),
            missing_bones=("追加ボーン",),
            supported_iks=("左足ＩＫ",),
            missing_iks=("補助ＩＫ",),
            translation_height_ratio=0.12,
            target_model_matches=False,
            unsupported_sections=("セルフ影",),
        ),
    )
    player = VmdMotionPlayer()
    player.reset(motion)

    assert "対応 ボーン1/2" in player.status_text
    assert "最大移動/身長12%" in player.status_text
    assert "対象モデル名不一致" in player.status_text
    assert "未検出 追加ボーン、補助ＩＫ" in player.status_text
    assert "未対応 セルフ影" in player.status_text


def test_confirmation_vmd_uses_only_supported_model_bones(tmp_path: Path) -> None:
    path = tmp_path / "confirmation.vmd"

    write_confirmation_vmd(path, ("センター", "上半身", "首", "右腕"), target_model_name="人工PMX")
    motion = read_vmd_motion(path)

    assert path.is_file()
    assert motion.target_model_name == "人工PMX"
    assert {frame.bone_name for frame in motion.bone_frames} == {"センター", "上半身", "首"}
    assert motion.frame_count == 180
    assert motion.morph_frames == ()
    assert motion.bone_frames[0].interpolation == ((30, 10, 97, 117),) * 4
    for bone_name in ("センター", "上半身", "首"):
        frames = [frame for frame in motion.bone_frames if frame.bone_name == bone_name]
        assert [frame.frame_number for frame in frames] == [0, 45, 90, 135, 180]
        assert frames[0].translation == pytest.approx(frames[-1].translation)
        assert frames[0].rotation == pytest.approx(frames[-1].rotation)
    assert max(frame.translation[1] for frame in motion.bone_frames) == pytest.approx(0.024)
    assert max(abs(frame.rotation[0]) for frame in motion.bone_frames) > 0.02
    assert max(abs(frame.rotation[2]) for frame in motion.bone_frames) > 0.015


def test_vmd_player_reports_non_loop_completion_and_reset() -> None:
    motion = UiVmdMotion(
        name="終了テスト",
        frame_count=30,
        bone_frames=(
            UiVmdBoneFrame(
                "頭",
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        loop=False,
    )
    preview = UiAvatarModelPreview(
        model_name="終了テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("頭", -1, (0.0, 0.0, 0.0)),),
    )
    player = VmdMotionPlayer()
    player.reset(motion)

    player.advance_motion(preview, motion, 1.1, True)

    assert player.is_finished is True
    player.reset(None)
    assert player.is_finished is False


def test_confirmation_vmd_requires_a_supported_bone(tmp_path: Path) -> None:
    with pytest.raises(VmdMotionError, match="ボーン"):
        write_confirmation_vmd(tmp_path / "confirmation.vmd", ("左足",))


def test_listening_idle_vmd_keeps_feet_and_face_out_of_the_motion(tmp_path: Path) -> None:
    path = tmp_path / "listening_idle.vmd"

    write_listening_idle_vmd(
        path,
        (
            "腰",
            "上半身",
            "上半身2",
            "首",
            "頭",
            "左腕",
            "左ひじ",
            "左手首",
            "右腕",
            "右ひじ",
            "右手首",
            "左足",
            "右足ＩＫ",
        ),
        target_model_name="人工PMX",
    )

    motion = read_vmd_motion(path)

    assert motion.target_model_name == "人工PMX"
    assert motion.frame_count == 360
    assert {frame.bone_name for frame in motion.bone_frames} == {
        "腰",
        "上半身",
        "上半身2",
        "首",
        "頭",
        "左腕",
        "左ひじ",
        "左手首",
        "右腕",
        "右ひじ",
        "右手首",
    }
    assert not motion.morph_frames
    assert all(frame.translation == (0.0, 0.0, 0.0) for frame in motion.bone_frames)
    for name in {frame.bone_name for frame in motion.bone_frames}:
        seam = [frame for frame in motion.bone_frames if frame.bone_name == name]
        assert seam[0].frame_number == 0
        assert seam[-1].frame_number == 360
        assert seam[0].rotation == seam[-1].rotation


def test_gentle_sway_idle_vmd_keeps_feet_and_face_out_of_the_motion(tmp_path: Path) -> None:
    path = tmp_path / "gentle_sway_idle.vmd"
    bones = (
        "腰",
        "上半身",
        "上半身2",
        "首",
        "頭",
        "左腕",
        "左ひじ",
        "左手首",
        "右腕",
        "右ひじ",
        "右手首",
        "左足",
        "右足ＩＫ",
    )

    write_gentle_sway_idle_vmd(path, bones, target_model_name="人工PMX")

    motion = read_vmd_motion(path)

    assert motion.target_model_name == "人工PMX"
    assert motion.frame_count == 450
    assert {frame.bone_name for frame in motion.bone_frames} == set(bones[:-2])
    assert not motion.morph_frames
    assert all(frame.translation == (0.0, 0.0, 0.0) for frame in motion.bone_frames)
    assert len({frame.rotation for frame in motion.bone_frames if frame.bone_name == "上半身"}) > 1
    for name in {frame.bone_name for frame in motion.bone_frames}:
        seam = [frame for frame in motion.bone_frames if frame.bone_name == name]
        assert seam[0].frame_number == 0
        assert seam[-1].frame_number == 450
        assert seam[0].rotation == seam[-1].rotation
