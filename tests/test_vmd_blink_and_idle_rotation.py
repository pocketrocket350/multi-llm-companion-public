from __future__ import annotations

import math
import os
import struct
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.dense_pmx_widget import DensePmxWidget
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarTexturedTriangle,
    UiMmdAvatarCommand,
    UiVmdBoneFrame,
    UiVmdMorphFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.natural_motion import NaturalMotionFrame


def _motion(name: str, blink: float | None = None) -> UiVmdMotion:
    return UiVmdMotion(
        name=name,
        frame_count=30,
        bone_frames=(UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        morph_frames=() if blink is None else (UiVmdMorphFrame("まばたき", 0, blink),),
    )


@pytest.fixture
def renderer():
    app = QApplication.instance() or QApplication([])
    widget = DensePmxWidget()
    widget.set_preview(
        UiAvatarModelPreview(
            model_name="閉眼と待機切替テスト",
            vertex_count=3,
            face_count=1,
            points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
            mouth_morphs={"a": ((0, 0.0, 0.4, 0.0),)},
            named_vertex_morphs={"まばたき": ((0, 0.2, 0.0, 0.0),)},
            dense_textured_triangles=(
                UiAvatarTexturedTriangle(
                    indexes=(0, 1, 2),
                    uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                    texture_path="",
                    diffuse=(1.0, 1.0, 1.0, 1.0),
                ),
            ),
            vmd_motion=_motion("標準"),
            idle_motions=(_motion("待機1", 0.4), _motion("待機2")),
        )
    )
    yield widget
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("source", ["base", "preview", "gesture"])
@pytest.mark.parametrize("blink", [None, 0.0, 0.8])
def test_active_vmd_blink_owns_fast_vertices_without_double_application(
    renderer: DensePmxWidget, source: str, blink: float | None
) -> None:
    motion = _motion("対象", blink)
    if source == "base":
        renderer.set_preview(replace(renderer._preview, vmd_motion=motion))
    else:
        # 背景待機の閉眼は、前面のモーションへ混ぜない。
        renderer._vmd_morph_weights = {"まばたき": 0.3}
        if source == "preview":
            renderer.start_motion_preview(motion)
        else:
            renderer.set_mmd_command(
                UiMmdAvatarCommand(
                    emotion="neutral",
                    emotion_intensity=0.0,
                    emotion_morph="",
                    gaze="forward",
                    gaze_bone="",
                    gesture="nod",
                    gesture_target="",
                    gesture_motion=motion,
                    gesture_motion_type="reaction",
                )
            )
            renderer.begin_conversation_motion()
    renderer.set_mouth_shape("a")
    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=1.0, breath=0.1))
    expected = 1.0 if blink is None else blink
    assert renderer._blink_weight == pytest.approx(expected)
    assert struct.unpack_from("<3f", renderer._groups[0].vertices, 0) == pytest.approx(
        (expected * 0.2, 0.4, 0.0)
    )
    assert "まばたき" not in renderer._pose_settings()[3]
    assert renderer._effective_natural_motion().breath == 0.0


@pytest.mark.parametrize("stop_method", ["stop_motion_preview", "interrupt_motion"])
def test_preview_blink_restores_default_track_on_stop(
    renderer: DensePmxWidget, stop_method: str
) -> None:
    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=0.6))
    renderer.start_motion_preview(_motion("プレビュー", 1.0))
    assert renderer._blink_weight == 1.0
    getattr(renderer, stop_method)()
    assert renderer._blink_weight == 0.6


def test_vmd_blink_follows_interpolated_playback_values(renderer: DensePmxWidget) -> None:
    motion = replace(
        _motion("開閉"),
        loop=False,
        morph_frames=(
            UiVmdMorphFrame("まばたき", 0, 0.0),
            UiVmdMorphFrame("まばたき", 15, 1.0),
            UiVmdMorphFrame("まばたき", 30, 0.0),
        ),
    )
    renderer.start_motion_preview(motion)
    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=0.2))
    for expected in (0.5, 1.0, 0.5):
        renderer.advance_heavy_animation(0.25, True)
        assert renderer._blink_weight == pytest.approx(expected)


def test_idle_rotates_once_then_returns_without_reloading_model(renderer: DensePmxWidget) -> None:
    original = renderer._preview
    physics = renderer._physics
    renderer._advance_idle_rotation(59.0)
    assert renderer._idle_motion.name == "標準"
    renderer._advance_idle_rotation(1.0)
    assert renderer._idle_motion.name == "待機1"
    assert not renderer._idle_motion.loop
    assert renderer._preview is original
    assert renderer._physics is physics
    for _ in range(30):
        renderer.advance_heavy_animation(0.1, True)
    renderer._vmd_player.advance_motion(renderer._preview, renderer._idle_motion, 1.0, True)
    renderer._advance_idle_rotation(0.1)
    assert renderer._idle_motion is original.vmd_motion
    for _ in range(30):
        renderer.advance_heavy_animation(0.1, True)
    renderer._advance_idle_rotation(60.0)
    assert renderer._idle_motion.name == "待機2"


def test_idle_rotation_does_not_interrupt_voice_or_manual_preview(renderer: DensePmxWidget) -> None:
    renderer._speaking = True
    renderer._advance_idle_rotation(65.0)
    assert not renderer._idle_variant_active
    renderer._speaking = False
    renderer._advance_idle_rotation(60.0)
    assert renderer._idle_variant_active
    renderer.start_motion_preview(_motion("手動"))
    renderer._advance_idle_rotation(65.0)
    assert renderer._preview_motion.name == "手動"
    assert renderer._idle_motion.name == "標準"
    renderer.stop_motion_preview()
    renderer._advance_idle_rotation(59.0)
    assert not renderer._idle_variant_active


def test_idle_rotation_disabled_when_motion_is_paused(renderer: DensePmxWidget) -> None:
    renderer._idle_elapsed = 59.9
    renderer.advance_heavy_animation(1.0, False)
    assert not renderer._idle_variant_active
    assert renderer._idle_elapsed == 59.9


def test_gaze_only_command_does_not_override_base_vmd_blink(renderer: DensePmxWidget) -> None:
    renderer.set_preview(replace(renderer._preview, vmd_motion=_motion("標準", 0.8)))
    renderer.set_mmd_command(
        UiMmdAvatarCommand("neutral", 0.0, "", "left", "", "none", ""),
        smooth_gaze_transition=True,
    )
    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=0.0))
    assert renderer._blink_weight == 0.8


def test_idle_transition_reaches_target_without_final_pose_jump(renderer: DensePmxWidget) -> None:
    motion = replace(
        _motion("別待機"),
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("センター", 0, (0.6, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    )
    renderer._pose_settings()
    renderer._switch_idle_motion(motion)
    positions = []
    for _ in range(5):
        renderer.advance_heavy_animation(0.6, True)
        positions.append(renderer._pose_settings()[2].get(0, (0.0, 0.0, 0.0))[0])
    assert positions == pytest.approx([0.09375, 0.3, 0.50625, 0.6, 0.6])
    assert renderer._vmd_player._elapsed_seconds == pytest.approx(3.0)
    renderer.advance_heavy_animation(0.12, True)
    assert renderer._vmd_player._elapsed_seconds == pytest.approx(3.12)


@pytest.mark.parametrize("delta", [0.03, 0.12, 0.3])
def test_idle_transition_clock_does_not_depend_on_pose_query_count(
    renderer: DensePmxWidget, delta: float
) -> None:
    renderer._switch_idle_motion(_motion("次の待機"))
    for _ in range(round(1.8 / delta)):
        renderer.advance_heavy_animation(delta, True)
        for _ in range(4):
            renderer._pose_settings()
    assert renderer._idle_transition_elapsed == pytest.approx(1.8)
    assert renderer._idle_transition_from is not None


@pytest.mark.parametrize("loop", [False, True])
def test_reaction_finishes_after_short_voice_and_can_be_interrupted(
    renderer: DensePmxWidget,
    loop: bool,
) -> None:
    motion = replace(_motion("反応", 0.8), loop=loop, frame_count=90)
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "happy",
            0.7,
            "",
            "forward",
            "",
            "nod",
            "",
            gesture_motion=motion,
            gesture_motion_type="reaction",
        )
    )
    renderer.begin_conversation_motion()
    renderer.advance_conversation_animation(0.3)
    renderer.end_conversation_motion()
    assert renderer.is_conversation_motion_active
    assert renderer.continues_conversation_motion_without_voice
    assert not renderer._command.gesture_motion.loop
    for _ in range(8):
        renderer.advance_conversation_animation(0.3)
    renderer.advance_conversation_animation(0.29)
    assert renderer._gesture_vmd_player._elapsed_seconds == pytest.approx(2.99)
    assert renderer.is_conversation_motion_active
    renderer.advance_conversation_animation(0.02)
    assert not renderer.is_conversation_motion_active
    renderer.begin_conversation_motion()
    renderer.advance_conversation_animation(0.3)
    renderer.interrupt_motion()
    assert not renderer.is_conversation_motion_active


@pytest.mark.parametrize("intensity", [0.0, 0.2, 0.3919, 1.0])
@pytest.mark.parametrize("full_body", [False, True])
@pytest.mark.parametrize("asset_scale", [0.5, 1.0])
def test_full_body_vmd_preserves_pose_without_emotion_attenuation(
    renderer: DensePmxWidget, intensity: float, full_body: bool, asset_scale: float
) -> None:
    motion = replace(
        _motion("配布動作"),
        bone_frames=(UiVmdBoneFrame("センター", 0, (0.8, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    )
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "happy",
            intensity,
            "",
            "forward",
            "",
            "playful_shoot",
            "",
            gesture_motion=motion,
            gesture_motion_type="reaction",
            gesture_body_mask=("full_body",) if full_body else (),
            gesture_intensity=intensity,
            gesture_intensity_scale=asset_scale,
        )
    )
    renderer.begin_conversation_motion()
    for _ in range(80):
        renderer._pose_settings()
    expected = 0.8 * asset_scale * (1.0 if full_body else intensity)
    assert renderer._motion_mixer.current.translations.get(0, (0.0, 0.0, 0.0))[0] == pytest.approx(
        expected
    )
    renderer.end_conversation_motion()
    assert renderer.is_conversation_motion_active
    renderer.interrupt_motion()
    assert not renderer.is_conversation_motion_active


def test_reaction_return_keeps_center_height_then_eases_to_idle(renderer: DensePmxWidget) -> None:
    motion = replace(
        _motion("低い姿勢の反応"),
        bone_frames=(UiVmdBoneFrame("センター", 0, (0.0, -0.8, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    )
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "neutral",
            0.5,
            "",
            "forward",
            "",
            "think",
            "",
            gesture_motion=motion,
            gesture_motion_type="reaction",
            gesture_intensity=0.5,
        )
    )
    renderer.begin_conversation_motion()
    for _ in range(20):
        renderer._pose_settings()
    before = renderer._motion_mixer.current.translations[0][1]
    assert before < -0.3
    renderer._finish_managed_motion()
    assert renderer._pose_settings()[2][0][1] == pytest.approx(before)
    heights = [before]
    for _ in range(30):
        renderer.advance_heavy_animation(0.1, True)
        heights.append(renderer._pose_settings()[2].get(0, (0.0, 0.0, 0.0))[1])
    assert before < heights[1] < heights[6] < 0.0
    assert heights[-1] == pytest.approx(0.0)
    assert all(
        left <= right + 1e-8 <= 1e-8 for left, right in zip(heights, heights[1:], strict=False)
    )


def test_preview_keeps_valid_rotation_above_old_angle_cap(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "_queue_pose_mesh_refresh", lambda **kwargs: None)
    angle = 3.1
    motion = replace(
        _motion("大きな回転"),
        bone_frames=(
            UiVmdBoneFrame(
                "センター", 0, (0, 0, 0), (0, 0, math.sin(angle / 2), math.cos(angle / 2))
            ),
        ),
    )
    renderer.start_motion_preview(motion)
    renderer._pose_settings()
    assert renderer._motion_mixer.current.rotations[0][2] == pytest.approx(angle)
