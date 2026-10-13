"""全身の低い復帰と、no/yesの頭・首だけの合成を検証する。"""

import math
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.dense_pmx_widget import (
    _ARM_RETURN_SECONDS,
    DensePmxWidget,
    _arm_return_seconds,
)
from multi_llm_companion.ui.mmd_motion_mixer import MixedMotion, MotionLayer
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiMmdAvatarCommand,
    UiVmdBoneFrame,
    UiVmdMorphFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.pmx_pose import matrix_vector, rotation_matrix
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer


def tail_return_renderer(preview: bool) -> tuple[DensePmxWidget, UiVmdMotion]:
    renderer = renderer_for("両手後ろ")
    idle = UiVmdMotion(
        name="両手後ろ",
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0.2, 0, -0.2, 0.9591663)),),
    )
    renderer.set_preview(replace(renderer._preview, vmd_motion=idle))
    motion = UiVmdMotion(
        name="ot0510_やれやれ",
        frame_count=90,
        loop=False,
        morph_frames=(UiVmdMorphFrame("笑い", 0, 0.0), UiVmdMorphFrame("笑い", 90, 1.0)),
        bone_frames=(
            UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0, 0, math.sin(-0.2), math.cos(-0.2))),
            UiVmdBoneFrame("左ひじ", 0, (0, 0, 0), (0.2, 0, 0, 0.9797959)),
            UiVmdBoneFrame("左ひじ", 90, (0, 0, 0), (0, 0, 0, 1)),
            UiVmdBoneFrame("頭", 0, (0, 0, 0), (0, 0, 0, 1)),
            UiVmdBoneFrame("頭", 90, (0, 0.1, 0), (0.3, 0, 0, 0.9539392)),
        ),
    )
    if preview:
        renderer.start_motion_preview(motion)
    else:
        renderer.set_mmd_command(
            UiMmdAvatarCommand(
                "neutral",
                0.6,
                "",
                "user",
                "頭",
                "shrug",
                "頭",
                gesture_motion=motion,
                gesture_motion_type="reaction",
                gesture_body_mask=("full_body",),
            )
        )
        renderer.begin_conversation_motion()
    return renderer, motion


@pytest.mark.parametrize("preview", [False, True])
def test_tail_return_starts_before_end_and_keeps_body_timeline(preview: bool) -> None:
    renderer, motion = tail_return_renderer(preview)
    advance = (
        (lambda delta: renderer.advance_heavy_animation(delta, True))
        if preview
        else renderer.advance_conversation_animation
    )
    raw_player = VmdMotionPlayer()
    raw_player.reset(motion)
    for delta in (1.0, 1.0, 0.59):
        advance(delta)
        raw_player.advance_motion(renderer._preview, motion, delta, True)
    assert renderer._tail_return_motion is None
    advance(0.21)
    raw, translations, morphs = raw_player.advance_motion(renderer._preview, motion, 0.21, True)
    active = (
        renderer.is_motion_preview_active if preview else renderer.is_conversation_motion_active
    )
    assert active
    assert renderer._idle_transition_from is None
    assert renderer._tail_return_elapsed == pytest.approx(0.2)
    controlled = renderer._preview_vmd_rotations if preview else renderer._gesture_vmd_rotations
    assert controlled[0] != pytest.approx(raw[0])
    assert controlled[4] == pytest.approx(raw[4])
    controlled_translations = (
        renderer._preview_vmd_translations if preview else renderer._gesture_vmd_translations
    )
    assert controlled_translations[4] == pytest.approx(translations[4])
    controlled_morphs = (
        renderer._preview_vmd_morph_weights if preview else renderer._gesture_vmd_morph_weights
    )
    assert controlled_morphs == pytest.approx(morphs)
    # 腕だけは合成器で再度遅らせず、通常会話でも同じ軌道にする。
    assert renderer._pose_settings()[0][0] == pytest.approx(controlled[0])
    advance(0.2)
    assert not renderer.is_motion_preview_active
    assert not renderer.is_conversation_motion_active
    assert renderer._idle_return_tail[2] == pytest.approx(0.4)
    assert renderer._idle_transition_elapsed == 0.0
    before = renderer._pose_settings()[0][0]
    renderer.advance_heavy_animation(0.05, True)
    after = renderer._pose_settings()[0][0]
    assert math.dist(before, after) > 0.005
    assert not renderer._tail_return_rotations
    renderer.close()


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("stop", ["interrupt", "manual", "next"])
def test_tail_return_does_not_leak_into_manual_stop_or_next_motion(
    preview: bool, stop: str
) -> None:
    renderer, motion = tail_return_renderer(preview)
    for delta in (1.0, 1.0, 0.8):
        if preview:
            renderer.advance_heavy_animation(delta, True)
        else:
            renderer.advance_conversation_animation(delta)
    before = renderer._pose_settings()[0][0]
    assert renderer._tail_return_rotations
    if stop == "next":
        renderer.start_motion_preview(replace(motion, name="ot0510_yes"))
    elif stop == "manual" and preview:
        renderer.stop_motion_preview()
    else:
        renderer.interrupt_motion()
    assert not renderer._tail_return_rotations
    assert renderer._idle_return_tail is None
    if stop != "next":
        assert renderer._vmd_rotations[0] == pytest.approx(before)
    renderer.close()


@pytest.mark.parametrize(
    "name,loop",
    [
        ("ot0510_yes", False),
        ("ot0510_no", False),
        ("ot0510_待機_元気_補正", True),
        ("ot0510_待機_猫背_補正", True),
        ("未調整", False),
        ("ot0510_やれやれ", True),
    ],
)
def test_tail_return_preserves_excluded_motions(name: str, loop: bool) -> None:
    renderer, motion = tail_return_renderer(True)
    excluded = replace(motion, name=name, loop=loop)
    player = VmdMotionPlayer()
    player.reset(excluded)
    for delta in (1.0, 1.0, 0.8):
        rotations, _, _ = player.advance_motion(renderer._preview, excluded, delta, True)
    before = dict(rotations)
    renderer._blend_motion_tail(excluded, player, rotations)
    assert rotations == before
    assert renderer._tail_return_motion is None
    renderer.close()


@pytest.mark.parametrize("name", list(_ARM_RETURN_SECONDS))
@pytest.mark.parametrize("bent", [False, True])
def test_all_return_profiles_keep_position_and_velocity_across_original_end(name: str, bent: bool):
    renderer, original = tail_return_renderer(True)
    motion = replace(
        original,
        name=name,
        bone_frames=tuple(
            replace(frame, rotation=(0.3, 0, 0, 0.9539392))
            if bent and frame.bone_name == "左ひじ" and frame.frame_number == 90
            else frame
            for frame in original.bone_frames
        ),
    )
    player = VmdMotionPlayer()
    player.reset(motion)
    epsilon = 0.00001
    for delta in (1.0, 1.0, 1.0 - epsilon):
        before, _, _ = player.advance_motion(renderer._preview, motion, delta, True)
    renderer._blend_motion_tail(motion, player, before)
    terminal, _, _ = player.advance_motion(renderer._preview, motion, epsilon, True)
    renderer._blend_motion_tail(motion, player, terminal)
    indexes = set(renderer._tail_return_rotations)
    assert indexes
    renderer._switch_idle_motion(
        renderer._default_idle_motion, source=MixedMotion(rotations=terminal), source_motion=motion
    )
    for index in indexes:
        assert renderer._vmd_rotations[index] == pytest.approx(terminal[index], abs=1e-9)
    renderer._vmd_rotations, _, _ = renderer._vmd_player.advance_motion(
        renderer._preview, renderer._default_idle_motion, epsilon, True
    )
    renderer._blend_idle_transition(epsilon)
    for index in indexes:
        incoming = tuple(
            (b - a) / epsilon for a, b in zip(before[index], terminal[index], strict=True)
        )
        outgoing = tuple(
            (b - a) / epsilon
            for a, b in zip(terminal[index], renderer._vmd_rotations[index], strict=True)
        )
        assert incoming == pytest.approx(outgoing, abs=0.001)
    renderer.close()


@pytest.mark.parametrize("variant", ["右手腰あて", "左手腰あて", "元気", "猫背"])
def test_idle_variant_also_keeps_tail_clock(variant: str):
    renderer, motion = tail_return_renderer(True)
    renderer._clear_motion_preview()
    renderer._idle_variant_active = True
    motion = replace(motion, name=f"ot0510_待機_{variant}_まばたき付き")
    renderer._preview = replace(renderer._preview, idle_motions=(motion,))
    renderer._switch_idle_motion(motion)
    for _ in range(28):
        renderer.advance_heavy_animation(0.1, True)
    assert renderer._tail_return_elapsed == pytest.approx(0.2)
    assert renderer._idle_variant_active
    assert renderer._idle_transition_from is None
    for _ in range(3):
        renderer.advance_heavy_animation(0.1, True)
    assert not renderer._idle_variant_active
    assert renderer._idle_return_tail[2] == pytest.approx(0.4)
    renderer.close()


@pytest.mark.parametrize("variant", ["元気", "猫背"])
def test_low_arm_tail_preserves_route_speed_and_exact_display(variant: str):
    renderer, original = tail_return_renderer(True)
    motion = replace(original, name=f"ot0510_待機_{variant}_補正")
    renderer.start_motion_preview(motion)
    for _ in range(29):
        renderer.advance_heavy_animation(0.1, True)
    assert renderer._tail_return_motion is motion
    assert renderer._tail_return_elapsed == pytest.approx(0.3)
    expected = renderer._sample_return_arms(
        renderer._tail_return_rotations,
        renderer._tail_return_target,
        0.4,
        1.2,
        renderer._low_arm_waypoint(),
        False,
        0.25,
    )
    renderer.advance_heavy_animation(0.1, True)
    assert renderer._idle_return_tail[2] == pytest.approx(0.4)
    assert renderer._idle_return_arm_seconds == 1.2
    assert not renderer._idle_return_distance_timing
    assert renderer._idle_return_forearm_split == 0.25
    source, target, offset = renderer._idle_return_tail
    for elapsed in (0.0, 0.05, 0.3, 0.8):
        if elapsed:
            renderer.advance_heavy_animation(elapsed - renderer._idle_transition_elapsed, True)
            expected = renderer._sample_return_arms(
                source, target, elapsed + offset, 1.2, renderer._low_arm_waypoint(), False, 0.25
            )
        displayed = renderer._pose_settings()[0]
        for index in source:
            assert displayed[index] == pytest.approx(expected[index])
        # 頭は腕用時計へ前倒しせず、従来の2.4秒で復帰する。
        progress = elapsed / 2.4
        weight = progress * progress * (3 - 2 * progress)
        head = renderer._idle_transition_from.rotations[4][0]
        assert renderer._vmd_rotations[4][0] == pytest.approx(head * (1 - weight))
    renderer.close()


def test_conversation_tail_entry_continues_original_display_filter(monkeypatch):
    renderer, _ = tail_return_renderer(False)
    reference, _ = tail_return_renderer(False)
    monkeypatch.setattr(reference, "_blend_motion_tail", lambda *args: None)
    for delta in (1.0, 1.0, 0.5999, 0.0002):
        renderer.advance_conversation_animation(delta)
        reference.advance_conversation_animation(delta)
    assert renderer._tail_return_elapsed == pytest.approx(0.0001)
    # 重ね始めは元の表示と一致し、未補間のVMDへ飛ばない。
    for index in (0, 2):
        assert renderer._motion_mixer.current.rotations[index] == pytest.approx(
            reference._motion_mixer.current.rotations[index], abs=1e-7
        )
    renderer.close()
    reference.close()


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.5])
def test_scaled_gesture_is_not_overridden_by_exact_tail_path(scale: float):
    renderer, _ = tail_return_renderer(False)
    renderer._command = replace(renderer._command, gesture_intensity_scale=scale)
    for delta in (1.0, 1.0, 0.8):
        renderer.advance_conversation_animation(delta)
    assert renderer._tail_return_motion is None
    renderer.close()


def test_disabling_motion_clears_tail_without_another_timer_tick():
    from multi_llm_companion.ui.pmx_preview_widget import PmxPreviewWidget

    renderer, motion = tail_return_renderer(True)
    wrapper = PmxPreviewWidget()
    wrapper.set_preview(renderer._preview)
    renderer.close()
    wrapper.start_motion_preview(motion)
    dense = wrapper._dense_view
    for delta in (1.0, 1.0, 0.8):
        dense.advance_heavy_animation(delta, True)
    assert dense._tail_return_rotations
    before = dense._motion_mixer.current.rotations[0]
    wrapper.set_motion_enabled(False)
    assert not wrapper._heavy_timer.isActive()
    assert not dense.is_motion_preview_active
    assert dense._tail_return_motion is None
    assert dense._idle_return_tail is None
    assert dense._idle_transition_from.rotations[0] == pytest.approx(before)
    wrapper.close()


def renderer_for(name: str) -> DensePmxWidget:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="復帰経路テスト",
            vertex_count=0,
            face_count=0,
            points=(),
            bones=tuple(
                UiAvatarBone(bone, -1, (0.0, 0.0, 0.0))
                for bone in (
                    "左腕",
                    "右腕",
                    "左ひじ",
                    "右ひじ",
                    "頭",
                    "首",
                    "上半身",
                    "左足",
                    "両目",
                    "左肩",
                    "右肩",
                    "左手首",
                    "右手首",
                )
            ),
            vmd_motion=UiVmdMotion(name=name, frame_count=30, bone_frames=()),
        )
    )
    return renderer


@pytest.mark.parametrize(
    "name,seconds",
    [
        ("ot0510_ドヤ（管理プレビュー）", 1.2),
        ("ot0510_喜びジャンプ", 1.4),
        ("ot0510_驚き_腕バタバタ_ループ", 1.6),
        ("ot0510_待機_右手腰あて_まばたき付き（権利確認済み）", 1.2),
        ("ot0510_待機_左手腰あて_まばたき付き_右袖余裕8度（権利確認済み）", 1.2),
        ("ot0510_待機_元気_補正（管理プレビュー）", 1.2),
        ("ot0510_待機_猫背_補正（管理プレビュー）", 1.2),
        ("未調整の動作", 2.2),
    ],
)
@pytest.mark.parametrize("route", ["preview", "preview_interrupt", "conversation", "interrupt"])
def test_each_return_route_preserves_the_source_motion_for_speed(
    name: str, seconds: float, route: str
) -> None:
    renderer = renderer_for("両手後ろ")
    motion = UiVmdMotion(name=name, frame_count=30, bone_frames=())
    if route.startswith("preview"):
        renderer.start_motion_preview(motion)
        if route == "preview":
            renderer.stop_motion_preview()
        else:
            renderer.interrupt_motion()
    else:
        renderer.set_mmd_command(
            UiMmdAvatarCommand(
                emotion="neutral",
                emotion_intensity=0.6,
                emotion_morph="",
                gaze="user",
                gaze_bone="頭",
                gesture="proud",
                gesture_target="頭",
                gesture_motion=motion,
                gesture_motion_type="reaction",
            )
        )
        renderer.begin_conversation_motion()
        if route == "conversation":
            renderer._finish_managed_motion()
        else:
            renderer.interrupt_motion()
    assert renderer._idle_motion is renderer._default_idle_motion
    assert renderer._idle_return_arm_seconds == seconds
    assert renderer._idle_return_open_rotations
    assert renderer._idle_return_distance_timing is (
        "元気" not in name and "猫背" not in name and "未調整" not in name
    )
    renderer.close()


@pytest.mark.parametrize("side,seconds", [("右", 1.2), ("左", 1.2)])
@pytest.mark.parametrize("route", ["direct", "finished", "speaking"])
def test_idle_rotation_return_uses_its_individual_speed(
    side: str, seconds: float, route: str
) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(
        name=f"ot0510_待機_{side}手腰あて_まばたき付き",
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("右腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
    )
    renderer._idle_variant_active = True
    if route == "direct":
        renderer._return_to_default_idle()
    else:
        motion = replace(renderer._idle_motion, frame_count=30, loop=False)
        renderer._idle_motion = motion
        renderer._preview = replace(renderer._preview, idle_motions=(motion,))
        renderer._vmd_player.advance_motion(
            renderer._preview, motion, 1.0 if route == "finished" else 0.5, True
        )
        assert renderer._vmd_player.is_finished is (route == "finished")
        renderer._speaking = route == "speaking"
        renderer._advance_idle_rotation(0.01)
    assert renderer._idle_motion is renderer._default_idle_motion
    assert renderer._idle_return_arm_seconds == seconds
    assert renderer._idle_return_open_rotations
    assert not renderer._idle_variant_active
    renderer.close()


@pytest.mark.parametrize("seconds", [1.2, 1.4, 1.6])
def test_individual_return_duration_preserves_nonzero_waypoint_velocity(seconds: float) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion, source=MixedMotion(rotations={0: (0.0, 0.4, -0.5)})
    )
    renderer._idle_return_arm_seconds = seconds
    poses = []
    step = 0.0001
    for moment in (seconds * 0.45 - step, seconds * 0.45, seconds * 0.45 + step):
        renderer._vmd_rotations = {0: (0.0, -0.4, -0.6)}
        renderer._blend_idle_transition(moment - renderer._idle_transition_elapsed)
        poses.append(renderer._vmd_rotations[0])
    before = tuple((b - a) / step for a, b in zip(poses[0], poses[1], strict=True))
    after = tuple((b - a) / step for a, b in zip(poses[1], poses[2], strict=True))
    assert before == pytest.approx(after, abs=0.002)
    assert abs(before[1]) > 0.49
    renderer.close()


def test_unmatched_motion_keeps_existing_return_speed() -> None:
    assert _arm_return_seconds(None) == 2.2
    assert (
        _arm_return_seconds(UiVmdMotion(name="ot0510_未確認", frame_count=30, bone_frames=()))
        == 2.2
    )


@pytest.mark.parametrize(
    ("name", "pitch"),
    [
        ("ot0510_笑い_口あて", -0.3),
        ("ot0510_驚き_腕バタバタ_ループ", -0.3),
        ("ot0510_ひらめく", 0.0),
        ("ot0510_待機_猫背_補正", 0.0),
    ],
)
def test_back_clearance_only_changes_selected_return_waypoints(name: str, pitch: float) -> None:
    renderer, original = tail_return_renderer(True)
    motion = replace(original, name=name)
    waypoint = renderer._low_arm_waypoint(motion)
    for side, sign in (("左", -1), ("右", 1)):
        index = renderer._bone_indexes_by_name[side + "腕"]
        assert waypoint[index] == pytest.approx((pitch, 0.0, sign * 0.4))
    renderer._switch_idle_motion(renderer._default_idle_motion, source_motion=motion)
    assert renderer._idle_return_open_rotations == waypoint
    assert renderer._idle_return_arm_seconds == _arm_return_seconds(motion)
    renderer.close()


def test_nearby_clearance_pose_does_not_consume_half_the_return() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(name="ot0510_やれやれ", frame_count=90, bone_frames=())
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: (0.0, 0.0, -0.33), 2: (0.0, 0.0, 0.0)}),
    )
    end = {0: (-0.402, -0.337, -0.574), 2: (0.446, 0.539, 0.125)}
    poses = []
    for _ in range(1, 41):
        renderer._vmd_rotations = dict(end)
        renderer._blend_idle_transition(0.01)
        poses.append(renderer._vmd_rotations[0])
    # 近い通過点のために前半を費やさず、0.2秒時点で後方へ進む。
    assert poses[19][0] < -0.02
    assert poses[19][1] < -0.01
    # 前半を全走査し、始動直後を除いてほぼ静止する区間を作らない。
    for before, after in zip(poses[9:-1], poses[10:], strict=True):
        assert sum((b - a) ** 2 for a, b in zip(before, after, strict=True)) ** 0.5 > 0.002
    renderer.close()


@pytest.mark.parametrize("seconds", [1.2, 1.4, 1.6])
def test_near_clearance_return_starts_backwards_without_another_outward_detour(
    seconds: float,
) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(name="ot0510_やれやれ", frame_count=90, bone_frames=())
    start = (0.0, 0.0, -0.33)
    end = (-0.402, -0.337, -0.574)
    renderer._switch_idle_motion(
        renderer._default_idle_motion, source=MixedMotion(rotations={0: start, 2: (0, 0, 0)})
    )
    renderer._idle_return_arm_seconds = seconds
    previous = start
    for _ in range(round(seconds * 100)):
        renderer._vmd_rotations = {0: end, 2: (0.446, 0.539, 0.125)}
        renderer._blend_idle_transition(0.01)
        current = renderer._vmd_rotations[0]
        # 最初の10msから後方へ進み、前方への折り返しを挟まない。
        assert current[0] < previous[0]
        assert current[1] < previous[1]
        previous = current
    assert previous == pytest.approx(end)
    renderer.close()


def test_distance_timed_return_is_not_smoothed_again_before_display() -> None:
    renderer = renderer_for("両手後ろ")
    renderer.set_preview(
        replace(
            renderer._preview,
            vmd_motion=UiVmdMotion(
                name="両手後ろ",
                frame_count=90,
                bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
            ),
        )
    )
    renderer._idle_motion = UiVmdMotion(name="ot0510_やれやれ", frame_count=90, bone_frames=())
    source = MixedMotion(rotations={0: (0.0, 0.0, -0.33), 6: (0.2, 0, 0)})
    renderer._motion_mixer.mix(
        delta_seconds=1.0, vmd=MotionLayer(rotations=source.rotations, exact=True)
    )
    renderer._switch_idle_motion(renderer._default_idle_motion, source=source)
    renderer._vmd_rotations = {0: (-0.402, -0.337, -0.574)}
    renderer._blend_idle_transition(0.2)
    expected = dict(renderer._vmd_rotations)
    renderer._next_mix_delta_seconds = 0.05
    actual = renderer._pose_settings()[0]
    assert actual[0] == pytest.approx(expected[0])
    assert actual[6] == pytest.approx(expected[6])
    assert renderer._motion_mixer.current.rotations[0] == pytest.approx(expected[0])
    renderer.close()


@pytest.mark.parametrize("seconds", [1.2, 1.4, 1.6])
@pytest.mark.parametrize("start", [(0.0, 0.0, -0.33), (0.4, 0.7, -0.9)])
def test_distance_timed_waypoint_has_matching_velocity(seconds: float, start) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(name="ot0510_やれやれ", frame_count=90, bone_frames=())
    renderer._switch_idle_motion(
        # 上腕が通過点に近くても、曲がった肘には従来の迂回が必要。
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: start, 2: (0.4, 0, 0)}),
    )
    renderer._idle_return_arm_seconds = seconds
    end, opened = (-0.402, -0.337, -0.574), (0, 0, -0.4)
    incoming = sum((a - b) ** 2 for a, b in zip(start, opened, strict=True)) ** 0.5
    outgoing = sum((a - b) ** 2 for a, b in zip(opened, end, strict=True)) ** 0.5
    split = max(0.08, min(0.45, incoming / (incoming + outgoing)))
    poses = []
    step = 0.00001
    for moment in (seconds * split - step, seconds * split, seconds * split + step):
        renderer._vmd_rotations = {0: end}
        renderer._blend_idle_transition(moment - renderer._idle_transition_elapsed)
        poses.append(renderer._vmd_rotations[0])
    before = tuple((b - a) / step for a, b in zip(poses[0], poses[1], strict=True))
    after = tuple((b - a) / step for a, b in zip(poses[1], poses[2], strict=True))
    assert poses[1] == pytest.approx(opened)
    assert before == pytest.approx(after, abs=0.002)
    assert sum(value * value for value in before) ** 0.5 > 0.5
    renderer.close()


def test_return_keeps_arm_and_elbow_on_the_same_clearance_clock() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(name="ot0510_やれやれ", frame_count=90, bone_frames=())
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: (0.4, 0.7, -0.9), 2: (0, 0, 0)}),
    )
    renderer._vmd_rotations = {0: (-0.402, -0.337, -0.574), 2: (0.446, 0.539, 0.125)}
    renderer._blend_idle_transition(1.2 * 0.45)
    assert renderer._vmd_rotations[0] == pytest.approx((0, 0, -0.4))
    assert renderer._vmd_rotations[2] == pytest.approx((0, 0, 0))
    renderer.close()


def test_hands_behind_return_sweeps_low_arms_without_a_middle_hold() -> None:
    renderer = renderer_for("両手後ろ")
    default = renderer._default_idle_motion
    source = MixedMotion(
        rotations={
            0: (0.1, 0.2, -0.6),
            1: (0.1, -0.2, 0.6),
            2: (1.0, 0.0, 0.0),
            4: (0.7, 0.0, 0.0),
        },
        translations={4: (0.0, 1.0, 0.0)},
        morphs={"まばたき": 0.5},
    )
    renderer._switch_idle_motion(default, source=source)
    elapsed = 0.0
    previous = source.rotations[0]
    for moment in (0.1, 0.4, 0.8, 1.2, 1.6, 2.0, 2.2, 2.39, 2.4):
        # 実更新と同じく、凍結中のVMDから目標姿勢を毎回読み直す。
        renderer._vmd_rotations = {}
        renderer._vmd_translations = {}
        renderer._vmd_morph_weights = {}
        renderer._blend_idle_transition(moment - elapsed)
        elapsed = moment
        current = renderer._vmd_rotations[0]
        if moment <= 2.2:
            assert current != pytest.approx(previous)
        else:
            assert current == pytest.approx((0, 0, 0))
        previous = current
        if moment < 2.4:
            if moment < 2.2:
                assert -0.6 < current[2] < 0.0
            else:
                assert current == pytest.approx((0, 0, 0))
            assert 0.0 < renderer._vmd_rotations[4][0] < 0.7
            assert renderer._idle_transition_from is not None
        else:
            assert all(
                value == pytest.approx((0, 0, 0)) for value in renderer._vmd_rotations.values()
            )
            assert renderer._vmd_translations[4] == pytest.approx((0, 0, 0))
            assert renderer._vmd_morph_weights["まばたき"] == 0
            assert renderer._idle_transition_from is None
            assert renderer._idle_return_open_rotations == {}
    assert renderer._default_idle_motion is default
    assert default.bone_frames == ()
    renderer.close()


def test_faster_arm_return_keeps_body_timing_and_follows_the_live_target() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(
            rotations={0: (0.2, 0.3, -0.5), 4: (0.6, 0, 0)},
            translations={4: (0, 1, 0)},
            morphs={"まばたき": 0.5},
        ),
    )
    for moment in (2.2, 2.3, 2.39, 2.4):
        target_arm = (0.1 * moment, 0.2, -0.6)
        renderer._vmd_rotations = {0: target_arm}
        renderer._vmd_translations = {}
        renderer._vmd_morph_weights = {}
        renderer._blend_idle_transition(moment - renderer._idle_transition_elapsed)
        assert renderer._vmd_rotations[0] == pytest.approx(target_arm)
        progress = moment / 2.4
        remaining = 1.0 - progress * progress * (3.0 - 2.0 * progress)
        assert renderer._vmd_rotations[4][0] == pytest.approx(0.6 * remaining)
        assert renderer._vmd_translations[4][1] == pytest.approx(remaining)
        assert renderer._vmd_morph_weights["まばたき"] == pytest.approx(0.5 * remaining)
        assert (renderer._idle_transition_from is not None) is (moment < 2.4)
    renderer.close()


@pytest.mark.parametrize("variant", ["元気", "猫背"])
@pytest.mark.parametrize("returning", [False, True])
def test_low_arm_variant_finishes_arms_in_1_2_seconds_and_keeps_following(
    variant: str, returning: bool
) -> None:
    renderer = renderer_for("両手後ろ")
    variant_motion = UiVmdMotion(
        name=f"ot0510_待機_{variant}_補正", frame_count=600, bone_frames=()
    )
    target = variant_motion
    if returning:
        renderer._idle_motion = variant_motion
        target = renderer._default_idle_motion
    renderer._switch_idle_motion(
        target,
        source=MixedMotion(rotations={0: (0.2, 0.3, -0.5), 4: (0.6, 0, 0)}),
    )
    assert renderer._idle_return_arm_seconds == 1.2
    poses = []
    for moment in (0.5399, 0.54, 0.5401, 1.2, 1.3, 2.4):
        target_arm = (0.1, -0.2, -0.6) if moment < 1.2 else (moment * 0.1, -0.2, -0.6)
        renderer._vmd_rotations = {0: target_arm}
        renderer._vmd_translations = {}
        renderer._vmd_morph_weights = {}
        renderer._blend_idle_transition(moment - renderer._idle_transition_elapsed)
        if moment < 1.2:
            poses.append(renderer._vmd_rotations[0])
        else:
            assert renderer._vmd_rotations[0] == pytest.approx(target_arm)
        progress = moment / 2.4
        remaining = 1.0 - progress * progress * (3.0 - 2.0 * progress)
        assert renderer._vmd_rotations[4][0] == pytest.approx(0.6 * remaining)
        assert (renderer._idle_transition_from is not None) is (moment < 2.4)
    before = tuple((b - a) / 0.0001 for a, b in zip(poses[0], poses[1], strict=True))
    after = tuple((b - a) / 0.0001 for a, b in zip(poses[1], poses[2], strict=True))
    assert before == pytest.approx(after, abs=0.002)
    assert sum(value * value for value in before) ** 0.5 > 0.37
    renderer.close()


def test_arm_sweep_has_continuous_nonzero_velocity_at_the_middle() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: (0.0, 0.4, -0.5)}),
    )
    poses = []
    elapsed = 0.0
    for moment in (0.989, 0.99, 0.991):
        renderer._vmd_rotations = {0: (0.0, -0.4, -0.6)}
        renderer._blend_idle_transition(moment - elapsed)
        elapsed = moment
        poses.append(renderer._vmd_rotations[0])
    before = tuple((b - a) / 0.001 for a, b in zip(poses[0], poses[1], strict=True))
    after = tuple((b - a) / 0.001 for a, b in zip(poses[1], poses[2], strict=True))
    assert before == pytest.approx(after, abs=0.002)
    assert abs(before[1]) > 0.49
    assert poses[1] == pytest.approx((0.0, 0.0, -0.4))
    renderer.close()


def test_other_idle_starts_transition_without_initial_hold() -> None:
    renderer = renderer_for("別の待機")
    renderer._switch_idle_motion(
        renderer._default_idle_motion, source=MixedMotion(rotations={0: (0.5, 0, 0)})
    )
    assert renderer._idle_return_open_rotations == {}
    renderer._vmd_rotations = {}
    renderer._blend_idle_transition(0.1)
    assert 0.0 < renderer._vmd_rotations[0][0] < 0.5
    renderer._vmd_rotations = {}
    renderer._blend_idle_transition(2.3)
    assert renderer._idle_transition_from is None
    assert renderer._vmd_rotations[0] == (0, 0, 0)
    renderer.close()


@pytest.mark.parametrize("xy", [(0.0, 0.0), (0.002, -0.003)])
def test_arm_sweep_does_not_stall_when_start_and_end_have_similar_fore_aft_angles(xy) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: (0.0, 0.0, -0.8)}),
    )
    poses = []
    elapsed = 0.0
    # 高速化で差分の加速度誤差が増えるため、接点近傍を細かく測る。
    step = 0.0001
    for moment in (0.99 - step, 0.99, 0.99 + step):
        renderer._vmd_rotations = {0: (*xy, -0.7)}
        renderer._blend_idle_transition(moment - elapsed)
        elapsed = moment
        poses.append(renderer._vmd_rotations[0])
    before = tuple((b - a) / step for a, b in zip(poses[0], poses[1], strict=True))
    after = tuple((b - a) / step for a, b in zip(poses[1], poses[2], strict=True))
    assert before == pytest.approx(after, abs=0.002)
    assert sum(value * value for value in before) ** 0.5 > 0.37
    assert poses[1] == pytest.approx((0.0, 0.0, -0.4))
    renderer.close()


@pytest.mark.parametrize(
    "start_z,end_z", [(-0.58, -0.48), (-0.48, -0.58), (-0.58, -0.06), (-0.06, -0.58)]
)
def test_low_arm_waypoint_keeps_the_projected_hand_moving(start_z: float, end_z: float) -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={0: (-0.4, -0.34, start_z)}),
    )
    positions = []
    elapsed = 0.0
    for moment in (0.989, 0.99, 0.991):
        renderer._vmd_rotations = {0: (0.03, 0.001, end_z)}
        renderer._blend_idle_transition(moment - elapsed)
        elapsed = moment
        # 伸びた腕が体の正面平面にある時も、回転値だけでなく手先が動く。
        positions.append(matrix_vector(rotation_matrix(*renderer._vmd_rotations[0]), (1, -1, 0)))
    before = tuple((b - a) / 0.001 for a, b in zip(positions[0], positions[1], strict=True))
    after = tuple((b - a) / 0.001 for a, b in zip(positions[1], positions[2], strict=True))
    assert before == pytest.approx(after, abs=0.005)
    assert (before[0] ** 2 + before[1] ** 2) ** 0.5 > 0.3
    renderer.close()


def test_switching_away_clears_the_pending_arm_opening() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(renderer._default_idle_motion)
    assert renderer._idle_return_open_rotations
    other = UiVmdMotion(name="猫背", frame_count=30, bone_frames=())
    renderer._switch_idle_motion(other)
    assert renderer._idle_return_open_rotations == {}
    renderer.set_preview(None)
    assert renderer._idle_return_open_rotations == {}
    renderer.close()


def test_elbow_and_wrist_clear_early_then_move_through_the_arm_waypoint() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._idle_motion = UiVmdMotion(
        name="ot0510_待機_猫背_補正", frame_count=600, bone_frames=()
    )
    renderer._switch_idle_motion(
        renderer._default_idle_motion,
        source=MixedMotion(rotations={2: (0.2, 0.1, 0.0), 11: (0.1, 0.2, 0.3)}),
    )
    assert 2 in renderer._idle_return_open_rotations
    assert 11 in renderer._idle_return_open_rotations
    renderer._vmd_rotations = {2: (0.8, 0.5, 0.2), 11: (0.3, 0.4, 0.5)}
    renderer._blend_idle_transition(0.3)
    assert renderer._vmd_rotations[2] == pytest.approx((0, 0, 0))
    assert renderer._vmd_rotations[11] == pytest.approx((0, 0, 0))
    renderer._vmd_rotations = {2: (0.8, 0.5, 0.2), 11: (0.3, 0.4, 0.5)}
    renderer._blend_idle_transition(0.24)
    before = renderer._vmd_rotations[2]
    assert before != pytest.approx((0, 0, 0))
    assert renderer._vmd_rotations[11] != pytest.approx((0, 0, 0))
    renderer._vmd_rotations = {2: (0.8, 0.5, 0.2), 11: (0.3, 0.4, 0.5)}
    renderer._blend_idle_transition(0.001)
    assert all(b > a for a, b in zip(before, renderer._vmd_rotations[2], strict=True))
    renderer.close()


def test_other_reactions_keep_the_slower_forearm_clearance() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._switch_idle_motion(
        renderer._default_idle_motion, source=MixedMotion(rotations={2: (1.2, 0.4, 0.0)})
    )
    assert renderer._idle_return_forearm_split == 0.45
    renderer._vmd_rotations = {2: (0.4, 0.5, 0.2)}
    renderer._blend_idle_transition(0.55)
    assert renderer._vmd_rotations[2] != pytest.approx((0, 0, 0))
    renderer._vmd_rotations = {2: (0.4, 0.5, 0.2)}
    renderer._blend_idle_transition(0.44)
    assert renderer._vmd_rotations[2] == pytest.approx((0, 0, 0))
    renderer.close()


def test_energetic_idle_entry_uses_opening_without_changing_default() -> None:
    renderer = renderer_for("両手後ろ")
    default = renderer._default_idle_motion
    energetic = UiVmdMotion(name="ot0510_待機_元気_まばたき付き", frame_count=600, bone_frames=())
    renderer._switch_idle_motion(energetic)
    assert renderer._idle_return_open_rotations
    assert renderer._idle_motion is energetic
    assert renderer._default_idle_motion is default
    renderer.close()


@pytest.mark.parametrize(
    "name", ["ot0510_ひらめく", "ot0510_考える_ループ", "ot0510_苦笑い_ループ"]
)
@pytest.mark.parametrize("preview", [False, True])
def test_skirt_safe_entry_keeps_vmd_clock_and_existing_return(name: str, preview: bool) -> None:
    renderer, original = tail_return_renderer(preview)
    motion = replace(original, name=name)
    if preview:
        renderer.start_motion_preview(motion)

        def advance(dt):
            renderer.advance_heavy_animation(dt, True)

        player = renderer._preview_vmd_player
    else:
        renderer.set_mmd_command(replace(renderer._command, gesture_motion=motion))
        renderer.begin_conversation_motion()
        advance = renderer.advance_conversation_animation
        player = renderer._gesture_vmd_player
    assert renderer._idle_return_open_rotations
    assert renderer._idle_return_arm_seconds == 0.65
    advance(0.4)
    assert renderer._idle_transition_from is not None
    advance(0.25)
    assert player._elapsed_seconds == pytest.approx(0.65)
    assert renderer._vmd_player._elapsed_seconds == pytest.approx(0.65)
    assert renderer._idle_transition_from is None
    assert renderer._default_idle_motion.name == "両手後ろ"
    renderer._switch_idle_motion(renderer._default_idle_motion, source_motion=motion)
    assert renderer._idle_return_arm_seconds == 1.2
    assert renderer._idle_return_distance_timing
    assert renderer._idle_return_forearm_split == 0.45
    renderer.close()


@pytest.mark.parametrize(
    "name", ["ot0510_待機_猫背_補正", "ot0510_待機_元気_まばたき付き", "ot0510_no", "ot0510_yes"]
)
def test_preview_uses_arm_entry_only_for_catback_and_energetic(name: str) -> None:
    renderer = renderer_for("両手後ろ")
    default = renderer._default_idle_motion
    motion = UiVmdMotion(
        name=name,
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
    )
    renderer.start_motion_preview(motion)
    agreement = name in {"ot0510_no", "ot0510_yes"}
    assert bool(renderer._idle_return_open_rotations) is not agreement
    for step in range(3):
        renderer.advance_heavy_animation(0.8, True)
        expected = (step + 1) * 0.8
        assert renderer._preview_vmd_player._elapsed_seconds == pytest.approx(expected)
        if agreement:
            assert renderer._preview_vmd_rotations == {}
            assert renderer._idle_motion is default
        else:
            assert renderer._preview_vmd_rotations == renderer._vmd_rotations
    assert renderer._idle_transition_from is None
    renderer.advance_heavy_animation(0.1, True)
    assert renderer._preview_vmd_player._elapsed_seconds == pytest.approx(2.5)
    renderer.stop_motion_preview()
    assert renderer._idle_motion is default
    assert not renderer._idle_variant_active
    assert renderer._default_idle_motion is default
    renderer.close()


@pytest.mark.parametrize("suffix", ["", "_右袖余裕8度（管理プレビュー）"])
def test_left_waist_preview_blends_into_pose_without_extra_arm_opening(suffix: str) -> None:
    renderer = renderer_for("両手後ろ")
    source = (0.0, 0.2, 0.5)
    source_rotations = {1: source, 10: (0.1, 0.2, 0.3)}
    renderer._vmd_rotations = source_rotations
    renderer._motion_mixer.mix(
        delta_seconds=0.35, idle=MotionLayer(rotations=source_rotations, exact=True)
    )
    motion = UiVmdMotion(
        name=f"ot0510_待機_左手腰あて_まばたき付き{suffix}",
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("右腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
    )
    renderer.start_motion_preview(motion)
    assert renderer._idle_motion is motion
    assert renderer._idle_transition_from is not None
    assert renderer._idle_return_open_rotations == {}
    assert renderer._preview_vmd_rotations[1] == pytest.approx(source)
    assert renderer._preview_vmd_rotations[10] == pytest.approx(source_rotations[10])
    for step in range(6):
        renderer.advance_heavy_animation(0.4, True)
        assert renderer._preview_vmd_player._elapsed_seconds == pytest.approx((step + 1) * 0.4)
        assert renderer._preview_vmd_rotations == renderer._vmd_rotations
    assert renderer._idle_transition_from is None
    renderer.advance_heavy_animation(0.1, True)
    assert renderer._preview_vmd_player._elapsed_seconds == pytest.approx(2.5)
    # 056にキーがない肩へ、両手後ろの回転を基底として残さない。
    assert renderer._preview_vmd_rotations.get(10, (0, 0, 0)) == pytest.approx((0, 0, 0))
    renderer.stop_motion_preview()
    assert renderer._idle_motion is renderer._default_idle_motion
    assert renderer._idle_return_arm_seconds == 1.2
    renderer.close()


def test_switching_from_catback_preview_restores_the_default_idle_layer() -> None:
    renderer = renderer_for("両手後ろ")
    default = renderer._default_idle_motion
    renderer.start_motion_preview(
        UiVmdMotion(name="待機_猫背_補正", frame_count=600, bone_frames=())
    )
    renderer.start_motion_preview(UiVmdMotion(name="別の動作", frame_count=600, bone_frames=()))
    assert renderer._idle_motion is default
    assert not renderer._idle_variant_active
    renderer.close()


def test_preview_entry_captures_bones_owned_only_by_the_running_reaction() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._conversation_motion_active = True
    renderer._gesture_body_indexes = None
    renderer._gesture_vmd_rotations = {4: (0.7, 0.0, 0.0)}
    renderer._motion_mixer.mix(
        delta_seconds=0.35,
        gesture=MotionLayer(rotations={4: (0.7, 0.0, 0.0)}, exact=True),
    )
    renderer.start_motion_preview(
        UiVmdMotion(name="待機_猫背_補正", frame_count=30, bone_frames=())
    )
    assert renderer._idle_transition_from.rotations[4] == pytest.approx((0.7, 0.0, 0.0))
    assert not renderer.is_conversation_motion_active
    renderer.stop_motion_preview()
    assert renderer._idle_motion is renderer._default_idle_motion
    assert not renderer._idle_variant_active
    renderer.close()


@pytest.mark.parametrize("name", ["ot0510_no", "ot0510_yes"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_agreement_entry_preserves_motion_clock_and_default_idle(
    name: str, interrupt: bool
) -> None:
    renderer = renderer_for("両手後ろ")
    default = renderer._default_idle_motion
    motion = UiVmdMotion(
        name=name,
        frame_count=90,
        loop=False,
        bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
    )
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            emotion="neutral",
            emotion_intensity=0.6,
            emotion_morph="",
            gaze="user",
            gaze_bone="頭",
            gesture="nod",
            gesture_target="頭",
            gesture_motion=motion,
            gesture_motion_type="reaction",
            gesture_body_mask=("full_body",),
        )
    )
    renderer.begin_conversation_motion()
    assert renderer._idle_motion is default
    for step in range(3):
        renderer.advance_conversation_animation(0.8)
        assert renderer._gesture_vmd_player._elapsed_seconds == pytest.approx((step + 1) * 0.8)
        assert renderer._managed_gesture_pose() == ({}, {})
        assert not renderer._idle_return_open_rotations
        renderer.end_conversation_motion()
        assert renderer.is_conversation_motion_active
    assert renderer._idle_transition_from is None
    renderer.advance_conversation_animation(0.1)
    assert renderer._gesture_vmd_player._elapsed_seconds == pytest.approx(2.5)
    if interrupt:
        renderer.interrupt_motion()
    else:
        # 元の3秒を省略せず再生し、終端直後に保持なしで復帰する。
        renderer.advance_conversation_animation(0.49)
        assert renderer.is_conversation_motion_active
        assert renderer._gesture_vmd_player._elapsed_seconds == pytest.approx(2.99)
        renderer.advance_conversation_animation(0.02)
        assert not renderer.is_conversation_motion_active
        for _ in range(3):
            renderer.advance_conversation_animation(0.1)
    assert not renderer.is_conversation_motion_active
    assert renderer._idle_motion is default
    assert not renderer._idle_variant_active
    assert not renderer._idle_return_open_rotations
    assert renderer._agreement_return_from is not None
    renderer.close()


@pytest.mark.parametrize("stop_method", ["stop_motion_preview", "interrupt_motion"])
def test_preview_stop_uses_the_displayed_pose_and_same_return_path(stop_method: str) -> None:
    renderer = renderer_for("両手後ろ")
    renderer.start_motion_preview(UiVmdMotion(name="確認動作", frame_count=30, bone_frames=()))
    renderer._preview_vmd_rotations = {0: (0.5, 0.0, 0.0)}
    renderer._motion_mixer.mix(
        delta_seconds=0.35,
        gesture=MotionLayer(rotations={0: (0.5, 0.0, 0.0)}, exact=True),
    )
    getattr(renderer, stop_method)()
    assert not renderer.is_motion_preview_active
    assert renderer._idle_return_open_rotations
    assert renderer._idle_transition_from.rotations[0] == pytest.approx((0.5, 0.0, 0.0))
    renderer.close()


@pytest.mark.parametrize("preview", [False, True])
def test_agreement_head_moves_without_arm_entry_and_pause_freezes_both_clocks(
    preview: bool,
) -> None:
    renderer = renderer_for("両手後ろ")
    renderer.set_preview(
        replace(
            renderer._preview,
            vmd_motion=UiVmdMotion(
                name="両手後ろ",
                frame_count=600,
                bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0, 0, 0, 1)),),
            ),
        )
    )
    motion = UiVmdMotion(
        name="ot0510_yes",
        frame_count=90,
        loop=False,
        bone_frames=(
            UiVmdBoneFrame("頭", 0, (0, 0, 0), (0, 0, 0, 1)),
            UiVmdBoneFrame("頭", 30, (0, 0, 0), (0.19866933, 0, 0, 0.98006658)),
        ),
    )
    if preview:
        renderer.start_motion_preview(motion)

        def advance(delta: float) -> None:
            renderer.advance_heavy_animation(delta, True)

        player = renderer._preview_vmd_player
    else:
        renderer.set_mmd_command(
            UiMmdAvatarCommand(
                emotion="neutral",
                emotion_intensity=0.6,
                emotion_morph="",
                gaze="user",
                gaze_bone="頭",
                gesture="nod",
                gesture_target="頭",
                gesture_motion=motion,
                gesture_motion_type="reaction",
            )
        )
        renderer.begin_conversation_motion()
        advance = renderer.advance_conversation_animation
        player = renderer._gesture_vmd_player
    advance(1.0)
    advance(1.0)
    assert player._elapsed_seconds == pytest.approx(2.0)
    assert renderer._vmd_player._elapsed_seconds == pytest.approx(2.0)
    assert renderer._idle_transition_from is None
    displayed = renderer._preview_vmd_rotations if preview else renderer._gesture_vmd_rotations
    assert displayed[4][0] > 0.1
    renderer.set_lightweight_animation(0.0, False, False)
    if preview:
        renderer.advance_heavy_animation(0.5, False)
    else:
        renderer.advance_conversation_animation(0.5)
    assert player._elapsed_seconds == pytest.approx(2.0)
    assert renderer._vmd_player._elapsed_seconds == pytest.approx(2.0)
    renderer.close()


@pytest.mark.parametrize("name", ["ot0510_no", "ot0510_yes"])
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("finish", ["natural", "stop", "interrupt"])
@pytest.mark.parametrize("variant", [False, True])
def test_agreement_preserves_idle_body_through_playback_and_return(
    name: str, preview: bool, finish: str, variant: bool
) -> None:
    renderer = renderer_for("両手後ろ")
    idle = UiVmdMotion(
        name="両手後ろ",
        frame_count=600,
        bone_frames=tuple(
            UiVmdBoneFrame(bone, 0, (0.02, 0, 0), (0.19866933, 0, 0, 0.98006658))
            for bone in (
                "左腕",
                "右腕",
                "左ひじ",
                "右ひじ",
                "上半身",
                "左足",
                "頭",
                "首",
                "左肩",
                "右肩",
                "左手首",
                "右手首",
            )
        ),
    )
    renderer.set_preview(replace(renderer._preview, vmd_motion=idle))
    if variant:
        idle = replace(idle, name="待機_猫背_補正")
        renderer._idle_motion = idle
        renderer._idle_variant_active = True
        renderer._vmd_player.reset(idle)
    # 姿勢の切替ではなく、現在の待機時計を保つことも確認する。
    renderer.advance_heavy_animation(0.4, True)
    baseline = renderer._pose_settings()
    body_indexes = (0, 1, 2, 3, 6, 7, 9, 10, 11, 12)
    motion = UiVmdMotion(
        name=name,
        frame_count=30,
        loop=False,
        bone_frames=tuple(
            UiVmdBoneFrame(bone, frame, (0, 0.1, 0), rotation)
            for bone in (
                "左腕",
                "右腕",
                "左ひじ",
                "右ひじ",
                "頭",
                "首",
                "上半身",
                "左足",
                "両目",
                "左肩",
                "右肩",
                "左手首",
                "右手首",
            )
            for frame, rotation in (
                (0, (0, 0, 0, 1)),
                (30, (0.38941834, 0, 0, 0.92106099)),
            )
        ),
    )
    if preview:
        renderer.start_motion_preview(motion)

        def advance(delta: float) -> None:
            renderer.advance_heavy_animation(delta, True)
    else:
        renderer.set_mmd_command(
            UiMmdAvatarCommand(
                emotion="neutral",
                emotion_intensity=0.6,
                emotion_morph="",
                gaze="user",
                gaze_bone="頭",
                gesture="nod",
                gesture_target="頭",
                gesture_motion=motion,
                gesture_motion_type="reaction",
                gesture_body_mask=("full_body",),
            )
        )
        renderer.begin_conversation_motion()
        advance = renderer.advance_conversation_animation
    heads = []
    for _ in range(5 if finish != "natural" else 11):
        advance(0.1)
        pose = renderer._pose_settings()
        heads.append(pose[0].get(4))
        for index in body_indexes:
            assert pose[0].get(index) == pytest.approx(baseline[0][index])
            assert pose[2].get(index) == pytest.approx(baseline[2][index])
        assert renderer._idle_motion is idle
        assert renderer._idle_transition_from is None
        assert renderer._idle_return_open_rotations == {}
        if preview and renderer.is_motion_preview_active:
            assert set(renderer._preview_vmd_rotations) == {4, 5}
        elif renderer.is_conversation_motion_active:
            assert set(renderer._managed_gesture_pose()[0]) == {4, 5}
    assert heads[0] != heads[3]
    if finish == "interrupt":
        renderer.interrupt_motion()
    elif finish == "stop":
        if preview:
            renderer.stop_motion_preview()
        else:
            # 発話終了でも原モーションは完走させる。
            renderer.end_conversation_motion()
            assert renderer.is_conversation_motion_active
            for _ in range(6):
                advance(0.1)
                renderer._pose_settings()
    assert not renderer.is_motion_preview_active
    assert not renderer.is_conversation_motion_active
    for _ in range(5):
        renderer.advance_heavy_animation(0.1, True)
        pose = renderer._pose_settings()
        for index in body_indexes:
            assert pose[0].get(index) == pytest.approx(baseline[0][index])
            assert pose[2].get(index) == pytest.approx(baseline[2][index])
        assert not renderer._idle_return_open_rotations
    assert renderer._agreement_return_from is None
    assert renderer._vmd_player._elapsed_seconds > 1.3
    renderer.close()


@pytest.mark.parametrize("preview", [False, True])
def test_switching_no_to_yes_preserves_idle_clock_and_arm_pose(preview: bool) -> None:
    renderer = renderer_for("両手後ろ")
    idle = UiVmdMotion(
        name="両手後ろ",
        frame_count=600,
        bone_frames=(UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0.19866933, 0, 0, 0.98006658)),),
    )
    renderer.set_preview(replace(renderer._preview, vmd_motion=idle))
    arm = renderer._pose_settings()[0][0]
    for name in ("ot0510_no", "ot0510_yes", "ot0510_no"):
        motion = UiVmdMotion(
            name=name,
            frame_count=90,
            loop=False,
            bone_frames=(
                UiVmdBoneFrame("頭", 0, (0, 0, 0), (0.19866933, 0, 0, 0.98006658)),
                UiVmdBoneFrame("左腕", 0, (0, 0, 0), (0.38941834, 0, 0, 0.92106099)),
            ),
        )
        before = renderer._vmd_player._elapsed_seconds
        if preview:
            renderer.start_motion_preview(motion)
            renderer.advance_heavy_animation(0.3, True)
        else:
            renderer.set_mmd_command(
                UiMmdAvatarCommand(
                    "neutral",
                    0.5,
                    "",
                    "user",
                    "頭",
                    "nod",
                    "頭",
                    gesture_motion=motion,
                    gesture_motion_type="reaction",
                )
            )
            renderer.begin_conversation_motion()
            renderer.advance_conversation_animation(0.3)
        assert renderer._pose_settings()[0][0] == pytest.approx(arm)
        assert renderer._vmd_player._elapsed_seconds == pytest.approx(before + 0.3)
        assert not renderer._idle_return_open_rotations
        assert renderer._idle_transition_from is None
    renderer.interrupt_motion()
    renderer.advance_heavy_animation(0.5, True)
    assert renderer._pose_settings()[0][0] == pytest.approx(arm)
    assert renderer._agreement_return_from is None
    renderer.close()


def test_non_agreement_reaction_returns_without_an_added_end_hold() -> None:
    renderer = renderer_for("両手後ろ")
    motion = UiVmdMotion(
        name="既存の反応",
        frame_count=30,
        loop=False,
        bone_frames=(UiVmdBoneFrame("頭", 0, (0, 0, 0), (0, 0, 0, 1)),),
    )
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            emotion="neutral",
            emotion_intensity=0.6,
            emotion_morph="",
            gaze="user",
            gaze_bone="頭",
            gesture="nod",
            gesture_target="頭",
            gesture_motion=motion,
            gesture_motion_type="reaction",
        )
    )
    renderer.begin_conversation_motion()
    for _ in range(9):
        renderer.advance_conversation_animation(0.1)
    renderer.end_conversation_motion()
    assert renderer.is_conversation_motion_active
    renderer.advance_conversation_animation(0.11)
    assert not renderer.is_conversation_motion_active
    assert renderer._idle_transition_from is not None
    renderer.close()


def test_conversation_interrupt_captures_pose_and_idle_interrupt_keeps_return() -> None:
    renderer = renderer_for("両手後ろ")
    renderer._conversation_motion_active = True
    renderer._gesture_vmd_rotations = {0: (0.5, 0.0, 0.0)}
    renderer._gesture_body_indexes = None
    renderer._motion_mixer.mix(
        delta_seconds=0.35,
        gesture=MotionLayer(rotations={0: (0.5, 0.0, 0.0)}, exact=True),
    )
    renderer.interrupt_motion()
    source = renderer._idle_transition_from
    assert not renderer.is_conversation_motion_active
    assert renderer._idle_return_open_rotations
    assert source.rotations[0] == pytest.approx((0.5, 0.0, 0.0))
    renderer._blend_idle_transition(1.0)
    renderer.interrupt_motion()
    assert renderer._idle_transition_from is source
    assert renderer._idle_transition_elapsed == 1.0
    renderer.close()
