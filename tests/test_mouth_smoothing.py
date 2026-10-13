"""発話時の急な開閉、停止後の再適用、口以外のモーフへの回帰。"""

import math
import struct
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui import dense_pmx_widget, pmx_preview_widget
from multi_llm_companion.ui.models import UiAvatarModelPreview, UiAvatarTexturedTriangle
from multi_llm_companion.ui.pmx_preview_widget import PmxPreviewWidget


@pytest.fixture
def mouth_view(monkeypatch):
    QApplication.instance() or QApplication([])
    view = PmxPreviewWidget()
    view._timer.stop()
    view._heavy_timer.stop()
    now = [0.0]
    monkeypatch.setattr(pmx_preview_widget.time, "perf_counter", lambda: now[0])

    def advance(seconds):
        now[0] += seconds
        view._advance_mouth_transition()

    yield view, advance
    view.close()


def _preview():
    return UiAvatarModelPreview(
        model_name="口の補間テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),), "i": ((0, 0.4, 0.0, 0.0),)},
        dense_textured_triangles=(
            UiAvatarTexturedTriangle(
                indexes=(0, 1, 2),
                uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
                texture_path="",
                diffuse=(1.0, 1.0, 1.0, 1.0),
            ),
        ),
    )


def test_opening_is_gradual_but_reaches_the_vowel_within_150_ms(mouth_view):
    view, advance = mouth_view
    view.set_voice_active(True)
    view.set_lip_sync("a")
    assert not view._mouth_weights
    advance(0.033)
    assert 0.3 < view._mouth_weights["a"] < 0.5
    advance(0.117)
    assert view._mouth_weights["a"] > 0.9


def test_vowels_crossfade_without_closing_or_adding_two_full_mouths(mouth_view):
    view, advance = mouth_view
    view.set_voice_active(True)
    view.set_lip_sync("a")
    advance(0.3)
    view.set_lip_sync("i")
    advance(0.033)
    assert 0.5 < view._mouth_weights["a"] < 0.7
    assert 0.3 < view._mouth_weights["i"] < 0.5
    assert sum(view._mouth_weights.values()) == pytest.approx(1.0)
    advance(0.3)
    assert view._mouth_weights == {"i": 1.0}


def test_short_silence_does_not_snap_closed_but_long_silence_does(mouth_view):
    view, advance = mouth_view
    view.set_voice_active(True)
    view.set_lip_sync("a")
    advance(0.3)
    view.set_lip_sync("neutral")
    advance(0.033)
    assert 0.5 < view._mouth_weights["a"] < 0.7
    advance(0.267)
    assert not view._mouth_weights
    assert not view._dense_view._mouth_weights


def test_stop_discards_interpolation_and_next_speech_starts_closed(mouth_view):
    view, advance = mouth_view
    view.set_voice_active(True)
    view.set_lip_sync("o")
    advance(0.033)
    view.set_voice_active(False)
    assert view._mouth_shape == "neutral"
    assert not view._mouth_weights
    assert not view._dense_view._mouth_weights
    advance(0.1)
    view.set_voice_active(True)
    advance(0.1)
    assert not view._mouth_weights


def test_mouth_continues_when_body_motion_is_disabled(mouth_view):
    view, advance = mouth_view
    view.set_motion_enabled(False)
    view.set_voice_active(True)
    view.set_lip_sync("u")
    assert view._timer.isActive()
    assert not view._heavy_timer.isActive()
    idle_phase = view._idle_phase
    advance(0.033)
    view._advance_lightweight_motion()
    assert 0 < view._mouth_weights["u"] < 1
    assert view._idle_phase == idle_phase
    view.set_voice_active(False)
    assert not view._timer.isActive()


def test_manual_morph_preview_remains_immediate(mouth_view):
    view, _advance = mouth_view
    view.set_lip_sync("e")
    assert view._mouth_weights == {"e": 1.0}
    assert view._dense_view._mouth_weights == {"e": 1.0}
    assert not view._timer.isActive()
    view.set_lip_sync("neutral")
    assert not view._mouth_weights


def test_interpolation_is_independent_of_render_frame_rate(mouth_view):
    view, advance = mouth_view
    view.set_voice_active(True)
    view.set_lip_sync("a")
    for _ in range(10):
        advance(0.01)
    fine = view._mouth_weights["a"]
    view.set_voice_active(False)
    view.set_voice_active(True)
    view.set_lip_sync("a")
    advance(0.1)
    assert view._mouth_weights["a"] == pytest.approx(fine)


def test_weighted_gpu_and_fallback_match_without_rebuilding_model(mouth_view, monkeypatch):
    view, advance = mouth_view
    preview = _preview()
    view.set_preview(preview)
    view._dense_view._refresh_pose_mesh()

    def unexpected_build(*_args, **_kwargs):
        pytest.fail("口更新でモデル全体を再構築しました")

    monkeypatch.setattr(dense_pmx_widget, "_build_groups", unexpected_build)
    view.set_voice_active(True)
    view.set_lip_sync("a")
    advance(0.3)
    view.set_lip_sync("i")
    advance(0.033)
    weights = view._mouth_weights
    # 次の姿勢更新でも、口だけの変化を全体の変更と誤認しない。
    view._dense_view._refresh_pose_mesh()
    view._dense_view._queue_pose_mesh_refresh()
    expected = (weights["i"] * 0.4, weights["a"] * 0.5, 0.0)
    actual = struct.unpack_from("<3f", view._dense_view._groups[0].vertices)
    assert actual == pytest.approx(expected)
    # 簡易表示固有の正面ピッチ0.1を除けば同じ口の変形。
    fallback_expected = (expected[0], expected[1] * math.cos(0.1), expected[1] * math.sin(0.1))
    assert view._transformed_points(preview, 0)[0] == pytest.approx(fallback_expected)


def test_blend_keeps_expression_and_blink_after_pose_replacement(mouth_view):
    view, _advance = mouth_view
    preview = replace(_preview(), named_vertex_morphs={"笑い": ((0, 0, 0, 0.2),)})
    view.set_preview(preview)
    dense = view._dense_view
    dense._blink_morph_name = "笑い"
    dense._blink_occurrences = dense_pmx_widget._build_mouth_occurrences(preview)
    dense._blink_weight = 0.5
    dense._set_mouth_weights("i", {"a": 0.6, "i": 0.4})
    assert struct.unpack_from("<3f", dense._groups[0].vertices) == pytest.approx((0.16, 0.3, 0.1))
    # 姿勢を再構築しても現在の混合口形を適用する。
    groups = dense._build_pose_groups({}, {}, {}, {"笑い": 0.25})
    assert struct.unpack_from("<3f", groups[0].vertices) == pytest.approx((0.16, 0.3, 0.15))


def test_late_pose_install_uses_current_blend_and_does_not_reopen_stopped_mouth(mouth_view):
    view, advance = mouth_view
    view.set_preview(_preview())
    dense = view._dense_view
    view.set_voice_active(True)
    view.set_lip_sync("a")
    advance(0.1)
    request = dense._pose_build_request()
    view.set_lip_sync("i")
    advance(0.033)
    weights = dict(view._mouth_weights)
    dense._install_pose_groups(request, dense_pmx_widget._build_groups(dense._preview, "neutral"))
    assert struct.unpack_from("<3f", dense._groups[0].vertices) == pytest.approx(
        (weights["i"] * 0.4, weights["a"] * 0.5, 0.0)
    )
    view.set_voice_active(False)
    dense._install_pose_groups(request, dense_pmx_widget._build_groups(dense._preview, "neutral"))
    assert struct.unpack_from("<3f", dense._groups[0].vertices) == pytest.approx((0.0, 0.0, 0.0))
