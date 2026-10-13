from __future__ import annotations

import inspect
import math
import os
import struct
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QSurfaceFormat, QVector3D
from PySide6.QtWidgets import QApplication, QLabel

import multi_llm_companion.ui.dense_pmx_widget as dense_pmx_widget
import multi_llm_companion.ui.pmx_pose as pmx_pose
from multi_llm_companion.ui.bullet_pmx_physics import ThinClothSourceSurface
from multi_llm_companion.ui.dense_pmx_widget import (
    _FRAGMENT_SHADER,
    _RESOLVE_FRAGMENT_SHADER,
    _RESOLVE_VERTEX_SHADER,
    _VERTEX_SHADER,
    DensePmxWidget,
    _apply_mouth_shape_to_groups,
    _body_mask_bone_indexes,
    _build_groups,
    _build_mouth_occurrences,
    _bust_center,
    _ClothState,
    _create_texture,
    _effective_gaze_bone_indexes,
    _mmd_outline_camera_scale,
    _model_center,
    _natural_motion_centers,
    _natural_rest_rotations,
    _opengl_texture_image,
    _prepare_pose_data,
    _safe_supersample_factor,
    _separate_thin_cloth_mesh,
    _set_scalar_uniform,
    _surface_format,
    _texture_image,
)
from multi_llm_companion.ui.mmd_motion_mixer import MixedMotion, MotionLayer
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarMaterialMorph,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
    UiAvatarTexturedTriangle,
    UiMmdAvatarCommand,
    UiVmdBoneFrame,
    UiVmdCameraFrame,
    UiVmdIkFrame,
    UiVmdIkState,
    UiVmdLightFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.natural_motion import NaturalMotionFrame
from multi_llm_companion.ui.pmx_physics import (
    PhysicsPose,
    PmxPhysicsSimulator,
    _body_anchors,
)
from multi_llm_companion.ui.pmx_pose import (
    apply_bone_pose,
    bone_transforms,
    matrix_vector,
    prepare_ik_solver,
    solve_ik_rotations,
)
from multi_llm_companion.ui.pmx_preview_widget import PmxPreviewWidget, _camera_rotate
from multi_llm_companion.ui.render_quality import (
    configure_default_surface_format,
    normalized_render_quality,
    requested_sample_count,
)


def test_thin_cloth_mesh_separation_moves_only_target_vertices() -> None:
    points = np.asarray(
        ((0.4, 0.0, 0.0), (0.0, 0.0, 2.0), (0.2, 0.0, 0.0)),
        dtype=np.float32,
    )
    _separate_thin_cloth_mesh(
        points,
        np.asarray((0, 1), dtype=np.int32),
        (
            ThinClothSourceSurface(
                position=(0.0, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
                radius=1.0,
                half_length=0.0,
            ),
        ),
    )

    assert points[0, 0] > 1.0
    assert tuple(points[1]) == pytest.approx((0.0, 0.0, 2.0))
    assert tuple(points[2]) == pytest.approx((0.2, 0.0, 0.0))


def test_prepared_thin_cloth_targets_match_configured_display_bones() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        normals=((0.0, 0.0, 1.0),) * 3,
    )
    preview = UiAvatarModelPreview(
        model_name="薄布対象ボーンテスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        bones=(
            UiAvatarBone("Skirt_0", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("Jacket_Chest_0", -1, (1.0, 0.0, 0.0)),
            UiAvatarBone("対象外", -1, (2.0, 0.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),), ((1, 1.0),), ((2, 1.0),)),
        dense_textured_triangles=(triangle,),
        thin_cloth_contact_enabled=True,
        thin_cloth_target_body_patterns=("Skirt_", "Jacket_Chest_"),
    )

    prepared = _prepare_pose_data(preview)

    assert prepared is not None
    assert prepared.thin_cloth_target_vertex_indexes.tolist() == [0, 1]


def test_fixed_size_matrix_helpers_keep_the_original_results() -> None:
    """周期処理向けの直接計算でも、3次元数学の結果を変えない。"""

    first = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    second = ((9.0, 8.0, 7.0), (6.0, 5.0, 4.0), (3.0, 2.0, 1.0))

    assert pmx_pose.matrix_multiply(first, second) == (
        (30.0, 24.0, 18.0),
        (84.0, 69.0, 54.0),
        (138.0, 114.0, 90.0),
    )
    assert pmx_pose.matrix_transpose(first) == (
        (1.0, 4.0, 7.0),
        (2.0, 5.0, 8.0),
        (3.0, 6.0, 9.0),
    )
    assert pmx_pose.matrix_vector(first, (2.0, 3.0, 4.0)) == (20.0, 47.0, 74.0)
    assert pmx_pose.add((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == (5.0, 7.0, 9.0)
    assert pmx_pose.subtract((4.0, 5.0, 6.0), (1.0, 2.0, 3.0)) == (3.0, 3.0, 3.0)
    assert pmx_pose.dot((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == 32.0


def test_pmx_preview_widget_draws_triangular_mesh() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    widget.resize(240, 280)
    widget.set_preview(
        UiAvatarModelPreview(
            model_name="テストメッシュ",
            vertex_count=3,
            face_count=1,
            points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            triangles=((0, 1, 2),),
            mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
            emotion_morphs={"happy": ((0, 0.5, 0.0, 0.0),)},
        )
    )
    widget.set_motion_enabled(False)
    widget.set_mmd_command(UiMmdAvatarCommand("happy", 0.5, "笑顔", "forward", "", "none", ""))
    widget.set_lip_sync("a")
    app.processEvents()

    assert not widget.grab().isNull()
    assert widget._preview is not None
    transformed = widget._transformed_points(widget._preview, 0.0)
    assert transformed[0][0] == pytest.approx(-0.75)
    assert transformed[0][1] == pytest.approx(0.5 * 0.995004165)
    assert widget._timer.isActive() is False
    widget.close()


def test_pmx_preview_fallback_combines_multiple_expression_morphs() -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    preview = UiAvatarModelPreview(
        model_name="複数表情の簡易描画テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        named_vertex_morphs={
            "笑い": ((0, 0.2, 0.0, 0.0),),
            "にこり": ((0, 0.0, 0.3, 0.0),),
        },
    )
    widget.set_preview(preview)
    widget.set_motion_enabled(False)
    widget.set_mmd_command(
        UiMmdAvatarCommand(
            "happy",
            0.8,
            "笑い",
            "forward",
            "",
            "none",
            "",
            expression_preset="gentle_smile",
            expression_morphs=(("笑い", 0.5), ("にこり", 0.25)),
        )
    )

    transformed = widget._transformed_points(preview, 0.0)

    assert transformed[0][0] == pytest.approx(0.1)
    assert transformed[0][1] == pytest.approx(0.075 * 0.995004165)
    assert widget._face_target_indexes(preview) == {0}
    widget.close()


@pytest.mark.parametrize("motion_enabled", [True, False])
def test_rest_expression_returns_after_interrupt_without_old_command(motion_enabled) -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    preview = UiAvatarModelPreview(
        model_name="標準表情",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        named_vertex_morphs={"じと目": ((0, 0.0, -0.2, 0.0),)},
    )
    try:
        widget.set_preview(preview)
        widget.set_motion_enabled(motion_enabled)
        widget._set_rest_expression({"じと目": 0.3})
        assert widget._target_expression_weights == {"じと目": 0.3}
        happy = UiMmdAvatarCommand(
            "happy", 0.7, "", "left", "", "nod", "", expression_morphs=(("笑い", 0.7),)
        )
        widget.set_mmd_command(happy)
        assert widget._target_expression_weights == {"笑い": 0.7}
        widget.interrupt_motion()
        assert widget._command is None
        assert widget._dense_view._command is None
        assert widget._target_expression_weights == {"じと目": 0.3}
        for _ in range(30):
            morphs = widget._dense_view._pose_settings()[3]
        assert morphs["じと目"] == pytest.approx(0.3, abs=0.001)
        assert morphs.get("笑い", 0) < 0.001
        assert not widget.is_conversation_motion_active
        if not motion_enabled:
            assert widget._expression_weights == {"じと目": 0.3}
        widget.start_expression_preview(happy)
        widget.stop_expression_preview()
        assert widget._target_expression_weights == {"じと目": 0.3}
    finally:
        widget.close()


def test_pmx_preview_fallback_replays_generated_vmd_bones() -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    motion = UiVmdMotion(
        name="自然待機",
        frame_count=30,
        bone_frames=(
            UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame("センター", 30, (0.0, 0.3, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
    )
    preview = UiAvatarModelPreview(
        model_name="VMD簡易描画テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vertex_weights=(((0, 1.0),),),
        vmd_motion=motion,
    )
    widget.set_preview(preview)
    widget._dense_renderer_available = False
    widget._last_heavy_at -= 0.5

    widget._advance_heavy_motion()

    assert widget._fallback_vmd_translations[0][1] == pytest.approx(0.15, abs=0.01)
    assert widget._transformed_points(preview, 0.0)[0][1] > 0.14
    assert "VMD再生: 自然待機" in widget.renderer_status_text
    widget.close()


def test_pmx_preview_fallback_keeps_idle_vmd_moving_during_voice() -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    motion = UiVmdMotion(
        name="自然待機",
        frame_count=30,
        bone_frames=(
            UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame("センター", 30, (0.0, 0.3, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
    )
    preview = UiAvatarModelPreview(
        model_name="発話中VMD簡易描画テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vertex_weights=(((0, 1.0),),),
        vmd_motion=motion,
    )
    widget.set_preview(preview)
    widget._dense_renderer_available = False
    widget.set_voice_active(True)
    widget._last_heavy_at -= 0.5

    widget._advance_heavy_motion()

    assert widget._fallback_vmd_translations[0][1] == pytest.approx(0.15, abs=0.01)
    widget.close()


def test_quality_status_change_is_forwarded_after_first_gpu_frame() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    notifications: list[str] = []
    widget.renderer_status_changed.connect(lambda: notifications.append("changed"))

    widget._dense_view.quality_status_changed.emit()
    app.processEvents()

    assert notifications == ["changed"]
    widget.close()


def test_renderer_status_shows_native_bullet_physics() -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    widget._preview = UiAvatarModelPreview(
        model_name="物理状態表示テスト",
        vertex_count=3,
        face_count=1,
        points=(),
        dense_textured_triangles=(
            UiAvatarTexturedTriangle(
                indexes=(0, 1, 2),
                uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                texture_path="",
                diffuse=(1.0, 1.0, 1.0, 1.0),
            ),
        ),
    )
    widget._dense_renderer_available = True

    assert "ネイティブBullet 2.75互換物理・PMX剛体/6DoFばね/ジョイント" in (
        widget.renderer_status_text
    )
    widget.close()


def test_pmx_preview_does_not_rebuild_unchanged_model_or_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    preview = UiAvatarModelPreview(
        model_name="再構築防止テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    command = UiMmdAvatarCommand("neutral", 0.0, "", "forward", "", "none", "")
    preview_calls: list[UiAvatarModelPreview | None] = []
    command_calls: list[UiMmdAvatarCommand | None] = []
    monkeypatch.setattr(widget._dense_view, "set_preview", preview_calls.append)
    monkeypatch.setattr(widget._dense_view, "set_mmd_command", command_calls.append)

    widget.set_preview(preview)
    widget.set_preview(preview)
    widget.set_mmd_command(command)
    widget.set_mmd_command(command)

    assert preview_calls == [preview]
    assert command_calls == [command]
    widget.close()


def test_pmx_preview_widget_does_not_cover_model_with_status_labels() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    widget.resize(320, 280)
    widget.set_preview(
        UiAvatarModelPreview(
            model_name="状態表示テスト",
            vertex_count=3,
            face_count=1,
            points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            triangles=((0, 1, 2),),
        )
    )
    widget.show()
    app.processEvents()

    assert widget.findChildren(QLabel) == []
    widget.close()


def test_pmx_preview_widget_applies_gaze_bone_rotation() -> None:
    app = QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    widget.set_motion_enabled(False)
    widget.set_preview(
        UiAvatarModelPreview(
            model_name="ボーンテスト",
            vertex_count=3,
            face_count=1,
            points=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            triangles=((0, 1, 2),),
            bones=(UiAvatarBone("両目", -1, (0.0, 0.0, 0.0)),),
            vertex_weights=(((0, 1.0),), ((0, 1.0),), ((0, 1.0),)),
        )
    )
    widget.set_mmd_command(UiMmdAvatarCommand("neutral", 0.0, "", "left", "両目", "none", ""))
    app.processEvents()

    assert widget._preview is not None
    transformed = widget._transformed_points(widget._preview, 0.0)
    assert transformed[0][0] < 1.0
    assert transformed[0][2] < 0.0
    widget.close()


def test_pmx_preview_widget_hides_face_closeup_when_no_morph_is_active() -> None:
    widget = PmxPreviewWidget()
    widget.set_motion_enabled(False)
    preview = UiAvatarModelPreview(
        model_name="顔テスト",
        vertex_count=3,
        face_count=1,
        points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        triangles=((0, 1, 2),),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
        emotion_morphs={"happy": ((1, 0.0, 0.5, 0.0),)},
    )
    widget.set_preview(preview)

    assert widget._face_target_indexes(preview) == set()
    widget.set_lip_sync("a")
    assert widget._face_target_indexes(preview) == {0}
    widget.close()


def test_pmx_preview_widget_draws_texture_and_resets_camera(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    texture_path = str(tmp_path / "texture.png")
    texture = QImage(8, 8, QImage.Format.Format_ARGB32)
    texture.fill(QColor("#e88ab5"))
    assert texture.save(texture_path)
    widget = PmxPreviewWidget()
    widget.resize(240, 280)
    widget.set_motion_enabled(False)
    widget.set_preview(
        UiAvatarModelPreview(
            model_name="テクスチャテスト",
            vertex_count=3,
            face_count=1,
            points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            triangles=((0, 1, 2),),
            textured_triangles=(
                UiAvatarTexturedTriangle(
                    indexes=(0, 1, 2),
                    uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
                    texture_path=texture_path,
                    diffuse=(1.0, 1.0, 1.0, 1.0),
                ),
            ),
        )
    )
    widget._yaw = 0.7
    widget._zoom = 2.0
    app.processEvents()

    assert not widget.grab().isNull()
    assert not widget._texture_cache[texture_path].isNull()
    assert widget._dense_view._yaw == math.pi
    widget.reset_camera()
    assert widget._yaw == math.pi
    assert widget._zoom == 1.0
    widget.close()


def test_dense_renderer_uses_full_texture_triangle_data() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="local/texture.png",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        draw_outline=True,
        edge_color=(0.1, 0.2, 0.3, 0.8),
        edge_size=1.25,
        edge_scales=(0.5, 1.0, 1.5),
    )
    preview = UiAvatarModelPreview(
        model_name="全描画テスト",
        vertex_count=3,
        face_count=1,
        points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        textured_triangles=(),
        dense_textured_triangles=(triangle,),
    )

    groups = _build_groups(preview)

    assert len(groups) == 1
    assert groups[0].count == 3
    assert groups[0].draw_outline is True
    assert groups[0].edge_color == pytest.approx((0.1, 0.2, 0.3, 0.8))
    assert groups[0].edge_size == pytest.approx(1.25)
    assert len(groups[0].vertices) == 3 * 11 * 4
    values = struct.unpack("<33f", groups[0].vertices)
    assert (values[10], values[21], values[32]) == pytest.approx((0.5, 1.0, 1.5))
    assert _model_center(preview) == (0.0, 0.5, 0.0)
    assert _bust_center(preview) == (0.0, 0.84, 0.0)


def test_bust_camera_uses_mouth_morph_position_when_available() -> None:
    preview = UiAvatarModelPreview(
        model_name="口中心テスト",
        vertex_count=3,
        face_count=0,
        points=((0.0, 1.0, 0.0), (2.0, 8.0, 1.0), (4.0, 10.0, 3.0)),
        mouth_morphs={"a": ((1, 0.0, 0.1, 0.0), (2, 0.0, 0.1, 0.0))},
    )

    assert _bust_center(preview) == (3.0, 9.0, 2.0)


def test_dense_renderer_keeps_pmx_texture_orientation() -> None:
    image = QImage(2, 2, QImage.Format.Format_ARGB32)
    image.setPixelColor(0, 0, QColor("#ff0000"))
    image.setPixelColor(1, 0, QColor("#00ff00"))
    image.setPixelColor(0, 1, QColor("#0000ff"))
    image.setPixelColor(1, 1, QColor("#ffffff"))

    converted = _opengl_texture_image(image)

    assert converted.pixelColor(0, 0) == QColor("#ff0000")
    assert converted.pixelColor(1, 0) == QColor("#00ff00")
    assert converted.pixelColor(0, 1) == QColor("#0000ff")
    assert converted.pixelColor(1, 1) == QColor("#ffffff")


def test_dense_renderer_disables_texture_atlas_mipmaps() -> None:
    source = inspect.getsource(_create_texture)

    assert "DontGenerateMipMaps" in source


def test_dense_renderer_discards_fully_transparent_texture_pixels() -> None:
    assert "discard;" in _FRAGMENT_SHADER
    assert "color.a < 0.02" in _FRAGMENT_SHADER
    assert "sphereMode" in _FRAGMENT_SHADER
    assert "toonTextureSampler" in _FRAGMENT_SHADER
    assert "outlinePass" in _FRAGMENT_SHADER
    assert "outlineWidth" in _VERTEX_SHADER
    assert "in float edgeScale;" in _VERTEX_SHADER
    assert "uniform vec2 viewportSize;" in _VERTEX_SHADER
    assert "pixelWidth = outlineWidth * max(edgeScale, 0.0)" in _VERTEX_SHADER


def test_dense_renderer_calibrates_pmx_edge_width_to_mmd_pixels() -> None:
    assert _mmd_outline_camera_scale(1.0, "bust") == pytest.approx(0.6)
    assert _mmd_outline_camera_scale(1.75, "bust") == pytest.approx(1.05)
    assert _mmd_outline_camera_scale(0.4, "bust") == pytest.approx(0.35)
    assert _mmd_outline_camera_scale(1.0, "full") == pytest.approx(0.35)
    assert _mmd_outline_camera_scale(3.0, "bust") == pytest.approx(1.8)
    source = inspect.getsource(DensePmxWidget._draw_group)
    assert "self._supersample_factor" in source
    assert "_mmd_outline_camera_scale(self._zoom, self._view_mode)" in source


def test_dense_renderer_preserves_pmx_outline_rgba() -> None:
    source = inspect.getsource(DensePmxWidget._draw_group)

    assert "QVector4D(*group.edge_color)" in source
    assert "_mmd_reference_outline_color" not in source


def test_dense_renderer_preserves_bright_face_texture_details() -> None:
    assert "vec3 lightColor = ambient + toon * sceneLightColor;" in _FRAGMENT_SHADER
    assert "float maxLight" in _FRAGMENT_SHADER
    assert "lightColor / max(maxLight, 1.0)" in _FRAGMENT_SHADER


def test_dense_renderer_uses_locations_for_scalar_shader_uniforms() -> None:
    source = inspect.getsource(DensePmxWidget._paint_model) + inspect.getsource(
        DensePmxWidget._draw_group
    )

    assert '_set_scalar_uniform(program, "shininess"' in source
    assert '_set_scalar_uniform(program, "sphereMode"' in source
    assert 'program.setUniformValue("mvp"' in source


def test_render_quality_requests_safe_msaa_levels() -> None:
    assert _surface_format("low").samples() == 0
    assert _surface_format("standard").samples() == 4
    assert _surface_format("high").samples() == 8


def test_default_surface_format_uses_the_same_quality_request() -> None:
    previous = QSurfaceFormat.defaultFormat()
    try:
        configured = configure_default_surface_format("high")

        assert configured.samples() == 8
        assert configured.swapInterval() == 0
        assert QSurfaceFormat.defaultFormat().samples() == 8
        assert QSurfaceFormat.defaultFormat().swapInterval() == 0
        assert requested_sample_count("standard") == 4
        assert normalized_render_quality("UNKNOWN") == "standard"
    finally:
        QSurfaceFormat.setDefaultFormat(previous)


@pytest.mark.parametrize("quality", ["low", "standard", "high"])
def test_renderer_format_does_not_block_timer_updates_on_vsync(quality: str) -> None:
    surface = _surface_format(quality)
    assert surface.swapInterval() == 0
    assert surface.depthBufferSize() == 24
    assert surface.stencilBufferSize() == 8


def test_supersampling_is_quality_dependent_and_memory_bounded() -> None:
    assert _safe_supersample_factor("low", 1000, 800) == 1.0
    assert _safe_supersample_factor("standard", 1000, 800) == 2.0
    assert _safe_supersample_factor("high", 1000, 800) == 2.0
    assert _safe_supersample_factor("high", 4000, 3000) == 1.0


def test_supersampling_resolves_on_the_gpu_without_a_cpu_readback() -> None:
    source = inspect.getsource(DensePmxWidget._resolve_supersample_target)

    assert "source.texture()" in source
    assert "glDrawArrays" in source
    assert "toImage" not in source
    assert "resolvedTexture" in _RESOLVE_FRAGMENT_SHADER
    assert "gl_VertexID" in _RESOLVE_VERTEX_SHADER


def test_dense_renderer_reuses_gpu_textures_by_path() -> None:
    source = inspect.getsource(DensePmxWidget._upload_groups)

    assert "self._texture_cache" in source
    assert "_texture_cache_key" in source


def test_translucent_materials_keep_opaque_scene_alpha_for_qt_composition() -> None:
    from unittest.mock import Mock

    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer._program = Mock()
    renderer._program.uniformLocation.return_value = -1
    functions = Mock()
    renderer._paint_model(functions)
    functions.glBlendFuncSeparate.assert_called_once_with(0x0302, 0x0303, 1, 0x0303)
    functions.glBlendFunc.assert_not_called()
    renderer._program = None
    renderer.close()


def test_scalar_shader_uniform_uses_typed_opengl_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunctions:
        def __init__(self) -> None:
            self.float_values: list[tuple[int, float]] = []
            self.int_values: list[tuple[int, int]] = []

        def glUniform1f(self, location: int, value: float) -> None:  # noqa: N802
            self.float_values.append((location, value))

        def glUniform1i(self, location: int, value: int) -> None:  # noqa: N802
            self.int_values.append((location, value))

    functions = FakeFunctions()

    class FakeContext:
        def extraFunctions(self) -> FakeFunctions:  # noqa: N802
            return functions

    class FakeProgram:
        def uniformLocation(self, name: str) -> int:  # noqa: N802
            return {"shininess": 7, "sphereMode": 9}[name]

        def setUniformValue(self, location: int, value: int | float) -> None:  # noqa: N802
            raise AssertionError("有効なOpenGLコンテキストではQtの曖昧なscalar overloadを使わない")

    program = FakeProgram()
    monkeypatch.setattr(
        dense_pmx_widget.QOpenGLContext,
        "currentContext",
        lambda: FakeContext(),
    )

    _set_scalar_uniform(program, "shininess", 24.0)  # type: ignore[arg-type]
    _set_scalar_uniform(program, "sphereMode", 2)  # type: ignore[arg-type]

    assert functions.float_values == [(7, 24.0)]
    assert functions.int_values == [(9, 2)]


def test_dense_renderer_records_render_errors_without_raising() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer._available = True

    renderer._handle_render_error(ValueError("test"))

    assert renderer.is_available is False
    assert "GPU描画設定に失敗しました" in renderer.error_text
    renderer.close()


def test_dense_renderer_uses_white_texture_for_an_untextured_material() -> None:
    image = _texture_image("")

    assert image.size().width() == 1
    assert image.pixelColor(0, 0) == QColor("#ffffff")


def test_dense_renderer_generates_a_shared_toon_texture_without_external_assets() -> None:
    image = _texture_image("\0shared-toon-4")

    assert image.size().width() == 1
    assert image.size().height() == 256
    assert image.pixelColor(0, 0) != image.pixelColor(0, 255)


def test_dense_renderer_applies_a_mouth_morph_to_gpu_vertex_data() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="local/texture.png",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="口モーフテスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
        dense_textured_triangles=(triangle,),
    )

    neutral = _build_groups(preview)
    opened = _build_groups(preview, "a")

    assert neutral[0].vertices != opened[0].vertices


def test_dense_renderer_switches_mouth_without_rebuilding_all_triangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import multi_llm_companion.ui.dense_pmx_widget as dense_module

    QApplication.instance() or QApplication([])
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="口差分更新テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
        dense_textured_triangles=(triangle,),
    )
    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    neutral_vertices = renderer._groups[0].vertices
    build_count = 0
    original_build_groups = dense_module._build_groups

    def tracked_build_groups(*args: object, **kwargs: object) -> list[object]:
        nonlocal build_count
        build_count += 1
        return original_build_groups(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(dense_module, "_build_groups", tracked_build_groups)

    renderer.set_mouth_shape("a")
    opened_vertices = renderer._groups[0].vertices
    renderer.set_mouth_shape("a")
    renderer.set_animation(renderer._idle_phase, renderer._motion_enabled, True)

    assert opened_vertices != neutral_vertices
    assert renderer._groups[0].vertices == opened_vertices
    assert build_count == 0
    renderer.close()


def test_fast_mouth_update_adds_smile_without_erasing_it() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="笑顔と口の合成テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
        named_vertex_morphs={"笑い": ((0, 0.4, 0.0, 0.0),)},
        dense_textured_triangles=(triangle,),
    )
    groups = _build_groups(preview, "neutral", morph_weights={"笑い": 0.5})

    _apply_mouth_shape_to_groups(
        groups,
        preview,
        "a",
        _build_mouth_occurrences(preview),
    )

    x, y, z = struct.unpack_from("<3f", groups[0].vertices, 0)
    assert (x, y, z) == pytest.approx((0.2, 0.5, 0.0))


def test_fast_mouth_update_keeps_blink_and_brow_morphs_added() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="まばたきと眉と口の合成テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.4, 0.0),)},
        named_vertex_morphs={
            "まばたき": ((0, 0.1, 0.0, 0.0),),
            "眉上": ((0, 0.0, 0.0, 0.2),),
        },
        dense_textured_triangles=(triangle,),
    )
    groups = _build_groups(
        preview,
        "neutral",
        morph_weights={"まばたき": 0.8, "眉上": 0.5},
    )

    _apply_mouth_shape_to_groups(
        groups,
        preview,
        "a",
        _build_mouth_occurrences(preview),
    )

    x, y, z = struct.unpack_from("<3f", groups[0].vertices, 0)
    assert (x, y, z) == pytest.approx((0.08, 0.4, 0.1))


def test_fast_mouth_update_keeps_material_and_uv_expression() -> None:
    QApplication.instance() or QApplication([])
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(0.8, 0.7, 0.6, 1.0),
        additional_uvs=(
            ((0.4, 0.5, 0.0, 0.0),),
            ((0.0, 0.0, 0.0, 0.0),),
            ((0.0, 0.0, 0.0, 0.0),),
        ),
    )
    material_morphs = (
        UiAvatarMaterialMorph(
            material_index=0,
            operation=0,
            diffuse=(0.5, 1.0, 1.0, 0.5),
            specular=(1.0, 1.0, 1.0),
            shininess=1.0,
            ambient=(1.0, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            edge_size=1.0,
            texture_factor=(0.5, 1.0, 1.0, 1.0),
            sphere_factor=(1.0, 1.0, 1.0, 1.0),
            toon_factor=(1.0, 1.0, 1.0, 1.0),
        ),
        UiAvatarMaterialMorph(
            material_index=-1,
            operation=1,
            diffuse=(0.1, 0.0, 0.0, 0.1),
            specular=(0.0, 0.0, 0.0),
            shininess=0.0,
            ambient=(0.0, 0.0, 0.0),
            edge_color=(0.0, 0.0, 0.0, 0.0),
            edge_size=0.0,
            texture_factor=(0.2, 0.0, 0.0, 1.0),
            sphere_factor=(0.0, 0.0, 0.0, 0.0),
            toon_factor=(0.0, 0.0, 0.0, 0.0),
        ),
    )
    preview = UiAvatarModelPreview(
        model_name="材質UVと口の合成テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.4, 0.0),)},
        morph_names=("表情",),
        named_uv_morphs={
            "表情": (
                (0, 0, 0.2, -0.1, 0.0, 0.0),
                (1, 0, 0.1, 0.2, 0.0, 0.0),
            )
        },
        named_material_morphs={"表情": material_morphs},
        dense_textured_triangles=(triangle,),
    )
    direct = _build_groups(preview, morph_weights={"表情": 0.5})[0]
    direct_values = struct.unpack_from("<10f", direct.vertices, 0)
    assert direct_values[6:10] == pytest.approx((0.1, -0.05, 0.45, 0.6))
    assert direct.diffuse == pytest.approx((0.65, 0.7, 0.6, 0.8))
    assert direct.is_transparent is True
    assert direct.texture_multiply == pytest.approx((0.75, 1.0, 1.0, 1.0))
    assert direct.texture_add == pytest.approx((0.1, 0.0, 0.0, 0.5))

    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    renderer._command = UiMmdAvatarCommand(
        "joy",
        1.0,
        "表情",
        "forward",
        "",
        "none",
        "",
        expression_morphs=(("表情", 0.5),),
    )
    renderer._refresh_pose_mesh(force=True)
    expression_values = struct.unpack_from("<10f", renderer._groups[0].vertices, 0)
    expression_diffuse = renderer._groups[0].diffuse

    renderer.set_mouth_shape("a")
    combined_values = struct.unpack_from("<10f", renderer._groups[0].vertices, 0)

    assert expression_values[6:10] != pytest.approx((0.0, 0.0, 0.4, 0.5))
    assert expression_diffuse != pytest.approx(triangle.diffuse)
    assert renderer._groups[0].is_transparent is True
    assert combined_values[1] == pytest.approx(0.4)
    assert combined_values[6:10] == pytest.approx(expression_values[6:10])
    assert renderer._groups[0].diffuse == pytest.approx(expression_diffuse)
    renderer.close()


def test_dense_renderer_combines_vmd_expression_and_mouth_morph() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="local/texture.png",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="表情合成テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.5, 0.0),)},
        named_vertex_morphs={"笑い": ((0, 0.4, 0.0, 0.0),)},
        dense_textured_triangles=(triangle,),
    )

    expression = _build_groups(preview, "neutral", morph_weights={"笑い": 0.5})
    combined = _build_groups(preview, "a", morph_weights={"笑い": 0.5})

    assert expression[0].vertices != combined[0].vertices


def test_dense_renderer_applies_weighted_bone_rotation_to_gpu_vertex_data() -> None:
    preview = UiAvatarModelPreview(
        model_name="ボーン変形テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 1.0, 0.0),),
        bones=(UiAvatarBone("中心", -1, (0.0, 0.0, 0.0)),),
        vertex_weights=(((0, 1.0),),),
    )

    posed = apply_bone_pose(list(preview.points), preview, {0: (0.0, 0.0, math.pi / 2)})

    assert posed[0][0] == pytest.approx(-1.0)
    assert posed[0][1] == pytest.approx(0.0, abs=0.0001)


def test_pmx_ik_solver_moves_the_effector_toward_the_controller_goal() -> None:
    preview = UiAvatarModelPreview(
        model_name="IKテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (1.0, 1.0, 0.0), 3, 32, 0.35, (2, 1)),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
    )
    goal = preview.bones[0].position
    before = bone_transforms(preview, {})[3][1]
    solved = solve_ik_rotations(preview, {}, {0: (0.0, 0.0, 0.0)})
    after = bone_transforms(preview, solved)[3][1]

    before_distance = math.dist(before, goal)
    after_distance = math.dist(after, goal)

    assert after_distance < before_distance


def test_pmx_ik_solver_enables_controller_without_explicit_offset() -> None:
    preview = UiAvatarModelPreview(
        model_name="既定IKテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("左つま先ＩＫ", -1, (1.0, 1.0, 0.0), 3, 4, 0.35, (2, 1)),
            UiAvatarBone("左足首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左足先", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足先EX", 2, (0.0, 2.0, 0.0)),
        ),
    )
    goal = preview.bones[0].position
    before = bone_transforms(preview, {})[3][1]

    solved = solve_ik_rotations(preview, {}, {})
    after = bone_transforms(preview, solved)[3][1]

    assert math.dist(after, goal) < math.dist(before, goal)


def test_pmx_ik_solver_skips_vmd_disabled_controller() -> None:
    preview = UiAvatarModelPreview(
        model_name="IK無効テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("左つま先ＩＫ", -1, (1.0, 1.0, 0.0), 3, 4, 0.35, (2, 1)),
            UiAvatarBone("左足首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左足先", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足先EX", 2, (0.0, 2.0, 0.0)),
        ),
    )

    solved = solve_ik_rotations(
        preview,
        {},
        {},
        disabled_controller_indexes=frozenset({0}),
    )

    assert solved == {}


def test_pmx_ik_solver_stops_when_constraints_cannot_reduce_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = UiAvatarModelPreview(
        model_name="IK停滞終了テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("つま先ＩＫ", -1, (0.0, 3.0, 0.0), 2, 100, 0.35, (1,)),
            UiAvatarBone("足首", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("足先", 1, (0.0, 1.0, 0.0)),
        ),
    )
    calls = 0
    original = pmx_pose.bone_transforms

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pmx_pose, "bone_transforms", counted)

    solve_ik_rotations(preview, {}, {})

    assert calls == 1


def test_pmx_ik_solver_reuses_model_metadata_without_rescanning_bones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = UiAvatarModelPreview(
        model_name="IK準備データテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (1.0, 1.0, 0.0), 3, 32, 0.35, (2, 1)),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
    )
    prepared = prepare_ik_solver(preview)
    expected = solve_ik_rotations(preview, {}, {})

    def fail_prepare(_preview: UiAvatarModelPreview) -> object:
        raise AssertionError("モデル不変のIK情報を再走査している")

    monkeypatch.setattr(pmx_pose, "prepare_ik_solver", fail_prepare)
    actual = solve_ik_rotations(preview, {}, {}, prepared=prepared)

    assert prepared.controller_indexes == frozenset({0})
    assert prepared.link_indexes == frozenset({1, 2})
    assert actual.keys() == expected.keys()
    for index in actual:
        assert actual[index] == pytest.approx(expected[index])


def test_pmx_ik_solver_uses_vmd_translation_of_controller_as_goal() -> None:
    preview = UiAvatarModelPreview(
        model_name="VMD移動IKテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (0.0, 2.0, 0.0), 3, 32, 0.35, (2, 1)),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
    )
    translations = {0: (0.8, -0.2, 0.0)}
    goal = (0.8, 1.8, 0.0)
    before = bone_transforms(preview, {}, translations)[3][1]
    solved = solve_ik_rotations(preview, {}, {}, translations)
    after = bone_transforms(preview, solved, translations)[3][1]

    assert math.dist(after, goal) < math.dist(before, goal)


def test_bone_transform_subset_matches_full_transform_for_requested_bones() -> None:
    preview = UiAvatarModelPreview(
        model_name="部分変換テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("子", 0, (0.0, 1.0, 0.0)),
            UiAvatarBone("孫", 1, (0.0, 2.0, 0.0)),
            UiAvatarBone("無関係", -1, (5.0, 0.0, 0.0)),
        ),
    )
    rotations = {0: (0.0, 0.0, 0.2), 1: (0.0, 0.0, -0.1)}
    translations = {0: (0.3, 0.0, 0.0)}

    full = bone_transforms(preview, rotations, translations)
    subset = bone_transforms(preview, rotations, translations, frozenset({2}))

    for actual_row, expected_row in zip(subset[2][0], full[2][0], strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert subset[2][1] == pytest.approx(full[2][1])
    assert 3 not in subset


def test_bone_transform_applies_rotation_and_translation_inheritance() -> None:
    preview = UiAvatarModelPreview(
        model_name="付与変形テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("付与親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone(
                "付与子",
                -1,
                (0.0, 0.0, 0.0),
                inherit_parent_index=0,
                inherit_weight=0.5,
                inherit_rotation=True,
                inherit_translation=True,
            ),
        ),
    )

    transforms = bone_transforms(
        preview,
        {0: (0.0, 0.0, math.pi / 2)},
        {0: (2.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0)},
    )
    rotated = matrix_vector(transforms[1][0], (1.0, 0.0, 0.0))

    assert rotated == pytest.approx((math.sqrt(0.5), math.sqrt(0.5), 0.0), abs=0.0001)
    assert transforms[1][1] == pytest.approx((2.0, 0.0, 0.0))


def test_child_translation_uses_rotated_parent_local_coordinates() -> None:
    preview = UiAvatarModelPreview(
        model_name="移動座標テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("子", 0, (0.0, 1.0, 0.0)),
        ),
    )

    transforms = bone_transforms(
        preview,
        {0: (0.0, 0.0, math.pi / 2)},
        {1: (1.0, 0.0, 0.0)},
    )

    assert transforms[1][1] == pytest.approx((-1.0, 1.0, 0.0), abs=0.0001)


def test_fixed_and_local_axes_constrain_bone_rotation() -> None:
    fixed_preview = UiAvatarModelPreview(
        model_name="固定軸テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("固定", -1, (0.0, 0.0, 0.0), fixed_axis=(0.0, 0.0, 1.0)),),
    )
    fixed = bone_transforms(fixed_preview, {0: (1.0, 0.0, math.pi / 2)})
    assert matrix_vector(fixed[0][0], (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=0.0001)

    local_preview = UiAvatarModelPreview(
        model_name="ローカル軸テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone(
                "ローカル",
                -1,
                (0.0, 0.0, 0.0),
                local_axis_x=(0.0, 1.0, 0.0),
                local_axis_z=(0.0, 0.0, 1.0),
            ),
        ),
    )
    local = bone_transforms(local_preview, {0: (math.pi / 2, 0.0, 0.0)})
    assert matrix_vector(local[0][0], (0.0, 0.0, 1.0)) == pytest.approx((1.0, 0.0, 0.0), abs=0.0001)


def test_pmx_ik_solver_clamps_each_link_to_its_axis_limits() -> None:
    limits = ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05))
    preview = UiAvatarModelPreview(
        model_name="IK制限テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone(
                "左足ＩＫ",
                -1,
                (1.0, 1.0, 0.0),
                3,
                32,
                0.35,
                (2, 1),
                ik_link_limits=(limits, limits),
            ),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
    )

    solved = solve_ik_rotations(preview, {}, {0: (0.0, 0.0, 0.0)})

    assert solved
    for index in (1, 2):
        assert all(-0.05 <= value <= 0.05 for value in solved[index])


def test_pmx_physics_moves_a_dynamic_bone_within_a_safe_limit() -> None:
    preview = UiAvatarModelPreview(
        model_name="物理テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("中心", -1, (0.0, 0.0, 0.0)), UiAvatarBone("髪", 0, (0.0, 1.0, 0.0))),
        rigid_bodies=(
            UiAvatarRigidBody(0, (0.0, 0.0, 0.0), 1.0, 0.8, 0),
            UiAvatarRigidBody(1, (0.0, 1.0, 0.0), 1.0, 0.1, 1),
        ),
        physics_joints=(
            UiAvatarPhysicsJoint(
                0,
                1,
                (0.0, 0.5, 0.0),
                (8.0, 8.0, 8.0),
                translation_min=(-0.2, -0.2, -0.2),
                translation_max=(0.2, 0.2, 0.2),
            ),
        ),
    )
    physics = PmxPhysicsSimulator()

    offsets = {}
    for _ in range(10):
        offsets = physics.step(preview, 0.05, True)

    assert 1 in offsets
    assert offsets[1][1] < 0.0
    assert math.dist(offsets[1], (0.0, 0.0, 0.0)) <= 0.25
    assert physics.step(preview, 0.05, False) == {}


def test_pmx_physics_uses_fixed_steps_and_drops_excess_catch_up() -> None:
    preview = UiAvatarModelPreview(
        model_name="固定刻みテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("髪", -1, (0.0, 1.0, 0.0)),),
        rigid_bodies=(UiAvatarRigidBody(0, (0.0, 1.0, 0.0), 1.0, 0.1, 1),),
    )
    physics = PmxPhysicsSimulator()

    physics.step_pose(preview, 0.2, True)

    assert physics.last_substeps == 4
    assert physics.dropped_seconds > 0.0


def test_pmx_physics_respects_collision_groups_and_masks() -> None:
    def preview(mask: int) -> UiAvatarModelPreview:
        return UiAvatarModelPreview(
            model_name="衝突マスクテスト",
            vertex_count=0,
            face_count=0,
            points=(),
            bones=(
                UiAvatarBone("固定", -1, (0.0, 0.0, 0.0)),
                UiAvatarBone("動的", -1, (1.2, 0.0, 0.0)),
            ),
            rigid_bodies=(
                UiAvatarRigidBody(
                    0,
                    (0.0, 0.0, 0.0),
                    1.0,
                    0.1,
                    0,
                    collision_group=0,
                    non_collision_mask=mask,
                    size=(0.5, 0.5, 0.5),
                ),
                UiAvatarRigidBody(
                    1,
                    (1.2, 0.0, 0.0),
                    1.0,
                    0.1,
                    1,
                    collision_group=1,
                    size=(0.5, 0.5, 0.5),
                ),
            ),
        )

    colliding_physics = PmxPhysicsSimulator()
    colliding_preview = preview(0)
    colliding_physics.reset(colliding_preview)
    colliding_physics._states[1].position = (0.8, 0.0, 0.0)
    colliding = colliding_physics.step_pose(colliding_preview, 1.0 / 60.0, True)

    excluded_physics = PmxPhysicsSimulator()
    excluded_preview = preview(1 << 1)
    excluded_physics.reset(excluded_preview)
    excluded_physics._states[1].position = (0.8, 0.0, 0.0)
    excluded = excluded_physics.step_pose(excluded_preview, 1.0 / 60.0, True)

    assert colliding.translations[1][0] > excluded.translations[1][0]
    assert excluded.translations[1][0] == pytest.approx(-0.4)


def test_pmx_physics_preserves_rest_overlap_instead_of_exploding() -> None:
    preview = UiAvatarModelPreview(
        model_name="初期重なりテスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("固定", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("スカート", -1, (0.8, 0.0, 0.0)),
        ),
        rigid_bodies=(
            UiAvatarRigidBody(
                0,
                (0.0, 0.0, 0.0),
                1.0,
                0.1,
                0,
                size=(0.5, 0.5, 0.5),
            ),
            UiAvatarRigidBody(
                1,
                (0.8, 0.0, 0.0),
                1.0,
                0.1,
                1,
                size=(0.5, 0.5, 0.5),
            ),
        ),
    )

    pose = PmxPhysicsSimulator().step_pose(preview, 1.0 / 60.0, True)

    assert pose.translations[1][0] == pytest.approx(0.0)
    assert pose.translations[1][2] == pytest.approx(0.0)
    assert abs(pose.translations[1][1]) < 0.001


def test_pmx_physics_converts_world_displacement_to_parent_local_space() -> None:
    preview = UiAvatarModelPreview(
        model_name="座標変換テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("袖", 0, (0.0, 1.0, 0.0)),
        ),
        rigid_bodies=(UiAvatarRigidBody(1, (0.0, 1.0, 0.0), 1.0, 0.1, 1),),
    )
    quarter_turn = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    transforms = {
        0: (quarter_turn, (0.0, 0.0, 0.0)),
        1: (quarter_turn, (-1.0, 0.0, 0.0)),
    }
    physics = PmxPhysicsSimulator()
    physics.reset(preview)
    anchors = _body_anchors(preview, transforms)
    physics._states[0].position = (
        anchors[0].position[0] + 1.0,
        anchors[0].position[1],
        anchors[0].position[2],
    )

    pose = physics._bone_pose(preview, anchors)

    assert pose.translations[1] == pytest.approx((0.0, -1.0, 0.0))


def test_pmx_physics_mode_two_keeps_position_and_returns_rotation() -> None:
    preview = UiAvatarModelPreview(
        model_name="ボーン追従テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("袖", -1, (0.0, 1.0, 0.0)),),
        rigid_bodies=(UiAvatarRigidBody(0, (0.0, 1.0, 0.0), 1.0, 0.1, 2),),
    )
    physics = PmxPhysicsSimulator()
    physics.reset(preview)
    physics._states[0].rotation = (0.2, 0.0, 0.0)

    pose = physics.step_pose(preview, 1.0 / 60.0, True)

    assert pose.translations == {}
    assert pose.rotations[0][0] > 0.0


def test_dense_renderer_keeps_material_units_separate_even_with_same_texture() -> None:
    front = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="local/clothes.png",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        material_index=0,
    )
    back = UiAvatarTexturedTriangle(
        indexes=(2, 1, 0),
        uvs=((0.5, 1.0), (1.0, 0.0), (0.0, 0.0)),
        texture_path="local/clothes.png",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        material_index=1,
    )
    preview = UiAvatarModelPreview(
        model_name="材質順序テスト",
        vertex_count=3,
        face_count=2,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dense_textured_triangles=(front, back),
    )

    groups = _build_groups(preview)

    assert [group.material_index for group in groups] == [0, 1]


def test_dense_renderer_draws_transparent_materials_in_pmx_order_with_depth_write() -> None:
    triangles = tuple(
        UiAvatarTexturedTriangle(
            indexes=(0, 1, 2),
            uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
            texture_path="",
            diffuse=(1.0, 1.0, 1.0, alpha),
            material_index=material_index,
            is_transparent=alpha < 1.0,
        )
        for material_index, alpha in ((0, 0.5), (1, 1.0), (2, 0.4))
    )
    preview = UiAvatarModelPreview(
        model_name="透明材質順序テスト",
        vertex_count=3,
        face_count=3,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dense_textured_triangles=triangles,
    )

    groups = _build_groups(preview)
    source = inspect.getsource(DensePmxWidget._paint_model)

    assert [group.material_index for group in groups] == [0, 1, 2]
    assert source.count("for group in ready_groups:") == 2
    assert "_split_render_groups" not in source
    assert "glDepthMask(False)" not in source


def test_front_camera_converts_pmx_handedness_without_mirroring_x() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.resize(400, 400)
    renderer.set_camera(math.pi, 0.0, 1.0)

    matrix = renderer._mvp()
    left = matrix.map(QVector3D(-1.0, 0.0, 0.0))
    right = matrix.map(QVector3D(1.0, 0.0, 0.0))

    assert left.x() < right.x()
    assert _camera_rotate((1.0, 0.0, 0.0), math.pi, 0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert "glFrontFace(0x0900)" in inspect.getsource(DensePmxWidget._paint_model)
    renderer.close()


def test_vmd_camera_and_light_require_explicit_view_mode() -> None:
    QApplication.instance() or QApplication([])
    motion = UiVmdMotion(
        "VMD演出",
        30,
        (),
        camera_frames=(
            UiVmdCameraFrame(0, -20.0, (0.0, 5.0, 0.0), (0.1, 0.2, 0.0), view_angle=40),
        ),
        light_frames=(UiVmdLightFrame(0, (0.4, 0.5, 0.6), (1.0, 2.0, 3.0)),),
    )
    preview = UiAvatarModelPreview(
        "VMD演出",
        2,
        0,
        ((-1.0, 0.0, 0.0), (1.0, 10.0, 0.0)),
        vmd_motion=motion,
    )
    renderer = DensePmxWidget()
    renderer.resize(400, 400)
    renderer.set_preview(preview)
    renderer.advance_heavy_animation(0.01, True)

    manual_matrix = renderer._mvp()
    assert renderer._scene_light() == ((-0.35, 0.70, 0.55), (1.0, 1.0, 1.0))
    assert "待機（通常表示を維持）" in renderer.vmd_status_text

    renderer.set_view_mode("vmd")
    vmd_matrix = renderer._mvp()

    assert vmd_matrix != manual_matrix
    assert renderer._scene_light() == ((1.0, 2.0, 3.0), (0.4, 0.5, 0.6))
    assert "明示再生中" in renderer.vmd_status_text
    renderer.close()


def test_dense_renderer_resolves_vmd_disabled_ik_name_to_controller_index() -> None:
    QApplication.instance() or QApplication([])
    motion = UiVmdMotion(
        "IK切替",
        30,
        (),
        ik_frames=(UiVmdIkFrame(0, True, (UiVmdIkState("左足ＩＫ", False),)),),
    )
    preview = UiAvatarModelPreview(
        "IK切替",
        0,
        0,
        (),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (1.0, 1.0, 0.0), 3, 4, 0.35, (2, 1)),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
        vmd_motion=motion,
    )
    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    renderer._vmd_player.advance(preview, 0.01, True)

    assert renderer._disabled_ik_indexes() == frozenset({0})
    renderer.close()


@pytest.mark.parametrize("body_mask", [("upper_body",), ("full_body",), ()])
@pytest.mark.parametrize("gesture_ik_enabled", [False, True])
def test_conversation_ik_switch_respects_body_mask(
    body_mask: tuple[str, ...], gesture_ik_enabled: bool
) -> None:
    QApplication.instance() or QApplication([])
    idle = UiVmdMotion(
        "待機",
        30,
        (),
        ik_frames=(UiVmdIkFrame(0, True, (UiVmdIkState("左足ＩＫ", not gesture_ik_enabled),)),),
    )
    gesture = UiVmdMotion(
        "会話",
        30,
        (),
        ik_frames=(UiVmdIkFrame(0, True, (UiVmdIkState("左足ＩＫ", gesture_ik_enabled),)),),
    )
    preview = UiAvatarModelPreview(
        "IK所有範囲",
        0,
        0,
        (),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("上半身", -1, (0.0, 1.0, 0.0)),
        ),
        vmd_motion=idle,
    )
    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "neutral",
            0.0,
            "",
            "forward",
            "",
            "explain",
            "",
            gesture_motion=gesture,
            gesture_body_mask=body_mask,
        )
    )
    renderer._conversation_motion_active = True
    renderer._vmd_player.advance(preview, 0.01, True)
    renderer._gesture_vmd_player.advance_motion(preview, gesture, 0.01, True)
    expected_enabled = (
        not gesture_ik_enabled if body_mask == ("upper_body",) else gesture_ik_enabled
    )
    assert renderer._disabled_ik_indexes() == (frozenset() if expected_enabled else frozenset({0}))
    renderer.end_conversation_motion()
    assert renderer._disabled_ik_indexes() == (
        frozenset({0}) if gesture_ik_enabled else frozenset()
    )
    renderer.start_motion_preview(gesture)
    assert renderer._disabled_ik_indexes() == (
        frozenset() if gesture_ik_enabled else frozenset({0})
    )
    renderer.stop_motion_preview()
    assert renderer._disabled_ik_indexes() == (
        frozenset({0}) if gesture_ik_enabled else frozenset()
    )
    renderer.close()


def test_dense_renderer_keeps_idle_and_speaking_animation_state() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()

    renderer.set_animation(1.5, True, True)
    renderer.set_view_mode("bust")

    assert renderer._idle_phase == 1.5
    assert renderer._motion_enabled is True
    assert renderer._speaking is True
    assert renderer._view_mode == "bust"
    renderer.release_resources()
    assert renderer.is_available is False
    renderer.close()


def test_management_preview_uses_separate_clock_and_never_resumes_after_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    idle_motion = UiVmdMotion(
        "待機",
        30,
        (UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    )
    preview = UiAvatarModelPreview(
        model_name="管理プレビュー分離テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        vmd_motion=idle_motion,
    )
    managed_motion = UiVmdMotion(
        "一回だけ",
        1,
        (UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        loop=False,
    )
    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    monkeypatch.setattr(renderer, "_advance_physics", lambda *_args, **_kwargs: PhysicsPose())
    monkeypatch.setattr(renderer, "_queue_pose_mesh_refresh", lambda **_kwargs: None)

    renderer.start_motion_preview(managed_motion)

    assert renderer.is_motion_preview_active is True
    assert renderer._preview is preview
    assert renderer._preview.vmd_motion is idle_motion
    assert renderer._preview_vmd_player is not renderer._vmd_player

    renderer.advance_heavy_animation(1.0, True)
    assert renderer.is_motion_preview_active is False

    renderer.advance_heavy_animation(0.1, True)
    assert renderer.is_motion_preview_active is False
    renderer.close()


def test_interruption_discards_looping_management_preview() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="管理プレビュー割り込みテスト",
            vertex_count=0,
            face_count=0,
            points=(),
        )
    )
    renderer.start_motion_preview(
        UiVmdMotion(
            "繰り返し",
            30,
            (UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
            loop=True,
        )
    )

    renderer.interrupt_motion()

    assert renderer.is_motion_preview_active is False
    assert renderer._preview_motion is None
    renderer.close()


def test_dance_survives_voice_end_only_until_non_loop_vmd_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    motion = UiVmdMotion(
        "短いダンス",
        1,
        (UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        loop=False,
    )
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="ダンス終了テスト",
            vertex_count=0,
            face_count=0,
            points=(),
            bones=(UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),),
        )
    )
    monkeypatch.setattr(renderer, "_advance_physics", lambda *_args, **_kwargs: PhysicsPose())
    monkeypatch.setattr(renderer, "_queue_pose_mesh_refresh", lambda **_kwargs: None)
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "neutral",
            0.0,
            "",
            "forward",
            "",
            "dance",
            "",
            gesture_motion=motion,
            gesture_motion_type="dance",
        )
    )
    renderer.begin_conversation_motion()

    renderer.end_conversation_motion()
    assert renderer.continues_conversation_motion_without_voice is True

    renderer.advance_conversation_animation(1.0)
    assert renderer.continues_conversation_motion_without_voice is False
    renderer.close()


def test_dense_renderer_lightweight_animation_does_not_run_heavy_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    heavy_calls: list[str] = []

    monkeypatch.setattr(
        renderer._vmd_player,
        "advance",
        lambda *_args: heavy_calls.append("vmd") or ({}, {}, {}),
    )
    monkeypatch.setattr(
        renderer._physics,
        "step",
        lambda *_args: heavy_calls.append("physics") or {},
    )
    monkeypatch.setattr(
        renderer,
        "_refresh_pose_mesh",
        lambda **_kwargs: heavy_calls.append("mesh"),
    )

    renderer.set_lightweight_animation(1.5, True, True)

    assert renderer._idle_phase == 1.5
    assert renderer._motion_enabled is True
    assert renderer._speaking is True
    assert heavy_calls == []
    renderer.close()


def test_dense_renderer_heavy_animation_runs_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    calls: list[str] = []

    monkeypatch.setattr(
        renderer._vmd_player,
        "advance_motion",
        lambda *_args: calls.append("vmd") or ({}, {}, {}),
    )
    monkeypatch.setattr(
        renderer,
        "_advance_physics",
        lambda *_args: calls.append("physics") or PhysicsPose(),
    )
    renderer.advance_heavy_animation(0.5, True)
    assert renderer._pose_future is not None
    renderer._pose_future.result(timeout=1.0)
    renderer._poll_pose_build()

    assert calls == ["vmd", "physics"]
    assert renderer.heavy_update_count == 1
    assert renderer.last_heavy_update_ms >= 0.0
    renderer.close()


def test_dense_renderer_defers_physics_until_first_vmd_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    reset_calls: list[UiAvatarModelPreview | None] = []
    monkeypatch.setattr(renderer._physics, "reset", reset_calls.append)

    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="初回物理姿勢テスト",
            vertex_count=1,
            face_count=0,
            points=((0.0, 0.0, 0.0),),
        )
    )

    assert reset_calls == [None]
    renderer.close()


def test_dense_renderer_freezes_only_a_ready_large_pmx_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 0, 0),
        uvs=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    renderer._preview = UiAvatarModelPreview(
        model_name="GC周期停止テスト",
        vertex_count=1,
        face_count=50_000,
        points=((0.0, 0.0, 0.0),),
        dense_textured_triangles=(triangle,) * 50_000,
    )

    class ReadyPhysics:
        is_initialized = True

    renderer._physics = ReadyPhysics()  # type: ignore[assignment]
    calls: list[str] = []
    monkeypatch.setattr("multi_llm_companion.ui.dense_pmx_widget.gc.collect", lambda: 0)
    monkeypatch.setattr("multi_llm_companion.ui.dense_pmx_widget.gc.get_freeze_count", lambda: 0)
    monkeypatch.setattr(
        "multi_llm_companion.ui.dense_pmx_widget.gc.freeze", lambda: calls.append("freeze")
    )
    monkeypatch.setattr(
        "multi_llm_companion.ui.dense_pmx_widget.gc.unfreeze",
        lambda: calls.append("unfreeze"),
    )

    renderer._freeze_long_lived_pmx_graph()
    renderer._freeze_long_lived_pmx_graph()
    renderer._release_gc_freeze()

    assert calls == ["freeze", "unfreeze"]
    renderer.shutdown()


def test_pmx_preview_keeps_idle_motion_during_voice_without_conversation_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    lightweight_calls: list[tuple[float, bool, bool]] = []
    heavy_calls: list[tuple[float, bool]] = []
    conversation_calls: list[float] = []

    monkeypatch.setattr(
        widget._dense_view,
        "set_lightweight_animation",
        lambda phase, enabled, speaking: lightweight_calls.append((phase, enabled, speaking)),
    )
    monkeypatch.setattr(
        widget._dense_view,
        "advance_heavy_animation",
        lambda delta, enabled: heavy_calls.append((delta, enabled)),
    )
    monkeypatch.setattr(
        widget._dense_view,
        "advance_conversation_animation",
        lambda delta: conversation_calls.append(delta),
    )

    widget.set_voice_active(True)
    widget._last_heavy_at -= 0.36
    phase_at_start = widget._idle_phase
    widget._advance_lightweight_motion()
    widget._advance_heavy_motion()

    widget.set_voice_active(False)
    assert widget._idle_phase >= phase_at_start
    assert [call[2] for call in lightweight_calls] == [True, True, False]
    assert heavy_calls[0][0] == pytest.approx(0.36, abs=0.03)
    assert heavy_calls[0][1] is True
    assert conversation_calls == []

    widget._last_heavy_at -= 0.36
    widget._advance_heavy_motion()
    assert heavy_calls[1][0] == pytest.approx(0.36, abs=0.03)
    assert heavy_calls[1][1] is True
    assert widget._timer.interval() == 33
    assert widget._heavy_timer.interval() == 33
    assert widget._timer.timerType() == Qt.TimerType.PreciseTimer
    assert widget._heavy_timer.timerType() == Qt.TimerType.PreciseTimer
    assert widget._dense_view._pose_poll_timer.interval() == 2
    assert widget._dense_view._pose_poll_timer.timerType() == Qt.TimerType.PreciseTimer
    widget.close()


def test_pmx_preview_uses_conversation_motion_when_a_gesture_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    widget = PmxPreviewWidget()
    heavy_calls: list[tuple[float, bool]] = []
    conversation_calls: list[float] = []
    monkeypatch.setattr(
        widget._dense_view,
        "advance_heavy_animation",
        lambda delta, enabled: heavy_calls.append((delta, enabled)),
    )
    monkeypatch.setattr(
        widget._dense_view,
        "advance_conversation_animation",
        lambda delta: conversation_calls.append(delta),
    )
    widget._dense_view._conversation_motion_active = True
    widget.set_voice_active(True)
    widget._last_heavy_at -= 0.36

    widget._advance_heavy_motion()

    assert heavy_calls == []
    assert conversation_calls[0] == pytest.approx(0.36, abs=0.03)
    widget.close()


def test_prepared_numpy_groups_match_reference_bone_pose() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        normals=((0.0, 0.0, 1.0),) * 3,
    )
    preview = UiAvatarModelPreview(
        model_name="NumPy姿勢一致テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        bones=(
            UiAvatarBone("親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("子", 0, (0.0, 1.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),), ((0, 0.5), (1, 0.5)), ((1, 1.0),)),
        dense_textured_triangles=(triangle,),
    )
    rotations = {0: (0.1, 0.2, -0.1), 1: (-0.2, 0.0, 0.15)}
    reference = _build_groups(preview, rotations=rotations)
    prepared = _prepare_pose_data(preview)
    accelerated = _build_groups(preview, rotations=rotations, prepared_pose=prepared)

    assert prepared is not None
    assert tuple(indexes.tolist() for indexes in prepared.influence_vertex_indexes) == (
        [0, 1, 2],
        [1],
    )
    assert prepared.weighted_vertex_mask.tolist() == [True, True, True]
    assert accelerated[0].count == reference[0].count
    assert struct.unpack("<33f", accelerated[0].vertices) == pytest.approx(
        struct.unpack("<33f", reference[0].vertices),
        abs=0.00001,
    )


def test_prepared_pose_reuses_already_solved_ik_without_changing_vertices() -> None:
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        normals=((0.0, 0.0, 1.0),) * 3,
    )
    preview = UiAvatarModelPreview(
        model_name="IK再利用テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
        bones=(
            UiAvatarBone("左足ＩＫ", -1, (1.0, 1.0, 0.0), 3, 32, 0.35, (2, 1)),
            UiAvatarBone("左足", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左ひざ", 1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左足首", 2, (0.0, 2.0, 0.0)),
        ),
        vertex_weights=(((1, 1.0),), ((2, 1.0),), ((3, 1.0),)),
        dense_textured_triangles=(triangle,),
    )
    prepared = _prepare_pose_data(preview)
    solved = solve_ik_rotations(preview, {}, {})

    calculated = _build_groups(preview, prepared_pose=prepared)
    reused = _build_groups(
        preview,
        rotations=solved,
        prepared_pose=prepared,
        ik_already_solved=True,
    )

    assert prepared is not None
    assert struct.unpack("<33f", reused[0].vertices) == pytest.approx(
        struct.unpack("<33f", calculated[0].vertices),
        abs=0.00001,
    )


def test_conversation_motion_returns_gradually_instead_of_warping() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    motion = UiVmdMotion(
        name="手を上げる",
        frame_count=30,
        bone_frames=(UiVmdBoneFrame("頭", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        loop=True,
    )
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="復帰補間テスト",
            vertex_count=1,
            face_count=0,
            points=((0.0, 0.0, 0.0),),
            bones=(UiAvatarBone("頭", -1, (0.0, 0.0, 0.0)),),
            vertex_weights=(((0, 1.0),),),
        )
    )
    renderer._command = UiMmdAvatarCommand(
        "neutral",
        0.0,
        "",
        "forward",
        "",
        "head_scratch",
        "",
        gesture_motion=motion,
    )
    renderer._conversation_motion_active = True
    renderer._gesture_vmd_rotations = {0: (1.0, 0.0, 0.0)}
    renderer._next_mix_delta_seconds = 0.35
    before, _ik, _translations, _morphs = renderer._pose_settings()

    renderer.end_conversation_motion()
    after = renderer._motion_mixer.current.rotations

    assert before[0][0] == pytest.approx(1.0)
    assert 0.8 < after[0][0] < before[0][0]
    renderer.close()


def test_idle_vmd_does_not_override_a_motion_return_transition() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    idle_motion = UiVmdMotion(
        name="待機",
        frame_count=30,
        bone_frames=(UiVmdBoneFrame("頭", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
    )
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="復帰中の待機VMDテスト",
            vertex_count=0,
            face_count=0,
            points=(),
            bones=(UiAvatarBone("頭", -1, (0.0, 0.0, 0.0)),),
            vmd_motion=idle_motion,
        )
    )
    renderer._motion_mixer.mix(
        delta_seconds=0.35,
        gesture=MotionLayer(rotations={0: (1.0, 0.0, 0.0)}),
    )
    renderer._switch_idle_motion(
        idle_motion,
        source=MixedMotion(rotations={0: (1.0, 0.0, 0.0)}),
    )
    renderer._next_mix_delta_seconds = 0.05

    rotations, _ik, _translations, _morphs = renderer._pose_settings()

    assert 0.0 < rotations[0][0] < 1.0
    renderer.close()


def test_restarting_cloth_discards_only_the_previous_cloth_world() -> None:
    class Simulation:
        closed = False

        def close(self) -> None:
            self.closed = True

    state = _ClothState(simulation=Simulation(), initialized=True, error="前の失敗")

    state.restart()

    assert state.simulation is None
    assert state.initialized is False
    assert state.error == ""


def test_motion_mixer_keeps_hand_gesture_gaze_expression_and_mouth_together() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    preview = UiAvatarModelPreview(
        model_name="動作ミキサー統合テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        mouth_morphs={"a": ((0, 0.0, 0.2, 0.0),)},
        named_vertex_morphs={
            "あ": ((0, 0.0, 0.2, 0.0),),
            "笑顔": ((1, 0.1, 0.0, 0.0),),
        },
        bones=(
            UiAvatarBone("右手", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),
        ),
        dense_textured_triangles=(triangle,),
    )
    renderer.set_preview(preview)
    renderer._vmd_rotations = {0: (0.1, 0.0, 0.0)}
    renderer._vmd_morph_weights = {"あ": 1.0}
    renderer._command = UiMmdAvatarCommand(
        "happy",
        0.8,
        "笑顔",
        "left",
        "両目",
        "wave",
        "右手",
    )
    renderer._conversation_motion_active = True
    renderer._heavy_phase = 1.0
    renderer.set_mouth_shape("a")

    rotations, _ik, _translations, morphs = renderer._pose_settings()

    assert 0 in rotations
    assert abs(rotations[0][2]) > 0.1
    assert 1 in rotations
    assert rotations[1][1] > 0.0
    assert morphs["笑顔"] > 0.0
    assert "あ" not in morphs
    assert renderer._mouth_shape == "a"
    renderer.close()


def test_motion_mixer_applies_multiple_expression_preset_morphs() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="複数表情テスト",
            vertex_count=1,
            face_count=0,
            points=((0.0, 0.0, 0.0),),
            named_vertex_morphs={
                "笑い": ((0, 0.1, 0.0, 0.0),),
                "にこり": ((0, 0.0, 0.1, 0.0),),
            },
        )
    )
    renderer._command = UiMmdAvatarCommand(
        "happy",
        0.8,
        "笑い",
        "forward",
        "",
        "none",
        "",
        expression_preset="gentle_smile",
        expression_morphs=(("笑い", 0.32), ("にこり", 0.16)),
    )

    _rotations, _ik, _translations, morphs = renderer._pose_settings()

    assert morphs["笑い"] > 0.0
    assert morphs["にこり"] > 0.0
    renderer.close()


def test_conversation_vmd_runs_without_reloading_pmx_and_keeps_gaze() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    preview = UiAvatarModelPreview(
        model_name="会話VMD統合テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        bones=(
            UiAvatarBone("頭", -1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左目", 0, (0.2, 1.0, 0.0)),
            UiAvatarBone("右目", 0, (-0.2, 1.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),), ((1, 1.0),), ((2, 1.0),)),
    )
    motion = UiVmdMotion(
        "権利確認済みうなずき",
        30,
        (
            UiVmdBoneFrame("頭", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame(
                "頭",
                30,
                (0.0, 0.0, 0.0),
                (math.sin(0.2), 0.0, 0.0, math.cos(0.2)),
            ),
        ),
        loop=False,
    )
    renderer.set_preview(preview)
    preview_identity = renderer._preview
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            emotion="neutral",
            emotion_intensity=0.0,
            emotion_morph="",
            gaze="left",
            gaze_bone="左目",
            gesture="nod",
            gesture_target="",
            gesture_motion_id="nod_small",
            gesture_motion=motion,
            gesture_intensity_scale=0.5,
        )
    )

    revision = renderer.begin_conversation_motion()
    renderer.advance_conversation_animation(0.5)
    rotations, _ik, _translations, _morphs = renderer._pose_settings()

    assert revision is not None
    assert renderer._preview is preview_identity
    assert rotations[0][0] > 0.05
    assert rotations[1][1] > 0.1

    renderer.end_conversation_motion()
    assert renderer._gesture_vmd_rotations == {}
    renderer.close()


def test_conversation_gaze_keeps_the_idle_vmd_advancing() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    idle_motion = UiVmdMotion(
        "待機VMD",
        30,
        (
            UiVmdBoneFrame("センター", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            UiVmdBoneFrame("センター", 30, (0.0, 0.3, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
    )
    preview = UiAvatarModelPreview(
        model_name="会話視線中の待機VMDテスト",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        bones=(
            UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("左目", 0, (0.0, 1.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),), ((1, 1.0),)),
        vmd_motion=idle_motion,
    )
    renderer.set_preview(preview)
    renderer.set_mmd_command(UiMmdAvatarCommand("neutral", 0.0, "", "left", "左目", "none", ""))

    assert renderer.begin_conversation_motion() is not None
    renderer.advance_conversation_animation(0.5)
    rotations, _ik, translations, _morphs = renderer._pose_settings()

    assert renderer._vmd_translations[0][1] == pytest.approx(0.15, abs=0.01)
    assert translations[0][1] > 0.14
    assert rotations[1][1] > 0.1
    renderer.close()


def test_gaze_uses_weighted_eye_bones_when_both_eyes_is_only_a_relay() -> None:
    preview = UiAvatarModelPreview(
        model_name="中継ボーン視線テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        bones=(
            UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左目", -1, (0.2, 1.0, 0.0)),
            UiAvatarBone("右目", -1, (-0.2, 1.0, 0.0)),
        ),
        vertex_weights=(((1, 1.0),), ((1, 1.0),), ((2, 1.0),)),
    )

    assert _effective_gaze_bone_indexes(preview, "両目") == (1, 2)


def test_conversation_motion_starts_wave_and_gaze_then_releases_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    preview = UiAvatarModelPreview(
        model_name="会話動作寿命テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        bones=(
            UiAvatarBone("右腕", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),
            UiAvatarBone("左目", -1, (0.2, 1.0, 0.0)),
            UiAvatarBone("右目", -1, (-0.2, 1.0, 0.0)),
        ),
        vertex_weights=(((0, 1.0),), ((2, 1.0),), ((3, 1.0),)),
    )
    renderer.set_preview(preview)
    assert renderer._influential_bone_indexes == frozenset({0, 2, 3})
    assert renderer._natural_gaze_indexes == (2, 3)

    def fail_vertex_scan(*_args: object) -> bool:
        raise AssertionError("姿勢更新ごとに全頂点を再走査している")

    monkeypatch.setattr(dense_pmx_widget, "_bone_has_vertex_influence", fail_vertex_scan)
    renderer.set_mmd_command(UiMmdAvatarCommand("neutral", 0.0, "", "left", "両目", "wave", "右腕"))

    revision = renderer.begin_conversation_motion()
    rotations, _ik, _translations, _morphs = renderer._pose_settings()

    assert revision is not None
    assert abs(rotations[0][2]) > 0.1
    assert rotations[2][1] > 0.1
    assert rotations[3][1] > 0.1
    assert 1 not in rotations

    renderer.end_conversation_motion()
    released, _ik, _translations, _morphs = renderer._pose_settings()
    # 終了直後も前の位置を残し、複数フレームで待機姿勢へ戻す。
    assert released[0][2] < 0.0
    assert 0.0 < released[2][1] < rotations[2][1]
    assert 0.0 < released[3][1] < rotations[3][1]
    renderer.close()


def test_down_gaze_uses_the_pmx_visible_pitch_direction() -> None:
    QApplication.instance() or QApplication([])
    preview = UiAvatarModelPreview(
        model_name="下向き視線テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),),
        vertex_weights=(((0, 1.0),),),
    )
    command = UiMmdAvatarCommand(
        emotion="neutral",
        emotion_intensity=0.0,
        emotion_morph="",
        gaze="down",
        gaze_bone="両目",
        gesture="none",
        gesture_target="",
    )
    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    renderer.set_mmd_command(command)
    assert renderer.begin_conversation_motion() is not None
    renderer._next_mix_delta_seconds = 0.35

    rotations, _ik, _translations, _morphs = renderer._pose_settings()

    assert rotations[0][0] == pytest.approx(-0.12)
    renderer.close()

    fallback = PmxPreviewWidget()
    fallback.set_preview(preview)
    fallback.set_mmd_command(command)
    assert fallback._pose_rotations(preview)[0][0] == pytest.approx(-0.12)
    fallback.close()


def test_interrupting_motion_invalidates_an_old_pose_request() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="古い姿勢の破棄テスト",
            vertex_count=0,
            face_count=0,
            points=(),
        )
    )
    old_request = renderer._pose_build_request()

    renderer.interrupt_motion()
    new_request = renderer._pose_build_request()

    assert new_request.motion_revision > old_request.motion_revision
    assert old_request.motion_revision != renderer._motion_mixer.revision
    renderer.close()


def test_completed_old_pose_is_not_installed_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="計算済み旧姿勢の破棄テスト",
            vertex_count=0,
            face_count=0,
            points=(),
        )
    )
    old_request = renderer._pose_build_request()
    renderer._motion_mixer.interrupt()
    future: Future[tuple[object, list[object], float]] = Future()
    future.set_result((old_request, [], 1.0))
    renderer._pose_future = future  # type: ignore[assignment]
    installed: list[bool] = []
    monkeypatch.setattr(
        renderer,
        "_install_pose_groups",
        lambda *_args: installed.append(True),
    )

    renderer._poll_pose_build()

    assert installed == []
    renderer.close()


def test_natural_motion_uses_actual_hip_chest_and_neck_bone_heights() -> None:
    preview = UiAvatarModelPreview(
        model_name="自然動作支点テスト",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.0, 20.0, 0.0)),
        bones=(
            UiAvatarBone("腰", -1, (0.0, 11.0, 0.0)),
            UiAvatarBone("上半身2", 0, (0.0, 13.0, 0.0)),
            UiAvatarBone("首", 1, (0.0, 16.0, 0.0)),
        ),
    )

    assert _natural_motion_centers(preview) == (0.0, 11.0, 13.0, 16.0)


def test_vertex_shader_localizes_breath_weight_shift_and_nod() -> None:
    assert "float chestMask" in _VERTEX_SHADER
    assert "result.x *= 1.0 + naturalMotion.x * chestMask" in _VERTEX_SHADER
    assert "float weightMask" in _VERTEX_SHADER
    assert "float balanceMask" in _VERTEX_SHADER
    assert "rotateZ(-naturalMotion.w * 2.2 * balanceMask)" in _VERTEX_SHADER
    assert "float headMask" in _VERTEX_SHADER
    assert "naturalMotion.z + naturalPosture.y" in _VERTEX_SHADER
    assert "naturalSecondary.x * headMask" in _VERTEX_SHADER
    assert "result.y +=" not in _VERTEX_SHADER


def test_fragment_shader_flips_normals_for_double_sided_back_faces() -> None:
    assert "gl_FrontFacing ? normalize(surfaceNormal) : -normalize(surfaceNormal)" in (
        _FRAGMENT_SHADER
    )


def test_shader_uses_additional_uv_for_subtexture_sphere_mode() -> None:
    assert "in vec2 sphereUv;" in _VERTEX_SHADER
    assert "sphereMode == 3" in _FRAGMENT_SHADER
    assert "sphereTextureUv" in _FRAGMENT_SHADER


def test_natural_rest_pose_lowers_both_arms_without_moving_legs() -> None:
    preview = UiAvatarModelPreview(
        model_name="待機姿勢テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(
            UiAvatarBone("右肩", -1, (-0.2, 1.5, 0.0)),
            UiAvatarBone("右腕", 0, (-0.5, 1.4, 0.0)),
            UiAvatarBone("右ひじ", 1, (-1.0, 1.2, 0.0)),
            UiAvatarBone("左肩", -1, (0.2, 1.5, 0.0)),
            UiAvatarBone("左腕", 3, (0.5, 1.4, 0.0)),
            UiAvatarBone("左ひじ", 4, (1.0, 1.2, 0.0)),
            UiAvatarBone("右足", -1, (-0.2, 0.8, 0.0)),
            UiAvatarBone("左足", -1, (0.2, 0.8, 0.0)),
        ),
    )

    rotations = _natural_rest_rotations(preview)

    assert rotations[1][2] > 0.7
    assert rotations[4][2] < -0.7
    assert 6 not in rotations
    assert 7 not in rotations


def test_upper_body_vmd_mask_keeps_center_and_legs_owned_by_idle_motion() -> None:
    QApplication.instance() or QApplication([])
    preview = UiAvatarModelPreview(
        model_name="部位所有権テスト",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(
            UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("下半身", 0, (0.0, 1.0, 0.0)),
            UiAvatarBone("右足", 1, (-0.2, 0.5, 0.0)),
            UiAvatarBone("上半身", 0, (0.0, 1.2, 0.0)),
            UiAvatarBone("頭", 3, (0.0, 1.8, 0.0)),
            UiAvatarBone("右腕", 3, (-0.4, 1.5, 0.0)),
        ),
    )
    allowed = _body_mask_bone_indexes(preview, ("upper_body", "arms", "head"))

    assert allowed == frozenset({3, 4, 5})

    renderer = DensePmxWidget()
    renderer.set_preview(preview)
    renderer.set_mmd_command(
        UiMmdAvatarCommand(
            "neutral",
            0.0,
            "",
            "forward",
            "",
            "head_scratch",
            "",
            gesture_body_mask=("upper_body", "arms", "head"),
        )
    )
    renderer._gesture_vmd_rotations = {
        0: (0.1, 0.0, 0.0),
        2: (0.2, 0.0, 0.0),
        3: (0.3, 0.0, 0.0),
        5: (0.4, 0.0, 0.0),
    }
    renderer._gesture_vmd_translations = {
        0: (0.1, 0.0, 0.0),
        3: (0.2, 0.0, 0.0),
    }

    rotations, translations = renderer._managed_gesture_pose()

    assert rotations == {3: (0.3, 0.0, 0.0), 5: (0.4, 0.0, 0.0)}
    assert translations == {3: (0.2, 0.0, 0.0)}
    renderer.close()


def test_natural_rest_pose_does_not_override_vmd_bones() -> None:
    preview = UiAvatarModelPreview(
        model_name="VMD優先テスト",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        bones=(UiAvatarBone("右腕", -1, (0.0, 1.0, 0.0)),),
        vmd_motion=UiVmdMotion(
            name="確認用",
            frame_count=30,
            bone_frames=(UiVmdBoneFrame("右腕", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),),
        ),
    )

    assert _natural_rest_rotations(preview) == {}


def test_base_vmd_keeps_its_body_pose_while_natural_gaze_remains_active() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="VMD自然動作抑止テスト",
            vertex_count=3,
            face_count=0,
            points=((0.0, 0.0, 0.0), (0.2, 1.0, 0.0), (-0.2, 1.0, 0.0)),
            bones=(
                UiAvatarBone("センター", -1, (0.0, 0.0, 0.0)),
                UiAvatarBone("両目", 0, (0.0, 1.0, 0.0)),
                UiAvatarBone("左目", 1, (0.2, 1.0, 0.0)),
                UiAvatarBone("右目", 1, (-0.2, 1.0, 0.0)),
            ),
            vertex_weights=(((0, 1.0),), ((2, 1.0),), ((3, 1.0),)),
            vmd_motion=UiVmdMotion(
                name="確認用",
                frame_count=30,
                bone_frames=(
                    UiVmdBoneFrame("センター", 0, (0.0, 0.2, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    UiVmdBoneFrame(
                        "両目",
                        0,
                        (0.0, 0.0, 0.0),
                        (0.0, math.sin(0.05), 0.0, math.cos(0.05)),
                    ),
                ),
            ),
        )
    )
    renderer.set_natural_motion(
        NaturalMotionFrame(
            breath=0.01,
            gaze_yaw=0.15,
            gaze_pitch=0.04,
            weight_shift=0.01,
            blink_weight=1.0,
        )
    )

    rotations, _ik, translations, _morphs = renderer._pose_settings()

    assert renderer._effective_natural_motion() == NaturalMotionFrame()
    assert translations[0] == pytest.approx((0.0, 0.2, 0.0))
    assert 1 not in rotations
    assert rotations[2] == pytest.approx(rotations[3])
    assert rotations[2][0] > 0.01
    assert rotations[2][1] > 0.05
    renderer.close()


def test_fast_blink_keeps_the_current_mouth_shape() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    triangle = UiAvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="高速まばたき合成テスト",
            vertex_count=3,
            face_count=1,
            points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            mouth_morphs={"a": ((0, 0.0, 0.4, 0.0),)},
            named_vertex_morphs={"まばたき": ((0, 0.2, 0.0, 0.0),)},
            dense_textured_triangles=(triangle,),
        )
    )
    renderer.set_mouth_shape("a")
    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=1.0))

    x, y, z = struct.unpack_from("<3f", renderer._groups[0].vertices, 0)
    assert (x, y, z) == pytest.approx((0.2, 0.4, 0.0))

    renderer.set_natural_motion(NaturalMotionFrame(blink_weight=0.0))
    x, y, z = struct.unpack_from("<3f", renderer._groups[0].vertices, 0)
    assert (x, y, z) == pytest.approx((0.0, 0.4, 0.0))
    renderer.close()


def test_external_vmd_cannot_own_automatic_blink_morph() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="VMDまばたき優先順位テスト",
            vertex_count=1,
            face_count=0,
            points=((0.0, 0.0, 0.0),),
            named_vertex_morphs={"まばたき": ((0, 0.2, 0.0, 0.0),)},
        )
    )
    renderer._vmd_morph_weights = {"まばたき": 1.0}

    _rotations, _ik_offsets, _translations, morphs = renderer._pose_settings()

    assert "まばたき" not in morphs
    renderer.close()


def test_thinking_gaze_uses_both_weighted_eye_bones() -> None:
    QApplication.instance() or QApplication([])
    renderer = DensePmxWidget()
    renderer.set_preview(
        UiAvatarModelPreview(
            model_name="自然視線テスト",
            vertex_count=2,
            face_count=0,
            points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            bones=(
                UiAvatarBone("両目", -1, (0.0, 1.0, 0.0)),
                UiAvatarBone("左目", -1, (0.2, 1.0, 0.0)),
                UiAvatarBone("右目", -1, (-0.2, 1.0, 0.0)),
            ),
            vertex_weights=(((1, 1.0),), ((2, 1.0),)),
        )
    )
    renderer.set_natural_motion(NaturalMotionFrame(gaze_yaw=0.15, gaze_pitch=0.04))

    rotations, _ik, _translations, _morphs = renderer._pose_settings()

    assert rotations[1] == pytest.approx(rotations[2])
    assert rotations[1][0] > 0.01
    assert rotations[1][1] > 0.05
    assert 0 not in rotations
    renderer.close()
