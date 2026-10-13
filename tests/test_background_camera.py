"""人物と背景の相対位置・大きさを画角変更後も保つ。"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QVector3D
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.companion_background import _cover_rect, _standing_background_rect
from multi_llm_companion.ui.dense_pmx_widget import DensePmxWidget
from multi_llm_companion.ui.models import UiAvatarModelPreview


@pytest.fixture
def renderer():
    QApplication.instance() or QApplication([])
    widget = DensePmxWidget()
    widget._background_image = QImage(853, 1844, QImage.Format.Format_RGB32)
    widget.set_preview(
        UiAvatarModelPreview("画角検査", 4, 0, ((-2, 0, 0), (2, 0, 0), (-2, 10, 0), (2, 10, 0)))
    )
    yield widget
    widget.close()


@pytest.mark.parametrize("mode", ["full", "bust"])
@pytest.mark.parametrize("zoom", [0.4, 1.0, 3.0])
@pytest.mark.parametrize("size", [(320, 600), (900, 600)])
def test_feet_and_height_stay_at_same_background_coordinates(renderer, mode, zoom, size):
    renderer.resize(*size)
    renderer.set_view_mode(mode)
    renderer.set_camera(math.pi, 0, zoom)
    crop = renderer._background_crop()
    matrix = renderer._mvp()

    def source_position(point):
        projected = matrix.map(QVector3D(*point))
        return (
            (crop.x() + (projected.x() + 1) / 2 * crop.width()) / 853,
            (crop.y() + (1 - projected.y()) / 2 * crop.height()) / 1844,
        )

    # 両端が画面外でも、原本上の立ち位置と身長は同じ。
    assert source_position((0, 0, 0)) == pytest.approx((0.60, 0.95), abs=1e-6)
    assert source_position((0, 10, 0)) == pytest.approx((0.60, 0.15), abs=1e-6)
    assert crop.width() / crop.height() == pytest.approx(size[0] / size[1])


def test_background_magnifies_by_same_factor_as_camera(renderer):
    renderer.resize(600, 800)
    renderer.set_view_mode("bust")
    original = renderer._background_crop()
    renderer.set_camera(math.pi, 0, 2)
    enlarged = renderer._background_crop()
    assert enlarged.width() == pytest.approx(original.width() / 2)
    assert enlarged.height() == pytest.approx(original.height() / 2)


def test_changing_to_face_depth_does_not_move_the_room_in_world_space(renderer):
    renderer.resize(600, 800)
    renderer._bust_center = (0.3, 8.4, -0.4)
    renderer.set_view_mode("bust")
    crop = renderer._background_crop()
    foot = renderer._mvp().map(QVector3D(0, 0, 0))
    assert (crop.x() + (foot.x() + 1) / 2 * crop.width()) / 853 == pytest.approx(0.60)
    assert (crop.y() + (1 - foot.y()) / 2 * crop.height()) / 1844 == pytest.approx(0.95)


def test_background_does_not_follow_model_rotation_or_idle_breathing(renderer):
    renderer.resize(600, 800)
    original = renderer._background_crop()
    renderer.set_camera(1.8, 0.5, 1.0)
    renderer.set_lightweight_animation(30.0, True, True)
    assert renderer._background_crop() == original


def test_shared_background_rect_preserves_original_aspect_and_standing_point():
    feet = QPointF(320, 560)
    rect = _standing_background_rect(853, 1844, feet, 480)
    assert rect.width() / rect.height() == pytest.approx(853 / 1844)
    assert rect.left() + rect.width() * 0.60 == pytest.approx(feet.x())
    assert rect.top() + rect.height() * 0.95 == pytest.approx(feet.y())
    assert 480 / rect.height() == pytest.approx(0.80)


@pytest.mark.parametrize("without_model", [False, True])
def test_legacy_camera_and_missing_model_keep_background_cover(renderer, without_model):
    renderer.resize(700, 500)
    if without_model:
        renderer.set_preview(None)
    else:
        renderer.set_view_mode("vmd")
    assert renderer._background_crop() == _cover_rect(853, 1844, 700, 500)


@pytest.mark.parametrize("size", [(0, 0), (320, 0), (0, 320)])
def test_collapsed_background_does_not_divide_by_zero(renderer, size):
    renderer.resize(*size)
    crop = renderer._background_crop()
    assert all(math.isfinite(value) for value in crop.getRect())


def test_unavailable_background_image_keeps_safe_crop(renderer):
    renderer._background_image = QImage()
    crop = renderer._background_crop()
    assert all(math.isfinite(value) for value in crop.getRect())
