from __future__ import annotations

import math
import os
import struct
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui.dense_pmx_widget import (
    DensePmxWidget,
    _build_groups,
    _build_pose_groups_in_background,
    _prepare_pose_data,
)
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarRigidBody,
    UiAvatarTexturedTriangle,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms


def _model(mode: int = 1) -> UiAvatarModelPreview:
    return UiAvatarModelPreview(
        model_name="物理移動の表示テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
        bones=(
            UiAvatarBone("親", -1, (0.0, 0.0, 0.0)),
            UiAvatarBone("布", 0, (0.0, 1.0, 0.0), allows_translation=False),
        ),
        vertex_weights=(((1, 1.0),),) * 3,
        rigid_bodies=(UiAvatarRigidBody(1, (0.0, 1.0, 0.0), 1.0, 0.5, mode, size=(0.2,) * 3),),
        dense_textured_triangles=(
            UiAvatarTexturedTriangle(
                indexes=(0, 1, 2),
                uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                texture_path="",
                diffuse=(1.0,) * 4,
                normals=((0.0, 0.0, 1.0),) * 3,
            ),
        ),
    )


@pytest.mark.parametrize("mode", (0, 1, 2))
def test_only_dynamic_body_translation_bypasses_animation_lock(mode: int) -> None:
    model = _model(mode)
    animation = {1: (9.0, 0.0, 0.0)}
    physics = {1: (0.25, 0.0, 0.0)}
    assert bone_transforms(model, {}, animation)[1][1] == (0.0, 1.0, 0.0)
    result = bone_transforms(model, {}, animation, physics_translations=physics)
    assert result[1][1] == pytest.approx((0.25 if mode == 1 else 0.0, 1.0, 0.0))
    assert not model.bones[1].allows_translation
    assert physics == {1: (0.25, 0.0, 0.0)}


def test_physical_translation_keeps_parent_local_coordinates_and_unbound_lock() -> None:
    model = _model()
    result = bone_transforms(
        model,
        {0: (0.0, 0.0, math.pi / 2)},
        physics_translations={1: (0.25, 0.0, 0.0)},
    )
    assert result[1][1] == pytest.approx((-1.0, 0.25, 0.0))
    unbound = replace(model, rigid_bodies=())
    assert bone_transforms(unbound, {}, physics_translations={1: (2.0, 0.0, 0.0)})[1][1] == (
        0.0,
        1.0,
        0.0,
    )


@pytest.mark.parametrize("accelerated", (False, True))
def test_physics_translation_reaches_rendered_vertices(accelerated: bool) -> None:
    model = _model()
    prepared = _prepare_pose_data(model) if accelerated else None
    groups = _build_groups(
        model,
        prepared_pose=prepared,
        physics_offsets={1: (9.0, 0.0, 0.0)},
        physics_translations={1: (0.25, 0.0, 0.0)},
    )
    values = struct.unpack("<33f", groups[0].vertices)
    assert values[0:3] == pytest.approx((0.25, 1.0, 0.0))
    assert values[11:14] == pytest.approx((1.25, 1.0, 0.0))


def test_pose_request_snapshots_physics_and_updates_signature() -> None:
    QApplication.instance() or QApplication([])
    widget = DensePmxWidget()
    widget.set_preview(_model())
    widget._physics_offsets = {1: (0.25, 0.0, 0.0)}
    first = widget._pose_build_request()
    widget._physics_offsets[1] = (0.5, 0.0, 0.0)
    second = widget._pose_build_request()
    assert first.physics_translations == {1: (0.25, 0.0, 0.0)}
    assert first.signature != second.signature
    values = struct.unpack("<33f", _build_pose_groups_in_background(first)[1][0].vertices)
    assert values[0] == pytest.approx(0.25)
    widget.set_preview(_model(2))
    assert widget._pose_build_request().physics_translations == {}
    widget.close()


def test_stopping_animation_keeps_physical_translation() -> None:
    QApplication.instance() or QApplication([])
    widget = DensePmxWidget()
    widget.set_preview(_model())
    widget._physics_offsets = {1: (0.25, 0.0, 0.0)}
    widget.advance_heavy_animation(1 / 30, False)
    assert widget._pose_build_request().physics_translations == {1: (0.25, 0.0, 0.0)}
    widget.close()


def test_animation_translation_remains_unchanged_for_movable_bone() -> None:
    model = _model()
    model = replace(model, bones=(model.bones[0], replace(model.bones[1], allows_translation=True)))
    result = bone_transforms(
        model, {}, {1: (0.5, 0.0, 0.0)}, physics_translations={1: (0.25, 0.0, 0.0)}
    )
    assert result[1][1] == (0.5, 1.0, 0.0)
