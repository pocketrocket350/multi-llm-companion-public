from concurrent.futures import Future
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from test_numpy_skinning_cache import _fixture

from multi_llm_companion.ui.dense_pmx_widget import (
    DensePmxWidget,
    _build_groups,
    _prepare_pose_data,
)


def _model(ground=False):
    model, _, rotations = _fixture(42, ground)
    first = model.dense_textured_triangles[0]
    model = replace(
        model,
        dense_textured_triangles=tuple(
            replace(first, indexes=(i, i + 1, i + 2), material_index=int(i > 2))
            for i in range(0, 126, 3)
        ),
        named_vertex_morphs={"笑い": ((0, 0.1, 0.2, -0.1),)},
        mouth_morphs={"a": ((1, 0.0, 0.3, 0.0),)},
    )
    return model, rotations


def _shader_positions(group):
    source = np.frombuffer(group.vertices, np.float32).reshape(-1, 11)[:, :3]
    if group.gpu_pose is None:
        return source
    palette = group.gpu_pose.palette
    skin = np.frombuffer(group.skin_vertices, np.float32).reshape(-1, 8)
    positions = []
    # シェーダーへ渡した配列だけから、頂点ごとの式を独立して評価する。
    for point, influences in zip(source, skin, strict=True):
        result = np.zeros(3, np.float32)
        total = 0.0
        for bone, weight in zip(influences[:4].astype(int), influences[4:], strict=True):
            if weight > 0:
                result += (
                    palette[bone, :3, :3] @ (point - palette[bone, 4, :3]) + palette[bone, 3, :3]
                ) * weight
                total += weight
        if total == 0:
            result = point.copy()
        result[1] += group.gpu_pose.ground_offset
        positions.append(result)
    return np.asarray(positions)


@pytest.mark.parametrize("ground", (False, True))
@pytest.mark.parametrize("pose", ("rest", "rotated", "translated", "physics"))
@pytest.mark.parametrize("morph", (False, True))
def test_gpu_palette_preserves_cpu_vertices_and_materials(ground, pose, morph):
    model, rotations = _model(ground)
    prepared = _prepare_pose_data(model)
    before = prepared.points.copy()
    kwargs = dict(
        prepared_pose=prepared,
        rotations={} if pose == "rest" else rotations,
        physics_offsets={0: (0.1, 0.5, -0.1)} if pose == "translated" else {},
        physics_translations={2: (0.2, -0.1, 0.0)} if pose == "physics" else {},
        morph_weights={"笑い": 0.7} if morph else {},
        mouth_shape="a" if morph else "neutral",
        ik_already_solved=True,
    )
    cpu = _build_groups(model, **kwargs)
    gpu = _build_groups(model, **kwargs, gpu_skinning=True)
    assert gpu[0].gpu_pose is None
    assert gpu[1].gpu_pose is not None
    for expected, actual in zip(cpu, gpu, strict=True):
        expected_values = np.frombuffer(expected.vertices, np.float32).reshape(-1, 11)
        actual_values = np.frombuffer(actual.vertices, np.float32).reshape(-1, 11)
        np.testing.assert_allclose(_shader_positions(actual), expected_values[:, :3], atol=2e-6)
        np.testing.assert_array_equal(actual_values[:, 3:], expected_values[:, 3:])
        assert actual.diffuse == expected.diffuse
    np.testing.assert_array_equal(prepared.points, before)
    assert not prepared.cpu_vertex_mask.flags.writeable


def test_gpu_static_vertices_are_reused_for_changed_pose():
    model, rotations = _model()
    prepared = _prepare_pose_data(model)
    first = _build_groups(model, prepared_pose=prepared, gpu_skinning=True)
    second = _build_groups(model, rotations=rotations, prepared_pose=prepared, gpu_skinning=True)
    assert first[1].vertices is second[1].vertices is prepared.groups[1].rest_vertices
    assert first[1].skin_vertices is second[1].skin_vertices
    assert not np.array_equal(first[1].gpu_pose.palette, second[1].gpu_pose.palette)


@pytest.mark.parametrize("uv", [False, True])
def test_expression_material_updates_while_other_materials_use_gpu(uv):
    model, rotations = _model()
    model = replace(
        model,
        named_vertex_morphs={"じと目": ((3, 0.1, -0.2, 0.05),)},
        named_uv_morphs={"じと目": ((0, 3, 0.1, 0, 0, 0),)} if uv else {},
        dense_textured_triangles=tuple(
            replace(t, material_index=2 if t.indexes[0] > 5 else t.material_index)
            for t in model.dense_textured_triangles
        ),
    )
    prepared = _prepare_pose_data(model)
    options = dict(rotations=rotations, prepared_pose=prepared, morph_weights={"じと目": 0.3})
    cpu = _build_groups(model, **options)
    gpu = _build_groups(model, **options, gpu_skinning=True)
    assert gpu[1].gpu_pose is None
    assert gpu[2].gpu_pose is not None
    assert gpu[1].vertices != prepared.groups[1].rest_vertices
    for expected, actual in zip(cpu, gpu, strict=True):
        values = np.frombuffer(expected.vertices, np.float32).reshape(-1, 11)
        np.testing.assert_allclose(_shader_positions(actual), values[:, :3], atol=2e-6)
    cleared = _build_groups(model, prepared_pose=prepared, gpu_skinning=True)
    assert cleared[2].vertices is prepared.groups[2].rest_vertices


def test_gpu_does_not_limit_models_with_more_than_four_influences():
    model, rotations = _model()
    model = replace(model, vertex_weights=(((0, 0.2),) * 5,) * len(model.points))
    prepared = _prepare_pose_data(model)
    groups = _build_groups(model, rotations=rotations, prepared_pose=prepared, gpu_skinning=True)
    reference = _build_groups(model, rotations=rotations, prepared_pose=prepared)
    assert all(g.gpu_pose is None for g in groups)
    assert [g.vertices for g in groups] == [g.vertices for g in reference]


def test_skirt_material_keeps_cpu_cloth_and_normal_path():
    model, _ = _model()
    model = replace(model, bones=(replace(model.bones[0], name="Skirt_0"), *model.bones[1:]))
    prepared = _prepare_pose_data(model)
    assert all(not g.skin_vertices for g in prepared.groups)
    assert prepared.cpu_vertex_mask[:126].all()


def test_missing_bones_do_not_create_an_empty_gpu_texture():
    model, _ = _model()
    model = replace(model, bones=())
    prepared = _prepare_pose_data(model)
    assert all(not g.skin_vertices for g in prepared.groups)
    assert all(
        g.gpu_pose is None for g in _build_groups(model, prepared_pose=prepared, gpu_skinning=True)
    )


def test_gpu_keeps_uv_morphs_on_static_geometry():
    model, rotations = _model()
    model = replace(
        model, named_uv_morphs={"UV": ((0, 3, 0.1, -0.2, 0, 0), (1, 4, 0.2, 0.3, 0, 0))}
    )
    prepared = _prepare_pose_data(model)
    options = dict(rotations=rotations, morph_weights={"UV": 0.7}, prepared_pose=prepared)
    cpu = _build_groups(model, **options)
    gpu = _build_groups(model, **options, gpu_skinning=True)
    assert gpu[1].gpu_pose is not None
    for expected, actual in zip(cpu, gpu, strict=True):
        a = np.frombuffer(expected.vertices, np.float32).reshape(-1, 11)
        b = np.frombuffer(actual.vertices, np.float32).reshape(-1, 11)
        np.testing.assert_array_equal(a[:, 3:], b[:, 3:])


def test_gpu_upload_error_uses_existing_render_failure_path(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    errors = []
    try:
        request = widget._pose_build_request()
        future = Future()
        future.set_result((request, [], 1.0))
        widget._pose_future = future
        widget._pending_pose_request = request

        def fail(*args):
            raise RuntimeError("GPU upload failed")

        monkeypatch.setattr(widget, "_install_pose_groups", fail)
        monkeypatch.setattr(widget, "_handle_render_error", errors.append)
        widget._poll_pose_build()
        assert len(errors) == 1
        assert widget._pending_pose_request is None
        assert not widget._pose_poll_timer.isActive()
    finally:
        widget.shutdown()


def test_pose_install_does_not_retransfer_static_gpu_vertices():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    model, rotations = _model()
    widget = DensePmxWidget()

    class Buffer:
        uploads = 0

        def bind(self):
            pass

        def release(self):
            pass

        def allocate(self, *args):
            self.uploads += 1

        def destroy(self):
            pass

    try:
        widget.set_preview(model)
        prepared = widget._prepared_pose
        widget._groups = _build_groups(model, prepared_pose=prepared, gpu_skinning=True)
        buffer = Buffer()
        widget._groups[1].buffer = buffer
        updated = _build_groups(
            model, rotations=rotations, prepared_pose=prepared, gpu_skinning=True
        )
        widget._install_pose_groups(widget._pose_build_request(), updated)
        assert buffer.uploads == 0
        assert widget._groups[1].gpu_pose is updated[1].gpu_pose
    finally:
        widget._groups[1].buffer = None
        widget.shutdown()
