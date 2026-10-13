"""モデルを変更しない布計算と描画接続の回帰確認。"""

from copy import deepcopy
from dataclasses import replace
from math import sin
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import BoundingBox, LPoint3f, NodePath, TransformState
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui import dense_pmx_widget
from multi_llm_companion.ui.cloth_process import ProcessClothSimulation, _serve
from multi_llm_companion.ui.dense_pmx_widget import DensePmxWidget
from multi_llm_companion.ui.independent_cloth import ClothNonFiniteError, SkirtClothSimulation
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarTexturedTriangle,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms


def cloth_model():
    points = tuple((x * 0.5, 4.0 - y * 0.5, 0.0) for y in range(5) for x in range(5))
    faces = []
    for y in range(4):
        for x in range(4):
            i = y * 5 + x
            faces.extend(((i, i + 5, i + 1), (i + 1, i + 5, i + 6)))
    return UiAvatarModelPreview(
        "布テスト",
        len(points),
        len(faces),
        points,
        bones=(UiAvatarBone("下半身", -1, (0, 4, 0)), UiAvatarBone("Skirt_test", 0, (0, 4, 0))),
        vertex_weights=tuple(((1, 1.0),) for _ in points),
        dense_textured_triangles=tuple(
            UiAvatarTexturedTriangle(f, ((0, 0),) * 3, "", (1, 1, 1, 1)) for f in faces
        ),
    )


def test_cloth_moves_under_gravity_without_mutating_model():
    model = cloth_model()
    original = deepcopy(model)
    points = np.array(model.points)
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, points, transforms)
    assert cloth is not None
    try:
        positions = cloth.advance(0, transforms).copy()
        assert np.isfinite(positions).all()
        assert np.max(np.abs(positions - points[cloth.vertex_indexes])) > 0.001
        assert positions[12, 1] <= points[12, 1] + 0.03
        np.testing.assert_allclose(positions[:5], points[:5], atol=0.03)
        np.testing.assert_array_equal(cloth.advance(0, transforms), positions)
        assert model == original
        np.testing.assert_array_equal(points, np.array(model.points))
    finally:
        cloth.close()


def test_render_uses_separate_cloth_and_releases_on_model_change():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    model = cloth_model()
    original = deepcopy(model)
    widget = DensePmxWidget()
    try:
        widget.set_preview(model)
        widget._refresh_pose_mesh(force=True)
        cloth = widget._cloth_state.simulation
        assert cloth is not None
        values = np.frombuffer(widget._groups[0].vertices, np.float32).reshape(-1, 11)
        source = widget._prepared_pose.groups[0].vertex_indexes
        positions = np.array(model.points)
        positions[cloth.vertex_indexes] = cloth._positions
        np.testing.assert_allclose(values[:, :3], positions[source])
        assert model == original
        widget.set_preview(None)
        assert widget._cloth_state.simulation is None
    finally:
        widget.shutdown()


def test_unrelated_model_has_no_cloth():
    model = UiAvatarModelPreview("なし", 0, 0, ())
    assert SkirtClothSimulation.create(model, np.empty((0, 3)), {}) is None


@pytest.mark.parametrize("contact_bone", ["下半身", "左手首"])
def test_waist_contacts_keep_separate_surfaces_without_bridging_the_gap(contact_bone):
    model = cloth_model()
    bones = model.bones + (UiAvatarBone("左手首", 0, (0, 0, 0)),)
    bone_index = next(i for i, bone in enumerate(bones) if bone.name == contact_bone)
    points, weights, triangles = (
        list(model.points),
        list(model.vertex_weights),
        list(model.dense_textured_triangles),
    )
    for material, center in enumerate((-1.0, 1.0), start=1):
        first = len(points)
        points.extend(
            (center + x, y, z)
            for x, y, z in ((-0.2, -0.2, -0.2), (0.2, -0.2, -0.2), (0, 0.2, -0.2), (0, 0, 0.2))
        )
        weights.extend([((bone_index, 1.0),)] * 4)
        triangles.extend(
            UiAvatarTexturedTriangle(
                tuple(first + i for i in face),
                ((0, 0),) * 3,
                f"surface-{material}.png",
                (1, 1, 1, 1),
                material_index=material,
            )
            for face in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
        )
    model = replace(
        model,
        bones=bones,
        points=tuple(points),
        vertex_count=len(points),
        face_count=len(triangles),
        vertex_weights=tuple(weights),
        dense_textured_triangles=tuple(triangles),
    )
    original = deepcopy(model)
    cloth = SkirtClothSimulation.create(
        model, np.array(model.points), bone_transforms(model, {}, {})
    )
    assert cloth is not None
    probe = BulletRigidBodyNode("gap-probe")
    probe.add_shape(BulletSphereShape(0.003))
    path = NodePath(probe)
    cloth._world.attach_rigid_body(probe)
    try:
        body = next(node for i, node, _ in cloth._colliders if i == bone_index)
        for x in (-1.0, 1.0):
            path.set_pos(x * 0.08, 0, 0)
            assert cloth._world.contact_test_pair(body, probe).get_num_contacts() > 0
        path.set_pos(0, 0, 0)
        gap_contacts = cloth._world.contact_test_pair(body, probe).get_num_contacts()
        if contact_bone == "下半身":
            assert gap_contacts == 0
            assert body.get_num_shapes() == 2
        else:
            assert gap_contacts > 0
            assert body.get_num_shapes() == 1
        assert model == original
    finally:
        cloth._world.remove_rigid_body(probe)
        cloth.close()


@pytest.mark.parametrize("steps", (4, 12, 30))
def test_cloth_anchor_moves_at_frame_velocity_across_physics_substeps(steps):
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, np.array(model.points), transforms)
    assert cloth is not None
    world = cloth._world
    velocities = []
    try:
        cloth.advance(0, transforms)

        def record_step(*args):
            world.do_physics(*args)
            velocities.append(tuple(cloth._anchor.get_linear_velocity()))

        cloth._world = SimpleNamespace(do_physics=record_step)
        cloth.advance(steps / 120, bone_transforms(model, {}, {0: (0.01 * steps, 0, 0)}))
        assert len(velocities) == steps
        # 1フレーム分を最初の1/120秒へ押し込まず、接触速度を一定にする。
        np.testing.assert_allclose(velocities, [(0.096, 0, 0)] * steps, atol=1e-5)
    finally:
        cloth._world = world
        cloth.close()


def test_cloth_keeps_last_simulated_transform_until_a_substep_is_available():
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, np.array(model.points), transforms)
    assert cloth is not None
    try:
        cloth.advance(0, transforms)
        before = cloth._anchor_path.get_pos()
        cloth.advance(1 / 240, bone_transforms(model, {}, {0: (0.02, 0, 0)}))
        assert cloth._anchor_path.get_pos() == before
        assert cloth._previous_transforms == {i: transforms[i] for i in cloth.bone_indexes}
        target = bone_transforms(model, {}, {0: (0.04, 0, 0)})
        cloth.advance(1 / 120, target)
        assert cloth._anchor_path.get_x() == pytest.approx(0.04 * 0.08)
        assert cloth._previous_transforms == {i: target[i] for i in cloth.bone_indexes}
    finally:
        cloth.close()


@pytest.mark.parametrize("coordinate", [-1e20, 1e20])
def test_finite_but_exploded_cloth_stops_before_the_next_substep(coordinate):
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, np.array(model.points), transforms)
    assert cloth is not None
    body, world = cloth._cloth, cloth._world
    previous = cloth._positions.copy()
    bounds = BoundingBox(LPoint3f(coordinate, 0, 0), LPoint3f(coordinate, 1, 1))
    assert np.isfinite(tuple(bounds.get_min())).all()
    steps = []
    cloth._cloth = SimpleNamespace(get_aabb=lambda: bounds)
    cloth._world = SimpleNamespace(do_physics=lambda *args: steps.append(args))
    try:
        with pytest.raises(ClothNonFiniteError, match="モデル寸法"):
            cloth.advance(0, transforms)
        assert len(steps) == 1
        np.testing.assert_array_equal(cloth._positions, previous)
    finally:
        cloth._cloth, cloth._world = body, world
        cloth.close()


def test_cloth_bounds_follow_the_current_anchor_not_the_world_origin():
    model = cloth_model()
    translation = np.array((100, 200, -300))
    transforms = bone_transforms(model, {}, {0: tuple(translation)})
    cloth = SkirtClothSimulation.create(model, np.array(model.points) + translation, transforms)
    assert cloth is not None
    try:
        positions = cloth.advance(0, transforms)
        np.testing.assert_allclose(positions[:5] - translation, model.points[:5], atol=0.03)
    finally:
        cloth.close()


def test_cloth_rest_shape_does_not_inherit_a_deformed_native_skirt():
    model = cloth_model()
    transforms = bone_transforms(model, {}, {0: (0.4, 0.2, 0.1)})
    points = np.array(model.points)
    distorted = points.copy()
    distorted[10:, 1] += 1.5
    reference = SkirtClothSimulation.create(model, points, transforms)
    actual = SkirtClothSimulation.create(model, distorted, transforms)
    assert reference is not None and actual is not None
    try:
        np.testing.assert_allclose(actual.advance(0, transforms), reference.advance(0, transforms))
        np.testing.assert_array_equal(distorted[10:, 1], points[10:, 1] + 1.5)
    finally:
        reference.close()
        actual.close()


def test_moving_body_pushes_cloth_without_changing_source_geometry():
    model = cloth_model()
    center = (1.0, 3.0, -0.5)
    cube = tuple(
        (center[0] + x, center[1] + y, center[2] + z)
        for x in (-0.3, 0.3)
        for y in (-0.3, 0.3)
        for z in (-0.3, 0.3)
    )
    model = replace(
        model,
        points=model.points + cube,
        vertex_count=model.vertex_count + 8,
        bones=model.bones + (UiAvatarBone("左手首", -1, center),),
        vertex_weights=model.vertex_weights + (((2, 1.0),),) * 8,
    )
    original = deepcopy(model)
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, np.array(model.points), transforms)
    assert cloth is not None
    try:
        cloth.advance(0, transforms)
        for step in range(1, 61):
            transforms = bone_transforms(model, {}, {2: (0, 0, step / 60 * 0.6)})
            positions = cloth.advance(step / 60, transforms)
        assert positions[12, 2] > 0.25
        assert model == original
    finally:
        cloth.close()


def test_cloth_is_computed_by_pose_worker(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    thread_ids = []
    create = ProcessClothSimulation.create

    def record_create(*args):
        thread_ids.append(get_ident())
        return create(*args)

    try:
        widget.set_preview(cloth_model())
        monkeypatch.setattr(ProcessClothSimulation, "create", staticmethod(record_create))
        widget._queue_pose_mesh_refresh(force=True)
        widget._pose_future.result(timeout=3)
        widget._poll_pose_build()
        assert thread_ids and all(t != get_ident() for t in thread_ids)
        assert widget._cloth_state.simulation is not None
    finally:
        widget.shutdown()


def test_model_switch_does_not_recreate_closed_cloth_world(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    model = cloth_model()
    entered, release = Event(), Event()
    build = dense_pmx_widget._build_groups

    def blocked_build(preview, *args, **kwargs):
        if preview is model:
            entered.set()
            assert release.wait(timeout=3)
        return build(preview, *args, **kwargs)

    try:
        widget.set_preview(model)
        old_state = widget._cloth_state
        monkeypatch.setattr(dense_pmx_widget, "_build_groups", blocked_build)
        widget._queue_pose_mesh_refresh(force=True)
        future = widget._pose_future
        assert entered.wait(timeout=3)
        widget.set_preview(None)
        release.set()
        future.result(timeout=3)
        assert old_state.closed
        assert old_state.simulation is None
        assert widget._cloth_state is not old_state
        assert widget._cloth_state.simulation is None
    finally:
        release.set()
        widget.shutdown()


def test_old_cloth_time_does_not_rewind_or_move_colliders():
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    cloth = SkirtClothSimulation.create(model, np.array(model.points), transforms)
    assert cloth is not None
    try:
        positions = cloth.advance(1.0, transforms).copy()
        moved = bone_transforms(model, {}, {0: (1, 0, 0)})
        np.testing.assert_array_equal(cloth.advance(0.5, moved), positions)
        assert cloth._time == 1.0
        np.testing.assert_allclose(tuple(cloth._anchor_path.get_pos()), (0, 0.32, 0))
    finally:
        cloth.close()


def test_pruned_pose_keeps_non_cloth_geometry_and_texture_coordinates():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    model = cloth_model()
    model = replace(
        model,
        points=model.points + ((3, 3, 1), (4, 3, 1), (3, 2, 1)),
        vertex_count=28,
        vertex_weights=model.vertex_weights + (((0, 1.0),),) * 3,
        dense_textured_triangles=model.dense_textured_triangles
        + (
            UiAvatarTexturedTriangle(
                (25, 26, 27), ((0, 0),) * 3, "", (1, 1, 1, 1), material_index=1
            ),
        ),
    )
    widget = DensePmxWidget()
    try:
        widget.set_preview(model)
        request = replace(widget._pose_build_request(), rotations={0: (0.1, 0.2, 0.3)})
        build = dense_pmx_widget._build_pose_groups_in_background
        reference = build(replace(request, cloth_state=None))[1]
        build(request)
        groups = build(request)[1]
        assert groups[1].vertices == reference[1].vertices
        before = np.frombuffer(reference[0].vertices, np.float32).reshape(-1, 11)
        after = np.frombuffer(groups[0].vertices, np.float32).reshape(-1, 11)
        np.testing.assert_array_equal(before[:, 6:], after[:, 6:])
    finally:
        widget.shutdown()


def test_render_dependencies_include_ancestors_of_non_cloth_vertices():
    model = cloth_model()
    model = replace(
        model,
        points=model.points + ((3, 3, 1),),
        vertex_count=26,
        bones=model.bones + (UiAvatarBone("布以外の子", 1, (3, 3, 1)),),
        vertex_weights=model.vertex_weights + (((2, 1.0),),),
    )
    cloth = SkirtClothSimulation.create(
        model, np.array(model.points), bone_transforms(model, {}, {})
    )
    assert cloth is not None
    try:
        assert cloth.render_bone_indexes == frozenset((0, 1, 2))
    finally:
        cloth.close()


def test_render_and_cloth_share_one_pose_transform_without_changing_vertices(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    model = cloth_model()
    try:
        widget.set_preview(model)
        request = replace(
            widget._pose_build_request(), rotations={0: (0.1, 0.2, 0.3)}, ik_already_solved=True
        )
        dense_pmx_widget._build_pose_groups_in_background(request)
        request = replace(request, cloth_seconds=0.5)
        expected = dense_pmx_widget._build_groups(
            model,
            rotations=request.rotations,
            prepared_pose=request.prepared_pose,
            ik_already_solved=True,
        )
        dense_pmx_widget._apply_independent_cloth(request, expected)
        calls = []
        original = dense_pmx_widget.bone_transforms

        def counted(*args, **kwargs):
            calls.append(kwargs.get("required_indexes"))
            return original(*args, **kwargs)

        monkeypatch.setattr(dense_pmx_widget, "bone_transforms", counted)
        _, actual, _ = dense_pmx_widget._build_pose_groups_in_background(request)
        assert len(calls) == 1
        assert [g.vertices for g in actual] == [g.vertices for g in expected]
        assert widget._cloth_state.simulation.bone_indexes <= calls[0]
    finally:
        widget.shutdown()


def test_process_cloth_matches_original_for_moving_anchor_and_pending_clock():
    model = cloth_model()
    original = deepcopy(model)
    points = np.array(model.points)
    transforms = bone_transforms(model, {}, {})
    reference = SkirtClothSimulation.create(model, points, transforms)
    process = ProcessClothSimulation.create(model, points, transforms)
    assert reference is not None and process is not None
    try:
        for step in range(61):
            transforms = bone_transforms(model, {}, {0: (step / 120, 0, 0)})
            expected = reference.advance(step / 30, transforms)
            previous = process._positions.copy()
            process._begin_advance(step / 30, transforms)
            if step:
                np.testing.assert_array_equal(process.advance((step - 1) / 30, {}), previous)
            actual = process.advance(step / 30, transforms)
            np.testing.assert_array_equal(actual, expected)
        assert model == original
        np.testing.assert_array_equal(points, np.array(model.points))
    finally:
        reference.close()
        process.close()
    assert process._process.poll() == 0
    process.close()


def test_cloth_worker_does_not_accumulate_unused_transform_states():
    model = cloth_model()

    def requests():
        yield model, np.array(model.points), bone_transforms(model, {}, {})
        for step in range(2001):
            yield step / 30, bone_transforms(model, {}, {0: (sin(step / 120) * 0.1, 0, 0)})
        yield None

    class Connection:
        def __init__(self):
            self.requests = iter(requests())
            self.states = []
            self.closed = False

        def recv(self):
            return next(self.requests)

        def send(self, response):
            status, payload = response
            assert status == "ok"
            if isinstance(payload, np.ndarray):
                assert np.isfinite(payload).all()
                self.states.append(TransformState.get_num_unused_states())

        def close(self):
            self.closed = True

    connection = Connection()
    _serve(connection)

    assert connection.closed
    assert len(connection.states) == 2001
    assert max(connection.states[1000:]) <= connection.states[0] + 256


def test_process_cloth_closes_with_inflight_result_and_rejects_updates():
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    process = ProcessClothSimulation.create(model, np.array(model.points), transforms)
    assert process is not None
    process._begin_advance(0, transforms)
    process.close()
    assert process._process.poll() is not None
    with pytest.raises(RuntimeError, match="終了した"):
        process.advance(1, transforms)


def test_process_cloth_exits_when_parent_connection_closes():
    model = cloth_model()
    process = ProcessClothSimulation.create(
        model, np.array(model.points), bone_transforms(model, {}, {})
    )
    assert process is not None
    try:
        process._connection.close()
        assert process._process.wait(timeout=3) == 0
    finally:
        process.close()


def test_process_cloth_timeout_stops_child(monkeypatch):
    model = cloth_model()
    transforms = bone_transforms(model, {}, {})
    process = ProcessClothSimulation.create(model, np.array(model.points), transforms)
    assert process is not None
    try:
        monkeypatch.setattr(process._connection, "poll", lambda timeout: False)
        with pytest.raises(RuntimeError, match="応答がありません"):
            process.advance(0, transforms)
        assert process._process.poll() is not None
        assert process._closed
    finally:
        process.close()


def test_nonfinite_cloth_reinitializes_once_at_current_pose(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    try:
        widget.set_preview(cloth_model())
        widget._refresh_pose_mesh(force=True)
        broken = widget._cloth_state.simulation
        assert broken is not None
        create = ProcessClothSimulation.create
        created = []

        def fail_once(_seconds, _transforms):
            raise ClothNonFiniteError("独立した布計算の頂点が有限値ではありません")

        def recreate(*args):
            created.append(args)
            return create(*args)

        monkeypatch.setattr(broken, "advance", fail_once)
        monkeypatch.setattr(ProcessClothSimulation, "create", staticmethod(recreate))

        widget._heavy_phase += 1 / 30
        widget._refresh_pose_mesh(force=True)

        assert len(created) == 1
        assert widget._cloth_state.simulation is not broken
        assert not widget._cloth_state.error
    finally:
        widget.shutdown()


def test_failed_cloth_process_holds_last_frame_and_recovers_on_model_reload():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    widget = DensePmxWidget()
    model = cloth_model()
    try:
        widget.set_preview(model)
        widget._refresh_pose_mesh(force=True)
        process = widget._cloth_state.simulation
        assert process is not None
        previous = [g.vertices for g in widget._groups]
        process._process.kill()
        process._process.wait(timeout=3)
        widget._heavy_phase += 1 / 30
        widget._queue_pose_mesh_refresh(force=True)
        with pytest.raises(RuntimeError):
            widget._pose_future.result(timeout=3)
        widget._poll_pose_build()
        assert "布計算: 停止" in widget.physics_status_text
        assert [g.vertices for g in widget._groups] == previous
        widget._queue_pose_mesh_refresh(force=True)
        assert widget._pose_future is None
        widget.set_preview(None)
        widget.set_preview(model)
        widget._refresh_pose_mesh(force=True)
        assert not widget._cloth_state.error
        assert not widget._error_text
        assert widget._cloth_state.simulation._process.poll() is None
    finally:
        widget.shutdown()
