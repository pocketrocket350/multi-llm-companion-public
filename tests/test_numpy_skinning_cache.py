from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from multi_llm_companion.ui.dense_pmx_widget import _apply_prepared_bone_pose, _prepare_pose_data
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarTexturedTriangle,
)
from multi_llm_companion.ui.pmx_pose import bone_transforms


def _fixture(seed, ground):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(128, 3)).astype(np.float32)
    weights = (
        (),
        ((0, 1.0),),
        ((0, 0.2), (1, 0.3)),
        ((0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4)),
        ((-1, 1.0), (99, 1.0), (2, 0.0)),
        ((99, 1.0), (1, 2.0), (2, 1.0)),
    )
    model = UiAvatarModelPreview(
        model_name="骨変形キャッシュの回帰テスト",
        vertex_count=len(points),
        face_count=1,
        points=tuple(map(tuple, points.tolist())),
        bones=tuple(
            UiAvatarBone(
                "左足首" if ground and i == 0 else f"骨{i}",
                i - 1,
                (0.1 * i, 0.3 * i, -0.1 * i),
            )
            for i in range(4)
        ),
        vertex_weights=tuple(weights[i % len(weights)] for i in range(len(points))),
        dense_textured_triangles=(
            UiAvatarTexturedTriangle(
                indexes=(0, 1, 2),
                uvs=((0.0, 0.0),) * 3,
                texture_path="",
                diffuse=(1.0,) * 4,
                normals=((0.0, 0.0, 1.0),) * 3,
            ),
        ),
    )
    rotations = {i: tuple(rng.uniform(-0.4, 0.4, 3)) for i in range(4)}
    return model, _prepare_pose_data(model), rotations


def _reference(points, model, prepared, rotations):
    # 最適化前と同じ抽出・演算順序を独立した期待値として保持する。
    transforms = bone_transforms(model, rotations, {})
    matrices = np.asarray([transforms[i][0] for i in range(4)], dtype=np.float32)
    positions = np.asarray([transforms[i][1] for i in range(4)], dtype=np.float32)
    offsets = points - prepared.points
    output = np.zeros_like(points)
    for column, affected in enumerate(prepared.influence_vertex_indexes):
        selected = prepared.bone_indexes[affected, column]
        matrix = matrices[selected]
        rest = prepared.points - prepared.bone_rest_positions[prepared.bone_indexes[:, column]]
        local = rest[affected] + offsets[affected]
        weights = prepared.normalized_bone_weights[affected, column]
        position = positions[selected]
        for axis in range(3):
            output[affected, axis] += (
                matrix[:, axis, 0] * local[:, 0]
                + matrix[:, axis, 1] * local[:, 1]
                + matrix[:, axis, 2] * local[:, 2]
                + position[:, axis]
            ) * weights
    output[~prepared.weighted_vertex_mask] = points[~prepared.weighted_vertex_mask]
    if prepared.ground_height is not None:
        output[:, 1] += prepared.ground_height - float(
            output[prepared.ground_vertex_indexes, 1].min()
        )
    return output


@pytest.mark.parametrize("seed", (0, 42, 100))
@pytest.mark.parametrize("ground", (False, True))
@pytest.mark.parametrize("morph", (False, True))
def test_cached_skinning_matches_previous_float32_operations(seed, ground, morph):
    model, prepared, rotations = _fixture(seed, ground)
    points = prepared.points.copy()
    if morph:
        points[::3] += (0.1, -0.03, 0.02)
    points.setflags(write=False)
    before = points.copy()
    result = _apply_prepared_bone_pose(
        points, model, prepared, rotations, {}, {}, ik_already_solved=True
    )
    np.testing.assert_array_equal(result, _reference(points, model, prepared, rotations))
    np.testing.assert_array_equal(points, before)


def test_cached_influences_are_immutable_and_safe_for_parallel_poses():
    model, prepared, rotations = _fixture(42, True)
    for arrays in (
        prepared.bone_local_offsets,
        prepared.influence_bone_indexes,
        prepared.influence_weights,
    ):
        assert all(not a.flags.writeable and a.flags.c_contiguous for a in arrays)
    inputs = [prepared.points + np.float32(i * 0.01) for i in range(4)]

    def apply(points):
        return _apply_prepared_bone_pose(
            points, model, prepared, rotations, {}, {}, ik_already_solved=True
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(apply, inputs))
    for points, result in zip(inputs, results, strict=True):
        np.testing.assert_array_equal(result, _reference(points, model, prepared, rotations))


def test_unused_bones_do_not_change_skinning_result():
    model, prepared, rotations = _fixture(42, False)
    expected = _reference(prepared.points, model, prepared, rotations)
    model = replace(
        model,
        bones=model.bones + tuple(UiAvatarBone(f"未使用{i}", -1, (10, 10, 10)) for i in range(40)),
    )
    prepared = _prepare_pose_data(model)
    assert prepared.required_bone_indexes == frozenset(range(4))
    result = _apply_prepared_bone_pose(
        prepared.points, model, prepared, rotations, {}, {}, ik_already_solved=True
    )
    np.testing.assert_array_equal(result, expected)
