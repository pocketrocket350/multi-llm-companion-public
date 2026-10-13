from __future__ import annotations

import struct
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest

from multi_llm_companion.config import Avatar3DConfig, AvatarConfig, MmdMappingConfig
from multi_llm_companion.domain import AssistantResponse, EmotionState
from multi_llm_companion.integrations import (
    PmxPreviewError,
    build_avatar_controller,
    read_pmx_preview,
)
from multi_llm_companion.integrations.base import AvatarVmdCameraFrame, AvatarVmdMotion
from multi_llm_companion.integrations.pmx_preview import (
    _build_textured_triangles,
    _PmxMaterial,
    _PmxReader,
    _read_bones,
    _read_materials,
    _read_vertex_morphs,
    _resolve_mapped_bone,
)
from multi_llm_companion.integrations.vmd_motion import write_confirmation_vmd


def _write_minimum_pmx(
    path: Path,
    morph_name: str = "",
    group_morph_name: str = "",
    bone_name: str = "",
    texture_name: str = "",
    include_physics: bool = False,
    draw_outline: bool = False,
) -> None:
    globals_data = bytes((1, 0, 4, 4, 4, 4, 4, 4))
    texts = ("テストモデル", "Test model", "", "")
    data = bytearray(b"PMX ")
    data.extend(struct.pack("<fB", 2.0, len(globals_data)))
    data.extend(globals_data)
    for text in texts:
        encoded = text.encode("utf-8")
        data.extend(struct.pack("<i", len(encoded)))
        data.extend(encoded)
    data.extend(struct.pack("<i", 3))
    for position, edge_scale in zip(
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        (0.5, 1.0, 1.5),
        strict=True,
    ):
        data.extend(struct.pack("<3f", *position))
        data.extend(struct.pack("<3f", 0.0, 1.0, 0.0))
        data.extend(struct.pack("<2f", 0.0, 0.0))
        data.extend(struct.pack("<Bif", 0, 0, edge_scale))
    data.extend(struct.pack("<i3i", 3, 0, 1, 2))
    data.extend(struct.pack("<i", int(bool(texture_name))))
    if texture_name:
        encoded = texture_name.encode("utf-8")
        data.extend(struct.pack("<i", len(encoded)))
        data.extend(encoded)
    data.extend(struct.pack("<i", int(bool(texture_name))))
    if texture_name:
        for text in ("材質", "Material"):
            encoded = text.encode("utf-8")
            data.extend(struct.pack("<i", len(encoded)))
            data.extend(encoded)
        data.extend(struct.pack("<4f", 1.0, 0.8, 0.7, 1.0))
        data.extend(
            struct.pack(
                "<3ff3fB4ff",
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0x10 if draw_outline else 0,
                0.1,
                0.2,
                0.3,
                0.4,
                1.25,
            )
        )
        data.extend(struct.pack("<iiBBi", 0, -1, 0, 0, -1))
        data.extend(struct.pack("<i", 0))
        data.extend(struct.pack("<i", 3))
    data.extend(struct.pack("<i", int(bool(bone_name))))
    if bone_name:
        for text in (bone_name, bone_name):
            encoded = text.encode("utf-8")
            data.extend(struct.pack("<i", len(encoded)))
            data.extend(encoded)
        data.extend(struct.pack("<3fiih3f", 0.0, 0.0, 0.0, -1, 0, 0, 0.0, 0.0, 0.0))
    data.extend(struct.pack("<i", int(bool(morph_name)) + int(bool(group_morph_name))))
    if morph_name:
        for text in (morph_name, morph_name):
            encoded = text.encode("utf-8")
            data.extend(struct.pack("<i", len(encoded)))
            data.extend(encoded)
        data.extend(struct.pack("<BBiI3f", 1, 1, 1, 0, 0.0, 0.5, 0.0))
    if group_morph_name:
        for text in (group_morph_name, group_morph_name):
            encoded = text.encode("utf-8")
            data.extend(struct.pack("<i", len(encoded)))
            data.extend(encoded)
        data.extend(struct.pack("<BBiif", 1, 0, 1, 0, 0.5))
    data.extend(struct.pack("<i", 0))  # 表示枠
    if include_physics:
        data.extend(struct.pack("<i", 2))
        for name, position, mode in (("固定", (0.0, 0.0, 0.0), 0), ("髪", (0.0, 1.0, 0.0), 1)):
            for text in (name, name):
                encoded = text.encode("utf-8")
                data.extend(struct.pack("<i", len(encoded)))
                data.extend(encoded)
            collision_group = 0 if mode == 0 else 3
            non_collision_mask = 0 if mode == 0 else 5
            data.extend(struct.pack("<iBH", 0, collision_group, non_collision_mask))
            shape = 0 if mode == 0 else 2
            size = (1.0, 1.0, 1.0) if mode == 0 else (0.2, 0.8, 0.2)
            rotation = (0.0, 0.0, 0.0) if mode == 0 else (0.1, 0.2, 0.3)
            data.extend(
                struct.pack(
                    "<B3f3f3f5fB",
                    shape,
                    *size,
                    *position,
                    *rotation,
                    1.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    mode,
                )
            )
        data.extend(struct.pack("<i", 1))
        for text in ("髪ジョイント", "Hair joint"):
            encoded = text.encode("utf-8")
            data.extend(struct.pack("<i", len(encoded)))
            data.extend(encoded)
        data.extend(struct.pack("<Bii", 0, 0, 1))
        joint_values = (
            (0.0, 0.5, 0.0)
            + (0.1, 0.2, 0.3)
            + (-1.0, -2.0, -3.0)
            + (1.0, 2.0, 3.0)
            + (-0.1, -0.2, -0.3)
            + (0.1, 0.2, 0.3)
            + (8.0, 8.0, 8.0)
            + (4.0, 5.0, 6.0)
        )
        data.extend(
            struct.pack(
                "<24f",
                *joint_values,
            )
        )
    else:
        data.extend(struct.pack("<i", 0))  # 剛体
        data.extend(struct.pack("<i", 0))  # ジョイント
    path.write_bytes(data)


def test_pmx_preview_reads_geometry_without_loading_textures(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path)

    preview = read_pmx_preview(path)

    assert preview.model_name == "テストモデル"
    assert preview.vertex_count == 3
    assert preview.face_count == 1
    assert preview.points == ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    assert preview.triangles == ((0, 1, 2),)


def test_pmx_preview_reads_uv_and_local_texture_reference(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, texture_name="textures\\face.png")

    preview = read_pmx_preview(path)

    assert len(preview.textured_triangles) == 1
    triangle = preview.textured_triangles[0]
    assert triangle.indexes == (0, 1, 2)
    assert triangle.uvs == ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert triangle.texture_path == str(tmp_path / "textures" / "face.png")
    assert triangle.diffuse == pytest.approx((1.0, 0.8, 0.7, 1.0))
    assert preview.dense_textured_triangles == preview.textured_triangles


def test_pmx_preview_reads_material_outline(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, texture_name="face.png", draw_outline=True)

    triangle = read_pmx_preview(path).dense_textured_triangles[0]

    assert triangle.draw_outline is True
    assert triangle.edge_color == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert triangle.edge_size == pytest.approx(1.25)
    assert triangle.edge_scales == pytest.approx((0.5, 1.0, 1.5))


def test_pmx_preview_keeps_untextured_material_faces_for_dense_rendering(tmp_path: Path) -> None:
    triangles = _build_textured_triangles(
        tmp_path / "model.pmx",
        (),
        (_PmxMaterial(-1, (0.2, 0.3, 0.4, 1.0), 1),),
        [(0, (0, 1, 2))],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    assert len(triangles) == 1
    assert triangles[0].texture_path == ""
    assert triangles[0].diffuse == pytest.approx((0.2, 0.3, 0.4, 1.0))


def test_pmx_preview_keeps_each_material_index_for_draw_order(tmp_path: Path) -> None:
    triangles = _build_textured_triangles(
        tmp_path / "model.pmx",
        ("clothes.png",),
        (
            _PmxMaterial(0, (1.0, 1.0, 1.0, 1.0), 1),
            _PmxMaterial(0, (1.0, 1.0, 1.0, 1.0), 1),
        ),
        [(0, (0, 1, 2)), (1, (2, 1, 0))],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    assert [triangle.material_index for triangle in triangles] == [0, 1]


def test_pmx_preview_uses_explicit_back_material_without_interior_outline(
    tmp_path: Path,
) -> None:
    triangles = _build_textured_triangles(
        tmp_path / "model.pmx",
        ("skirt.png",),
        (
            _PmxMaterial(
                0,
                (1.0, 1.0, 1.0, 1.0),
                1,
                name="skirt",
                draw_flags=0x11,
            ),
            _PmxMaterial(
                0,
                (1.0, 1.0, 1.0, 1.0),
                1,
                name="skirt_back",
            ),
        ),
        [(0, (0, 1, 2)), (1, (2, 1, 0))],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    assert triangles[0].is_double_sided is False
    assert triangles[0].draw_outline is False
    assert triangles[1].is_double_sided is False


def test_pmx_preview_keeps_static_material_properties_for_gpu_rendering(tmp_path: Path) -> None:
    triangles = _build_textured_triangles(
        tmp_path / "model.pmx",
        ("base.png", "sphere.sph", "toon.bmp"),
        (
            _PmxMaterial(
                0,
                (0.8, 0.7, 0.6, 1.0),
                1,
                specular=(0.3, 0.2, 0.1),
                shininess=24.0,
                ambient=(0.1, 0.2, 0.3),
                draw_flags=0,
                sphere_texture_index=1,
                sphere_mode=2,
                toon_texture_index=2,
                edge_color=(0.2, 0.3, 0.4, 0.5),
                edge_size=1.5,
            ),
        ),
        [(0, (0, 1, 2))],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
    )

    triangle = triangles[0]

    assert triangle.sphere_texture_path == str(tmp_path / "sphere.sph")
    assert triangle.sphere_mode == 2
    assert triangle.toon_texture_path == str(tmp_path / "toon.bmp")
    assert triangle.is_double_sided is False
    assert triangle.specular == pytest.approx((0.3, 0.2, 0.1))
    assert triangle.ambient == pytest.approx((0.1, 0.2, 0.3))
    assert triangle.draw_outline is False
    assert triangle.edge_color == pytest.approx((0.2, 0.3, 0.4, 0.5))
    assert triangle.edge_size == pytest.approx(1.5)


def test_pmx_material_reader_keeps_shared_toon_reference() -> None:
    data = bytearray(struct.pack("<i", 1))
    for text in ("共有Toon材質", "Shared toon material"):
        encoded = text.encode("utf-8")
        data.extend(struct.pack("<i", len(encoded)))
        data.extend(encoded)
    data.extend(
        struct.pack("<4f3ff3fB4ff", *(1.0,) * 4, *(0.0,) * 3, 0.0, *(0.0,) * 3, 0, *(0.0,) * 4, 1.0)
    )
    data.extend(struct.pack("<iiBBB", -1, -1, 0, 1, 4))
    data.extend(struct.pack("<ii", 0, 3))

    materials = _read_materials(_PmxReader(BytesIO(data)), 1, 4)

    assert materials[0].toon_texture_index == -1
    assert materials[0].toon_shared_index == 4


def test_pmx_morph_reader_combines_additional_uv_and_material_group() -> None:
    data = bytearray(struct.pack("<i", 3))

    def write_text(value: str) -> None:
        encoded = value.encode("utf-8")
        data.extend(struct.pack("<i", len(encoded)))
        data.extend(encoded)

    for value in ("追加UV", "Additional UV"):
        write_text(value)
    data.extend(struct.pack("<BBiI4f", 4, 4, 1, 1, 0.2, -0.1, 0.3, 0.4))

    for value in ("材質変更", "Material change"):
        write_text(value)
    data.extend(struct.pack("<BBiiB", 4, 8, 1, 2, 0))
    material_values = (
        (0.5, 0.6, 0.7, 0.8)
        + (0.1, 0.2, 0.3)
        + (2.0,)
        + (0.2, 0.3, 0.4)
        + (0.1, 0.2, 0.3, 0.4)
        + (0.5,)
        + (0.9, 0.8, 0.7, 1.0)
        + (1.0, 0.9, 0.8, 1.0)
        + (0.8, 0.9, 1.0, 1.0)
    )
    data.extend(struct.pack("<28f", *material_values))

    for value in ("複合", "Combined"):
        write_text(value)
    data.extend(struct.pack("<BBiifif", 4, 0, 2, 0, 0.5, 1, 0.25))

    (
        _mouth,
        _emotion,
        vertex_names,
        named_vertex,
        morph_names,
        named_uv,
        named_material,
    ) = _read_vertex_morphs(
        _PmxReader(BytesIO(data)),
        1,
        3,
        4,
        4,
        4,
        4,
        4,
        {},
        {},
    )

    assert vertex_names == ()
    assert named_vertex == {}
    assert morph_names == ("追加UV", "材質変更", "複合")
    assert named_uv["複合"][0] == pytest.approx((1, 1, 0.1, -0.05, 0.15, 0.2))
    assert named_material["複合"][0].material_index == 2
    assert named_material["複合"][0].operation == 0
    assert named_material["複合"][0].weight_scale == pytest.approx(0.25)


def test_pmx_preview_reads_bone_and_vertex_weights(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, bone_name="目")

    preview = read_pmx_preview(path)

    assert preview.bones[0].name == "目"
    assert preview.bones[0].parent_index == -1
    assert preview.vertex_weights == (((0, 1.0),), ((0, 1.0),), ((0, 1.0),))


def test_pmx_bone_reader_keeps_inheritance_axes_order_and_ik_limits() -> None:
    data = bytearray(struct.pack("<i", 3))

    def write_text(value: str) -> None:
        encoded = value.encode("utf-8")
        data.extend(struct.pack("<i", len(encoded)))
        data.extend(encoded)

    def write_header(
        name: str,
        position: tuple[float, float, float],
        parent: int,
        layer: int,
        flags: int,
    ) -> None:
        write_text(name)
        write_text(name)
        data.extend(struct.pack("<3fiiH3f", *position, parent, layer, flags, 0.0, 1.0, 0.0))

    write_header("付与親", (0.0, 0.0, 0.0), -1, 0, 0x0002 | 0x0004)
    flags = 0x0002 | 0x0004 | 0x0100 | 0x0200 | 0x0400 | 0x0800 | 0x1000
    write_header("付与子", (0.0, 1.0, 0.0), 0, 5, flags)
    data.extend(struct.pack("<if", 0, 0.5))
    data.extend(struct.pack("<3f", 0.0, 0.0, 1.0))
    data.extend(struct.pack("<3f3f", 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    write_header("足ＩＫ", (1.0, 1.0, 0.0), -1, 1, 0x0002 | 0x0020)
    data.extend(struct.pack("<iifI", 1, 12, 0.4, 1))
    data.extend(struct.pack("<iB3f3f", 1, 1, -1.0, 0.0, 0.0, -0.1, 0.0, 0.0))

    bones = _read_bones(_PmxReader(BytesIO(data)), encoding=1, bone_index_size=4)

    assert bones[1].deform_layer == 5
    assert bones[1].inherit_parent_index == 0
    assert bones[1].inherit_weight == pytest.approx(0.5)
    assert bones[1].inherit_rotation is True
    assert bones[1].inherit_translation is True
    assert bones[1].fixed_axis == pytest.approx((0.0, 0.0, 1.0))
    assert bones[1].local_axis_x == pytest.approx((0.0, 1.0, 0.0))
    assert bones[1].after_physics is True
    assert bones[2].ik_iterations == 12
    assert bones[2].ik_link_limits[0] is not None
    assert bones[2].ik_link_limits[0][0] == pytest.approx((-1.0, 0.0, 0.0))


def test_pmx_preview_reads_rigid_bodies_and_joints(tmp_path: Path) -> None:
    path = tmp_path / "physics-model.pmx"
    _write_minimum_pmx(path, bone_name="髪", include_physics=True)

    preview = read_pmx_preview(path)

    assert len(preview.rigid_bodies) == 2
    assert preview.rigid_bodies[1].name == "髪"
    assert preview.rigid_bodies[1].mode == 1
    assert preview.rigid_bodies[1].collision_group == 3
    assert preview.rigid_bodies[1].non_collision_mask == 5
    assert preview.rigid_bodies[1].shape == 2
    assert preview.rigid_bodies[1].size == pytest.approx((0.2, 0.8, 0.2))
    assert preview.rigid_bodies[1].rotation == pytest.approx((0.1, 0.2, 0.3))
    assert preview.rigid_bodies[1].angular_damping == pytest.approx(0.2)
    assert preview.rigid_bodies[1].restitution == pytest.approx(0.3)
    assert preview.rigid_bodies[1].friction == pytest.approx(0.4)
    assert len(preview.physics_joints) == 1
    assert preview.physics_joints[0].name == "髪ジョイント"
    assert preview.physics_joints[0].spring_translation == pytest.approx((8.0, 8.0, 8.0))
    assert preview.physics_joints[0].rotation == pytest.approx((0.1, 0.2, 0.3))
    assert preview.physics_joints[0].translation_min == pytest.approx((-1.0, -2.0, -3.0))
    assert preview.physics_joints[0].translation_max == pytest.approx((1.0, 2.0, 3.0))
    assert preview.physics_joints[0].rotation_min == pytest.approx((-0.1, -0.2, -0.3))
    assert preview.physics_joints[0].rotation_max == pytest.approx((0.1, 0.2, 0.3))
    assert preview.physics_joints[0].spring_rotation == pytest.approx((4.0, 5.0, 6.0))


def test_pmx_preview_rejects_non_pmx_file(tmp_path: Path) -> None:
    path = tmp_path / "not-model.txt"
    path.write_text("not a model", encoding="utf-8")

    with pytest.raises(PmxPreviewError, match=".pmx"):
        read_pmx_preview(path)


def test_pmx_preview_reads_configured_vertex_mouth_morph(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, morph_name="あ")

    preview = read_pmx_preview(path, mouth_morph_names={"a": "あ"})

    assert preview.mouth_morphs == {"a": ((0, 0.0, 0.5, 0.0),)}
    assert preview.vertex_morph_names == ("あ",)


def test_pmx_preview_reads_configured_vertex_emotion_morph(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, morph_name="笑顔")

    preview = read_pmx_preview(path, emotion_morph_names={"happy": "笑顔"})

    assert preview.emotion_morphs == {"happy": ((0, 0.0, 0.5, 0.0),)}
    assert preview.vertex_morph_names == ("笑顔",)


def test_pmx_preview_expands_configured_group_emotion_morph(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(
        path,
        morph_name="笑顔の基本",
        group_morph_name="笑顔",
    )

    preview = read_pmx_preview(path, emotion_morph_names={"happy": "笑顔"})

    assert preview.emotion_morphs == {"happy": ((0, 0.0, 0.25, 0.0),)}


def test_pmx_preview_ignores_circular_group_morph(tmp_path: Path) -> None:
    path = tmp_path / "model.pmx"
    _write_minimum_pmx(path, group_morph_name="循環")

    preview = read_pmx_preview(path, emotion_morph_names={"happy": "循環"})

    assert preview.emotion_morphs == {}


def test_pmx_controller_reads_only_models_under_local_mmd_folder(tmp_path: Path) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path)
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
        )
    )

    controller = build_avatar_controller(config, tmp_path)

    assert controller.is_available is True, controller.status_text
    assert controller.model_preview is not None
    outside = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="outside.pmx",
        )
    )
    assert build_avatar_controller(outside, tmp_path).is_available is False


def test_pmx_controller_loads_only_cataloged_compatible_vmd(tmp_path: Path) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    motion_dir = tmp_path / "local" / "motions"
    license_dir = tmp_path / "local" / "licenses"
    model_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    motion_path = motion_dir / "nod.vmd"
    _write_minimum_pmx(model_path, bone_name="頭")
    write_confirmation_vmd(motion_path, ("頭",))
    (license_dir / "nod.yaml").write_text(
        "title: 自作確認モーション\n"
        "author: テスト作者\n"
        "obtained_date: 2026-07-24\n"
        "allowed_use: アプリ内確認\n"
        "credit_required: false\n"
        "redistribution_allowed: false\n"
        "usage_confirmed: true\n",
        encoding="utf-8",
    )
    (tmp_path / "local" / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  nod_small:\n"
        "    file: local/motions/nod.vmd\n"
        "    motion_type: reaction\n"
        "    action: nod\n"
        "    moods: [neutral]\n"
        "    energy: medium\n"
        "    weight: 1\n"
        "    audio_required: false\n"
        "    category: head\n"
        "    loop: false\n"
        "    license_record: local/licenses/nod.yaml\n"
        "    compatibility:\n"
        "      model_names: [テストモデル]\n"
        "      required_bones: [頭]\n",
        encoding="utf-8",
    )
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            vmd_catalog_path="local/vmd_catalog.yaml",
            vmd_motion_id="nod_small",
        )
    )

    controller = build_avatar_controller(config, tmp_path)

    assert controller.is_available is True
    assert controller.model_preview is not None
    assert controller.model_preview.vmd_motion is not None
    assert controller.model_preview.vmd_motion.name == "nod（権利確認済み）"


def test_pmx_controller_selects_cataloged_vmd_as_conversation_gesture(tmp_path: Path) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    motion_dir = tmp_path / "local" / "motions"
    license_dir = tmp_path / "local" / "licenses"
    model_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    motion_path = motion_dir / "nod.vmd"
    broken_motion_path = motion_dir / "broken.vmd"
    _write_minimum_pmx(model_path, bone_name="頭")
    write_confirmation_vmd(motion_path, ("頭",))
    broken_motion_path.write_bytes(b"not a vmd")
    (license_dir / "nod.yaml").write_text(
        "title: 自作確認モーション\n"
        "author: テスト作者\n"
        "obtained_date: 2026-07-24\n"
        "allowed_use: アプリ内確認\n"
        "credit_required: false\n"
        "redistribution_allowed: false\n"
        "usage_confirmed: true\n",
        encoding="utf-8",
    )
    (tmp_path / "local" / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  nod_small:\n"
        "    file: local/motions/nod.vmd\n"
        "    intent: small_nod\n"
        "    category: head\n"
        "    body_mask: [head, neck]\n"
        "    loop: false\n"
        "    cooldown_seconds: 8\n"
        "    license_record: local/licenses/nod.yaml\n"
        "    compatibility:\n"
        "      model_names: [テストモデル]\n"
        "      required_bones: [頭]\n"
        "  broken_motion:\n"
        "    file: local/motions/broken.vmd\n"
        "    motion_type: reaction\n"
        "    action: nod\n"
        "    category: head\n"
        "    license_record: local/licenses/nod.yaml\n"
        "    compatibility:\n"
        "      required_bones: [頭]\n",
        encoding="utf-8",
    )
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            vmd_catalog_path="local/vmd_catalog.yaml",
            motion_library_enabled=True,
        )
    )
    controller = build_avatar_controller(config, tmp_path)

    controller.apply_response(AssistantResponse(speech="はい", gesture="nod"))

    command = controller.mmd_command
    assert command is not None
    assert command.gesture == "nod"
    assert command.gesture_motion_id == "nod_small"
    assert command.gesture_body_mask == ("head", "neck")
    assert command.gesture_motion is not None
    assert command.gesture_motion.name == "nod（会話ジェスチャー・権利確認済み）"
    assert command.gesture_target == ""
    assert controller.model_preview is not None
    assert controller.model_preview.vmd_motion is None
    assert "会話VMD互換: 対応" in controller.status_text
    assert "利用不可 1件" in controller.status_text

    controller.apply_response(AssistantResponse(speech="もう一度", gesture="nod"))

    repeated = controller.mmd_command
    assert repeated is not None
    assert repeated.gesture == "none"
    assert repeated.gesture_motion is None


def test_motion_library_keeps_pmx_available_when_selected_idle_is_invalid(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    motion_dir = tmp_path / "local" / "motions"
    license_dir = tmp_path / "local" / "licenses"
    model_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    _write_minimum_pmx(model_dir / "test-model.pmx", bone_name="頭")
    (motion_dir / "broken.vmd").write_bytes(b"not a vmd")
    (license_dir / "motion.yaml").write_text(
        "title: 自作確認モーション\n"
        "author: テスト作者\n"
        "obtained_date: 2026-08-27\n"
        "allowed_use: 自動テスト\n"
        "credit_required: false\n"
        "redistribution_allowed: false\n"
        "usage_confirmed: true\n",
        encoding="utf-8",
    )
    (tmp_path / "local" / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  broken_idle:\n"
        "    file: local/motions/broken.vmd\n"
        "    motion_type: idle\n"
        "    action: idle\n"
        "    category: idle\n"
        "    license_record: local/licenses/motion.yaml\n"
        "    compatibility:\n"
        "      required_bones: [頭]\n",
        encoding="utf-8",
    )
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            vmd_catalog_path="local/vmd_catalog.yaml",
            vmd_motion_id="broken_idle",
            motion_library_enabled=True,
        )
    )

    controller = build_avatar_controller(config, tmp_path)

    assert controller.is_available is True
    assert controller.model_preview is not None
    assert controller.model_preview.vmd_motion is None
    assert "利用不可 1件" in controller.status_text


def test_motion_library_skips_camera_only_conversation_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    motion_dir = tmp_path / "local" / "motions"
    license_dir = tmp_path / "local" / "licenses"
    model_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    license_dir.mkdir(parents=True)
    _write_minimum_pmx(model_dir / "test-model.pmx", bone_name="頭")
    (motion_dir / "camera.vmd").write_bytes(b"placeholder")
    (license_dir / "motion.yaml").write_text(
        "title: 自作確認モーション\n"
        "author: テスト作者\n"
        "obtained_date: 2026-08-27\n"
        "allowed_use: 自動テスト\n"
        "credit_required: false\n"
        "redistribution_allowed: false\n"
        "usage_confirmed: true\n",
        encoding="utf-8",
    )
    (tmp_path / "local" / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  camera_reaction:\n"
        "    file: local/motions/camera.vmd\n"
        "    motion_type: reaction\n"
        "    action: nod\n"
        "    category: camera\n"
        "    license_record: local/licenses/motion.yaml\n"
        "    compatibility:\n"
        "      required_bones: [頭]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "multi_llm_companion.integrations.pmx_preview.read_vmd_motion",
        lambda *_args, **_kwargs: AvatarVmdMotion(
            name="camera",
            frame_count=30,
            bone_frames=(),
            camera_frames=(
                AvatarVmdCameraFrame(
                    0,
                    35.0,
                    (0.0, 10.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
            ),
        ),
    )
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            vmd_catalog_path="local/vmd_catalog.yaml",
            motion_library_enabled=True,
        )
    )

    controller = build_avatar_controller(config, tmp_path)
    controller.apply_response(AssistantResponse(speech="はい", gesture="nod"))

    assert controller.is_available is True, controller.status_text
    assert controller.mmd_command is not None
    assert controller.mmd_command.gesture_motion is None
    assert controller.mmd_command.gesture == "none"
    assert controller.mmd_command.gesture_motion_id == ""
    assert "利用不可 1件" in controller.status_text


@pytest.mark.parametrize("gesture", ["wave", "nod", "small_nod"])
def test_pmx_controller_keeps_emotion_and_gaze_without_unregistered_gesture(
    tmp_path: Path, gesture: str
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path, morph_name="笑顔")
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            mmd_mapping=MmdMappingConfig(
                emotion_morphs={"happy": "笑顔"},
                gaze_bones={"left": "両目"},
                pose_bones={"head": "頭", "upper_body": "上半身"},
                gesture_targets={"wave": "手を振る", "nod": "頭"},
            ),
        )
    )
    controller = build_avatar_controller(config, tmp_path)
    controller.apply_response(
        AssistantResponse(
            speech="こんにちは",
            emotion="happy",
            emotion_intensity=0.7,
            gaze="left",
            gesture=gesture,
        )
    )
    controller.apply_emotion_state(
        EmotionState(
            conversation_id="test",
            emotion="happy",
            intensity=0.5,
            updated_at=datetime.now(UTC),
        )
    )

    command = controller.mmd_command
    assert command is not None
    assert command.emotion_morph == "笑顔"
    assert command.emotion_intensity == pytest.approx(0.4549)
    assert command.gaze_bone == "両目"
    assert command.head_bone == "頭"
    assert command.upper_body_bone == "上半身"
    assert command.gesture == "none"
    assert command.gesture_target == ""
    assert command.gesture_motion_id == ""
    assert command.gesture_motion is None
    assert controller.model_preview is not None
    assert controller.model_preview.emotion_morphs == {"happy": ((0, 0.0, 0.5, 0.0),)}


def test_pmx_controller_resolves_expression_preset_to_multiple_model_morphs(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path, morph_name="笑い")
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            expression_presets={
                "gentle_smile": {"morphs": {"smile": 0.4, "blush": 0.1}},
            },
            emotion_presets={"happy": "gentle_smile"},
            mmd_mapping=MmdMappingConfig(
                expression_morphs={"smile": "笑い", "blush": "頬染め"},
            ),
        )
    )
    controller = build_avatar_controller(config, tmp_path)

    controller.apply_response(
        AssistantResponse(
            speech="こんにちは",
            emotion="happy",
            emotion_intensity=0.5,
        )
    )

    command = controller.mmd_command
    assert command is not None
    assert command.expression_preset == "gentle_smile"
    assert tuple(name for name, _weight in command.expression_morphs) == ("笑い", "頬染め")
    assert tuple(weight for _name, weight in command.expression_morphs) == pytest.approx(
        (0.18196, 0.04549)
    )
    assert command.emotion_morph == "笑い"


def test_pmx_controller_keeps_verified_display_morph_at_full_strength(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path, morph_name="照れ")
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            expression_presets={"effect": {"morphs": {"blush": 1.0}}},
            emotion_presets={"embarrassed": "effect"},
            mmd_mapping=MmdMappingConfig(expression_morphs={"blush": "照れ"}),
        )
    )
    controller = build_avatar_controller(config, tmp_path)

    controller.apply_response(
        AssistantResponse(
            speech="照れます",
            emotion="embarrassed",
            emotion_intensity=1.0,
        )
    )

    command = controller.mmd_command
    assert command is not None
    assert command.emotion_intensity < 1.0
    assert command.expression_morphs == (("照れ", 1.0),)


def test_pmx_controller_keeps_one_expression_variant_through_emotion_smoothing(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path, morph_name="笑い")
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            expression_presets={
                "happy_a": {"morphs": {"smile_a": 0.4}},
                "happy_b": {"morphs": {"smile_b": 0.6}},
            },
            emotion_presets={"happy": "happy_a"},
            emotion_preset_variants={"happy": ["happy_a", "happy_b"]},
            mmd_mapping=MmdMappingConfig(
                expression_morphs={"smile_a": "笑い", "smile_b": "笑い"},
            ),
        )
    )
    controller = build_avatar_controller(config, tmp_path)
    response = AssistantResponse(speech="うれしい", emotion="happy", emotion_intensity=1.0)
    state = EmotionState(
        conversation_id="test",
        emotion="happy",
        intensity=1.0,
        updated_at=datetime.now(UTC),
    )

    controller.apply_response(response)
    first_preset = controller.mmd_command.expression_preset
    preview = controller.expression_preview_command("happy")
    assert preview.expression_preset == first_preset
    controller.apply_emotion_state(state)
    assert controller.mmd_command.expression_preset == first_preset

    controller.apply_response(response)
    second_preset = controller.mmd_command.expression_preset
    controller.apply_emotion_state(state)
    assert second_preset != first_preset
    assert controller.mmd_command.expression_preset == second_preset


def test_pmx_controller_builds_non_persistent_expression_preview_command(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path, morph_name="笑い")
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            model_path="local/models/mmd/test-model.pmx",
            expression_presets={"gentle_smile": {"morphs": {"smile": 0.4}}},
            emotion_presets={"happy": "gentle_smile"},
            mmd_mapping=MmdMappingConfig(expression_morphs={"smile": "笑い"}),
        )
    )
    controller = build_avatar_controller(config, tmp_path)

    command = controller.expression_preview_command("happy")

    assert command.expression_preset == "gentle_smile"
    assert command.expression_morphs == (("笑い", 0.328),)
    assert command.gesture == "none"
    assert command.gaze == "forward"
    assert controller.mmd_command.expression_morphs == ()


def test_missing_gesture_and_user_gaze_mapping_fall_back_to_real_mmd_bones() -> None:
    bone_names = {"両目", "右腕", "右ひじ"}

    gaze = _resolve_mapped_bone(
        {"left": "両目"},
        "user",
        bone_names,
        mapping_fallbacks=("user", "forward", "left"),
        bone_fallbacks=("両目", "頭"),
    )
    wave = _resolve_mapped_bone(
        {"wave": "手を振る"},
        "wave",
        bone_names,
        bone_fallbacks=("右腕", "左腕"),
    )

    assert gaze == "両目"
    assert wave == "右腕"


def test_mmd_runtime_base_safely_uses_the_legacy_preview_until_the_runtime_exists(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local" / "models" / "mmd"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test-model.pmx"
    _write_minimum_pmx(model_path)
    config = AvatarConfig(
        three_d=Avatar3DConfig(
            enabled=True,
            provider="pmx_preview",
            renderer="mmd_runtime",
            model_path="local/models/mmd/test-model.pmx",
        )
    )

    controller = build_avatar_controller(config, tmp_path)

    assert controller.is_available is True
    assert controller.controller_name == "MMDランタイム基盤"
    assert "未実装" in controller.status_text
    assert controller.model_preview is not None
