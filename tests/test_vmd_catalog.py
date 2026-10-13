from __future__ import annotations

from pathlib import Path

import pytest

from multi_llm_companion.integrations.base import (
    AvatarBone,
    AvatarModelPreview,
    AvatarVmdBoneFrame,
    AvatarVmdCameraFrame,
    AvatarVmdIkFrame,
    AvatarVmdIkState,
    AvatarVmdMorphFrame,
    AvatarVmdMotion,
    AvatarVmdShadowFrame,
)
from multi_llm_companion.integrations.vmd_catalog import (
    VmdCatalogError,
    diagnose_vmd_model_compatibility,
    inspect_vmd_motion_library,
    list_vmd_catalog_actions,
    load_vmd_catalog_motion,
    load_vmd_catalog_motions,
    load_vmd_motion_library,
    save_vmd_catalog_motion_enabled,
    validate_vmd_model_compatibility,
)
from multi_llm_companion.integrations.vmd_motion import write_confirmation_vmd


def test_compatibility_diagnostic_reports_supported_missing_and_body_ratio() -> None:
    preview = AvatarModelPreview(
        model_name="現在モデル",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        morph_names=("笑顔",),
        bones=(
            AvatarBone("センター", -1, (0.0, 5.0, 0.0)),
            AvatarBone("左足ＩＫ", -1, (0.0, 0.0, 0.0), 0, 1, 0.3, (0,)),
        ),
    )
    motion = AvatarVmdMotion(
        "診断",
        10,
        (
            AvatarVmdBoneFrame("センター", 0, (0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            AvatarVmdBoneFrame("追加ボーン", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ),
        morph_frames=(
            AvatarVmdMorphFrame("笑顔", 0, 1.0),
            AvatarVmdMorphFrame("未登録表情", 0, 1.0),
        ),
        target_model_name="別モデル",
        shadow_frames=(AvatarVmdShadowFrame(0, 1, 10.0),),
        ik_frames=(
            AvatarVmdIkFrame(
                0,
                True,
                (
                    AvatarVmdIkState("左足ＩＫ", True),
                    AvatarVmdIkState("補助ＩＫ", False),
                ),
            ),
        ),
    )

    report = diagnose_vmd_model_compatibility(preview, motion)

    assert report.supported_bones == ("センター",)
    assert report.missing_bones == ("追加ボーン",)
    assert report.supported_morphs == ("笑顔",)
    assert report.missing_morphs == ("未登録表情",)
    assert report.supported_iks == ("左足ＩＫ",)
    assert report.missing_iks == ("補助ＩＫ",)
    assert report.translation_height_ratio == pytest.approx(0.2)
    assert report.target_model_matches is False
    assert report.unsupported_sections == ("セルフ影", "モデル表示切替")


def _write_catalog(
    root: Path,
    *,
    usage_confirmed: bool = True,
    motion_file: str = "local/motions/nod.vmd",
    required_bone: str = "頭",
) -> None:
    local = root / "local"
    (local / "motions").mkdir(parents=True)
    (local / "licenses").mkdir(parents=True)
    if motion_file.startswith("local/"):
        target = root / motion_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test-vmd")
    (local / "licenses" / "nod.yaml").write_text(
        "title: 自作うなずき\n"
        "author: テスト作者\n"
        "source_url: ''\n"
        "terms_url: ''\n"
        "obtained_date: 2026-07-24\n"
        "allowed_use: 個人利用とアプリ内再生\n"
        "modifications: なし\n"
        "credit_required: false\n"
        "credit_text: ''\n"
        "redistribution_allowed: false\n"
        f"usage_confirmed: {'true' if usage_confirmed else 'false'}\n"
        "notes: テスト用記録\n",
        encoding="utf-8",
    )
    (local / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  nod_small:\n"
        f"    file: {motion_file}\n"
        "    intent: small_nod\n"
        "    category: head\n"
        "    body_mask: [head, neck]\n"
        "    duration_seconds: 1.0\n"
        "    loop: false\n"
        "    intensity_scale: 0.6\n"
        "    cooldown_seconds: 8\n"
        "    license_record: local/licenses/nod.yaml\n"
        "    compatibility:\n"
        "      model_names: [テストモデル]\n"
        f"      required_bones: [{required_bone}]\n"
        "      required_morphs: []\n"
        "    enabled: true\n",
        encoding="utf-8",
    )


def test_catalog_resolves_registered_rights_confirmed_motion(tmp_path: Path) -> None:
    _write_catalog(tmp_path)

    selected = load_vmd_catalog_motion(
        tmp_path,
        "local/vmd_catalog.yaml",
        "nod_small",
    )

    assert selected.motion_id == "nod_small"
    assert selected.file_path == (tmp_path / "local" / "motions" / "nod.vmd").resolve()
    assert selected.motion.intent == "small_nod"
    assert selected.license.usage_confirmed is True


def test_catalog_rejects_unconfirmed_rights_record(tmp_path: Path) -> None:
    _write_catalog(tmp_path, usage_confirmed=False)

    with pytest.raises(VmdCatalogError, match="利用条件の確認が完了していません"):
        load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")


def test_catalog_rejects_motion_outside_local_folder(tmp_path: Path) -> None:
    outside = tmp_path / "outside.vmd"
    outside.write_bytes(b"test-vmd")
    _write_catalog(tmp_path, motion_file=str(outside))

    with pytest.raises(VmdCatalogError, match="localフォルダー内"):
        load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")


def test_catalog_rejects_unknown_motion_id(tmp_path: Path) -> None:
    _write_catalog(tmp_path)

    with pytest.raises(VmdCatalogError, match="登録されていません"):
        load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "wave_unknown")


def test_catalog_rejects_duplicate_motion_id(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    catalog_path = tmp_path / "local" / "vmd_catalog.yaml"
    original = catalog_path.read_text(encoding="utf-8")
    duplicated_entry = original.split("  nod_small:\n", 1)[1]
    catalog_path.write_text(
        original + "  nod_small:\n" + duplicated_entry,
        encoding="utf-8",
    )

    with pytest.raises(VmdCatalogError, match="重複"):
        load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")


def test_model_compatibility_rejects_missing_required_bone(tmp_path: Path) -> None:
    _write_catalog(tmp_path, required_bone="右腕")
    selected = load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")
    preview = AvatarModelPreview(
        model_name="テストモデル",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(AvatarBone("頭", -1, (0.0, 1.0, 0.0)),),
    )
    motion = AvatarVmdMotion(
        "nod",
        30,
        (
            AvatarVmdBoneFrame(
                "頭",
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )

    with pytest.raises(VmdCatalogError, match="右腕"):
        validate_vmd_model_compatibility(selected, preview, motion)


def test_model_compatibility_accepts_matching_model_and_track(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    selected = load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")
    preview = AvatarModelPreview(
        model_name="テストモデル",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(AvatarBone("頭", -1, (0.0, 1.0, 0.0)),),
    )
    motion = AvatarVmdMotion(
        "nod",
        30,
        (
            AvatarVmdBoneFrame(
                "頭",
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )

    validate_vmd_model_compatibility(selected, preview, motion)


def test_model_compatibility_accepts_perspective_camera_only_motion(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    selected = load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")
    preview = AvatarModelPreview(
        model_name="テストモデル",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(AvatarBone("頭", -1, (0.0, 1.0, 0.0)),),
    )
    motion = AvatarVmdMotion(
        "camera",
        30,
        (),
        target_model_name="カメラ・照明",
        camera_frames=(
            AvatarVmdCameraFrame(
                0,
                35.0,
                (0.0, 10.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        ),
    )

    validate_vmd_model_compatibility(selected, preview, motion)

    report = diagnose_vmd_model_compatibility(preview, motion)
    assert report.target_model_matches is None


def test_model_compatibility_rejects_vmd_without_usable_track(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    selected = load_vmd_catalog_motion(tmp_path, "local/vmd_catalog.yaml", "nod_small")
    preview = AvatarModelPreview(
        model_name="テストモデル",
        vertex_count=0,
        face_count=0,
        points=(),
        bones=(AvatarBone("頭", -1, (0.0, 1.0, 0.0)),),
    )
    motion = AvatarVmdMotion(
        "other-model-motion",
        30,
        (
            AvatarVmdBoneFrame(
                "未知ボーン",
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )

    with pytest.raises(VmdCatalogError, match="利用できるボーン、モーフ、IK"):
        validate_vmd_model_compatibility(selected, preview, motion)


def test_motion_library_loads_all_enabled_entries_with_phase6_metadata(tmp_path: Path) -> None:
    local = tmp_path / "local"
    motions = local / "motions"
    licenses = local / "licenses"
    motions.mkdir(parents=True)
    licenses.mkdir()
    (licenses / "shared.yaml").write_text(
        "title: 自作テストモーション\n"
        "author: テスト作者\n"
        "obtained_date: 2026-08-27\n"
        "allowed_use: 自動テスト\n"
        "credit_required: false\n"
        "redistribution_allowed: false\n"
        "usage_confirmed: true\n",
        encoding="utf-8",
    )
    entries: list[str] = ["motions:"]
    for index in range(20):
        (motions / f"motion_{index}.vmd").write_bytes(b"test")
        entries.extend(
            (
                f"  motion_{index}:",
                f"    file: local/motions/motion_{index}.vmd",
                "    motion_type: reaction",
                "    action: react",
                "    moods: [happy, neutral]",
                "    energy: medium",
                f"    weight: {(index % 3) + 1}",
                "    audio_required: false",
                "    category: upper_body",
                "    license_record: local/licenses/shared.yaml",
                "    compatibility:",
                "      required_bones: [上半身]",
            )
        )
    (local / "vmd_catalog.yaml").write_text("\n".join(entries) + "\n", encoding="utf-8")

    loaded = load_vmd_catalog_motions(tmp_path, "local/vmd_catalog.yaml")
    actions = list_vmd_catalog_actions(tmp_path, "local/vmd_catalog.yaml")

    assert len(loaded) == 20
    assert all(item.motion.motion_type == "reaction" for item in loaded)
    assert all(item.motion.selection_action == "react" for item in loaded)
    assert actions == ("react",)


def test_motion_library_rejects_unknown_type_and_energy(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  broken:\n"
        "    file: local/motions/broken.vmd\n"
        "    motion_type: unknown\n"
        "    action: react\n"
        "    energy: extreme\n"
        "    category: upper_body\n"
        "    license_record: local/licenses/broken.yaml\n"
        "    compatibility:\n"
        "      required_bones: [上半身]\n",
        encoding="utf-8",
    )

    with pytest.raises(VmdCatalogError, match="motion_type"):
        load_vmd_catalog_motions(tmp_path, "local/vmd_catalog.yaml")


def test_motion_library_excludes_a_missing_file_and_reports_the_reason(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  missing_motion:\n"
        "    file: local/motions/missing.vmd\n"
        "    motion_type: reaction\n"
        "    action: react\n"
        "    category: upper_body\n"
        "    license_record: local/licenses/missing.yaml\n"
        "    compatibility:\n"
        "      required_bones: [上半身]\n",
        encoding="utf-8",
    )

    result = load_vmd_motion_library(tmp_path, "local/vmd_catalog.yaml")

    assert result.motions == ()
    assert len(result.issues) == 1
    assert "missing_motion" in result.issues[0]
    assert "見つかりません" in result.issues[0]


def test_motion_library_inspection_lists_disabled_and_audio_unavailable_entries(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)
    catalog_path = tmp_path / "local" / "vmd_catalog.yaml"
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8")
        .replace("    intent: small_nod", "    intent: small_nod\n    motion_type: dance")
        .replace("    enabled: true", "    audio_required: true\n    enabled: false"),
        encoding="utf-8",
    )
    write_confirmation_vmd(tmp_path / "local" / "motions" / "nod.vmd", ("頭",))
    preview = AvatarModelPreview(
        model_name="テストモデル",
        vertex_count=2,
        face_count=0,
        points=((0.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
        bones=(AvatarBone("頭", -1, (0.0, 8.0, 0.0)),),
    )

    entries = inspect_vmd_motion_library(
        tmp_path,
        "local/vmd_catalog.yaml",
        preview,
        audio_available=False,
    )

    assert len(entries) == 1
    assert entries[0].motion_id == "nod_small"
    assert entries[0].motion.motion_type == "dance"
    assert entries[0].motion.enabled is False
    assert entries[0].available is False
    assert entries[0].parsed_motion is not None
    assert entries[0].issues == ("音源が利用できません。",)


def test_audio_required_actions_are_exposed_only_when_voice_output_is_available(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)
    path = tmp_path / "local" / "vmd_catalog.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    enabled: true",
            "    audio_required: true\n    enabled: true",
        ),
        encoding="utf-8",
    )

    assert (
        list_vmd_catalog_actions(
            tmp_path,
            "local/vmd_catalog.yaml",
            audio_available=False,
        )
        == ()
    )
    assert list_vmd_catalog_actions(
        tmp_path,
        "local/vmd_catalog.yaml",
        audio_available=True,
    ) == ("small_nod",)


def test_motion_enabled_save_preserves_comments_and_other_values(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    catalog_path = tmp_path / "local" / "vmd_catalog.yaml"
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace(
            "    enabled: true",
            "    # 利用者が残した説明\n    enabled: true",
        ),
        encoding="utf-8",
    )

    save_vmd_catalog_motion_enabled(
        tmp_path,
        "local/vmd_catalog.yaml",
        "nod_small",
        False,
    )

    saved = catalog_path.read_text(encoding="utf-8")
    assert "# 利用者が残した説明" in saved
    assert "intensity_scale: 0.6" in saved
    assert "enabled: false" in saved
    assert "enabled: true" not in saved
