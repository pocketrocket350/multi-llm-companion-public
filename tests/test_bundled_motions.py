"""同梱モーションの初回導入・保護と実カタログの読み込みを確認する。"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from multi_llm_companion.config import AvatarConfig
from multi_llm_companion.integrations.factory import build_avatar_controller
from multi_llm_companion.integrations.noop import NullAvatarController
from multi_llm_companion.integrations.vmd_catalog import (
    VmdCatalog,
    VmdLicenseRecord,
    load_vmd_catalog_motions,
)
from multi_llm_companion.integrations.vmd_motion import read_vmd_motion
from scripts.install_bundled_motions import _install_bundled_motions


def _bundle(root: Path) -> dict[str, bytes]:
    files = {
        "motions/otokam/idle/original/test.vmd": b"artificial motion",
        "licenses/otokam/idle.yaml": b"artificial rights record",
        "licenses/otokam/idle/Readme.txt": b"artificial terms",
        "vmd_catalog.yaml": b"motions: {}\n",
    }
    entries = []
    for name, content in files.items():
        p = root / "bundled" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        entries.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    (root / "bundled/FILE_MANIFEST.json").write_text(
        json.dumps({"files": entries}), encoding="utf-8"
    )
    return files


def test_fresh_install_and_second_install_keep_files(tmp_path: Path) -> None:
    files = _bundle(tmp_path)
    _install_bundled_motions(tmp_path)
    for name, expected in files.items():
        assert (tmp_path / "local" / name).read_bytes() == expected
    mtimes = {name: (tmp_path / "local" / name).stat().st_mtime_ns for name in files}
    assert "既存のVMDカタログ" in _install_bundled_motions(tmp_path)
    assert {name: (tmp_path / "local" / name).stat().st_mtime_ns for name in files} == mtimes


def test_existing_catalog_skips_all_material_changes(tmp_path: Path) -> None:
    catalog = tmp_path / "local/vmd_catalog.yaml"
    catalog.parent.mkdir()
    catalog.write_bytes(b"user-owned catalog")
    _install_bundled_motions(tmp_path)
    assert catalog.read_bytes() == b"user-owned catalog"
    assert list(catalog.parent.iterdir()) == [catalog]


def test_bundle_corruption_stops_before_any_copy(tmp_path: Path) -> None:
    _bundle(tmp_path)
    (tmp_path / "bundled/vmd_catalog.yaml").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="ハッシュ"):
        _install_bundled_motions(tmp_path)
    assert not (tmp_path / "local").exists()


def test_catalog_directory_is_an_error_without_modifying_it(tmp_path: Path) -> None:
    catalog = tmp_path / "local/vmd_catalog.yaml"
    catalog.mkdir(parents=True)
    with pytest.raises(ValueError, match="ファイルではありません"):
        _install_bundled_motions(tmp_path)
    assert catalog.is_dir() and not list(catalog.iterdir())


def test_conflicting_user_material_is_not_overwritten(tmp_path: Path) -> None:
    _bundle(tmp_path)
    existing = tmp_path / "local/motions/otokam/idle/original/test.vmd"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"user-owned motion")
    with pytest.raises(ValueError, match="競合"):
        _install_bundled_motions(tmp_path)
    assert existing.read_bytes() == b"user-owned motion"
    assert not (tmp_path / "local/vmd_catalog.yaml").exists()
    assert not (tmp_path / "local/licenses").exists()


def test_partial_install_with_identical_files_can_resume(tmp_path: Path) -> None:
    files = _bundle(tmp_path)
    name = "motions/otokam/idle/original/test.vmd"
    existing = tmp_path / "local" / name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(files[name])
    before = existing.stat().st_mtime_ns
    _install_bundled_motions(tmp_path)
    assert existing.stat().st_mtime_ns == before
    assert (tmp_path / "local/vmd_catalog.yaml").read_bytes() == files["vmd_catalog.yaml"]


def test_manifest_cannot_write_outside_local_motion_locations(tmp_path: Path) -> None:
    _bundle(tmp_path)
    manifest = tmp_path / "bundled/FILE_MANIFEST.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["files"][0]["path"] = "motions/otokam/../../../.env"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="許可範囲外"):
        _install_bundled_motions(tmp_path)
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "local").exists()


def test_delivered_26_motions_and_credits_resolve_in_local(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "bundled", tmp_path / "bundled")
    _install_bundled_motions(tmp_path)
    catalog = VmdCatalog.model_validate(
        yaml.safe_load((tmp_path / "local/vmd_catalog.yaml").read_text(encoding="utf-8"))
    )
    manifest = json.loads((tmp_path / "bundled/FILE_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(catalog.motions) == 26
    assert sum(item["modified"] for item in manifest["motions"]) == 3
    for entry in catalog.motions.values():
        motion = read_vmd_motion(tmp_path / entry.file)
        assert motion.bone_frames
        record = VmdLicenseRecord.model_validate(
            yaml.safe_load((tmp_path / entry.license_record).read_text(encoding="utf-8"))
        )
        assert record.usage_confirmed and record.redistribution_allowed
        assert record.credit_required and record.source_url in record.credit_text
    for item in manifest["files"]:
        assert (
            hashlib.sha256((tmp_path / "local" / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    loaded = load_vmd_catalog_motions(tmp_path, "local/vmd_catalog.yaml")
    assert {motion.motion_id for motion in loaded} == {
        name for name, entry in catalog.motions.items() if entry.enabled
    }
    assert any(motion.motion_id == "sample_otokam_048" for motion in loaded)


def test_bundled_defaults_keep_mock_3d_disabled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = AvatarConfig.model_validate(
        yaml.safe_load((root / "config/avatar.example.yaml").read_text(encoding="utf-8"))
    )
    assert not config.three_d.enabled and config.three_d.provider == "mock"
    assert config.three_d.vmd_motion_id == "sample_otokam_048"
    assert config.three_d.motion_library_enabled
    assert isinstance(build_avatar_controller(config, tmp_path), NullAvatarController)
