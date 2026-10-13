"""権利確認済みVMDだけをローカルカタログから選ぶ。"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml.nodes import MappingNode, ScalarNode

from .base import AvatarModelPreview, AvatarVmdCompatibility, AvatarVmdMotion

_MAX_YAML_FILE_SIZE = 1024 * 1024
_MOTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class VmdCatalogError(RuntimeError):
    """利用者へ安全に示せるVMDカタログエラー。"""


class VmdCompatibility(BaseModel):
    """選んだVMDを使用できるPMXの条件。"""

    model_config = ConfigDict(extra="forbid")

    model_names: list[str] = Field(default_factory=list)
    required_bones: list[str] = Field(default_factory=list)
    required_morphs: list[str] = Field(default_factory=list)

    @field_validator("model_names", "required_bones", "required_morphs")
    @classmethod
    def normalize_names(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("モデル適合条件へ空の名前は指定できません")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_a_compatibility_condition(self) -> VmdCompatibility:
        if not (self.model_names or self.required_bones or self.required_morphs):
            raise ValueError("モデル名、必須ボーン、必須モーフのいずれかを指定してください")
        return self


class VmdCatalogMotion(BaseModel):
    """VMD一件の用途と安全情報。"""

    model_config = ConfigDict(extra="forbid")

    file: str
    intent: str = ""
    motion_type: Literal["idle", "talk", "reaction", "dance"] = "talk"
    action: str = ""
    moods: list[str] = Field(default_factory=list)
    energy: Literal["low", "medium", "high"] = "medium"
    weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    audio_required: bool = False
    category: str
    body_mask: list[str] = Field(default_factory=list)
    duration_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)
    loop: bool = False
    intensity_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    cooldown_seconds: float = Field(default=0.0, ge=0.0, le=3600.0)
    license_record: str
    compatibility: VmdCompatibility
    enabled: bool = True

    @field_validator("file", "category", "license_record")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("VMDカタログの文字列は空にできません")
        return normalized

    @field_validator("intent", "action")
    @classmethod
    def normalize_optional_action(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("moods")
    @classmethod
    def normalize_moods(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value]
        if any(not item for item in normalized):
            raise ValueError("moodsへ空の名前は指定できません")
        return list(dict.fromkeys(normalized))

    @field_validator("body_mask")
    @classmethod
    def normalize_body_mask(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value]
        if any(not item for item in normalized):
            raise ValueError("body_maskへ空の名前は指定できません")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_action_or_legacy_intent(self) -> VmdCatalogMotion:
        if not (self.action or self.intent):
            raise ValueError("actionを指定してください")
        return self

    @property
    def selection_action(self) -> str:
        """新しいactionを優先し、既存カタログのintentも読み取る。"""

        return self.action or self.intent


class VmdCatalog(BaseModel):
    """ローカルVMDカタログ全体。"""

    model_config = ConfigDict(extra="forbid")

    motions: dict[str, VmdCatalogMotion]

    @field_validator("motions")
    @classmethod
    def validate_motion_ids(
        cls,
        value: dict[str, VmdCatalogMotion],
    ) -> dict[str, VmdCatalogMotion]:
        if not value:
            raise ValueError("VMDカタログにmotionsがありません")
        for motion_id in value:
            if not _MOTION_ID_PATTERN.fullmatch(motion_id):
                raise ValueError(
                    "VMD識別名は半角英数字で始め、英数字・点・ハイフン・下線だけを使ってください"
                )
        return value


class VmdLicenseRecord(BaseModel):
    """利用者が確認したVMDの出所と利用条件。"""

    model_config = ConfigDict(extra="forbid")

    title: str
    author: str
    source_url: str = ""
    terms_url: str = ""
    obtained_date: date
    allowed_use: str
    modifications: str = ""
    credit_required: bool
    credit_text: str = ""
    redistribution_allowed: bool
    usage_confirmed: bool = False
    notes: str = ""

    @field_validator("title", "author", "allowed_use")
    @classmethod
    def require_rights_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("権利確認記録のタイトル、作者、許可用途は空にできません")
        return normalized

    @field_validator(
        "source_url",
        "terms_url",
        "modifications",
        "credit_text",
        "notes",
    )
    @classmethod
    def strip_optional_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def require_credit_text_when_needed(self) -> VmdLicenseRecord:
        if self.credit_required and not self.credit_text:
            raise ValueError("クレジットが必要な場合はcredit_textを記録してください")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedVmdCatalogMotion:
    """検証済みのVMDカタログ選択。"""

    motion_id: str
    file_path: Path
    motion: VmdCatalogMotion
    license: VmdLicenseRecord


@dataclass(frozen=True, slots=True)
class VmdMotionLibraryLoad:
    """利用可能なVMDと、個別に除外した理由。"""

    motions: tuple[ResolvedVmdCatalogMotion, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InspectedVmdCatalogMotion:
    """管理画面へ渡す一件分の検査結果。"""

    motion_id: str
    motion: VmdCatalogMotion
    available: bool
    issues: tuple[str, ...]
    parsed_motion: AvatarVmdMotion | None = None


def load_vmd_catalog_motion(
    project_root: Path,
    catalog_path: str,
    motion_id: str,
) -> ResolvedVmdCatalogMotion:
    """カタログ、権利記録、ローカル境界を検証して一件を返す。"""

    normalized_id = motion_id.strip()
    if not normalized_id:
        raise VmdCatalogError("使用するVMD識別名が指定されていません。")
    return load_vmd_catalog_motions(
        project_root,
        catalog_path,
        (normalized_id,),
    )[0]


def load_vmd_catalog_motions(
    project_root: Path,
    catalog_path: str,
    motion_ids: Iterable[str] | None = None,
) -> tuple[ResolvedVmdCatalogMotion, ...]:
    """カタログを一度だけ読み、指定分または有効な全VMDを検証する。"""

    resolved_catalog = _resolve_local_file(
        project_root,
        catalog_path,
        allowed_suffixes={".yaml", ".yml"},
        label="VMDカタログ",
    )
    catalog = _validate_yaml_model(resolved_catalog, VmdCatalog, "VMDカタログ")
    requested_ids = _normalize_requested_ids(motion_ids)
    if requested_ids is None:
        selected_items = tuple(
            (motion_id, entry) for motion_id, entry in catalog.motions.items() if entry.enabled
        )
    else:
        selected_items = tuple(
            (motion_id, _require_enabled_motion(catalog, motion_id)) for motion_id in requested_ids
        )
    license_cache: dict[Path, VmdLicenseRecord] = {}
    return tuple(
        _resolve_catalog_motion(
            project_root,
            motion_id,
            entry,
            license_cache,
        )
        for motion_id, entry in selected_items
    )


def list_vmd_catalog_actions(
    project_root: Path,
    catalog_path: str,
    motion_ids: Iterable[str] | None = None,
    *,
    audio_available: bool = False,
) -> tuple[str, ...]:
    """LLMへ渡せる抽象actionだけをカタログから返す。"""

    resolved_catalog = _resolve_local_file(
        project_root,
        catalog_path,
        allowed_suffixes={".yaml", ".yml"},
        label="VMDカタログ",
    )
    catalog = _validate_yaml_model(resolved_catalog, VmdCatalog, "VMDカタログ")
    requested_ids = _normalize_requested_ids(motion_ids)
    if requested_ids is None:
        entries = tuple(entry for entry in catalog.motions.values() if entry.enabled)
    else:
        entries = tuple(_require_enabled_motion(catalog, motion_id) for motion_id in requested_ids)
    return tuple(
        dict.fromkeys(
            entry.selection_action
            for entry in entries
            if entry.motion_type != "idle" and (audio_available or not entry.audio_required)
        )
    )


def load_vmd_motion_library(
    project_root: Path,
    catalog_path: str,
) -> VmdMotionLibraryLoad:
    """有効な全件を検査し、個別ファイルの失敗だけ候補から除外する。"""

    resolved_catalog = _resolve_local_file(
        project_root,
        catalog_path,
        allowed_suffixes={".yaml", ".yml"},
        label="VMDカタログ",
    )
    catalog = _validate_yaml_model(resolved_catalog, VmdCatalog, "VMDカタログ")
    license_cache: dict[Path, VmdLicenseRecord] = {}
    motions: list[ResolvedVmdCatalogMotion] = []
    issues: list[str] = []
    for motion_id, entry in catalog.motions.items():
        if not entry.enabled:
            continue
        try:
            motions.append(
                _resolve_catalog_motion(
                    project_root,
                    motion_id,
                    entry,
                    license_cache,
                )
            )
        except VmdCatalogError as exc:
            issues.append(f"{motion_id}: {exc}")
    return VmdMotionLibraryLoad(tuple(motions), tuple(issues))


def inspect_vmd_motion_library(
    project_root: Path,
    catalog_path: str,
    preview: AvatarModelPreview | None,
    *,
    audio_available: bool,
) -> tuple[InspectedVmdCatalogMotion, ...]:
    """無効項目も含め、管理画面用に全VMDを検査する。"""

    from .vmd_motion import VmdMotionError, read_vmd_motion

    resolved_catalog = _resolve_local_file(
        project_root,
        catalog_path,
        allowed_suffixes={".yaml", ".yml"},
        label="VMDカタログ",
    )
    catalog = _validate_yaml_model(resolved_catalog, VmdCatalog, "VMDカタログ")
    license_cache: dict[Path, VmdLicenseRecord] = {}
    inspected: list[InspectedVmdCatalogMotion] = []
    for motion_id, entry in catalog.motions.items():
        issues: list[str] = []
        parsed_motion: AvatarVmdMotion | None = None
        try:
            resolved = _resolve_catalog_motion(
                project_root,
                motion_id,
                entry,
                license_cache,
            )
            parsed_motion = read_vmd_motion(resolved.file_path, loop=entry.loop)
            if preview is None:
                raise VmdCatalogError("PMXモデルが読み込まれていません。")
            validate_vmd_model_compatibility(resolved, preview, parsed_motion)
            if entry.motion_type != "idle":
                validate_conversation_vmd_tracks(motion_id, parsed_motion)
            parsed_motion = replace(
                parsed_motion,
                name=f"{resolved.file_path.stem}（管理プレビュー）",
                intensity_scale=entry.intensity_scale,
                compatibility=diagnose_vmd_model_compatibility(preview, parsed_motion),
            )
        except (VmdCatalogError, VmdMotionError) as exc:
            issues.append(str(exc))
        if entry.audio_required and not audio_available:
            issues.append("音源が利用できません。")
        inspected.append(
            InspectedVmdCatalogMotion(
                motion_id=motion_id,
                motion=entry,
                available=not issues,
                issues=tuple(issues),
                parsed_motion=parsed_motion,
            )
        )
    return tuple(inspected)


def save_vmd_catalog_motion_enabled(
    project_root: Path,
    catalog_path: str,
    motion_id: str,
    enabled: bool,
) -> None:
    """他の記述とコメントを保ち、enabledだけを保存する。"""

    resolved_catalog = _resolve_local_file(
        project_root,
        catalog_path,
        allowed_suffixes={".yaml", ".yml"},
        label="VMDカタログ",
    )
    catalog = _validate_yaml_model(resolved_catalog, VmdCatalog, "VMDカタログ")
    if motion_id not in catalog.motions:
        raise VmdCatalogError(f"VMDカタログに「{motion_id}」が登録されていません。")
    text = resolved_catalog.read_text(encoding="utf-8")
    root = yaml.compose(text)
    entry = _catalog_motion_node(root, motion_id)
    replacement = "true" if enabled else "false"
    enabled_value = _mapping_value(entry, "enabled")
    if enabled_value is not None:
        updated = (
            text[: enabled_value.start_mark.index]
            + replacement
            + text[enabled_value.end_mark.index :]
        )
    else:
        insert_at = entry.end_mark.index
        indent = " " * (entry.start_mark.column + 2)
        prefix = "" if insert_at == 0 or text[insert_at - 1] == "\n" else "\n"
        updated = text[:insert_at] + f"{prefix}{indent}enabled: {replacement}\n" + text[insert_at:]
    try:
        resolved_catalog.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise VmdCatalogError(f"VMDカタログを保存できません: {resolved_catalog}") from exc


def validate_conversation_vmd_tracks(motion_id: str, motion: AvatarVmdMotion) -> None:
    """会話用VMDにモデルへ適用できるトラックがあることを確認する。"""

    if motion.bone_frames or motion.morph_frames or motion.ik_frames:
        return
    raise VmdCatalogError(
        f"会話モーション「{motion_id}」に利用できるボーン、モーフ、IKがありません。"
    )


def _catalog_motion_node(root: object, motion_id: str) -> MappingNode:
    if not isinstance(root, MappingNode):
        raise VmdCatalogError("VMDカタログの先頭は項目名で始めてください。")
    motions = _mapping_value(root, "motions")
    if not isinstance(motions, MappingNode):
        raise VmdCatalogError("VMDカタログにmotionsがありません。")
    entry = _mapping_value(motions, motion_id)
    if not isinstance(entry, MappingNode):
        raise VmdCatalogError(f"VMDカタログに「{motion_id}」が登録されていません。")
    return entry


def _mapping_value(node: MappingNode, key: str) -> object | None:
    for key_node, value_node in node.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return value_node
    return None


def _normalize_requested_ids(motion_ids: Iterable[str] | None) -> tuple[str, ...] | None:
    if motion_ids is None:
        return None
    normalized = tuple(motion_id.strip() for motion_id in motion_ids)
    if any(not motion_id for motion_id in normalized):
        raise VmdCatalogError("使用するVMD識別名が指定されていません。")
    if len(set(normalized)) != len(normalized):
        raise VmdCatalogError("使用するVMD識別名が重複しています。")
    return normalized


def _require_enabled_motion(catalog: VmdCatalog, motion_id: str) -> VmdCatalogMotion:
    entry = catalog.motions.get(motion_id)
    if entry is None:
        raise VmdCatalogError(f"VMDカタログに「{motion_id}」が登録されていません。")
    if not entry.enabled:
        raise VmdCatalogError(f"VMD「{motion_id}」はカタログで無効になっています。")
    return entry


def _resolve_catalog_motion(
    project_root: Path,
    motion_id: str,
    entry: VmdCatalogMotion,
    license_cache: dict[Path, VmdLicenseRecord],
) -> ResolvedVmdCatalogMotion:
    motion_path = _resolve_local_file(
        project_root,
        entry.file,
        allowed_suffixes={".vmd"},
        label="VMDモーション",
    )
    license_path = _resolve_local_file(
        project_root,
        entry.license_record,
        allowed_suffixes={".yaml", ".yml"},
        label="VMD権利確認記録",
    )
    license_record = license_cache.get(license_path)
    if license_record is None:
        license_record = _validate_yaml_model(
            license_path,
            VmdLicenseRecord,
            "VMD権利確認記録",
        )
        license_cache[license_path] = license_record
    if not license_record.usage_confirmed:
        raise VmdCatalogError(
            f"VMD「{motion_id}」は利用条件の確認が完了していません。"
            "権利確認記録のusage_confirmedを確認してください。"
        )
    return ResolvedVmdCatalogMotion(motion_id, motion_path, entry, license_record)


def validate_vmd_model_compatibility(
    selected: ResolvedVmdCatalogMotion,
    preview: AvatarModelPreview,
    motion: AvatarVmdMotion,
) -> None:
    """カタログ条件と実VMDトラックを、読み込んだPMXへ照合する。"""

    compatibility = selected.motion.compatibility
    if compatibility.model_names and preview.model_name.casefold() not in {
        name.casefold() for name in compatibility.model_names
    }:
        raise VmdCatalogError(
            f"VMD「{selected.motion_id}」はモデル「{preview.model_name}」を対象にしていません。"
        )
    model_bones = {bone.name for bone in preview.bones}
    model_morphs = set(preview.morph_names) | set(preview.named_vertex_morphs)
    missing_bones = sorted(set(compatibility.required_bones) - model_bones)
    if missing_bones:
        raise VmdCatalogError(
            f"VMD「{selected.motion_id}」に必要なボーンがモデルにありません: "
            + "、".join(missing_bones)
        )
    missing_morphs = sorted(set(compatibility.required_morphs) - model_morphs)
    if missing_morphs:
        raise VmdCatalogError(
            f"VMD「{selected.motion_id}」に必要なモーフがモデルにありません: "
            + "、".join(missing_morphs)
        )
    motion_bones = {frame.bone_name for frame in motion.bone_frames}
    motion_morphs = {frame.morph_name for frame in motion.morph_frames}
    model_iks = {bone.name for bone in preview.bones if bone.ik_target_index >= 0 and bone.ik_links}
    motion_iks = {state.ik_name for frame in motion.ik_frames for state in frame.states}
    has_supported_camera = any(frame.perspective for frame in motion.camera_frames)
    if not (
        motion_bones & model_bones
        or motion_morphs & model_morphs
        or motion_iks & model_iks
        or has_supported_camera
        or motion.light_frames
    ):
        raise VmdCatalogError(
            f"VMD「{selected.motion_id}」には、利用できるボーン、モーフ、IK、"
            "透視カメラ、照明がありません。"
        )


def diagnose_vmd_model_compatibility(
    preview: AvatarModelPreview,
    motion: AvatarVmdMotion,
) -> AvatarVmdCompatibility:
    """実VMDの各トラックをPMXへ照合し、再生を止めない診断結果を返す。"""

    model_bones = {bone.name for bone in preview.bones}
    model_morphs = set(preview.morph_names)
    model_iks = {bone.name for bone in preview.bones if bone.ik_target_index >= 0 and bone.ik_links}
    motion_bones = {frame.bone_name for frame in motion.bone_frames}
    motion_morphs = {frame.morph_name for frame in motion.morph_frames}
    motion_iks = {state.ik_name for frame in motion.ik_frames for state in frame.states}
    height = 0.0
    if preview.points:
        ys = [point[1] for point in preview.points]
        height = max(ys) - min(ys)
    maximum_translation = max(
        (
            math.sqrt(sum(component * component for component in frame.translation))
            for frame in motion.bone_frames
        ),
        default=0.0,
    )
    unsupported: list[str] = []
    if motion.shadow_frames:
        unsupported.append("セルフ影")
    if motion.ik_frames:
        unsupported.append("モデル表示切替")
    if any(not frame.perspective for frame in motion.camera_frames):
        unsupported.append("平行投影カメラ")
    has_model_tracks = bool(motion_bones or motion_morphs or motion_iks)
    target_matches = (
        motion.target_model_name.casefold() == preview.model_name.casefold()
        if motion.target_model_name and has_model_tracks
        else None
    )
    return AvatarVmdCompatibility(
        supported_bones=tuple(sorted(motion_bones & model_bones)),
        missing_bones=tuple(sorted(motion_bones - model_bones)),
        supported_morphs=tuple(sorted(motion_morphs & model_morphs)),
        missing_morphs=tuple(sorted(motion_morphs - model_morphs)),
        supported_iks=tuple(sorted(motion_iks & model_iks)),
        missing_iks=tuple(sorted(motion_iks - model_iks)),
        translation_height_ratio=maximum_translation / height if height > 0.000001 else 0.0,
        target_model_matches=target_matches,
        unsupported_sections=tuple(unsupported),
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """同じキーを上書きせず設定ミスとして扱うSafeLoader。"""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in mapping
        except TypeError as exc:
            raise VmdCatalogError("YAMLの項目名は文字列で指定してください。") from exc
        if duplicated:
            raise VmdCatalogError(f"YAML内で「{key}」が重複しています。")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_yaml_model[T: BaseModel](
    path: Path,
    model_type: type[T],
    label: str,
) -> T:
    data = _read_yaml(path, label)
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first["loc"])
        detail = first["msg"]
        raise VmdCatalogError(f"{label}の形式が正しくありません: {location}: {detail}") from exc


def _read_yaml(path: Path, label: str) -> Any:
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_YAML_FILE_SIZE:
            raise VmdCatalogError(f"{label}のサイズが正しくありません（上限は1MBです）。")
        with path.open("r", encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except VmdCatalogError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise VmdCatalogError(f"{label}を読み取れません。") from exc


def _resolve_local_file(
    project_root: Path,
    configured_path: str,
    *,
    allowed_suffixes: set[str],
    label: str,
) -> Path:
    local_root = (project_root / "local").resolve()
    normalized_path = configured_path.strip()
    if not normalized_path:
        raise VmdCatalogError(f"{label}のファイルが指定されていません。")
    candidate = Path(normalized_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(local_root)
    except ValueError as exc:
        raise VmdCatalogError(f"{label}はlocalフォルダー内へ置いてください。") from exc
    if resolved.suffix.lower() not in allowed_suffixes:
        suffixes = "、".join(sorted(allowed_suffixes))
        raise VmdCatalogError(f"{label}は{suffixes}形式のファイルを指定してください。")
    if not resolved.is_file():
        raise VmdCatalogError(f"{label}が見つかりません: {configured_path}")
    return resolved
