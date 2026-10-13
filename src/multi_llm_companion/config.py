"""YAML設定と環境変数を安全に読み込む。"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from json import dumps as json_dumps
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(RuntimeError):
    """利用者に日本語で示せる設定エラー。"""


class AppConfig(BaseModel):
    """アプリ全体の設定。"""

    model_config = ConfigDict(extra="forbid")

    active_provider: str = "mock"
    providers_config_path: str = "local/providers.yaml"
    persona_path: str = "local/persona.yaml"
    database_path: str = "data/conversations.db"
    audio_config_path: str = "local/audio.yaml"
    memory_config_path: str = "local/memory.yaml"
    avatar_config_path: str = "local/avatar.yaml"


class SpeechRecognitionConfig(BaseModel):
    """マイク音声認識の設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "whisper"
    model_path: str = "local/models/vosk-model-small-ja-0.22"
    input_device: int | str | None = None
    sample_rate: int = Field(default=16000, gt=0)
    silence_threshold: float = Field(default=0.015, ge=0.001, le=1.0)
    silence_duration: float = Field(default=1.2, ge=0.3, le=10.0)
    max_recording_duration: float = Field(default=30.0, ge=3.0, le=120.0)
    partial_interval: float = Field(default=2.5, ge=1.0, le=15.0)
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"


class VoiceExpressionConfig(BaseModel):
    """VoiSona Talkへ渡す、ボイス名に依存しない音声表現設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_speed_change: float = Field(default=0.12, ge=0.0, le=0.25)
    max_intonation_change: float = Field(default=0.30, ge=0.0, le=0.50)


class VoiceOutputConfig(BaseModel):
    """VoiSona Talk音声出力の設定。秘密値は環境変数から読む。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "voisona_talk"
    base_url: str = "http://127.0.0.1:32766/api/talk/v1"
    username_env: str = "VOISONA_TALK_USERNAME"
    password_env: str = "VOISONA_TALK_PASSWORD"
    voice_name_env: str = "VOISONA_TALK_VOICE_NAME"
    language: str = "ja_JP"
    timeout: float = Field(default=15.0, gt=0.0)
    playback_preroll: float = Field(default=0.12, ge=0.0, le=1.0)
    lip_sync_start_delay: float = Field(default=0.0, ge=0.0, le=5.0)
    lip_sync_character_duration: float = Field(default=0.22, ge=0.04, le=1.0)
    lip_sync_timing_mode: str = "auto"
    expression: VoiceExpressionConfig = Field(default_factory=VoiceExpressionConfig)

    @field_validator("lip_sync_timing_mode")
    @classmethod
    def validate_lip_sync_timing_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "estimated"}:
            raise ValueError("口パク同期方式は auto または estimated を指定してください")
        return normalized


class AudioConfig(BaseModel):
    """第2段階の音声機能設定。"""

    model_config = ConfigDict(extra="forbid")

    speech_recognition: SpeechRecognitionConfig = Field(default_factory=SpeechRecognitionConfig)
    voice_output: VoiceOutputConfig = Field(default_factory=VoiceOutputConfig)


class MemoryConfig(BaseModel):
    """第3段階の長期記憶の設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    auto_approve_candidates: bool = True
    max_context_memories: int = Field(default=5, ge=1, le=10)
    min_relevance_score: int = Field(default=2, ge=1, le=20)
    duplicate_similarity_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
    max_memory_age_days: int = Field(default=365, ge=0, le=3650)
    max_recent_messages: int = Field(default=12, ge=2, le=50)
    summary_max_characters: int = Field(default=900, ge=100, le=4000)


class GesturePolicyConfig(BaseModel):
    """第34段階のジェスチャー間隔と連続抑制。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_interval_seconds: float = Field(default=3.5, ge=0.0, le=60.0)
    same_gesture_cooldown_seconds: float = Field(default=20.0, ge=0.0, le=3600.0)
    same_category_cooldown_seconds: float = Field(default=6.0, ge=0.0, le=3600.0)
    max_large_per_minute: int = Field(default=2, ge=0, le=60)
    recent_motion_count: int = Field(default=5, ge=0, le=50)


class ExpressionPresetConfig(BaseModel):
    """モデル名に依存しない、表情モーフの役割と基準強度。"""

    model_config = ConfigDict(extra="forbid")

    morphs: dict[str, float]

    @field_validator("morphs")
    @classmethod
    def normalize_morphs(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for role, weight in value.items():
            role_name = role.strip().lower()
            if not role_name:
                raise ValueError("表情プリセットのモーフ役割は空にできません")
            if not 0.0 < weight <= 1.0:
                raise ValueError("表情プリセットの強さは0より大きく1以下にしてください")
            normalized[role_name] = weight
        if not normalized:
            raise ValueError("表情プリセットにはモーフを1つ以上設定してください")
        return normalized


class ThinClothContactConfig(BaseModel):
    """特定モデルの薄布接触を局所的に補うための任意設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    source_body_patterns: list[str] = Field(default_factory=list)
    target_body_patterns: list[str] = Field(default_factory=list)
    maximum_correction_per_step: float = Field(default=0.008, ge=0.001, le=0.02)

    @field_validator("source_body_patterns", "target_body_patterns")
    @classmethod
    def normalize_body_patterns(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("薄布接触の剛体名候補は空にできません")
        if len(set(normalized)) != len(normalized):
            raise ValueError("薄布接触の剛体名候補が重複しています")
        return normalized

    @model_validator(mode="after")
    def require_patterns_when_enabled(self) -> ThinClothContactConfig:
        if self.enabled and (not self.source_body_patterns or not self.target_body_patterns):
            raise ValueError("薄布接触を有効にするには対象剛体名候補を指定してください")
        return self


class Avatar3DConfig(BaseModel):
    """3Dアバター接続と描画方式の設定。素材や秘密値は持たない。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "mock"
    model_path: str = ""
    vmd_catalog_path: str = "local/vmd_catalog.yaml"
    vmd_motion_id: str = ""
    motion_library_enabled: bool = False
    gesture_motion_ids: list[str] = Field(default_factory=list)
    gesture_policy: GesturePolicyConfig = Field(default_factory=GesturePolicyConfig)
    vmd_path: str = ""
    vmd_loop: bool = True
    renderer: str = "legacy_pmx_preview"
    fallback_renderer: str = "legacy_pmx_preview"
    render_quality: str = "standard"
    thin_cloth_contact: ThinClothContactConfig = Field(
        default_factory=lambda: ThinClothContactConfig()
    )
    expression_presets: dict[str, ExpressionPresetConfig] = Field(default_factory=dict)
    emotion_presets: dict[str, str] = Field(default_factory=dict)
    emotion_preset_variants: dict[str, list[str]] = Field(default_factory=dict)
    mmd_mapping: MmdMappingConfig = Field(default_factory=lambda: MmdMappingConfig())

    @field_validator("provider")
    @classmethod
    def strip_provider(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("3D連携方式を指定してください")
        return normalized

    @field_validator("model_path", "vmd_catalog_path", "vmd_motion_id", "vmd_path")
    @classmethod
    def strip_model_path(cls, value: str) -> str:
        return value.strip()

    @field_validator("gesture_motion_ids")
    @classmethod
    def normalize_gesture_motion_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("会話ジェスチャーVMD識別名は空にできません")
        if len(set(normalized)) != len(normalized):
            raise ValueError("会話ジェスチャーVMD識別名が重複しています")
        return normalized

    @field_validator("renderer", "fallback_renderer")
    @classmethod
    def validate_renderer(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"legacy_pmx_preview", "mmd_runtime"}:
            raise ValueError("描画方式は legacy_pmx_preview または mmd_runtime を指定してください")
        return normalized

    @field_validator("render_quality")
    @classmethod
    def validate_render_quality(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"low", "standard", "high"}:
            raise ValueError("描画品質は low、standard、high のいずれかを指定してください")
        return normalized

    @field_validator("expression_presets")
    @classmethod
    def normalize_expression_presets(
        cls,
        value: dict[str, ExpressionPresetConfig],
    ) -> dict[str, ExpressionPresetConfig]:
        normalized: dict[str, ExpressionPresetConfig] = {}
        for name, preset in value.items():
            preset_name = name.strip().lower()
            if not preset_name:
                raise ValueError("表情プリセット名は空にできません")
            normalized[preset_name] = preset
        return normalized

    @field_validator("emotion_presets")
    @classmethod
    def normalize_emotion_presets(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for emotion, preset in value.items():
            emotion_name = emotion.strip().lower()
            preset_name = preset.strip().lower()
            if not emotion_name or not preset_name:
                raise ValueError("感情と表情プリセットの名前は空にできません")
            normalized[emotion_name] = preset_name
        return normalized

    @field_validator("emotion_preset_variants")
    @classmethod
    def normalize_emotion_preset_variants(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for emotion, presets in value.items():
            emotion_name = emotion.strip().lower()
            names = [preset.strip().lower() for preset in presets if preset.strip()]
            if not emotion_name or not names:
                raise ValueError("感情ごとの表情候補は空にできません")
            normalized[emotion_name] = list(dict.fromkeys(names))
        return normalized

    @model_validator(mode="after")
    def require_catalog_for_selected_vmd(self) -> Avatar3DConfig:
        if self.vmd_path:
            raise ValueError(
                "VMDの直接指定は使用できません。vmd_catalog_pathとvmd_motion_idを指定してください"
            )
        if self.vmd_motion_id and not self.vmd_catalog_path:
            raise ValueError("VMD識別名を使う場合はvmd_catalog_pathを指定してください")
        if self.gesture_motion_ids and not self.vmd_catalog_path:
            raise ValueError("会話ジェスチャーVMDを使う場合はvmd_catalog_pathを指定してください")
        if self.motion_library_enabled and not self.vmd_catalog_path:
            raise ValueError("モーションライブラリを使う場合はvmd_catalog_pathを指定してください")
        if self.vmd_motion_id and self.vmd_motion_id in self.gesture_motion_ids:
            raise ValueError("同じVMDを待機動作と会話ジェスチャーへ同時指定できません")
        unknown_presets = set(self.emotion_presets.values()) - set(self.expression_presets)
        if unknown_presets:
            names = "、".join(sorted(unknown_presets))
            raise ValueError(f"未定義の表情プリセットが指定されています: {names}")
        unknown_variants = {
            preset for presets in self.emotion_preset_variants.values() for preset in presets
        } - set(self.expression_presets)
        if unknown_variants:
            names = "、".join(sorted(unknown_variants))
            raise ValueError(f"未定義の表情プリセット候補が指定されています: {names}")
        return self


class MmdMappingConfig(BaseModel):
    """モデルごとに異なるモーフ・ボーン名の対応表。"""

    model_config = ConfigDict(extra="forbid")

    emotion_morphs: dict[str, str] = Field(default_factory=dict)
    expression_morphs: dict[str, str] = Field(default_factory=dict)
    mouth_morphs: dict[str, str] = Field(default_factory=dict)
    gaze_bones: dict[str, str] = Field(default_factory=dict)
    pose_bones: dict[str, str] = Field(default_factory=dict)
    gesture_targets: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "emotion_morphs",
        "expression_morphs",
        "mouth_morphs",
        "gaze_bones",
        "pose_bones",
        "gesture_targets",
    )
    @classmethod
    def normalize_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for source, target in value.items():
            source_name = source.strip().lower()
            target_name = target.strip()
            if not source_name or not target_name:
                raise ValueError("MMD対応表の名前は空にできません")
            normalized[source_name] = target_name
        return normalized


class AvatarConfig(BaseModel):
    """第7段階のオリジナルアバター表示設定。"""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(default="紬", min_length=1, max_length=30)
    hair_color: str = "#684765"
    outfit_color: str = "#b87aa4"
    accent_color: str = "#d99ac1"
    three_d: Avatar3DConfig = Field(default_factory=Avatar3DConfig)

    @field_validator("hair_color", "outfit_color", "accent_color")
    @classmethod
    def require_hex_color(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", normalized):
            raise ValueError("色は #RRGGBB 形式で指定してください")
        return normalized


class ProviderConfig(BaseModel):
    """秘密情報を含まない、1つのプロバイダー設定。"""

    model_config = ConfigDict(extra="forbid")

    provider: str
    base_url_env: str
    api_key_env: str
    model_env: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    timeout: float = Field(default=60.0, gt=0.0)

    @field_validator("provider", "base_url_env", "api_key_env", "model_env")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ProvidersConfig(BaseModel):
    """利用できるプロバイダーの一覧。"""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]

    @field_validator("providers")
    @classmethod
    def require_provider(cls, value: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        if not value:
            raise ValueError("プロバイダーが1件もありません")
        return value


class CharacterConfig(BaseModel):
    """キャラクター固有の設定。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    first_person: str = "私"
    speaking_style: str = ""
    values: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    character_notes: list[str] = Field(default_factory=list)

    @field_validator("name", "first_person", "speaking_style")
    @classmethod
    def strip_character_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("values", "preferences", "character_notes")
    @classmethod
    def normalize_character_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class BehaviorPolicyConfig(BaseModel):
    """キャラクターに依存しない会話上の行動方針。"""

    model_config = ConfigDict(extra="forbid")

    response_length: str = ""
    question_frequency: str = ""
    initiative: str = ""
    disagreement_style: str = ""
    user_autonomy: str = ""
    intimacy_boundary: str = ""
    technical_guidance: str = ""
    memory_usage: str = ""

    @field_validator("*")
    @classmethod
    def strip_policy_text(cls, value: str) -> str:
        return value.strip()


class DialogueExampleConfig(BaseModel):
    """人格の話し方を示す短い会話例。"""

    model_config = ConfigDict(extra="forbid")

    user: str = ""
    assistant: str = ""

    @field_validator("user", "assistant")
    @classmethod
    def strip_example_text(cls, value: str) -> str:
        return value.strip()


class DialogueModePolicyConfig(BaseModel):
    """1つの会話モードで行うことと避けること。"""

    model_config = ConfigDict(extra="forbid")

    applies_when: str = ""
    do: str = ""
    avoid: str = ""

    @field_validator("applies_when", "do", "avoid")
    @classmethod
    def strip_mode_text(cls, value: str) -> str:
        return value.strip()


_DIALOGUE_MODE_NAMES = frozenset(
    {"casual", "listening", "supporting", "playful", "explaining", "planning", "boundary"}
)


class PersonaConfig(BaseModel):
    """キャラクターと会話方針を分けた人格設定。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    character: CharacterConfig | None = None
    behavior_policy: BehaviorPolicyConfig | None = None
    system_prompt_appendix: str = ""
    official_facts: list[str] = Field(default_factory=list)
    interpretations: list[str] = Field(default_factory=list)
    forbidden_inferences: list[str] = Field(default_factory=list)
    style_anchors: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    dialogue_examples: list[DialogueExampleConfig] = Field(default_factory=list)
    dialogue_modes: dict[str, DialogueModePolicyConfig] = Field(default_factory=dict)
    legacy_name: str | None = Field(default=None, alias="name")
    legacy_system_prompt: str | None = Field(default=None, alias="system_prompt")

    @field_validator("system_prompt_appendix", "legacy_name", "legacy_system_prompt")
    @classmethod
    def strip_persona_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator(
        "official_facts",
        "interpretations",
        "forbidden_inferences",
        "style_anchors",
        "anti_patterns",
    )
    @classmethod
    def normalize_persona_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("dialogue_modes")
    @classmethod
    def validate_dialogue_modes(
        cls,
        value: dict[str, DialogueModePolicyConfig],
    ) -> dict[str, DialogueModePolicyConfig]:
        normalized: dict[str, DialogueModePolicyConfig] = {}
        for name, policy in value.items():
            mode_name = name.strip().lower()
            if mode_name not in _DIALOGUE_MODE_NAMES:
                raise ValueError(f"未対応の会話モードです: {name}")
            if mode_name in normalized:
                raise ValueError(f"会話モードが重複しています: {mode_name}")
            normalized[mode_name] = policy
        return normalized

    @model_validator(mode="after")
    def validate_format(self) -> PersonaConfig:
        uses_legacy = self.legacy_name is not None or self.legacy_system_prompt is not None
        uses_structured = (
            self.character is not None
            or self.behavior_policy is not None
            or bool(self.system_prompt_appendix)
            or bool(self.official_facts)
            or bool(self.interpretations)
            or bool(self.forbidden_inferences)
            or bool(self.style_anchors)
            or bool(self.anti_patterns)
            or bool(self.dialogue_examples)
            or bool(self.dialogue_modes)
        )
        if uses_legacy and uses_structured:
            raise ValueError("人格設定の旧形式と新形式は同時に指定できません")
        if uses_legacy:
            if not self.legacy_name or not self.legacy_system_prompt:
                raise ValueError("旧形式ではnameとsystem_promptの両方が必要です")
            return self
        if self.character is None or self.behavior_policy is None:
            raise ValueError("新形式ではcharacterとbehavior_policyの両方が必要です")
        return self

    @property
    def is_legacy(self) -> bool:
        """旧形式から読み込んだ設定かを返す。"""

        return self.legacy_name is not None

    @property
    def name(self) -> str:
        """画面などで使う名前を形式に関係なく返す。"""

        if self.character is not None:
            return self.character.name
        return self.legacy_name or ""

    @property
    def system_prompt(self) -> str:
        """旧形式の自由記述を互換用に返す。"""

        return self.legacy_system_prompt or ""


@dataclass(frozen=True, slots=True)
class ResolvedProviderConfig:
    """環境変数を解決した実行時設定。APIキーは保存しない。"""

    name: str
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    temperature: float
    max_output_tokens: int
    timeout: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """起動に必要な設定をまとめたもの。"""

    project_root: Path
    app_config_path: Path
    app: AppConfig
    providers: ProvidersConfig
    persona: PersonaConfig
    active_provider: ResolvedProviderConfig
    database_path: Path
    audio: AudioConfig = field(default_factory=AudioConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"設定ファイルが見つかりません: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"設定ファイルを読み込めません: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAMLの書式が正しくありません: {path}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"設定ファイルの先頭は項目名で始めてください: {path}")
    return raw


def _load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate(_read_yaml(path))
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"設定内容が正しくありません: {path} ({details})") from exc


def _absolute_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _environment_value(variable_name: str) -> str:
    if not variable_name:
        return ""
    return os.getenv(variable_name, "").strip()


def resolve_provider(name: str, config: ProviderConfig) -> ResolvedProviderConfig:
    """環境変数名から接続値を取得する。"""

    implementation = config.provider.lower()
    base_url = _environment_value(config.base_url_env)
    api_key = _environment_value(config.api_key_env)
    model = _environment_value(config.model_env)

    if implementation == "mock":
        return ResolvedProviderConfig(
            name=name,
            provider=implementation,
            base_url=base_url,
            api_key="",
            model=model or "mock-model",
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout=config.timeout,
        )

    if implementation != "openai_compatible":
        raise ConfigError(f"プロバイダー「{name}」の種類「{config.provider}」には未対応です。")

    missing = [
        label
        for label, variable_name, value in (
            ("ベースURL", config.base_url_env, base_url),
            ("APIキー", config.api_key_env, api_key),
            ("モデル名", config.model_env, model),
        )
        if not variable_name or not value
    ]
    if missing:
        joined = "、".join(missing)
        raise ConfigError(f"プロバイダー「{name}」の{joined}が環境変数に設定されていません。")

    return ResolvedProviderConfig(
        name=name,
        provider=implementation,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout=config.timeout,
    )


def load_runtime_config(
    project_root: Path,
    app_config_path: Path | None = None,
) -> RuntimeConfig:
    """ファイルと環境変数から、起動用設定を作る。"""

    root = project_root.resolve()
    load_dotenv(root / ".env", override=False)
    app_path = (app_config_path or root / "local" / "app.yaml").resolve()
    app = _load_model(app_path, AppConfig)
    providers_path = _absolute_path(root, app.providers_config_path)
    persona_path = _absolute_path(root, app.persona_path)
    providers = _load_model(providers_path, ProvidersConfig)
    persona = _load_model(persona_path, PersonaConfig)
    audio_path = _absolute_path(root, app.audio_config_path)
    audio = _load_model(audio_path, AudioConfig) if audio_path.exists() else AudioConfig()
    memory_path = _absolute_path(root, app.memory_config_path)
    memory = _load_model(memory_path, MemoryConfig) if memory_path.exists() else MemoryConfig()
    avatar_path = _absolute_path(root, app.avatar_config_path)
    avatar = _load_model(avatar_path, AvatarConfig) if avatar_path.exists() else AvatarConfig()

    selected = providers.providers.get(app.active_provider)
    if selected is None:
        raise ConfigError(f"使用中プロバイダー「{app.active_provider}」が一覧にありません。")

    return RuntimeConfig(
        project_root=root,
        app_config_path=app_path,
        app=app,
        providers=providers,
        persona=persona,
        active_provider=resolve_provider(app.active_provider, selected),
        database_path=_absolute_path(root, app.database_path),
        audio=audio,
        memory=memory,
        avatar=avatar,
    )


def save_persona_config(path: Path, persona: PersonaConfig, *, expected: PersonaConfig) -> None:
    """既存形式と未編集部分を保ち、人格の変更項目だけを保存する。"""

    try:
        text = path.read_text(encoding="utf-8")
        current = PersonaConfig.model_validate(yaml.safe_load(text))
        persona = PersonaConfig.model_validate(persona.model_dump(by_alias=True))
        if current != expected:
            raise ConfigError(
                "人格設定が別の場所で変更されました。アプリを再起動してから編集してください。"
            )
        if current.is_legacy != persona.is_legacy:
            raise ConfigError("人格設定の形式は変更できません。")
        if not persona.name.strip():
            raise ConfigError("名前を入力してください。")
        previous = current.model_dump(by_alias=True, exclude_none=True)
        for key, value in persona.model_dump(by_alias=True, exclude_none=True).items():
            if previous.get(key) == value:
                continue
            root = yaml.compose(text)
            assert isinstance(root, yaml.MappingNode)
            entry = next((node for name, node in root.value if name.value == key), None)
            if entry is None:
                if root.flow_style:
                    at = root.end_mark.index - 1
                    text = (
                        text[:at]
                        + f", {json_dumps(key)}: {json_dumps(value, ensure_ascii=False)}"
                        + text[at:]
                    )
                else:
                    at = root.end_mark.index
                    prefix = "" if text[:at].endswith("\n") else "\n"
                    text = (
                        text[:at]
                        + prefix
                        + yaml.safe_dump({key: value}, allow_unicode=True, sort_keys=False)
                        + text[at:]
                    )
            else:
                replacement = json_dumps(value, ensure_ascii=False)
                # ブロック値の終端は、次のキーの字下げまでを含む。
                block = (
                    isinstance(entry, (yaml.MappingNode, yaml.SequenceNode))
                    and not entry.flow_style
                ) or (isinstance(entry, yaml.ScalarNode) and entry.style in {"|", ">"})
                if block:
                    replacement += "\n" + " " * entry.end_mark.column
                text = text[: entry.start_mark.index] + replacement + text[entry.end_mark.index :]
        if PersonaConfig.model_validate(yaml.safe_load(text)) != persona:
            raise ConfigError("人格設定を保存形式へ変換できませんでした。")
        if current == persona:
            return
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
            ) as output:
                temporary = Path(output.name)
                output.write(text)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(
            "人格設定を保存できませんでした。入力内容と保存先を確認してください。"
        ) from exc


def save_active_provider(app_config_path: Path, provider_name: str) -> None:
    """選択したプロバイダー名だけをアプリ設定へ保存する。"""

    raw = _read_yaml(app_config_path)
    raw["active_provider"] = provider_name
    try:
        serialized = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        app_config_path.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"設定を保存できません: {app_config_path}") from exc


def save_speech_recognition_settings(
    audio_config_path: Path,
    provider: str,
    input_device: int | str | None,
    silence_threshold: float | None = None,
    silence_duration: float | None = None,
    max_recording_duration: float | None = None,
    partial_interval: float | None = None,
) -> None:
    """認識方式とマイクだけを音声設定へ保存する。"""

    raw = _read_yaml(audio_config_path)
    speech = raw.setdefault("speech_recognition", {})
    if not isinstance(speech, dict):
        raise ConfigError(f"音声認識設定が正しくありません: {audio_config_path}")
    speech["provider"] = provider
    speech["input_device"] = input_device
    values = {
        "silence_threshold": silence_threshold,
        "silence_duration": silence_duration,
        "max_recording_duration": max_recording_duration,
        "partial_interval": partial_interval,
    }
    speech.update({key: value for key, value in values.items() if value is not None})
    try:
        serialized = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        audio_config_path.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"音声設定を保存できません: {audio_config_path}") from exc


def save_memory_enabled(memory_config_path: Path, enabled: bool) -> None:
    """長期記憶を会話へ渡すかだけをローカル設定へ保存する。"""

    raw = _read_yaml(memory_config_path) if memory_config_path.exists() else {}
    raw["enabled"] = enabled
    try:
        memory_config_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        memory_config_path.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"記憶設定を保存できません: {memory_config_path}") from exc


def save_memory_auto_approve_candidates(memory_config_path: Path, enabled: bool) -> None:
    """候補の自動承認設定だけをローカル設定へ保存する。"""

    raw = _read_yaml(memory_config_path) if memory_config_path.exists() else {}
    raw["auto_approve_candidates"] = enabled
    try:
        memory_config_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        memory_config_path.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"記憶設定を保存できません: {memory_config_path}") from exc


def save_avatar_expression_preset(
    avatar_config_path: Path,
    preset_name: str,
    emotion: str,
    morph_weights: dict[str, float],
) -> AvatarConfig:
    """カタログで選んだPMXモーフを既存の表情設定へ保存する。"""

    normalized_name = preset_name.strip().lower()
    normalized_emotion = emotion.strip().lower()
    if not normalized_name:
        raise ConfigError("保存名を入力してください。")
    if normalized_emotion and normalized_emotion not in {
        "neutral",
        "happy",
        "sad",
        "angry",
        "embarrassed",
        "surprised",
    }:
        raise ConfigError("会話に使う感情を選び直してください。")
    if not morph_weights:
        raise ConfigError("組合せへモーフを1つ以上追加してください。")
    raw = _read_yaml(avatar_config_path)
    three_d = raw.get("three_d")
    if not isinstance(three_d, dict):
        raise ConfigError(f"3D設定が正しくありません: {avatar_config_path}")
    presets = three_d.get("expression_presets")
    emotions = three_d.get("emotion_presets")
    variants = three_d.setdefault("emotion_preset_variants", {})
    mapping = three_d.get("mmd_mapping")
    if (
        not isinstance(presets, dict)
        or not isinstance(emotions, dict)
        or not isinstance(variants, dict)
        or not isinstance(mapping, dict)
    ):
        raise ConfigError(f"表情設定が正しくありません: {avatar_config_path}")
    expression_morphs = mapping.get("expression_morphs")
    if not isinstance(expression_morphs, dict):
        raise ConfigError(f"表情モーフ対応が正しくありません: {avatar_config_path}")
    previous_emotions = dict(emotions)
    previous_preset = presets.get(normalized_name)
    previous_roles = (
        set(previous_preset.get("morphs", {}))
        if isinstance(previous_preset, dict) and isinstance(previous_preset.get("morphs"), dict)
        else set()
    )
    normalized_weights = {
        name.strip(): weight for name, weight in morph_weights.items() if name.strip()
    }
    if len(normalized_weights) != len(morph_weights):
        raise ConfigError("モーフ名は空にできません。")
    roles = {
        f"catalog_{normalized_name}_{index}": name.strip()
        for index, name in enumerate(normalized_weights, start=1)
    }
    presets[normalized_name] = {
        "morphs": {role: normalized_weights[name] for role, name in roles.items()}
    }
    expression_morphs.update(roles)

    for emotion_name, previous in list(variants.items()):
        if not isinstance(previous, list):
            raise ConfigError("感情ごとの表情候補が正しくありません。")
        candidates = [
            name.strip().lower()
            for name in previous
            if isinstance(name, str) and name.strip() and name.strip().lower() != normalized_name
        ]
        if candidates:
            variants[emotion_name] = list(dict.fromkeys(candidates))
        else:
            variants.pop(emotion_name)
    for emotion_name, selected in list(emotions.items()):
        if not isinstance(selected, str):
            raise ConfigError("感情ごとの表情設定が正しくありません。")
        if selected.strip().lower() != normalized_name:
            continue
        candidates = variants.get(emotion_name, [])
        if candidates:
            emotions[emotion_name] = candidates[-1]
        else:
            emotions.pop(emotion_name)
    if normalized_emotion:
        existing_preset = emotions.get(normalized_emotion, "")
        emotions[normalized_emotion] = normalized_name
        previous = variants.get(normalized_emotion, [])
        if not isinstance(previous, list):
            raise ConfigError("感情ごとの表情候補が正しくありません。")
        candidates = [
            name.strip().lower()
            for name in previous
            if isinstance(name, str)
            and name.strip().lower()
            and name.strip().lower() != normalized_name
        ]
        if isinstance(existing_preset, str):
            existing_name = existing_preset.strip().lower()
            if (
                existing_name
                and existing_name != normalized_name
                and existing_name not in candidates
            ):
                candidates.append(existing_name)
        variants[normalized_emotion] = [*candidates, normalized_name]
    referenced_roles = {
        role
        for preset in presets.values()
        if isinstance(preset, dict) and isinstance(preset.get("morphs"), dict)
        for role in preset["morphs"]
    }
    generated_prefix = f"catalog_{normalized_name}_"
    removed_roles = {
        role
        for role in previous_roles
        if role.startswith(generated_prefix) and role not in referenced_roles
    }
    for role in removed_roles:
        expression_morphs.pop(role, None)
    try:
        avatar = AvatarConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"表情設定を保存できません: {details}") from exc
    text = avatar_config_path.read_text(encoding="utf-8")
    text = _upsert_avatar_mapping_entry(
        text,
        ("three_d", "expression_presets"),
        normalized_name,
        {"morphs": presets[normalized_name]["morphs"]},
    )
    for role in removed_roles:
        text = _delete_avatar_mapping_entry(
            text,
            ("three_d", "mmd_mapping", "expression_morphs"),
            role,
        )
    for role, morph_name in roles.items():
        text = _upsert_avatar_mapping_entry(
            text,
            ("three_d", "mmd_mapping", "expression_morphs"),
            role,
            morph_name,
        )
    for emotion_name in set(previous_emotions) | set(emotions):
        selected = emotions.get(emotion_name)
        if isinstance(selected, str):
            text = _upsert_avatar_mapping_entry(
                text,
                ("three_d", "emotion_presets"),
                emotion_name,
                selected,
            )
        else:
            text = _delete_avatar_mapping_entry(
                text,
                ("three_d", "emotion_presets"),
                emotion_name,
            )
    text = _upsert_avatar_mapping_entry(
        text,
        ("three_d",),
        "emotion_preset_variants",
        variants,
    )
    try:
        avatar_config_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"表情設定を保存できません: {avatar_config_path}") from exc
    return avatar


def delete_avatar_expression_preset(
    avatar_config_path: Path,
    preset_name: str,
) -> AvatarConfig:
    """表情プリセットと、その専用モーフ対応だけを削除する。"""

    normalized_name = preset_name.strip().lower()
    if not normalized_name:
        raise ConfigError("削除する表情プリセットを選んでください。")
    raw = _read_yaml(avatar_config_path)
    three_d = raw.get("three_d")
    if not isinstance(three_d, dict):
        raise ConfigError(f"3D設定が正しくありません: {avatar_config_path}")
    presets = three_d.get("expression_presets")
    emotions = three_d.get("emotion_presets")
    variants = three_d.setdefault("emotion_preset_variants", {})
    mapping = three_d.get("mmd_mapping")
    if (
        not isinstance(presets, dict)
        or not isinstance(emotions, dict)
        or not isinstance(variants, dict)
        or not isinstance(mapping, dict)
    ):
        raise ConfigError(f"表情設定が正しくありません: {avatar_config_path}")
    expression_morphs = mapping.get("expression_morphs")
    if not isinstance(expression_morphs, dict):
        raise ConfigError(f"表情モーフ対応が正しくありません: {avatar_config_path}")
    removed_preset = presets.pop(normalized_name, None)
    if not isinstance(removed_preset, dict):
        raise ConfigError(f"表情プリセット「{normalized_name}」が見つかりません。")
    previous_emotions = dict(emotions)
    removed_roles = set(removed_preset.get("morphs", {}))
    for emotion_name, previous in list(variants.items()):
        if not isinstance(previous, list):
            raise ConfigError("感情ごとの表情候補が正しくありません。")
        candidates = [
            name.strip().lower()
            for name in previous
            if isinstance(name, str) and name.strip() and name.strip().lower() != normalized_name
        ]
        if candidates:
            variants[emotion_name] = list(dict.fromkeys(candidates))
        else:
            variants.pop(emotion_name)
    for emotion_name, selected in list(emotions.items()):
        if not isinstance(selected, str):
            raise ConfigError("感情ごとの表情設定が正しくありません。")
        if selected.strip().lower() != normalized_name:
            continue
        candidates = variants.get(emotion_name, [])
        if candidates:
            emotions[emotion_name] = candidates[-1]
        else:
            emotions.pop(emotion_name)
    referenced_roles = {
        role
        for preset in presets.values()
        if isinstance(preset, dict) and isinstance(preset.get("morphs"), dict)
        for role in preset["morphs"]
    }
    generated_prefix = f"catalog_{normalized_name}_"
    orphaned_roles = {
        role
        for role in removed_roles
        if role.startswith(generated_prefix) and role not in referenced_roles
    }
    for role in orphaned_roles:
        expression_morphs.pop(role, None)
    try:
        avatar = AvatarConfig.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise ConfigError(f"表情設定を削除できません: {details}") from exc
    text = avatar_config_path.read_text(encoding="utf-8")
    text = _delete_avatar_mapping_entry(
        text,
        ("three_d", "expression_presets"),
        normalized_name,
    )
    for role in orphaned_roles:
        text = _delete_avatar_mapping_entry(
            text,
            ("three_d", "mmd_mapping", "expression_morphs"),
            role,
        )
    for emotion_name in set(previous_emotions) | set(emotions):
        selected = emotions.get(emotion_name)
        if isinstance(selected, str):
            text = _upsert_avatar_mapping_entry(
                text,
                ("three_d", "emotion_presets"),
                emotion_name,
                selected,
            )
        else:
            text = _delete_avatar_mapping_entry(
                text,
                ("three_d", "emotion_presets"),
                emotion_name,
            )
    text = _upsert_avatar_mapping_entry(
        text,
        ("three_d",),
        "emotion_preset_variants",
        variants,
    )
    try:
        avatar_config_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"表情設定を削除できません: {avatar_config_path}") from exc
    return avatar


def _upsert_avatar_mapping_entry(
    text: str,
    path: tuple[str, ...],
    key: str,
    value: str | dict[str, dict[str, float]] | dict[str, list[str]] | list[str],
) -> str:
    root = yaml.compose(text)
    mapping: yaml.MappingNode | None = root if isinstance(root, yaml.MappingNode) else None
    parent_key_column = 0
    container_entry: tuple[yaml.Node, yaml.Node] | None = None
    for path_key in path:
        if mapping is None:
            break
        entry_node = next(
            (
                (child_key, child_value)
                for child_key, child_value in mapping.value
                if child_key.value == path_key
            ),
            None,
        )
        if entry_node is None:
            mapping = None
            break
        parent_key, value_node = entry_node
        container_entry = (parent_key, value_node)
        parent_key_column = parent_key.start_mark.column
        mapping = value_node if isinstance(value_node, yaml.MappingNode) else None
    if mapping is None:
        joined = ".".join(path)
        raise ConfigError(f"表情設定の保存先が見つかりません: {joined}")
    if not mapping.value and mapping.flow_style and container_entry is not None:
        key_node, value_node = container_entry
        child_indent = " " * (key_node.start_mark.column + 2)
        child = _render_avatar_mapping_entry(child_indent, key, value)
        replacement = f"{' ' * key_node.start_mark.column}{json_dumps(path[-1])}:\n{child}"
        line_start = text.rfind("\n", 0, key_node.start_mark.index) + 1
        return text[:line_start] + replacement + text[value_node.end_mark.index :]
    entry = next(
        (
            (child_key, child_value)
            for child_key, child_value in mapping.value
            if child_key.value == key
        ),
        None,
    )
    indent = " " * (
        entry[0].start_mark.column
        if entry is not None
        else (mapping.value[0][0].start_mark.column if mapping.value else parent_key_column + 2)
    )
    rendered = _render_avatar_mapping_entry(indent, key, value)
    if entry is not None:
        key_node, value_node = entry
        line_start = text.rfind("\n", 0, key_node.start_mark.index) + 1
        tail_start = value_node.end_mark.index
        tail_line_start = text.rfind("\n", 0, tail_start) + 1
        if tail_line_start > line_start and not text[tail_line_start:tail_start].strip():
            tail_start = tail_line_start
        tail = text[tail_start:]
        separator = "" if not tail or tail.startswith(("\n", "\r")) else "\n"
        return text[:line_start] + rendered + separator + tail
    insert_at = mapping.end_mark.index
    line_start = text.rfind("\n", 0, insert_at) + 1
    if insert_at < len(text) and text[line_start:insert_at].isspace():
        insert_at = line_start
    prefix = "" if insert_at == 0 or text[insert_at - 1] == "\n" else "\n"
    suffix = "" if insert_at >= len(text) or text[insert_at] == "\n" else "\n"
    return text[:insert_at] + prefix + rendered + suffix + text[insert_at:]


def _delete_avatar_mapping_entry(
    text: str,
    path: tuple[str, ...],
    key: str,
) -> str:
    """YAML全体を再整形せず、指定したマッピング要素だけを外す。"""

    root = yaml.compose(text)
    mapping: yaml.MappingNode | None = root if isinstance(root, yaml.MappingNode) else None
    container_entry: tuple[yaml.Node, yaml.Node] | None = None
    for path_key in path:
        if mapping is None:
            return text
        entry_node = next(
            (
                (child_key, child_value)
                for child_key, child_value in mapping.value
                if child_key.value == path_key
            ),
            None,
        )
        if entry_node is None:
            return text
        container_entry = entry_node
        mapping = entry_node[1] if isinstance(entry_node[1], yaml.MappingNode) else None
    if mapping is None:
        return text
    entry_index = next(
        (
            index
            for index, (child_key, _value) in enumerate(mapping.value)
            if child_key.value == key
        ),
        None,
    )
    if entry_index is None:
        return text
    key_node, value_node = mapping.value[entry_index]
    line_start = text.rfind("\n", 0, key_node.start_mark.index) + 1
    if entry_index + 1 < len(mapping.value):
        next_key = mapping.value[entry_index + 1][0]
        block_end = text.rfind("\n", 0, next_key.start_mark.index) + 1
    else:
        end_index = value_node.end_mark.index
        end_line_start = text.rfind("\n", 0, end_index) + 1
        if end_line_start > line_start and not text[end_line_start:end_index].strip():
            block_end = end_line_start
        else:
            newline = text.find("\n", end_index)
            block_end = len(text) if newline < 0 else newline + 1
    if len(mapping.value) == 1 and container_entry is not None:
        container_key = container_entry[0]
        container_start = text.rfind("\n", 0, container_key.start_mark.index) + 1
        indent = " " * container_key.start_mark.column
        replacement = f"{indent}{json_dumps(path[-1], ensure_ascii=False)}: {{}}"
        suffix = "\n" if block_end < len(text) or text.endswith("\n") else ""
        return text[:container_start] + replacement + suffix + text[block_end:]
    return text[:line_start] + text[block_end:]


def _render_avatar_mapping_entry(
    indent: str,
    key: str,
    value: str | dict[str, dict[str, float]] | dict[str, list[str]] | list[str],
) -> str:
    quoted_key = json_dumps(key, ensure_ascii=False)
    if isinstance(value, str):
        return f"{indent}{quoted_key}: {json_dumps(value, ensure_ascii=False)}"
    if isinstance(value, list):
        lines = [f"{indent}{quoted_key}:"]
        lines.extend(f"{indent}  - {json_dumps(name, ensure_ascii=False)}" for name in value)
        return "\n".join(lines)
    if not value:
        return f"{indent}{quoted_key}: {{}}"
    if "morphs" in value:
        morphs = value["morphs"]
        lines = [f"{indent}{quoted_key}:", f"{indent}  morphs:"]
        lines.extend(
            f"{indent}    {json_dumps(role, ensure_ascii=False)}: {weight}"
            for role, weight in morphs.items()
        )
        return "\n".join(lines)
    lines = [f"{indent}{quoted_key}:"]
    for emotion, presets in value.items():
        lines.append(f"{indent}  {json_dumps(emotion, ensure_ascii=False)}:")
        lines.extend(f"{indent}    - {json_dumps(name, ensure_ascii=False)}" for name in presets)
    return "\n".join(lines)
