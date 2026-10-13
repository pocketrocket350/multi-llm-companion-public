from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from multi_llm_companion.config import (
    AudioConfig,
    Avatar3DConfig,
    AvatarConfig,
    ConfigError,
    PersonaConfig,
    ProviderConfig,
    delete_avatar_expression_preset,
    load_runtime_config,
    resolve_provider,
    save_active_provider,
    save_avatar_expression_preset,
    save_memory_auto_approve_candidates,
    save_speech_recognition_settings,
)
from multi_llm_companion.integrations.expression_morphs import ExpressionMorphResolver


def _write_settings(root: Path, active_provider: str = "mock") -> Path:
    local = root / "local"
    local.mkdir()
    app_path = local / "app.yaml"
    app_path.write_text(
        "\n".join(
            (
                f"active_provider: {active_provider}",
                "providers_config_path: local/providers.yaml",
                "persona_path: local/persona.yaml",
                "database_path: data/conversations.db",
            )
        ),
        encoding="utf-8",
    )
    (local / "providers.yaml").write_text(
        """providers:
  mock:
    provider: mock
    base_url_env: MOCK_BASE_URL
    api_key_env: MOCK_API_KEY
    model_env: MOCK_MODEL
    temperature: 0.7
    max_output_tokens: 128
    timeout: 5
  openai:
    provider: openai_compatible
    base_url_env: TEST_OPENAI_BASE_URL
    api_key_env: TEST_OPENAI_API_KEY
    model_env: TEST_OPENAI_MODEL
    temperature: 0.5
    max_output_tokens: 256
    timeout: 10
""",
        encoding="utf-8",
    )
    (local / "persona.yaml").write_text(
        "name: テスト案内役\nsystem_prompt: 丁寧に答えてください。\n",
        encoding="utf-8",
    )
    return app_path


def test_mock_starts_without_api_environment_variables(tmp_path: Path) -> None:
    _write_settings(tmp_path)

    runtime = load_runtime_config(tmp_path)

    assert runtime.active_provider.name == "mock"
    assert runtime.active_provider.model == "mock-model"
    assert runtime.active_provider.api_key == ""
    assert runtime.database_path == (tmp_path / "data" / "conversations.db")


def test_openai_compatible_reports_missing_environment_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_settings(tmp_path, active_provider="openai")
    for variable in (
        "TEST_OPENAI_BASE_URL",
        "TEST_OPENAI_API_KEY",
        "TEST_OPENAI_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(ConfigError, match="環境変数") as error:
        load_runtime_config(tmp_path)

    assert "sk-" not in str(error.value)


def test_active_provider_can_be_changed_without_other_settings(tmp_path: Path) -> None:
    app_path = _write_settings(tmp_path)

    save_active_provider(app_path, "openai")

    saved = app_path.read_text(encoding="utf-8")
    assert "active_provider: openai" in saved
    assert "database_path: data/conversations.db" in saved


def test_memory_auto_approval_can_be_saved_without_other_settings(tmp_path: Path) -> None:
    path = tmp_path / "memory.yaml"
    path.write_text("enabled: true\nmax_context_memories: 4\n", encoding="utf-8")

    save_memory_auto_approve_candidates(path, False)

    saved = path.read_text(encoding="utf-8")
    assert "enabled: true" in saved
    assert "max_context_memories: 4" in saved
    assert "auto_approve_candidates: false" in saved


def test_resolved_provider_repr_hides_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-that-must-not-appear"
    monkeypatch.setenv("SAFE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SAFE_API_KEY", secret)
    monkeypatch.setenv("SAFE_MODEL", "example-model")
    configured = ProviderConfig(
        provider="openai_compatible",
        base_url_env="SAFE_BASE_URL",
        api_key_env="SAFE_API_KEY",
        model_env="SAFE_MODEL",
        temperature=0.7,
        max_output_tokens=32,
        timeout=5,
    )

    resolved = resolve_provider("example", configured)

    assert secret not in repr(resolved)


def test_shipped_example_files_start_with_mock(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parents[1] / "config"
    local = tmp_path / "local"
    local.mkdir()
    shutil.copyfile(examples / "app.example.yaml", local / "app.yaml")
    shutil.copyfile(examples / "providers.example.yaml", local / "providers.yaml")
    shutil.copyfile(examples / "persona.example.yaml", local / "persona.yaml")

    runtime = load_runtime_config(tmp_path)

    assert runtime.active_provider.provider == "mock"
    assert runtime.persona.name == "ひなた"
    assert runtime.persona.is_legacy is False
    assert runtime.persona.character is not None
    assert runtime.persona.character.first_person == "私"
    assert runtime.persona.behavior_policy is not None
    assert "専門用語" in runtime.persona.behavior_policy.technical_guidance
    assert "listening" in runtime.persona.dialogue_modes


def test_local_legacy_persona_still_loads_without_changes(tmp_path: Path) -> None:
    app_path = _write_settings(tmp_path)

    runtime = load_runtime_config(tmp_path, app_config_path=app_path)

    assert runtime.persona == PersonaConfig(
        name="テスト案内役", system_prompt="丁寧に答えてください。"
    )


def test_shipped_avatar_example_has_valid_expression_presets() -> None:
    example_path = Path(__file__).resolve().parents[1] / "config" / "avatar.example.yaml"

    avatar = AvatarConfig.model_validate(yaml.safe_load(example_path.read_text(encoding="utf-8")))

    assert avatar.three_d.emotion_presets["happy"] == "gentle_smile"
    assert avatar.three_d.mmd_mapping.expression_morphs["smile"] == "笑顔"


def test_speech_engine_and_microphone_can_be_saved(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.yaml"
    audio_path.write_text(
        "speech_recognition:\n  enabled: true\n  provider: vosk\nvoice_output:\n  enabled: false\n",
        encoding="utf-8",
    )

    save_speech_recognition_settings(audio_path, "whisper", 3)

    saved = audio_path.read_text(encoding="utf-8")
    assert "provider: whisper" in saved
    assert "input_device: 3" in saved


def test_speech_turn_settings_can_be_saved_without_changing_format(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.yaml"
    audio_path.write_text("speech_recognition:\n  provider: whisper\n", encoding="utf-8")

    save_speech_recognition_settings(
        audio_path,
        "whisper",
        None,
        silence_threshold=0.02,
        silence_duration=1.5,
        max_recording_duration=45.0,
        partial_interval=3.0,
    )

    saved = yaml.safe_load(audio_path.read_text(encoding="utf-8"))
    assert saved["speech_recognition"] == {
        "provider": "whisper",
        "input_device": None,
        "silence_threshold": 0.02,
        "silence_duration": 1.5,
        "max_recording_duration": 45.0,
        "partial_interval": 3.0,
    }


def test_voice_expression_defaults_preserve_existing_audio_behavior() -> None:
    settings = AudioConfig.model_validate(
        {
            "voice_output": {
                "enabled": True,
                "provider": "voisona_talk",
            }
        }
    )

    assert settings.voice_output.expression.enabled is False
    assert settings.voice_output.expression.max_speed_change == 0.12
    assert settings.voice_output.expression.max_intonation_change == 0.30


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_speed_change", -0.01),
        ("max_speed_change", 0.26),
        ("max_intonation_change", -0.01),
        ("max_intonation_change", 0.51),
    ],
)
def test_voice_expression_rejects_values_outside_safe_setting_bounds(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        AudioConfig.model_validate(
            {
                "voice_output": {
                    "expression": {
                        "enabled": True,
                        field: value,
                    }
                }
            }
        )


def test_avatar_appearance_can_be_loaded_from_local_config(tmp_path: Path) -> None:
    _write_settings(tmp_path)
    (tmp_path / "local" / "avatar.yaml").write_text(
        "display_name: オリジナル案内役\n"
        "hair_color: '#112233'\n"
        "outfit_color: '#445566'\n"
        "accent_color: '#778899'\n"
        "three_d:\n"
        "  enabled: true\n"
        "  provider: pmx_preview\n"
        "  model_path: local/models/mmd/test.pmx\n"
        "  mmd_mapping:\n"
        "    emotion_morphs:\n"
        "      happy: 笑顔\n"
        "    gaze_bones:\n"
        "      left: 両目\n"
        "    gesture_targets:\n"
        "      wave: 手を振る\n",
        encoding="utf-8",
    )

    runtime = load_runtime_config(tmp_path)

    assert runtime.avatar.display_name == "オリジナル案内役"
    assert runtime.avatar.hair_color == "#112233"
    assert runtime.avatar.three_d.enabled is True
    assert runtime.avatar.three_d.provider == "pmx_preview"
    assert runtime.avatar.three_d.model_path == "local/models/mmd/test.pmx"
    assert runtime.avatar.three_d.mmd_mapping.emotion_morphs["happy"] == "笑顔"


def test_avatar_renderer_setting_accepts_the_mmd_runtime_base() -> None:
    settings = Avatar3DConfig(renderer="mmd_runtime", fallback_renderer="legacy_pmx_preview")

    assert settings.renderer == "mmd_runtime"
    assert settings.fallback_renderer == "legacy_pmx_preview"


def test_avatar_renderer_setting_rejects_unknown_renderer() -> None:
    with pytest.raises(ValueError, match="描画方式"):
        Avatar3DConfig(renderer="unknown_renderer")


def test_avatar_render_quality_accepts_three_safe_levels() -> None:
    assert Avatar3DConfig(render_quality="low").render_quality == "low"
    assert Avatar3DConfig(render_quality="standard").render_quality == "standard"
    assert Avatar3DConfig(render_quality="HIGH").render_quality == "high"


def test_avatar_render_quality_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="描画品質"):
        Avatar3DConfig(render_quality="ultra")


def test_avatar_thin_cloth_contact_requires_named_local_targets() -> None:
    settings = Avatar3DConfig(
        thin_cloth_contact={
            "enabled": True,
            "source_body_patterns": ["右腕", "右袖"],
            "target_body_patterns": ["Skirt_"],
            "maximum_correction_per_step": 0.006,
        }
    )

    assert settings.thin_cloth_contact.enabled is True
    assert settings.thin_cloth_contact.maximum_correction_per_step == 0.006
    with pytest.raises(ValueError, match="対象剛体名候補"):
        Avatar3DConfig(thin_cloth_contact={"enabled": True})


def test_avatar_expression_presets_are_separate_from_model_morph_names() -> None:
    settings = Avatar3DConfig(
        expression_presets={
            " Gentle_Smile ": {
                "morphs": {" Smile ": 0.35, "Eye_Smile": 0.18},
            }
        },
        emotion_presets={" Happy ": " Gentle_Smile "},
        mmd_mapping={
            "expression_morphs": {" Smile ": "笑い", "Eye_Smile": "にこり"},
        },
    )

    assert settings.emotion_presets == {"happy": "gentle_smile"}
    assert settings.expression_presets["gentle_smile"].morphs == {
        "smile": 0.35,
        "eye_smile": 0.18,
    }
    assert settings.mmd_mapping.expression_morphs == {
        "smile": "笑い",
        "eye_smile": "にこり",
    }


def test_avatar_expression_preset_rejects_unknown_name_and_invalid_weight() -> None:
    with pytest.raises(ValueError, match="未定義の表情プリセット"):
        Avatar3DConfig(emotion_presets={"happy": "missing"})

    with pytest.raises(ValueError, match="強さ"):
        Avatar3DConfig(
            expression_presets={"broken": {"morphs": {"smile": 1.1}}},
        )


def test_avatar_expression_catalog_preset_preserves_existing_settings_and_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "avatar.yaml"
    path.write_text(
        "display_name: テスト雨衣\n"
        "three_d:\n"
        "  # 元からある表情は残す\n"
        "  expression_presets:\n"
        "    neutral_soft:\n"
        "      morphs:\n"
        "        soft_smile: 0.06\n"
        "    happy_existing:\n"
        "      morphs:\n"
        "        soft_smile: 0.4\n"
        "  emotion_presets:\n"
        "    neutral: neutral_soft\n"
        "    happy: happy_existing\n"
        "  mmd_mapping:\n"
        "    expression_morphs:\n"
        "      soft_smile: 笑い\n",
        encoding="utf-8",
    )

    avatar = save_avatar_expression_preset(
        path,
        "happy_tear",
        "happy",
        {"涙": 0.7, "照れ": 0.2},
    )
    avatar = save_avatar_expression_preset(
        path,
        "happy_smile",
        "happy",
        {"笑い": 0.6},
    )

    saved = path.read_text(encoding="utf-8")
    assert "# 元からある表情は残す" in saved
    assert avatar.three_d.emotion_presets["happy"] == "happy_smile"
    assert avatar.three_d.emotion_preset_variants["happy"] == [
        "happy_existing",
        "happy_tear",
        "happy_smile",
    ]
    preset = avatar.three_d.expression_presets["happy_tear"]
    assert tuple(preset.morphs.values()) == (0.7, 0.2)
    mapped = avatar.three_d.mmd_mapping.expression_morphs
    assert tuple(mapped[role] for role in preset.morphs) == ("涙", "照れ")
    random_values = iter((0.0, 0.0, 0.9))
    resolver = ExpressionMorphResolver(
        expression_presets={
            name: item.morphs for name, item in avatar.three_d.expression_presets.items()
        },
        emotion_presets=avatar.three_d.emotion_presets,
        emotion_preset_variants=avatar.three_d.emotion_preset_variants,
        model_morphs=mapped,
        random_source=lambda: next(random_values),
    )
    assert resolver.resolve("happy", 1.0).morph_weights == (("笑い", 0.4),)
    assert resolver.resolve("happy", 1.0).morph_weights == (("涙", 0.7), ("照れ", 0.2))
    assert resolver.resolve("happy", 1.0).morph_weights == (("笑い", 0.6),)


def test_avatar_expression_catalog_preset_can_be_reassigned_updated_and_deleted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "avatar.yaml"
    path.write_text(
        "display_name: テスト雨衣\n"
        "three_d:\n"
        "  # このコメントは残す\n"
        "  expression_presets:\n"
        "    happy_base:\n"
        "      morphs:\n"
        "        smile: 0.7\n"
        "  emotion_presets:\n"
        "    happy: happy_base\n"
        "  mmd_mapping:\n"
        "    expression_morphs:\n"
        "      smile: 笑い\n",
        encoding="utf-8",
    )

    save_avatar_expression_preset(path, "custom", "happy", {"照れ": 0.4, "涙": 0.2})
    updated = save_avatar_expression_preset(path, "custom", "sad", {"困る": 0.8})

    assert updated.three_d.emotion_preset_variants["happy"] == ["happy_base"]
    assert updated.three_d.emotion_presets["happy"] == "happy_base"
    assert updated.three_d.emotion_preset_variants["sad"] == ["custom"]
    assert updated.three_d.emotion_presets["sad"] == "custom"
    assert "catalog_custom_2" not in updated.three_d.mmd_mapping.expression_morphs
    saved = path.read_text(encoding="utf-8")
    assert "# このコメントは残す" in saved
    assert "catalog_custom_2" not in saved

    deleted = delete_avatar_expression_preset(path, "custom")

    assert "custom" not in deleted.three_d.expression_presets
    assert "sad" not in deleted.three_d.emotion_presets
    assert "sad" not in deleted.three_d.emotion_preset_variants
    assert "catalog_custom_1" not in deleted.three_d.mmd_mapping.expression_morphs
    saved = path.read_text(encoding="utf-8")
    assert "# このコメントは残す" in saved
    assert yaml.safe_load(saved)["three_d"]["emotion_preset_variants"] == {"happy": ["happy_base"]}


def test_avatar_vmd_requires_catalog_selection() -> None:
    with pytest.raises(ValueError, match="VMDの直接指定"):
        Avatar3DConfig(vmd_path="local/models/mmd/old-motion.vmd")

    settings = Avatar3DConfig(
        vmd_catalog_path="local/vmd_catalog.yaml",
        vmd_motion_id="nod_small",
    )
    assert settings.vmd_motion_id == "nod_small"


def test_avatar_gesture_policy_and_motion_ids_are_validated() -> None:
    settings = Avatar3DConfig(
        motion_library_enabled=True,
        gesture_motion_ids=["wave_small", "nod_small"],
        gesture_policy={
            "min_interval_seconds": 2.0,
            "same_gesture_cooldown_seconds": 12.0,
            "same_category_cooldown_seconds": 4.0,
            "max_large_per_minute": 1,
            "recent_motion_count": 5,
        },
    )

    assert settings.motion_library_enabled is True
    assert settings.gesture_motion_ids == ["wave_small", "nod_small"]
    assert settings.gesture_policy.same_gesture_cooldown_seconds == 12.0
    assert settings.gesture_policy.recent_motion_count == 5
    with pytest.raises(ValueError, match="重複"):
        Avatar3DConfig(gesture_motion_ids=["nod_small", "nod_small"])
    with pytest.raises(ValueError, match="同じVMD"):
        Avatar3DConfig(vmd_motion_id="nod_small", gesture_motion_ids=["nod_small"])
