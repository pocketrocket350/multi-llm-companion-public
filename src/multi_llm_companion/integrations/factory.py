"""設定から音声連携を組み立てる。"""

from __future__ import annotations

import os
from pathlib import Path

from multi_llm_companion.config import AudioConfig, AvatarConfig

from .base import AvatarControlPort, SpeechRecognitionPort, VoiceOutputPort
from .mmd_renderer import (
    LegacyPmxPreviewRenderer,
    MmdRuntimeRenderer,
    RendererBackedAvatarController,
)
from .mock_avatar import Mock3DAvatarController
from .noop import NullAvatarController, NullSpeechRecognizer, NullVoiceOutput
from .pmx_preview import PmxPreviewAvatarController, PmxPreviewError
from .vmd_catalog import (
    VmdCatalogError,
    load_vmd_catalog_motion,
    load_vmd_catalog_motions,
    load_vmd_motion_library,
)
from .vmd_motion import VmdMotionError
from .voisona_talk import VoiSonaTalkVoiceOutput
from .vosk_recognizer import VoskSpeechRecognizer
from .whisper_recognizer import WhisperSpeechRecognizer


def build_speech_recognizer(
    config: AudioConfig,
    project_root: Path,
) -> SpeechRecognitionPort:
    settings = config.speech_recognition
    if not settings.enabled:
        return NullSpeechRecognizer()
    implementation = settings.provider.lower()
    common = {
        "input_device": settings.input_device,
        "sample_rate": settings.sample_rate,
        "silence_threshold": settings.silence_threshold,
        "silence_duration": settings.silence_duration,
        "max_recording_duration": settings.max_recording_duration,
    }
    if implementation == "whisper":
        return WhisperSpeechRecognizer(
            model_name=settings.whisper_model,
            partial_interval=settings.partial_interval,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            model_root=(project_root / "local" / "models" / "faster-whisper-small").resolve(),
            **common,
        )
    if implementation != "vosk":
        return NullSpeechRecognizer(f"未対応の音声認識: {settings.provider}")
    model_path = Path(settings.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path
    return VoskSpeechRecognizer(
        model_path.resolve(),
        **common,
    )


def build_voice_output(config: AudioConfig) -> VoiceOutputPort:
    settings = config.voice_output
    if not settings.enabled:
        return NullVoiceOutput()
    if settings.provider.lower() != "voisona_talk":
        return NullVoiceOutput(f"未対応の音声出力: {settings.provider}")
    return VoiSonaTalkVoiceOutput(
        base_url=settings.base_url,
        username=os.getenv(settings.username_env, "").strip(),
        password=os.getenv(settings.password_env, "").strip(),
        voice_name=os.getenv(settings.voice_name_env, "").strip(),
        language=settings.language,
        timeout=settings.timeout,
        playback_preroll=settings.playback_preroll,
        lip_sync_start_delay=settings.lip_sync_start_delay,
        lip_sync_character_duration=settings.lip_sync_character_duration,
        lip_sync_timing_mode=settings.lip_sync_timing_mode,
        expression_enabled=settings.expression.enabled,
        max_speed_change=settings.expression.max_speed_change,
        max_intonation_change=settings.expression.max_intonation_change,
    )


def build_avatar_controller(
    config: AvatarConfig,
    project_root: Path,
    *,
    audio_available: bool = False,
) -> AvatarControlPort:
    """第10段階の3D連携を組み立てる。PMXはローカル配下だけを読む。"""

    settings = config.three_d
    if not settings.enabled:
        return NullAvatarController()
    if settings.provider.lower() == "mock":
        return Mock3DAvatarController()
    if settings.provider.lower() == "pmx_preview":
        if not settings.model_path:
            return NullAvatarController("PMXモデルが未選択です。")
        model_path = _resolve_mmd_model_path(project_root, settings.model_path)
        if model_path is None:
            return NullAvatarController("PMXモデルは local\\models\\mmd 内に置いてください。")
        try:
            requested_ids = tuple(
                dict.fromkeys((settings.vmd_motion_id, *settings.gesture_motion_ids))
            )
            requested_ids = tuple(motion_id for motion_id in requested_ids if motion_id)
            library_issues: tuple[str, ...] = ()
            if settings.motion_library_enabled:
                library_load = load_vmd_motion_library(
                    project_root,
                    settings.vmd_catalog_path,
                )
                loaded_vmds = library_load.motions
                library_issues = library_load.issues
            else:
                loaded_vmds = (
                    load_vmd_catalog_motions(
                        project_root,
                        settings.vmd_catalog_path,
                        requested_ids,
                    )
                    if requested_ids
                    else ()
                )
            loaded_by_id = {item.motion_id: item for item in loaded_vmds}
            selected_vmd = loaded_by_id.get(settings.vmd_motion_id)
            if settings.vmd_motion_id and selected_vmd is None:
                if settings.motion_library_enabled:
                    if not any(
                        issue.startswith(f"{settings.vmd_motion_id}:") for issue in library_issues
                    ):
                        try:
                            load_vmd_catalog_motion(
                                project_root,
                                settings.vmd_catalog_path,
                                settings.vmd_motion_id,
                            )
                        except VmdCatalogError as exc:
                            library_issues += (f"{settings.vmd_motion_id}: {exc}",)
                else:
                    selected_vmd = load_vmd_catalog_motion(
                        project_root,
                        settings.vmd_catalog_path,
                        settings.vmd_motion_id,
                    )
            if settings.motion_library_enabled:
                gesture_vmds = tuple(
                    item
                    for item in loaded_vmds
                    if item.motion_id != settings.vmd_motion_id
                    and item.motion.motion_type in {"talk", "reaction", "dance"}
                )
                gesture_ids = {item.motion_id for item in gesture_vmds}
                validation_vmds = tuple(
                    item
                    for item in loaded_vmds
                    if item.motion_id != settings.vmd_motion_id
                    and item.motion_id not in gesture_ids
                )
            else:
                gesture_vmds = tuple(
                    loaded_by_id[motion_id] for motion_id in settings.gesture_motion_ids
                )
                validation_vmds = ()
        except VmdCatalogError as exc:
            return NullAvatarController(str(exc))
        try:
            controller = PmxPreviewAvatarController(
                model_path,
                settings.mmd_mapping,
                expression_presets={
                    name: preset.morphs for name, preset in settings.expression_presets.items()
                },
                emotion_presets=settings.emotion_presets,
                emotion_preset_variants=settings.emotion_preset_variants,
                selected_vmd=selected_vmd,
                gesture_vmds=gesture_vmds,
                gesture_policy=settings.gesture_policy,
                skip_invalid_gesture_vmds=settings.motion_library_enabled,
                skip_invalid_selected_vmd=settings.motion_library_enabled,
                gesture_library_issues=library_issues,
                library_validation_vmds=validation_vmds,
                audio_available=audio_available,
                thin_cloth_contact_enabled=settings.thin_cloth_contact.enabled,
                thin_cloth_source_body_patterns=tuple(
                    settings.thin_cloth_contact.source_body_patterns
                ),
                thin_cloth_target_body_patterns=tuple(
                    settings.thin_cloth_contact.target_body_patterns
                ),
                thin_cloth_maximum_correction_per_step=(
                    settings.thin_cloth_contact.maximum_correction_per_step
                ),
            )
        except (PmxPreviewError, VmdCatalogError, VmdMotionError) as exc:
            return NullAvatarController(str(exc))
        return _with_renderer(controller, settings.renderer, settings.fallback_renderer)
    return NullAvatarController(f"未対応の3D連携: {settings.provider}")


def _with_renderer(
    controller: AvatarControlPort,
    renderer_name: str,
    fallback_name: str,
) -> AvatarControlPort:
    """本格MMD描画が未完成でも、既存の簡易表示へ安全に戻す。"""

    fallback = LegacyPmxPreviewRenderer()
    if fallback_name != "legacy_pmx_preview":
        return NullAvatarController(f"未対応の予備描画方式: {fallback_name}")
    if renderer_name == "legacy_pmx_preview":
        return RendererBackedAvatarController(controller, fallback)
    if renderer_name == "mmd_runtime":
        return RendererBackedAvatarController(controller, MmdRuntimeRenderer(fallback))
    return NullAvatarController(f"未対応のMMD描画方式: {renderer_name}")


def _resolve_mmd_model_path(project_root: Path, configured_path: str) -> Path | None:
    """公開対象外のローカルMMDフォルダに限定してモデルを読む。"""

    model_root = (project_root / "local" / "models" / "mmd").resolve()
    candidate = Path(configured_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(model_root)
    except ValueError:
        return None
    return resolved
