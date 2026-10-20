"""設定、会話処理、画面をつなぐアプリケーション層。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event

from multi_llm_companion.config import (
    AvatarConfig,
    ConfigError,
    PersonaConfig,
    RuntimeConfig,
    delete_avatar_expression_preset,
    resolve_provider,
    save_active_provider,
    save_avatar_expression_preset,
    save_memory_auto_approve_candidates,
    save_memory_enabled,
    save_persona_config,
    save_speech_recognition_settings,
)
from multi_llm_companion.domain import (
    AssistantResponse,
    ConversationState,
    EmotionState,
    ExpressionChannel,
    ExpressionPolicy,
    MemoryCategory,
    MemoryEntry,
    MemoryStatus,
    MessageRole,
    TokenUsageSummary,
)
from multi_llm_companion.domain.models import _recover_markdown_speech
from multi_llm_companion.integrations import (
    build_avatar_controller,
    build_speech_recognizer,
    build_voice_output,
)
from multi_llm_companion.integrations.base import AvatarVmdMotion, MmdAvatarCommand
from multi_llm_companion.integrations.vmd_catalog import (
    VmdCatalogError,
    inspect_vmd_motion_library,
    list_vmd_catalog_actions,
    save_vmd_catalog_motion_enabled,
)
from multi_llm_companion.providers import ChatProvider, ProviderError, build_provider
from multi_llm_companion.services import (
    ChatService,
    ConversationContextService,
    DialogueDirector,
    EmotionService,
    MemoryService,
    build_persona_prompt,
)
from multi_llm_companion.services.chat import (
    ProactiveTurnInvalidatedError as ServiceProactiveTurnInvalidatedError,
)
from multi_llm_companion.services.image_sharing import (
    ImageSharingSession,
    OpenAIImageAnalysis,
    PreparedImage,
)
from multi_llm_companion.services.web_search import OpenAIWebSearch, WebSearchSession
from multi_llm_companion.storage import SQLiteConversationRepository, StorageError
from multi_llm_companion.streaming import SpeechStreamFilter
from multi_llm_companion.ui.models import (
    BackendStatus,
    MemorySettings,
    SendOutcome,
    SpeechSettings,
    UiAudioDevice,
    UiAvatar3DStatus,
    UiAvatarAction,
    UiAvatarAppearance,
    UiAvatarBone,
    UiAvatarMaterialMorph,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
    UiAvatarTexturedTriangle,
    UiConversation,
    UiEmotionState,
    UiExpressionPreset,
    UiLipSyncState,
    UiMemory,
    UiMessage,
    UiMmdAvatarCommand,
    UiMotionLibraryItem,
    UiTokenUsageSummary,
    UiVmdBoneFrame,
    UiVmdCameraFrame,
    UiVmdCompatibility,
    UiVmdIkFrame,
    UiVmdIkState,
    UiVmdLightFrame,
    UiVmdMorphFrame,
    UiVmdMotion,
    UiVmdShadowFrame,
)

_UNSET_AVATAR_PREVIEW = object()


class ApplicationError(RuntimeError):
    """画面へ安全に表示できるアプリ全体のエラー。"""


class StreamTurnInvalidatedError(ApplicationError):
    """利用者の新しい発話により、古い生成結果を破棄した。"""


class ProactiveTurnInvalidatedError(ApplicationError):
    """利用者の操作により、古い能動発話を破棄した。"""


class ApplicationBackend:
    """PySide6画面へ、実装詳細に依存しない操作を提供する。"""

    def __init__(self, runtime: RuntimeConfig) -> None:
        self._runtime = runtime
        self._active_provider_name = runtime.active_provider.name
        audio_path = runtime.project_root / runtime.app.audio_config_path
        self._audio_config_path = audio_path.resolve()
        memory_path = runtime.project_root / runtime.app.memory_config_path
        self._memory_config_path = memory_path.resolve()
        avatar_path = runtime.project_root / runtime.app.avatar_config_path
        self._avatar_config_path = avatar_path.resolve()
        try:
            repository = SQLiteConversationRepository(runtime.database_path)
            provider = build_provider(runtime.active_provider)
        except (StorageError, ProviderError) as exc:
            raise ApplicationError(str(exc)) from exc
        self._repository = repository
        self._memory_service = MemoryService(repository, runtime.memory)
        self._emotion_service = EmotionService(repository)
        speech_recognizer = build_speech_recognizer(runtime.audio, runtime.project_root)
        voice_output = (
            build_voice_output(runtime.audio)
            if runtime.environment.use_environment
            else build_voice_output(runtime.audio, environment=runtime.environment)
        )
        motion_actions = _supported_motion_actions(
            runtime,
            audio_available=voice_output.is_available,
        )
        self._expression_policy = ExpressionPolicy(
            supported_gesture_intents=motion_actions,
            supported_gaze_intents=tuple(runtime.avatar.three_d.mmd_mapping.gaze_bones),
            voice_expression_enabled=runtime.audio.voice_output.expression.enabled,
        )
        self._avatar_controller = build_avatar_controller(
            runtime.avatar,
            runtime.project_root,
            audio_available=voice_output.is_available,
        )
        # PMXは数万面あるため、状態表示の更新ごとに画面用データへ変換し直さない。
        self._avatar_preview_source: object = _UNSET_AVATAR_PREVIEW
        self._avatar_preview_cache: UiAvatarModelPreview | None = None
        self._motion_preview_cache: dict[str, UiVmdMotion] = {}
        self._service = ChatService(
            provider=provider,
            repository=repository,
            system_prompt=build_persona_prompt(runtime.persona),
            engine_contract=_engine_contract(runtime, motion_actions),
            dialogue_director=DialogueDirector(runtime.persona),
            speech_recognizer=speech_recognizer,
            voice_output=voice_output,
            avatar_controller=self._avatar_controller,
            memory_service=self._memory_service,
            emotion_service=self._emotion_service,
            conversation_context=ConversationContextService(
                max_recent_messages=runtime.memory.max_recent_messages,
                summary_max_characters=runtime.memory.summary_max_characters,
            ),
        )
        self._configure_web_search(runtime.active_provider)

    def _configure_web_search(self, provider) -> None:
        web_enabled = self._service.web_search.enabled
        image_enabled = self._service.image_sharing.enabled
        client = None
        if (
            provider.base_url.rstrip("/") == "https://api.openai.com/v1"
            and provider.provider == "openai_compatible"
        ):
            # モデル名で固定せず、未対応の機能はAPIのエラーとして伝える。
            client = OpenAIWebSearch(api_key=provider.api_key, model=provider.model)
        self._service.web_search = WebSearchSession(client)
        self._service.image_sharing = ImageSharingSession(
            OpenAIImageAnalysis(api_key=provider.api_key, model=provider.model)
            if client is not None
            else None
        )
        self._service.web_search.set_enabled(web_enabled)
        self._service.image_sharing.enabled = image_enabled

    def image_sharing_available(self) -> bool:
        return self._service.image_sharing.client is not None

    def image_sharing_enabled(self) -> bool:
        return self._service.image_sharing.enabled

    def set_image_sharing_enabled(self, enabled: bool) -> None:
        self._service.set_image_sharing_enabled(enabled)

    def web_search_available(self) -> bool:
        return self._service.web_search.client is not None

    def web_search_enabled(self) -> bool:
        return self._service.web_search.enabled

    def set_web_search_enabled(self, enabled: bool) -> None:
        self._service.set_web_search_enabled(enabled)

    def approve_web_search(self, text: str, query: str, detailed: bool, region: str) -> None:
        self._service.web_search.approve(text, query, detailed=detailed, region=region)

    def persona_settings(self) -> PersonaConfig:
        """編集用のコピーを返し、キャンセル時に稼働中の人格を変えない。"""

        return self._runtime.persona.model_copy(deep=True)

    def change_persona_settings(self, persona: PersonaConfig) -> None:
        """保存成功後に、次の通常会話・能動会話へ人格を反映する。"""

        if self._service.is_busy:
            raise ApplicationError("返答の処理が終わってから人格を変更してください。")
        path = Path(self._runtime.app.persona_path)
        if not path.is_absolute():
            path = self._runtime.project_root / path
        try:
            save_persona_config(path, persona, expected=self._runtime.persona)
        except ConfigError as exc:
            raise ApplicationError(str(exc)) from exc
        self._service.set_persona(persona)
        self._runtime = replace(self._runtime, persona=persona.model_copy(deep=True))

    def status(self) -> BackendStatus:
        provider_kind = self._runtime.providers.providers[
            self._active_provider_name
        ].provider.lower()
        connection = "API不要" if provider_kind == "mock" else "未確認"
        return BackendStatus(
            provider_name=self._service.provider_name,
            model_name=self._service.model_name,
            connection_text=connection,
            speech_recognition_text=self._service.speech_recognizer.status_text,
            voice_output_text=self._service.voice_output.status_text,
        )

    def start_speech_recognition(
        self,
        on_recognized: Callable[[str], None],
        on_error: Callable[[str], None],
        on_partial: Callable[[str], None],
        on_level: Callable[[int], None],
    ) -> None:
        self._service.speech_recognizer.start(
            on_recognized,
            on_error,
            on_partial,
            on_level,
        )

    def stop_speech_recognition(self) -> None:
        self._service.speech_recognizer.stop()

    def finish_speech_recognition(self) -> None:
        self._service.speech_recognizer.finish()

    def _invalidate_active_stream(self) -> None:
        """古い生成の表示・保存を止める。Providerへ取消要求は送らない。"""

        self._service._invalidate_active_stream()

    def cancel_proactive_turn(self) -> None:
        """能動発話を取り消し、後から返る結果も保存させない。"""

        self._service.invalidate_proactive_turn()

    def is_voice_speaking(self) -> bool:
        return self._service.voice_output.is_speaking

    def voice_lip_sync_state(self) -> UiLipSyncState:
        state = self._service.voice_output.lip_sync_state
        return UiLipSyncState(
            state.is_active,
            state.mouth_shape,
            state.is_estimated,
            state.is_timeline_active,
        )

    def stop_voice_output(self) -> None:
        self._service.voice_output.stop()

    def start_voice_output(self, text: str) -> str | None:
        """3D姿勢の準備後に読み上げを開始し、画面用エラーを返す。"""

        return self._start_voice_output(text, reject_invalidated_stream=True)

    def start_proactive_voice_output(self, text: str) -> str | None:
        """能動発話にも、その発話自身の表情と音声設定を適用する。"""

        return self._start_voice_output(text, reject_invalidated_stream=False)

    def _start_voice_output(
        self,
        text: str,
        *,
        reject_invalidated_stream: bool,
        cancel_event: Event | None = None,
    ) -> str | None:
        voice_output = self._service.voice_output
        text = self._spoken_text(text)
        if reject_invalidated_stream and self._service._stream_was_invalidated:
            raise StreamTurnInvalidatedError()
        speak_with_expression = getattr(voice_output, "speak_with_expression", None)
        expression = self._voice_expression() if callable(speak_with_expression) else None
        guarded_start = getattr(voice_output, "_speak_unless_cancelled", None)
        if cancel_event is not None and callable(guarded_start):
            guarded_start(text, expression, cancel_event)
        elif cancel_event is not None and cancel_event.is_set():
            return None
        elif expression is not None:
            speak_with_expression(text, expression)
        else:
            voice_output.speak(text)
        return voice_output.last_error

    def _voice_expression(self) -> ExpressionChannel | None:
        """確定済みの回答と感情から、音声専用の表現強度を得る。"""

        published = self._service._published_response
        if published is None or published[1] is None:
            return None
        response, state = published
        intent = self._expression_policy.normalize(response)
        intent = self._expression_policy.apply_emotion_state(intent, state.emotion, state.intensity)
        return intent.outputs.voice if intent.outputs is not None else None

    def _update_stream_voice_expression(self) -> None:
        update = getattr(self._service.voice_output, "_set_stream_expression", None)
        if callable(update):
            update(self._voice_expression())

    def start_stream_voice_output(self, text: str) -> str | None:
        """文単位の音声先読みを開始し、画面用エラーを返す。"""

        voice_output = self._service.voice_output
        start = getattr(voice_output, "_start_stream_with_expression", None)
        if callable(start):
            start(self._spoken_text(text), self._voice_expression())
        else:
            voice_output.start_stream(self._spoken_text(text))
        return voice_output.last_error

    def enqueue_stream_voice_output(self, text: str) -> str | None:
        """生成中の次の文を音声キューへ追加する。"""

        voice_output = self._service.voice_output
        text = self._spoken_text(text)
        if text.strip():
            voice_output.enqueue_stream(text)
        return voice_output.last_error

    def finish_stream_voice_output(self) -> None:
        """音声キューへ追加する文がなくなったことを通知する。"""

        self._service.voice_output.finish_stream()

    def _spoken_text(self, text: str) -> str:
        published = self._service._published_response
        response = published[0] if published else None
        return re.sub(r"\[S\d+\]", "", text) if response and response.sources else text

    def is_speech_recognition_available(self) -> bool:
        return self._service.speech_recognizer.is_available

    def is_listening(self) -> bool:
        return self._service.speech_recognizer.is_listening

    def speech_settings(self) -> SpeechSettings:
        recognizer = self._service.speech_recognizer
        return SpeechSettings(
            engine=recognizer.engine_name,
            input_device=recognizer.input_device,
            devices=[
                UiAudioDevice(device.id, device.name) for device in recognizer.list_input_devices()
            ],
            silence_threshold=self._runtime.audio.speech_recognition.silence_threshold,
            silence_duration=self._runtime.audio.speech_recognition.silence_duration,
            max_recording_duration=self._runtime.audio.speech_recognition.max_recording_duration,
            partial_interval=self._runtime.audio.speech_recognition.partial_interval,
        )

    def change_speech_settings(
        self,
        engine: str,
        input_device: int | str | None,
        silence_threshold: float,
        silence_duration: float,
        max_recording_duration: float,
        partial_interval: float,
    ) -> BackendStatus:
        normalized = engine.lower()
        if normalized not in {"whisper", "vosk"}:
            raise ApplicationError("音声認識方式はWhisperまたはVoskを選んでください。")
        settings = self._runtime.audio.speech_recognition
        previous_engine = settings.provider
        previous_device = settings.input_device
        previous_values = (
            settings.silence_threshold,
            settings.silence_duration,
            settings.max_recording_duration,
            settings.partial_interval,
        )
        settings.provider = normalized
        settings.input_device = input_device
        settings.silence_threshold = silence_threshold
        settings.silence_duration = silence_duration
        settings.max_recording_duration = max_recording_duration
        settings.partial_interval = partial_interval
        try:
            recognizer = build_speech_recognizer(
                self._runtime.audio,
                self._runtime.project_root,
            )
            save_speech_recognition_settings(
                self._audio_config_path,
                normalized,
                input_device,
                silence_threshold,
                silence_duration,
                max_recording_duration,
                partial_interval,
            )
            self._service.set_speech_recognizer(recognizer)
        except (ConfigError, RuntimeError) as exc:
            settings.provider = previous_engine
            settings.input_device = previous_device
            (
                settings.silence_threshold,
                settings.silence_duration,
                settings.max_recording_duration,
                settings.partial_interval,
            ) = previous_values
            raise ApplicationError(str(exc)) from exc
        return self.status()

    def provider_names(self) -> list[str]:
        return list(self._runtime.providers.providers)

    def current_provider_name(self) -> str:
        return self._active_provider_name

    def environment_settings_context(self) -> tuple[Path, str]:
        source = "環境変数" if self._runtime.environment.use_environment else "画面入力"
        return self._runtime.project_root, source

    def environment_app_config_path(self) -> Path:
        return self._runtime.app_config_path

    def memory_settings(self) -> MemorySettings:
        return MemorySettings(
            enabled=self._memory_service.enabled,
            auto_approve_candidates=self._memory_service.auto_approve_candidates,
        )

    def conversation_context_status(self) -> str:
        """外部LLMへ渡す会話の範囲を、画面用に返す。"""

        return self._service.context_status

    def set_memory_enabled(self, enabled: bool) -> None:
        previous = self._memory_service.enabled
        self._memory_service.set_enabled(enabled)
        try:
            save_memory_enabled(self._memory_config_path, enabled)
        except ConfigError as exc:
            self._memory_service.set_enabled(previous)
            raise ApplicationError(str(exc)) from exc

    def set_memory_auto_approve_candidates(self, enabled: bool) -> None:
        previous = self._memory_service.auto_approve_candidates
        self._memory_service.set_auto_approve_candidates(enabled)
        try:
            save_memory_auto_approve_candidates(self._memory_config_path, enabled)
        except ConfigError as exc:
            self._memory_service.set_auto_approve_candidates(previous)
            raise ApplicationError(str(exc)) from exc

    def list_memories(self, status: str | None = None) -> list[UiMemory]:
        try:
            memory_status = MemoryStatus(status) if status is not None else None
            return [
                _to_ui_memory(item) for item in self._memory_service.list_memories(memory_status)
            ]
        except (StorageError, ValueError) as exc:
            raise ApplicationError(str(exc)) from exc

    def approve_memory(self, memory_id: int, content: str, category: str) -> UiMemory:
        """旧UI利用者向けに残す互換用の入口。"""
        try:
            return _to_ui_memory(
                self._memory_service.approve(memory_id, content, MemoryCategory(category))
            )
        except (StorageError, ValueError) as exc:
            raise ApplicationError(str(exc)) from exc

    def update_memory(self, memory_id: int, content: str, category: str) -> UiMemory:
        try:
            return _to_ui_memory(
                self._memory_service.update(memory_id, content, MemoryCategory(category))
            )
        except (StorageError, ValueError) as exc:
            raise ApplicationError(str(exc)) from exc

    def delete_memory(self, memory_id: int) -> None:
        try:
            if not self._memory_service.delete(memory_id):
                raise ApplicationError("削除する記憶が見つかりませんでした。")
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc

    def delete_all_memories(self) -> int:
        try:
            return self._memory_service.delete_all()
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc

    def emotion_state(self, conversation_id: str) -> UiEmotionState | None:
        try:
            state = self._emotion_service.current(conversation_id)
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc
        return _to_ui_emotion_state(state) if state is not None else None

    def avatar_action(self, conversation_id: str) -> UiAvatarAction:
        """会話内の最後の構造化回答から、表示中の動作と視線を戻す。"""

        try:
            messages = self._repository.list_messages(conversation_id)
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc
        response = next(
            (
                item.response
                for item in reversed(messages)
                if item.role == MessageRole.ASSISTANT and item.response is not None
            ),
            None,
        )
        if response is None:
            return UiAvatarAction(gesture="", gaze="")
        intent = self._expression_policy.normalize(
            response,
            conversation_state=ConversationState.REACTING,
        )
        try:
            state = self._emotion_service.current(conversation_id)
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc
        if state is not None:
            intent = self._expression_policy.apply_emotion_state(
                intent,
                state.emotion,
                state.intensity,
            )
        outputs = intent.outputs
        assert outputs is not None
        return UiAvatarAction(
            gesture=intent.gesture_intent,
            gaze=intent.gaze_intent,
            expression_intensity=outputs.expression.intensity,
            gesture_intensity=outputs.gesture.intensity,
            gaze_intensity=outputs.gaze.intensity,
            natural_motion_intensity=outputs.natural_motion.intensity,
            status_text=outputs.status_text,
        )

    def avatar_appearance(self) -> UiAvatarAppearance:
        return _to_ui_avatar_appearance(self._runtime.avatar)

    def avatar_3d_status(self) -> UiAvatar3DStatus:
        controller = self._avatar_controller
        return UiAvatar3DStatus(
            controller_name=controller.controller_name,
            status_text=controller.status_text,
            is_available=controller.is_available,
            render_quality=self._runtime.avatar.three_d.render_quality,
        )

    def avatar_model_preview(self) -> UiAvatarModelPreview | None:
        preview = self._avatar_controller.model_preview
        if preview is self._avatar_preview_source:
            return self._avatar_preview_cache
        if preview is None:
            self._avatar_preview_source = None
            self._avatar_preview_cache = None
            return None
        converted = UiAvatarModelPreview(
            model_name=preview.model_name,
            vertex_count=preview.vertex_count,
            face_count=preview.face_count,
            points=preview.points,
            normals=preview.normals,
            triangles=preview.triangles,
            mouth_morphs=preview.mouth_morphs,
            vertex_morph_names=preview.vertex_morph_names,
            emotion_morphs=preview.emotion_morphs,
            named_vertex_morphs=preview.named_vertex_morphs,
            uvs=preview.uvs,
            additional_uvs=preview.additional_uvs,
            morph_names=preview.morph_names,
            named_uv_morphs=preview.named_uv_morphs,
            named_material_morphs={
                name: tuple(
                    UiAvatarMaterialMorph(
                        material_index=item.material_index,
                        operation=item.operation,
                        diffuse=item.diffuse,
                        specular=item.specular,
                        shininess=item.shininess,
                        ambient=item.ambient,
                        edge_color=item.edge_color,
                        edge_size=item.edge_size,
                        texture_factor=item.texture_factor,
                        sphere_factor=item.sphere_factor,
                        toon_factor=item.toon_factor,
                        weight_scale=item.weight_scale,
                    )
                    for item in items
                )
                for name, items in preview.named_material_morphs.items()
            },
            bones=tuple(
                UiAvatarBone(
                    name=bone.name,
                    parent_index=bone.parent_index,
                    position=bone.position,
                    ik_target_index=bone.ik_target_index,
                    ik_iterations=bone.ik_iterations,
                    ik_limit_radians=bone.ik_limit_radians,
                    ik_links=bone.ik_links,
                    deform_layer=bone.deform_layer,
                    inherit_parent_index=bone.inherit_parent_index,
                    inherit_weight=bone.inherit_weight,
                    inherit_rotation=bone.inherit_rotation,
                    inherit_translation=bone.inherit_translation,
                    fixed_axis=bone.fixed_axis,
                    local_axis_x=bone.local_axis_x,
                    local_axis_z=bone.local_axis_z,
                    after_physics=bone.after_physics,
                    allows_rotation=bone.allows_rotation,
                    allows_translation=bone.allows_translation,
                    ik_link_limits=bone.ik_link_limits,
                )
                for bone in preview.bones
            ),
            vertex_weights=preview.vertex_weights,
            rigid_bodies=tuple(
                UiAvatarRigidBody(
                    bone_index=body.bone_index,
                    position=body.position,
                    mass=body.mass,
                    linear_damping=body.linear_damping,
                    mode=body.mode,
                    name=body.name,
                    collision_group=body.collision_group,
                    non_collision_mask=body.non_collision_mask,
                    shape=body.shape,
                    size=body.size,
                    rotation=body.rotation,
                    angular_damping=body.angular_damping,
                    restitution=body.restitution,
                    friction=body.friction,
                )
                for body in preview.rigid_bodies
            ),
            physics_joints=tuple(
                UiAvatarPhysicsJoint(
                    rigid_body_a=joint.rigid_body_a,
                    rigid_body_b=joint.rigid_body_b,
                    position=joint.position,
                    spring_translation=joint.spring_translation,
                    name=joint.name,
                    rotation=joint.rotation,
                    translation_min=joint.translation_min,
                    translation_max=joint.translation_max,
                    rotation_min=joint.rotation_min,
                    rotation_max=joint.rotation_max,
                    spring_rotation=joint.spring_rotation,
                )
                for joint in preview.physics_joints
            ),
            vmd_motion=(
                _to_ui_vmd_motion(preview.vmd_motion) if preview.vmd_motion is not None else None
            ),
            idle_motions=tuple(_to_ui_vmd_motion(motion) for motion in preview.idle_motions),
            textured_triangles=tuple(
                UiAvatarTexturedTriangle(
                    indexes=triangle.indexes,
                    uvs=triangle.uvs,
                    texture_path=triangle.texture_path,
                    diffuse=triangle.diffuse,
                    additional_uvs=triangle.additional_uvs,
                    material_index=triangle.material_index,
                    is_transparent=triangle.is_transparent,
                    normals=triangle.normals,
                    sphere_texture_path=triangle.sphere_texture_path,
                    sphere_mode=triangle.sphere_mode,
                    toon_texture_path=triangle.toon_texture_path,
                    toon_shared_index=triangle.toon_shared_index,
                    is_double_sided=triangle.is_double_sided,
                    specular=triangle.specular,
                    shininess=triangle.shininess,
                    ambient=triangle.ambient,
                    draw_outline=triangle.draw_outline,
                    edge_color=triangle.edge_color,
                    edge_size=triangle.edge_size,
                    edge_scales=triangle.edge_scales,
                )
                for triangle in preview.textured_triangles
            ),
            dense_textured_triangles=tuple(
                UiAvatarTexturedTriangle(
                    indexes=triangle.indexes,
                    uvs=triangle.uvs,
                    texture_path=triangle.texture_path,
                    diffuse=triangle.diffuse,
                    additional_uvs=triangle.additional_uvs,
                    material_index=triangle.material_index,
                    is_transparent=triangle.is_transparent,
                    normals=triangle.normals,
                    sphere_texture_path=triangle.sphere_texture_path,
                    sphere_mode=triangle.sphere_mode,
                    toon_texture_path=triangle.toon_texture_path,
                    toon_shared_index=triangle.toon_shared_index,
                    is_double_sided=triangle.is_double_sided,
                    specular=triangle.specular,
                    shininess=triangle.shininess,
                    ambient=triangle.ambient,
                    draw_outline=triangle.draw_outline,
                    edge_color=triangle.edge_color,
                    edge_size=triangle.edge_size,
                    edge_scales=triangle.edge_scales,
                )
                for triangle in preview.dense_textured_triangles
            ),
            thin_cloth_contact_enabled=preview.thin_cloth_contact_enabled,
            thin_cloth_source_body_patterns=preview.thin_cloth_source_body_patterns,
            thin_cloth_target_body_patterns=preview.thin_cloth_target_body_patterns,
            thin_cloth_maximum_correction_per_step=(preview.thin_cloth_maximum_correction_per_step),
        )
        self._avatar_preview_source = preview
        self._avatar_preview_cache = converted
        return converted

    def mmd_avatar_command(self) -> UiMmdAvatarCommand | None:
        command = self._avatar_controller.mmd_command
        if command is None:
            return None
        return self._to_ui_mmd_avatar_command(command)

    def expression_preview_command(self, emotion: str) -> UiMmdAvatarCommand | None:
        """会話を変更しない表情確認命令を、PMX画面用に返す。"""

        create = getattr(self._avatar_controller, "expression_preview_command", None)
        if not callable(create):
            return None
        command = create(emotion)
        return self._to_ui_mmd_avatar_command(command) if command is not None else None

    def save_expression_preset(
        self,
        preset_name: str,
        emotion: str,
        morph_weights: dict[str, float],
    ) -> None:
        """実在PMXモーフの組合せだけを会話表情として保存する。"""

        if self._service.is_busy or self._service.voice_output.is_speaking:
            raise ApplicationError("回答または音声の処理中は表情を保存できません。")
        preview = self._avatar_controller.model_preview
        if preview is None:
            raise ApplicationError("表情を保存できるPMXモデルが読み込まれていません。")
        selected_names = tuple(name.strip() for name in morph_weights)
        missing = sorted(set(selected_names) - set(preview.morph_names))
        if missing:
            raise ApplicationError(f"PMXにないモーフは保存できません: {'、'.join(missing)}")
        mouth_names = set(self._runtime.avatar.three_d.mmd_mapping.mouth_morphs.values())
        protected = sorted(set(selected_names) & mouth_names)
        if protected:
            raise ApplicationError(
                f"口形は音声同期専用のため保存できません: {'、'.join(protected)}"
            )
        try:
            avatar = save_avatar_expression_preset(
                self._avatar_config_path,
                preset_name,
                emotion,
                morph_weights,
            )
        except ConfigError as exc:
            raise ApplicationError(str(exc)) from exc
        self._replace_avatar_config(avatar)

    def list_expression_presets(self) -> list[UiExpressionPreset]:
        """保存済み表情を、PMXの実モーフ名へ解決して返す。"""

        settings = self._runtime.avatar.three_d
        assigned: dict[str, set[str]] = {}
        for emotion, preset_name in settings.emotion_presets.items():
            assigned.setdefault(preset_name, set()).add(emotion)
        for emotion, preset_names in settings.emotion_preset_variants.items():
            for preset_name in preset_names:
                assigned.setdefault(preset_name, set()).add(emotion)
        result: list[UiExpressionPreset] = []
        mapping = settings.mmd_mapping.expression_morphs
        for name, preset in sorted(settings.expression_presets.items()):
            resolved: dict[str, float] = {}
            unresolved: list[str] = []
            for role, weight in preset.morphs.items():
                morph_name = mapping.get(role)
                if morph_name is None:
                    unresolved.append(role)
                    continue
                resolved[morph_name] = min(1.0, resolved.get(morph_name, 0.0) + weight)
            result.append(
                UiExpressionPreset(
                    name=name,
                    morph_weights=tuple(resolved.items()),
                    emotions=tuple(sorted(assigned.get(name, set()))),
                    unresolved_roles=tuple(unresolved),
                )
            )
        return result

    def delete_expression_preset(self, preset_name: str) -> None:
        """選択した表情だけを削除し、アバター制御へ即時反映する。"""

        if self._service.is_busy or self._service.voice_output.is_speaking:
            raise ApplicationError("回答または音声の処理中は表情を削除できません。")
        try:
            avatar = delete_avatar_expression_preset(
                self._avatar_config_path,
                preset_name,
            )
        except ConfigError as exc:
            raise ApplicationError(str(exc)) from exc
        self._replace_avatar_config(avatar)

    def _replace_avatar_config(self, avatar: AvatarConfig) -> None:
        """保存後の設定から制御器を一度だけ作り直す。"""

        self._runtime = replace(self._runtime, avatar=avatar)
        controller = build_avatar_controller(
            avatar,
            self._runtime.project_root,
            audio_available=self._service.voice_output.is_available,
        )
        self._avatar_controller = controller
        self._service.replace_avatar_controller(controller)
        self._avatar_preview_source = _UNSET_AVATAR_PREVIEW
        self._avatar_preview_cache = None

    @staticmethod
    def _to_ui_mmd_avatar_command(command: MmdAvatarCommand) -> UiMmdAvatarCommand:
        """制御器の命令を、画面専用の不変データへ写す。"""

        return UiMmdAvatarCommand(
            emotion=command.emotion,
            emotion_intensity=command.emotion_intensity,
            emotion_morph=command.emotion_morph,
            gaze=command.gaze,
            gaze_bone=command.gaze_bone,
            gesture=command.gesture,
            gesture_target=command.gesture_target,
            gesture_motion_id=command.gesture_motion_id,
            gesture_motion_type=command.gesture_motion_type,
            gesture_body_mask=command.gesture_body_mask,
            gesture_motion=(
                _to_ui_vmd_motion(command.gesture_motion)
                if command.gesture_motion is not None
                else None
            ),
            gesture_intensity_scale=command.gesture_intensity_scale,
            gesture_intensity=command.gesture_intensity,
            gaze_intensity=command.gaze_intensity,
            natural_motion_intensity=command.natural_motion_intensity,
            head_bone=command.head_bone,
            upper_body_bone=command.upper_body_bone,
            expression_preset=command.expression_preset,
            expression_morphs=command.expression_morphs,
            expression_status_text=command.expression_status_text,
        )

    def list_motion_library(self) -> list[UiMotionLibraryItem]:
        """無効項目と利用不可理由を含むローカルVMD一覧を返す。"""

        settings = self._runtime.avatar.three_d
        if not settings.motion_library_enabled or not settings.vmd_catalog_path:
            self._motion_preview_cache.clear()
            return []
        try:
            inspected = inspect_vmd_motion_library(
                self._runtime.project_root,
                settings.vmd_catalog_path,
                self._avatar_controller.model_preview,
                audio_available=self._service.voice_output.is_available,
            )
        except VmdCatalogError as exc:
            raise ApplicationError(str(exc)) from exc
        self._motion_preview_cache = {
            item.motion_id: _to_ui_vmd_motion(item.parsed_motion)
            for item in inspected
            if item.available and item.parsed_motion is not None
        }
        return [
            UiMotionLibraryItem(
                motion_id=item.motion_id,
                source_name=Path(item.motion.file).name,
                motion_type=item.motion.motion_type,
                action=item.motion.selection_action,
                moods=tuple(item.motion.moods),
                energy=item.motion.energy,
                enabled=item.motion.enabled,
                available=item.available,
                status_text="利用可能" if item.available else " / ".join(item.issues),
                audio_required=item.motion.audio_required,
                frame_count=item.parsed_motion.frame_count if item.parsed_motion is not None else 0,
            )
            for item in inspected
        ]

    def load_motion_preview(self, motion_id: str) -> UiVmdMotion:
        """検査済みVMDを本番命令とは別のプレビューへ渡す。"""

        normalized = motion_id.strip()
        if normalized not in self._motion_preview_cache:
            self.list_motion_library()
        motion = self._motion_preview_cache.get(normalized)
        if motion is None:
            raise ApplicationError(f"モーション「{normalized}」はプレビューできません。")
        return motion

    def set_motion_enabled(self, motion_id: str, enabled: bool) -> None:
        """カタログの一件だけを保存し、会話候補を即時再読込する。"""

        settings = self._runtime.avatar.three_d
        if not settings.motion_library_enabled or not settings.vmd_catalog_path:
            raise ApplicationError("モーションライブラリが有効ではありません。")
        if self._service.is_busy or self._service.voice_output.is_speaking:
            raise ApplicationError("回答または音声の処理中はモーション設定を変更できません。")
        try:
            save_vmd_catalog_motion_enabled(
                self._runtime.project_root,
                settings.vmd_catalog_path,
                motion_id,
                enabled,
            )
        except VmdCatalogError as exc:
            raise ApplicationError(str(exc)) from exc
        controller = build_avatar_controller(
            self._runtime.avatar,
            self._runtime.project_root,
            audio_available=self._service.voice_output.is_available,
        )
        self._avatar_controller = controller
        self._service.replace_avatar_controller(controller)
        self._avatar_preview_source = _UNSET_AVATAR_PREVIEW
        self._avatar_preview_cache = None
        self._motion_preview_cache.clear()
        motion_actions = _supported_motion_actions(
            self._runtime,
            audio_available=self._service.voice_output.is_available,
        )
        self._expression_policy = ExpressionPolicy(
            supported_gesture_intents=motion_actions,
            supported_gaze_intents=tuple(self._runtime.avatar.three_d.mmd_mapping.gaze_bones),
            voice_expression_enabled=self._runtime.audio.voice_output.expression.enabled,
        )
        self._service.set_engine_contract(_engine_contract(self._runtime, motion_actions))

    def generate_confirmation_vmd(self) -> str:
        """画面から要求された場合だけ、ローカルの確認用VMDを作る。"""

        try:
            path = self._avatar_controller.generate_confirmation_vmd()
        except RuntimeError as exc:
            raise ApplicationError(str(exc)) from exc
        try:
            return str(Path(path).resolve().relative_to(self._runtime.project_root))
        except ValueError:
            return path

    def token_usage_summary(
        self,
        conversation_id: str | None = None,
    ) -> UiTokenUsageSummary:
        try:
            summary = self._repository.get_token_usage_summary(conversation_id)
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc
        return _to_ui_token_usage_summary(summary)

    def list_conversations(self) -> list[UiConversation]:
        try:
            return [
                UiConversation(id=item.id, title=item.title)
                for item in self._service.list_conversations()
            ]
        except StorageError as exc:
            raise ApplicationError(str(exc)) from exc

    def load_messages(self, conversation_id: str) -> list[UiMessage]:
        try:
            return [
                UiMessage(
                    role=item.role.value,
                    content=(
                        (_recover_markdown_speech(item.content) or item.content)
                        if item.role.value == "assistant"
                        else item.content
                    ),
                    sources=tuple(item.response.sources) if item.response else (),
                    image_attachment=item.response.image_attachment if item.response else None,
                    created_at=item.created_at,
                )
                for item in self._service.load_conversation(conversation_id)
                if item.role.value in {"user", "assistant"}
            ]
        except (StorageError, RuntimeError) as exc:
            raise ApplicationError(str(exc)) from exc

    def create_conversation(self, title: str) -> str:
        try:
            return self._service.start_new_conversation(title).id
        except (StorageError, RuntimeError) as exc:
            raise ApplicationError(str(exc)) from exc

    def delete_conversation(self, conversation_id: str) -> None:
        try:
            deleted = self._service.delete_conversation(conversation_id)
        except (StorageError, RuntimeError) as exc:
            raise ApplicationError(str(exc)) from exc
        if not deleted:
            raise ApplicationError("削除する会話履歴が見つかりませんでした。")

    def generate_proactive_turn(self, conversation_id: str) -> SendOutcome:
        """利用者入力なしの短い会話を生成し、通常の応答と同じ形で返す。"""
        return self._generate_proactive_turn(conversation_id)

    def _generate_proactive_turn(self, conversation_id: str, *, cancelled=None) -> SendOutcome:
        """開始前の取消も確認し、同じ世代の結果状態でUI応答を組み立てる。"""
        try:
            with self._service._state_lock:
                if cancelled is not None and cancelled():
                    raise ServiceProactiveTurnInvalidatedError()
                if self._service.current_conversation_id != conversation_id:
                    self._service.load_conversation(conversation_id)
                generation = self._service._proactive_generation
            return self._service._generate_proactive_turn(
                cancelled=cancelled,
                expected_generation=generation,
                finalize=lambda response: self._build_send_outcome(
                    response, include_memory_feedback=False
                ),
            )
        except ServiceProactiveTurnInvalidatedError as exc:
            raise ProactiveTurnInvalidatedError() from exc
        except (ProviderError, StorageError, RuntimeError) as exc:
            raise ApplicationError(str(exc)) from exc
        except Exception as exc:
            raise ApplicationError(
                f"能動発話の処理中に予期しないエラーが発生しました（{type(exc).__name__}）。\n"
                "アプリを終了し、run_debug.bat で起動し直してください。\n"
                "解決しない場合は、この表示を控えて報告してください。"
            ) from exc

    def change_provider(self, provider_name: str) -> BackendStatus:
        configured = self._runtime.providers.providers.get(provider_name)
        if configured is None:
            raise ApplicationError(f"プロバイダー「{provider_name}」が見つかりません。")

        try:
            resolved = resolve_provider(provider_name, configured, self._runtime.environment)
            new_provider = build_provider(resolved)
        except (ConfigError, ProviderError) as exc:
            raise ApplicationError(str(exc)) from exc

        previous: ChatProvider = self._service.provider
        try:
            self._service.set_provider(new_provider)
            save_active_provider(self._runtime.app_config_path, provider_name)
        except Exception as exc:
            self._service.set_provider(previous)
            if isinstance(exc, (ConfigError, ProviderError, RuntimeError)):
                raise ApplicationError(str(exc)) from exc
            raise ApplicationError("プロバイダー設定を保存できませんでした。") from exc

        self._active_provider_name = provider_name
        self._configure_web_search(resolved)
        return self.status()

    @property
    def _uses_research_flow(self) -> bool:
        return self._service._uses_research_flow

    @property
    def _research_phase(self) -> str:
        return self._service._research_phase

    def stream_message(
        self,
        conversation_id: str,
        user_text: str,
        on_chunk: Callable[[str], None],
        *,
        image: PreparedImage | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> SendOutcome:
        stream_filter = SpeechStreamFilter(require_structured=True)
        try:
            if self._service.current_conversation_id != conversation_id:
                self._service.load_conversation(conversation_id)
            for raw_chunk in self._service.stream_message(
                user_text, image=image, cancelled=cancelled
            ):
                speech_chunk = stream_filter.feed(raw_chunk)
                if speech_chunk:
                    on_chunk(speech_chunk)
        except (ProviderError, StorageError, RuntimeError) as exc:
            raise ApplicationError(str(exc)) from exc
        except Exception as exc:
            # 予期しない例外の詳細に秘密情報が含まれる可能性を避ける。
            raise ApplicationError(
                f"回答の処理中に予期しないエラーが発生しました（{type(exc).__name__}）。\n"
                "アプリを終了し、run_debug.bat で起動し直してください。\n"
                "解決しない場合は、この表示を控えて報告してください。"
            ) from exc

        if self._service._stream_was_invalidated:
            self._memory_service.clear_last_candidates()
            raise StreamTurnInvalidatedError()

        response = self._service.last_response
        if response is None:
            raise ApplicationError("LLMから回答本文を取得できませんでした。")
        if remaining := stream_filter.finish(response.speech):
            on_chunk(remaining)
        return self._build_send_outcome(response)

    def _build_send_outcome(
        self,
        response: AssistantResponse,
        *,
        include_memory_feedback: bool = True,
    ) -> SendOutcome:
        usage = self._service.last_usage
        command = self._avatar_controller.mmd_command
        intent = self._expression_policy.normalize(
            response,
            conversation_state=ConversationState.REACTING,
        )
        published = self._service._published_response
        emotion_state = published[1] if published and published[0] is response else None
        if emotion_state is not None:
            intent = self._expression_policy.apply_emotion_state(
                intent,
                emotion_state.emotion,
                emotion_state.intensity,
            )
        outputs = intent.outputs
        assert outputs is not None
        effective_gesture = command.gesture if command is not None else intent.gesture_intent
        effective_gaze = command.gaze if command is not None else intent.gaze_intent
        return SendOutcome(
            speech=response.speech,
            sources=tuple(response.sources),
            image_attachment=response.image_attachment,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            gesture=effective_gesture,
            gaze=effective_gaze,
            # 音声は、画面が会話姿勢の準備完了を確認してから開始する。
            voice_error=None,
            memory_candidates=[_to_ui_memory(item) for item in self._memory_service.last_candidates]
            if include_memory_feedback
            else [],
            used_memories=[
                _to_ui_memory(item) for item in self._memory_service.last_context_memories
            ]
            if include_memory_feedback
            else [],
            emotion_state=_to_ui_emotion_state(emotion_state),
            expression_intensity=outputs.expression.intensity,
            gesture_intensity=outputs.gesture.intensity,
            gaze_intensity=outputs.gaze.intensity,
            natural_motion_intensity=outputs.natural_motion.intensity,
            expression_status_text=outputs.status_text,
        )

    def close(self) -> None:
        try:
            self._service.close()
        finally:
            close_voice = getattr(self._service.voice_output, "close", None)
            if callable(close_voice):
                close_voice()
            wait_for_cleanup = getattr(self._service.voice_output, "_wait_for_cleanup", None)
            if callable(wait_for_cleanup) and wait_for_cleanup() is False:
                raise RuntimeError("音声の終了処理を完了できませんでした。")


def _system_prompt(runtime: RuntimeConfig) -> str:
    """旧来の確認用に、Static PersonaとEngine Contractを結合して返す。"""

    return f"{build_persona_prompt(runtime.persona).rstrip()}\n\n{_engine_contract(runtime)}"


def _engine_contract(
    runtime: RuntimeConfig,
    motion_actions: tuple[str, ...] | None = None,
) -> str:
    """人格や会話状態と分離した、回答JSONの機械契約。"""

    actions = _supported_motion_actions(runtime) if motion_actions is None else motion_actions
    gesture_options = list(dict.fromkeys(("none", "wave", "nod", *actions)))
    gesture_names = "、".join(gesture_options)
    gesture_guidance = _gesture_action_guidance(tuple(gesture_options))
    response_contract = f"""
回答は必ず次のキーを持つJSONオブジェクト1つだけにし、speechを最初のキーにしてください。
speech: 利用者へ伝える発言文字列
emotion: neutral, happy, sad, angry, embarrassed, surprised のいずれか
emotion_intensity: 0から1の数値
gesture: {gesture_names} のいずれか。動作が不要ならnone。ローカルファイル名ではなく、利用者の発言に合う抽象的なactionだけを返す。
{gesture_guidance}
gaze: user、forward、left、right、up、down のいずれか。視線を指定しない場合はforward。
memory_candidates: 旧形式との互換キー。常に空の配列にする。
memory_updates: 「会話の意味による記憶」の規則と形式で、本人の今回の発言に基づく追加・訂正・削除を配列にする。変更がない場合、記憶が無効の場合、検索・画像の結果回答、能動発話では空配列。規則と根拠が渡されていなければ空配列。正確な時刻のアラームや外部通知を約束してはいけない。
dialogue_mode: casual, listening, supporting, playful, explaining, planning, boundary のいずれか。
dialogue_act: acknowledge, answer, ask, suggest, comfort, explain, plan, play, boundary, close のいずれか。
current_topic: 今回の短い話題。確定した利用者情報の保存には使わない。
open_loop: 自然な次回継続候補。閉じた話題や不要な問いは空文字にする。
dialogue_mode、dialogue_act、current_topic、open_loopは後方互換の任意メタデータであり、欠落しても処理できる。
APIキー、パスワード、認証情報、利用者が記憶しないよう求めた情報は絶対に含めない。
利用者が「忘れて」と求め、対象を特定できないときは記憶を削除したと断言せず、「記憶の管理から選んで削除できます」と短く案内してください。
Markdownのコード囲みは付けないでください。
""".strip()
    return response_contract


_GESTURE_ACTION_GUIDANCE = {
    "wave": "挨拶または別れで小さく手を振る",
    "nod": "同意・了承・短い相づちを示す",
    "shake_head_no": "明確な否定を穏やかに示す",
    "unsure": "答えを断定できない軽い迷いを示す",
    "compare": "二つ以上の選択肢を比較する",
    "explain": "理由・手順・仕組みを説明する",
    "remember": "思い出した、または気づいた内容を話す",
    "head_scratch": "困惑または照れを控えめに示す",
    "hair_flip": "気分を切り替える軽い仕草に限る",
    "happy_sway": "はっきりした喜びを示す",
    "stretch": "休憩や伸び・あくびを発言で直接扱うときだけ使う",
    "cold": "寒さや冷えを発言で直接扱うときだけ使う",
    "delight": "喜びジャンプ。利用者から『ジャンプして』『跳んで』と動作を頼まれたとき、または特別にうれしい出来事への強い喜びに使う",
    "laugh": "実際に可笑しい内容へ笑って応じる",
    "sad": "静かな悲しみを身体で補う",
    "cry": "強い悲しみを直接扱うときだけ使い、通常の共感では使わない",
    "scared": "明確な恐怖や強い驚きを直接扱うときだけ使う",
    "shrug": "軽い呆れや困った状況へ肩をすくめる。深刻な悩みには使わない",
    "playful_shoot": "指でバキュンする遊びを明示的に求められたときだけ使う",
    "rhythm_small": "音楽やリズムの話題で小さくリズムに乗る",
    "rhythm_large": "大きくリズムに乗る動作を明示的に求められたときだけ使う",
    "rhythm_extra_large": "特大のリズム動作を明示的に求められたときだけ使う",
    "point_direction": "方向を指し示す動作を求められたときに使う",
    "cherish": "大切な気持ちや感謝を胸に留める仕草で示す",
    "reassure": "頑張りをねぎらい、落ち着いて励ます",
    "curious": "利用者の話へ興味を示し、詳しく聞きたいときに使う",
    "lean_in": "近くで見たい・覗き込みたいという文脈に限る",
    "surprise": "予想外の出来事へのはっきりした驚きを示す",
    "proud": "達成を少し得意げに喜ぶ。利用者を見下す場面には使わない",
    "peace": "記念や成功を明るいピースで祝う",
    "cheer": "これから挑戦する利用者へ明るい応援を送る",
    "walk_right": "右へ歩く動作を明示的に求められたときだけ使う",
    "walk_left": "左へ歩く動作を明示的に求められたときだけ使う",
    "think": "相談や難しい問いについて考えを巡らせる",
    "awkward_smile": "軽い失敗や気まずさへ苦笑いする。深刻な悩みには使わない",
    "enter": "登場の演技を明示的に求められたときだけ使う",
    "playful_airplane": "飛行機ごっこを明示的に求められたときだけ使う",
    "startled": "腕を慌てて動かすほどの強い驚きや演技の要求に限る",
    "doubt": "矛盾や不確かな話へ慎重な疑問を示す",
    "small_success": "小さな成功や良い知らせを控えめなガッツポーズで喜ぶ",
    "robot_dance": "ロボットダンスを明示的に求められたときだけ使う",
    "reject": "明確な拒否を身体で示す必要があるときに限る",
    "determined": "拳を突き出す気合いの演技を求められたときだけ使う",
    "flustered": "突然の話や照れる話題で目が泳ぐような戸惑いを示す",
    "secret": "内緒話の仕草を示す。秘密情報の開示を促さない",
    "embarrassed": "自分の照れや恥ずかしさを顔を覆う仕草で示す",
}


def _gesture_action_guidance(actions: tuple[str, ...]) -> str:
    """有効な抽象actionだけへ、人格に合う控えめな用途を添える。"""

    lines = [
        "gestureは発話内容を身体で補う必要があるときだけ一つ選び、"
        "感情名だけを理由に大きな反応を使わない。迷う場合はnone。",
        "表情・顔だけを求める依頼はemotionとemotion_intensityで表し、gestureはnoneにする。"
        "『うれしい顔』『悲しい顔』『照れた顔』『驚いた顔』だけでは身体動作を付けない。",
        "身体動作も明示的に求められた場合は表情と動作を独立に指定する。"
        "禁止・否定された動作は選ばない。emotionには上記の6値だけを使い、"
        "遊びの雰囲気はdialogue_modeのplayfulで表す。",
    ]
    lines.extend(
        f"gesture用途 - {action}: {_GESTURE_ACTION_GUIDANCE[action]}。"
        for action in actions
        if action in _GESTURE_ACTION_GUIDANCE
    )
    return "\n".join(lines)


def _supported_motion_actions(
    runtime: RuntimeConfig,
    *,
    audio_available: bool = False,
) -> tuple[str, ...]:
    """ローカル識別名を隠し、カタログの抽象actionだけを返す。"""

    settings = runtime.avatar.three_d
    if not settings.motion_library_enabled and not settings.gesture_motion_ids:
        return ()
    motion_ids = None if settings.motion_library_enabled else settings.gesture_motion_ids
    try:
        return list_vmd_catalog_actions(
            runtime.project_root,
            settings.vmd_catalog_path,
            motion_ids,
            audio_available=audio_available,
        )
    except VmdCatalogError:
        return ()


def _to_ui_memory(memory: MemoryEntry) -> UiMemory:
    """ドメインの記憶を画面向けの小さな型へ変換する。"""

    return UiMemory(
        id=memory.id,
        content=memory.content,
        category=memory.category.value,
        status=memory.status.value,
        created_at=memory.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
        updated_at=memory.updated_at.astimezone().strftime("%Y-%m-%d %H:%M"),
        source_conversation_id=memory.source_conversation_id or "",
    )


def _to_ui_vmd_motion(motion: AvatarVmdMotion) -> UiVmdMotion:
    """検証済みVMDを、PMX本体を複製せず画面用へ変換する。"""

    return UiVmdMotion(
        name=motion.name,
        frame_count=motion.frame_count,
        bone_frames=tuple(
            UiVmdBoneFrame(
                bone_name=frame.bone_name,
                frame_number=frame.frame_number,
                translation=frame.translation,
                rotation=frame.rotation,
                interpolation=frame.interpolation,
            )
            for frame in motion.bone_frames
        ),
        loop=motion.loop,
        morph_frames=tuple(
            UiVmdMorphFrame(
                morph_name=frame.morph_name,
                frame_number=frame.frame_number,
                weight=frame.weight,
            )
            for frame in motion.morph_frames
        ),
        intensity_scale=motion.intensity_scale,
        target_model_name=motion.target_model_name,
        camera_frames=tuple(
            UiVmdCameraFrame(
                frame_number=frame.frame_number,
                distance=frame.distance,
                position=frame.position,
                rotation=frame.rotation,
                interpolation=frame.interpolation,
                view_angle=frame.view_angle,
                perspective=frame.perspective,
            )
            for frame in motion.camera_frames
        ),
        light_frames=tuple(
            UiVmdLightFrame(
                frame_number=frame.frame_number,
                color=frame.color,
                position=frame.position,
            )
            for frame in motion.light_frames
        ),
        shadow_frames=tuple(
            UiVmdShadowFrame(
                frame_number=frame.frame_number,
                mode=frame.mode,
                distance=frame.distance,
            )
            for frame in motion.shadow_frames
        ),
        ik_frames=tuple(
            UiVmdIkFrame(
                frame_number=frame.frame_number,
                display_model=frame.display_model,
                states=tuple(
                    UiVmdIkState(ik_name=state.ik_name, enabled=state.enabled)
                    for state in frame.states
                ),
            )
            for frame in motion.ik_frames
        ),
        compatibility=(
            UiVmdCompatibility(
                supported_bones=motion.compatibility.supported_bones,
                missing_bones=motion.compatibility.missing_bones,
                supported_morphs=motion.compatibility.supported_morphs,
                missing_morphs=motion.compatibility.missing_morphs,
                supported_iks=motion.compatibility.supported_iks,
                missing_iks=motion.compatibility.missing_iks,
                translation_height_ratio=motion.compatibility.translation_height_ratio,
                target_model_matches=motion.compatibility.target_model_matches,
                unsupported_sections=motion.compatibility.unsupported_sections,
            )
            if motion.compatibility is not None
            else None
        ),
    )


def _to_ui_emotion_state(state: EmotionState | None) -> UiEmotionState | None:
    if state is None:
        return None
    return UiEmotionState(emotion=state.emotion.value, intensity=state.intensity)


def _to_ui_token_usage_summary(summary: TokenUsageSummary) -> UiTokenUsageSummary:
    return UiTokenUsageSummary(
        input_tokens=summary.input_tokens,
        output_tokens=summary.output_tokens,
        total_tokens=summary.total_tokens,
        response_count=summary.response_count,
    )


def _to_ui_avatar_appearance(config: AvatarConfig) -> UiAvatarAppearance:
    return UiAvatarAppearance(
        display_name=config.display_name,
        hair_color=config.hair_color,
        outfit_color=config.outfit_color,
        accent_color=config.accent_color,
    )
