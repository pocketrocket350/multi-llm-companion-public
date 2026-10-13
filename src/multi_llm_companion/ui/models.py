"""画面とアプリ本体の間で受け渡す小さなデータ型。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.domain.models import ImageAttachmentInfo, WebSource
from multi_llm_companion.services.image_sharing import PreparedImage


@dataclass(frozen=True, slots=True)
class UiConversation:
    id: str
    title: str


@dataclass(frozen=True, slots=True)
class UiMessage:
    role: str
    content: str
    sources: tuple[WebSource, ...] = ()
    image_attachment: ImageAttachmentInfo | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BackendStatus:
    provider_name: str
    model_name: str
    connection_text: str
    speech_recognition_text: str = "無効"
    voice_output_text: str = "無効"


@dataclass(frozen=True, slots=True)
class UiAudioDevice:
    id: int
    name: str


@dataclass(frozen=True, slots=True)
class SpeechSettings:
    engine: str
    input_device: int | str | None
    devices: list[UiAudioDevice]
    silence_threshold: float = 0.015
    silence_duration: float = 1.2
    max_recording_duration: float = 30.0
    partial_interval: float = 2.5


@dataclass(frozen=True, slots=True)
class UiMemory:
    id: int
    content: str
    category: str
    status: str
    created_at: str = ""
    updated_at: str = ""
    source_conversation_id: str = ""


@dataclass(frozen=True, slots=True)
class UiEmotionState:
    emotion: str
    intensity: float


@dataclass(frozen=True, slots=True)
class UiAvatarAction:
    gesture: str
    gaze: str
    expression_intensity: float = 0.0
    gesture_intensity: float = 1.0
    gaze_intensity: float = 1.0
    natural_motion_intensity: float = 1.0
    status_text: str = ""


@dataclass(frozen=True, slots=True)
class UiAvatarAppearance:
    display_name: str
    hair_color: str
    outfit_color: str
    accent_color: str


@dataclass(frozen=True, slots=True)
class UiAvatar3DStatus:
    controller_name: str
    status_text: str
    is_available: bool
    render_quality: str = "standard"


@dataclass(frozen=True, slots=True)
class UiAvatarBone:
    """画面描画に必要なPMXボーン階層・軸制約・IK情報。"""

    name: str
    parent_index: int
    position: tuple[float, float, float]
    ik_target_index: int = -1
    ik_iterations: int = 0
    ik_limit_radians: float = 0.0
    ik_links: tuple[int, ...] = ()
    deform_layer: int = 0
    inherit_parent_index: int = -1
    inherit_weight: float = 0.0
    inherit_rotation: bool = False
    inherit_translation: bool = False
    fixed_axis: tuple[float, float, float] | None = None
    local_axis_x: tuple[float, float, float] | None = None
    local_axis_z: tuple[float, float, float] | None = None
    after_physics: bool = False
    allows_rotation: bool = True
    allows_translation: bool = True
    ik_link_limits: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]] | None, ...
    ] = ()


@dataclass(frozen=True, slots=True)
class UiAvatarRigidBody:
    """画面で物理計算に使うPMX剛体情報。"""

    bone_index: int
    position: tuple[float, float, float]
    mass: float
    linear_damping: float
    mode: int
    name: str = ""
    collision_group: int = 0
    non_collision_mask: int = 0
    shape: int = 0
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_damping: float = 0.0
    restitution: float = 0.0
    friction: float = 0.0


@dataclass(frozen=True, slots=True)
class UiAvatarPhysicsJoint:
    """画面で物理計算に使う6自由度ばねジョイント情報。"""

    rigid_body_a: int
    rigid_body_b: int
    position: tuple[float, float, float]
    spring_translation: tuple[float, float, float]
    name: str = ""
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spring_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class UiVmdBoneFrame:
    """画面用のVMDボーン1フレーム。"""

    bone_name: str
    frame_number: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    interpolation: tuple[tuple[int, int, int, int], ...] = (
        (20, 20, 107, 107),
        (20, 20, 107, 107),
        (20, 20, 107, 107),
        (20, 20, 107, 107),
    )


@dataclass(frozen=True, slots=True)
class UiVmdMorphFrame:
    """画面用のVMD表情モーフ1フレーム。"""

    morph_name: str
    frame_number: int
    weight: float


@dataclass(frozen=True, slots=True)
class UiVmdCameraFrame:
    """画面用のVMDカメラ1フレーム。"""

    frame_number: int
    distance: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    interpolation: tuple[tuple[int, int, int, int], ...] = ((20, 20, 107, 107),) * 6
    view_angle: int = 30
    perspective: bool = True


@dataclass(frozen=True, slots=True)
class UiVmdLightFrame:
    """画面用のVMD照明1フレーム。"""

    frame_number: int
    color: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class UiVmdShadowFrame:
    """画面用のVMDセルフ影1フレーム。"""

    frame_number: int
    mode: int
    distance: float


@dataclass(frozen=True, slots=True)
class UiVmdIkState:
    """画面用のVMD IK状態。"""

    ik_name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class UiVmdIkFrame:
    """画面用のVMDモデル表示・IK切替1フレーム。"""

    frame_number: int
    display_model: bool
    states: tuple[UiVmdIkState, ...] = ()


@dataclass(frozen=True, slots=True)
class UiVmdCompatibility:
    """画面へ渡すVMDとPMXの互換性診断。"""

    supported_bones: tuple[str, ...] = ()
    missing_bones: tuple[str, ...] = ()
    supported_morphs: tuple[str, ...] = ()
    missing_morphs: tuple[str, ...] = ()
    supported_iks: tuple[str, ...] = ()
    missing_iks: tuple[str, ...] = ()
    translation_height_ratio: float = 0.0
    target_model_matches: bool | None = None
    unsupported_sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiVmdMotion:
    """画面再生用に整えたVMDモーション。"""

    name: str
    frame_count: int
    bone_frames: tuple[UiVmdBoneFrame, ...]
    loop: bool = True
    morph_frames: tuple[UiVmdMorphFrame, ...] = ()
    intensity_scale: float = 1.0
    target_model_name: str = ""
    camera_frames: tuple[UiVmdCameraFrame, ...] = ()
    light_frames: tuple[UiVmdLightFrame, ...] = ()
    shadow_frames: tuple[UiVmdShadowFrame, ...] = ()
    ik_frames: tuple[UiVmdIkFrame, ...] = ()
    compatibility: UiVmdCompatibility | None = None


@dataclass(frozen=True, slots=True)
class UiMotionLibraryItem:
    """モーション管理画面へ表示する一件分の状態。"""

    motion_id: str
    motion_type: str
    action: str
    moods: tuple[str, ...]
    energy: str
    enabled: bool
    available: bool
    status_text: str
    audio_required: bool = False
    frame_count: int = 0
    source_name: str = ""


@dataclass(frozen=True, slots=True)
class UiExpressionPreset:
    """表情管理画面へ渡す、PMX実名へ解決済みのプリセット。"""

    name: str
    morph_weights: tuple[tuple[str, float], ...]
    emotions: tuple[str, ...] = ()
    unresolved_roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiAvatarMaterialMorph:
    """画面描画へ渡すPMX材質モーフ一要素。"""

    material_index: int
    operation: int
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float]
    shininess: float
    ambient: tuple[float, float, float]
    edge_color: tuple[float, float, float, float]
    edge_size: float
    texture_factor: tuple[float, float, float, float]
    sphere_factor: tuple[float, float, float, float]
    toon_factor: tuple[float, float, float, float]
    weight_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class UiAvatarTexturedTriangle:
    """画面描画に必要なテクスチャ付きPMX三角形。"""

    indexes: tuple[int, int, int]
    uvs: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    texture_path: str
    diffuse: tuple[float, float, float, float]
    additional_uvs: tuple[
        tuple[tuple[float, float, float, float], ...],
        tuple[tuple[float, float, float, float], ...],
        tuple[tuple[float, float, float, float], ...],
    ] = ((), (), ())
    material_index: int = 0
    is_transparent: bool = False
    normals: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    sphere_texture_path: str = ""
    sphere_mode: int = 0
    toon_texture_path: str = ""
    toon_shared_index: int = -1
    is_double_sided: bool = True
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shininess: float = 0.0
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    draw_outline: bool = False
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 0.0
    edge_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class UiAvatarModelPreview:
    model_name: str
    vertex_count: int
    face_count: int
    points: tuple[tuple[float, float, float], ...]
    normals: tuple[tuple[float, float, float], ...] = ()
    triangles: tuple[tuple[int, int, int], ...] = ()
    mouth_morphs: dict[str, tuple[tuple[int, float, float, float], ...]] = field(
        default_factory=dict
    )
    vertex_morph_names: tuple[str, ...] = ()
    emotion_morphs: dict[str, tuple[tuple[int, float, float, float], ...]] = field(
        default_factory=dict
    )
    named_vertex_morphs: dict[str, tuple[tuple[int, float, float, float], ...]] = field(
        default_factory=dict
    )
    uvs: tuple[tuple[float, float], ...] = ()
    additional_uvs: tuple[tuple[tuple[float, float, float, float], ...], ...] = ()
    morph_names: tuple[str, ...] = ()
    named_uv_morphs: dict[str, tuple[tuple[int, int, float, float, float, float], ...]] = field(
        default_factory=dict
    )
    named_material_morphs: dict[str, tuple[UiAvatarMaterialMorph, ...]] = field(
        default_factory=dict
    )
    bones: tuple[UiAvatarBone, ...] = ()
    vertex_weights: tuple[tuple[tuple[int, float], ...], ...] = ()
    rigid_bodies: tuple[UiAvatarRigidBody, ...] = ()
    physics_joints: tuple[UiAvatarPhysicsJoint, ...] = ()
    vmd_motion: UiVmdMotion | None = None
    textured_triangles: tuple[UiAvatarTexturedTriangle, ...] = ()
    dense_textured_triangles: tuple[UiAvatarTexturedTriangle, ...] = ()
    thin_cloth_contact_enabled: bool = False
    thin_cloth_source_body_patterns: tuple[str, ...] = ()
    thin_cloth_target_body_patterns: tuple[str, ...] = ()
    thin_cloth_maximum_correction_per_step: float = 0.008
    idle_motions: tuple[UiVmdMotion, ...] = ()


@dataclass(frozen=True, slots=True)
class UiMmdAvatarCommand:
    emotion: str
    emotion_intensity: float
    emotion_morph: str
    gaze: str
    gaze_bone: str
    gesture: str
    gesture_target: str
    head_bone: str = ""
    upper_body_bone: str = ""
    gesture_motion_id: str = ""
    gesture_motion: UiVmdMotion | None = None
    gesture_intensity_scale: float = 1.0
    gesture_intensity: float = 1.0
    gaze_intensity: float = 1.0
    natural_motion_intensity: float | None = None
    expression_preset: str = ""
    expression_morphs: tuple[tuple[str, float], ...] = ()
    expression_status_text: str = ""
    gesture_motion_type: str = ""
    gesture_body_mask: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiLipSyncState:
    """画面用に整えた、現在の推定口形。"""

    is_active: bool
    mouth_shape: str = "neutral"
    is_estimated: bool = True
    is_timeline_active: bool = False


@dataclass(frozen=True, slots=True)
class UiTokenUsageSummary:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_count: int


@dataclass(frozen=True, slots=True)
class MemorySettings:
    enabled: bool
    auto_approve_candidates: bool = True


@dataclass(frozen=True, slots=True)
class SendOutcome:
    speech: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    gesture: str = ""
    gaze: str = ""
    voice_error: str | None = None
    memory_candidates: list[UiMemory] = field(default_factory=list)
    used_memories: list[UiMemory] = field(default_factory=list)
    emotion_state: UiEmotionState | None = None
    expression_intensity: float = 0.0
    gesture_intensity: float = 0.0
    gaze_intensity: float = 0.0
    natural_motion_intensity: float = 0.0
    expression_status_text: str = ""
    sources: tuple[WebSource, ...] = ()
    image_attachment: ImageAttachmentInfo | None = None


class ChatBackend(Protocol):
    """画面が利用する機能。実装の細部を画面から分離する。"""

    def status(self) -> BackendStatus: ...

    def web_search_available(self) -> bool: ...

    def web_search_enabled(self) -> bool: ...

    def set_web_search_enabled(self, enabled: bool) -> None: ...

    def approve_web_search(self, text: str, query: str, detailed: bool, region: str) -> None: ...

    def image_sharing_available(self) -> bool: ...

    def image_sharing_enabled(self) -> bool: ...

    def set_image_sharing_enabled(self, enabled: bool) -> None: ...

    def provider_names(self) -> list[str]: ...

    def current_provider_name(self) -> str: ...

    def persona_settings(self) -> PersonaConfig: ...

    def change_persona_settings(self, persona: PersonaConfig) -> None: ...

    def list_conversations(self) -> list[UiConversation]: ...

    def load_messages(self, conversation_id: str) -> list[UiMessage]: ...

    def create_conversation(self, title: str) -> str: ...

    def delete_conversation(self, conversation_id: str) -> None: ...

    def generate_proactive_turn(self, conversation_id: str) -> SendOutcome: ...

    def cancel_proactive_turn(self) -> None: ...

    def memory_settings(self) -> MemorySettings: ...

    def conversation_context_status(self) -> str: ...

    def set_memory_enabled(self, enabled: bool) -> None: ...

    def set_memory_auto_approve_candidates(self, enabled: bool) -> None: ...

    def list_memories(self, status: str | None = None) -> list[UiMemory]: ...

    def approve_memory(self, memory_id: int, content: str, category: str) -> UiMemory: ...

    def update_memory(self, memory_id: int, content: str, category: str) -> UiMemory: ...

    def delete_memory(self, memory_id: int) -> None: ...

    def delete_all_memories(self) -> int: ...

    def emotion_state(self, conversation_id: str) -> UiEmotionState | None: ...

    def avatar_action(self, conversation_id: str) -> UiAvatarAction: ...

    def avatar_appearance(self) -> UiAvatarAppearance: ...

    def avatar_3d_status(self) -> UiAvatar3DStatus: ...

    def avatar_model_preview(self) -> UiAvatarModelPreview | None: ...

    def mmd_avatar_command(self) -> UiMmdAvatarCommand | None: ...

    def save_expression_preset(
        self,
        preset_name: str,
        emotion: str,
        morph_weights: dict[str, float],
    ) -> None: ...

    def list_expression_presets(self) -> list[UiExpressionPreset]: ...

    def delete_expression_preset(self, preset_name: str) -> None: ...

    def list_motion_library(self) -> list[UiMotionLibraryItem]: ...

    def load_motion_preview(self, motion_id: str) -> UiVmdMotion: ...

    def set_motion_enabled(self, motion_id: str, enabled: bool) -> None: ...

    def generate_confirmation_vmd(self) -> str: ...

    def token_usage_summary(
        self,
        conversation_id: str | None = None,
    ) -> UiTokenUsageSummary: ...

    def change_provider(self, provider_name: str) -> BackendStatus: ...

    def start_speech_recognition(
        self,
        on_recognized: Callable[[str], None],
        on_error: Callable[[str], None],
        on_partial: Callable[[str], None],
        on_level: Callable[[int], None],
    ) -> None: ...

    def stop_speech_recognition(self) -> None: ...

    def finish_speech_recognition(self) -> None: ...

    def is_voice_speaking(self) -> bool: ...

    def voice_lip_sync_state(self) -> UiLipSyncState: ...

    def stop_voice_output(self) -> None: ...

    def start_voice_output(self, text: str) -> str | None: ...

    def start_proactive_voice_output(self, text: str) -> str | None: ...

    def start_stream_voice_output(self, text: str) -> str | None: ...

    def enqueue_stream_voice_output(self, text: str) -> str | None: ...

    def finish_stream_voice_output(self) -> None: ...

    def is_speech_recognition_available(self) -> bool: ...

    def is_listening(self) -> bool: ...

    def speech_settings(self) -> SpeechSettings: ...

    def change_speech_settings(
        self,
        engine: str,
        input_device: int | str | None,
        silence_threshold: float,
        silence_duration: float,
        max_recording_duration: float,
        partial_interval: float,
    ) -> BackendStatus: ...

    def stream_message(
        self,
        conversation_id: str,
        user_text: str,
        on_chunk: Callable[[str], None],
        *,
        image: PreparedImage | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> SendOutcome: ...


class BackendFactory(Protocol):
    def __call__(self, provider_name: str) -> ChatBackend: ...
