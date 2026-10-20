"""音声とアバターを会話エンジンから分離するインターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from multi_llm_companion.domain import AssistantResponse, EmotionState

from .lip_sync import LipSyncState

type RecognitionCallback = Callable[[str], None]
type IntegrationErrorCallback = Callable[[str], None]
type LevelCallback = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class AvatarBone:
    """PMXボーンの階層、付与変形、軸制約、IK情報。"""

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
class AvatarRigidBody:
    """PMX物理演算に必要な剛体情報。"""

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
class AvatarPhysicsJoint:
    """PMX剛体をつなぐ6自由度ばねジョイント情報。"""

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
class AvatarVmdBoneFrame:
    """VMD内のボーン1フレーム。"""

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
class AvatarVmdMorphFrame:
    """VMD内の表情モーフ1フレーム。"""

    morph_name: str
    frame_number: int
    weight: float


@dataclass(frozen=True, slots=True)
class AvatarVmdCameraFrame:
    """VMD内のカメラ1フレーム。"""

    frame_number: int
    distance: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    interpolation: tuple[tuple[int, int, int, int], ...] = ((20, 20, 107, 107),) * 6
    view_angle: int = 30
    perspective: bool = True


@dataclass(frozen=True, slots=True)
class AvatarVmdLightFrame:
    """VMD内の照明1フレーム。"""

    frame_number: int
    color: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AvatarVmdShadowFrame:
    """VMD内のセルフ影1フレーム。"""

    frame_number: int
    mode: int
    distance: float


@dataclass(frozen=True, slots=True)
class AvatarVmdIkState:
    """VMD内のIK名と有効状態。"""

    ik_name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class AvatarVmdIkFrame:
    """VMD内のモデル表示・IK切替1フレーム。"""

    frame_number: int
    display_model: bool
    states: tuple[AvatarVmdIkState, ...] = ()


@dataclass(frozen=True, slots=True)
class AvatarVmdCompatibility:
    """VMDとPMXの対応・未検出・未対応項目。"""

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
class AvatarVmdMotion:
    """安全に読み取ったVMDモーション。"""

    name: str
    frame_count: int
    bone_frames: tuple[AvatarVmdBoneFrame, ...]
    loop: bool = True
    morph_frames: tuple[AvatarVmdMorphFrame, ...] = ()
    intensity_scale: float = 1.0
    target_model_name: str = ""
    camera_frames: tuple[AvatarVmdCameraFrame, ...] = ()
    light_frames: tuple[AvatarVmdLightFrame, ...] = ()
    shadow_frames: tuple[AvatarVmdShadowFrame, ...] = ()
    ik_frames: tuple[AvatarVmdIkFrame, ...] = ()
    compatibility: AvatarVmdCompatibility | None = None


@dataclass(frozen=True, slots=True)
class AvatarMaterialMorph:
    """PMX材質モーフ一要素の乗算・加算係数。"""

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
class AvatarTexturedTriangle:
    """テクスチャ付き表示に必要な、PMXの三角形と材質情報。"""

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
class AvatarModelPreview:
    """画面で安全に扱う、3Dモデルの軽量なメッシュ情報。"""

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
    named_material_morphs: dict[str, tuple[AvatarMaterialMorph, ...]] = field(default_factory=dict)
    bones: tuple[AvatarBone, ...] = ()
    vertex_weights: tuple[tuple[tuple[int, float], ...], ...] = ()
    rigid_bodies: tuple[AvatarRigidBody, ...] = ()
    physics_joints: tuple[AvatarPhysicsJoint, ...] = ()
    vmd_motion: AvatarVmdMotion | None = None
    textured_triangles: tuple[AvatarTexturedTriangle, ...] = ()
    dense_textured_triangles: tuple[AvatarTexturedTriangle, ...] = ()
    thin_cloth_contact_enabled: bool = False
    thin_cloth_source_body_patterns: tuple[str, ...] = ()
    thin_cloth_target_body_patterns: tuple[str, ...] = ()
    thin_cloth_maximum_correction_per_step: float = 0.008
    idle_motions: tuple[AvatarVmdMotion, ...] = ()


@dataclass(frozen=True, slots=True)
class MmdAvatarCommand:
    """将来のMMD表示へ渡す、設定済みの表情・視線・動作命令。"""

    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    emotion_morph: str = ""
    gaze: str = "forward"
    gaze_bone: str = ""
    gesture: str = "none"
    gesture_target: str = ""
    head_bone: str = ""
    upper_body_bone: str = ""
    gesture_motion_id: str = ""
    gesture_motion: AvatarVmdMotion | None = None
    gesture_intensity_scale: float = 1.0
    gesture_intensity: float = 1.0
    gaze_intensity: float = 1.0
    natural_motion_intensity: float = 1.0
    expression_preset: str = ""
    expression_morphs: tuple[tuple[str, float], ...] = ()
    expression_status_text: str = ""
    gesture_motion_type: str = ""
    gesture_body_mask: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MmdRendererStatus:
    """描画部品の能力と、画面へ出す安全な状態。"""

    renderer_id: str
    renderer_name: str
    status_text: str
    is_full_mmd: bool
    is_fallback: bool


class AudioInputDevice:
    """画面へ渡すマイク情報。"""

    def __init__(self, device_id: int, name: str) -> None:
        self.id = device_id
        self.name = name


class SpeechRecognitionPort(ABC):
    """将来のマイク音声認識が実装する入口。"""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """現在利用可能ならTrue。"""

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """マイクから音を受け取っている間はTrue。"""

    @property
    def _is_running(self) -> bool:
        """設定交換前に、停止後も残る処理の有無を確認する。"""

        return self.is_listening

    @property
    @abstractmethod
    def status_text(self) -> str:
        """画面へ表示する短い日本語の状態。"""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """設定画面へ表示する認識方式名。"""

    @property
    @abstractmethod
    def input_device(self) -> int | str | None:
        """現在選択している入力機器。"""

    @abstractmethod
    def list_input_devices(self) -> list[AudioInputDevice]:
        """利用可能なマイクを返す。"""

    @abstractmethod
    def start(
        self,
        on_recognized: RecognitionCallback,
        on_error: IntegrationErrorCallback | None = None,
        on_partial: RecognitionCallback | None = None,
        on_level: LevelCallback | None = None,
    ) -> None:
        """認識を開始し、文字列をコールバックへ渡す。"""

    @abstractmethod
    def stop(self) -> None:
        """認識を停止する。"""

    @abstractmethod
    def finish(self) -> None:
        """録音を終え、取得済み音声の確定認識を続ける。"""


class VoiceOutputPort(ABC):
    """将来のVoiSona Talk出力が実装する入口。"""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """現在利用可能ならTrue。"""

    @property
    @abstractmethod
    def status_text(self) -> str:
        """画面へ表示する短い日本語の状態。"""

    @property
    @abstractmethod
    def last_error(self) -> str | None:
        """直近の安全な日本語エラー。"""

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """音声の処理または再生中ならTrue。"""

    @property
    @abstractmethod
    def playback_status_text(self) -> str:
        """画面へ表示する短い音声再生状態。"""

    @property
    @abstractmethod
    def lip_sync_state(self) -> LipSyncState:
        """現在の推定口形。音素時刻を取得できない場合も安全に返す。"""

    @abstractmethod
    def speak(self, text: str) -> None:
        """文字列を音声として再生する。"""

    def start_stream(self, text: str) -> None:
        """文単位の先読みを開始する。未対応実装は通常発話へ戻す。"""

        self.speak(text)

    def enqueue_stream(self, text: str) -> None:
        """再生中の文の後へ、次の文を追加する。"""

        self.speak(text)

    def finish_stream(self) -> None:
        """先読み入力の終了を通知する。"""

        return None

    @abstractmethod
    def stop(self) -> None:
        """再生中の音声を停止する。"""


class AvatarControlPort(ABC):
    """将来の汎用アバター制御が実装する入口。"""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """現在利用可能ならTrue。"""

    @property
    @abstractmethod
    def controller_name(self) -> str:
        """画面へ表示するアバター制御方式の名前。"""

    @property
    @abstractmethod
    def status_text(self) -> str:
        """画面へ表示する短い3D連携状態。"""

    @property
    @abstractmethod
    def model_preview(self) -> AvatarModelPreview | None:
        """表示可能な3Dモデルの軽量情報。未読込ならNone。"""

    @property
    @abstractmethod
    def mmd_command(self) -> MmdAvatarCommand | None:
        """現在のMMD向け命令。MMD連携以外ではNone。"""

    @abstractmethod
    def apply_response(self, response: AssistantResponse) -> None:
        """感情、動作、視線をアバターへ反映する。"""

    @abstractmethod
    def apply_emotion_state(self, state: EmotionState) -> None:
        """なだらかに変化した感情状態をアバターへ反映する。"""

    @abstractmethod
    def reset(self) -> None:
        """アバターを通常状態へ戻す。"""

    @abstractmethod
    def generate_confirmation_vmd(self) -> str:
        """モデルに合わせた確認用VMDを作り、保存先を返す。"""


class MmdRendererPort(ABC):
    """会話処理から分離したMMD描画方式の入口。"""

    @property
    @abstractmethod
    def status(self) -> MmdRendererStatus:
        """描画方式と利用可能な範囲を返す。"""

    @abstractmethod
    def load_model(self, preview: AvatarModelPreview | None) -> None:
        """安全に読み込まれたモデル情報を受け取る。"""

    @abstractmethod
    def apply_command(self, command: MmdAvatarCommand | None) -> None:
        """表情・視線・動作の共通命令を受け取る。"""
