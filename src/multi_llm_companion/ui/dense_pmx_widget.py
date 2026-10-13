"""PMXの全テクスチャ面をGPUで描くための軽量なOpenGLビュー。"""

from __future__ import annotations

import gc
import math
import struct
import time
from _thread import LockType
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMatrix4x4,
    QOpenGLContext,
    QSurfaceFormat,
    QVector2D,
    QVector3D,
    QVector4D,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from shiboken6 import VoidPtr

from multi_llm_companion.integrations.lip_sync_diagnostic import write_lip_sync_trace
from multi_llm_companion.ui.bullet_pmx_physics import (
    ThinClothSourceSurface,
)
from multi_llm_companion.ui.cloth_process import ProcessClothSimulation
from multi_llm_companion.ui.companion_background import (
    _cover_rect,
    _local_background,
    _standing_background_rect,
)
from multi_llm_companion.ui.independent_cloth import ClothNonFiniteError
from multi_llm_companion.ui.mmd_motion_mixer import MixedMotion, MmdMotionMixer, MotionLayer
from multi_llm_companion.ui.models import (
    UiAvatarModelPreview,
    UiMmdAvatarCommand,
    UiVmdCameraFrame,
    UiVmdLightFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.natural_motion import NaturalMotionFrame
from multi_llm_companion.ui.pmx_physics import PhysicsPose
from multi_llm_companion.ui.pmx_pose import (
    BoneTransform,
    PreparedIkSolver,
    apply_bone_pose,
    bone_transforms,
    matrix_vector,
    prepare_ik_solver,
    solve_ik_rotations,
)
from multi_llm_companion.ui.render_quality import (
    normalized_render_quality,
    surface_format,
)
from multi_llm_companion.ui.umt_native_physics import UmtNativePmxPhysicsSimulator
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer, _interpolate_vector_maps

# Python起動時の固定分は、別のPMXビューによる固定とは区別する。
_INITIAL_GC_FREEZE_COUNT = gc.get_freeze_count()
_VERTEX_FLOAT_COUNT = 11
_VERTEX_STRIDE = _VERTEX_FLOAT_COUNT * 4
_MMD_EDGE_REFERENCE_VIEW_SCALE = 2.2
_MMD_EDGE_PROJECTION_SCALE = 0.6
# 登録動作ごとの復帰候補。未調整の動作は従来の時間を保つ。
_ARM_RETURN_SECONDS = {
    "ot0510_やれやれ": 1.2,
    "ot0510_バキュン": 1.2,
    "ot0510_リズム_大": 1.2,
    "ot0510_リズム_小": 1.2,
    "ot0510_喜びジャンプ": 1.4,
    "ot0510_指令": 1.2,
    "ot0510_胸に秘めたる": 1.2,
    "ot0510_腕ポンポン": 1.2,
    "ot0510_興味深い": 1.2,
    "ot0510_驚き_背中見せ": 1.4,
    "ot0510_ひらめく": 1.2,
    "ot0510_ドヤ": 1.2,
    "ot0510_ピース": 1.2,
    "ot0510_応援_ループ": 1.4,
    "ot0510_笑い_口あて": 1.2,
    "ot0510_考える_ループ": 1.2,
    "ot0510_苦笑い_ループ": 1.2,
    "ot0510_飛行機ごっこ_ループ": 1.4,
    "ot0510_驚き_腕バタバタ_ループ": 1.6,
    "ot0510_待機_元気_": 1.2,
    "ot0510_待機_右手腰あて_": 1.2,
    "ot0510_待機_左手腰あて_": 1.2,
    "ot0510_待機_猫背_": 1.2,
}


@dataclass(slots=True)
class _GpuPose:
    palette: np.ndarray
    ground_offset: float = 0.0


@dataclass(slots=True)
class _MeshGroup:
    material_index: int
    texture_path: str
    diffuse: tuple[float, float, float, float]
    is_transparent: bool
    sphere_texture_path: str
    sphere_mode: int
    toon_texture_path: str
    is_double_sided: bool
    specular: tuple[float, float, float]
    shininess: float
    ambient: tuple[float, float, float]
    vertices: bytes
    count: int
    draw_outline: bool = False
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 0.0
    toon_shared_index: int = -1
    texture_multiply: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture_add: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sphere_multiply: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    sphere_add: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    toon_multiply: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    toon_add: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    buffer: QOpenGLBuffer | None = None
    texture: QOpenGLTexture | None = None
    sphere_texture: QOpenGLTexture | None = None
    toon_texture: QOpenGLTexture | None = None
    skin_vertices: bytes = b""
    skin_buffer: QOpenGLBuffer | None = None
    gpu_pose: _GpuPose | None = None


@dataclass(frozen=True, slots=True)
class _PreparedMeshGroup:
    """姿勢ごとに変わらない材質・法線・UVと頂点参照。"""

    material_index: int
    texture_path: str
    diffuse: tuple[float, float, float, float]
    is_transparent: bool
    sphere_texture_path: str
    sphere_mode: int
    toon_texture_path: str
    is_double_sided: bool
    specular: tuple[float, float, float]
    shininess: float
    ambient: tuple[float, float, float]
    vertex_indexes: np.ndarray
    vertex_template: np.ndarray
    draw_outline: bool
    edge_color: tuple[float, float, float, float]
    edge_size: float
    toon_shared_index: int
    skin_vertices: bytes = b""
    rest_vertices: bytes = b""


@dataclass(frozen=True, slots=True)
class _MaterialState:
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float]
    shininess: float
    ambient: tuple[float, float, float]
    edge_color: tuple[float, float, float, float]
    edge_size: float
    texture_multiply: tuple[float, float, float, float]
    texture_add: tuple[float, float, float, float]
    sphere_multiply: tuple[float, float, float, float]
    sphere_add: tuple[float, float, float, float]
    toon_multiply: tuple[float, float, float, float]
    toon_add: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _PreparedPoseData:
    """NumPyで一括ボーン変形するため、PMX読込時に一度だけ作る配列。"""

    points: np.ndarray
    bone_indexes: np.ndarray
    bone_weights: np.ndarray
    bone_rest_positions: np.ndarray
    bone_local_offsets: tuple[np.ndarray, ...]
    normalized_bone_weights: np.ndarray
    influence_vertex_indexes: tuple[np.ndarray, ...]
    influence_bone_indexes: tuple[np.ndarray, ...]
    influence_weights: tuple[np.ndarray, ...]
    required_bone_indexes: frozenset[int]
    weighted_vertex_mask: np.ndarray
    ground_vertex_indexes: np.ndarray
    ground_height: float | None
    thin_cloth_target_vertex_indexes: np.ndarray
    groups: tuple[_PreparedMeshGroup, ...]
    cpu_vertex_mask: np.ndarray


@dataclass(slots=True)
class _ClothState:
    simulation: ProcessClothSimulation | None = None
    initialized: bool = False
    closed: bool = False
    error: str = ""
    lock: LockType = field(default_factory=Lock)

    def close(self) -> None:
        # モデル切替・終了と計算中の世界破棄を競合させない。
        with self.lock:
            self.closed = True
            if self.simulation is not None:
                self.simulation.close()
                self.simulation = None

    def fail(self, exc: Exception) -> None:
        # 呼出し側がロックを保持する。失敗した世界を自動で再生成しない。
        self.error = str(exc)
        if self.simulation is not None:
            self.simulation.close()

    def restart(self) -> None:
        """動作境界で、現在の姿勢を初期値にして布計算を再開する。"""

        with self.lock:
            if self.closed:
                return
            if self.simulation is not None:
                self.simulation.close()
            self.simulation = None
            self.initialized = False
            self.error = ""


@dataclass(frozen=True, slots=True)
class _PoseBuildRequest:
    generation: int
    motion_revision: int
    preview: UiAvatarModelPreview | None
    rotations: dict[int, tuple[float, float, float]]
    ik_offsets: dict[int, tuple[float, float, float]]
    translations: dict[int, tuple[float, float, float]]
    morph_weights: dict[str, float]
    prepared_pose: _PreparedPoseData | None
    ik_already_solved: bool
    disabled_ik_indexes: frozenset[int]
    thin_cloth_source_surfaces: tuple[ThinClothSourceSurface, ...]
    signature: tuple[object, ...]
    physics_translations: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    cloth_seconds: float = 0.0
    cloth_state: _ClothState | None = None
    gpu_skinning: bool = False
    pose_transforms: dict[int, BoneTransform] | None = None


def _physics_anchor_motion(
    preview: UiAvatarModelPreview | None,
    vmd_rotations: dict[int, tuple[float, float, float]],
    vmd_translations: dict[int, tuple[float, float, float]],
    gesture_rotations: dict[int, tuple[float, float, float]],
    gesture_translations: dict[int, tuple[float, float, float]],
    displayed_motion: MixedMotion,
    *,
    prefer_displayed: bool = False,
) -> tuple[
    dict[int, tuple[float, float, float]],
    dict[int, tuple[float, float, float]],
]:
    """画面の姿勢を衝突用剛体へ渡し、物理結果の自己入力だけを除く。"""

    rotations = {**vmd_rotations, **gesture_rotations}
    translations = {**vmd_translations, **gesture_translations}
    if preview is None:
        return rotations, translations
    dynamic_bones = {
        body.bone_index
        for body in preview.rigid_bodies
        if body.mode != 0 and 0 <= body.bone_index < len(preview.bones)
    }
    displayed_rotations = {
        index: value
        for index, value in displayed_motion.rotations.items()
        if index not in dynamic_bones
    }
    displayed_translations = {
        index: value
        for index, value in displayed_motion.translations.items()
        if index not in dynamic_bones
    }
    if prefer_displayed:
        rotations.update(displayed_rotations)
        translations.update(displayed_translations)
    else:
        rotations.update(
            {index: value for index, value in displayed_rotations.items() if index not in rotations}
        )
        translations.update(
            {
                index: value
                for index, value in displayed_translations.items()
                if index not in translations
            }
        )
    return rotations, translations


class DensePmxWidget(QOpenGLWidget):
    """全三角形をGPUへ渡し、隙間のない基本テクスチャ表示を行う。"""

    renderer_ready = Signal(bool)
    quality_status_changed = Signal()
    conversation_pose_ready = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        render_quality: str = "standard",
    ) -> None:
        super().__init__(parent)
        self._render_quality = normalized_render_quality(render_quality)
        self.setFormat(_surface_format(self._render_quality))
        self._groups: list[_MeshGroup] = []
        self._preview: UiAvatarModelPreview | None = None
        self._rest_expression_weights: dict[str, float] = {}
        self._prepared_pose: _PreparedPoseData | None = None
        self._bone_indexes_by_name: dict[str, int] = {}
        self._influential_bone_indexes: frozenset[int] = frozenset()
        self._physics_bone_indexes: frozenset[int] = frozenset()
        self._natural_gaze_indexes: tuple[int, ...] = ()
        self._natural_rest_pose: dict[int, tuple[float, float, float]] = {}
        self._prepared_ik: PreparedIkSolver | None = None
        self._mouth_shape = "neutral"
        self._mouth_weights: dict[str, float] = {}
        self._program: QOpenGLShaderProgram | None = None
        self._resolve_program: QOpenGLShaderProgram | None = None
        self._background_program: QOpenGLShaderProgram | None = None
        self._background_texture: QOpenGLTexture | None = None
        self._background_image = _local_background()
        self._skin_texture: QOpenGLTexture | None = None
        self._gpu_skinning_enabled = True
        self._texture_cache: dict[str, QOpenGLTexture] = {}
        self._supersample_buffer: QOpenGLFramebufferObject | None = None
        self._supersample_factor = 1.0
        self._supersample_disabled = False
        self._last_reported_quality_status = ""
        self._initialized = False
        self._available = False
        self._error_text = ""
        self._yaw = math.pi
        self._pitch = 0.0
        self._zoom = 1.0
        self._model_scale = 1.0
        self._model_center = (0.0, 0.0, 0.0)
        self._bust_center = (0.0, 0.0, 0.0)
        self._view_mode = "full"
        self._motion_enabled = True
        self._idle_phase = 0.0
        self._heavy_phase = 0.0
        self._cloth_state = _ClothState()
        self._speaking = False
        self._command: UiMmdAvatarCommand | None = None
        self._gesture_body_indexes: frozenset[int] | None = None
        self._conversation_motion_active = False
        self._next_mix_delta_seconds = 0.12
        self._blend_transition_steps = 0
        self._motion_mixer = MmdMotionMixer()
        self._pose_signature: tuple[object, ...] | None = None
        self._physics = UmtNativePmxPhysicsSimulator()
        self._physics_offsets: dict[int, tuple[float, float, float]] = {}
        self._physics_rotations: dict[int, tuple[float, float, float]] = {}
        self._physics_ik_rotations: dict[int, tuple[float, float, float]] = {}
        self._physics_ik_ready = False
        self._vmd_player = VmdMotionPlayer()
        self._default_idle_motion: UiVmdMotion | None = None
        self._idle_motion: UiVmdMotion | None = None
        self._idle_elapsed = 0.0
        self._idle_next_index = 0
        self._idle_variant_active = False
        self._idle_transition_from: MixedMotion | None = None
        self._idle_transition_preserves_displayed_pose = False
        self._idle_transition_elapsed = 0.0
        self._idle_return_open_rotations: dict[int, tuple[float, float, float]] = {}
        self._idle_return_forearm_split = 0.45
        self._idle_return_arm_seconds = 2.2
        self._idle_return_distance_timing = False
        self._idle_return_tail: (
            tuple[
                dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]], float
            ]
            | None
        ) = None
        self._tail_return_motion: UiVmdMotion | None = None
        self._tail_return_rotations: dict[int, tuple[float, float, float]] = {}
        self._tail_return_target: dict[int, tuple[float, float, float]] = {}
        self._tail_source_mixer: MmdMotionMixer | None = None
        self._tail_return_elapsed = 0.0
        self._agreement_return_from: MixedMotion | None = None
        self._agreement_return_elapsed = 0.0
        self._vmd_rotations: dict[int, tuple[float, float, float]] = {}
        self._vmd_translations: dict[int, tuple[float, float, float]] = {}
        self._vmd_morph_weights: dict[str, float] = {}
        self._gesture_vmd_player = VmdMotionPlayer()
        self._gesture_vmd_rotations: dict[int, tuple[float, float, float]] = {}
        self._gesture_vmd_translations: dict[int, tuple[float, float, float]] = {}
        self._gesture_vmd_morph_weights: dict[str, float] = {}
        self._preview_motion: UiVmdMotion | None = None
        self._preview_motion_active = False
        self._preview_vmd_player = VmdMotionPlayer()
        self._preview_vmd_rotations: dict[int, tuple[float, float, float]] = {}
        self._preview_vmd_translations: dict[int, tuple[float, float, float]] = {}
        self._preview_vmd_morph_weights: dict[str, float] = {}
        self._mouth_occurrences: dict[int, dict[int, tuple[int, ...]]] = {}
        self._blink_occurrences: dict[int, dict[int, tuple[int, ...]]] = {}
        self._blink_morph_name = ""
        self._blink_weight = 0.0
        self._natural_motion = NaturalMotionFrame()
        self._natural_centers = (0.0, 0.35, 0.62, 0.82)
        self._model_height = 1.0
        self._neutral_group_vertices: dict[int, bytes] = {}
        self._mouth_pose_rotations: dict[int, tuple[float, float, float]] = {}
        self._mouth_pose_ik_offsets: dict[int, tuple[float, float, float]] = {}
        self._mouth_pose_translations: dict[int, tuple[float, float, float]] = {}
        self._mouth_pose_disabled_ik_indexes: frozenset[int] = frozenset()
        self._mouth_pose_ik_already_solved = False
        self._last_heavy_update_ms = 0.0
        self._heavy_update_count = 0
        self._pose_generation = 0
        self._pose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pmx-pose")
        self._pose_future: Future[tuple[_PoseBuildRequest, list[_MeshGroup], float]] | None = None
        self._pending_pose_request: _PoseBuildRequest | None = None
        self._pose_poll_timer = QTimer(self)
        self._pose_poll_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._pose_poll_timer.setInterval(2)
        self._pose_poll_timer.timeout.connect(self._poll_pose_build)
        self._executor_closed = False
        self._gc_freeze_owned = False
        self.setAutoFillBackground(False)

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def error_text(self) -> str:
        return self._error_text

    @property
    def vmd_status_text(self) -> str:
        text = self._vmd_player.status_text
        motion = self._idle_motion
        if motion is not None and (motion.camera_frames or motion.light_frames):
            presentation = (
                "VMDカメラ・照明: 明示再生中"
                if self._view_mode == "vmd"
                else "VMDカメラ・照明: 待機（通常表示を維持）"
            )
            return f"{text} / {presentation}"
        return text

    @property
    def quality_status_text(self) -> str:
        labels = {"low": "低", "standard": "標準", "high": "高"}
        samples = (
            self.context().format().samples() if self._initialized else self.format().samples()
        )
        if samples > 1:
            smoothing = f"MSAA {samples}x"
        elif self._supersample_factor > 1.0:
            smoothing = f"SSAA {self._supersample_factor:.1f}x"
        elif self._render_quality == "low":
            smoothing = "平滑化なし"
        else:
            smoothing = "平滑化を使用できません"
        return f"描画品質: {labels[self._render_quality]}（{smoothing}）"

    @property
    def physics_status_text(self) -> str:
        """現在使っているPMX物理方式を画面へ返す。"""

        status = self._physics.backend_status_text
        if self._cloth_state.error:
            status += f" / 布計算: 停止（{self._cloth_state.error}）"
        return status

    @property
    def last_heavy_update_ms(self) -> float:
        """直近のボーン・物理・全メッシュ更新時間。"""

        return self._last_heavy_update_ms

    @property
    def heavy_update_count(self) -> int:
        return self._heavy_update_count

    def set_preview(self, preview: UiAvatarModelPreview | None) -> None:
        self._clear_tail_return()
        self._idle_return_tail = None
        if self._cloth_state.error:
            self._error_text = ""
        self._cloth_state.close()
        self._cloth_state = _ClothState()
        self._release_gc_freeze()
        self._pose_generation += 1
        self._pending_pose_request = None
        if self._pose_future is not None:
            self._pose_future.cancel()
            self._pose_future = None
        self._pose_poll_timer.stop()
        self._preview = preview
        self._prepared_pose = _prepare_pose_data(preview)
        self._bone_indexes_by_name = (
            {bone.name: index for index, bone in enumerate(preview.bones)} if preview else {}
        )
        self._gesture_body_indexes = _body_mask_bone_indexes(
            preview,
            self._command.gesture_body_mask if self._command is not None else (),
        )
        self._influential_bone_indexes = _influential_bone_indexes(preview)
        self._physics_bone_indexes = (
            frozenset(
                body.bone_index
                for body in preview.rigid_bodies
                if 0 <= body.bone_index < len(preview.bones)
            )
            if preview is not None
            else frozenset()
        )
        self._natural_gaze_indexes = (
            _natural_gaze_bone_indexes(preview, self._influential_bone_indexes) if preview else ()
        )
        self._natural_rest_pose = _natural_rest_rotations(preview) if preview else {}
        self._prepared_ik = prepare_ik_solver(preview) if preview else None
        self._conversation_motion_active = False
        self._blend_transition_steps = 0
        self._motion_mixer = MmdMotionMixer()
        # 最初のVMD姿勢が決まってから物理世界を作り、T字姿勢からの衝撃を避ける。
        self._physics.reset(None)
        self._physics_offsets = {}
        self._physics_rotations = {}
        self._physics_ik_rotations = {}
        self._physics_ik_ready = False
        self._default_idle_motion = preview.vmd_motion if preview is not None else None
        self._idle_motion = self._default_idle_motion
        self._idle_elapsed = 0.0
        self._idle_next_index = 0
        self._idle_variant_active = False
        self._idle_transition_from = None
        self._idle_transition_preserves_displayed_pose = False
        self._idle_transition_elapsed = 0.0
        self._idle_return_open_rotations = {}
        self._idle_return_forearm_split = 0.45
        self._idle_return_arm_seconds = 2.2
        self._idle_return_distance_timing = False
        self._agreement_return_from = None
        self._agreement_return_elapsed = 0.0
        self._vmd_player.reset(self._default_idle_motion)
        (
            self._vmd_rotations,
            self._vmd_translations,
            self._vmd_morph_weights,
        ) = self._vmd_player.advance(preview, 0.0, True)
        self._gesture_vmd_player.reset(None)
        self._gesture_vmd_rotations = {}
        self._gesture_vmd_translations = {}
        self._gesture_vmd_morph_weights = {}
        self._clear_motion_preview()
        self._pose_signature = None
        self._mouth_occurrences = _build_mouth_occurrences(preview)
        self._blink_morph_name = _blink_morph_name(preview)
        self._blink_occurrences = _build_morph_occurrences(
            preview,
            preview.named_vertex_morphs.get(self._blink_morph_name, ()) if preview else (),
        )
        self._blink_weight = 0.0
        self._natural_motion = NaturalMotionFrame()
        self._natural_centers = _natural_motion_centers(preview)
        self._model_height = _model_height(preview)
        self._neutral_group_vertices = {}
        if self._initialized:
            self.makeCurrent()
        self._release_groups()
        self._groups = self._build_pose_groups(*self._pose_settings())
        self._model_scale = _model_scale(preview)
        self._model_center = _model_center(preview)
        self._bust_center = _bust_center(preview)
        if self._initialized:
            self._upload_groups()
            self.doneCurrent()
        self.update()

    def set_mouth_shape(self, mouth_shape: str) -> None:
        """口形に対応するPMX頂点モーフを、GPUの頂点列へ反映する。"""

        normalized = mouth_shape if mouth_shape in {"a", "i", "u", "e", "o"} else "neutral"
        self._set_mouth_weights(normalized, {} if normalized == "neutral" else {normalized: 1.0})

    def _set_mouth_weights(self, mouth_shape: str, weights: dict[str, float]) -> None:
        """補間済みの口だけを更新し、表情とまばたきを保持する。"""

        if mouth_shape == self._mouth_shape and weights == self._mouth_weights:
            return
        started_at = time.perf_counter_ns()
        self._mouth_shape = mouth_shape
        self._mouth_weights = dict(weights)
        used_fast_path = self._refresh_mouth_mesh()
        if not used_fast_path:
            self._refresh_pose_mesh(force=True)
        elif self._pose_signature is not None:
            signature = list(self._pose_signature)
            signature[4] = self._mouth_shape
            self._pose_signature = tuple(signature)
        write_lip_sync_trace(
            "renderer",
            "mouth_mesh_updated",
            mouth=mouth_shape,
            fast_path=used_fast_path,
            affected_materials=len(self._mouth_occurrences),
            duration_ms=round((time.perf_counter_ns() - started_at) / 1_000_000, 3),
        )
        self.update()

    def _build_pose_groups(
        self,
        rotations: dict[int, tuple[float, float, float]],
        ik_offsets: dict[int, tuple[float, float, float]],
        translations: dict[int, tuple[float, float, float]],
        morph_weights: dict[str, float],
    ) -> list[_MeshGroup]:
        """全体姿勢を作り、口だけを後から差し替えられる基準頂点を保持する。"""

        disabled_ik_indexes = self._disabled_ik_indexes()
        groups = _build_groups(
            self._preview,
            "neutral",
            rotations,
            ik_offsets,
            translations,
            morph_weights,
            self._prepared_pose,
            disabled_ik_indexes=disabled_ik_indexes,
            physics_translations=self._physics_offsets,
        )
        self._neutral_group_vertices = {group.material_index: group.vertices for group in groups}
        self._mouth_pose_rotations = dict(rotations)
        self._mouth_pose_ik_offsets = dict(ik_offsets)
        self._mouth_pose_translations = dict(translations)
        self._mouth_pose_disabled_ik_indexes = disabled_ik_indexes
        self._mouth_pose_ik_already_solved = False
        _apply_mouth_shape_to_groups(
            groups,
            self._preview,
            self._mouth_shape,
            self._mouth_occurrences,
            rotations,
            ik_offsets,
            translations,
            disabled_ik_indexes=self._mouth_pose_disabled_ik_indexes,
            weights=self._mouth_weights,
        )
        _apply_vertex_morph_to_groups(
            groups,
            self._preview,
            self._blink_morph_name,
            self._blink_weight,
            self._blink_occurrences,
            rotations,
            ik_offsets,
            translations,
            disabled_ik_indexes=self._mouth_pose_disabled_ik_indexes,
        )
        return groups

    def _refresh_mouth_mesh(self) -> bool:
        """口に使う材質だけを更新し、全三角形の再構築を避ける。"""

        preview = self._preview
        if preview is None or not self._neutral_group_vertices:
            return preview is None
        groups_by_material = {group.material_index: group for group in self._groups}
        affected_groups: list[_MeshGroup] = []
        affected_materials = self._mouth_occurrences.keys() | self._blink_occurrences.keys()
        for material_index in affected_materials:
            group = groups_by_material.get(material_index)
            neutral = self._neutral_group_vertices.get(material_index)
            if group is None or neutral is None:
                return False
            group.vertices = neutral
            affected_groups.append(group)
        _apply_mouth_shape_to_groups(
            affected_groups,
            preview,
            self._mouth_shape,
            self._mouth_occurrences,
            self._mouth_pose_rotations,
            self._mouth_pose_ik_offsets,
            self._mouth_pose_translations,
            ik_already_solved=self._mouth_pose_ik_already_solved,
            disabled_ik_indexes=self._mouth_pose_disabled_ik_indexes,
            weights=self._mouth_weights,
        )
        _apply_vertex_morph_to_groups(
            affected_groups,
            preview,
            self._blink_morph_name,
            self._blink_weight,
            self._blink_occurrences,
            self._mouth_pose_rotations,
            self._mouth_pose_ik_offsets,
            self._mouth_pose_translations,
            ik_already_solved=self._mouth_pose_ik_already_solved,
            disabled_ik_indexes=self._mouth_pose_disabled_ik_indexes,
        )
        if self._initialized and affected_groups:
            self.makeCurrent()
            try:
                for group in affected_groups:
                    if group.buffer is None:
                        continue
                    group.buffer.bind()
                    group.buffer.allocate(group.vertices, len(group.vertices))
                    group.buffer.release()
            finally:
                self.doneCurrent()
        return True

    def set_natural_motion(self, frame: NaturalMotionFrame) -> None:
        """GPU局所変形と高速まばたきへ、最新の自然動作を渡す。"""

        self._natural_motion = frame
        self._sync_blink_weight()
        self.update()

    def _sync_blink_weight(self) -> None:
        """再生中VMDの閉眼値を優先し、トラックがなければ自然まばたきを使う。"""

        weights = self._vmd_morph_weights
        if self._preview_motion_active:
            weights = self._preview_vmd_morph_weights
        elif (
            self._conversation_motion_active
            and self._command is not None
            and self._command.gesture_motion is not None
        ):
            weights = self._gesture_vmd_morph_weights
        previous_blink = self._blink_weight
        self._blink_weight = max(
            0.0,
            min(1.0, weights.get(self._blink_morph_name, self._natural_motion.blink_weight)),
        )
        if self._blink_morph_name and not math.isclose(
            previous_blink, self._blink_weight, abs_tol=0.025
        ):
            self._refresh_mouth_mesh()

    def _base_vmd_controls_pose(self) -> bool:
        """通常VMD再生中は作者が定義した姿勢だけを表示する。"""

        motion = self._idle_motion
        return motion is not None and bool(motion.bone_frames)

    def _effective_natural_motion(self) -> NaturalMotionFrame:
        """VMD再生へアプリ独自の全身変形を重ねない。"""

        return NaturalMotionFrame() if self._base_vmd_controls_pose() else self._natural_motion

    def _set_rest_expression(self, weights: dict[str, float]) -> None:
        """会話命令から独立した基礎表情を、既存の表情層へ渡す。"""

        if weights == self._rest_expression_weights:
            return
        self._rest_expression_weights = dict(weights)
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    def set_mmd_command(
        self,
        command: UiMmdAvatarCommand | None,
        *,
        smooth_gaze_transition: bool = False,
    ) -> None:
        """会話からの視線・姿勢命令を、GPUメッシュのボーン変形へ渡す。"""

        if (
            command is not None
            and command.gesture_motion_type == "reaction"
            and command.gesture_motion is not None
            and command.gesture_motion.loop
        ):
            command = replace(command, gesture_motion=replace(command.gesture_motion, loop=False))
        if command == self._command and not smooth_gaze_transition:
            return
        self._clear_tail_return()
        self._motion_mixer.interrupt()
        self._clear_motion_preview()
        if command is None or not _is_agreement_motion(command.gesture_motion):
            self._return_to_default_idle()
        self._command = command
        if smooth_gaze_transition:
            self._conversation_motion_active = bool(
                self._motion_enabled
                and command is not None
                and command.gaze not in {"", "forward", "user"}
            )
            self._blend_transition_steps = 6
            self._next_mix_delta_seconds = 0.06
        self._gesture_body_indexes = _body_mask_bone_indexes(
            self._preview,
            command.gesture_body_mask if command is not None else (),
        )
        self._reset_gesture_vmd(command.gesture_motion if command is not None else None)
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    def begin_conversation_motion(self) -> int | None:
        """視線・会話ジェスチャーの初期姿勢を作り、準備番号を返す。"""

        command = self._command
        if (
            not self._motion_enabled
            or command is None
            or (command.gesture in {"", "none"} and command.gaze in {"", "forward", "user"})
        ):
            return None
        entry_source = (
            self._capture_motion_return_pose(
                self._gesture_vmd_rotations, self._gesture_vmd_translations, {}
            )
            if _uses_low_arm_entry(command.gesture_motion)
            or _uses_short_arm_entry(command.gesture_motion)
            else None
        )
        self._conversation_motion_active = True
        self._clear_tail_return()
        self._agreement_return_from = None
        self._reset_gesture_vmd(command.gesture_motion)
        if entry_source is not None:
            self._idle_variant_active = True
            self._switch_idle_motion(command.gesture_motion, source=entry_source)
            self._gesture_vmd_rotations = dict(self._vmd_rotations)
            self._gesture_vmd_translations = dict(self._vmd_translations)
            self._gesture_vmd_morph_weights = dict(self._vmd_morph_weights)
        agreement = _is_agreement_motion(command.gesture_motion)
        self._blend_transition_steps = 6 if agreement else 0
        self._next_mix_delta_seconds = 0.05 if agreement else 0.35
        revision = self._motion_mixer.interrupt()
        self._queue_pose_mesh_refresh(force=True)
        return revision

    def end_conversation_motion(self) -> None:
        """読み上げ後は反応VMDを完走させ、組み込み動作と視線だけ解除する。"""

        if not self._conversation_motion_active:
            return
        command = self._command
        if (
            command is not None
            and command.gesture_motion_type in {"reaction", "dance"}
            and command.gesture_motion is not None
            and not self._gesture_vmd_player.is_finished
        ):
            return
        self._finish_managed_motion()

    def _finish_managed_motion(self) -> None:
        """動作を外し、終端姿勢を保持せず滑らかに待機へ戻す。"""

        return_pose = None
        return_motion = self._command.gesture_motion if self._command is not None else None
        if self._command is not None and _is_agreement_motion(self._command.gesture_motion):
            self._start_agreement_return()
        elif self._command is not None and self._command.gesture_motion_type == "reaction":
            rotations, translations = self._managed_gesture_pose()
            return_pose = self._capture_motion_return_pose(
                rotations, translations, self._gesture_vmd_morph_weights
            )
        self._conversation_motion_active = False
        self._reset_gesture_vmd(None)
        if return_pose is not None:
            self._idle_variant_active = False
            self._switch_idle_motion(
                self._default_idle_motion, source=return_pose, source_motion=return_motion
            )
        # 補間時間と同じ値を一度に渡すと、手が待機位置へ瞬間移動する。
        self._blend_transition_steps = 6
        self._next_mix_delta_seconds = 0.05
        self._motion_mixer.interrupt()
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    def _capture_motion_return_pose(
        self,
        rotations: dict[int, tuple[float, float, float]],
        translations: dict[int, tuple[float, float, float]],
        morphs: dict[str, float],
    ) -> MixedMotion:
        """表示中の所有ボーンだけを引き継ぎ、衣装物理の結果は固定しない。"""
        current = self._motion_mixer.current
        dynamic = (
            {body.bone_index for body in self._preview.rigid_bodies if body.mode != 0}
            if self._preview is not None
            else set()
        )
        return MixedMotion(
            {
                index: current.rotations.get(index, (0.0, 0.0, 0.0))
                for index in self._vmd_rotations.keys() | rotations.keys()
                if index not in dynamic
            },
            {
                index: current.translations.get(index, (0.0, 0.0, 0.0))
                for index in self._vmd_translations.keys() | translations.keys()
                if index not in dynamic
            },
            {**self._vmd_morph_weights, **morphs},
        )

    def interrupt_motion(self) -> None:
        """利用者の新しい発言で、古い会話ジェスチャーを安全に解除する。"""

        self._clear_tail_return()
        return_pose = None
        return_motion = (
            self._preview_motion
            if self._preview_motion_active
            else self._command.gesture_motion
            if self._command is not None
            else None
        )
        agreement = (
            self._preview_motion_active and _is_agreement_motion(self._preview_motion)
        ) or (
            self._conversation_motion_active
            and self._command is not None
            and _is_agreement_motion(self._command.gesture_motion)
        )
        if agreement:
            self._start_agreement_return()
        elif self._preview_motion_active:
            return_pose = self._capture_motion_return_pose(
                self._preview_vmd_rotations,
                self._preview_vmd_translations,
                {},
            )
        elif self._conversation_motion_active:
            rotations, translations = self._managed_gesture_pose()
            return_pose = self._capture_motion_return_pose(
                rotations, translations, self._gesture_vmd_morph_weights
            )
        self._motion_mixer.interrupt()
        self._conversation_motion_active = False
        self._blend_transition_steps = 0
        self._command = None
        self._reset_gesture_vmd(None)
        self._clear_motion_preview()
        if not agreement:
            self._return_to_default_idle()
        if return_pose is not None:
            self._switch_idle_motion(
                self._default_idle_motion, source=return_pose, source_motion=return_motion
            )
            self._blend_transition_steps = 6
            self._next_mix_delta_seconds = 0.05
        self._pending_pose_request = None
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    @property
    def continues_conversation_motion_without_voice(self) -> bool:
        """音声終了後も継続する反応・ダンスかを返す。"""

        command = self._command
        return bool(
            self._conversation_motion_active
            and command is not None
            and command.gesture_motion_type in {"reaction", "dance"}
            and command.gesture_motion is not None
        )

    @property
    def is_conversation_motion_active(self) -> bool:
        return self._conversation_motion_active

    @property
    def is_motion_preview_active(self) -> bool:
        return self._preview_motion_active

    def start_motion_preview(self, motion: UiVmdMotion) -> None:
        """本番の会話動作を止め、PMXを再読込せず管理用VMDを試す。"""

        self._clear_tail_return()
        entry_rotations, entry_translations = (
            self._managed_gesture_pose()
            if self._conversation_motion_active
            else (self._preview_vmd_rotations, self._preview_vmd_translations)
        )
        entry_source = self._capture_motion_return_pose(entry_rotations, entry_translations, {})
        self._motion_mixer.interrupt()
        if not _is_agreement_motion(motion):
            self._return_to_default_idle()
        self._agreement_return_from = None
        self._conversation_motion_active = False
        self._command = None
        self._reset_gesture_vmd(None)
        self._preview_motion = motion
        self._preview_motion_active = True
        self._preview_vmd_player.reset(motion)
        (
            self._preview_vmd_rotations,
            self._preview_vmd_translations,
            self._preview_vmd_morph_weights,
        ) = self._preview_vmd_player.advance_motion(self._preview, motion, 0.0, True)
        self._filter_agreement_preview()
        if (
            _uses_low_arm_entry(motion)
            or _uses_short_arm_entry(motion)
            or motion.name.startswith("ot0510_待機_左手腰あて_")
        ):
            # 通常巡回と同じ進入を使い、腰当ての袖を瞬間移動で裾へ入れない。
            self._idle_variant_active = True
            self._switch_idle_motion(motion, source=entry_source)
            self._preview_vmd_rotations = dict(self._vmd_rotations)
            self._preview_vmd_translations = dict(self._vmd_translations)
            self._preview_vmd_morph_weights = dict(self._vmd_morph_weights)
        self._blend_transition_steps = 6
        self._next_mix_delta_seconds = 0.05
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    def stop_motion_preview(self) -> None:
        """管理用プレビューだけを停止し、通常待機へ戻す。"""

        if not self._preview_motion_active:
            return
        if not self._preview_vmd_player.is_finished:
            self._clear_tail_return()
        # プレビューの閉眼・表情は残さず、従来どおり待機側へ返す。
        return_motion = self._preview_motion
        agreement = _is_agreement_motion(return_motion)
        if agreement:
            self._start_agreement_return()
        else:
            return_pose = self._capture_motion_return_pose(
                self._preview_vmd_rotations,
                self._preview_vmd_translations,
                {},
            )
        self._clear_motion_preview()
        if not agreement:
            self._idle_variant_active = False
            self._switch_idle_motion(
                self._default_idle_motion, source=return_pose, source_motion=return_motion
            )
        self._blend_transition_steps = 6
        self._next_mix_delta_seconds = 0.05
        self._motion_mixer.interrupt()
        self._queue_pose_mesh_refresh(force=True)
        self.update()

    def _clear_motion_preview(self) -> None:
        self._preview_motion = None
        self._preview_motion_active = False
        self._preview_vmd_player.reset(None)
        self._preview_vmd_rotations = {}
        self._preview_vmd_translations = {}
        self._preview_vmd_morph_weights = {}

    def _refresh_pose_mesh(self, *, force: bool = False) -> None:
        if self._cloth_state.error:
            return
        request = self._pose_build_request()
        if not force and request.signature == self._pose_signature:
            return
        self._pose_signature = request.signature
        updated = _build_pose_groups_in_background(request)[1]
        self._install_pose_groups(request, updated)

    def _pose_build_request(self) -> _PoseBuildRequest:
        self._sync_blink_weight()
        rotations, ik_offsets, physics_offsets, morph_weights = self._pose_settings()
        disabled_ik_indexes = self._disabled_ik_indexes()
        ik_already_solved = not ik_offsets and self._physics_ik_ready
        thin_cloth_source_surfaces = self._physics.thin_cloth_source_surfaces()
        if ik_already_solved:
            rotations.update(self._physics_ik_rotations)
        signature = (
            self._heavy_phase if self._cloth_state.simulation is not None else None,
            tuple(
                sorted(
                    (index, tuple(round(value, 4) for value in values))
                    for index, values in rotations.items()
                )
            ),
            tuple(sorted((name, round(weight, 4)) for name, weight in morph_weights.items())),
            tuple(
                sorted(
                    (index, tuple(round(value, 4) for value in values))
                    for index, values in ik_offsets.items()
                )
            ),
            self._mouth_shape,
            tuple(
                sorted(
                    (index, tuple(round(value, 4) for value in values))
                    for index, values in physics_offsets.items()
                )
            ),
            tuple(sorted(disabled_ik_indexes)),
            tuple(
                sorted(
                    (index, tuple(round(value, 4) for value in values))
                    for index, values in self._physics_offsets.items()
                )
            ),
            tuple(
                (
                    tuple(round(value, 4) for value in surface.position),
                    tuple(round(value, 4) for value in surface.axis),
                    round(surface.radius, 4),
                    round(surface.half_length, 4),
                )
                for surface in thin_cloth_source_surfaces
            ),
        )
        return _PoseBuildRequest(
            generation=self._pose_generation,
            motion_revision=self._motion_mixer.revision,
            preview=self._preview,
            rotations=dict(rotations),
            ik_offsets=dict(ik_offsets),
            translations=dict(physics_offsets),
            morph_weights=dict(morph_weights),
            prepared_pose=self._prepared_pose,
            ik_already_solved=ik_already_solved,
            disabled_ik_indexes=disabled_ik_indexes,
            thin_cloth_source_surfaces=thin_cloth_source_surfaces,
            signature=signature,
            physics_translations=dict(self._physics_offsets),
            cloth_seconds=self._heavy_phase / 0.92,
            cloth_state=self._cloth_state,
            gpu_skinning=self._initialized and self._gpu_skinning_enabled,
        )

    def _queue_pose_mesh_refresh(self, *, force: bool = False) -> None:
        """全メッシュのCPU計算を描画スレッドの外へ予約する。"""

        if self._executor_closed or self._cloth_state.error:
            return
        request = self._pose_build_request()
        if not force and request.signature == self._pose_signature:
            return
        self._pose_signature = request.signature
        if self._pose_future is not None:
            self._pending_pose_request = request
            return
        self._start_pose_build(request)

    def _start_pose_build(self, request: _PoseBuildRequest) -> None:
        self._pose_future = self._pose_executor.submit(_build_pose_groups_in_background, request)
        self._pose_poll_timer.start()

    def _poll_pose_build(self) -> None:
        """完了したCPU計算だけを画面スレッドでGPUへ反映する。"""

        future = self._pose_future
        if future is None or not future.done():
            return
        self._pose_future = None
        try:
            request, updated, duration_ms = future.result()
        except Exception as exc:  # noqa: BLE001 - バックグラウンド例外を画面終了へ波及させない
            self._error_text = f"PMX姿勢計算に失敗しました: {exc}"
            if self._cloth_state.error:
                self._pending_pose_request = None
                self._pose_poll_timer.stop()
                self.renderer_ready.emit(self._available)
                return
        else:
            self._last_heavy_update_ms = duration_ms
            self._heavy_update_count += 1
            if (
                request.generation == self._pose_generation
                and request.motion_revision == self._motion_mixer.revision
                and request.preview is self._preview
            ):
                try:
                    self._install_pose_groups(request, updated)
                except (RuntimeError, TypeError, ValueError) as exc:
                    self._handle_render_error(exc)
                    self.doneCurrent()
                    self._pending_pose_request = None
                    self._pose_poll_timer.stop()
                    return
                else:
                    if self._conversation_motion_active:
                        self.conversation_pose_ready.emit(request.motion_revision)
                    self.update()
        pending = self._pending_pose_request
        self._pending_pose_request = None
        if pending is not None:
            self._start_pose_build(pending)
        else:
            self._pose_poll_timer.stop()

    def _install_pose_groups(
        self,
        request: _PoseBuildRequest,
        updated: list[_MeshGroup],
    ) -> None:
        """計算済み頂点を現在の口形と合成し、GPUバッファへ反映する。"""

        self._neutral_group_vertices = {group.material_index: group.vertices for group in updated}
        self._mouth_pose_rotations = dict(request.rotations)
        self._mouth_pose_ik_offsets = dict(request.ik_offsets)
        self._mouth_pose_translations = dict(request.translations)
        self._mouth_pose_disabled_ik_indexes = request.disabled_ik_indexes
        self._mouth_pose_ik_already_solved = request.ik_already_solved
        _apply_mouth_shape_to_groups(
            updated,
            self._preview,
            self._mouth_shape,
            self._mouth_occurrences,
            request.rotations,
            request.ik_offsets,
            request.translations,
            ik_already_solved=request.ik_already_solved,
            disabled_ik_indexes=request.disabled_ik_indexes,
            weights=self._mouth_weights,
        )
        _apply_vertex_morph_to_groups(
            updated,
            self._preview,
            self._blink_morph_name,
            self._blink_weight,
            self._blink_occurrences,
            request.rotations,
            request.ik_offsets,
            request.translations,
            ik_already_solved=request.ik_already_solved,
            disabled_ik_indexes=request.disabled_ik_indexes,
        )
        can_replace = len(updated) == len(self._groups) and all(
            current.material_index == replacement.material_index
            and current.texture_path == replacement.texture_path
            and current.sphere_texture_path == replacement.sphere_texture_path
            and current.sphere_mode == replacement.sphere_mode
            and current.toon_texture_path == replacement.toon_texture_path
            and current.toon_shared_index == replacement.toon_shared_index
            and current.is_double_sided == replacement.is_double_sided
            and current.draw_outline == replacement.draw_outline
            for current, replacement in zip(self._groups, updated, strict=True)
        )
        if not can_replace:
            if self._initialized:
                self.makeCurrent()
            self._release_groups()
            self._groups = updated
            if self._initialized:
                self._upload_groups()
                self.doneCurrent()
            return
        if self._initialized:
            self.makeCurrent()
            self._upload_skin_pose(updated)
        for current, replacement in zip(self._groups, updated, strict=True):
            vertices_changed = current.vertices != replacement.vertices
            current.vertices = replacement.vertices
            current.count = replacement.count
            current.diffuse = replacement.diffuse
            current.is_transparent = replacement.is_transparent
            current.specular = replacement.specular
            current.shininess = replacement.shininess
            current.ambient = replacement.ambient
            current.edge_color = replacement.edge_color
            current.edge_size = replacement.edge_size
            current.texture_multiply = replacement.texture_multiply
            current.texture_add = replacement.texture_add
            current.sphere_multiply = replacement.sphere_multiply
            current.sphere_add = replacement.sphere_add
            current.toon_multiply = replacement.toon_multiply
            current.toon_add = replacement.toon_add
            current.gpu_pose = replacement.gpu_pose
            if current.skin_vertices != replacement.skin_vertices:
                current.skin_vertices = replacement.skin_vertices
                if self._initialized:
                    self._upload_skin_buffer(current)
            if current.buffer is not None and vertices_changed:
                current.buffer.bind()
                current.buffer.allocate(current.vertices, len(current.vertices))
                current.buffer.release()
        if self._initialized:
            self.doneCurrent()

    def _pose_settings(
        self,
    ) -> tuple[
        dict[int, tuple[float, float, float]],
        dict[int, tuple[float, float, float]],
        dict[int, tuple[float, float, float]],
        dict[str, float],
    ]:
        preview = self._preview
        command = self._command
        if preview is None:
            return {}, {}, {}, {}
        if command is None:
            command = UiMmdAvatarCommand("neutral", 0.0, "", "forward", "", "none", "")
        morph_weights: dict[str, float] = {}
        expression_morphs: dict[str, float] = {}
        natural_rest_rotations: dict[int, tuple[float, float, float]] = {}
        natural_gaze_rotations: dict[int, tuple[float, float, float]] = {}
        gaze_rotations: dict[int, tuple[float, float, float]] = {}
        gesture_rotations: dict[int, tuple[float, float, float]] = {}
        gesture_translations: dict[int, tuple[float, float, float]] = {}
        gesture_morphs: dict[str, float] = {}
        gesture_ik_offsets: dict[int, tuple[float, float, float]] = {}
        gesture_pose_weight = 1.0
        if command.expression_morphs:
            expression_morphs.update(
                {
                    name: max(0.0, min(1.0, weight))
                    for name, weight in command.expression_morphs
                    if name
                }
            )
        elif command.emotion_morph:
            expression_morphs[command.emotion_morph] = max(0.0, min(1.0, command.emotion_intensity))
        elif command.emotion == "neutral":
            expression_morphs.update(self._rest_expression_weights)
        indexes = self._bone_indexes_by_name
        gaze_owned_indexes = {
            index for name in ("両目", "左目", "右目") if (index := indexes.get(name)) is not None
        }
        vmd_rotations = {
            index: rotation
            for index, rotation in self._vmd_rotations.items()
            if index not in gaze_owned_indexes
        }
        if self._motion_enabled and not self._base_vmd_controls_pose():
            natural_rest_rotations = dict(self._natural_rest_pose)
        if self._motion_enabled:
            natural_gaze_rotation = (
                self._natural_motion.gaze_pitch,
                self._natural_motion.gaze_yaw,
                0.0,
            )
            for natural_gaze_index in self._natural_gaze_indexes:
                natural_gaze_rotations[natural_gaze_index] = natural_gaze_rotation
        gaze_indexes = (
            _effective_gaze_bone_indexes(
                preview,
                command.gaze_bone,
                self._influential_bone_indexes,
            )
            or self._natural_gaze_indexes
        )
        if self._conversation_motion_active:
            gaze_rotation = (
                {"up": 0.12, "down": -0.12}.get(command.gaze, 0.0) * command.gaze_intensity,
                {"left": 0.22, "right": -0.22}.get(command.gaze, 0.0) * command.gaze_intensity,
                0.0,
            )
            for gaze_index in gaze_indexes:
                gaze_rotations[gaze_index] = gaze_rotation
        head_index = indexes.get(command.head_bone)
        if (
            head_index is not None
            and command.gesture == "nod"
            and command.gesture_motion is None
            and self._motion_enabled
            and self._conversation_motion_active
        ):
            nod = math.sin(self._heavy_phase * 1.8) * 0.14 * command.gesture_intensity
            gesture_rotations[head_index] = (nod, 0.0, 0.0)
            upper_body_index = indexes.get(command.upper_body_bone)
            if upper_body_index is not None:
                gesture_rotations[upper_body_index] = (nod * 0.25, 0.0, 0.0)
        target_index = indexes.get(command.gesture_target)
        if (
            target_index is not None
            and command.gesture not in {"", "none", "nod"}
            and command.gesture_motion is None
            and self._conversation_motion_active
        ):
            direction = -1.0 if "右" in command.gesture_target else 1.0
            wave_angle = (
                direction
                * (0.48 + math.sin(self._heavy_phase * 2.2) * 0.16)
                * command.gesture_intensity
            )
            gesture_rotations[target_index] = (
                0.0,
                0.0,
                wave_angle,
            )
        if (
            target_index is not None
            and preview.bones[target_index].ik_target_index >= 0
            and self._conversation_motion_active
        ):
            gesture_ik_offsets[target_index] = (
                math.sin(self._heavy_phase * 1.5) * 0.12 * command.gesture_intensity,
                0.0,
                0.0,
            )
        if self._conversation_motion_active and command.gesture_motion is not None:
            managed_rotations, managed_translations = self._managed_gesture_pose()
            managed_rotations = {
                index: rotation
                for index, rotation in managed_rotations.items()
                if index not in gaze_owned_indexes
            }
            scale = max(
                0.0,
                min(2.0, command.gesture_intensity_scale),
            ) * max(0.0, min(1.0, command.gesture_intensity))
            # 全身VMDの形は感情で薄めず、明示された資産側の倍率だけ使う。
            gesture_pose_weight = (
                max(0.0, min(2.0, command.gesture_intensity_scale))
                if "full_body" in command.gesture_body_mask
                else scale
            )
            gesture_rotations.update(managed_rotations)
            gesture_translations.update(managed_translations)
            gesture_morphs.update(
                {
                    name: max(0.0, min(1.0, value * scale))
                    for name, value in self._gesture_vmd_morph_weights.items()
                }
            )
        elif self._preview_motion_active:
            gesture_rotations.update(self._preview_vmd_rotations)
            gesture_translations.update(self._preview_vmd_translations)
            gesture_morphs.update(self._preview_vmd_morph_weights)
        protected_morphs = frozenset(
            {
                *preview.mouth_morphs,
                self._blink_morph_name,
                "あ",
                "い",
                "う",
                "え",
                "お",
                "a",
                "i",
                "u",
                "e",
                "o",
            }
        )
        mix_delta_seconds = self._next_mix_delta_seconds
        if self._blend_transition_steps > 0:
            self._blend_transition_steps -= 1
            self._next_mix_delta_seconds = (
                mix_delta_seconds if self._blend_transition_steps > 0 else 0.12
            )
        else:
            self._next_mix_delta_seconds = 0.12
        # 末尾の腕だけは復帰軌道を直接表示し、胴・顔の合成は変えない。
        for index in self._tail_return_rotations.keys() & gesture_rotations.keys():
            vmd_rotations[index] = gesture_rotations.pop(index)
        mixed = self._motion_mixer.mix(
            delta_seconds=mix_delta_seconds,
            physics=MotionLayer(
                rotations=dict(self._physics_rotations),
                translations=dict(self._physics_offsets),
                rotation_limit=1.2,
                translation_limit=2.0,
            ),
            vmd=MotionLayer(
                rotations=vmd_rotations,
                translations=dict(self._vmd_translations),
                morphs=dict(self._vmd_morph_weights),
                rotation_limit=2.8,
                translation_limit=2.0,
                # 調整済み復帰は表示姿勢から補間済みなので、再度遅らせない。
                exact=(
                    not self._idle_transition_preserves_displayed_pose
                    or (
                        self._idle_transition_from is not None
                        and (
                            self._idle_return_distance_timing or self._idle_return_tail is not None
                        )
                    )
                ),
            ),
            idle=MotionLayer(
                rotations={**natural_rest_rotations, **natural_gaze_rotations},
            ),
            gesture=MotionLayer(
                rotations=gesture_rotations,
                translations=gesture_translations,
                morphs=gesture_morphs,
                rotation_limit=math.pi if self._preview_motion_active else 2.8,
                translation_limit=2.0,
                pose_weight=gesture_pose_weight,
                exact=(
                    self._preview_motion_active and not _is_agreement_motion(self._preview_motion)
                ),
            ),
            gaze=MotionLayer(rotations=gaze_rotations),
            expression=MotionLayer(morphs=expression_morphs),
            protected_morphs=protected_morphs,
        )
        # 動きを止めても、基礎表情への復帰は簡易描画と同じく反映する。
        morph_weights.update(
            mixed.morphs
            if self._motion_enabled
            else {
                name: weight
                for name, weight in expression_morphs.items()
                if name not in protected_morphs
            }
        )
        return (
            dict(mixed.rotations),
            gesture_ik_offsets,
            dict(mixed.translations),
            morph_weights,
        )

    def _disabled_ik_indexes(self) -> frozenset[int]:
        """待機VMDと会話VMDのIK切替を、PMXコントローラー番号へ変換する。"""

        states = self._vmd_player.ik_states
        if self._conversation_motion_active:
            # 上半身だけの会話VMDで、待機側の足IKを切り替えない。
            allowed = self._gesture_body_indexes
            if self._command is not None and _is_agreement_motion(self._command.gesture_motion):
                allowed = self._agreement_bone_indexes()
            states.update(
                (name, enabled)
                for name, enabled in self._gesture_vmd_player.ik_states.items()
                if allowed is None or self._bone_indexes_by_name.get(name) in allowed
            )
        elif self._preview_motion_active and not _is_agreement_motion(self._preview_motion):
            states.update(self._preview_vmd_player.ik_states)
        return frozenset(
            index
            for name, enabled in states.items()
            if not enabled and (index := self._bone_indexes_by_name.get(name)) is not None
        )

    def set_camera(self, yaw: float, pitch: float, zoom: float) -> None:
        self._yaw = yaw
        self._pitch = pitch
        self._zoom = zoom
        self.update()

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode if mode in {"full", "bust", "vmd"} else "full"
        self.update()

    def set_lightweight_animation(
        self,
        idle_phase: float,
        motion_enabled: bool,
        speaking: bool,
    ) -> None:
        """GPU行列だけを更新し、頂点再構築や物理計算は行わない。"""

        if motion_enabled != self._motion_enabled:
            self._motion_mixer.set_emergency_stopped(not motion_enabled)
        self._idle_phase = idle_phase
        self._motion_enabled = motion_enabled
        self._speaking = speaking
        self.update()

    def advance_heavy_animation(self, delta_seconds: float, motion_enabled: bool) -> None:
        """ボーン・VMD・物理と全メッシュを、低頻度の別経路で更新する。"""

        safe_delta = max(0.0, min(1.0, delta_seconds))
        preview_entry = (
            self._preview_motion_active
            and self._idle_motion is self._preview_motion
            and self._idle_transition_from is not None
        )
        if motion_enabled:
            self._heavy_phase += safe_delta * 0.92
            self._advance_idle_rotation(safe_delta)
        self._vmd_rotations, self._vmd_translations, self._vmd_morph_weights = (
            self._vmd_player.advance_motion(
                self._preview,
                self._idle_motion,
                safe_delta,
                motion_enabled,
            )
        )
        if motion_enabled:
            self._blend_idle_transition(safe_delta)
            self._blend_agreement_return(safe_delta)
        if preview_entry:
            self._preview_vmd_player.advance_motion(
                self._preview, self._preview_motion, safe_delta, motion_enabled
            )
            self._preview_vmd_rotations = dict(self._vmd_rotations)
            self._preview_vmd_translations = dict(self._vmd_translations)
            self._preview_vmd_morph_weights = dict(self._vmd_morph_weights)
        elif self._preview_motion_active:
            (
                self._preview_vmd_rotations,
                self._preview_vmd_translations,
                self._preview_vmd_morph_weights,
            ) = self._preview_vmd_player.advance_motion(
                self._preview,
                self._preview_motion,
                safe_delta,
                motion_enabled,
            )
            self._filter_agreement_preview()
        if self._preview_motion_active and motion_enabled and not preview_entry:
            self._blend_motion_tail(
                self._preview_motion, self._preview_vmd_player, self._preview_vmd_rotations
            )
        elif (
            motion_enabled
            and self._idle_variant_active
            and not self._preview_motion_active
            and not self._conversation_motion_active
        ):
            self._blend_motion_tail(self._idle_motion, self._vmd_player, self._vmd_rotations)
        physics_pose = (
            self._advance_physics(safe_delta, motion_enabled, include_gesture=True)
            if self._preview_motion_active
            else self._advance_physics(safe_delta, motion_enabled)
        )
        if motion_enabled:
            self._physics_offsets = dict(physics_pose.translations)
            self._physics_rotations = dict(physics_pose.rotations)
        self._queue_pose_mesh_refresh()
        self.update()
        if self._preview_motion_active and self._preview_vmd_player.is_finished:
            self.stop_motion_preview()

    def _advance_idle_rotation(self, delta_seconds: float) -> None:
        """標準待機を60秒保ち、別の待機を一度だけ順番に再生する。"""

        preview = self._preview
        if preview is None or not preview.idle_motions or self._default_idle_motion is None:
            return
        if self._preview_motion_active or self._conversation_motion_active or self._speaking:
            self._idle_elapsed = 0.0
            if not self._preview_motion_active and not self._conversation_motion_active:
                self._return_to_default_idle()
            return
        if self._idle_transition_from is not None:
            return
        if self._idle_variant_active:
            if self._vmd_player.is_finished:
                self._return_to_default_idle()
            return
        self._idle_elapsed += delta_seconds
        if self._idle_elapsed < 60.0:
            return
        motion = preview.idle_motions[self._idle_next_index % len(preview.idle_motions)]
        self._idle_next_index += 1
        self._idle_variant_active = True
        self._switch_idle_motion(replace(motion, loop=False))

    def _return_to_default_idle(self) -> None:
        self._idle_elapsed = 0.0
        if self._idle_variant_active:
            self._idle_variant_active = False
            self._switch_idle_motion(self._default_idle_motion)

    def _clear_tail_return(self) -> None:
        self._tail_return_motion = None
        self._tail_return_rotations = {}
        self._tail_return_target = {}
        self._tail_source_mixer = None
        self._tail_return_elapsed = 0.0

    def _blend_motion_tail(
        self,
        motion: UiVmdMotion | None,
        player: VmdMotionPlayer,
        rotations: dict[int, tuple[float, float, float]],
    ) -> None:
        """原動作の末尾から低い腕の復帰を重ね、終端で時計を切らない。"""
        if (
            motion is None
            or motion.loop
            or self._idle_transition_from is not None
            or self._default_idle_motion is None
            or "両手後ろ" not in self._default_idle_motion.name
            or not any(motion.name.startswith(prefix) for prefix in _ARM_RETURN_SECONDS)
        ):
            return
        if player is self._gesture_vmd_player:
            command = self._command
            if (
                command is None
                or "full_body" not in command.gesture_body_mask
                or command.gesture_intensity_scale != 1.0
            ):
                # 倍率付き・部分動作の合成契約は、この全身用経路で上書きしない。
                return
        duration = max(1, motion.frame_count) / 30.0
        overlap = min(0.4, duration * 0.25)
        elapsed = max(0.0, player._elapsed_seconds - (duration - overlap))
        if elapsed <= 0.0:
            return
        if self._tail_return_motion is not motion:
            self._clear_tail_return()
            self._tail_return_motion = motion
            sampler = VmdMotionPlayer()
            sampler.reset(motion)
            remaining = duration
            terminal = {}
            while remaining > 0.0:
                step = min(1.0, remaining)
                terminal, _, _ = sampler.advance_motion(self._preview, motion, step, True)
                remaining -= step
            self._tail_return_rotations = {
                index: terminal.get(index, (0.0, 0.0, 0.0))
                for index in self._low_arm_waypoint()
                if index in rotations
                and (
                    player is not self._gesture_vmd_player
                    or self._gesture_body_indexes is None
                    or index in self._gesture_body_indexes
                )
            }
            sampler.reset(self._default_idle_motion)
            self._tail_return_target, _, _ = sampler.advance_motion(
                self._preview, self._default_idle_motion, 0.0, True
            )
            if player is self._gesture_vmd_player:
                # 元動作側の表示補間も継続し、重ね始めに生VMDへ飛ばさない。
                self._tail_source_mixer = MmdMotionMixer(
                    transition_seconds=self._motion_mixer._transition_seconds
                )
                self._tail_source_mixer.mix(
                    delta_seconds=1.0,
                    vmd=MotionLayer(
                        rotations={
                            index: self._motion_mixer.current.rotations.get(index, (0.0, 0.0, 0.0))
                            for index in self._tail_return_rotations
                        },
                        rotation_limit=2.8,
                        exact=True,
                    ),
                )
        self._tail_return_elapsed = min(overlap, elapsed)
        blend = min(1.0, elapsed / overlap)
        blend = blend * blend * (3.0 - 2.0 * blend)
        path = self._sample_return_arms(
            self._tail_return_rotations,
            self._tail_return_target,
            self._tail_return_elapsed,
            _arm_return_seconds(motion),
            self._low_arm_waypoint(motion),
            not _uses_low_arm_entry(motion),
            0.25 if _uses_low_arm_entry(motion) else 0.45,
        )
        originals = rotations
        if self._tail_source_mixer is not None:
            originals = self._tail_source_mixer.mix(
                delta_seconds=self._next_mix_delta_seconds,
                gesture=MotionLayer(
                    rotations={index: rotations[index] for index in self._tail_return_rotations},
                    rotation_limit=2.8,
                ),
            ).rotations
        for index in self._tail_return_rotations:
            original = originals.get(index, (0.0, 0.0, 0.0))
            rotations[index] = tuple(
                value + (target - value) * blend
                for value, target in zip(original, path[index], strict=True)
            )

    def _low_arm_waypoint(
        self, motion: UiVmdMotion | None = None
    ) -> dict[int, tuple[float, float, float]]:
        # 袖が背面をすくう2動作は、低いまま先に後方の余裕を作る。
        back_pitch = (
            -0.3
            if motion is not None
            and motion.name.startswith(("ot0510_笑い_口あて", "ot0510_驚き_腕バタバタ_ループ"))
            else 0.0
        )
        return {
            index: (back_pitch, 0.0, sign * -0.4) if part == "腕" else (0.0, 0.0, 0.0)
            for side, sign in (("左", 1.0), ("右", -1.0))
            for part in ("肩", "腕", "腕捩", "ひじ", "手捩", "手首")
            if (index := self._bone_indexes_by_name.get(side + part)) is not None
        }

    def _switch_idle_motion(
        self,
        motion: UiVmdMotion | None,
        *,
        source: MixedMotion | None = None,
        source_motion: UiVmdMotion | None = None,
    ) -> None:
        """PMX・物理を再初期化せず、待機VMDの時計だけを切り替える。"""

        if self._preview is None:
            return
        self._idle_transition_from = source or MixedMotion(
            dict(self._vmd_rotations), dict(self._vmd_translations), dict(self._vmd_morph_weights)
        )
        self._idle_transition_preserves_displayed_pose = source is not None
        self._idle_transition_elapsed = 0.0
        self._idle_return_open_rotations = {}
        self._idle_return_tail = None
        origin = source_motion or self._idle_motion
        low_arm_entry = _uses_low_arm_entry(motion) or _uses_low_arm_entry(origin)
        self._idle_return_distance_timing = (
            motion is self._default_idle_motion
            and not low_arm_entry
            and origin is not None
            and any(origin.name.startswith(prefix) for prefix in _ARM_RETURN_SECONDS)
        )
        self._idle_return_forearm_split = 0.25 if low_arm_entry else 0.45
        self._idle_return_arm_seconds = 2.2
        if low_arm_entry:
            self._idle_return_arm_seconds = 1.2
        elif _uses_short_arm_entry(motion):
            self._idle_return_arm_seconds = 0.65
        elif motion is self._default_idle_motion:
            self._idle_return_arm_seconds = _arm_return_seconds(origin)
        if motion is not None and (
            (motion is self._default_idle_motion and "両手後ろ" in motion.name)
            or _uses_low_arm_entry(motion)
            or _uses_short_arm_entry(motion)
        ):
            # 待機姿勢は変えず、承認された移行だけ袖を裾から離す。
            self._idle_return_open_rotations = self._low_arm_waypoint(
                origin if motion is self._default_idle_motion else None
            )
        self._idle_motion = motion
        self._vmd_player.reset(motion)
        self._vmd_rotations, self._vmd_translations, self._vmd_morph_weights = (
            self._vmd_player.advance_motion(self._preview, motion, 0.0, True)
        )
        if (
            motion is self._default_idle_motion
            and origin is not None
            and origin is self._tail_return_motion
            and self._tail_return_elapsed >= min(0.4, max(1, origin.frame_count) / 120.0) - 1e-8
        ):
            # 腕の始点・終点・時計を引き継ぎ、終端で経路を作り直さない。
            self._idle_return_tail = (
                self._tail_return_rotations,
                self._tail_return_target,
                self._tail_return_elapsed,
            )
        self._clear_tail_return()
        self._motion_mixer.interrupt()
        self._pending_pose_request = None
        self._idle_elapsed = 0.0
        self._blend_idle_transition(0.0)

    def _blend_idle_transition(self, delta_seconds: float) -> None:
        """途中で保持しない復帰姿勢を、描画と物理アンカーの両方へ渡す。"""

        source = self._idle_transition_from
        if source is None:
            return
        self._idle_transition_elapsed += delta_seconds
        elapsed = self._idle_transition_elapsed
        open_rotations = self._idle_return_open_rotations
        body_seconds = 0.35 if _uses_short_arm_entry(self._idle_motion) else 2.4
        progress = max(0.0, min(1.0, elapsed / body_seconds))
        ratio = progress * progress * (3.0 - 2.0 * progress)
        target_rotations = self._vmd_rotations
        self._vmd_rotations = _interpolate_vector_maps(source.rotations, self._vmd_rotations, ratio)
        if open_rotations:
            self._vmd_rotations.update(
                self._sample_return_arms(
                    source.rotations,
                    target_rotations,
                    elapsed,
                    self._idle_return_arm_seconds,
                    open_rotations,
                    self._idle_return_distance_timing,
                    self._idle_return_forearm_split,
                )
            )
            if self._idle_return_tail is not None:
                arm_source, arm_target, offset = self._idle_return_tail
                continued = self._sample_return_arms(
                    arm_source,
                    arm_target,
                    elapsed + offset,
                    self._idle_return_arm_seconds,
                    open_rotations,
                    self._idle_return_distance_timing,
                    self._idle_return_forearm_split,
                )
                self._vmd_rotations.update({index: continued[index] for index in arm_source})
        self._vmd_translations = _interpolate_vector_maps(
            source.translations, self._vmd_translations, ratio
        )
        # 中立へ戻る最終値0も渡し、Mixer側の再補間で復帰を遅らせない。
        for index in source.rotations:
            self._vmd_rotations.setdefault(index, (0.0, 0.0, 0.0))
        for index in source.translations:
            self._vmd_translations.setdefault(index, (0.0, 0.0, 0.0))
        # 開眼値0のキーも残し、移行途中で自然まばたきへ切り替えない。
        self._vmd_morph_weights = {
            name: source.morphs.get(name, 0.0)
            + (self._vmd_morph_weights.get(name, 0.0) - source.morphs.get(name, 0.0)) * ratio
            for name in source.morphs.keys() | self._vmd_morph_weights.keys()
        }
        finished = progress >= 1.0 and (
            not open_rotations or elapsed >= self._idle_return_arm_seconds
        )
        if finished:
            self._idle_transition_from = None
            self._idle_transition_preserves_displayed_pose = False
            self._idle_return_open_rotations = {}
            self._idle_return_tail = None

    def _sample_return_arms(
        self,
        source: dict[int, tuple[float, float, float]],
        target_rotations: dict[int, tuple[float, float, float]],
        elapsed: float,
        seconds: float,
        open_rotations: dict[int, tuple[float, float, float]],
        distance_timing: bool,
        forearm_split: float = 0.45,
    ) -> dict[int, tuple[float, float, float]]:
        """末尾と復帰で同じ低い腕軌道を使う。"""
        result = {}
        # 復帰元ごとの総時間を保ち、必要な腕だけ裾を避けて迂回する。
        arm_progress = max(0.0, min(1.0, elapsed / seconds))
        distance_splits = {}
        direct_indexes = set()
        if distance_timing:
            for side in ("左", "右"):
                arm_index = self._bone_indexes_by_name.get(side + "腕")
                if arm_index not in open_rotations:
                    continue
                opened_arm = open_rotations[arm_index]
                incoming = math.dist(source.get(arm_index, (0, 0, 0)), opened_arm)
                outgoing = math.dist(opened_arm, target_rotations.get(arm_index, (0, 0, 0)))
                if incoming + outgoing <= 0.0001:
                    continue
                # 近い通過点だけ前倒しし、同じ腕の肘・手首も同じ時計で動かす。
                split = max(0.08, min(0.45, incoming / (incoming + outgoing)))
                side_indexes = {
                    index
                    for part in ("肩", "腕", "腕捩", "ひじ", "手捩", "手首")
                    if (index := self._bone_indexes_by_name.get(side + part)) is not None
                }
                # 既に低く伸びた腕を、もう一度前・外へ開き直さない。
                if all(
                    math.dist(source.get(index, (0, 0, 0)), open_rotations[index]) <= 0.12
                    for index in side_indexes
                ):
                    direct_indexes.update(side_indexes)
                for part in ("肩", "腕", "腕捩", "ひじ", "手捩", "手首"):
                    index = self._bone_indexes_by_name.get(side + part)
                    if index is not None:
                        distance_splits[index] = split
        # 低い通過点を保ち、上下と前後の折り返しを同時にしない。
        for index, opened in open_rotations.items():
            start = source.get(index, (0.0, 0.0, 0.0))
            end = target_rotations.get(index, (0.0, 0.0, 0.0))
            if index in direct_indexes:
                # 始動をゼロ速度に戻さず、終端だけ滑らかに減速する。
                weight = arm_progress + arm_progress**2 - arm_progress**3
                result[index] = tuple(a + (b - a) * weight for a, b in zip(start, end, strict=True))
                continue
            # 肘・手首を先に伸ばし、低い腕の通過時には次の姿勢へ動かす。
            split = (
                0.45
                if index
                in {
                    self._bone_indexes_by_name.get(side + part)
                    for side in ("左", "右")
                    for part in ("肩", "腕")
                }
                else forearm_split
            )
            split = distance_splits.get(index, split)
            first = arm_progress <= split
            span = split if first else 1.0 - split
            step = arm_progress / span if first else (arm_progress - split) / span
            h00 = 2.0 * step**3 - 3.0 * step**2 + 1.0
            h10 = step**3 - 2.0 * step**2 + step
            h01 = -2.0 * step**3 + 3.0 * step**2
            h11 = step**3 - step**2
            tangent = tuple(
                (b - a) * 6.0 * split * (1.0 - split) if axis < 2 else 0.0
                for axis, (a, b) in enumerate(zip(start, end, strict=True))
            )
            if index in {
                self._bone_indexes_by_name.get("左腕"),
                self._bone_indexes_by_name.get("右腕"),
            }:
                # 同じ前後角へ戻る場合も、低い頂点を前後へ通り抜ける。
                speed = math.hypot(tangent[0], tangent[1])
                if speed < 0.9:
                    tangent = (
                        (tangent[0] * 0.9 / speed, tangent[1] * 0.9 / speed, 0.0)
                        if speed > 0.0001
                        else (0.0, math.copysign(0.9, opened[2]), 0.0)
                    )
                # 前後回転だけでは、正面の手先が止まって見える。
                roll_direction = end[2] - opened[2]
                if abs(roll_direction) < 0.0001:
                    roll_direction = opened[2] - start[2]
                tangent = (*tangent[:2], math.copysign(0.6, roll_direction))
            left, right = (start, opened) if first else (opened, end)
            result[index] = tuple(
                h00 * a + h01 * b + (h11 if first else h10) * velocity * span
                for a, b, velocity in zip(left, right, tangent, strict=True)
            )
        return result

    def _agreement_bone_indexes(self) -> frozenset[int]:
        """no/yesは頭・首だけを所有し、胴・腕・視線を上書きしない。"""
        return frozenset(
            index
            for name in ("頭", "首")
            if (index := self._bone_indexes_by_name.get(name)) is not None
        )

    def _filter_agreement_preview(self) -> None:
        """管理プレビューも描画・物理に同じ頭・首の姿勢を渡す。"""
        if not _is_agreement_motion(self._preview_motion):
            return
        allowed = self._agreement_bone_indexes()
        self._preview_vmd_rotations = {
            index: value for index, value in self._preview_vmd_rotations.items() if index in allowed
        }
        self._preview_vmd_translations = {
            index: value
            for index, value in self._preview_vmd_translations.items()
            if index in allowed
        }

    def _start_agreement_return(self) -> None:
        """待機時計を戻さず、表示中の頭・首だけを短く補間する。"""
        current = self._motion_mixer.current
        allowed = self._agreement_bone_indexes()
        self._agreement_return_from = MixedMotion(
            rotations={index: current.rotations.get(index, (0.0, 0.0, 0.0)) for index in allowed},
            translations={
                index: current.translations.get(index, (0.0, 0.0, 0.0)) for index in allowed
            },
        )
        self._agreement_return_elapsed = 0.0
        self._blend_agreement_return(0.0)

    def _blend_agreement_return(self, delta_seconds: float) -> None:
        """腕を含む待機の動きは続け、頭・首を0.35秒で戻す。"""
        source = self._agreement_return_from
        if source is None:
            return
        self._agreement_return_elapsed += delta_seconds
        progress = min(1.0, self._agreement_return_elapsed / 0.35)
        weight = progress * progress * (3.0 - 2.0 * progress)
        for before, target in (
            (source.rotations, self._vmd_rotations),
            (source.translations, self._vmd_translations),
        ):
            for index, start in before.items():
                end = target.get(index, (0.0, 0.0, 0.0))
                target[index] = tuple(
                    start[axis] + (end[axis] - start[axis]) * weight for axis in range(3)
                )
        if progress >= 1.0:
            self._agreement_return_from = None

    def advance_conversation_animation(self, delta_seconds: float) -> None:
        """発話中の会話VMDと衣装物理を、同じ低頻度経路で更新する。"""

        if not self._motion_enabled or not self._conversation_motion_active:
            return
        safe_delta = max(0.0, min(1.0, delta_seconds))
        command = self._command
        gesture_motion = command.gesture_motion if command is not None else None
        gesture_entry = (
            gesture_motion is not None
            and self._idle_motion is gesture_motion
            and self._idle_transition_from is not None
        )
        self._heavy_phase += safe_delta * 0.92
        self._vmd_rotations, self._vmd_translations, self._vmd_morph_weights = (
            self._vmd_player.advance_motion(
                self._preview,
                self._idle_motion,
                safe_delta,
                True,
            )
        )
        self._blend_idle_transition(safe_delta)
        self._blend_agreement_return(safe_delta)
        if gesture_entry:
            # 進入中も元VMDを進め、表示と物理に同じ合成姿勢を渡す。
            self._gesture_vmd_player.advance_motion(self._preview, gesture_motion, safe_delta, True)
            self._gesture_vmd_rotations = dict(self._vmd_rotations)
            self._gesture_vmd_translations = dict(self._vmd_translations)
            self._gesture_vmd_morph_weights = dict(self._vmd_morph_weights)
        else:
            (
                self._gesture_vmd_rotations,
                self._gesture_vmd_translations,
                self._gesture_vmd_morph_weights,
            ) = self._gesture_vmd_player.advance_motion(
                self._preview,
                gesture_motion,
                safe_delta,
                True,
            )
        if not gesture_entry and command is not None and command.gesture_motion_type == "reaction":
            self._blend_motion_tail(
                gesture_motion, self._gesture_vmd_player, self._gesture_vmd_rotations
            )
        physics_pose = self._advance_physics(safe_delta, True, include_gesture=True)
        self._physics_offsets = dict(physics_pose.translations)
        self._physics_rotations = dict(physics_pose.rotations)
        self._queue_pose_mesh_refresh()
        self.update()
        if self._gesture_vmd_player.is_finished:
            self._finish_managed_motion()

    def _reset_gesture_vmd(self, motion: UiVmdMotion | None) -> None:
        """会話VMDの時計と出力だけを初期化し、PMX本体は再転送しない。"""

        self._gesture_vmd_player.reset(motion)
        (
            self._gesture_vmd_rotations,
            self._gesture_vmd_translations,
            self._gesture_vmd_morph_weights,
        ) = self._gesture_vmd_player.advance_motion(self._preview, motion, 0.0, True)

    def set_animation(self, idle_phase: float, motion_enabled: bool, speaking: bool) -> None:
        """旧呼出し互換。新規コードは軽量・重量更新を分けて呼ぶ。"""

        animation_advanced = (
            not math.isclose(idle_phase, self._idle_phase, abs_tol=0.000001)
            or motion_enabled != self._motion_enabled
        )
        self.set_lightweight_animation(idle_phase, motion_enabled, speaking)
        if animation_advanced:
            self.advance_heavy_animation(0.12, motion_enabled)

    def _advance_physics(
        self,
        delta_seconds: float,
        enabled: bool,
        *,
        include_gesture: bool = False,
    ) -> PhysicsPose:
        """物理を30分の1秒刻みで進め、メッシュ更新は最後の1回にする。"""

        if not enabled:
            return self._physics.step_pose(self._preview, 0.0, False)
        if isinstance(self._physics, UmtNativePmxPhysicsSimulator):
            cloth = self._cloth_state.simulation
            self._physics._output_bone_indexes = (
                cloth.render_bone_indexes if cloth is not None else None
            )
        gesture_rotations: dict[int, tuple[float, float, float]] = {}
        gesture_translations: dict[int, tuple[float, float, float]] = {}
        if include_gesture:
            if self._preview_motion_active:
                gesture_rotations = self._preview_vmd_rotations
                gesture_translations = self._preview_vmd_translations
            else:
                gesture_rotations, gesture_translations = self._managed_gesture_pose()
        rotations, translations = _physics_anchor_motion(
            self._preview,
            self._vmd_rotations,
            self._vmd_translations,
            gesture_rotations,
            gesture_translations,
            self._motion_mixer.current,
            prefer_displayed=(
                self._blend_transition_steps > 0 or self._idle_transition_from is not None
            ),
        )
        solved_rotations = (
            solve_ik_rotations(
                self._preview,
                rotations,
                {},
                translations,
                self._prepared_ik,
                disabled_controller_indexes=self._disabled_ik_indexes(),
            )
            if self._preview is not None
            else {}
        )
        if self._prepared_ik is not None:
            self._physics_ik_rotations = {
                index: value
                for index, value in solved_rotations.items()
                if index in self._prepared_ik.link_indexes
            }
        self._physics_ik_ready = self._prepared_ik is not None
        transforms = (
            bone_transforms(
                self._preview,
                solved_rotations,
                translations,
                required_indexes=self._physics_bone_indexes,
            )
            if self._preview is not None
            else {}
        )
        pose = self._physics.step_pose(
            self._preview,
            delta_seconds,
            True,
            transforms,
        )
        self._freeze_long_lived_pmx_graph()
        return pose

    def _managed_gesture_pose(
        self,
    ) -> tuple[
        dict[int, tuple[float, float, float]],
        dict[int, tuple[float, float, float]],
    ]:
        """登録された部位だけを会話VMDと物理アンカーへ渡す。"""

        allowed = self._gesture_body_indexes
        if self._command is not None and _is_agreement_motion(self._command.gesture_motion):
            allowed = self._agreement_bone_indexes()
        if allowed is None:
            return self._gesture_vmd_rotations, self._gesture_vmd_translations
        return (
            {
                index: value
                for index, value in self._gesture_vmd_rotations.items()
                if index in allowed
            },
            {
                index: value
                for index, value in self._gesture_vmd_translations.items()
                if index in allowed
            },
        )

    def _freeze_long_lived_pmx_graph(self) -> None:
        """実行中ずっと使うPMX/Bulletを、周期的な全走査から外す。"""

        preview = self._preview
        if (
            self._gc_freeze_owned
            or preview is None
            or len(preview.dense_textured_triangles) < 50_000
            or not self._physics.is_initialized
            or gc.get_freeze_count() > _INITIAL_GC_FREEZE_COUNT
        ):
            return
        # 98,000面級のPMXと301ジョイントを毎回GCがたどると、実測で
        # 100ms超の停止が生じる。モデル変更まで生存するデータだけを固定する。
        gc.collect()
        gc.freeze()
        self._gc_freeze_owned = True

    def _release_gc_freeze(self) -> None:
        """このビューが固定したデータを、モデル変更・終了時に元へ戻す。"""

        if not self._gc_freeze_owned:
            return
        gc.unfreeze()
        self._gc_freeze_owned = False

    def initializeGL(self) -> None:  # noqa: N802
        try:
            program = QOpenGLShaderProgram(self)
            if not program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX_SHADER
            ):
                raise RuntimeError(program.log())
            if not program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT_SHADER
            ):
                raise RuntimeError(program.log())
            if not program.link():
                raise RuntimeError(program.log())
            resolve_program = QOpenGLShaderProgram(self)
            if not resolve_program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, _RESOLVE_VERTEX_SHADER
            ):
                raise RuntimeError(resolve_program.log())
            if not resolve_program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, _RESOLVE_FRAGMENT_SHADER
            ):
                raise RuntimeError(resolve_program.log())
            if not resolve_program.link():
                raise RuntimeError(resolve_program.log())
            self._program = program
            self._resolve_program = resolve_program
            if not self._background_image.isNull():
                background_program = QOpenGLShaderProgram(self)
                if not background_program.addShaderFromSourceCode(
                    QOpenGLShader.ShaderTypeBit.Vertex, _RESOLVE_VERTEX_SHADER
                ):
                    raise RuntimeError(background_program.log())
                if (
                    not background_program.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Fragment, _BACKGROUND_FRAGMENT_SHADER
                    )
                    or not background_program.link()
                ):
                    raise RuntimeError(background_program.log())
                self._background_program = background_program
                self._background_texture = _create_texture(self._background_image)
            self._initialized = True
            self.context().aboutToBeDestroyed.connect(self.release_resources)
            self._upload_groups()
            self._available = True
        except RuntimeError as exc:
            self._available = False
            self._error_text = str(exc)
        self.renderer_ready.emit(self._available)

    def paintGL(self) -> None:  # noqa: N802
        functions = self.context().extraFunctions()
        try:
            target = self._supersample_target()
            self._notify_quality_status_if_changed()
            if target is None:
                self._clear_current_target(functions)
                self._paint_background(functions)
                if self._available and self._program is not None:
                    self._paint_model(functions)
                return
            if not target.bind():
                self._disable_supersampling()
                self._bind_widget_framebuffer(functions)
                self._clear_current_target(functions)
                self._paint_background(functions)
                if self._available and self._program is not None:
                    self._paint_model(functions)
                return
            functions.glViewport(0, 0, target.width(), target.height())
            self._clear_current_target(functions)
            self._paint_background(functions)
            if self._available and self._program is not None:
                self._paint_model(functions)
            self._resolve_supersample_target(functions, target)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._handle_render_error(exc)

    def _clear_current_target(self, functions: Any) -> None:
        functions.glClearColor(232 / 255, 245 / 255, 249 / 255, 1.0)
        functions.glClear(0x00004000 | 0x00000100)  # 色・深度

    def _paint_background(self, functions: Any) -> None:
        program, texture = self._background_program, self._background_texture
        if program is None or texture is None:
            return
        image = self._background_image
        crop = self._background_crop()
        functions.glDisable(0x0B71)  # GL_DEPTH_TEST
        functions.glDisable(0x0B44)  # GL_CULL_FACE
        functions.glDisable(0x0BE2)  # GL_BLEND
        texture.bind(0)
        program.bind()
        _set_scalar_uniform(program, "backgroundTexture", 0)
        program.setUniformValue(
            "crop",
            QVector4D(
                crop.x() / image.width(),
                crop.y() / image.height(),
                crop.width() / image.width(),
                crop.height() / image.height(),
            ),
        )
        functions.glDrawArrays(0x0004, 0, 3)
        program.release()
        texture.release()

    def _background_crop(self) -> QRectF:
        """通常画角は人物と同じ投影倍率・注視点で背景を切り取る。"""
        image = self._background_image
        if (
            self._preview is None
            or not self._preview.points
            or self._view_mode == "vmd"
            or image.isNull()
            or self.width() == 0
            or self.height() == 0
        ):
            return _cover_rect(image.width(), image.height(), self.width(), self.height())
        matrix = self._mvp(rotate_model=False)
        # 向きのドラッグや呼吸は人物だけに適用し、部屋は揺らさない。
        foot_y = self._model_center[1] - self._model_height / 2
        foot = matrix.map(QVector3D(self._model_center[0], foot_y, self._model_center[2]))
        top = matrix.map(
            QVector3D(self._model_center[0], foot_y + self._model_height, self._model_center[2])
        )
        target = _standing_background_rect(
            image.width(),
            image.height(),
            QPointF((foot.x() + 1) * self.width() / 2, (1 - foot.y()) * self.height() / 2),
            (top.y() - foot.y()) * self.height() / 2,
        )
        return QRectF(
            -target.left() / target.width() * image.width(),
            -target.top() / target.height() * image.height(),
            self.width() / target.width() * image.width(),
            self.height() / target.height() * image.height(),
        )

    def _supersample_target(self) -> QOpenGLFramebufferObject | None:
        """MSAAが使えない環境だけ、拡大した内部画面を用意する。"""

        if (
            self._supersample_disabled
            or self._render_quality == "low"
            or self.context().format().samples() > 1
        ):
            self._supersample_factor = 1.0
            return None
        width = max(1, round(self.width() * self.devicePixelRatioF()))
        height = max(1, round(self.height() * self.devicePixelRatioF()))
        factor = _safe_supersample_factor(self._render_quality, width, height)
        if factor <= 1.0:
            self._supersample_factor = 1.0
            return None
        size = QSize(max(1, round(width * factor)), max(1, round(height * factor)))
        current = self._supersample_buffer
        if current is not None and current.size() == size and current.isValid():
            self._supersample_factor = factor
            return current
        self._supersample_buffer = None
        buffer_format = QOpenGLFramebufferObjectFormat()
        buffer_format.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        buffer = QOpenGLFramebufferObject(size, buffer_format)
        if not buffer.isValid():
            self._supersample_disabled = True
            self._supersample_factor = 1.0
            return None
        self._supersample_buffer = buffer
        self._supersample_factor = factor
        return buffer

    def _resolve_supersample_target(
        self,
        functions: Any,
        source: QOpenGLFramebufferObject,
    ) -> None:
        """大きな内部画面を、Qtの3D表示欄へ滑らかに縮小する。"""

        width = max(1, round(self.width() * self.devicePixelRatioF()))
        height = max(1, round(self.height() * self.devicePixelRatioF()))
        self._bind_widget_framebuffer(functions)
        functions.glViewport(0, 0, width, height)
        self._clear_current_target(functions)
        program = self._resolve_program
        if program is None:
            raise RuntimeError("SSAA縮小用のGPU処理を準備できませんでした。")
        functions.glDisable(0x0B71)  # GL_DEPTH_TEST
        functions.glDisable(0x0B44)  # GL_CULL_FACE
        functions.glDisable(0x0BE2)  # GL_BLEND
        functions.glActiveTexture(0x84C0)  # GL_TEXTURE0
        functions.glBindTexture(0x0DE1, source.texture())  # GL_TEXTURE_2D
        functions.glTexParameteri(0x0DE1, 0x2801, 0x2601)  # MIN_FILTER, LINEAR
        functions.glTexParameteri(0x0DE1, 0x2800, 0x2601)  # MAG_FILTER, LINEAR
        program.bind()
        _set_scalar_uniform(program, "resolvedTexture", 0)
        functions.glDrawArrays(0x0004, 0, 3)
        program.release()
        functions.glBindTexture(0x0DE1, 0)

    def _bind_widget_framebuffer(self, functions: Any) -> None:
        functions.glBindFramebuffer(0x8D40, self.defaultFramebufferObject())  # GL_FRAMEBUFFER

    def _disable_supersampling(self) -> None:
        self._supersample_buffer = None
        self._supersample_disabled = True
        self._supersample_factor = 1.0
        self._notify_quality_status_if_changed()

    def _notify_quality_status_if_changed(self) -> None:
        """最初の実描画で確定した平滑化方式を状態欄へ通知する。"""

        status = self.quality_status_text
        if status == self._last_reported_quality_status:
            return
        self._last_reported_quality_status = status
        self.quality_status_changed.emit()

    def _paint_model(self, functions: Any) -> None:
        """描画例外を画面部品の外へ出さず、簡易表示へ戻せるよう分離する。"""

        if self._program is None:
            return
        functions.glEnable(0x0B71)  # GL_DEPTH_TEST
        functions.glEnable(0x0BE2)  # GL_BLEND
        # 色は通常の半透明合成、αは背景の不透明性を維持してQtの再合成へ渡す。
        functions.glBlendFuncSeparate(0x0302, 0x0303, 1, 0x0303)
        functions.glFrontFace(0x0900)  # GL_CW
        program = self._program
        program.bind()
        if self._skin_texture is not None:
            self._skin_texture.bind(3)
        _set_scalar_uniform(program, "bonePalette", 3)
        program.setUniformValue("mvp", self._mvp())
        program.setUniformValue(
            "viewportSize",
            QVector2D(
                max(1, round(self.width() * self.devicePixelRatioF())),
                max(1, round(self.height() * self.devicePixelRatioF())),
            ),
        )
        natural_motion = self._effective_natural_motion()
        program.setUniformValue(
            "naturalMotion",
            QVector4D(
                natural_motion.breath,
                natural_motion.neck_yaw,
                natural_motion.neck_pitch,
                natural_motion.weight_shift,
            ),
        )
        program.setUniformValue(
            "naturalPosture",
            QVector3D(
                natural_motion.posture_pitch,
                natural_motion.nod_pitch,
                self._model_height,
            ),
        )
        program.setUniformValue(
            "naturalSecondary",
            QVector4D(natural_motion.head_roll, 0.0, 0.0, 0.0),
        )
        program.setUniformValue("motionCenters", QVector4D(*self._natural_centers))
        _set_scalar_uniform(program, "baseTextureSampler", 0)
        _set_scalar_uniform(program, "sphereTextureSampler", 1)
        _set_scalar_uniform(program, "toonTextureSampler", 2)
        ready_groups = [group for group in self._groups if _group_is_ready(group)]
        if self._render_quality != "low":
            for group in ready_groups:
                if group.draw_outline and group.edge_size > 0.0:
                    self._draw_group(functions, program, group, outline=True)
        for group in ready_groups:
            self._draw_group(functions, program, group)
        functions.glFrontFace(0x0901)  # GL_CCW
        functions.glDisable(0x0B44)  # GL_CULL_FACE
        program.release()
        if self._skin_texture is not None:
            self._skin_texture.release(3)

    def _draw_group(
        self,
        functions: Any,
        program: QOpenGLShaderProgram,
        group: _MeshGroup,
        *,
        outline: bool = False,
    ) -> None:
        """一つの材質を、通常面またはMMD輪郭線として描く。"""

        if not _group_is_ready(group):
            return
        if outline:
            functions.glEnable(0x0B44)  # GL_CULL_FACE
            functions.glCullFace(0x0404)  # GL_FRONT
        elif group.is_double_sided:
            functions.glDisable(0x0B44)
        else:
            functions.glEnable(0x0B44)
            functions.glCullFace(0x0405)  # GL_BACK
        group.texture.bind(0)
        group.sphere_texture.bind(1)
        group.toon_texture.bind(2)
        program.setUniformValue("diffuse", QVector4D(*group.diffuse))
        program.setUniformValue("ambient", QVector3D(*group.ambient))
        program.setUniformValue("specular", QVector3D(*group.specular))
        program.setUniformValue("outlineColor", QVector4D(*group.edge_color))
        program.setUniformValue("textureMultiply", QVector4D(*group.texture_multiply))
        program.setUniformValue("textureAdd", QVector4D(*group.texture_add))
        program.setUniformValue("sphereMultiply", QVector4D(*group.sphere_multiply))
        program.setUniformValue("sphereAdd", QVector4D(*group.sphere_add))
        program.setUniformValue("toonMultiply", QVector4D(*group.toon_multiply))
        program.setUniformValue("toonAdd", QVector4D(*group.toon_add))
        light_direction, light_color = self._scene_light()
        program.setUniformValue("sceneLightDirection", QVector3D(*light_direction))
        program.setUniformValue("sceneLightColor", QVector3D(*light_color))
        _set_scalar_uniform(program, "shininess", group.shininess)
        _set_scalar_uniform(program, "sphereMode", group.sphere_mode)
        _set_scalar_uniform(program, "outlinePass", int(outline))
        outline_width = (
            group.edge_size
            * self._supersample_factor
            * _mmd_outline_camera_scale(self._zoom, self._view_mode)
            if outline
            else 0.0
        )
        _set_scalar_uniform(program, "outlineWidth", outline_width)
        _set_scalar_uniform(
            program,
            "hasToonTexture",
            int(bool(group.toon_texture_path) or group.toon_shared_index >= 0),
        )
        group.buffer.bind()
        program.enableAttributeArray("position")
        program.setAttributeBuffer("position", 0x1406, 0, 3, _VERTEX_STRIDE)
        program.enableAttributeArray("normal")
        program.setAttributeBuffer("normal", 0x1406, 12, 3, _VERTEX_STRIDE)
        program.enableAttributeArray("uv")
        program.setAttributeBuffer("uv", 0x1406, 24, 2, _VERTEX_STRIDE)
        program.enableAttributeArray("sphereUv")
        program.setAttributeBuffer("sphereUv", 0x1406, 32, 2, _VERTEX_STRIDE)
        program.enableAttributeArray("edgeScale")
        program.setAttributeBuffer("edgeScale", 0x1406, 40, 1, _VERTEX_STRIDE)
        skinned = group.gpu_pose is not None and group.skin_buffer is not None
        _set_scalar_uniform(program, "gpuSkinning", int(skinned))
        _set_scalar_uniform(
            program, "groundOffset", group.gpu_pose.ground_offset if skinned else 0.0
        )
        if skinned:
            group.skin_buffer.bind()
            program.enableAttributeArray("boneIndexes")
            program.setAttributeBuffer("boneIndexes", 0x1406, 0, 4, 32)
            program.enableAttributeArray("boneWeights")
            program.setAttributeBuffer("boneWeights", 0x1406, 16, 4, 32)
        else:
            program.disableAttributeArray("boneIndexes")
            program.disableAttributeArray("boneWeights")
        functions.glDrawArrays(0x0004, 0, group.count)
        group.buffer.release()
        group.texture.release(0)
        group.sphere_texture.release(1)
        group.toon_texture.release(2)

    def _upload_skin_pose(self, groups: list[_MeshGroup]) -> None:
        """各材質で共有するボーン配列だけを転送する。"""

        pose = next((g.gpu_pose for g in groups if g.gpu_pose is not None), None)
        if pose is None:
            return
        texture = self._skin_texture
        if texture is None:
            texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
            texture.setFormat(QOpenGLTexture.TextureFormat.RGBA32F)
            texture.setSize(5, len(pose.palette))
            texture.setMipLevels(1)
            texture.setMinMagFilters(QOpenGLTexture.Filter.Nearest, QOpenGLTexture.Filter.Nearest)
            texture.allocateStorage()
            if not texture.isStorageAllocated():
                raise RuntimeError("GPUボーン配列を確保できませんでした。")
            self._skin_texture = texture
        texture.setData(
            QOpenGLTexture.PixelFormat.RGBA,
            QOpenGLTexture.PixelType.Float32,
            VoidPtr(pose.palette.ctypes.data),
        )

    def _upload_skin_buffer(self, group: _MeshGroup) -> None:
        """モデルの影響ボーンとウェイトは、切替時に一度だけ転送する。"""

        if group.skin_buffer is not None:
            group.skin_buffer.destroy()
            group.skin_buffer = None
        if not group.skin_vertices:
            return
        buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        if not buffer.create():
            raise RuntimeError("GPUウェイト配列を確保できませんでした。")
        buffer.bind()
        buffer.allocate(group.skin_vertices, len(group.skin_vertices))
        buffer.release()
        group.skin_buffer = buffer

    def _handle_render_error(self, exc: Exception) -> None:
        """GPU描画だけを停止し、アプリ全体の強制終了を防ぐ。"""

        self._error_text = f"GPU描画設定に失敗しました: {exc}"
        self._available = False
        if self._program is not None:
            self._program.release()
        self._release_groups()
        self.renderer_ready.emit(False)

    def _mvp(self, *, rotate_model: bool = True) -> QMatrix4x4:
        matrix = QMatrix4x4()
        aspect = max(0.1, self.width() / max(1, self.height()))
        vmd_camera: UiVmdCameraFrame | None = (
            self._active_vmd_player().camera_frame if self._view_mode == "vmd" else None
        )
        if vmd_camera is not None and vmd_camera.perspective:
            matrix.perspective(
                max(15.0, min(60.0, float(vmd_camera.view_angle))),
                aspect,
                0.05,
                200.0,
            )
            distance = max(0.4, min(80.0, abs(vmd_camera.distance) * self._model_scale))
            matrix.translate(0.0, 0.0, -distance)
            matrix.rotate(-vmd_camera.rotation[2] * 57.2958, 0.0, 0.0, 1.0)
            matrix.rotate(vmd_camera.rotation[0] * 57.2958, 1.0, 0.0, 0.0)
            matrix.rotate(vmd_camera.rotation[1] * 57.2958, 0.0, 1.0, 0.0)
            scale = self._model_scale
            matrix.scale(scale, scale, -scale)
            matrix.translate(
                -vmd_camera.position[0],
                -vmd_camera.position[1],
                -vmd_camera.position[2],
            )
            return matrix
        matrix.perspective(32.0, aspect, 0.1, 100.0)
        # 細い表示欄では頭が横に切れないよう、画角だけを抑える。
        view_scale = min(3.4, aspect * 4.8) if self._view_mode == "bust" else 1.0
        center = self._bust_center if self._view_mode == "bust" else self._model_center
        matrix.translate(0.0, 0.0, -3.2 / max(0.4, self._zoom))
        if rotate_model:
            matrix.rotate(self._pitch * 57.2958, 1.0, 0.0, 0.0)
            matrix.rotate((self._yaw - math.pi) * 57.2958, 0.0, 1.0, 0.0)
        scale = self._model_scale * view_scale
        matrix.scale(scale, scale, -scale)
        matrix.translate(-center[0], -center[1], -center[2])
        return matrix

    def _scene_light(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """明示したVMD演出表示だけ照明キーを使い、通常品質は固定値を保つ。"""

        frame: UiVmdLightFrame | None = (
            self._active_vmd_player().light_frame if self._view_mode == "vmd" else None
        )
        if frame is None or sum(value * value for value in frame.position) < 0.000001:
            return (-0.35, 0.70, 0.55), (1.0, 1.0, 1.0)
        return frame.position, tuple(max(0.0, min(1.0, value)) for value in frame.color)  # type: ignore[return-value]

    def _active_vmd_player(self) -> VmdMotionPlayer:
        """カメラ・照明は現在の所有者を優先し、通常時は待機VMDを使う。"""

        if self._preview_motion_active:
            return self._preview_vmd_player
        if self._conversation_motion_active:
            return self._gesture_vmd_player
        return self._vmd_player

    def _upload_groups(self) -> None:
        texture_keys = {
            key
            for group in self._groups
            for key in (
                _texture_cache_key(group.texture_path),
                _texture_cache_key(group.sphere_texture_path),
                _toon_texture_cache_key(group.toon_texture_path, group.toon_shared_index),
            )
        }
        for key in sorted(texture_keys):
            if key in self._texture_cache:
                continue
            image = _texture_image(key)
            if image.isNull():
                continue
            self._texture_cache[key] = _create_texture(image)
        for group in self._groups:
            textures = (
                self._texture_cache.get(_texture_cache_key(group.texture_path)),
                self._texture_cache.get(_texture_cache_key(group.sphere_texture_path)),
                self._texture_cache.get(
                    _toon_texture_cache_key(group.toon_texture_path, group.toon_shared_index)
                ),
            )
            if any(texture is None for texture in textures):
                continue
            texture, sphere_texture, toon_texture = textures
            buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not buffer.create():
                continue
            buffer.bind()
            buffer.allocate(group.vertices, len(group.vertices))
            buffer.release()
            group.texture = texture
            group.sphere_texture = sphere_texture
            group.toon_texture = toon_texture
            group.buffer = buffer
            self._upload_skin_buffer(group)
        self._upload_skin_pose(self._groups)

    def _release_groups(self) -> None:
        if self._skin_texture is not None:
            self._skin_texture.destroy()
            self._skin_texture = None
        for group in self._groups:
            if group.skin_buffer is not None:
                group.skin_buffer.destroy()
                group.skin_buffer = None
            if group.buffer is not None:
                group.buffer.destroy()
                group.buffer = None
            if group.texture is not None:
                group.texture = None
            if group.sphere_texture is not None:
                group.sphere_texture = None
            if group.toon_texture is not None:
                group.toon_texture = None
        for texture in self._texture_cache.values():
            texture.destroy()
        self._texture_cache.clear()

    def release_resources(self) -> None:
        """現在のOpenGLコンテキスト内でGPU資源を解放する。"""

        if not self._initialized:
            return
        self.makeCurrent()
        try:
            self._supersample_buffer = None
            self._release_groups()
            if self._background_texture is not None:
                self._background_texture.destroy()
                self._background_texture = None
        finally:
            self.doneCurrent()
            self._initialized = False
            self._available = False

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]

    def shutdown(self) -> None:
        """バックグラウンド計算とOpenGL資源を安全に終了する。"""

        self._cloth_state.close()
        self._release_gc_freeze()
        self._pose_poll_timer.stop()
        self._pending_pose_request = None
        if self._pose_future is not None:
            self._pose_future.cancel()
            self._pose_future = None
        if not self._executor_closed:
            self._pose_executor.shutdown(wait=False, cancel_futures=True)
            self._executor_closed = True
        self.release_resources()


def _apply_independent_cloth(request: _PoseBuildRequest, groups: list[_MeshGroup]) -> None:
    """姿勢計算側で布を進め、表示用バッファだけへ反映する。"""

    state = request.cloth_state
    if state is None:
        return
    with state.lock:
        preview, prepared = request.preview, request.prepared_pose
        if state.closed or state.error or preview is None or prepared is None:
            return
        cloth = state.simulation
        if state.initialized and cloth is None:
            return
        transforms = request.pose_transforms
        if transforms is None:
            rotations = (
                request.rotations
                if request.ik_already_solved
                else solve_ik_rotations(
                    preview,
                    request.rotations,
                    request.ik_offsets,
                    request.translations,
                    disabled_controller_indexes=request.disabled_ik_indexes,
                )
            )
            transforms = bone_transforms(
                preview,
                rotations,
                request.translations,
                required_indexes=cloth.bone_indexes if cloth is not None else None,
                physics_translations=request.physics_translations,
            )
        points = prepared.points.copy()
        if not state.initialized:
            for group, source in zip(groups, prepared.groups, strict=True):
                vertices = np.frombuffer(group.vertices, dtype=np.float32).reshape(
                    -1, _VERTEX_FLOAT_COUNT
                )
                points[source.vertex_indexes] = vertices[:, :3]
            try:
                cloth = state.simulation = ProcessClothSimulation.create(
                    preview, points, transforms
                )
            except (RuntimeError, OSError) as exc:
                state.fail(exc)
                raise
            state.initialized = True
        if cloth is None:
            return
        indexes = cloth.vertex_indexes
        try:
            points[indexes] = cloth.advance(request.cloth_seconds, transforms)
        except ClothNonFiniteError:
            # 数値だけが壊れた世界は、同じ表示姿勢から一度だけ作り直す。
            cloth.close()
            state.simulation = None
            state.initialized = False
            try:
                cloth = state.simulation = ProcessClothSimulation.create(
                    preview, points, transforms
                )
            except (RuntimeError, OSError) as exc:
                state.fail(exc)
                raise
            state.initialized = True
            if cloth is None:
                return
            indexes = cloth.vertex_indexes
            try:
                points[indexes] = cloth.advance(request.cloth_seconds, transforms)
            except (RuntimeError, OSError) as exc:
                state.fail(exc)
                raise
        except (RuntimeError, OSError) as exc:
            state.fail(exc)
            raise
        selected = np.zeros(len(points), dtype=bool)
        selected[indexes] = True
        # 表示バッファだけを置き換え、元の頂点・ボーン・ウェイトは保持する。
        for group, source in zip(groups, prepared.groups, strict=True):
            mask = selected[source.vertex_indexes]
            if not np.any(mask):
                continue
            vertices = (
                np.frombuffer(group.vertices, dtype=np.float32)
                .reshape(-1, _VERTEX_FLOAT_COUNT)
                .copy()
            )
            vertices[mask, :3] = points[source.vertex_indexes[mask]]
            triangles = vertices[:, :3].reshape(-1, 3, 3)
            face_normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            normals = np.zeros_like(points)
            np.add.at(normals, source.vertex_indexes, np.repeat(face_normals, 3, axis=0))
            values = normals[source.vertex_indexes[mask]]
            lengths = np.linalg.norm(values, axis=1, keepdims=True)
            vertices[mask, 3:6] = values / np.maximum(lengths, 1e-8)
            group.vertices = vertices.tobytes()


def _build_pose_groups_in_background(
    request: _PoseBuildRequest,
) -> tuple[_PoseBuildRequest, list[_MeshGroup], float]:
    """OpenGLへ触れず、全頂点の姿勢だけをバックグラウンドで計算する。"""

    started_at = time.perf_counter_ns()
    prepared = request.prepared_pose
    state = request.cloth_state
    cloth_indexes: frozenset[int] | None = frozenset()
    if prepared is not None and state is not None:
        with state.lock:
            if not state.closed and not state.initialized:
                cloth_indexes = None
            if not state.closed and state.simulation is not None:
                cloth_indexes = state.simulation.bone_indexes
                # 独立布が必ず置き換える頂点用のボーン変換は重ねて行わない。
                prepared = replace(
                    prepared,
                    required_bone_indexes=(
                        prepared.required_bone_indexes & state.simulation.render_bone_indexes
                    ),
                )
    build_request = request
    if prepared is not None and request.preview is not None:
        rotations = (
            request.rotations
            if request.ik_already_solved
            else solve_ik_rotations(
                request.preview,
                request.rotations,
                request.ik_offsets,
                request.translations,
                disabled_controller_indexes=request.disabled_ik_indexes,
            )
        )
        transforms = bone_transforms(
            request.preview,
            rotations,
            request.translations,
            required_indexes=(
                None if cloth_indexes is None else prepared.required_bone_indexes | cloth_indexes
            ),
            physics_translations=request.physics_translations,
        )
        build_request = replace(
            request, rotations=rotations, ik_already_solved=True, pose_transforms=transforms
        )
        if state is not None:
            with state.lock:
                if not state.closed and not state.error and state.simulation is not None:
                    # 別プロセスの布計算と、親側の描画データ構築を並行する。
                    try:
                        state.simulation._begin_advance(request.cloth_seconds, transforms)
                    except (RuntimeError, OSError) as exc:
                        state.fail(exc)
                        raise
    groups = _build_groups(
        request.preview,
        "neutral",
        build_request.rotations,
        request.ik_offsets,
        request.translations,
        request.morph_weights,
        prepared,
        ik_already_solved=build_request.ik_already_solved,
        disabled_ik_indexes=request.disabled_ik_indexes,
        thin_cloth_source_surfaces=request.thin_cloth_source_surfaces,
        physics_translations=request.physics_translations,
        gpu_skinning=request.gpu_skinning,
        pose_transforms=build_request.pose_transforms,
    )
    _apply_independent_cloth(build_request, groups)
    duration_ms = (time.perf_counter_ns() - started_at) / 1_000_000
    return request, groups, duration_ms


def _prepare_pose_data(preview: UiAvatarModelPreview | None) -> _PreparedPoseData | None:
    """変化しないPMX情報を、毎フレーム作り直さないNumPy配列へ変換する。"""

    if preview is None or not preview.dense_textured_triangles:
        return None
    points = np.asarray(preview.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    maximum_weights = max((len(weights) for weights in preview.vertex_weights), default=0)
    bone_indexes = np.zeros((len(points), maximum_weights), dtype=np.int32)
    bone_weights = np.zeros((len(points), maximum_weights), dtype=np.float32)
    if len(preview.vertex_weights) == len(points):
        for vertex_index, weights in enumerate(preview.vertex_weights):
            column = 0
            for bone_index, weight in weights:
                if column >= maximum_weights:
                    break
                if 0 <= bone_index < len(preview.bones) and weight > 0.0:
                    bone_indexes[vertex_index, column] = bone_index
                    bone_weights[vertex_index, column] = weight
                    column += 1
    bone_rest_positions = np.asarray(
        [bone.position for bone in preview.bones],
        dtype=np.float32,
    ).reshape((-1, 3))
    total_weights = bone_weights.sum(axis=1, keepdims=True)
    normalized_bone_weights = np.divide(
        bone_weights,
        total_weights,
        out=np.zeros_like(bone_weights),
        where=total_weights > 0,
    )
    influence_vertex_indexes = tuple(
        np.flatnonzero(normalized_bone_weights[:, column] > 0.0).astype(
            np.int32,
            copy=False,
        )
        for column in range(maximum_weights)
    )
    weighted_vertex_mask = total_weights[:, 0] > 0.0
    influence_bone_indexes = tuple(
        bone_indexes[affected, column] for column, affected in enumerate(influence_vertex_indexes)
    )
    influence_weights = tuple(
        normalized_bone_weights[affected, column]
        for column, affected in enumerate(influence_vertex_indexes)
    )
    # 有効な影響だけを連続配列にし、描画ごとの同じ抽出を省く。
    bone_local_offsets = tuple(
        points[affected] - bone_rest_positions[selected]
        for affected, selected in zip(influence_vertex_indexes, influence_bone_indexes, strict=True)
    )
    ground_bones = {
        index
        for index, bone in enumerate(preview.bones)
        if bone.position[1] < 3.0
        and any(word in bone.name for word in ("足首", "足先", "つま先", "足ＩＫ", "足IK"))
    }
    ground_vertex_indexes = np.asarray(
        [
            vertex_index
            for vertex_index, weights in enumerate(preview.vertex_weights)
            if any(bone_index in ground_bones and weight >= 0.2 for bone_index, weight in weights)
        ],
        dtype=np.int32,
    )
    ground_height = (
        float(points[ground_vertex_indexes, 1].min()) if len(ground_vertex_indexes) else None
    )
    target_patterns = tuple(
        pattern.casefold() for pattern in preview.thin_cloth_target_body_patterns
    )
    target_bones = {
        body.bone_index
        for body in preview.rigid_bodies
        if body.mode != 0
        and body.bone_index >= 0
        and any(pattern in body.name.casefold() for pattern in target_patterns)
    }
    target_bones.update(
        index
        for index, bone in enumerate(preview.bones)
        if any(pattern in bone.name.casefold() for pattern in target_patterns)
    )
    thin_cloth_target_vertex_indexes = np.flatnonzero(
        np.any(
            (normalized_bone_weights > 0.0) & np.isin(bone_indexes, tuple(target_bones)),
            axis=1,
        )
        if preview.thin_cloth_contact_enabled and target_bones
        else np.zeros(len(points), dtype=bool)
    ).astype(np.int32, copy=False)
    triangles_by_material: dict[int, list[Any]] = {}
    for triangle in preview.dense_textured_triangles:
        triangles_by_material.setdefault(triangle.material_index, []).append(triangle)
    groups: list[_PreparedMeshGroup] = []
    # 表情と布を含む材質は、既存の局所更新・布法線計算をそのまま使う。
    cpu_vertex_mask = np.zeros(len(points), dtype=bool)
    for offsets in (*preview.named_vertex_morphs.values(), *preview.mouth_morphs.values()):
        for index, *_ in offsets:
            if 0 <= index < len(points):
                cpu_vertex_mask[index] = True
    skirt_bones = tuple(i for i, b in enumerate(preview.bones) if b.name.startswith("Skirt_"))
    cpu_vertex_mask |= np.any((bone_weights > 0) & np.isin(bone_indexes, skirt_bones), axis=1)
    dynamic_vertex_mask = cpu_vertex_mask.copy()
    for material_index, triangles in sorted(triangles_by_material.items()):
        first = triangles[0]
        vertex_indexes = np.asarray(
            [index for triangle in triangles for index in triangle.indexes],
            dtype=np.int32,
        )
        template = np.zeros((len(vertex_indexes), _VERTEX_FLOAT_COUNT), dtype=np.float32)
        template[:, :3] = points[vertex_indexes]
        template[:, 3:6] = np.asarray(
            [normal for triangle in triangles for normal in triangle.normals],
            dtype=np.float32,
        )
        template[:, 6:8] = np.asarray(
            [uv for triangle in triangles for uv in triangle.uvs],
            dtype=np.float32,
        )
        template[:, 8:10] = np.asarray(
            [
                vertex_uvs[0][:2] if vertex_uvs else (0.0, 0.0)
                for triangle in triangles
                for vertex_uvs in triangle.additional_uvs
            ],
            dtype=np.float32,
        )
        template[:, 10] = np.asarray(
            [scale for triangle in triangles for scale in triangle.edge_scales],
            dtype=np.float32,
        )
        vertex_indexes.setflags(write=False)
        template.setflags(write=False)
        skin_vertices = b""
        if (
            preview.bones
            and 0 < maximum_weights <= 4
            and not np.any(dynamic_vertex_mask[vertex_indexes])
        ):
            skin = np.zeros((len(vertex_indexes), 8), dtype=np.float32)
            skin[:, :maximum_weights] = bone_indexes[vertex_indexes]
            skin[:, 4 : 4 + maximum_weights] = normalized_bone_weights[vertex_indexes]
            skin_vertices = skin.tobytes()
        else:
            cpu_vertex_mask[vertex_indexes] = True
        groups.append(
            _PreparedMeshGroup(
                material_index=material_index,
                texture_path=first.texture_path,
                diffuse=first.diffuse,
                is_transparent=first.is_transparent,
                sphere_texture_path=first.sphere_texture_path,
                sphere_mode=first.sphere_mode,
                toon_texture_path=first.toon_texture_path,
                is_double_sided=first.is_double_sided,
                specular=first.specular,
                shininess=first.shininess,
                ambient=first.ambient,
                vertex_indexes=vertex_indexes,
                vertex_template=template,
                draw_outline=first.draw_outline,
                edge_color=first.edge_color,
                edge_size=first.edge_size,
                toon_shared_index=first.toon_shared_index,
                skin_vertices=skin_vertices,
                rest_vertices=template.tobytes(),
            )
        )
    cpu_vertex_mask[ground_vertex_indexes] = True
    for values in (
        cpu_vertex_mask,
        points,
        bone_indexes,
        bone_weights,
        bone_rest_positions,
        *bone_local_offsets,
        normalized_bone_weights,
        weighted_vertex_mask,
        ground_vertex_indexes,
        thin_cloth_target_vertex_indexes,
        *influence_vertex_indexes,
        *influence_bone_indexes,
        *influence_weights,
    ):
        values.setflags(write=False)
    return _PreparedPoseData(
        points=points,
        bone_indexes=bone_indexes,
        bone_weights=bone_weights,
        bone_rest_positions=bone_rest_positions,
        bone_local_offsets=bone_local_offsets,
        normalized_bone_weights=normalized_bone_weights,
        influence_vertex_indexes=influence_vertex_indexes,
        influence_bone_indexes=influence_bone_indexes,
        influence_weights=influence_weights,
        required_bone_indexes=frozenset(int(i) for i in np.unique(bone_indexes[bone_weights > 0])),
        weighted_vertex_mask=weighted_vertex_mask,
        ground_vertex_indexes=ground_vertex_indexes,
        ground_height=ground_height,
        thin_cloth_target_vertex_indexes=thin_cloth_target_vertex_indexes,
        groups=tuple(groups),
        cpu_vertex_mask=cpu_vertex_mask,
    )


def _combined_uv_offsets(
    preview: UiAvatarModelPreview,
    morph_weights: dict[str, float],
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """有効なUVモーフを通常UV・追加UV別に合算する。"""

    if not preview.named_uv_morphs or not morph_weights:
        return {}
    combined: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for name, weight in morph_weights.items():
        if not math.isfinite(weight):
            continue
        safe_weight = max(0.0, min(1.0, weight))
        for channel, vertex_index, x, y, z, w in preview.named_uv_morphs.get(name, ()):
            key = (channel, vertex_index)
            current = combined.get(key, (0.0, 0.0, 0.0, 0.0))
            combined[key] = tuple(
                current[axis] + value * safe_weight for axis, value in enumerate((x, y, z, w))
            )  # type: ignore[assignment]
    return combined


def _uv_delta_array(
    vertex_count: int,
    offsets: dict[tuple[int, int], tuple[float, float, float, float]],
    channel: int,
) -> np.ndarray | None:
    selected = {
        vertex_index: value[:2]
        for (offset_channel, vertex_index), value in offsets.items()
        if offset_channel == channel and 0 <= vertex_index < vertex_count
    }
    if not selected:
        return None
    result = np.zeros((vertex_count, 2), dtype=np.float32)
    indexes = np.fromiter(selected, dtype=np.int32)
    result[indexes] = np.asarray(tuple(selected.values()), dtype=np.float32)
    return result


def _material_state(
    preview: UiAvatarModelPreview,
    group: _PreparedMeshGroup,
    morph_weights: dict[str, float],
) -> _MaterialState:
    """PMX仕様どおり、材質値へ乗算を重ねてから加算する。"""

    if not preview.named_material_morphs or not any(
        name in preview.named_material_morphs for name in morph_weights
    ):
        return _MaterialState(
            diffuse=group.diffuse,
            specular=group.specular,
            shininess=group.shininess,
            ambient=group.ambient,
            edge_color=group.edge_color,
            edge_size=group.edge_size,
            texture_multiply=(1.0, 1.0, 1.0, 1.0),
            texture_add=(0.0, 0.0, 0.0, 0.0),
            sphere_multiply=(1.0, 1.0, 1.0, 1.0),
            sphere_add=(0.0, 0.0, 0.0, 0.0),
            toon_multiply=(1.0, 1.0, 1.0, 1.0),
            toon_add=(0.0, 0.0, 0.0, 0.0),
        )
    bases = {
        "diffuse": group.diffuse,
        "specular": group.specular,
        "shininess": (group.shininess,),
        "ambient": group.ambient,
        "edge_color": group.edge_color,
        "edge_size": (group.edge_size,),
        "texture": (1.0, 1.0, 1.0, 1.0),
        "sphere": (1.0, 1.0, 1.0, 1.0),
        "toon": (1.0, 1.0, 1.0, 1.0),
    }
    multiplied = {name: [1.0] * len(values) for name, values in bases.items()}
    added = {name: [0.0] * len(values) for name, values in bases.items()}
    for name, weight in morph_weights.items():
        if not math.isfinite(weight):
            continue
        for item in preview.named_material_morphs.get(name, ()):
            if item.material_index not in {-1, group.material_index}:
                continue
            effective_weight = max(-10.0, min(10.0, weight * item.weight_scale))
            values_by_name = {
                "diffuse": item.diffuse,
                "specular": item.specular,
                "shininess": (item.shininess,),
                "ambient": item.ambient,
                "edge_color": item.edge_color,
                "edge_size": (item.edge_size,),
                "texture": item.texture_factor,
                "sphere": item.sphere_factor,
                "toon": item.toon_factor,
            }
            for value_name, values in values_by_name.items():
                if item.operation == 0:
                    for index, value in enumerate(values):
                        multiplied[value_name][index] *= 1.0 + (value - 1.0) * effective_weight
                else:
                    for index, value in enumerate(values):
                        added[value_name][index] += value * effective_weight

    def resolved(value_name: str) -> tuple[float, ...]:
        return tuple(
            base * multiply + addition
            for base, multiply, addition in zip(
                bases[value_name],
                multiplied[value_name],
                added[value_name],
                strict=True,
            )
        )

    diffuse = resolved("diffuse")
    specular = resolved("specular")
    ambient = resolved("ambient")
    edge_color = resolved("edge_color")
    return _MaterialState(
        diffuse=(diffuse[0], diffuse[1], diffuse[2], diffuse[3]),
        specular=(specular[0], specular[1], specular[2]),
        shininess=max(0.0, resolved("shininess")[0]),
        ambient=(ambient[0], ambient[1], ambient[2]),
        edge_color=(edge_color[0], edge_color[1], edge_color[2], edge_color[3]),
        edge_size=max(0.0, resolved("edge_size")[0]),
        texture_multiply=tuple(multiplied["texture"]),  # type: ignore[arg-type]
        texture_add=tuple(added["texture"]),  # type: ignore[arg-type]
        sphere_multiply=tuple(multiplied["sphere"]),  # type: ignore[arg-type]
        sphere_add=tuple(added["sphere"]),  # type: ignore[arg-type]
        toon_multiply=tuple(multiplied["toon"]),  # type: ignore[arg-type]
        toon_add=tuple(added["toon"]),  # type: ignore[arg-type]
    )


def _build_prepared_groups(
    preview: UiAvatarModelPreview,
    prepared: _PreparedPoseData,
    mouth_shape: str,
    rotations: dict[int, tuple[float, float, float]],
    ik_target_offsets: dict[int, tuple[float, float, float]],
    translations: dict[int, tuple[float, float, float]],
    morph_weights: dict[str, float],
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
    thin_cloth_source_surfaces: tuple[ThinClothSourceSurface, ...] = (),
    physics_translations: dict[int, tuple[float, float, float]] | None = None,
    gpu_skinning: bool = False,
    pose_transforms: dict[int, BoneTransform] | None = None,
) -> list[_MeshGroup]:
    """ボーン変形と面展開をNumPyで一括処理する。"""

    points = prepared.points.copy()
    offsets: dict[int, tuple[float, float, float]] = {}
    for name, weight in morph_weights.items():
        _add_morph_offsets(offsets, preview.named_vertex_morphs.get(name, ()), weight)
    _add_morph_offsets(offsets, preview.mouth_morphs.get(mouth_shape, ()), 1.0)
    valid_offsets = {index: value for index, value in offsets.items() if 0 <= index < len(points)}
    if valid_offsets:
        indexes = np.fromiter(valid_offsets, dtype=np.int32)
        points[indexes] += np.asarray(tuple(valid_offsets.values()), dtype=np.float32)
    uv_offsets = _combined_uv_offsets(preview, morph_weights)
    main_uv_delta = _uv_delta_array(len(points), uv_offsets, 0)
    additional_uv_delta = _uv_delta_array(len(points), uv_offsets, 1)
    gpu_pose = None
    if (
        gpu_skinning
        and not thin_cloth_source_surfaces
        and any(g.skin_vertices for g in prepared.groups)
    ):
        palette = np.zeros((len(preview.bones), 5, 4), dtype=np.float32)
        palette[:, :3, :3] = np.eye(3, dtype=np.float32)
        palette[:, 3, :3] = prepared.bone_rest_positions
        palette[:, 4, :3] = prepared.bone_rest_positions
        gpu_pose = _GpuPose(palette)
    posed_points = _apply_prepared_bone_pose(
        points,
        preview,
        prepared,
        rotations,
        ik_target_offsets,
        translations,
        ik_already_solved=ik_already_solved,
        disabled_ik_indexes=disabled_ik_indexes,
        physics_translations=physics_translations,
        gpu_pose=gpu_pose,
        pose_transforms=pose_transforms,
    )
    _separate_thin_cloth_mesh(
        posed_points,
        prepared.thin_cloth_target_vertex_indexes,
        thin_cloth_source_surfaces,
    )
    result: list[_MeshGroup] = []
    for group in prepared.groups:
        skinned = gpu_pose is not None and bool(group.skin_vertices)
        if skinned and main_uv_delta is None and additional_uv_delta is None:
            vertex_bytes = group.rest_vertices
        else:
            vertices = group.vertex_template.copy()
            if not skinned:
                vertices[:, 0:3] = posed_points[group.vertex_indexes]
            if main_uv_delta is not None:
                vertices[:, 6:8] += main_uv_delta[group.vertex_indexes]
            if additional_uv_delta is not None:
                vertices[:, 8:10] += additional_uv_delta[group.vertex_indexes]
            vertex_bytes = vertices.tobytes()
        material = _material_state(preview, group, morph_weights)
        result.append(
            _MeshGroup(
                material_index=group.material_index,
                texture_path=group.texture_path,
                diffuse=material.diffuse,
                is_transparent=group.is_transparent or material.diffuse[3] < 0.999,
                sphere_texture_path=group.sphere_texture_path,
                sphere_mode=group.sphere_mode,
                toon_texture_path=group.toon_texture_path,
                is_double_sided=group.is_double_sided,
                specular=material.specular,
                shininess=material.shininess,
                ambient=material.ambient,
                vertices=vertex_bytes,
                count=len(group.vertex_indexes),
                draw_outline=group.draw_outline,
                edge_color=material.edge_color,
                edge_size=material.edge_size,
                toon_shared_index=group.toon_shared_index,
                texture_multiply=material.texture_multiply,
                texture_add=material.texture_add,
                sphere_multiply=material.sphere_multiply,
                sphere_add=material.sphere_add,
                toon_multiply=material.toon_multiply,
                toon_add=material.toon_add,
                skin_vertices=group.skin_vertices if skinned else b"",
                gpu_pose=gpu_pose if skinned else None,
            )
        )
    return result


def _apply_prepared_bone_pose(
    points: np.ndarray,
    preview: UiAvatarModelPreview,
    prepared: _PreparedPoseData,
    rotations: dict[int, tuple[float, float, float]],
    ik_target_offsets: dict[int, tuple[float, float, float]],
    translations: dict[int, tuple[float, float, float]],
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
    physics_translations: dict[int, tuple[float, float, float]] | None = None,
    gpu_pose: _GpuPose | None = None,
    pose_transforms: dict[int, BoneTransform] | None = None,
) -> np.ndarray:
    """既存のBDEF計算と同じ式を、全頂点へまとめて適用する。"""

    solved_rotations = (
        dict(rotations)
        if ik_already_solved
        else solve_ik_rotations(
            preview,
            rotations,
            ik_target_offsets,
            translations,
            disabled_controller_indexes=disabled_ik_indexes,
        )
    )
    if not solved_rotations and not translations and not physics_translations:
        return points
    if not preview.bones or prepared.bone_weights.shape[1] == 0:
        return points
    transforms = pose_transforms
    if transforms is None:
        transforms = bone_transforms(
            preview,
            solved_rotations,
            translations,
            required_indexes=prepared.required_bone_indexes,
            physics_translations=physics_translations,
        )
    # 未使用ボーンの計算を省き、元のボーン番号で参照する配列は保つ。
    matrices = np.zeros((len(preview.bones), 3, 3), dtype=np.float32)
    positions = np.zeros((len(preview.bones), 3), dtype=np.float32)
    if transforms:
        indexes = np.fromiter(transforms, dtype=np.int32)
        matrices[indexes] = np.asarray(
            [value[0] for value in transforms.values()], dtype=np.float32
        )
        positions[indexes] = np.asarray(
            [value[1] for value in transforms.values()], dtype=np.float32
        )
    if gpu_pose is not None:
        gpu_pose.palette[:, :3, :3] = matrices
        gpu_pose.palette[:, 3, :3] = positions
    point_offsets = None if np.array_equal(points, prepared.points) else points - prepared.points
    output = np.zeros_like(points)
    for column, affected in enumerate(prepared.influence_vertex_indexes):
        if not len(affected):
            continue
        selected = prepared.influence_bone_indexes[column]
        local = prepared.bone_local_offsets[column]
        selected_weights = prepared.influence_weights[column]
        if gpu_pose is not None:
            keep = prepared.cpu_vertex_mask[affected]
            affected, selected, local, selected_weights = (
                affected[keep],
                selected[keep],
                local[keep],
                selected_weights[keep],
            )
        selected_matrices = matrices[selected]
        if point_offsets is not None:
            local = local + point_offsets[affected]
        selected_positions = positions[selected]
        for axis in range(3):
            output[affected, axis] += (
                selected_matrices[:, axis, 0] * local[:, 0]
                + selected_matrices[:, axis, 1] * local[:, 1]
                + selected_matrices[:, axis, 2] * local[:, 2]
                + selected_positions[:, axis]
            ) * selected_weights
    output[~prepared.weighted_vertex_mask] = points[~prepared.weighted_vertex_mask]
    if prepared.ground_height is not None and len(prepared.ground_vertex_indexes):
        current_ground = float(output[prepared.ground_vertex_indexes, 1].min())
        output[:, 1] += prepared.ground_height - current_ground
        if gpu_pose is not None:
            gpu_pose.ground_offset = prepared.ground_height - current_ground
    return output


def _separate_thin_cloth_mesh(
    points: np.ndarray,
    target_vertex_indexes: np.ndarray,
    source_surfaces: tuple[ThinClothSourceSurface, ...],
) -> None:
    """腕・袖に食い込む指定済み布頂点だけを、表示時に局所的に離す。"""

    if not len(target_vertex_indexes) or not source_surfaces:
        return
    target_points = points[target_vertex_indexes]
    for surface in source_surfaces:
        if surface.radius <= 0.0:
            continue
        axis = np.asarray(surface.axis, dtype=np.float32)
        center = np.asarray(surface.position, dtype=np.float32)
        relative = target_points - center
        axial = np.clip(
            relative @ axis,
            -surface.half_length,
            surface.half_length,
        )
        nearest = center + axial[:, None] * axis
        radial = target_points - nearest
        distance = np.linalg.norm(radial, axis=1)
        # 表示メッシュは物理剛体と完全には一致しないため、薄い空隙を確保する。
        desired_gap = min(0.08, max(0.025, surface.radius * 0.08))
        penetration = surface.radius + desired_gap - distance
        touching = penetration > 0.0
        if not np.any(touching):
            continue
        directions = np.zeros_like(radial)
        nonzero = distance > 0.00001
        directions[nonzero] = radial[nonzero] / distance[nonzero, None]
        # 中心線上だけは固定方向を選び、フレームごとの反転を避ける。
        fallback = np.cross(axis, np.array((0.0, 0.0, 1.0), dtype=np.float32))
        if float(np.linalg.norm(fallback)) <= 0.00001:
            fallback = np.cross(axis, np.array((1.0, 0.0, 0.0), dtype=np.float32))
        fallback /= np.linalg.norm(fallback)
        directions[~nonzero] = fallback
        maximum_offset = min(0.80, max(0.12, surface.radius * 0.90))
        correction = np.minimum(penetration, maximum_offset)
        target_points[touching] += directions[touching] * correction[touching, None]
    points[target_vertex_indexes] = target_points


def _build_groups(
    preview: UiAvatarModelPreview | None,
    mouth_shape: str = "neutral",
    rotations: dict[int, tuple[float, float, float]] | None = None,
    ik_target_offsets: dict[int, tuple[float, float, float]] | None = None,
    physics_offsets: dict[int, tuple[float, float, float]] | None = None,
    morph_weights: dict[str, float] | None = None,
    prepared_pose: _PreparedPoseData | None = None,
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
    thin_cloth_source_surfaces: tuple[ThinClothSourceSurface, ...] = (),
    physics_translations: dict[int, tuple[float, float, float]] | None = None,
    gpu_skinning: bool = False,
    pose_transforms: dict[int, BoneTransform] | None = None,
) -> list[_MeshGroup]:
    if preview is None:
        return []
    if prepared_pose is not None:
        return _build_prepared_groups(
            preview,
            prepared_pose,
            mouth_shape,
            rotations or {},
            ik_target_offsets or {},
            physics_offsets or {},
            morph_weights or {},
            ik_already_solved=ik_already_solved,
            disabled_ik_indexes=disabled_ik_indexes,
            thin_cloth_source_surfaces=thin_cloth_source_surfaces,
            physics_translations=physics_translations,
            gpu_skinning=gpu_skinning,
            pose_transforms=pose_transforms,
        )
    points = list(preview.points)
    offsets: dict[int, tuple[float, float, float]] = {}
    for name, weight in (morph_weights or {}).items():
        _add_morph_offsets(offsets, preview.named_vertex_morphs.get(name, ()), weight)
    # 会話由来の感情はVMD表情へ加算し、発話中の口形は最後に優先する。
    _add_morph_offsets(offsets, preview.mouth_morphs.get(mouth_shape, ()), 1.0)
    for index, (offset_x, offset_y, offset_z) in offsets.items():
        if 0 <= index < len(points):
            x, y, z = points[index]
            points[index] = (x + offset_x, y + offset_y, z + offset_z)
    points = apply_bone_pose(
        points,
        preview,
        rotations or {},
        ik_target_offsets,
        physics_offsets,
        disabled_ik_indexes,
        physics_translations=physics_translations,
    )
    values: dict[int, array[float]] = {}
    material_info: dict[int, tuple[Any, ...]] = {}
    uv_offsets = _combined_uv_offsets(preview, morph_weights or {})
    for triangle in preview.dense_textured_triangles:
        try:
            triangle_points = tuple(points[index] for index in triangle.indexes)
        except IndexError:
            continue
        group = values.setdefault(triangle.material_index, array("f"))
        material_info.setdefault(
            triangle.material_index,
            (
                triangle.texture_path,
                triangle.diffuse,
                triangle.is_transparent,
                triangle.sphere_texture_path,
                triangle.sphere_mode,
                triangle.toon_texture_path,
                triangle.toon_shared_index,
                triangle.is_double_sided,
                triangle.specular,
                triangle.shininess,
                triangle.ambient,
                triangle.draw_outline,
                triangle.edge_color,
                triangle.edge_size,
            ),
        )
        for vertex_number, (vertex_index, point, normal, uv, edge_scale) in enumerate(
            zip(
                triangle.indexes,
                triangle_points,
                triangle.normals,
                triangle.uvs,
                triangle.edge_scales,
                strict=True,
            )
        ):
            uv_offset = uv_offsets.get((0, vertex_index), (0.0, 0.0, 0.0, 0.0))
            vertex_additional_uvs = triangle.additional_uvs[vertex_number]
            sphere_uv = vertex_additional_uvs[0][:2] if vertex_additional_uvs else (0.0, 0.0)
            sphere_offset = uv_offsets.get((1, vertex_index), (0.0, 0.0, 0.0, 0.0))
            group.extend(
                (
                    point[0],
                    point[1],
                    point[2],
                    normal[0],
                    normal[1],
                    normal[2],
                    uv[0] + uv_offset[0],
                    uv[1] + uv_offset[1],
                    sphere_uv[0] + sphere_offset[0],
                    sphere_uv[1] + sphere_offset[1],
                    edge_scale,
                )
            )
    groups = [
        _MeshGroup(
            material_index,
            texture_path,
            diffuse,
            is_transparent,
            sphere_texture_path,
            sphere_mode,
            toon_texture_path,
            is_double_sided,
            specular,
            shininess,
            ambient,
            values.tobytes(),
            len(values) // _VERTEX_FLOAT_COUNT,
            draw_outline=draw_outline,
            edge_color=edge_color,
            edge_size=edge_size,
            toon_shared_index=toon_shared_index,
        )
        for material_index, values in sorted(values.items())
        for (
            texture_path,
            diffuse,
            is_transparent,
            sphere_texture_path,
            sphere_mode,
            toon_texture_path,
            toon_shared_index,
            is_double_sided,
            specular,
            shininess,
            ambient,
            draw_outline,
            edge_color,
            edge_size,
        ) in (material_info[material_index],)
    ]
    for group in groups:
        prepared_group = _PreparedMeshGroup(
            material_index=group.material_index,
            texture_path=group.texture_path,
            diffuse=group.diffuse,
            is_transparent=group.is_transparent,
            sphere_texture_path=group.sphere_texture_path,
            sphere_mode=group.sphere_mode,
            toon_texture_path=group.toon_texture_path,
            is_double_sided=group.is_double_sided,
            specular=group.specular,
            shininess=group.shininess,
            ambient=group.ambient,
            vertex_indexes=np.empty(0, dtype=np.int32),
            vertex_template=np.empty((0, _VERTEX_FLOAT_COUNT), dtype=np.float32),
            draw_outline=group.draw_outline,
            edge_color=group.edge_color,
            edge_size=group.edge_size,
            toon_shared_index=group.toon_shared_index,
        )
        material = _material_state(preview, prepared_group, morph_weights or {})
        group.diffuse = material.diffuse
        group.is_transparent = group.is_transparent or material.diffuse[3] < 0.999
        group.specular = material.specular
        group.shininess = material.shininess
        group.ambient = material.ambient
        group.edge_color = material.edge_color
        group.edge_size = material.edge_size
        group.texture_multiply = material.texture_multiply
        group.texture_add = material.texture_add
        group.sphere_multiply = material.sphere_multiply
        group.sphere_add = material.sphere_add
        group.toon_multiply = material.toon_multiply
        group.toon_add = material.toon_add
    return groups


def _build_mouth_occurrences(
    preview: UiAvatarModelPreview | None,
) -> dict[int, dict[int, tuple[int, ...]]]:
    """口モーフ頂点が、材質別の展開済み頂点列に現れる位置を記録する。"""

    if preview is None:
        return {}
    mouth_indexes = {
        vertex_index
        for offsets in preview.mouth_morphs.values()
        for vertex_index, _offset_x, _offset_y, _offset_z in offsets
    }
    if not mouth_indexes:
        return {}
    positions: dict[int, dict[int, list[int]]] = {}
    material_counts: dict[int, int] = {}
    for triangle in preview.dense_textured_triangles:
        material_index = triangle.material_index
        cursor = material_counts.get(material_index, 0)
        material_positions = positions.setdefault(material_index, {})
        for vertex_index in triangle.indexes:
            if vertex_index in mouth_indexes:
                material_positions.setdefault(vertex_index, []).append(cursor * _VERTEX_STRIDE)
            cursor += 1
        material_counts[material_index] = cursor
    return {
        material_index: {
            vertex_index: tuple(byte_offsets)
            for vertex_index, byte_offsets in vertex_positions.items()
        }
        for material_index, vertex_positions in positions.items()
        if vertex_positions
    }


def _build_morph_occurrences(
    preview: UiAvatarModelPreview | None,
    offsets: tuple[tuple[int, float, float, float], ...],
) -> dict[int, dict[int, tuple[int, ...]]]:
    """任意の頂点モーフが展開済みGPU頂点列へ現れる位置を記録する。"""

    if preview is None or not offsets:
        return {}
    target_indexes = {index for index, _x, _y, _z in offsets}
    positions: dict[int, dict[int, list[int]]] = {}
    material_counts: dict[int, int] = {}
    for triangle in preview.dense_textured_triangles:
        material_index = triangle.material_index
        cursor = material_counts.get(material_index, 0)
        material_positions = positions.setdefault(material_index, {})
        for vertex_index in triangle.indexes:
            if vertex_index in target_indexes:
                material_positions.setdefault(vertex_index, []).append(cursor * _VERTEX_STRIDE)
            cursor += 1
        material_counts[material_index] = cursor
    return {
        material_index: {
            vertex_index: tuple(byte_offsets)
            for vertex_index, byte_offsets in vertex_positions.items()
        }
        for material_index, vertex_positions in positions.items()
        if vertex_positions
    }


def _apply_mouth_shape_to_groups(
    groups: list[_MeshGroup],
    preview: UiAvatarModelPreview | None,
    mouth_shape: str,
    occurrences: dict[int, dict[int, tuple[int, ...]]],
    rotations: dict[int, tuple[float, float, float]] | None = None,
    ik_target_offsets: dict[int, tuple[float, float, float]] | None = None,
    translations: dict[int, tuple[float, float, float]] | None = None,
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
    weights: dict[str, float] | None = None,
) -> None:
    """基準頂点列へ口の差分だけを足し、全メッシュ再構築を避ける。"""

    if preview is None:
        return
    source_offsets: dict[int, tuple[float, float, float]] = {}
    for shape, weight in (weights if weights is not None else {mouth_shape: 1.0}).items():
        _add_morph_offsets(source_offsets, preview.mouth_morphs.get(shape, ()), weight)
    if not source_offsets:
        return
    offsets = _posed_mouth_offsets(
        preview,
        tuple((index, *offset) for index, offset in source_offsets.items()),
        rotations or {},
        ik_target_offsets or {},
        translations or {},
        ik_already_solved=ik_already_solved,
        disabled_ik_indexes=disabled_ik_indexes,
    )
    for group in groups:
        vertex_positions = occurrences.get(group.material_index)
        if not vertex_positions:
            continue
        vertices = bytearray(group.vertices)
        for vertex_index, byte_offsets in vertex_positions.items():
            offset = offsets.get(vertex_index)
            if offset is None:
                continue
            for byte_offset in byte_offsets:
                if byte_offset < 0 or byte_offset + 12 > len(vertices):
                    continue
                x, y, z = struct.unpack_from("<3f", vertices, byte_offset)
                struct.pack_into(
                    "<3f",
                    vertices,
                    byte_offset,
                    x + offset[0],
                    y + offset[1],
                    z + offset[2],
                )
        group.vertices = bytes(vertices)


def _apply_vertex_morph_to_groups(
    groups: list[_MeshGroup],
    preview: UiAvatarModelPreview | None,
    morph_name: str,
    weight: float,
    occurrences: dict[int, dict[int, tuple[int, ...]]],
    rotations: dict[int, tuple[float, float, float]] | None = None,
    ik_target_offsets: dict[int, tuple[float, float, float]] | None = None,
    translations: dict[int, tuple[float, float, float]] | None = None,
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
) -> None:
    """基準頂点列へ高速まばたきなど一つの頂点モーフを加える。"""

    if preview is None or not morph_name or weight <= 0.0:
        return
    source_offsets = preview.named_vertex_morphs.get(morph_name)
    if not source_offsets:
        return
    posed = _posed_mouth_offsets(
        preview,
        source_offsets,
        rotations or {},
        ik_target_offsets or {},
        translations or {},
        ik_already_solved=ik_already_solved,
        disabled_ik_indexes=disabled_ik_indexes,
    )
    safe_weight = max(0.0, min(1.0, weight))
    for group in groups:
        vertex_positions = occurrences.get(group.material_index)
        if not vertex_positions:
            continue
        vertices = bytearray(group.vertices)
        for vertex_index, byte_offsets in vertex_positions.items():
            offset = posed.get(vertex_index)
            if offset is None:
                continue
            for byte_offset in byte_offsets:
                if byte_offset < 0 or byte_offset + 12 > len(vertices):
                    continue
                x, y, z = struct.unpack_from("<3f", vertices, byte_offset)
                struct.pack_into(
                    "<3f",
                    vertices,
                    byte_offset,
                    x + offset[0] * safe_weight,
                    y + offset[1] * safe_weight,
                    z + offset[2] * safe_weight,
                )
        group.vertices = bytes(vertices)


def _posed_mouth_offsets(
    preview: UiAvatarModelPreview,
    offsets: tuple[tuple[int, float, float, float], ...],
    rotations: dict[int, tuple[float, float, float]],
    ik_target_offsets: dict[int, tuple[float, float, float]],
    translations: dict[int, tuple[float, float, float]],
    *,
    ik_already_solved: bool = False,
    disabled_ik_indexes: frozenset[int] = frozenset(),
) -> dict[int, tuple[float, float, float]]:
    """口の頂点差分にも、現在のボーン回転と同じ向きを適用する。"""

    if len(preview.vertex_weights) != len(preview.points) or not preview.bones:
        return {
            index: (offset_x, offset_y, offset_z) for index, offset_x, offset_y, offset_z in offsets
        }
    posed_rotations = (
        dict(rotations)
        if ik_already_solved
        else solve_ik_rotations(
            preview,
            rotations,
            ik_target_offsets,
            disabled_controller_indexes=disabled_ik_indexes,
        )
    )
    transforms = bone_transforms(preview, posed_rotations, translations)
    result: dict[int, tuple[float, float, float]] = {}
    for index, offset_x, offset_y, offset_z in offsets:
        if not 0 <= index < len(preview.vertex_weights):
            continue
        weighted_rotations = [
            (transforms[bone_index][0], weight)
            for bone_index, weight in preview.vertex_weights[index]
            if bone_index in transforms and weight > 0.0
        ]
        total_weight = sum(weight for _rotation, weight in weighted_rotations)
        if total_weight <= 0.0:
            result[index] = (offset_x, offset_y, offset_z)
            continue
        transformed = (0.0, 0.0, 0.0)
        for rotation, weight in weighted_rotations:
            rotated = matrix_vector(rotation, (offset_x, offset_y, offset_z))
            normalized = weight / total_weight
            transformed = tuple(transformed[axis] + rotated[axis] * normalized for axis in range(3))  # type: ignore[assignment]
        result[index] = transformed
    return result


def _add_morph_offsets(
    destination: dict[int, tuple[float, float, float]],
    offsets: tuple[tuple[int, float, float, float], ...],
    weight: float,
) -> None:
    """複数の表情モーフを安全な範囲で合算する。"""

    safe_weight = max(0.0, min(1.0, weight))
    for index, offset_x, offset_y, offset_z in offsets:
        current_x, current_y, current_z = destination.get(index, (0.0, 0.0, 0.0))
        destination[index] = (
            current_x + offset_x * safe_weight,
            current_y + offset_y * safe_weight,
            current_z + offset_z * safe_weight,
        )


def _add_rotation(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _normalized_render_quality(value: str) -> str:
    """既存テストと内部利用のため、共通の品質正規化を呼び出す。"""

    return normalized_render_quality(value)


def _surface_format(render_quality: str) -> QSurfaceFormat:
    """既存テストと内部利用のため、共通のOpenGL形式を返す。"""

    return surface_format(render_quality)


def _safe_supersample_factor(render_quality: str, width: int, height: int) -> float:
    """GPUメモリを使いすぎない範囲で、内部描画倍率を決める。"""

    requested = {"low": 1.0, "standard": 2.0, "high": 2.0}[
        normalized_render_quality(render_quality)
    ]
    safe_width = max(1, width)
    safe_height = max(1, height)
    dimension_limit = min(4096 / safe_width, 4096 / safe_height)
    pixel_limit = math.sqrt(16_000_000 / (safe_width * safe_height))
    resolved = min(requested, dimension_limit, pixel_limit)
    return round(resolved, 2) if resolved >= 1.1 else 1.0


def _texture_cache_key(texture_path: str) -> str:
    return texture_path or "\0white-texture"


def _toon_texture_cache_key(texture_path: str, shared_index: int) -> str:
    if texture_path:
        return texture_path
    if 0 <= shared_index <= 9:
        return f"\0shared-toon-{shared_index}"
    return "\0white-texture"


def _group_is_ready(group: _MeshGroup) -> bool:
    return all(
        item is not None
        for item in (group.buffer, group.texture, group.sphere_texture, group.toon_texture)
    )


def _opengl_texture_image(image: QImage) -> QImage:
    """PMXのUVに合わせ、元画像の上下左右をそのまま保つ。"""

    return image


def _texture_image(texture_path: str) -> QImage:
    """単色材質には白い画像を使い、PMXの基本色で面を描く。"""

    if texture_path == "\0white-texture":
        texture_path = ""
    if texture_path.startswith("\0shared-toon-"):
        try:
            return _shared_toon_image(int(texture_path.rsplit("-", 1)[1]))
        except ValueError:
            return QImage()
    if texture_path:
        return QImage(texture_path)
    image = QImage(1, 1, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.white)
    return image


def _shared_toon_image(shared_index: int) -> QImage:
    """外部素材なしで共有Toon参照を安全な陰影テクスチャへ変換する。"""

    shadow_colors = (
        (150, 150, 150),
        (170, 142, 142),
        (142, 170, 142),
        (142, 142, 170),
        (170, 170, 142),
        (142, 170, 170),
        (170, 142, 170),
        (174, 156, 138),
        (138, 156, 174),
        (126, 126, 126),
    )
    shadow = shadow_colors[max(0, min(9, shared_index))]
    image = QImage(1, 256, QImage.Format.Format_RGBA8888)
    for y in range(image.height()):
        light = 1.0 - y / max(1, image.height() - 1)
        transition = min(1.0, max(0.0, light * 1.8 - 0.35))
        color = tuple(round(channel + (255 - channel) * transition) for channel in shadow)
        image.setPixelColor(0, y, QColor(color[0], color[1], color[2], 255))
    return image


def _create_texture(image: QImage) -> QOpenGLTexture:
    # PMX画像は複数部品を1枚へ並べたアトラスが多い。
    # 自動縮小画像を作ると別部品の色が混ざるため、元画像だけを使う。
    texture = QOpenGLTexture(
        _opengl_texture_image(image),
        QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps,
    )
    texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
    texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
    texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
    return texture


def _set_scalar_uniform(
    program: QOpenGLShaderProgram,
    name: str,
    value: int | float,
) -> None:
    """単一数値を型に対応したOpenGL APIでGPUへ渡す。"""

    location = program.uniformLocation(name)
    if location < 0:
        return
    context = QOpenGLContext.currentContext()
    if context is None:
        program.setUniformValue(location, value)
        return
    functions = context.extraFunctions()
    if isinstance(value, int):
        functions.glUniform1i(location, value)
    else:
        functions.glUniform1f(location, float(value))


def _mmd_outline_camera_scale(zoom: float, view_mode: str) -> float:
    """MMDのワールド幅に近づけ、投影後のモデル寸法へ輪郭幅を追従させる。"""

    view_scale = _MMD_EDGE_REFERENCE_VIEW_SCALE if view_mode == "bust" else 1.0
    projected_scale = (
        max(0.4, zoom) * view_scale / _MMD_EDGE_REFERENCE_VIEW_SCALE * _MMD_EDGE_PROJECTION_SCALE
    )
    return max(0.35, min(1.8, projected_scale))


def _effective_gaze_bone_indexes(
    preview: UiAvatarModelPreview,
    mapped_name: str,
    influential_bone_indexes: frozenset[int] | None = None,
) -> tuple[int, ...]:
    """中継の「両目」が頂点を持たないモデルでは、左右の実眼を動かす。"""

    indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
    mapped_index = indexes.get(mapped_name)
    if mapped_index is None:
        return ()

    def has_influence(index: int) -> bool:
        if influential_bone_indexes is not None:
            return index in influential_bone_indexes
        return _bone_has_vertex_influence(preview, index)

    if has_influence(mapped_index):
        return (mapped_index,)
    eye_indexes = tuple(
        indexes[name]
        for name in ("左目", "右目")
        if name in indexes and has_influence(indexes[name])
    )
    return eye_indexes or (mapped_index,)


def _natural_gaze_bone_indexes(
    preview: UiAvatarModelPreview,
    influential_bone_indexes: frozenset[int] | None = None,
) -> tuple[int, ...]:
    """自然視線に使える、頂点へ実際に影響する目ボーンを返す。"""

    names = {bone.name for bone in preview.bones}
    for candidate in ("両目", "左目", "右目"):
        if candidate in names:
            resolved = _effective_gaze_bone_indexes(
                preview,
                candidate,
                influential_bone_indexes,
            )
            if resolved:
                return resolved
    return ()


def _natural_rest_rotations(
    preview: UiAvatarModelPreview,
) -> dict[int, tuple[float, float, float]]:
    """T字の腕を下げ、VMDがないときだけ落ち着いた待機姿勢を作る。"""

    motion = preview.vmd_motion
    if motion is not None and motion.bone_frames:
        return {}
    indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
    requested = {
        "右肩": (0.0, 0.0, 0.06),
        "右腕": (0.0, 0.0, 0.82),
        "右ひじ": (0.0, 0.0, 0.14),
        "左肩": (0.0, 0.0, -0.06),
        "左腕": (0.0, 0.0, -0.82),
        "左ひじ": (0.0, 0.0, -0.14),
    }
    return {
        index: rotation
        for name, rotation in requested.items()
        if (index := indexes.get(name)) is not None
    }


def _bone_has_vertex_influence(preview: UiAvatarModelPreview, bone_index: int) -> bool:
    return any(
        index == bone_index and weight > 0.000001
        for weights in preview.vertex_weights
        for index, weight in weights
    )


def _influential_bone_indexes(
    preview: UiAvatarModelPreview | None,
) -> frozenset[int]:
    """頂点へ影響するボーンを、モデル変更時に一度だけ集める。"""

    if preview is None:
        return frozenset()
    return frozenset(
        index
        for weights in preview.vertex_weights
        for index, weight in weights
        if weight > 0.000001
    )


def _is_agreement_motion(motion: UiVmdMotion | None) -> bool:
    """承認された首振り・うなずきだけを識別する。"""
    return motion is not None and motion.name.startswith(("ot0510_no", "ot0510_yes"))


def _uses_low_arm_entry(motion: UiVmdMotion | None) -> bool:
    """開始時の低い腕開きが承認された動作だけを対象にする。"""
    return motion is not None and ("待機_猫背_" in motion.name or "待機_元気_" in motion.name)


def _uses_short_arm_entry(motion: UiVmdMotion | None) -> bool:
    """下ろした袖が裾をすくう反応だけ、開始の腕を低い経路へ通す。"""
    return motion is not None and motion.name.startswith(
        ("ot0510_ひらめく", "ot0510_考える_ループ", "ot0510_苦笑い_ループ")
    )


def _arm_return_seconds(motion: UiVmdMotion | None) -> float:
    """表示名の接尾辞によらず、確認対象の復帰時間を選ぶ。"""
    if motion is not None:
        for prefix, seconds in _ARM_RETURN_SECONDS.items():
            if motion.name.startswith(prefix):
                return seconds
    return 2.2


def _body_mask_bone_indexes(
    preview: UiAvatarModelPreview | None,
    body_mask: tuple[str, ...],
) -> frozenset[int] | None:
    """MMD標準ボーン階層から、カタログの所有部位を一度だけ解決する。"""

    normalized = {part.strip().lower() for part in body_mask if part.strip()}
    if preview is None or not normalized or "full_body" in normalized:
        return None
    root_names: set[str] = set()
    if normalized & {"upper_body", "torso"}:
        root_names.update({"上半身", "上半身2"})
    if normalized & {"arms", "both_arms"}:
        root_names.update({"左肩", "右肩", "左腕", "右腕"})
    if "head" in normalized:
        root_names.update({"首", "頭", "両目"})
    if "neck" in normalized:
        root_names.add("首")
    if normalized & {"lower_body", "legs"}:
        root_names.update({"下半身", "左足", "右足", "左足ＩＫ", "右足ＩＫ"})
    roots = {index for index, bone in enumerate(preview.bones) if bone.name in root_names}
    if not roots:
        return None
    allowed = set(roots)
    changed = True
    while changed:
        changed = False
        for index, bone in enumerate(preview.bones):
            if index not in allowed and bone.parent_index in allowed:
                allowed.add(index)
                changed = True
    return frozenset(allowed)


def _blink_morph_name(preview: UiAvatarModelPreview | None) -> str:
    if preview is None:
        return ""
    return next(
        (
            name
            for name in preview.named_vertex_morphs
            if name.casefold() in {"まばたき", "瞬き", "blink"}
        ),
        "",
    )


def _model_scale(preview: UiAvatarModelPreview | None) -> float:
    if preview is None:
        return 1.0
    maximum = max((max(abs(value) for value in point) for point in preview.points), default=1.0)
    return 1.7 / max(maximum, 0.001)


def _model_height(preview: UiAvatarModelPreview | None) -> float:
    if preview is None or not preview.points:
        return 1.0
    ys = [point[1] for point in preview.points]
    return max(0.001, max(ys) - min(ys))


def _natural_motion_centers(
    preview: UiAvatarModelPreview | None,
) -> tuple[float, float, float, float]:
    """足元・腰・胸郭・首の高さを実ボーンから取得する。"""

    if preview is None or not preview.points:
        return (0.0, 0.35, 0.62, 0.82)
    minimum_y = min(point[1] for point in preview.points)
    height = _model_height(preview)
    bone_y = {bone.name: bone.position[1] for bone in preview.bones}

    def resolved(names: tuple[str, ...], fallback_ratio: float) -> float:
        return next(
            (bone_y[name] for name in names if name in bone_y),
            minimum_y + height * fallback_ratio,
        )

    hip_y = resolved(("腰", "下半身", "センター"), 0.58)
    chest_y = resolved(("上半身2", "上半身"), 0.69)
    neck_y = resolved(("首", "頭"), 0.82)
    return (minimum_y, hip_y, chest_y, neck_y)


def _model_center(preview: UiAvatarModelPreview | None) -> tuple[float, float, float]:
    """足元ではなく、モデル全体の中心をカメラの中心へ合わせる。"""

    if preview is None or not preview.points:
        return (0.0, 0.0, 0.0)
    return tuple(
        (
            min(point[axis] for point in preview.points)
            + max(point[axis] for point in preview.points)
        )
        / 2.0
        for axis in range(3)
    )  # type: ignore[return-value]


def _bust_center(preview: UiAvatarModelPreview | None) -> tuple[float, float, float]:
    """口モーフがあれば口を中心にし、発話中の変形を確認しやすくする。"""

    center_x, _center_y, center_z = _model_center(preview)
    if preview is None or not preview.points:
        return (center_x, 0.0, center_z)
    mouth_indexes = {
        index
        for offsets in preview.mouth_morphs.values()
        for index, _offset_x, _offset_y, _offset_z in offsets
        if 0 <= index < len(preview.points)
    }
    if mouth_indexes:
        return tuple(
            (
                min(preview.points[index][axis] for index in mouth_indexes)
                + max(preview.points[index][axis] for index in mouth_indexes)
            )
            / 2.0
            for axis in range(3)
        )  # type: ignore[return-value]
    minimum_y = min(point[1] for point in preview.points)
    maximum_y = max(point[1] for point in preview.points)
    return (center_x, minimum_y + (maximum_y - minimum_y) * 0.84, center_z)


def _blink_weight(phase: float) -> float:
    """まばたき用の短い閉眼パルス。口・感情モーフとは別に合成する。"""

    progress = (phase * 0.11) % 4.2
    if not 3.72 <= progress <= 3.9:
        return 0.0
    return max(0.0, 1.0 - abs(progress - 3.81) / 0.09)


_VERTEX_SHADER = """
#version 150
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec2 sphereUv;
in float edgeScale;
in vec4 boneIndexes;
in vec4 boneWeights;
uniform sampler2D bonePalette;
uniform int gpuSkinning;
uniform float groundOffset;
uniform mat4 mvp;
uniform vec2 viewportSize;
uniform vec4 naturalMotion;
uniform vec3 naturalPosture;
uniform vec4 naturalSecondary;
uniform vec4 motionCenters;
uniform float outlineWidth;
out vec2 textureUv;
out vec2 sphereTextureUv;
out vec3 surfaceNormal;

vec3 skinnedPosition() {
    if (gpuSkinning == 0) return position;
    vec3 result = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 4; ++i) {
        float weight = boneWeights[i];
        if (weight <= 0.0) continue;
        int bone = int(boneIndexes[i]);
        vec3 local = position - texelFetch(bonePalette, ivec2(4, bone), 0).xyz;
        vec3 rotated = vec3(
            dot(texelFetch(bonePalette, ivec2(0, bone), 0).xyz, local),
            dot(texelFetch(bonePalette, ivec2(1, bone), 0).xyz, local),
            dot(texelFetch(bonePalette, ivec2(2, bone), 0).xyz, local)
        );
        result += (rotated + texelFetch(bonePalette, ivec2(3, bone), 0).xyz) * weight;
        total += weight;
    }
    if (total == 0.0) result = position;
    return result + vec3(0.0, groundOffset, 0.0);
}

mat3 rotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}

mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

mat3 rotateZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);
}

vec3 naturalPosition(vec3 source) {
    float modelHeight = max(naturalPosture.z, 0.001);
    vec3 result = source;

    // 胸郭周辺だけを横・奥行きへ広げ、上下浮遊を起こさない。
    float chestDistance = (source.y - motionCenters.z) / (modelHeight * 0.12);
    float chestMask = exp(-(chestDistance * chestDistance));
    result.x *= 1.0 + naturalMotion.x * chestMask;
    result.z *= 1.0 + naturalMotion.x * chestMask * 0.55;

    // 足元を固定したまま、腰より上へ重心移動を徐々に加える。
    float weightMask = smoothstep(
        motionCenters.x + modelHeight * 0.04,
        motionCenters.y + modelHeight * 0.08,
        source.y
    );
    result.x += naturalMotion.w * modelHeight * weightMask;

    // 腰が動く方向と反対へ上半身を少し傾け、横滑りではなく重心移動に見せる。
    float balanceMask = smoothstep(motionCenters.y, motionCenters.z, source.y);
    vec3 hipLocal = result - vec3(0.0, motionCenters.y, 0.0);
    hipLocal = rotateZ(-naturalMotion.w * 2.2 * balanceMask) * hipLocal;
    result = hipLocal + vec3(0.0, motionCenters.y, 0.0);

    // 発話・聞き取り姿勢は胸郭を支点にし、腰中心のうなずきを避ける。
    float postureMask = smoothstep(motionCenters.y, motionCenters.z, source.y);
    vec3 chestLocal = result - vec3(0.0, motionCenters.z, 0.0);
    chestLocal = rotateX(naturalPosture.x * postureMask) * chestLocal;
    result = chestLocal + vec3(0.0, motionCenters.z, 0.0);

    // 首振りと文章区切りのうなずきは、首より上だけへ適用する。
    float headMask = smoothstep(
        motionCenters.w - modelHeight * 0.015,
        motionCenters.w + modelHeight * 0.025,
        source.y
    );
    vec3 neckLocal = result - vec3(0.0, motionCenters.w, 0.0);
    neckLocal = rotateY(naturalMotion.y * headMask) * neckLocal;
    neckLocal = rotateX((naturalMotion.z + naturalPosture.y) * headMask) * neckLocal;
    neckLocal = rotateZ(naturalSecondary.x * headMask) * neckLocal;
    return neckLocal + vec3(0.0, motionCenters.w, 0.0);
}

void main() {
    vec3 posedPosition = skinnedPosition();
    vec3 naturalBase = naturalPosition(posedPosition);
    vec4 clipPosition = mvp * vec4(naturalBase, 1.0);
    float pixelWidth = outlineWidth * max(edgeScale, 0.0);
    if (pixelWidth > 0.0) {
        float normalProbe = max(naturalPosture.z * 0.001, 0.0001);
        vec3 naturalNormalPoint = naturalPosition(posedPosition + normalize(normal) * normalProbe);
        vec4 normalClip = mvp * vec4(naturalNormalPoint, 1.0);
        vec2 baseNdc = clipPosition.xy / max(abs(clipPosition.w), 0.00001);
        vec2 normalNdc = normalClip.xy / max(abs(normalClip.w), 0.00001);
        vec2 pixelDirection = (normalNdc - baseNdc) * viewportSize;
        if (length(pixelDirection) > 0.00001) {
            vec2 ndcOffset = normalize(pixelDirection) * pixelWidth * 2.0 / viewportSize;
            clipPosition.xy += ndcOffset * clipPosition.w;
        }
        clipPosition.z += 0.0001 * clipPosition.w;
    }
    gl_Position = clipPosition;
    textureUv = uv;
    sphereTextureUv = sphereUv;
    surfaceNormal = normal;
}
"""

_FRAGMENT_SHADER = """
#version 150
uniform sampler2D baseTextureSampler;
uniform sampler2D sphereTextureSampler;
uniform sampler2D toonTextureSampler;
uniform vec4 diffuse;
uniform vec3 ambient;
uniform vec3 specular;
uniform float shininess;
uniform int sphereMode;
uniform int hasToonTexture;
uniform int outlinePass;
uniform vec4 outlineColor;
uniform vec4 textureMultiply;
uniform vec4 textureAdd;
uniform vec4 sphereMultiply;
uniform vec4 sphereAdd;
uniform vec4 toonMultiply;
uniform vec4 toonAdd;
uniform vec3 sceneLightDirection;
uniform vec3 sceneLightColor;
in vec2 textureUv;
in vec2 sphereTextureUv;
in vec3 surfaceNormal;
out vec4 fragmentColor;
vec3 morphTextureColor(vec3 source, vec4 multiplied, vec4 added) {
    return source * mix(vec3(1.0), multiplied.rgb, multiplied.a)
        + mix(vec3(0.0), added.rgb, added.a);
}
void main() {
    vec4 baseSample = texture(baseTextureSampler, textureUv);
    vec4 color = vec4(
        morphTextureColor(baseSample.rgb, textureMultiply, textureAdd),
        baseSample.a
    ) * diffuse;
    if (color.a < 0.02) {
        discard;
    }
    if (outlinePass == 1) {
        fragmentColor = vec4(outlineColor.rgb, outlineColor.a * color.a);
        return;
    }
    // PMXの両面材質は裏側からも描画される。裏面では光を受ける向きも
    // 反転しないと、スカート裏などが黒い板状に見える。
    vec3 normal = gl_FrontFacing ? normalize(surfaceNormal) : -normalize(surfaceNormal);
    vec3 lightDirection = normalize(sceneLightDirection);
    float lightAmount = max(dot(normal, lightDirection), 0.0);
    vec3 toon = hasToonTexture == 1
        ? morphTextureColor(
            texture(toonTextureSampler, vec2(0.5, 1.0 - lightAmount)).rgb,
            toonMultiply,
            toonAdd
        )
        : vec3(0.58 + floor(lightAmount * 3.0) * 0.14);
    vec2 sphereCoordinates = sphereMode == 3
        ? sphereTextureUv
        : normal.xy * 0.5 + 0.5;
    vec3 sphere = morphTextureColor(
        texture(sphereTextureSampler, sphereCoordinates).rgb,
        sphereMultiply,
        sphereAdd
    );
    if (sphereMode == 1) {
        color.rgb *= sphere;
    } else if (sphereMode == 2) {
        color.rgb += sphere;
    } else if (sphereMode == 3) {
        color.rgb *= sphere;
    }
    vec3 viewDirection = vec3(0.0, 0.0, 1.0);
    vec3 reflected = reflect(-lightDirection, normal);
    float highlight = pow(max(dot(reflected, viewDirection), 0.0), max(1.0, shininess));
    vec3 lightColor = ambient + toon * sceneLightColor;
    float maxLight = max(lightColor.r, max(lightColor.g, lightColor.b));
    color.rgb = color.rgb * lightColor / max(maxLight, 1.0) + specular * highlight;
    fragmentColor = vec4(color.rgb, color.a);
}
"""

_RESOLVE_VERTEX_SHADER = """
#version 150
out vec2 textureUv;
void main() {
    vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    textureUv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
"""

_RESOLVE_FRAGMENT_SHADER = """
#version 150
uniform sampler2D resolvedTexture;
in vec2 textureUv;
out vec4 fragmentColor;
void main() {
    fragmentColor = texture(resolvedTexture, textureUv);
}
"""

_BACKGROUND_FRAGMENT_SHADER = """
#version 150
uniform sampler2D backgroundTexture;
uniform vec4 crop;
in vec2 textureUv;
out vec4 fragmentColor;
void main() {
    vec2 uv = vec2(textureUv.x, 1.0 - textureUv.y);
    vec2 sourceUv = crop.xy + uv * crop.zw;
    if (any(lessThan(sourceUv, vec2(0.0))) || any(greaterThan(sourceUv, vec2(1.0)))) {
        discard;
    }
    fragmentColor = texture(backgroundTexture, sourceUv);
}
"""
