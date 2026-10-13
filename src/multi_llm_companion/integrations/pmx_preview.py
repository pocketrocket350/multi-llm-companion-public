"""第9段階のPMX簡易プレビュー。外部素材を保存・配布しない。"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

from multi_llm_companion.config import GesturePolicyConfig, MmdMappingConfig
from multi_llm_companion.domain import (
    AssistantResponse,
    ConversationState,
    Emotion,
    EmotionState,
    ExpressionIntent,
    ExpressionPolicy,
)

from .base import (
    AvatarBone,
    AvatarControlPort,
    AvatarMaterialMorph,
    AvatarModelPreview,
    AvatarPhysicsJoint,
    AvatarRigidBody,
    AvatarTexturedTriangle,
    AvatarVmdCompatibility,
    AvatarVmdMotion,
    MmdAvatarCommand,
)
from .expression_morphs import ExpressionMorphResolver
from .gesture_selection import GestureCandidate, GestureSelection, GestureSelector
from .vmd_catalog import (
    ResolvedVmdCatalogMotion,
    VmdCatalogError,
    diagnose_vmd_model_compatibility,
    validate_conversation_vmd_tracks,
    validate_vmd_model_compatibility,
)
from .vmd_motion import VmdMotionError, read_vmd_motion, write_confirmation_vmd

_MAX_FILE_SIZE = 200 * 1024 * 1024
_MAX_VERTEX_COUNT = 1_000_000
_MAX_INDEX_COUNT = 9_000_000
_MAX_TEXT_SIZE = 1_000_000
_MAX_BONE_COUNT = 200_000
_MESH_VERTEX_LIMIT = 150_000
_PREVIEW_POINT_COUNT = 2_000
_PREVIEW_TRIANGLE_COUNT = 12_000
_TEXTURED_TRIANGLE_COUNT = 3_000
_FIXED_STRENGTH_EXPRESSION_MORPHS = frozenset(
    {
        "ハート目",
        "きらきら目",
        "ファイア目",
        "ハイライト拡大",
        "ハイライト消",
        "歯消し",
        "照れ",
        "涙",
        "汗",
        "ムカッ",
        "青ざめ",
    }
)


def _compatibility_status_text(motion: AvatarVmdMotion | None) -> str:
    """状態欄へVMDの対応数と未検出・未対応を短く示す。"""

    if motion is None or motion.compatibility is None:
        return ""
    report = motion.compatibility
    bone_total = len(report.supported_bones) + len(report.missing_bones)
    morph_total = len(report.supported_morphs) + len(report.missing_morphs)
    ik_total = len(report.supported_iks) + len(report.missing_iks)
    parts = [
        f"ボーン {len(report.supported_bones)}/{bone_total}",
        f"モーフ {len(report.supported_morphs)}/{morph_total}",
        f"IK {len(report.supported_iks)}/{ik_total}",
        f"最大移動/身長 {report.translation_height_ratio:.0%}",
    ]
    if report.target_model_matches is False:
        parts.append("対象モデル名不一致")
    missing = (*report.missing_bones, *report.missing_morphs, *report.missing_iks)
    if missing:
        parts.append("未検出 " + "、".join(missing[:3]))
    if report.unsupported_sections:
        parts.append("未対応 " + "、".join(report.unsupported_sections))
    return " / VMD互換: " + "・".join(parts)


def _gesture_compatibility_status_text(
    reports: tuple[AvatarVmdCompatibility, ...],
) -> str:
    """会話VMD全体の対応数を一つの状態へ集約する。"""

    if not reports:
        return ""
    supported = sum(
        len(report.supported_bones) + len(report.supported_morphs) + len(report.supported_iks)
        for report in reports
    )
    missing_names = tuple(
        name
        for report in reports
        for name in (*report.missing_bones, *report.missing_morphs, *report.missing_iks)
    )
    unsupported = tuple(
        dict.fromkeys(section for report in reports for section in report.unsupported_sections)
    )
    parts = [f"対応 {supported}", f"未検出 {len(missing_names)}"]
    if missing_names:
        parts.append("名前 " + "、".join(missing_names[:3]))
    if unsupported:
        parts.append("未対応 " + "、".join(unsupported))
    return " / 会話VMD互換: " + "・".join(parts)


class PmxPreviewError(RuntimeError):
    """利用者へ安全に示せるPMX読み込みエラー。"""


class PmxPreviewAvatarController(AvatarControlPort):
    """PMXの形状だけを読み取り、会話命令を将来用に受け取る。"""

    def __init__(
        self,
        model_path: Path,
        mapping: MmdMappingConfig | None = None,
        selected_vmd: ResolvedVmdCatalogMotion | None = None,
        gesture_vmds: tuple[ResolvedVmdCatalogMotion, ...] = (),
        gesture_policy: GesturePolicyConfig | None = None,
        skip_invalid_gesture_vmds: bool = False,
        skip_invalid_selected_vmd: bool = False,
        gesture_library_issues: tuple[str, ...] = (),
        library_validation_vmds: tuple[ResolvedVmdCatalogMotion, ...] = (),
        audio_available: bool = False,
        expression_presets: dict[str, dict[str, float]] | None = None,
        emotion_presets: dict[str, str] | None = None,
        emotion_preset_variants: dict[str, list[str]] | None = None,
        thin_cloth_contact_enabled: bool = False,
        thin_cloth_source_body_patterns: tuple[str, ...] = (),
        thin_cloth_target_body_patterns: tuple[str, ...] = (),
        thin_cloth_maximum_correction_per_step: float = 0.008,
    ) -> None:
        self._model_path = model_path
        self._mapping = mapping or MmdMappingConfig()
        self._preview = read_pmx_preview(
            model_path,
            mouth_morph_names=self._mouth_morph_names(),
            emotion_morph_names=self._mapping.emotion_morphs,
        )
        self._expression_morphs = ExpressionMorphResolver(
            expression_presets=expression_presets,
            emotion_presets=emotion_presets,
            emotion_preset_variants=emotion_preset_variants,
            model_morphs=self._mapping.expression_morphs,
            legacy_emotion_morphs=self._mapping.emotion_morphs,
            fixed_strength_morphs=(
                morph_name
                for morph_name in _FIXED_STRENGTH_EXPRESSION_MORPHS
                if morph_name in self._preview.named_vertex_morphs
            ),
        )
        self._preview = replace(
            self._preview,
            thin_cloth_contact_enabled=thin_cloth_contact_enabled,
            thin_cloth_source_body_patterns=thin_cloth_source_body_patterns,
            thin_cloth_target_body_patterns=thin_cloth_target_body_patterns,
            thin_cloth_maximum_correction_per_step=thin_cloth_maximum_correction_per_step,
        )
        library_issues = list(gesture_library_issues)
        if selected_vmd is not None:
            try:
                motion = read_vmd_motion(selected_vmd.file_path, loop=selected_vmd.motion.loop)
                validate_vmd_model_compatibility(selected_vmd, self._preview, motion)
                motion = replace(
                    motion,
                    compatibility=diagnose_vmd_model_compatibility(self._preview, motion),
                )
                self._preview = replace(
                    self._preview,
                    vmd_motion=replace(
                        motion,
                        name=f"{selected_vmd.file_path.stem}（権利確認済み）",
                        intensity_scale=selected_vmd.motion.intensity_scale,
                    ),
                )
            except (VmdCatalogError, VmdMotionError) as exc:
                if not skip_invalid_selected_vmd:
                    raise
                library_issues.append(f"{selected_vmd.motion_id}: {exc}")
        gesture_candidates: list[GestureCandidate] = []
        gesture_compatibility: list[AvatarVmdCompatibility] = []
        for selected_gesture in gesture_vmds:
            try:
                gesture_motion = read_vmd_motion(
                    selected_gesture.file_path,
                    loop=selected_gesture.motion.loop,
                )
                validate_vmd_model_compatibility(
                    selected_gesture,
                    self._preview,
                    gesture_motion,
                )
                validate_conversation_vmd_tracks(
                    selected_gesture.motion_id,
                    gesture_motion,
                )
            except (VmdCatalogError, VmdMotionError) as exc:
                if not skip_invalid_gesture_vmds:
                    raise
                library_issues.append(f"{selected_gesture.motion_id}: {exc}")
                continue
            compatibility = diagnose_vmd_model_compatibility(self._preview, gesture_motion)
            unsupported_sections = list(compatibility.unsupported_sections)
            if gesture_motion.camera_frames:
                unsupported_sections.append("会話VMDカメラ")
            if gesture_motion.light_frames:
                unsupported_sections.append("会話VMD照明")
            gesture_motion = replace(
                gesture_motion,
                compatibility=replace(
                    compatibility,
                    unsupported_sections=tuple(dict.fromkeys(unsupported_sections)),
                ),
            )
            if gesture_motion.compatibility is not None:
                gesture_compatibility.append(gesture_motion.compatibility)
            gesture_candidates.append(
                GestureCandidate(
                    gesture_id=selected_gesture.motion_id,
                    intent=selected_gesture.motion.selection_action,
                    category=selected_gesture.motion.category,
                    body_mask=tuple(selected_gesture.motion.body_mask),
                    cooldown_seconds=selected_gesture.motion.cooldown_seconds,
                    intensity_scale=selected_gesture.motion.intensity_scale,
                    motion=replace(
                        gesture_motion,
                        name=f"{selected_gesture.file_path.stem}（会話ジェスチャー・権利確認済み）",
                    ),
                    motion_type=selected_gesture.motion.motion_type,
                    moods=tuple(selected_gesture.motion.moods),
                    energy=selected_gesture.motion.energy,
                    weight=selected_gesture.motion.weight,
                    audio_required=selected_gesture.motion.audio_required,
                )
            )
        idle_motions: list[AvatarVmdMotion] = []
        for validation_vmd in library_validation_vmds:
            try:
                validation_motion = read_vmd_motion(
                    validation_vmd.file_path,
                    loop=validation_vmd.motion.loop,
                )
                validate_vmd_model_compatibility(
                    validation_vmd,
                    self._preview,
                    validation_motion,
                )
                if validation_vmd.motion.motion_type == "idle":
                    idle_motions.append(
                        replace(
                            validation_motion,
                            name=validation_vmd.file_path.stem,
                            loop=False,
                            intensity_scale=validation_vmd.motion.intensity_scale,
                        )
                    )
            except (VmdCatalogError, VmdMotionError) as exc:
                library_issues.append(f"{validation_vmd.motion_id}: {exc}")
        self._preview = replace(self._preview, idle_motions=tuple(idle_motions))
        policy = gesture_policy or GesturePolicyConfig()
        self._gesture_selector = GestureSelector(
            gesture_candidates,
            enabled=policy.enabled,
            min_interval_seconds=policy.min_interval_seconds,
            same_gesture_cooldown_seconds=policy.same_gesture_cooldown_seconds,
            same_category_cooldown_seconds=policy.same_category_cooldown_seconds,
            max_large_per_minute=policy.max_large_per_minute,
            recent_motion_count=policy.recent_motion_count,
        )
        self._audio_available = audio_available
        self._expression_policy = ExpressionPolicy(
            supported_gesture_intents=tuple(candidate.intent for candidate in gesture_candidates),
            supported_gaze_intents=tuple(self._mapping.gaze_bones),
        )
        self._gesture_vmd_count = len(gesture_candidates)
        self._gesture_compatibility = tuple(gesture_compatibility)
        self._gesture_library_issues = tuple(library_issues)
        self.last_response: AssistantResponse | None = None
        self.last_expression_intent: ExpressionIntent | None = None
        self.last_emotion_state: EmotionState | None = None
        self._gesture_selection = GestureSelection()
        self._command = MmdAvatarCommand()

    @property
    def is_available(self) -> bool:
        return True

    @property
    def controller_name(self) -> str:
        return "PMXメッシュプレビュー"

    @property
    def status_text(self) -> str:
        gesture_text = f" / 会話VMD {self._gesture_vmd_count}件" if self._gesture_vmd_count else ""
        issue_text = (
            f" / 利用不可 {len(self._gesture_library_issues)}件"
            if self._gesture_library_issues
            else ""
        )
        compatibility_text = _compatibility_status_text(self._preview.vmd_motion)
        gesture_compatibility_text = _gesture_compatibility_status_text(self._gesture_compatibility)
        return (
            f"読込済み: {self._preview.model_name}"
            f"（{self._preview.vertex_count:,}頂点）{gesture_text}{issue_text}{compatibility_text}"
            f"{gesture_compatibility_text}"
        )

    @property
    def model_preview(self) -> AvatarModelPreview:
        return self._preview

    @property
    def mmd_command(self) -> MmdAvatarCommand:
        return self._command

    def expression_preview_command(self, emotion: str) -> MmdAvatarCommand:
        """会話や保存状態を変えず、実在モーフだけで表情を確認する。"""

        normalized_emotion = emotion.strip().lower()
        try:
            preview_emotion = Emotion(normalized_emotion)
        except ValueError as exc:
            raise ValueError(f"確認できない表情です: {emotion}") from exc
        intent = self._expression_policy.normalize(
            AssistantResponse(
                speech="表情確認",
                emotion=preview_emotion,
                emotion_intensity=1.0,
            )
        )
        outputs = intent.outputs
        assert outputs is not None
        resolved_expression = self._expression_morphs.resolve(
            normalized_emotion,
            outputs.expression.intensity,
            advance_variant=False,
        )
        return MmdAvatarCommand(
            emotion=normalized_emotion,
            emotion_intensity=outputs.expression.intensity,
            emotion_morph=resolved_expression.primary_morph,
            gaze="forward",
            expression_preset=resolved_expression.preset_name,
            expression_morphs=resolved_expression.morph_weights,
            expression_status_text="表情確認（会話へは保存しません）",
        )

    def apply_response(self, response: AssistantResponse) -> None:
        self.last_response = response
        intent = self._expression_policy.normalize(
            response,
            conversation_state=ConversationState.REACTING,
        )
        self.last_expression_intent = intent
        self._gesture_selection = self._gesture_selector.select(
            intent.gesture_intent,
            mood=intent.emotion.value,
            energy=intent.intensity,
            audio_available=self._audio_available,
        )
        self._update_command(
            emotion=intent.emotion.value,
            intensity=intent.intensity,
            gaze=intent.gaze_intent,
            intent=intent,
            advance_expression_variant=True,
        )

    def apply_emotion_state(self, state: EmotionState) -> None:
        self.last_emotion_state = state
        intent = self.last_expression_intent
        if intent is not None:
            intent = self._expression_policy.apply_emotion_state(
                intent,
                state.emotion,
                state.intensity,
            )
            self.last_expression_intent = intent
        self._update_command(
            emotion=state.emotion.value,
            intensity=state.intensity,
            gaze=intent.gaze_intent if intent is not None else "forward",
            intent=intent,
            advance_expression_variant=False,
        )

    def reset(self) -> None:
        self.last_response = None
        self.last_expression_intent = None
        self.last_emotion_state = None
        self._gesture_selection = GestureSelection()
        self._gesture_selector.reset()
        self._expression_morphs.reset()
        self._command = MmdAvatarCommand()

    def generate_confirmation_vmd(self) -> str:
        """現在のPMXにあるボーン名で、ローカル確認用VMDを作る。"""

        output_path = self._model_path.parent / "multi_llm_companion_確認用モーション.vmd"
        write_confirmation_vmd(
            output_path,
            tuple(bone.name for bone in self._preview.bones),
            self._preview.morph_names,
            self._preview.model_name,
        )
        motion = read_vmd_motion(output_path)
        self._preview = replace(
            self._preview,
            vmd_motion=replace(
                motion,
                compatibility=diagnose_vmd_model_compatibility(self._preview, motion),
            ),
        )
        return str(output_path)

    def _update_command(
        self,
        *,
        emotion: str,
        intensity: float,
        gaze: str,
        intent: ExpressionIntent | None = None,
        advance_expression_variant: bool = True,
    ) -> None:
        normalized_emotion = emotion.strip().lower() or "neutral"
        normalized_gaze = gaze.strip().lower() or "forward"
        selection = self._gesture_selection
        normalized_gesture = selection.gesture
        bone_names = {bone.name for bone in self._preview.bones}
        gaze_bone = _resolve_mapped_bone(
            self._mapping.gaze_bones,
            normalized_gaze,
            bone_names,
            mapping_fallbacks=("user", "forward", "left", "right", "up", "down"),
            bone_fallbacks=("両目", "右目", "左目", "頭"),
        )
        gesture_target = _resolve_mapped_bone(
            self._mapping.gesture_targets,
            normalized_gesture,
            bone_names,
            bone_fallbacks=("右腕", "左腕", "右手首", "左手首")
            if normalized_gesture == "wave"
            and selection.candidate is not None
            and selection.candidate.motion is None
            else (),
        )
        selected_candidate = selection.candidate
        outputs = intent.outputs if intent is not None else None
        expression_intensity = outputs.expression.intensity if outputs is not None else intensity
        gesture_intensity = outputs.gesture.intensity if outputs is not None else intensity
        gaze_intensity = outputs.gaze.intensity if outputs is not None else intensity
        natural_motion_intensity = (
            outputs.natural_motion.intensity if outputs is not None else intensity
        )
        resolved_expression = self._expression_morphs.resolve(
            normalized_emotion,
            expression_intensity,
            advance_variant=advance_expression_variant,
        )
        self._command = MmdAvatarCommand(
            emotion=normalized_emotion,
            emotion_intensity=max(0.0, min(1.0, expression_intensity)),
            emotion_morph=resolved_expression.primary_morph,
            gaze=normalized_gaze,
            gaze_bone=gaze_bone,
            gesture=normalized_gesture,
            gesture_target=gesture_target,
            gesture_motion_id=(
                selected_candidate.gesture_id if selected_candidate is not None else ""
            ),
            gesture_motion_type=(
                selected_candidate.motion_type if selected_candidate is not None else ""
            ),
            gesture_body_mask=(
                selected_candidate.body_mask if selected_candidate is not None else ()
            ),
            gesture_motion=(selected_candidate.motion if selected_candidate is not None else None),
            gesture_intensity_scale=(
                selected_candidate.intensity_scale if selected_candidate is not None else 1.0
            ),
            gesture_intensity=max(0.0, min(1.0, gesture_intensity)),
            gaze_intensity=max(0.0, min(1.0, gaze_intensity)),
            natural_motion_intensity=max(0.0, min(1.0, natural_motion_intensity)),
            head_bone=self._mapping.pose_bones.get("head", ""),
            upper_body_bone=self._mapping.pose_bones.get("upper_body", ""),
            expression_preset=resolved_expression.preset_name,
            expression_morphs=resolved_expression.morph_weights,
            expression_status_text=(outputs.visual_status_text if outputs is not None else ""),
        )

    def _mouth_morph_names(self) -> dict[str, str]:
        """設定がない場合も、一般的な母音名のモーフを安全に試す。"""

        defaults = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}
        return defaults | self._mapping.mouth_morphs


def _resolve_mapped_bone(
    mapping: dict[str, str],
    key: str,
    bone_names: set[str],
    *,
    mapping_fallbacks: tuple[str, ...] = (),
    bone_fallbacks: tuple[str, ...] = (),
) -> str:
    """設定名を検証し、一般的なMMDボーン名へ安全に補う。"""

    configured_names = tuple(
        name
        for mapping_key in (key, *mapping_fallbacks)
        if (name := mapping.get(mapping_key, "").strip())
    )
    if not bone_names:
        return configured_names[0] if configured_names else ""
    return next(
        (name for name in (*configured_names, *bone_fallbacks) if name in bone_names),
        "",
    )


def read_pmx_preview(
    path: Path,
    *,
    mouth_morph_names: dict[str, str] | None = None,
    emotion_morph_names: dict[str, str] | None = None,
) -> AvatarModelPreview:
    """PMXの頂点と間引いた三角形面を、上限付きで読み取る。"""

    if path.suffix.lower() != ".pmx":
        raise PmxPreviewError("MMDモデルは .pmx ファイルを選んでください。")
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise PmxPreviewError("指定されたPMXモデルが見つかりません。") from exc
    except OSError as exc:
        raise PmxPreviewError("PMXモデルの情報を読み取れません。") from exc
    if size <= 0:
        raise PmxPreviewError("PMXモデルが空です。")
    if size > _MAX_FILE_SIZE:
        raise PmxPreviewError("PMXモデルが大きすぎます（上限は200MBです）。")

    try:
        with path.open("rb") as stream:
            reader = _PmxReader(stream)
            if reader.read_exact(4) != b"PMX ":
                raise PmxPreviewError("PMX形式ではないため読み込めません。")
            version = reader.read_float()
            if version not in {2.0, 2.1}:
                raise PmxPreviewError("対応していないPMXバージョンです（2.0または2.1のみ対応）。")
            globals_count = reader.read_byte()
            if globals_count < 8:
                raise PmxPreviewError("PMXの基本情報が不足しています。")
            globals_data = reader.read_exact(globals_count)
            encoding = globals_data[0]
            additional_uv_count = globals_data[1]
            vertex_index_size = globals_data[2]
            texture_index_size = globals_data[3]
            material_index_size = globals_data[4]
            bone_index_size = globals_data[5]
            morph_index_size = globals_data[6]
            rigid_body_index_size = globals_data[7]
            if (
                encoding not in {0, 1}
                or additional_uv_count > 4
                or any(
                    size not in {1, 2, 4}
                    for size in (
                        vertex_index_size,
                        texture_index_size,
                        material_index_size,
                        bone_index_size,
                        morph_index_size,
                        rigid_body_index_size,
                    )
                )
            ):
                raise PmxPreviewError("PMXの基本情報が正しくありません。")

            model_name = reader.read_text(encoding)
            reader.read_text(encoding)
            reader.read_text(encoding)
            reader.read_text(encoding)

            vertex_count = reader.read_int()
            if not 0 < vertex_count <= _MAX_VERTEX_COUNT:
                raise PmxPreviewError("PMXの頂点数が大きすぎるか、正しくありません。")
            mesh_available = vertex_count <= _MESH_VERTEX_LIMIT
            step = max(1, math.ceil(vertex_count / _PREVIEW_POINT_COUNT))
            points: list[tuple[float, float, float]] = []
            normals: list[tuple[float, float, float]] = []
            uvs: list[tuple[float, float]] = []
            additional_uvs: list[tuple[tuple[float, float, float, float], ...]] = []
            vertex_weights: list[tuple[tuple[int, float], ...]] = []
            edge_scales: list[float] = []
            for number in range(vertex_count):
                position = reader.read_vector3()
                normal = reader.read_vector3()
                uv = reader.read_vector2()
                vertex_additional_uvs = tuple(
                    reader.read_vector4() for _ in range(additional_uv_count)
                )
                deform_type = reader.read_byte()
                weights = _read_vertex_weights(reader, deform_type, bone_index_size)
                edge_scale = max(0.0, reader.read_float())
                if mesh_available or number % step == 0:
                    points.append(position)
                if mesh_available:
                    normals.append(normal)
                    uvs.append(uv)
                    additional_uvs.append(vertex_additional_uvs)
                    vertex_weights.append(weights)
                    edge_scales.append(edge_scale)

            index_count = reader.read_int()
            if index_count < 0 or index_count > _MAX_INDEX_COUNT or index_count % 3 != 0:
                raise PmxPreviewError("PMXの面情報が正しくありません。")
            face_count = index_count // 3
            face_step = max(1, math.ceil(face_count / _PREVIEW_TRIANGLE_COUNT))
            texture_face_step = max(1, math.ceil(face_count / _TEXTURED_TRIANGLE_COUNT))
            triangles: list[tuple[int, int, int]] = []
            texture_faces: list[tuple[int, tuple[int, int, int]]] = []
            dense_texture_faces: list[tuple[int, tuple[int, int, int]]] = []
            for face_number in range(face_count):
                indexes = tuple(reader.read_vertex_index(vertex_index_size) for _ in range(3))
                if any(index >= vertex_count for index in indexes):
                    raise PmxPreviewError("PMXの面が存在しない頂点を参照しています。")
                if mesh_available and face_number % face_step == 0:
                    triangles.append(indexes)
                if mesh_available and face_number % texture_face_step == 0:
                    texture_faces.append((face_number, indexes))
                if mesh_available:
                    dense_texture_faces.append((face_number, indexes))

            texture_names = _read_textures(reader, encoding)
            materials = _read_materials(reader, encoding, texture_index_size)
            textured_triangles = _build_textured_triangles(
                path,
                texture_names,
                materials,
                texture_faces,
                uvs,
                normals,
                additional_uvs,
                edge_scales,
            )
            dense_textured_triangles = _build_textured_triangles(
                path,
                texture_names,
                materials,
                dense_texture_faces,
                uvs,
                normals,
                additional_uvs,
                edge_scales,
            )
            bones = _read_bones(reader, encoding, bone_index_size)
            (
                mouth_morphs,
                emotion_morphs,
                vertex_morph_names,
                named_vertex_morphs,
                morph_names,
                named_uv_morphs,
                named_material_morphs,
            ) = _read_vertex_morphs(
                reader,
                encoding,
                vertex_count,
                vertex_index_size,
                material_index_size,
                bone_index_size,
                morph_index_size,
                rigid_body_index_size,
                mouth_morph_names or {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"},
                emotion_morph_names or {},
            )
            _skip_display_frames(reader, encoding, bone_index_size, morph_index_size)
            rigid_bodies = _read_rigid_bodies(reader, encoding, bone_index_size)
            physics_joints = _read_physics_joints(reader, encoding, rigid_body_index_size)
    except PmxPreviewError:
        raise
    except (OSError, OverflowError, struct.error) as exc:
        raise PmxPreviewError("PMXモデルの読み込み中にエラーが発生しました。") from exc

    return AvatarModelPreview(
        model_name=model_name or path.stem,
        vertex_count=vertex_count,
        face_count=index_count // 3,
        points=tuple(points),
        normals=tuple(normals) if mesh_available else (),
        triangles=tuple(triangles),
        mouth_morphs=mouth_morphs,
        vertex_morph_names=vertex_morph_names,
        emotion_morphs=emotion_morphs,
        named_vertex_morphs=named_vertex_morphs,
        uvs=tuple(uvs) if mesh_available else (),
        additional_uvs=tuple(additional_uvs) if mesh_available else (),
        morph_names=morph_names,
        named_uv_morphs=named_uv_morphs,
        named_material_morphs=named_material_morphs,
        bones=tuple(bones) if mesh_available else (),
        vertex_weights=tuple(vertex_weights) if mesh_available else (),
        rigid_bodies=tuple(rigid_bodies) if mesh_available else (),
        physics_joints=tuple(physics_joints) if mesh_available else (),
        textured_triangles=tuple(textured_triangles) if mesh_available else (),
        dense_textured_triangles=tuple(dense_textured_triangles) if mesh_available else (),
    )


def _deform_size(deform_type: int, index_size: int) -> int:
    if deform_type == 0:
        return index_size
    if deform_type == 1:
        return index_size * 2 + 4
    if deform_type in {2, 4}:
        return index_size * 4 + 16
    if deform_type == 3:
        return index_size * 2 + 40
    raise PmxPreviewError("対応していないPMXの頂点変形です。")


@dataclass(frozen=True, slots=True)
class _PmxMaterial:
    texture_index: int
    diffuse: tuple[float, float, float, float]
    face_count: int
    name: str = ""
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shininess: float = 0.0
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    draw_flags: int = 0
    sphere_texture_index: int = -1
    sphere_mode: int = 0
    toon_texture_index: int = -1
    toon_shared_index: int = -1
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    edge_size: float = 0.0


def _read_textures(reader: _PmxReader, encoding: int) -> tuple[str, ...]:
    return tuple(
        reader.read_text(encoding) for _ in range(_read_section_count(reader, "テクスチャ"))
    )


def _read_materials(
    reader: _PmxReader,
    encoding: int,
    texture_index_size: int,
) -> tuple[_PmxMaterial, ...]:
    materials: list[_PmxMaterial] = []
    for _ in range(_read_section_count(reader, "材質")):
        name = reader.read_text(encoding)
        reader.read_text(encoding)
        diffuse = tuple(reader.read_float() for _ in range(4))
        specular = tuple(reader.read_float() for _ in range(3))
        shininess = reader.read_float()
        ambient = tuple(reader.read_float() for _ in range(3))
        draw_flags = reader.read_byte()
        edge_color = tuple(reader.read_float() for _ in range(4))
        edge_size = reader.read_float()
        texture_index = reader.read_signed_index(texture_index_size)
        sphere_texture_index = reader.read_signed_index(texture_index_size)
        sphere_mode = reader.read_byte()
        toon_flag = reader.read_byte()
        if toon_flag not in {0, 1}:
            raise PmxPreviewError("PMXのToon参照情報が正しくありません。")
        toon_texture_index = reader.read_signed_index(texture_index_size) if toon_flag == 0 else -1
        toon_shared_index = reader.read_byte() if toon_flag == 1 else -1
        reader.read_text(encoding)
        face_vertex_count = reader.read_int()
        if face_vertex_count < 0 or face_vertex_count % 3 != 0:
            raise PmxPreviewError("PMXの材質ごとの面情報が正しくありません。")
        materials.append(
            _PmxMaterial(
                texture_index=texture_index,
                diffuse=(diffuse[0], diffuse[1], diffuse[2], diffuse[3]),
                face_count=face_vertex_count // 3,
                name=name,
                specular=(specular[0], specular[1], specular[2]),
                shininess=shininess,
                ambient=(ambient[0], ambient[1], ambient[2]),
                draw_flags=draw_flags,
                sphere_texture_index=sphere_texture_index,
                sphere_mode=sphere_mode if sphere_mode in {0, 1, 2, 3} else 0,
                toon_texture_index=toon_texture_index,
                toon_shared_index=(
                    toon_shared_index if toon_flag == 1 and 0 <= toon_shared_index <= 9 else -1
                ),
                edge_color=(
                    edge_color[0],
                    edge_color[1],
                    edge_color[2],
                    edge_color[3],
                ),
                edge_size=max(0.0, edge_size),
            )
        )
    return tuple(materials)


def _build_textured_triangles(
    model_path: Path,
    texture_names: tuple[str, ...],
    materials: tuple[_PmxMaterial, ...],
    texture_faces: list[tuple[int, tuple[int, int, int]]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]] | None = None,
    additional_uvs: list[tuple[tuple[float, float, float, float], ...]] | None = None,
    edge_scales: list[float] | None = None,
) -> tuple[AvatarTexturedTriangle, ...]:
    """材質の面範囲とUV座標から、安全なローカルテクスチャ面を作る。"""

    if not materials or not texture_faces:
        return ()
    material_ranges: list[tuple[int, int, _PmxMaterial]] = []
    start = 0
    for material in materials:
        end = start + material.face_count
        material_ranges.append((start, end, material))
        start = end

    resolved_textures = tuple(_resolve_texture_path(model_path, name) for name in texture_names)
    explicit_back_materials = {
        material.name.casefold()[: -len("_back")]
        for material in materials
        if material.name.casefold().endswith("_back")
    }
    result: list[AvatarTexturedTriangle] = []
    material_position = 0
    for face_number, indexes in texture_faces:
        while (
            material_position < len(material_ranges)
            and face_number >= material_ranges[material_position][1]
        ):
            material_position += 1
        if material_position >= len(material_ranges):
            break
        start, end, material = material_ranges[material_position]
        if face_number < start:
            continue
        if material.texture_index >= len(resolved_textures):
            continue
        texture_path = (
            "" if material.texture_index < 0 else resolved_textures[material.texture_index]
        )
        has_explicit_back_material = material.name.casefold() in explicit_back_materials
        try:
            triangle_uvs = tuple(uvs[index] for index in indexes)
            triangle_normals = tuple(
                normals[index] if normals is not None else (0.0, 1.0, 0.0) for index in indexes
            )
            triangle_additional_uvs = tuple(
                additional_uvs[index] if additional_uvs is not None else () for index in indexes
            )
            triangle_edge_scales = tuple(
                edge_scales[index] if edge_scales is not None else 1.0 for index in indexes
            )
        except IndexError:
            continue
        result.append(
            AvatarTexturedTriangle(
                indexes=indexes,
                uvs=(triangle_uvs[0], triangle_uvs[1], triangle_uvs[2]),
                texture_path=texture_path,
                diffuse=material.diffuse,
                additional_uvs=(
                    triangle_additional_uvs[0],
                    triangle_additional_uvs[1],
                    triangle_additional_uvs[2],
                ),
                material_index=material_position,
                is_transparent=material.diffuse[3] < 0.999,
                normals=(triangle_normals[0], triangle_normals[1], triangle_normals[2]),
                sphere_texture_path=_material_texture_path(
                    resolved_textures, material.sphere_texture_index
                ),
                sphere_mode=material.sphere_mode,
                toon_texture_path=_toon_texture_path(
                    model_path,
                    resolved_textures,
                    material.toon_texture_index,
                    material.toon_shared_index,
                ),
                toon_shared_index=material.toon_shared_index,
                is_double_sided=bool(material.draw_flags & 0x01) and not has_explicit_back_material,
                specular=material.specular,
                shininess=material.shininess,
                ambient=material.ambient,
                draw_outline=bool(material.draw_flags & 0x10) and not has_explicit_back_material,
                edge_color=material.edge_color,
                edge_size=material.edge_size,
                edge_scales=(
                    triangle_edge_scales[0],
                    triangle_edge_scales[1],
                    triangle_edge_scales[2],
                ),
            )
        )
    return tuple(result)


def _material_texture_path(textures: tuple[str, ...], texture_index: int) -> str:
    if texture_index < 0 or texture_index >= len(textures):
        return ""
    return textures[texture_index]


def _toon_texture_path(
    model_path: Path,
    textures: tuple[str, ...],
    texture_index: int,
    shared_index: int,
) -> str:
    explicit = _material_texture_path(textures, texture_index)
    if explicit or not 0 <= shared_index <= 9:
        return explicit
    local_shared = Path(_resolve_texture_path(model_path, f"toon{shared_index + 1:02}.bmp"))
    return str(local_shared) if local_shared.is_file() else ""


def _resolve_texture_path(model_path: Path, texture_name: str) -> str:
    """PMXの相対パスだけを許可し、モデル外への参照を防ぐ。"""

    if not texture_name.strip():
        return ""
    root = model_path.parent.resolve()
    candidate = (root / Path(texture_name.replace("\\", "/"))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return ""
    return str(candidate)


def _read_vertex_weights(
    reader: _PmxReader,
    deform_type: int,
    bone_index_size: int,
) -> tuple[tuple[int, float], ...]:
    """BDEFとSDEFの基本ウェイトを、安全な骨番号と重みに整える。"""

    if deform_type == 0:
        return _normalized_weights(((reader.read_signed_index(bone_index_size), 1.0),))
    if deform_type == 1:
        first = reader.read_signed_index(bone_index_size)
        second = reader.read_signed_index(bone_index_size)
        first_weight = reader.read_float()
        return _normalized_weights(((first, first_weight), (second, 1.0 - first_weight)))
    if deform_type in {2, 4}:
        bones = tuple(reader.read_signed_index(bone_index_size) for _ in range(4))
        weights = tuple(reader.read_float() for _ in range(4))
        return _normalized_weights(tuple(zip(bones, weights, strict=True)))
    if deform_type == 3:
        first = reader.read_signed_index(bone_index_size)
        second = reader.read_signed_index(bone_index_size)
        first_weight = reader.read_float()
        reader.skip(36)
        return _normalized_weights(((first, first_weight), (second, 1.0 - first_weight)))
    raise PmxPreviewError("対応していないPMXの頂点変形です。")


def _normalized_weights(
    raw_weights: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    valid = [
        (bone_index, weight)
        for bone_index, weight in raw_weights
        if bone_index >= 0 and math.isfinite(weight) and weight > 0.0
    ]
    total = sum(weight for _bone_index, weight in valid)
    if total <= 0.0:
        return ()
    return tuple((bone_index, weight / total) for bone_index, weight in valid)


def _read_bones(
    reader: _PmxReader,
    encoding: int,
    bone_index_size: int,
) -> tuple[AvatarBone, ...]:
    """PMXの変形順、付与変形、軸制約、IK制限を含むボーン情報を読む。"""

    count = _read_section_count(reader, "ボーン")
    if count > _MAX_BONE_COUNT:
        raise PmxPreviewError("PMXのボーン数が大きすぎます。")
    raw_bones: list[AvatarBone] = []
    for _ in range(count):
        name = reader.read_text(encoding)
        reader.read_text(encoding)
        position = reader.read_vector3()
        parent_index = reader.read_signed_index(bone_index_size)
        deform_layer = reader.read_int()
        flags = reader.read_ushort()
        reader.skip(bone_index_size if flags & 0x0001 else 12)
        inherit_parent_index = -1
        inherit_weight = 0.0
        if flags & 0x0300:
            inherit_parent_index = reader.read_signed_index(bone_index_size)
            inherit_weight = reader.read_float()
        fixed_axis = None
        if flags & 0x0400:
            fixed_axis = _safe_axis(reader.read_vector3())
        local_axis_x = None
        local_axis_z = None
        if flags & 0x0800:
            local_axis_x = _safe_axis(reader.read_vector3())
            local_axis_z = _safe_axis(reader.read_vector3())
        if flags & 0x2000:
            reader.skip(4)
        ik_target_index = -1
        ik_iterations = 0
        ik_limit_radians = 0.0
        ik_links: tuple[int, ...] = ()
        ik_link_limits: tuple[
            tuple[tuple[float, float, float], tuple[float, float, float]] | None, ...
        ] = ()
        if flags & 0x0020:
            ik_target_index = reader.read_signed_index(bone_index_size)
            ik_iterations = reader.read_int()
            ik_limit_radians = reader.read_float()
            links: list[int] = []
            link_limits: list[
                tuple[tuple[float, float, float], tuple[float, float, float]] | None
            ] = []
            for _ in range(_read_section_count(reader, "IKリンク")):
                link_index = reader.read_signed_index(bone_index_size)
                if reader.read_byte():
                    minimum = reader.read_vector3()
                    maximum = reader.read_vector3()
                    link_limits.append(_safe_angle_limits(minimum, maximum))
                else:
                    link_limits.append(None)
                links.append(link_index)
            ik_links = tuple(links)
            ik_link_limits = tuple(link_limits)
        raw_bones.append(
            AvatarBone(
                name=name,
                parent_index=parent_index,
                position=position,
                ik_target_index=ik_target_index,
                ik_iterations=max(0, min(ik_iterations, 256)),
                ik_limit_radians=max(0.0, min(ik_limit_radians, math.pi)),
                ik_links=ik_links,
                deform_layer=deform_layer,
                inherit_parent_index=inherit_parent_index,
                inherit_weight=max(-1.0, min(1.0, inherit_weight)),
                inherit_rotation=bool(flags & 0x0100),
                inherit_translation=bool(flags & 0x0200),
                fixed_axis=fixed_axis,
                local_axis_x=local_axis_x,
                local_axis_z=local_axis_z,
                after_physics=bool(flags & 0x1000),
                allows_rotation=bool(flags & 0x0002),
                allows_translation=bool(flags & 0x0004),
                ik_link_limits=ik_link_limits,
            )
        )

    normalized: list[AvatarBone] = []
    for index, bone in enumerate(raw_bones):
        valid_links: list[int] = []
        valid_limits: list[
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ] = []
        for link_position, link_index in enumerate(bone.ik_links):
            if not 0 <= link_index < count:
                continue
            valid_links.append(link_index)
            valid_limits.append(
                bone.ik_link_limits[link_position]
                if link_position < len(bone.ik_link_limits)
                else None
            )
        normalized.append(
            AvatarBone(
                name=bone.name,
                parent_index=(
                    bone.parent_index
                    if 0 <= bone.parent_index < count and bone.parent_index != index
                    else -1
                ),
                position=bone.position,
                ik_target_index=(bone.ik_target_index if 0 <= bone.ik_target_index < count else -1),
                ik_iterations=bone.ik_iterations,
                ik_limit_radians=bone.ik_limit_radians,
                ik_links=tuple(valid_links),
                deform_layer=bone.deform_layer,
                inherit_parent_index=(
                    bone.inherit_parent_index
                    if 0 <= bone.inherit_parent_index < count and bone.inherit_parent_index != index
                    else -1
                ),
                inherit_weight=bone.inherit_weight,
                inherit_rotation=bone.inherit_rotation,
                inherit_translation=bone.inherit_translation,
                fixed_axis=bone.fixed_axis,
                local_axis_x=bone.local_axis_x,
                local_axis_z=bone.local_axis_z,
                after_physics=bone.after_physics,
                allows_rotation=bone.allows_rotation,
                allows_translation=bone.allows_translation,
                ik_link_limits=tuple(valid_limits),
            )
        )
    return tuple(normalized)


def _safe_axis(value: tuple[float, float, float]) -> tuple[float, float, float] | None:
    """PMXの軸を有限な単位ベクトルへ整える。"""

    if not all(math.isfinite(item) for item in value):
        return None
    size = math.sqrt(sum(item * item for item in value))
    if size < 0.000001:
        return None
    return tuple(item / size for item in value)  # type: ignore[return-value]


def _safe_angle_limits(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """IK角度制限を有限な最小値・最大値へ整える。"""

    if not all(math.isfinite(item) for item in (*minimum, *maximum)):
        return None
    lower = tuple(min(minimum[index], maximum[index]) for index in range(3))
    upper = tuple(max(minimum[index], maximum[index]) for index in range(3))
    return lower, upper  # type: ignore[return-value]


def _read_vertex_morphs(
    reader: _PmxReader,
    encoding: int,
    vertex_count: int,
    vertex_index_size: int,
    material_index_size: int,
    bone_index_size: int,
    morph_index_size: int,
    rigid_body_index_size: int,
    requested_mouth_names: dict[str, str],
    requested_emotion_names: dict[str, str],
) -> tuple[
    dict[str, tuple[tuple[int, float, float, float], ...]],
    dict[str, tuple[tuple[int, float, float, float], ...]],
    tuple[str, ...],
    dict[str, tuple[tuple[int, float, float, float], ...]],
    tuple[str, ...],
    dict[str, tuple[tuple[int, int, float, float, float, float], ...]],
    dict[str, tuple[AvatarMaterialMorph, ...]],
]:
    """頂点・UV・材質モーフをグループ経由分も含めて保持する。"""

    mouth_by_name = {name: shape for shape, name in requested_mouth_names.items() if name}
    emotion_by_name = {name: emotion for emotion, name in requested_emotion_names.items() if name}
    vertex_names: list[str] = []
    morph_names: list[str] = []
    morph_types: list[int] = []
    vertex_offsets: list[tuple[tuple[int, float, float, float], ...]] = []
    uv_offsets: list[tuple[tuple[int, int, float, float, float, float], ...]] = []
    material_offsets: list[tuple[AvatarMaterialMorph, ...]] = []
    group_offsets: list[tuple[tuple[int, float], ...]] = []
    for _ in range(_read_section_count(reader, "モーフ")):
        name = reader.read_text(encoding)
        reader.read_text(encoding)
        reader.skip(1)
        morph_type = reader.read_byte()
        offset_count = _read_section_count(reader, "モーフ要素")
        morph_names.append(name)
        morph_types.append(morph_type)
        if morph_type == 1:
            vertex_names.append(name)
            offsets: list[tuple[int, float, float, float]] = []
            for _ in range(offset_count):
                vertex_index = reader.read_vertex_index(vertex_index_size)
                if vertex_index >= vertex_count:
                    raise PmxPreviewError("PMXのモーフが存在しない頂点を参照しています。")
                offset = reader.read_vector3()
                if not all(math.isfinite(value) for value in offset):
                    raise PmxPreviewError("PMXのモーフ変形量が正しくありません。")
                offsets.append((vertex_index, *offset))
            vertex_offsets.append(tuple(offsets))
            uv_offsets.append(())
            material_offsets.append(())
            group_offsets.append(())
            continue
        if morph_type in {3, 4, 5, 6, 7}:
            offsets_uv: list[tuple[int, int, float, float, float, float]] = []
            for _ in range(offset_count):
                vertex_index = reader.read_vertex_index(vertex_index_size)
                if vertex_index >= vertex_count:
                    raise PmxPreviewError("PMXのUVモーフが存在しない頂点を参照しています。")
                offset = reader.read_vector4()
                if not all(math.isfinite(value) for value in offset):
                    raise PmxPreviewError("PMXのUVモーフ変形量が正しくありません。")
                offsets_uv.append((morph_type - 3, vertex_index, *offset))
            vertex_offsets.append(())
            uv_offsets.append(tuple(offsets_uv))
            material_offsets.append(())
            group_offsets.append(())
            continue
        if morph_type == 8:
            offsets_material = tuple(
                _read_material_morph(reader, material_index_size) for _ in range(offset_count)
            )
            vertex_offsets.append(())
            uv_offsets.append(())
            material_offsets.append(offsets_material)
            group_offsets.append(())
            continue
        if morph_type in {0, 9}:
            group_offsets.append(
                tuple(
                    (reader.read_signed_index(morph_index_size), reader.read_float())
                    for _ in range(offset_count)
                )
            )
            vertex_offsets.append(())
            uv_offsets.append(())
            material_offsets.append(())
            continue
        vertex_offsets.append(())
        uv_offsets.append(())
        material_offsets.append(())
        group_offsets.append(())
        _skip_morph_offsets(
            reader,
            morph_type,
            offset_count,
            vertex_index_size,
            material_index_size,
            bone_index_size,
            morph_index_size,
            rigid_body_index_size,
        )

    indexes_by_name: dict[str, int] = {}
    for index, name in enumerate(morph_names):
        indexes_by_name.setdefault(name, index)
    vertex_cache: dict[int, tuple[tuple[int, float, float, float], ...]] = {}
    uv_cache: dict[int, tuple[tuple[int, int, float, float, float, float], ...]] = {}
    material_cache: dict[int, tuple[AvatarMaterialMorph, ...]] = {}

    def resolved_vertex(name: str) -> tuple[tuple[int, float, float, float], ...]:
        index = indexes_by_name.get(name)
        if index is None:
            return ()
        return _resolve_vertex_morph(
            index,
            morph_types,
            vertex_offsets,
            group_offsets,
            vertex_cache,
            set(),
        )

    def resolved_uv(name: str) -> tuple[tuple[int, int, float, float, float, float], ...]:
        index = indexes_by_name.get(name)
        if index is None:
            return ()
        return _resolve_uv_morph(
            index,
            morph_types,
            uv_offsets,
            group_offsets,
            uv_cache,
            set(),
        )

    def resolved_material(name: str) -> tuple[AvatarMaterialMorph, ...]:
        index = indexes_by_name.get(name)
        if index is None:
            return ()
        return _resolve_material_morph(
            index,
            morph_types,
            material_offsets,
            group_offsets,
            material_cache,
            set(),
        )

    found_mouth = {
        shape: offsets
        for name, shape in mouth_by_name.items()
        if (offsets := resolved_vertex(name))
    }
    found_emotion = {
        emotion: offsets
        for name, emotion in emotion_by_name.items()
        if (offsets := resolved_vertex(name))
    }
    # VMDはモーフ名で表情を指定するため、頂点に展開できる分を名前付きで保持する。
    named_vertex_morphs = {
        name: offsets for name in dict.fromkeys(morph_names) if (offsets := resolved_vertex(name))
    }
    named_uv_morphs = {
        name: offsets for name in dict.fromkeys(morph_names) if (offsets := resolved_uv(name))
    }
    named_material_morphs = {
        name: offsets for name in dict.fromkeys(morph_names) if (offsets := resolved_material(name))
    }
    supported_names = tuple(
        name
        for name in dict.fromkeys(morph_names)
        if name in named_vertex_morphs or name in named_uv_morphs or name in named_material_morphs
    )
    return (
        found_mouth,
        found_emotion,
        tuple(vertex_names),
        named_vertex_morphs,
        supported_names,
        named_uv_morphs,
        named_material_morphs,
    )


def _read_material_morph(reader: _PmxReader, material_index_size: int) -> AvatarMaterialMorph:
    """PMX材質モーフ一要素を有限値として読み取る。"""

    material_index = reader.read_signed_index(material_index_size)
    operation = reader.read_byte()
    diffuse = reader.read_vector4()
    specular = reader.read_vector3()
    shininess = reader.read_float()
    ambient = reader.read_vector3()
    edge_color = reader.read_vector4()
    edge_size = reader.read_float()
    texture_factor = reader.read_vector4()
    sphere_factor = reader.read_vector4()
    toon_factor = reader.read_vector4()
    values = (
        *diffuse,
        *specular,
        shininess,
        *ambient,
        *edge_color,
        edge_size,
        *texture_factor,
        *sphere_factor,
        *toon_factor,
    )
    if (
        operation not in {0, 1}
        or material_index < -1
        or not all(math.isfinite(value) for value in values)
    ):
        raise PmxPreviewError("PMXの材質モーフ情報が正しくありません。")
    return AvatarMaterialMorph(
        material_index=material_index,
        operation=operation,
        diffuse=diffuse,
        specular=specular,
        shininess=shininess,
        ambient=ambient,
        edge_color=edge_color,
        edge_size=edge_size,
        texture_factor=texture_factor,
        sphere_factor=sphere_factor,
        toon_factor=toon_factor,
    )


def _resolve_vertex_morph(
    index: int,
    morph_types: list[int],
    vertex_offsets: list[tuple[tuple[int, float, float, float], ...]],
    group_offsets: list[tuple[tuple[int, float], ...]],
    cache: dict[int, tuple[tuple[int, float, float, float], ...]],
    visiting: set[int],
) -> tuple[tuple[int, float, float, float], ...]:
    """グループモーフを再帰的に展開し、頂点ごとの差分へまとめる。"""

    if index < 0 or index >= len(morph_types) or index in visiting:
        return ()
    if index in cache:
        return cache[index]
    if morph_types[index] == 1:
        cache[index] = vertex_offsets[index]
        return cache[index]
    if morph_types[index] not in {0, 9}:
        cache[index] = ()
        return ()

    combined: dict[int, tuple[float, float, float]] = {}
    nested_visiting = visiting | {index}
    for referenced_index, weight in group_offsets[index]:
        if not math.isfinite(weight):
            continue
        safe_weight = max(-10.0, min(10.0, weight))
        for vertex_index, offset_x, offset_y, offset_z in _resolve_vertex_morph(
            referenced_index,
            morph_types,
            vertex_offsets,
            group_offsets,
            cache,
            nested_visiting,
        ):
            current_x, current_y, current_z = combined.get(
                vertex_index,
                (0.0, 0.0, 0.0),
            )
            combined[vertex_index] = (
                current_x + offset_x * safe_weight,
                current_y + offset_y * safe_weight,
                current_z + offset_z * safe_weight,
            )
    cache[index] = tuple((vertex_index, *offset) for vertex_index, offset in combined.items())
    return cache[index]


def _resolve_uv_morph(
    index: int,
    morph_types: list[int],
    uv_offsets: list[tuple[tuple[int, int, float, float, float, float], ...]],
    group_offsets: list[tuple[tuple[int, float], ...]],
    cache: dict[int, tuple[tuple[int, int, float, float, float, float], ...]],
    visiting: set[int],
) -> tuple[tuple[int, int, float, float, float, float], ...]:
    """グループモーフをUVチャンネル別の頂点差分へ展開する。"""

    if index < 0 or index >= len(morph_types) or index in visiting:
        return ()
    if index in cache:
        return cache[index]
    if morph_types[index] in {3, 4, 5, 6, 7}:
        cache[index] = uv_offsets[index]
        return cache[index]
    if morph_types[index] not in {0, 9}:
        cache[index] = ()
        return ()
    combined: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    nested_visiting = visiting | {index}
    for referenced_index, weight in group_offsets[index]:
        if not math.isfinite(weight):
            continue
        safe_weight = max(-10.0, min(10.0, weight))
        for channel, vertex_index, x, y, z, w in _resolve_uv_morph(
            referenced_index,
            morph_types,
            uv_offsets,
            group_offsets,
            cache,
            nested_visiting,
        ):
            current = combined.get((channel, vertex_index), (0.0, 0.0, 0.0, 0.0))
            combined[(channel, vertex_index)] = tuple(
                current[axis] + value * safe_weight for axis, value in enumerate((x, y, z, w))
            )  # type: ignore[assignment]
    cache[index] = tuple(
        (channel, vertex_index, *offset) for (channel, vertex_index), offset in combined.items()
    )
    return cache[index]


def _resolve_material_morph(
    index: int,
    morph_types: list[int],
    material_offsets: list[tuple[AvatarMaterialMorph, ...]],
    group_offsets: list[tuple[tuple[int, float], ...]],
    cache: dict[int, tuple[AvatarMaterialMorph, ...]],
    visiting: set[int],
) -> tuple[AvatarMaterialMorph, ...]:
    """材質モーフを順序を保ったままグループの率付きで展開する。"""

    if index < 0 or index >= len(morph_types) or index in visiting:
        return ()
    if index in cache:
        return cache[index]
    if morph_types[index] == 8:
        cache[index] = material_offsets[index]
        return cache[index]
    if morph_types[index] not in {0, 9}:
        cache[index] = ()
        return ()
    combined: list[AvatarMaterialMorph] = []
    nested_visiting = visiting | {index}
    for referenced_index, weight in group_offsets[index]:
        if not math.isfinite(weight):
            continue
        safe_weight = max(-10.0, min(10.0, weight))
        combined.extend(
            replace(item, weight_scale=item.weight_scale * safe_weight)
            for item in _resolve_material_morph(
                referenced_index,
                morph_types,
                material_offsets,
                group_offsets,
                cache,
                nested_visiting,
            )
        )
    cache[index] = tuple(combined)
    return cache[index]


def _skip_morph_offsets(
    reader: _PmxReader,
    morph_type: int,
    offset_count: int,
    vertex_index_size: int,
    material_index_size: int,
    bone_index_size: int,
    morph_index_size: int,
    rigid_body_index_size: int,
) -> None:
    sizes = {
        0: morph_index_size + 4,
        2: bone_index_size + 28,
        3: vertex_index_size + 16,
        4: vertex_index_size + 16,
        5: vertex_index_size + 16,
        6: vertex_index_size + 16,
        7: vertex_index_size + 16,
        8: material_index_size + 113,
        9: morph_index_size + 4,
        10: rigid_body_index_size + 25,
    }
    try:
        reader.skip(sizes[morph_type] * offset_count)
    except KeyError as exc:
        raise PmxPreviewError("対応していないPMXモーフ形式です。") from exc


def _skip_display_frames(
    reader: _PmxReader,
    encoding: int,
    bone_index_size: int,
    morph_index_size: int,
) -> None:
    """物理情報の前にある表示枠を、内容を保持せず安全に読み飛ばす。"""

    for _ in range(_read_section_count(reader, "表示枠")):
        reader.read_text(encoding)
        reader.read_text(encoding)
        reader.skip(1)
        for _ in range(_read_section_count(reader, "表示枠要素")):
            element_type = reader.read_byte()
            if element_type == 0:
                reader.skip(bone_index_size)
            elif element_type == 1:
                reader.skip(morph_index_size)
            else:
                raise PmxPreviewError("PMXの表示枠要素が正しくありません。")


def _read_rigid_bodies(
    reader: _PmxReader,
    encoding: int,
    bone_index_size: int,
) -> tuple[AvatarRigidBody, ...]:
    """PMX剛体の形状、衝突条件、物理係数を読む。"""

    bodies: list[AvatarRigidBody] = []
    for _ in range(_read_section_count(reader, "剛体")):
        name = reader.read_text(encoding)
        reader.read_text(encoding)
        bone_index = reader.read_signed_index(bone_index_size)
        collision_group = reader.read_byte()
        non_collision_mask = reader.read_ushort()
        shape = reader.read_byte()
        size = reader.read_vector3()
        position = reader.read_vector3()
        rotation = reader.read_vector3()
        mass = reader.read_float()
        linear_damping = reader.read_float()
        angular_damping = reader.read_float()
        restitution = reader.read_float()
        friction = reader.read_float()
        mode = reader.read_byte()
        values = (
            *size,
            *position,
            *rotation,
            mass,
            linear_damping,
            angular_damping,
            restitution,
            friction,
        )
        if not all(math.isfinite(value) for value in values):
            raise PmxPreviewError("PMXの剛体情報が正しくありません。")
        bodies.append(
            AvatarRigidBody(
                bone_index=bone_index,
                position=position,
                mass=max(0.001, min(mass, 1000.0)),
                linear_damping=max(0.0, min(linear_damping, 1.0)),
                mode=mode if mode in {0, 1, 2} else 0,
                name=name,
                collision_group=max(0, min(collision_group, 15)),
                non_collision_mask=non_collision_mask & 0xFFFF,
                shape=shape if shape in {0, 1, 2} else 0,
                size=tuple(max(0.0001, min(abs(value), 1000.0)) for value in size),
                rotation=tuple(max(-math.pi, min(value, math.pi)) for value in rotation),
                angular_damping=max(0.0, min(angular_damping, 1.0)),
                restitution=max(0.0, min(restitution, 1.0)),
                friction=max(0.0, min(friction, 1.0)),
            )
        )
    return tuple(bodies)


def _read_physics_joints(
    reader: _PmxReader,
    encoding: int,
    rigid_body_index_size: int,
) -> tuple[AvatarPhysicsJoint, ...]:
    """PMXの6自由度ばねジョイントと移動・回転制限を読む。"""

    joints: list[AvatarPhysicsJoint] = []
    for _ in range(_read_section_count(reader, "ジョイント")):
        name = reader.read_text(encoding)
        reader.read_text(encoding)
        joint_type = reader.read_byte()
        rigid_body_a = reader.read_signed_index(rigid_body_index_size)
        rigid_body_b = reader.read_signed_index(rigid_body_index_size)
        position = reader.read_vector3()
        rotation = reader.read_vector3()
        translation_min = reader.read_vector3()
        translation_max = reader.read_vector3()
        rotation_min = reader.read_vector3()
        rotation_max = reader.read_vector3()
        spring_translation = reader.read_vector3()
        spring_rotation = reader.read_vector3()
        if joint_type != 0:
            continue
        values = (
            *position,
            *rotation,
            *translation_min,
            *translation_max,
            *rotation_min,
            *rotation_max,
            *spring_translation,
            *spring_rotation,
        )
        if not all(math.isfinite(value) for value in values):
            raise PmxPreviewError("PMXのジョイント情報が正しくありません。")
        safe_translation_min, safe_translation_max = _safe_vector_limits(
            translation_min, translation_max, 1000.0
        )
        safe_rotation_min, safe_rotation_max = _safe_vector_limits(
            rotation_min, rotation_max, math.pi
        )
        joints.append(
            AvatarPhysicsJoint(
                rigid_body_a=rigid_body_a,
                rigid_body_b=rigid_body_b,
                position=position,
                spring_translation=tuple(
                    max(0.0, min(abs(value), 100.0)) for value in spring_translation
                ),
                name=name,
                rotation=tuple(max(-math.pi, min(value, math.pi)) for value in rotation),
                translation_min=safe_translation_min,
                translation_max=safe_translation_max,
                rotation_min=safe_rotation_min,
                rotation_max=safe_rotation_max,
                spring_rotation=tuple(
                    max(0.0, min(abs(value), 100.0)) for value in spring_rotation
                ),
            )
        )
    return tuple(joints)


def _safe_vector_limits(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    limit: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """最小・最大が逆のPMXでも、安全な範囲へ並べ直す。"""

    lower = tuple(
        max(-limit, min(limit, min(minimum[index], maximum[index]))) for index in range(3)
    )
    upper = tuple(
        max(-limit, min(limit, max(minimum[index], maximum[index]))) for index in range(3)
    )
    return lower, upper  # type: ignore[return-value]


def _read_section_count(reader: _PmxReader, label: str) -> int:
    count = reader.read_int()
    if count < 0 or count > _MAX_INDEX_COUNT:
        raise PmxPreviewError(f"PMXの{label}数が正しくありません。")
    return count


class _PmxReader:
    """PMX読込で巨大なデータを確保しない小さなバイナリ読込器。"""

    def __init__(self, stream: object) -> None:
        self._stream = stream

    def read_exact(self, size: int) -> bytes:
        if size < 0 or size > _MAX_FILE_SIZE:
            raise PmxPreviewError("PMXのデータサイズが正しくありません。")
        data = self._stream.read(size)  # type: ignore[attr-defined]
        if len(data) != size:
            raise PmxPreviewError("PMXモデルの途中でデータが終わっています。")
        return data

    def skip(self, size: int) -> None:
        self.read_exact(size)

    def read_byte(self) -> int:
        return self.read_exact(1)[0]

    def read_int(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def read_ushort(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def read_float(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def read_vector2(self) -> tuple[float, float]:
        return struct.unpack("<2f", self.read_exact(8))

    def read_vector3(self) -> tuple[float, float, float]:
        return struct.unpack("<3f", self.read_exact(12))

    def read_vector4(self) -> tuple[float, float, float, float]:
        return struct.unpack("<4f", self.read_exact(16))

    def read_vertex_index(self, size: int) -> int:
        formats = {1: "<B", 2: "<H", 4: "<I"}
        try:
            return int(struct.unpack(formats[size], self.read_exact(size))[0])
        except KeyError as exc:
            raise PmxPreviewError("PMXの頂点番号サイズが正しくありません。") from exc

    def read_signed_index(self, size: int) -> int:
        formats = {1: "<b", 2: "<h", 4: "<i"}
        try:
            return int(struct.unpack(formats[size], self.read_exact(size))[0])
        except KeyError as exc:
            raise PmxPreviewError("PMXのモーフ番号サイズが正しくありません。") from exc

    def read_text(self, encoding: int) -> str:
        size = self.read_int()
        if size < 0 or size > _MAX_TEXT_SIZE:
            raise PmxPreviewError("PMXの文字列サイズが正しくありません。")
        raw = self.read_exact(size)
        codec = "utf-16-le" if encoding == 0 else "utf-8"
        try:
            return raw.decode(codec)
        except UnicodeDecodeError as exc:
            raise PmxPreviewError("PMX内の文字列を読み取れません。") from exc
