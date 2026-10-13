"""第44段階のPMX物理基準を、権利素材を出力せず測定する。"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from multi_llm_companion.integrations.base import AvatarModelPreview
from multi_llm_companion.ui.bullet_pmx_physics import (
    BulletPmxPhysicsSimulator,
    _authored_collision,
)
from multi_llm_companion.ui.models import (
    UiAvatarBone,
    UiAvatarModelPreview,
    UiAvatarPhysicsJoint,
    UiAvatarRigidBody,
    UiVmdBoneFrame,
    UiVmdMotion,
)
from multi_llm_companion.ui.pmx_pose import (
    bone_transforms,
    prepare_ik_solver,
    solve_ik_rotations,
)
from multi_llm_companion.ui.vmd_player import VmdMotionPlayer

type Vector3 = tuple[float, float, float]

_APPLICATION_CADENCE_SECONDS = 1.0 / 30.0
_FRAME_BUDGET_MS = 1000.0 / 30.0
_SETTLE_HOLD_SECONDS = 0.6


@dataclass(frozen=True, slots=True)
class PhysicsScenarioFrame:
    """診断の一時刻でBulletへ渡すボーン姿勢。"""

    rotations: dict[int, Vector3]
    translations: dict[int, Vector3]


@dataclass(frozen=True, slots=True)
class PhysicsScenario:
    """再現可能な入力列と、動作停止時刻。"""

    name: str
    frames: tuple[PhysicsScenarioFrame, ...]
    delta_seconds: float
    physics_enabled: bool = True
    settle_from_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PhysicsScenarioResult:
    """一場面の性能・安定性・収束測定結果。"""

    name: str
    physics_enabled: bool
    update_count: int
    simulated_seconds: float
    initialization_ms: float
    mean_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float
    over_33ms_count: int
    over_cadence_count: int
    maximum_consecutive_over_33ms: int
    maximum_consecutive_over_cadence: int
    maximum_body_offset: float
    maximum_bone_translation: float
    settling_seconds: float | None
    initialization_stabilization_reset_count: int
    initialization_non_finite_reset_count: int
    initialization_offset_limit_reset_count: int
    stabilization_reset_count: int
    non_finite_reset_count: int
    offset_limit_reset_count: int
    non_finite_count: int
    dropped_seconds: float
    maximum_substeps: int
    maximum_auxiliary_hand_contacts: int


@dataclass(frozen=True, slots=True)
class PhysicsCollisionCoverage:
    """作者マスクと、応答なし仮診断の候補対数。"""

    hand_arm_sleeve_to_skirt_pairs: int
    authored_colliding_pairs: int
    diagnostic_only_excluded_pairs: int


@dataclass(frozen=True, slots=True)
class PhysicsBaselineReport:
    """公開可能な構造値と測定値だけを持つ第44段階結果。"""

    cadence_seconds: float
    frame_budget_ms: float
    rigid_body_count: int
    dynamic_body_count: int
    joint_count: int
    model_height: float
    vmd_configured: bool
    vmd_frame_count: int
    vmd_bone_frame_count: int
    collision_coverage: PhysicsCollisionCoverage
    scenarios: tuple[PhysicsScenarioResult, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def to_physics_ui_preview(preview: AvatarModelPreview) -> UiAvatarModelPreview:
    """読込済みPMXから、物理診断に必要な情報だけを画面型へ移す。"""

    vmd_motion = None
    if preview.vmd_motion is not None:
        vmd_motion = UiVmdMotion(
            name="設定済みVMD",
            frame_count=preview.vmd_motion.frame_count,
            bone_frames=tuple(
                UiVmdBoneFrame(
                    bone_name=frame.bone_name,
                    frame_number=frame.frame_number,
                    translation=frame.translation,
                    rotation=frame.rotation,
                    interpolation=frame.interpolation,
                )
                for frame in preview.vmd_motion.bone_frames
            ),
            loop=preview.vmd_motion.loop,
            intensity_scale=preview.vmd_motion.intensity_scale,
        )
    return UiAvatarModelPreview(
        model_name="ローカルPMX",
        vertex_count=preview.vertex_count,
        face_count=preview.face_count,
        points=preview.points,
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
        vmd_motion=vmd_motion,
        thin_cloth_contact_enabled=preview.thin_cloth_contact_enabled,
        thin_cloth_source_body_patterns=preview.thin_cloth_source_body_patterns,
        thin_cloth_target_body_patterns=preview.thin_cloth_target_body_patterns,
        thin_cloth_maximum_correction_per_step=(preview.thin_cloth_maximum_correction_per_step),
    )


def measure_physics_baseline(
    preview: UiAvatarModelPreview,
    *,
    cadence_seconds: float = _APPLICATION_CADENCE_SECONDS,
) -> PhysicsBaselineReport:
    """現在設定のPMX/VMDを、固定した代表場面で測る。"""

    cadence = max(1.0 / 60.0, min(0.25, cadence_seconds))
    scenarios = _default_scenarios(preview, cadence)
    results = tuple(_measure_scenario(preview, scenario) for scenario in scenarios)
    motion = preview.vmd_motion
    return PhysicsBaselineReport(
        cadence_seconds=cadence,
        frame_budget_ms=_FRAME_BUDGET_MS,
        rigid_body_count=len(preview.rigid_bodies),
        dynamic_body_count=sum(body.mode != 0 for body in preview.rigid_bodies),
        joint_count=len(preview.physics_joints),
        model_height=_model_height(preview),
        vmd_configured=motion is not None,
        vmd_frame_count=motion.frame_count if motion is not None else 0,
        vmd_bone_frame_count=len(motion.bone_frames) if motion is not None else 0,
        collision_coverage=_collision_coverage(preview),
        scenarios=results,
    )


def _measure_scenario(
    preview: UiAvatarModelPreview,
    scenario: PhysicsScenario,
) -> PhysicsScenarioResult:
    physics = BulletPmxPhysicsSimulator()
    prepared_ik = prepare_ik_solver(preview)
    first = scenario.frames[0] if scenario.frames else PhysicsScenarioFrame({}, {})
    first_rotations = solve_ik_rotations(
        preview,
        first.rotations,
        {},
        first.translations,
        prepared_ik,
    )
    first_transforms = bone_transforms(preview, first_rotations, first.translations)
    started = time.perf_counter_ns()
    physics.step_pose(preview, 0.0, scenario.physics_enabled, first_transforms)
    initialization_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    initialization_resets = physics.stabilization_reset_count
    initialization_non_finite_resets = physics.non_finite_reset_count
    initialization_offset_limit_resets = physics.offset_limit_reset_count
    physics.reset_diagnostic_counters()

    durations: list[float] = []
    previous_pose: dict[tuple[str, int], Vector3] = {}
    maximum_bone_translation = 0.0
    non_finite_count = 0
    maximum_substeps = 0
    maximum_contacts = 0
    quiet_started_at: float | None = None
    settling_seconds: float | None = None
    elapsed = 0.0
    translation_threshold = max(0.00001, _model_height(preview) * 0.00001)
    for frame in scenario.frames:
        solved_rotations = solve_ik_rotations(
            preview,
            frame.rotations,
            {},
            frame.translations,
            prepared_ik,
        )
        transforms = bone_transforms(preview, solved_rotations, frame.translations)
        started = time.perf_counter_ns()
        pose = physics.step_pose(
            preview,
            scenario.delta_seconds,
            scenario.physics_enabled,
            transforms,
        )
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
        elapsed += scenario.delta_seconds
        maximum_substeps = max(maximum_substeps, physics.last_substeps)
        maximum_contacts = max(maximum_contacts, physics.auxiliary_hand_contact_count())
        current_pose = {
            **{("translation", index): value for index, value in pose.translations.items()},
            **{("rotation", index): value for index, value in pose.rotations.items()},
        }
        for kind_index, value in current_pose.items():
            if not all(math.isfinite(component) for component in value):
                non_finite_count += 1
                continue
            if kind_index[0] == "translation":
                maximum_bone_translation = max(maximum_bone_translation, _length(value))
        if scenario.settle_from_seconds is not None and elapsed >= scenario.settle_from_seconds:
            changed = _pose_change(previous_pose, current_pose, translation_threshold)
            if changed:
                quiet_started_at = None
            elif quiet_started_at is None:
                quiet_started_at = elapsed
            elif elapsed - quiet_started_at >= _SETTLE_HOLD_SECONDS:
                settling_seconds = quiet_started_at - scenario.settle_from_seconds
        previous_pose = current_pose

    cadence_budget_ms = scenario.delta_seconds * 1000.0
    return PhysicsScenarioResult(
        name=scenario.name,
        physics_enabled=scenario.physics_enabled,
        update_count=len(durations),
        simulated_seconds=elapsed,
        initialization_ms=round(initialization_ms, 3),
        mean_ms=_rounded_mean(durations),
        p95_ms=_rounded_percentile(durations, 0.95),
        p99_ms=_rounded_percentile(durations, 0.99),
        maximum_ms=round(max(durations, default=0.0), 3),
        over_33ms_count=sum(value > _FRAME_BUDGET_MS for value in durations),
        over_cadence_count=sum(value > cadence_budget_ms for value in durations),
        maximum_consecutive_over_33ms=_maximum_consecutive(
            value > _FRAME_BUDGET_MS for value in durations
        ),
        maximum_consecutive_over_cadence=_maximum_consecutive(
            value > cadence_budget_ms for value in durations
        ),
        maximum_body_offset=round(physics.maximum_body_offset_observed, 6),
        maximum_bone_translation=round(maximum_bone_translation, 6),
        settling_seconds=round(settling_seconds, 3) if settling_seconds is not None else None,
        initialization_stabilization_reset_count=initialization_resets,
        initialization_non_finite_reset_count=initialization_non_finite_resets,
        initialization_offset_limit_reset_count=initialization_offset_limit_resets,
        stabilization_reset_count=physics.stabilization_reset_count,
        non_finite_reset_count=physics.non_finite_reset_count,
        offset_limit_reset_count=physics.offset_limit_reset_count,
        non_finite_count=non_finite_count,
        dropped_seconds=round(physics.dropped_seconds, 6),
        maximum_substeps=maximum_substeps,
        maximum_auxiliary_hand_contacts=maximum_contacts,
    )


def _default_scenarios(
    preview: UiAvatarModelPreview,
    cadence: float,
) -> tuple[PhysicsScenario, ...]:
    height = _model_height(preview)
    driver = _driver_bone_index(preview)
    idle = _constant_frames(6.0, cadence)
    movement = _movement_frames(driver, height, cadence)
    rotation = _rotation_frames(driver, cadence)
    arm_down = _arm_down_frames(preview, cadence)
    hand_sweep = _hand_sweep_frames(preview, height, cadence)
    scenarios = [
        PhysicsScenario("通常待機", idle, cadence),
        PhysicsScenario("移動開始・停止後の収束", movement, cadence, settle_from_seconds=2.0),
        PhysicsScenario("身体の左右回転・停止", rotation, cadence, settle_from_seconds=4.0),
        PhysicsScenario("腕を身体の横へ下ろす", arm_down, cadence, settle_from_seconds=2.0),
        PhysicsScenario("手が前面・側面を横切る", hand_sweep, cadence, settle_from_seconds=3.0),
        PhysicsScenario("物理無効時の移動比較", movement, cadence, physics_enabled=False),
        PhysicsScenario("30Hz固定刻み基準", _constant_frames(4.0, 1.0 / 30.0), 1.0 / 30.0),
    ]
    if preview.vmd_motion is not None and preview.vmd_motion.bone_frames:
        duration = max(1.0 / 30.0, preview.vmd_motion.frame_count / 30.0)
        loop_seconds = duration + 0.75
        scenarios.extend(
            (
                PhysicsScenario(
                    "VMDループ境界",
                    _vmd_frames(preview, loop_seconds, cadence),
                    cadence,
                ),
                PhysicsScenario(
                    "VMD停止から通常待機へ復帰",
                    _vmd_stop_frames(preview, cadence, stop_at=min(3.0, duration * 0.5)),
                    cadence,
                    settle_from_seconds=min(3.0, duration * 0.5),
                ),
                PhysicsScenario(
                    "利用者割り込み相当の姿勢解除",
                    _vmd_stop_frames(preview, cadence, stop_at=min(1.2, duration * 0.25)),
                    cadence,
                    settle_from_seconds=min(1.2, duration * 0.25),
                ),
            )
        )
    return tuple(scenarios)


def _constant_frames(duration: float, cadence: float) -> tuple[PhysicsScenarioFrame, ...]:
    return tuple(PhysicsScenarioFrame({}, {}) for _ in range(_frame_count(duration, cadence)))


def _movement_frames(
    driver: int | None,
    height: float,
    cadence: float,
) -> tuple[PhysicsScenarioFrame, ...]:
    frames = []
    for index in range(_frame_count(8.0, cadence)):
        elapsed = index * cadence
        ratio = _smoothstep(max(0.0, min(1.0, (elapsed - 1.0) / 1.0)))
        translations = {driver: (height * 0.02 * ratio, 0.0, 0.0)} if driver is not None else {}
        frames.append(PhysicsScenarioFrame({}, translations))
    return tuple(frames)


def _rotation_frames(driver: int | None, cadence: float) -> tuple[PhysicsScenarioFrame, ...]:
    frames = []
    for index in range(_frame_count(9.0, cadence)):
        elapsed = index * cadence
        if elapsed < 1.0:
            angle = 0.0
        elif elapsed < 2.0:
            angle = 0.28 * _smoothstep(elapsed - 1.0)
        elif elapsed < 3.0:
            angle = 0.28 - 0.56 * _smoothstep(elapsed - 2.0)
        elif elapsed < 4.0:
            angle = -0.28 + 0.28 * _smoothstep(elapsed - 3.0)
        else:
            angle = 0.0
        rotations = {driver: (0.0, angle, 0.0)} if driver is not None else {}
        frames.append(PhysicsScenarioFrame(rotations, {}))
    return tuple(frames)


def _arm_down_frames(
    preview: UiAvatarModelPreview,
    cadence: float,
) -> tuple[PhysicsScenarioFrame, ...]:
    indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
    targets = {
        indexes[name]: rotation
        for name, rotation in (("右腕", 0.82), ("左腕", -0.82))
        if name in indexes
    }
    frames = []
    for index in range(_frame_count(8.0, cadence)):
        elapsed = index * cadence
        ratio = _smoothstep(max(0.0, min(1.0, (elapsed - 1.0) / 1.0)))
        rotations = {bone: (0.0, 0.0, value * ratio) for bone, value in targets.items()}
        frames.append(PhysicsScenarioFrame(rotations, {}))
    return tuple(frames)


def _hand_sweep_frames(
    preview: UiAvatarModelPreview,
    height: float,
    cadence: float,
) -> tuple[PhysicsScenarioFrame, ...]:
    indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
    hand = next((indexes[name] for name in ("右手首", "右腕") if name in indexes), None)
    frames = []
    for index in range(_frame_count(8.0, cadence)):
        elapsed = index * cadence
        progress = max(0.0, min(1.0, (elapsed - 1.0) / 2.0))
        offset = math.sin(math.pi * progress) * height * 0.035 if 0.0 < progress < 1.0 else 0.0
        translations = {hand: (-offset, 0.0, offset * 0.45)} if hand is not None else {}
        frames.append(PhysicsScenarioFrame({}, translations))
    return tuple(frames)


def _vmd_frames(
    preview: UiAvatarModelPreview,
    duration: float,
    cadence: float,
) -> tuple[PhysicsScenarioFrame, ...]:
    player = VmdMotionPlayer()
    frames = []
    for _ in range(_frame_count(duration, cadence)):
        rotations, translations, _morphs = player.advance(preview, cadence, True)
        frames.append(PhysicsScenarioFrame(rotations, translations))
    return tuple(frames)


def _vmd_stop_frames(
    preview: UiAvatarModelPreview,
    cadence: float,
    *,
    stop_at: float,
) -> tuple[PhysicsScenarioFrame, ...]:
    player = VmdMotionPlayer()
    frames = []
    duration = stop_at + 6.0
    for index in range(_frame_count(duration, cadence)):
        elapsed = index * cadence
        if elapsed < stop_at:
            rotations, translations, _morphs = player.advance(preview, cadence, True)
        else:
            rotations, translations = {}, {}
        frames.append(PhysicsScenarioFrame(rotations, translations))
    return tuple(frames)


def _collision_coverage(preview: UiAvatarModelPreview) -> PhysicsCollisionCoverage:
    candidate_pairs = 0
    authored_pairs = 0
    for first_index, first in enumerate(preview.rigid_bodies):
        for second in preview.rigid_bodies[first_index + 1 :]:
            categories = {_body_category(preview, first), _body_category(preview, second)}
            if categories != {"hand_arm_sleeve", "skirt"}:
                continue
            candidate_pairs += 1
            authored_pairs += int(_authored_collision(first, second))
    return PhysicsCollisionCoverage(
        hand_arm_sleeve_to_skirt_pairs=candidate_pairs,
        authored_colliding_pairs=authored_pairs,
        diagnostic_only_excluded_pairs=candidate_pairs - authored_pairs,
    )


def _body_category(preview: UiAvatarModelPreview, body: UiAvatarRigidBody) -> str:
    bone_name = (
        preview.bones[body.bone_index].name if 0 <= body.bone_index < len(preview.bones) else ""
    )
    name = f"{body.name} {bone_name}".casefold()
    if any(token in name for token in ("skirt", "スカート", "裙")):
        return "skirt"
    if any(token in name for token in ("hand", "wrist", "arm", "sleeve", "手", "腕", "ひじ", "袖")):
        return "hand_arm_sleeve"
    return "other"


def _driver_bone_index(preview: UiAvatarModelPreview) -> int | None:
    indexes = {bone.name: index for index, bone in enumerate(preview.bones)}
    for name in ("全ての親", "センター", "グルーブ", "腰", "下半身"):
        if name in indexes:
            return indexes[name]
    return next(
        (
            body.bone_index
            for body in preview.rigid_bodies
            if body.mode == 0 and 0 <= body.bone_index < len(preview.bones)
        ),
        0 if preview.bones else None,
    )


def _model_height(preview: UiAvatarModelPreview) -> float:
    if not preview.points:
        return 1.0
    ys = [point[1] for point in preview.points]
    return max(0.001, max(ys) - min(ys))


def _pose_change(
    previous: dict[tuple[str, int], Vector3],
    current: dict[tuple[str, int], Vector3],
    translation_threshold: float,
) -> bool:
    for key in previous.keys() | current.keys():
        before = previous.get(key, (0.0, 0.0, 0.0))
        after = current.get(key, (0.0, 0.0, 0.0))
        threshold = translation_threshold if key[0] == "translation" else 0.0001
        if _length(tuple(after[axis] - before[axis] for axis in range(3))) > threshold:
            return True
    return False


def _maximum_consecutive(values: Iterable[bool]) -> int:
    current = 0
    maximum = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _rounded_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _rounded_percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 3)


def _frame_count(duration: float, cadence: float) -> int:
    return max(1, math.ceil(duration / cadence))


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _length(value: Vector3) -> float:
    return math.sqrt(sum(component * component for component in value))
