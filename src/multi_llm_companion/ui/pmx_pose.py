"""第39段階のPMX変形順・付与変形・軸制約・IKリンク制限。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from multi_llm_companion.ui.models import UiAvatarModelPreview

type Vector3 = tuple[float, float, float]
type Matrix3 = tuple[Vector3, Vector3, Vector3]
type BoneTransform = tuple[Matrix3, Vector3]


@dataclass(frozen=True, slots=True)
class PreparedIkSolver:
    """モデル変更まで不変なIKコントローラーとリンク参照。"""

    controller_indexes: frozenset[int]
    required_indexes: tuple[frozenset[int], ...]
    link_indexes: frozenset[int]


def prepare_ik_solver(preview: UiAvatarModelPreview) -> PreparedIkSolver:
    """IKの対象と必要ボーンをモデル読込時に一度だけ整理する。"""

    controller_indexes: set[int] = set()
    link_indexes: set[int] = set()
    required_indexes: list[frozenset[int]] = [frozenset() for _ in preview.bones]
    for index, bone in enumerate(preview.bones):
        if bone.ik_target_index < 0 or not bone.ik_links:
            continue
        controller_indexes.add(index)
        link_indexes.update(bone.ik_links)
        required_indexes[index] = frozenset((index, bone.ik_target_index, *bone.ik_links))
    return PreparedIkSolver(
        controller_indexes=frozenset(controller_indexes),
        required_indexes=tuple(required_indexes),
        link_indexes=frozenset(link_indexes),
    )


def apply_bone_pose(
    points: list[Vector3],
    preview: UiAvatarModelPreview,
    rotations: dict[int, Vector3],
    ik_target_offsets: dict[int, Vector3] | None = None,
    translations: dict[int, Vector3] | None = None,
    disabled_ik_indexes: frozenset[int] = frozenset(),
    *,
    physics_translations: dict[int, Vector3] | None = None,
) -> list[Vector3]:
    """BDEF/SDEFの頂点重みで、ボーン姿勢を頂点へ反映する。"""

    if len(preview.vertex_weights) != len(points):
        return points
    posed_rotations = solve_ik_rotations(
        preview,
        rotations,
        ik_target_offsets or {},
        translations,
        disabled_controller_indexes=disabled_ik_indexes,
    )
    if not posed_rotations and not translations and not physics_translations:
        return points
    transforms = bone_transforms(
        preview, posed_rotations, translations, physics_translations=physics_translations
    )
    result: list[Vector3] = []
    for point, weights in zip(points, preview.vertex_weights, strict=True):
        valid_weights = [
            (bone_index, weight)
            for bone_index, weight in weights
            if bone_index in transforms and weight > 0.0
        ]
        total_weight = sum(weight for _bone_index, weight in valid_weights)
        if total_weight <= 0.0:
            result.append(point)
            continue
        x = y = z = 0.0
        for bone_index, weight in valid_weights:
            rotation, translation = transforms[bone_index]
            rest_position = preview.bones[bone_index].position
            transformed = add(matrix_vector(rotation, subtract(point, rest_position)), translation)
            normalized_weight = weight / total_weight
            x += transformed[0] * normalized_weight
            y += transformed[1] * normalized_weight
            z += transformed[2] * normalized_weight
        result.append((x, y, z))
    return result


def solve_ik_rotations(
    preview: UiAvatarModelPreview,
    rotations: dict[int, Vector3],
    target_offsets: dict[int, Vector3],
    translations: dict[int, Vector3] | None = None,
    prepared: PreparedIkSolver | None = None,
    disabled_controller_indexes: frozenset[int] = frozenset(),
) -> dict[int, Vector3]:
    """PMXの反復数・1回の角度・リンク制限を使うCCD IK計算。"""

    solved = dict(rotations)
    # MMDではIKは既定で有効。VMDに移動キーがないつま先IKも解き、
    # つま先だけでなく、かかとを含む靴底全体を姿勢へ反映する。
    solver = prepared or prepare_ik_solver(preview)
    controller_indexes = (
        target_offsets.keys() | solver.controller_indexes
        if target_offsets
        else solver.controller_indexes
    )
    for controller_index in controller_indexes:
        if controller_index in disabled_controller_indexes:
            continue
        offset = target_offsets.get(controller_index, (0.0, 0.0, 0.0))
        if not 0 <= controller_index < len(preview.bones):
            continue
        controller = preview.bones[controller_index]
        if controller.ik_target_index < 0 or not controller.ik_links:
            continue
        required_indexes = solver.required_indexes[controller_index]
        initial_transforms = bone_transforms(
            preview,
            solved,
            translations,
            required_indexes,
        )
        controller_transform = initial_transforms.get(controller_index)
        if controller_transform is None:
            continue
        goal = add(controller_transform[1], offset)
        used_two_link_solver = _solve_two_link_ik(
            preview,
            solved,
            translations,
            controller_index,
            goal,
            required_indexes,
            initial_transforms,
        )
        # 人型の脚は解析解で大部分を合わせ、少数回のCCDで誤差だけを詰める。
        # 一般的な多リンクIKはPMX指定の反復数をそのまま使う。
        maximum_steps = (
            min(controller.ik_iterations or 1, 4)
            if used_two_link_solver
            else min(controller.ik_iterations or 1, 256)
        )
        transforms = initial_transforms
        if used_two_link_solver:
            transforms = bone_transforms(
                preview,
                solved,
                translations,
                required_indexes,
            )
        for _ in range(maximum_steps):
            effector = transforms.get(controller.ik_target_index)
            if effector is None:
                break
            distance_before = length(subtract(goal, effector[1]))
            if distance_before < 0.002:
                break
            for link_position, link_index in enumerate(controller.ik_links):
                transform = transforms.get(link_index)
                if transform is None:
                    continue
                to_effector = subtract(effector[1], transform[1])
                to_goal = subtract(goal, transform[1])
                if length(to_effector) < 0.00001 or length(to_goal) < 0.00001:
                    continue
                axis = cross(to_effector, to_goal)
                axis_length = length(axis)
                if axis_length < 0.00001:
                    continue
                angle = math.acos(
                    max(-1.0, min(1.0, dot(normalize(to_effector), normalize(to_goal))))
                )
                angle = min(angle, controller.ik_limit_radians or 0.35, 0.35)
                world_axis = tuple(value / axis_length for value in axis)
                parent_index = preview.bones[link_index].parent_index
                parent_rotation = (
                    transforms[parent_index][0] if parent_index in transforms else identity_matrix()
                )
                local_axis = matrix_vector(matrix_transpose(parent_rotation), world_axis)
                current = solved.get(link_index, (0.0, 0.0, 0.0))
                candidate = matrix_to_euler(
                    matrix_multiply(
                        axis_angle_matrix(local_axis, angle),
                        rotation_matrix(*current),
                    )
                )
                candidate = constrain_bone_rotation(preview.bones[link_index], candidate)
                if link_position < len(controller.ik_link_limits):
                    limits = controller.ik_link_limits[link_position]
                    if limits is not None:
                        candidate = clamp_vector(candidate, *limits)
                solved[link_index] = candidate
                transforms = bone_transforms(
                    preview,
                    solved,
                    translations,
                    required_indexes,
                )
                effector = transforms.get(controller.ik_target_index)
                if effector is None or length(subtract(goal, effector[1])) < 0.002:
                    break
            current_distance = (
                length(subtract(goal, effector[1])) if effector is not None else float("inf")
            )
            # 制約で動けないIKを指定回数だけ空回りさせず、改善中のIKだけ続行する。
            if distance_before - current_distance <= 0.00001:
                break
    return solved


def _solve_two_link_ik(
    preview: UiAvatarModelPreview,
    solved: dict[int, Vector3],
    translations: dict[int, Vector3] | None,
    controller_index: int,
    goal: Vector3,
    required_indexes: frozenset[int],
    initial_transforms: dict[int, BoneTransform],
) -> bool:
    """人型の脚で一般的な、ひざ1軸と足の2リンクIKを高速に解く。"""

    controller = preview.bones[controller_index]
    if len(controller.ik_links) != 2 or not controller.ik_link_limits:
        return False
    knee_index, hip_index = controller.ik_links
    knee_limits = controller.ik_link_limits[0]
    if knee_limits is None or preview.bones[knee_index].parent_index != hip_index:
        return False
    minimum, maximum = knee_limits
    if any(abs(value) > 0.000001 for value in (*minimum[1:], *maximum[1:])):
        return False

    transforms = initial_transforms
    hip = transforms.get(hip_index)
    knee = transforms.get(knee_index)
    effector = transforms.get(controller.ik_target_index)
    if hip is None or knee is None or effector is None:
        return False
    upper_length = length(subtract(knee[1], hip[1]))
    lower_length = length(subtract(effector[1], knee[1]))
    target_distance = length(subtract(goal, hip[1]))
    if upper_length < 0.00001 or lower_length < 0.00001 or target_distance < 0.00001:
        return False

    reachable = max(
        abs(upper_length - lower_length) + 0.00001,
        min(upper_length + lower_length - 0.00001, target_distance),
    )
    cosine = max(
        -1.0,
        min(
            1.0,
            (upper_length * upper_length + lower_length * lower_length - reachable * reachable)
            / (2.0 * upper_length * lower_length),
        ),
    )
    bend = math.acos(cosine) - math.pi
    if minimum[0] >= 0.0:
        bend = abs(bend)
    solved[knee_index] = clamp_vector((bend, 0.0, 0.0), minimum, maximum)

    transforms = bone_transforms(
        preview,
        solved,
        translations,
        required_indexes,
    )
    hip = transforms[hip_index]
    effector = transforms[controller.ik_target_index]
    current_direction = subtract(effector[1], hip[1])
    target_direction = subtract(goal, hip[1])
    axis = cross(current_direction, target_direction)
    axis_length = length(axis)
    if axis_length < 0.00001:
        return True
    angle = math.acos(
        max(-1.0, min(1.0, dot(normalize(current_direction), normalize(target_direction))))
    )
    parent_index = preview.bones[hip_index].parent_index
    parent_rotation = (
        transforms[parent_index][0] if parent_index in transforms else identity_matrix()
    )
    world_axis = tuple(value / axis_length for value in axis)
    local_axis = matrix_vector(matrix_transpose(parent_rotation), world_axis)
    current = solved.get(hip_index, (0.0, 0.0, 0.0))
    solved[hip_index] = constrain_bone_rotation(
        preview.bones[hip_index],
        matrix_to_euler(
            matrix_multiply(
                axis_angle_matrix(local_axis, angle),
                rotation_matrix(*current),
            )
        ),
    )
    return True


def bone_transforms(
    preview: UiAvatarModelPreview,
    rotations: dict[int, Vector3],
    translations: dict[int, Vector3] | None = None,
    required_indexes: frozenset[int] | None = None,
    *,
    physics_translations: dict[int, Vector3] | None = None,
) -> dict[int, BoneTransform]:
    """PMXの親子・付与親・軸・変形階層を考慮したワールド変換。

    ``required_indexes`` がある場合は、そのボーンと計算に必要な親だけを
    解決する。IK反復中に全身を毎回計算しないための高速経路である。
    """

    cached: dict[int, BoneTransform] = {}
    effective_rotations: dict[int, Vector3] = {}
    effective_translations: dict[int, Vector3] = {}
    physical_translations = (
        {
            body.bone_index: physics_translations[body.bone_index]
            for body in preview.rigid_bodies
            if body.mode == 1 and body.bone_index in physics_translations
        }
        if physics_translations
        else {}
    )

    def effective_rotation(index: int, visiting: set[int]) -> Vector3:
        if index in effective_rotations:
            return effective_rotations[index]
        bone = preview.bones[index]
        value = rotations.get(index, (0.0, 0.0, 0.0)) if bone.allows_rotation else (0.0, 0.0, 0.0)
        source_index = bone.inherit_parent_index
        if (
            bone.inherit_rotation
            and 0 <= source_index < len(preview.bones)
            and source_index not in visiting
        ):
            source = effective_rotation(source_index, visiting | {index})
            value = add(value, tuple(item * bone.inherit_weight for item in source))
        value = constrain_bone_rotation(bone, value)
        effective_rotations[index] = value
        return value

    def effective_translation(index: int, visiting: set[int]) -> Vector3:
        if index in effective_translations:
            return effective_translations[index]
        bone = preview.bones[index]
        # 編集用の移動禁止フラグで、動的剛体からの位置更新は捨てない。
        if not bone.allows_translation and index in physical_translations:
            value = physical_translations[index]
            effective_translations[index] = value
            return value
        value = (
            (translations or {}).get(index, (0.0, 0.0, 0.0))
            if bone.allows_translation
            else (0.0, 0.0, 0.0)
        )
        source_index = bone.inherit_parent_index
        if (
            bone.inherit_translation
            and 0 <= source_index < len(preview.bones)
            and source_index not in visiting
        ):
            source = effective_translation(source_index, visiting | {index})
            value = add(value, tuple(item * bone.inherit_weight for item in source))
        effective_translations[index] = value
        return value

    def resolve(index: int, visiting: set[int]) -> BoneTransform:
        if index in cached:
            return cached[index]
        bone = preview.bones[index]
        local_rotation = bone_rotation_matrix(bone, effective_rotation(index, set()))
        parent_index = bone.parent_index
        translation_offset = effective_translation(index, set())
        if 0 <= parent_index < len(preview.bones) and parent_index not in visiting:
            parent_rotation, parent_translation = resolve(parent_index, visiting | {index})
            local_position = subtract(bone.position, preview.bones[parent_index].position)
            transform = (
                matrix_multiply(parent_rotation, local_rotation),
                add(
                    matrix_vector(parent_rotation, add(local_position, translation_offset)),
                    parent_translation,
                ),
            )
        else:
            transform = (local_rotation, add(bone.position, translation_offset))
        cached[index] = transform
        return transform

    if required_indexes is None:
        targets = range(len(preview.bones))
    else:
        targets = (index for index in required_indexes if 0 <= index < len(preview.bones))
    order = sorted(
        targets,
        key=lambda index: (
            preview.bones[index].after_physics,
            preview.bones[index].deform_layer,
            index,
        ),
    )
    for index in order:
        resolve(index, set())
    return cached


def constrain_bone_rotation(bone: object, rotation: Vector3) -> Vector3:
    """固定軸ボーンの回転を、その軸方向だけへ制限する。"""

    axis = getattr(bone, "fixed_axis", None)
    if axis is None:
        return rotation if all(math.isfinite(item) for item in rotation) else (0.0, 0.0, 0.0)
    unit_axis = normalize(axis)
    amount = dot(rotation, unit_axis)
    return tuple(item * amount for item in unit_axis)  # type: ignore[return-value]


def bone_rotation_matrix(bone: object, rotation: Vector3) -> Matrix3:
    """PMXローカル軸があれば、その座標系で回転行列を作る。"""

    local_rotation = rotation_matrix(*rotation)
    axis_x = getattr(bone, "local_axis_x", None)
    axis_z = getattr(bone, "local_axis_z", None)
    if axis_x is None or axis_z is None:
        return local_rotation
    x_axis = normalize(axis_x)
    z_hint = normalize(axis_z)
    y_axis = normalize(cross(z_hint, x_axis))
    z_axis = normalize(cross(x_axis, y_axis))
    if length(x_axis) < 0.00001 or length(y_axis) < 0.00001 or length(z_axis) < 0.00001:
        return local_rotation
    basis: Matrix3 = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    return matrix_multiply(matrix_multiply(basis, local_rotation), matrix_transpose(basis))


def clamp_vector(value: Vector3, minimum: Vector3, maximum: Vector3) -> Vector3:
    """IKリンク角度をPMXの軸別範囲へ収める。"""

    return tuple(max(minimum[index], min(maximum[index], value[index])) for index in range(3))  # type: ignore[return-value]


def rotation_matrix(pitch: float, yaw: float, roll: float) -> Matrix3:
    cosine_x, sine_x = math.cos(pitch), math.sin(pitch)
    cosine_y, sine_y = math.cos(yaw), math.sin(yaw)
    cosine_z, sine_z = math.cos(roll), math.sin(roll)
    rotation_x: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cosine_x, -sine_x), (0.0, sine_x, cosine_x))
    rotation_y: Matrix3 = ((cosine_y, 0.0, sine_y), (0.0, 1.0, 0.0), (-sine_y, 0.0, cosine_y))
    rotation_z: Matrix3 = ((cosine_z, -sine_z, 0.0), (sine_z, cosine_z, 0.0), (0.0, 0.0, 1.0))
    return matrix_multiply(matrix_multiply(rotation_z, rotation_y), rotation_x)


def axis_angle_matrix(axis: Vector3, angle: float) -> Matrix3:
    """単位軸と角度から回転行列を作る。"""

    x_axis, y_axis, z_axis = normalize(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    remainder = 1.0 - cosine
    return (
        (
            cosine + x_axis * x_axis * remainder,
            x_axis * y_axis * remainder - z_axis * sine,
            x_axis * z_axis * remainder + y_axis * sine,
        ),
        (
            y_axis * x_axis * remainder + z_axis * sine,
            cosine + y_axis * y_axis * remainder,
            y_axis * z_axis * remainder - x_axis * sine,
        ),
        (
            z_axis * x_axis * remainder - y_axis * sine,
            z_axis * y_axis * remainder + x_axis * sine,
            cosine + z_axis * z_axis * remainder,
        ),
    )


def matrix_to_euler(matrix: Matrix3) -> Vector3:
    """Z-Y-X順の回転行列を、既存処理のX-Y-Z角度へ戻す。"""

    yaw = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    if abs(math.cos(yaw)) > 0.000001:
        pitch = math.atan2(matrix[2][1], matrix[2][2])
        roll = math.atan2(matrix[1][0], matrix[0][0])
    else:
        pitch = math.atan2(-matrix[1][2], matrix[1][1])
        roll = 0.0
    return (pitch, yaw, roll)


def identity_matrix() -> Matrix3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def matrix_multiply(first: Matrix3, second: Matrix3) -> Matrix3:
    """3×3行列を掛ける。周期処理なので一時リストや生成式を作らない。"""

    a00, a01, a02 = first[0]
    a10, a11, a12 = first[1]
    a20, a21, a22 = first[2]
    b00, b01, b02 = second[0]
    b10, b11, b12 = second[1]
    b20, b21, b22 = second[2]
    return (
        (
            a00 * b00 + a01 * b10 + a02 * b20,
            a00 * b01 + a01 * b11 + a02 * b21,
            a00 * b02 + a01 * b12 + a02 * b22,
        ),
        (
            a10 * b00 + a11 * b10 + a12 * b20,
            a10 * b01 + a11 * b11 + a12 * b21,
            a10 * b02 + a11 * b12 + a12 * b22,
        ),
        (
            a20 * b00 + a21 * b10 + a22 * b20,
            a20 * b01 + a21 * b11 + a22 * b21,
            a20 * b02 + a21 * b12 + a22 * b22,
        ),
    )


def matrix_transpose(matrix: Matrix3) -> Matrix3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    x, y, z = vector
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def add(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def subtract(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def dot(first: Vector3, second: Vector3) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def length(vector: Vector3) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vector3) -> Vector3:
    size = length(vector)
    return (0.0, 0.0, 0.0) if size < 0.00001 else tuple(value / size for value in vector)  # type: ignore[return-value]
