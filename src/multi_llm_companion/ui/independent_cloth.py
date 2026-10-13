"""PMXを書き換えず、別の布メッシュをBullet Soft Bodyで計算する。"""

from __future__ import annotations

import math

import numpy as np
from panda3d.bullet import (
    BulletConvexHullShape,
    BulletRigidBodyNode,
    BulletSoftBodyConfig,
    BulletSoftBodyNode,
    BulletSphereShape,
    BulletWorld,
)
from panda3d.core import LPoint3f, LVector3f, NodePath, PythonCallbackObject

from multi_llm_companion.ui.bullet_pmx_physics import _slerp_matrix, _transform_state
from multi_llm_companion.ui.models import UiAvatarModelPreview
from multi_llm_companion.ui.pmx_pose import BoneTransform

_SCALE = 0.08
_STEP = 1 / 120


class ClothNonFiniteError(RuntimeError):
    """再初期化できる、独立布ワーカーだけの数値失敗。"""


class SkirtClothSimulation:
    """姿勢計算側で直列に更新する、表示用の独立した布世界。"""

    @classmethod
    def create(
        cls,
        preview: UiAvatarModelPreview,
        posed_points: np.ndarray,
        transforms: dict[int, BoneTransform],
    ) -> SkirtClothSimulation | None:
        bone_names = {bone.name: i for i, bone in enumerate(preview.bones)}
        if "下半身" not in bone_names:
            return None
        skirt_bones = {i for name, i in bone_names.items() if name.startswith("Skirt_")}
        selected = np.asarray(
            [
                any(i in skirt_bones and w > 0.25 for i, w in weights)
                for weights in preview.vertex_weights
            ]
        )
        if len(selected) != len(posed_points) or not np.any(selected):
            return None
        faces = np.asarray([t.indexes for t in preview.dense_textured_triangles], dtype=np.int32)
        if not len(faces):
            return None
        faces = faces[np.all(selected[faces], axis=1)]
        if len(faces) < 8:
            return None
        return cls(preview, posed_points, transforms, faces, bone_names["下半身"])

    def __init__(self, preview, posed_points, transforms, faces, anchor_bone):
        self.vertex_indexes = np.unique(faces)
        self.vertex_indexes.setflags(write=False)
        covered = set(self.vertex_indexes)
        required = {
            i
            for j, weights in enumerate(preview.vertex_weights)
            if j not in covered
            for i, weight in weights
            if 0 <= i < len(preview.bones) and weight > 0
        }
        pending = list(required)
        while pending:
            bone = preview.bones[pending.pop()]
            parents = (
                bone.parent_index,
                bone.inherit_parent_index
                if (bone.inherit_rotation or bone.inherit_translation)
                else -1,
            )
            for parent in parents:
                if 0 <= parent < len(preview.bones) and parent not in required:
                    required.add(parent)
                    pending.append(parent)
        self.render_bone_indexes = frozenset(required)
        rest = np.asarray(preview.points, dtype=np.float64)
        # UVや裏面の重複頂点は計算面だけで共有し、元の頂点番号は保持する。
        _, representative, inverse = np.unique(
            np.round(rest[self.vertex_indexes], 5), axis=0, return_index=True, return_inverse=True
        )
        self._representatives = self.vertex_indexes[representative]
        self._inverse = inverse
        render_to_node = np.full(len(rest), -1, dtype=np.int32)
        render_to_node[self.vertex_indexes] = inverse
        node_faces = render_to_node[faces]
        valid = (
            (node_faces[:, 0] != node_faces[:, 1])
            & (node_faces[:, 1] != node_faces[:, 2])
            & (node_faces[:, 0] != node_faces[:, 2])
        )
        node_faces = node_faces[valid]
        _, unique_faces = np.unique(np.sort(node_faces, axis=1), axis=0, return_index=True)
        node_faces = node_faces[unique_faces]
        self._world = BulletWorld()
        self._world.set_gravity((0, -9.8, 0))
        info = self._world.get_world_info()
        info.set_gravity((0, -9.8, 0))
        self._filter = PythonCallbackObject(
            lambda data: data.set_collide(
                data.get_node_0().get_name() != "cloth-waist"
                and data.get_node_1().get_name() != "cloth-waist"
            )
        )
        self._world.set_filter_callback(self._filter)
        anchor_rotation, anchor_position = transforms[anchor_bone]
        # ネイティブ物理で既に折れた形を、布の基準形状として保存しない。
        points = (
            (rest[self._representatives] - preview.bones[anchor_bone].position)
            @ np.asarray(anchor_rotation).T
            + anchor_position
        ) * _SCALE
        self._cloth = BulletSoftBodyNode.make_tri_mesh(
            info, [LPoint3f(*p) for p in points], node_faces.ravel().tolist(), False
        )
        # Bullet側も頂点を統合・並べ替えるため、生成後の番号へ対応付ける。
        created = np.asarray([tuple(n.get_pos()) for n in self._cloth.get_nodes()])
        distances = np.sum((points[:, None] - created[None, :]) ** 2, axis=2)
        correspondence = np.argmin(distances, axis=1)
        if np.max(distances[np.arange(len(points)), correspondence]) > 1e-10:
            raise RuntimeError("独立した布メッシュの頂点対応を作れませんでした")
        self._inverse = correspondence[inverse]
        self._cloth.set_name("independent-skirt")
        self._cloth.set_total_mass(0.25)
        self._cloth.get_material(0).set_linear_stiffness(0.9)
        bending = self._cloth.append_material()
        bending.set_linear_stiffness(0.15)
        self._cloth.generate_bending_constraints(2, bending)
        config = self._cloth.get_cfg()
        config.set_positions_solver_iterations(8)
        config.set_damping_coefficient(0.02)
        config.set_dynamic_friction_coefficient(0.3)
        # プリーツの基準形状を弱い弾性として保持し、接触による変形は許す。
        self._cloth.set_pose(False, True)
        config.set_pose_matching_coefficient(0.05)
        # 点の間を手が通り抜けないよう、布の領域と剛体で接触を解く。
        config.set_collision_flag(BulletSoftBodyConfig.CF_sdf_rigid_soft, False)
        config.set_collision_flag(BulletSoftBodyConfig.CF_cluster_rigid_soft, True)
        config.set_collision_flag(BulletSoftBodyConfig.CF_cluster_soft_soft, True)
        config.set_collision_flag(BulletSoftBodyConfig.CF_cluster_self, True)
        self._cloth.generate_clusters(64, 128)
        self._cloth.get_shape(0).set_margin(0.001)
        self._anchor_bone = anchor_bone
        # 通常の揺れでは届かない距離で、有限値のまま進む数値発散も検出する。
        self._maximum_anchor_distance = (
            float(
                np.max(
                    np.linalg.norm(
                        rest[self._representatives] - preview.bones[anchor_bone].position, axis=1
                    )
                )
            )
            * _SCALE
            * 8
        )
        self._anchor = BulletRigidBodyNode("cloth-waist")
        self._anchor.add_shape(BulletSphereShape(0.001))
        self._anchor.set_kinematic(True)
        self._anchor_path = NodePath(self._anchor)
        self._move(self._anchor, self._anchor_path, transforms[anchor_bone])
        self._world.attach_rigid_body(self._anchor)
        heights = rest[self._representatives, 1]
        self._pinned = np.unique(correspondence[np.flatnonzero(heights >= heights.max() - 0.08)])
        anchor_rotation, anchor_position = transforms[anchor_bone]
        for index in self._pinned:
            pivot = np.asarray(anchor_rotation).T @ (
                created[index] - np.asarray(anchor_position) * _SCALE
            )
            self._cloth.append_anchor(int(index), self._anchor, LVector3f(*pivot), True)
        self._colliders = []
        surfaces: dict[tuple[str, int], set[int]] = {}
        for triangle in preview.dense_textured_triangles:
            key = (triangle.texture_path, -1 if triangle.texture_path else triangle.material_index)
            surfaces.setdefault(key, set()).update(triangle.indexes)
        for i, bone in enumerate(preview.bones):
            name = bone.name
            is_finger = name.startswith(("左", "右")) and any(
                term in name for term in ("親指", "人指", "人差指", "中指", "薬指", "小指")
            )
            if not is_finger and name not in (
                "下半身",
                "左足",
                "右足",
                "左手首",
                "右手首",
                "左手捩",
                "右手捩",
            ):
                continue
            indexes = [
                j
                for j, weights in enumerate(preview.vertex_weights)
                if any(b == i and weight > 0.25 for b, weight in weights) and render_to_node[j] < 0
            ]
            if len(indexes) < 4:
                continue
            node = BulletRigidBodyNode(f"cloth-contact-{i}")
            contact_groups = [set(indexes)]
            if i == anchor_bone:
                # 肌と衣装の間の空間を、腰の衝突面で埋めない。
                selected = set(indexes)
                separated = [selected & surface for surface in surfaces.values()]
                separated = [surface for surface in separated if len(surface) >= 4]
                if separated and set().union(*separated) == selected:
                    contact_groups = separated
            for surface in contact_groups:
                shape = BulletConvexHullShape()
                for point in np.unique(rest[sorted(surface)], axis=0) - bone.position:
                    shape.add_point(LPoint3f(*(point * _SCALE)))
                shape.set_margin(0.001)
                node.add_shape(shape)
            node.set_kinematic(True)
            path = NodePath(node)
            self._move(node, path, transforms[i])
            self._world.attach_rigid_body(node)
            self._colliders.append((i, node, path))
        # 初期配置を移動速度に数えないよう、布を入れる前に剛体を同期する。
        self._world.do_physics(_STEP, 1, _STEP)
        self._world.attach_soft_body(self._cloth)
        self.bone_indexes = frozenset((anchor_bone, *(i for i, _, _ in self._colliders)))
        self._previous_transforms = {i: transforms[i] for i in self.bone_indexes}
        self._time = None
        self._accumulator = 0.0
        self._positions = posed_points[self.vertex_indexes].copy()

    @staticmethod
    def _move(node, path, transform):
        rotation, position = transform
        path.set_transform(_transform_state(tuple(np.asarray(position) * _SCALE), rotation))
        node.set_transform_dirty()

    def advance(self, seconds: float, transforms: dict[int, BoneTransform]) -> np.ndarray:
        if self._time is not None and seconds <= self._time:
            return self._positions
        # 初回表示前だけ、腕を動かさず初期接触を落ち着かせる。
        elapsed = 0.5 if self._time is None else max(0.0, min(seconds - self._time, 0.25))
        self._time = seconds
        self._accumulator += elapsed
        steps = int((self._accumulator + 1e-10) / _STEP)
        bodies = ((self._anchor_bone, self._anchor, self._anchor_path), *self._colliders)
        for step in range(steps):
            # 1フレーム分の移動を最初の刻みへ集中させず、接触の衝撃を抑える。
            ratio = (step + 1) / steps
            for index, node, path in bodies:
                before, current = self._previous_transforms[index], transforms[index]
                rotation = (
                    current[0]
                    if before[0] == current[0]
                    else _slerp_matrix(before[0], current[0], ratio)
                )
                position = tuple(
                    a + (b - a) * ratio for a, b in zip(before[1], current[1], strict=True)
                )
                self._move(node, path, (rotation, position))
            self._world.do_physics(_STEP, 1, _STEP)
            # 次のサブステップや描画へ巨大な座標を渡す前に既存の回復経路へ戻す。
            bounds = self._cloth.get_aabb()
            anchor_position = self._anchor_path.get_pos()
            if any(
                not math.isfinite(value) or abs(value - anchor) > self._maximum_anchor_distance
                for edge in (bounds.get_min(), bounds.get_max())
                for value, anchor in zip(edge, anchor_position, strict=True)
            ):
                raise ClothNonFiniteError("独立した布計算の範囲がモデル寸法を大幅に超えました")
        self._accumulator -= steps * _STEP
        if steps:
            self._previous_transforms = {i: transforms[i] for i in self.bone_indexes}
        nodes = np.asarray([tuple(n.get_pos()) for n in self._cloth.get_nodes()], dtype=np.float32)
        self._positions = nodes[self._inverse] / _SCALE
        if not np.isfinite(self._positions).all():
            raise ClothNonFiniteError("独立した布計算の頂点が有限値ではありません")
        return self._positions

    def close(self) -> None:
        self._world.remove_soft_body(self._cloth)
        self._world.remove_rigid_body(self._anchor)
        for _, node, _ in self._colliders:
            self._world.remove_rigid_body(node)
        self._colliders.clear()
