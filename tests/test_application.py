from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from multi_llm_companion.application import (
    ApplicationBackend,
    _engine_contract,
    _system_prompt,
    _to_ui_vmd_motion,
)
from multi_llm_companion.config import (
    AppConfig,
    Avatar3DConfig,
    AvatarConfig,
    BehaviorPolicyConfig,
    CharacterConfig,
    PersonaConfig,
    ProviderConfig,
    ProvidersConfig,
    ResolvedProviderConfig,
    RuntimeConfig,
)
from multi_llm_companion.integrations.base import (
    AvatarBone,
    AvatarModelPreview,
    AvatarPhysicsJoint,
    AvatarRigidBody,
    AvatarTexturedTriangle,
    AvatarVmdBoneFrame,
    AvatarVmdCameraFrame,
    AvatarVmdCompatibility,
    AvatarVmdIkFrame,
    AvatarVmdIkState,
    AvatarVmdLightFrame,
    AvatarVmdMotion,
    AvatarVmdShadowFrame,
)
from multi_llm_companion.providers import MockProvider


def test_application_keeps_idle_vmd_intensity_scale() -> None:
    motion = AvatarVmdMotion(
        "待機VMD",
        30,
        (
            AvatarVmdBoneFrame(
                "右腕",
                0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        intensity_scale=0.75,
        target_model_name="対象モデル",
        camera_frames=(AvatarVmdCameraFrame(0, -45.0, (0.0, 10.0, 0.0), (0.0, 0.0, 0.0)),),
        light_frames=(AvatarVmdLightFrame(0, (1.0, 0.8, 0.6), (-0.5, 1.0, 0.5)),),
        shadow_frames=(AvatarVmdShadowFrame(0, 1, 10.0),),
        ik_frames=(AvatarVmdIkFrame(0, True, (AvatarVmdIkState("左足ＩＫ", False),)),),
        compatibility=AvatarVmdCompatibility(
            supported_bones=("右腕",),
            missing_iks=("補助ＩＫ",),
            unsupported_sections=("セルフ影",),
        ),
    )

    converted = _to_ui_vmd_motion(motion)

    assert converted.intensity_scale == 0.75
    assert converted.target_model_name == "対象モデル"
    assert converted.camera_frames[0].distance == -45.0
    assert converted.light_frames[0].color == (1.0, 0.8, 0.6)
    assert converted.shadow_frames[0].mode == 1
    assert converted.ik_frames[0].states[0].enabled is False
    assert converted.compatibility is not None
    assert converted.compatibility.missing_iks == ("補助ＩＫ",)


def _runtime(tmp_path: Path) -> RuntimeConfig:
    app_path = tmp_path / "local" / "app.yaml"
    app_path.parent.mkdir()
    app_path.write_text(
        "active_provider: mock\n"
        "providers_config_path: local/providers.yaml\n"
        "persona_path: local/persona.yaml\n"
        "database_path: data/conversations.db\n",
        encoding="utf-8",
    )
    provider_config = ProviderConfig(
        provider="mock",
        base_url_env="MOCK_BASE_URL",
        api_key_env="MOCK_API_KEY",
        model_env="MOCK_MODEL",
        temperature=0.7,
        max_output_tokens=128,
        timeout=5,
    )
    return RuntimeConfig(
        project_root=tmp_path,
        app_config_path=app_path,
        app=AppConfig(),
        providers=ProvidersConfig(providers={"mock": provider_config}),
        persona=PersonaConfig(name="案内役", system_prompt="短く答えてください。"),
        active_provider=ResolvedProviderConfig(
            name="mock",
            provider="mock",
            base_url="",
            api_key="",
            model="mock-model",
            temperature=0.7,
            max_output_tokens=128,
            timeout=5,
        ),
        database_path=tmp_path / "data" / "conversations.db",
    )


def test_external_permissions_default_on_and_provider_setup_preserves_opt_out(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    backend = ApplicationBackend(runtime)
    try:
        assert backend.web_search_enabled() and backend.image_sharing_enabled()
        assert not backend.web_search_available() and not backend.image_sharing_available()
        backend.set_web_search_enabled(False)
        backend.set_image_sharing_enabled(False)
        backend._configure_web_search(runtime.active_provider)
        assert not backend.web_search_enabled() and not backend.image_sharing_enabled()
        backend.set_web_search_enabled(True)
        backend._configure_web_search(runtime.active_provider)
        assert backend.web_search_enabled() and not backend.image_sharing_enabled()
        assert backend.list_conversations() == []
    finally:
        backend.close()


def test_application_backend_runs_mock_and_persists_history(tmp_path: Path) -> None:
    backend = ApplicationBackend(_runtime(tmp_path))
    conversation_id = backend.create_conversation("統合テスト")
    streamed: list[str] = []

    outcome = backend.stream_message(conversation_id, "こんにちは", streamed.append)

    assert "".join(streamed) == outcome.speech
    assert outcome.speech == "モック応答です。『こんにちは』を受け取りました。"
    assert outcome.input_tokens is not None
    assert outcome.emotion_state is not None
    assert outcome.emotion_state.emotion == "neutral"
    assert backend.emotion_state(conversation_id) == outcome.emotion_state
    assert backend.avatar_action(conversation_id).gesture == "none"
    assert backend.avatar_action(conversation_id).gaze == "forward"
    assert backend.avatar_3d_status().is_available is False
    usage_summary = backend.token_usage_summary(conversation_id)
    assert usage_summary.input_tokens == outcome.input_tokens
    assert usage_summary.output_tokens == outcome.output_tokens
    assert usage_summary.total_tokens == usage_summary.input_tokens + usage_summary.output_tokens
    assert usage_summary.response_count == 1
    messages = backend.load_messages(conversation_id)
    assert [(item.role, item.content) for item in messages] == [
        ("user", "こんにちは"),
        ("assistant", outcome.speech),
    ]
    assert backend.list_conversations()[0].id == conversation_id
    backend.delete_conversation(conversation_id)
    assert backend.list_conversations() == []
    backend.close()


def test_application_backend_hides_a_streamed_structure_leak(tmp_path: Path) -> None:
    raw = (
        "自然な返答です。JSON? Wait.\n"
        '{"speech":"自然な返答です。","emotion":"neutral","emotion_intensity":0.2,'
        '"gesture":"none","gaze":"forward","memory_candidates":[]}'
    )
    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(MockProvider(response_text=raw, chunk_size=3))  # type: ignore[attr-defined]
    conversation_id = backend.create_conversation("構造確認")
    streamed: list[str] = []

    outcome = backend.stream_message(conversation_id, "確認", streamed.append)

    assert "".join(streamed) == outcome.speech == "自然な返答です。"
    assert "JSON?" not in "".join(streamed)
    assert backend.load_messages(conversation_id)[-1].content == outcome.speech
    backend.close()


def test_markdown_control_leak_recovers_in_stream_and_history_without_rewriting_db(tmp_path):
    raw = 'こんばんは。 **speech**: "こんばんは。お話ししよう。" **emotion**: "happy" **memory_updates**: []'
    backend = ApplicationBackend(_runtime(tmp_path))
    try:
        backend._service.set_provider(MockProvider(response_text=raw, chunk_size=1))
        conv = backend.create_conversation("形式の検査")
        backend._repository.add_message(conv, "assistant", raw)
        backend._repository.add_message(conv, "user", raw)
        loaded = backend.load_messages(conv)
        assert loaded[0].content == "こんばんは。お話ししよう。"
        assert loaded[1].content == raw
        stored = backend._repository.list_messages(conv)
        assert stored[0].content == raw
        assert stored[0].to_chat_message().content == "こんばんは。お話ししよう。"
        assert stored[1].to_chat_message().content == raw
        streamed = []
        outcome = backend.stream_message(conv, "こんばんは", streamed.append)
        assert "".join(streamed) == outcome.speech == "こんばんは。お話ししよう。"
        assert backend.load_messages(conv)[-1].content == outcome.speech
        assert backend._repository.list_messages(conv)[0].content == raw
        assert backend.list_memories() == []
    finally:
        backend.close()


def test_application_backend_generates_and_persists_a_proactive_turn(tmp_path: Path) -> None:
    backend = ApplicationBackend(_runtime(tmp_path))
    conversation_id = backend.create_conversation("統合テスト")

    outcome = backend.generate_proactive_turn(conversation_id)

    assert [(item.role, item.content) for item in backend.load_messages(conversation_id)] == [
        ("assistant", outcome.speech)
    ]
    assert outcome.memory_candidates == []
    assert outcome.used_memories == []
    assert outcome.input_tokens is not None
    backend.close()


def test_application_backend_saves_catalog_expression_and_rebuilds_controller(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    avatar_path = tmp_path / "local" / "avatar.yaml"
    avatar_path.write_text(
        "display_name: テスト雨衣\n"
        "three_d:\n"
        "  expression_presets: {}\n"
        "  emotion_presets: {}\n"
        "  mmd_mapping:\n"
        "    expression_morphs: {}\n"
        "    mouth_morphs:\n"
        "      a: あ\n",
        encoding="utf-8",
    )
    backend = ApplicationBackend(runtime)
    source = AvatarModelPreview(
        model_name="表情保存PMX",
        vertex_count=1,
        face_count=0,
        points=((0.0, 0.0, 0.0),),
        morph_names=("笑い", "照れ", "あ"),
    )
    backend._avatar_controller = SimpleNamespace(model_preview=source)
    rebuilt = SimpleNamespace(model_preview=source, reset=lambda: None)
    monkeypatch.setattr(
        "multi_llm_companion.application.build_avatar_controller",
        lambda *_args, **_kwargs: rebuilt,
    )

    backend.save_expression_preset("happy_rich", "happy", {"笑い": 0.7, "照れ": 0.2})

    assert backend._avatar_controller is rebuilt
    assert backend._runtime.avatar.three_d.emotion_presets["happy"] == "happy_rich"
    assert "happy_rich" in (avatar_path.read_text(encoding="utf-8"))
    presets = backend.list_expression_presets()
    assert len(presets) == 1
    assert presets[0].name == "happy_rich"
    assert presets[0].morph_weights == (("笑い", 0.7), ("照れ", 0.2))
    assert presets[0].emotions == ("happy",)

    backend.delete_expression_preset("happy_rich")

    assert backend.list_expression_presets() == []
    assert "happy_rich" not in avatar_path.read_text(encoding="utf-8")
    backend.close()


def test_system_prompt_lists_registered_external_gesture_intents(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    local = tmp_path / "local"
    (local / "vmd_catalog.yaml").write_text(
        "motions:\n"
        "  private_file_name:\n"
        "    file: local/motions/private.vmd\n"
        "    motion_type: reaction\n"
        "    action: head_scratch\n"
        "    moods: [neutral]\n"
        "    energy: low\n"
        "    category: head\n"
        "    license_record: local/licenses/private.yaml\n"
        "    compatibility:\n"
        "      required_bones: [頭]\n",
        encoding="utf-8",
    )
    runtime = replace(
        runtime,
        avatar=AvatarConfig(
            three_d=Avatar3DConfig(
                motion_library_enabled=True,
                vmd_catalog_path="local/vmd_catalog.yaml",
            )
        ),
    )

    prompt = _system_prompt(runtime)

    assert "head_scratch" in prompt
    assert "private_file_name" not in prompt


def test_system_prompt_does_not_list_unconfigured_external_gestures(tmp_path: Path) -> None:
    prompt = _system_prompt(_runtime(tmp_path))

    assert "head_scratch" not in prompt


def test_engine_contract_limits_strong_gestures_to_matching_content(tmp_path: Path) -> None:
    prompt = _engine_contract(_runtime(tmp_path), ("explain", "cry"))

    assert "迷う場合はnone" in prompt
    assert "gesture用途 - explain: 理由・手順・仕組みを説明する。" in prompt
    assert "gesture用途 - cry: 強い悲しみを直接扱うときだけ使い" in prompt
    assert "gesture用途 - cold:" not in prompt


def test_engine_contract_maps_requested_jump_to_registered_delight(tmp_path: Path) -> None:
    prompt = _engine_contract(_runtime(tmp_path), ("delight",))

    assert "gesture用途 - delight: 喜びジャンプ" in prompt
    assert "ジャンプして" in prompt
    assert "跳んで" in prompt


def test_engine_contract_separates_face_requests_from_body_actions(tmp_path: Path) -> None:
    prompt = _engine_contract(_runtime(tmp_path), ("delight", "playful_shoot"))
    assert "表情・顔だけを求める依頼はemotionとemotion_intensityで表し、gestureはnone" in prompt
    assert "身体動作も明示的に求められた場合は表情と動作を独立に指定" in prompt
    assert "禁止・否定された動作は選ばない" in prompt
    assert "遊びの雰囲気はdialogue_modeのplayful" in prompt


def test_engine_contract_explains_adopted_reactions_and_restricts_staged_moves(
    tmp_path: Path,
) -> None:
    prompt = _engine_contract(_runtime(tmp_path), ("curious", "small_success", "walk_right"))
    assert "利用者の話へ興味を示し" in prompt
    assert "小さな成功や良い知らせ" in prompt
    assert "右へ歩く動作を明示的に求められたときだけ" in prompt
    assert "gesture用途 - robot_dance:" not in prompt


def test_application_transfers_idle_variants_without_losing_blink(tmp_path: Path) -> None:
    from multi_llm_companion.integrations.base import AvatarVmdMorphFrame

    backend = ApplicationBackend(_runtime(tmp_path))
    motion = AvatarVmdMotion(
        "別待機", 30, (), loop=False, morph_frames=(AvatarVmdMorphFrame("まばたき", 15, 1.0),)
    )
    source = AvatarModelPreview("テスト", 0, 0, (), idle_motions=(motion,))
    original = backend._avatar_controller
    backend._avatar_controller = SimpleNamespace(model_preview=source)
    converted = backend.avatar_model_preview()
    assert converted is not None
    assert len(converted.idle_motions) == 1
    assert converted.idle_motions[0].morph_frames[0].weight == 1.0
    assert not converted.idle_motions[0].loop
    backend._avatar_controller = original
    backend.close()


def test_system_prompt_keeps_persona_and_engine_contract_separate(tmp_path: Path) -> None:
    runtime = replace(
        _runtime(tmp_path),
        persona=PersonaConfig(
            character=CharacterConfig(name="案内役", speaking_style="穏やかな日本語"),
            behavior_policy=BehaviorPolicyConfig(response_length="短く答える"),
        ),
    )

    prompt = _system_prompt(runtime)

    character_position = prompt.index("【キャラクター設定】")
    policy_position = prompt.index("【行動方針】")
    contract_position = prompt.index("回答は必ず次のキーを持つJSON")
    assert character_position < policy_position < contract_position
    assert prompt.count("speech: 利用者へ伝える発言文字列") == 1


def test_application_backend_forwards_response_to_mock_3d_avatar(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime = RuntimeConfig(
        project_root=runtime.project_root,
        app_config_path=runtime.app_config_path,
        app=runtime.app,
        providers=runtime.providers,
        persona=runtime.persona,
        active_provider=runtime.active_provider,
        database_path=runtime.database_path,
        audio=runtime.audio,
        memory=runtime.memory,
        avatar=AvatarConfig(three_d=Avatar3DConfig(enabled=True, provider="mock")),
    )
    backend = ApplicationBackend(runtime)
    conversation_id = backend.create_conversation("3D接続テスト")

    backend.stream_message(conversation_id, "こんにちは", lambda _chunk: None)

    status = backend.avatar_3d_status()
    assert status.is_available is True
    assert status.controller_name == "3D連携Mock"
    assert status.render_quality == "standard"
    assert backend._avatar_controller.last_response is not None  # type: ignore[attr-defined]
    assert backend._avatar_controller.last_emotion_state is not None  # type: ignore[attr-defined]
    backend.close()


def test_application_backend_reuses_unchanged_avatar_preview(tmp_path: Path) -> None:
    backend = ApplicationBackend(_runtime(tmp_path))
    source = AvatarModelPreview(
        model_name="キャッシュテスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    controller = SimpleNamespace(model_preview=source)
    backend._avatar_controller = controller  # type: ignore[assignment]

    first = backend.avatar_model_preview()
    second = backend.avatar_model_preview()

    assert first is second
    controller.model_preview = replace(source, model_name="更新後")
    third = backend.avatar_model_preview()
    assert third is not first
    assert third is not None
    assert third.model_name == "更新後"
    backend.close()


def test_application_backend_keeps_pmx_vertex_edge_scales(tmp_path: Path) -> None:
    backend = ApplicationBackend(_runtime(tmp_path))
    triangle = AvatarTexturedTriangle(
        indexes=(0, 1, 2),
        uvs=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        texture_path="",
        diffuse=(1.0, 1.0, 1.0, 1.0),
        edge_scales=(0.25, 1.0, 1.75),
    )
    source = AvatarModelPreview(
        model_name="頂点エッジ倍率テスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        textured_triangles=(triangle,),
        dense_textured_triangles=(triangle,),
    )
    backend._avatar_controller = SimpleNamespace(model_preview=source)  # type: ignore[assignment]

    preview = backend.avatar_model_preview()

    assert preview is not None
    assert preview.textured_triangles[0].edge_scales == (0.25, 1.0, 1.75)
    assert preview.dense_textured_triangles[0].edge_scales == (0.25, 1.0, 1.75)
    backend.close()


def test_application_backend_keeps_pmx_bone_and_ik_rules(tmp_path: Path) -> None:
    backend = ApplicationBackend(_runtime(tmp_path))
    limits = (((-0.5, 0.0, 0.0), (0.1, 0.0, 0.0)),)
    source = AvatarModelPreview(
        model_name="ボーン受け渡しテスト",
        vertex_count=3,
        face_count=1,
        points=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        bones=(
            AvatarBone(
                name="左ひざ",
                parent_index=-1,
                position=(0.0, 1.0, 0.0),
                deform_layer=4,
                inherit_parent_index=1,
                inherit_weight=0.75,
                inherit_rotation=True,
                inherit_translation=True,
                fixed_axis=(1.0, 0.0, 0.0),
                local_axis_x=(1.0, 0.0, 0.0),
                local_axis_z=(0.0, 0.0, 1.0),
                after_physics=True,
                allows_rotation=True,
                allows_translation=False,
                ik_link_limits=limits,
            ),
        ),
        rigid_bodies=(
            AvatarRigidBody(
                0,
                (0.0, 1.0, 0.0),
                2.0,
                0.2,
                2,
                collision_group=3,
                non_collision_mask=4,
                shape=2,
                size=(0.2, 0.8, 0.2),
                rotation=(0.1, 0.2, 0.3),
                angular_damping=0.4,
                restitution=0.5,
                friction=0.6,
            ),
        ),
        physics_joints=(
            AvatarPhysicsJoint(
                0,
                0,
                (0.0, 1.0, 0.0),
                (1.0, 2.0, 3.0),
                rotation=(0.1, 0.0, 0.0),
                translation_min=(-0.2, -0.1, 0.0),
                translation_max=(0.2, 0.1, 0.0),
                rotation_min=(-0.3, 0.0, 0.0),
                rotation_max=(0.3, 0.0, 0.0),
                spring_rotation=(4.0, 5.0, 6.0),
            ),
        ),
    )
    backend._avatar_controller = SimpleNamespace(model_preview=source)  # type: ignore[assignment]

    preview = backend.avatar_model_preview()

    assert preview is not None
    bone = preview.bones[0]
    assert bone.deform_layer == 4
    assert bone.inherit_parent_index == 1
    assert bone.inherit_weight == 0.75
    assert bone.inherit_rotation is True
    assert bone.inherit_translation is True
    assert bone.fixed_axis == (1.0, 0.0, 0.0)
    assert bone.local_axis_x == (1.0, 0.0, 0.0)
    assert bone.local_axis_z == (0.0, 0.0, 1.0)
    assert bone.after_physics is True
    assert bone.allows_translation is False
    assert bone.ik_link_limits == limits
    body = preview.rigid_bodies[0]
    assert body.collision_group == 3
    assert body.non_collision_mask == 4
    assert body.shape == 2
    assert body.size == (0.2, 0.8, 0.2)
    assert body.angular_damping == 0.4
    assert body.restitution == 0.5
    assert body.friction == 0.6
    joint = preview.physics_joints[0]
    assert joint.translation_min == (-0.2, -0.1, 0.0)
    assert joint.rotation_max == (0.3, 0.0, 0.0)
    assert joint.spring_rotation == (4.0, 5.0, 6.0)
    backend.close()
