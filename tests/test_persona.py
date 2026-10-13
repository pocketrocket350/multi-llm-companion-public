from __future__ import annotations

import pytest
from pydantic import ValidationError

from multi_llm_companion.config import (
    BehaviorPolicyConfig,
    CharacterConfig,
    PersonaConfig,
)
from multi_llm_companion.domain import DialogueMode
from multi_llm_companion.services.persona import build_dialogue_mode_prompt, build_persona_prompt


def test_structured_persona_separates_character_and_behavior_policy() -> None:
    persona = PersonaConfig(
        character=CharacterConfig(
            name="ひなた",
            first_person="私",
            speaking_style="落ち着いた日本語",
            values=["正直に伝える"],
            preferences=["穏やかな会話"],
            character_notes=["オリジナルの案内役"],
        ),
        behavior_policy=BehaviorPolicyConfig(
            response_length="要点を先に短く答える",
            question_frequency="必要な場合だけ質問する",
            user_autonomy="最終判断は利用者に委ねる",
            memory_usage="承認済み記憶だけを参考にする",
        ),
        system_prompt_appendix="実在人物を再現しない。",
    )

    prompt = build_persona_prompt(persona)

    assert persona.name == "ひなた"
    assert "【キャラクター設定】" in prompt
    assert "価値観: 正直に伝える" in prompt
    assert "【行動方針】" in prompt
    assert "回答の長さ: 要点を先に短く答える" in prompt
    assert "記憶の利用: 承認済み記憶だけを参考にする" in prompt
    assert "【追加指示】\n実在人物を再現しない。" in prompt
    assert "speech:" not in prompt


def test_legacy_persona_keeps_original_system_prompt() -> None:
    persona = PersonaConfig(name="旧案内役", system_prompt="以前の指示を保ちます。")

    assert persona.is_legacy is True
    assert persona.name == "旧案内役"
    assert build_persona_prompt(persona) == "以前の指示を保ちます。"


def test_persona_rejects_mixed_legacy_and_structured_formats() -> None:
    with pytest.raises(ValidationError, match="旧形式と新形式"):
        PersonaConfig(
            name="旧案内役",
            system_prompt="以前の指示",
            character=CharacterConfig(name="新案内役"),
            behavior_policy=BehaviorPolicyConfig(),
        )


def test_structured_persona_requires_behavior_policy() -> None:
    with pytest.raises(ValidationError, match="characterとbehavior_policy"):
        PersonaConfig(character=CharacterConfig(name="案内役"))


def test_v2_static_sections_are_ordered_and_examples_are_safe() -> None:
    persona = PersonaConfig(
        character=CharacterConfig(name="案内役"),
        behavior_policy=BehaviorPolicyConfig(),
        official_facts=["公式に確認した事実"],
        interpretations=["事実から分離した解釈"],
        forbidden_inferences=["未確認設定を創作しない"],
        style_anchors=["穏やかに話す"],
        anti_patterns=["毎回質問しない"],
        dialogue_examples=[
            {
                "user": "一行目\n二行目",
                "assistant": "返答 " + "長" * 200,
            }
        ],
        system_prompt_appendix="追加",
    )

    prompt = build_persona_prompt(persona)

    titles = [
        "【キャラクター設定】",
        "【行動方針】",
        "【公式事実】",
        "【解釈】",
        "【推測禁止】",
        "【話し方の基準】",
        "【避ける表現】",
        "【会話例】",
        "【追加指示】",
    ]
    assert [prompt.index(title) for title in titles] == sorted(
        prompt.index(title) for title in titles
    )
    assert "利用者: 一行目 二行目" in prompt
    example_line = next(line for line in prompt.splitlines() if line.startswith("- 利用者:"))
    assert len(example_line) <= 350


def test_only_selected_dialogue_mode_is_added_to_runtime_prompt() -> None:
    persona = PersonaConfig(
        character=CharacterConfig(name="案内役"),
        behavior_policy=BehaviorPolicyConfig(),
        dialogue_modes={
            "listening": {"do": "静かに受け止める"},
            "planning": {"do": "手順を整理する"},
        },
    )

    prompt = build_dialogue_mode_prompt(persona, DialogueMode.LISTENING)

    assert "静かに受け止める" in prompt
    assert "手順を整理する" not in prompt
    assert "未定義" in build_dialogue_mode_prompt(persona, DialogueMode.CASUAL)


def test_any_v2_field_rejects_legacy_mix_and_unknown_mode() -> None:
    with pytest.raises(ValidationError, match="旧形式と新形式"):
        PersonaConfig(
            name="旧案内役",
            system_prompt="旧指示",
            official_facts=["新形式の事実"],
        )
    with pytest.raises(ValidationError, match="未対応の会話モード"):
        PersonaConfig(
            character=CharacterConfig(name="案内役"),
            behavior_policy=BehaviorPolicyConfig(),
            dialogue_modes={"unknown": {"do": "何かする"}},
        )
