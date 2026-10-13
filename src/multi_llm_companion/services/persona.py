"""人格設定をLLM向けの指示へ変換する。"""

from __future__ import annotations

from multi_llm_companion.config import DialogueModePolicyConfig, PersonaConfig
from multi_llm_companion.domain import DialogueMode

_EXAMPLE_TEXT_LIMIT = 160


def build_persona_prompt(persona: PersonaConfig) -> str:
    """旧形式との互換を保ち、人格と行動方針を別の節へ整形する。"""

    if persona.is_legacy:
        return persona.system_prompt

    character = persona.character
    policy = persona.behavior_policy
    if character is None or policy is None:  # PersonaConfigの検証後には到達しない。
        raise ValueError("構造化された人格設定が不足しています")

    character_lines = [
        "【キャラクター設定】",
        f"名前: {character.name}",
        f"一人称: {character.first_person}",
    ]
    _append_text(character_lines, "口調", character.speaking_style)
    _append_list(character_lines, "価値観", character.values)
    _append_list(character_lines, "好み", character.preferences)
    _append_list(character_lines, "固有設定", character.character_notes)

    policy_lines = ["【行動方針】"]
    for label, value in (
        ("回答の長さ", policy.response_length),
        ("質問頻度", policy.question_frequency),
        ("主体性", policy.initiative),
        ("反対意見", policy.disagreement_style),
        ("利用者の自律性", policy.user_autonomy),
        ("親密表現の上限", policy.intimacy_boundary),
        ("技術質問", policy.technical_guidance),
        ("記憶の利用", policy.memory_usage),
    ):
        _append_text(policy_lines, label, value)

    sections = ["\n".join(character_lines), "\n".join(policy_lines)]
    _append_section(sections, "公式事実", persona.official_facts)
    _append_section(sections, "解釈", persona.interpretations)
    _append_section(sections, "推測禁止", persona.forbidden_inferences)
    _append_section(sections, "話し方の基準", persona.style_anchors)
    _append_section(sections, "避ける表現", persona.anti_patterns)
    examples = [
        f"- 利用者: {_oneline(example.user)} / アシスタント: {_oneline(example.assistant)}"
        for example in persona.dialogue_examples
        if example.user or example.assistant
    ]
    if examples:
        sections.append("【会話例】\n" + "\n".join(examples))
    if persona.system_prompt_appendix:
        sections.append(f"【追加指示】\n{persona.system_prompt_appendix}")
    return "\n\n".join(sections)


def build_dialogue_mode_prompt(persona: PersonaConfig, mode: DialogueMode) -> str:
    """選択済みの1モードだけをRuntime指示へ整形する。"""

    policy = persona.dialogue_modes.get(mode.value)
    lines = [f"【今回の会話モード: {mode.value}】"]
    if policy is None:
        lines.append("未定義のため、静的人格と会話運営の共通規則に従う。")
        return "\n".join(lines)
    _append_mode_policy(lines, policy)
    if len(lines) == 1:
        lines.append("個別指定はないため、会話運営の共通規則に従う。")
    return "\n".join(lines)


def _append_text(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _append_list(lines: list[str], label: str, values: list[str]) -> None:
    if values:
        lines.append(f"{label}: {'、'.join(values)}")


def _append_section(sections: list[str], title: str, values: list[str]) -> None:
    if values:
        sections.append(f"【{title}】\n" + "\n".join(f"- {value}" for value in values))


def _append_mode_policy(lines: list[str], policy: DialogueModePolicyConfig) -> None:
    _append_text(lines, "適用場面", policy.applies_when)
    _append_text(lines, "行うこと", policy.do)
    _append_text(lines, "避けること", policy.avoid)


def _oneline(value: str) -> str:
    return " ".join(value.split())[:_EXAMPLE_TEXT_LIMIT]
