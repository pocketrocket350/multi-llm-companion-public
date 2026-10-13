"""長い会話を、最近の発言とローカルの要点へ整理する部品。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from multi_llm_companion.domain import ChatMessage

_CORRECTION_MARKER = re.compile(
    r"(?:実は|訂正|ではなく|じゃなく|変更(?:した|になった)?|前の話は違う|やっぱり.{0,24}(?:も|は|が))"
)
_MEMORY_BOUNDARY_MARKER = re.compile(
    r"(?:覚えないで|記憶しない|記憶に残さない|忘れて|秘密にして|質問.{0,12}(?:しないで|終わらせないで|いらない)|"
    r"提案.{0,12}(?:いらない|しないで)|触れないで|聞かないで|詮索しないで|"
    r"恋人のように|扱わないで|呼ばないで|使わないで|話.{0,8}終わり)"
)


@dataclass(frozen=True, slots=True)
class ConversationPromptContext:
    """Providerへ渡す最近の発言と、古い発言の短い要点。"""

    messages: list[ChatMessage]
    summary: str
    status_text: str


class ConversationContextService:
    """会話履歴を消さずに、送信範囲だけを小さく保つ。"""

    def __init__(self, *, max_recent_messages: int, summary_max_characters: int) -> None:
        self._max_recent_messages = max(2, max_recent_messages)
        self._summary_max_characters = max(100, summary_max_characters)

    def build(self, messages: list[ChatMessage]) -> ConversationPromptContext:
        """古い発言を要点化し、直近の発言だけをそのまま残す。"""

        if len(messages) <= self._max_recent_messages:
            return ConversationPromptContext(messages, "", f"直近 {len(messages)} 件")
        older = messages[: -self._max_recent_messages]
        recent = messages[-self._max_recent_messages :]
        query = messages[-1].content if messages[-1].role == "user" else ""
        summary = _summarize_locally(older, self._summary_max_characters, query)
        return ConversationPromptContext(
            recent,
            summary,
            f"要点 + 直近 {len(recent)} 件（全 {len(messages)} 件）",
        )


def _summarize_locally(messages: list[ChatMessage], limit: int, query: str = "") -> str:
    """追加のAI呼び出しをせず、古い会話の話題を短く残す。"""

    corrections = [
        message
        for message in messages
        if message.role == "user" and _CORRECTION_MARKER.search(message.content)
    ]
    boundaries = [
        message
        for message in messages
        if message.role == "user" and _MEMORY_BOUNDARY_MARKER.search(message.content)
    ]
    priority_ids = {id(message) for message in corrections + boundaries}
    correction_limit, boundary_limit = _priority_limits(limit, corrections, boundaries)
    correction_parts = _summary_parts(
        list(reversed(corrections)), correction_limit, marker=_CORRECTION_MARKER
    )
    used = sum(len(part) for part in correction_parts)
    boundary_parts = _summary_parts(
        list(reversed(boundaries)), boundary_limit, marker=_MEMORY_BOUNDARY_MARKER
    )
    used += sum(len(part) for part in boundary_parts)
    regular = [message for message in messages if id(message) not in priority_ids]
    regular_parts = _summary_parts(_relevant_clauses(regular, query) + regular, limit - used)
    if not correction_parts and not boundary_parts and not regular_parts:
        return ""
    sections: list[str] = []
    if correction_parts:
        sections.append(
            "【この会話で利用者が訂正した情報】\n新しい指定を先に記載。矛盾する旧情報は現在の事実にしない。\n"
            + "\n".join(correction_parts)
        )
    if boundary_parts:
        sections.append("【利用者が指定した記憶の境界】\n" + "\n".join(boundary_parts))
    if regular_parts:
        sections.append("【この会話の過去の要点】\n" + "\n".join(regular_parts))
    return "\n".join(sections)


def _relevant_clauses(messages: list[ChatMessage], query: str) -> list[ChatMessage]:
    """今回の語句と重なる本人発言を、長い前置きの後からも拾う。"""
    common = frozenset("のはがをにでとへもや")

    def bigrams(text: str) -> set[str]:
        text = "".join(character for character in text.lower() if character.isalnum())
        return {
            text[index : index + 2]
            for index in range(len(text) - 1)
            if not any(character in common for character in text[index : index + 2])
        }

    terms = bigrams(query)
    ranked = []
    for index, message in enumerate(messages):
        if message.role != "user":
            continue
        for clause in re.split(r"(?<=[。！？!?])", message.content):
            clause = " ".join(clause.split())
            score = len(terms & bigrams(clause))
            if score >= 2:
                ranked.append((score, index, clause))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [ChatMessage(role="user", content=clause) for _, _, clause in ranked]


def _priority_limits(
    limit: int,
    corrections: list[ChatMessage],
    boundaries: list[ChatMessage],
) -> tuple[int, int]:
    """訂正と記憶境界が同時にあるとき、どちらも残す余地を確保する。"""

    if corrections and boundaries:
        correction_limit = limit // 2
        return correction_limit, limit - correction_limit
    if corrections:
        return limit, 0
    return 0, limit


def _summary_parts(
    messages: list[ChatMessage], limit: int, *, marker: re.Pattern[str] | None = None
) -> list[str]:
    parts: list[str] = []
    seen: set[tuple[str, str]] = set()
    used = 0
    for message in messages:
        speaker = "利用者" if message.role == "user" else "アシスタント"
        content = " ".join(message.content.split())
        if marker is not None:
            # 長い前置きではなく、訂正・禁止が実際に書かれた節を残す。
            clauses = []
            for clause in re.split(r"(?<=[。！？])", content):
                match = marker.search(clause)
                if match is not None:
                    clauses.append(clause[max(0, match.start() - 16) :])
            content = " ".join(clauses)
        if not content:
            continue
        key = (message.role, content)
        if key in seen:
            continue
        seen.add(key)
        prefix = f"- {speaker}: "
        remaining = limit - used
        if remaining <= len(prefix):
            break
        fragment = prefix + content[: min(160, remaining - len(prefix))]
        parts.append(fragment)
        used += len(fragment)
    return parts
