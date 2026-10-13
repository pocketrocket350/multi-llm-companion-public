"""短期反復を避ける会話演技の実行時指示を組み立てる。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from multi_llm_companion.domain import DialogueMode, MessageRole, StoredMessage

_STOCK_ACKNOWLEDGEMENTS = (
    "わかった",
    "分かった",
    "そうなんだ",
    "なるほど",
    "了解",
)
_MOTIF_WORDS = ("雨", "光", "風", "色", "温度", "灯り", "水たまり")
_EXTREME_OR_HARMFUL_CLAIM = re.compile(
    r"(?:徹夜|寝ると負け|無視する|全部まずい|全部嫌い|絶対に|誰も|消えたい|傷つけ)"
)
_ASSISTANT_PREFERENCE_QUESTION = re.compile(
    r"(?:"
    r"(?:あなた|君|きみ|そっち|そちら).{0,12}(?:好き|嫌い|好み|苦手)"
    r"|(?:何|どんな).{0,8}(?:好き|嫌い|好み|苦手)"
    r"|(?:好き|嫌い|好み|苦手)な.{0,12}(?:ある|いる)"
    r"|(?:コーヒー|映画|音楽|本|パン|カフェ).{0,8}(?:派|好き).{0,6}[？?]"
    r")"
)
_USER_DECISION = re.compile(
    r"(?:"
    r"(?:だけ|まで|先に).{0,12}(?:する|やる|済ませる)"
    r"|(?:する|やる|済ませる)ことにする"
    r"|(?:それ|こう)で(?:いい|いく|決める)"
    r")"
)
_CORRECTION_MARKER = re.compile(
    r"(?:実は|訂正|ではなく|じゃなく|変更(?:した|になった)?|前の話は違う|やっぱり.{0,24}(?:も|は|が))"
)

PERFORMANCE_FEATURES = frozenset(
    {
        "stock_acknowledgement",
        "motif",
        "plain_language",
        "respect_user_decision",
        "respect_correction",
        "unearned_agreement",
        "small_opinion",
    }
)


@dataclass(frozen=True, slots=True)
class ConversationPerformancePlan:
    """保存せず、そのターンだけLLMへ渡す会話演技の方針。"""

    avoid_stock_acknowledgement: bool = False
    avoid_reused_motif: bool = False
    prefer_plain_language: bool = False
    respect_user_decision: bool = False
    respect_correction: bool = False
    avoid_unearned_agreement: bool = False
    allow_small_opinion: bool = False


def build_conversation_performance_plan(
    user_text: str,
    messages: Sequence[StoredMessage],
    *,
    mode: DialogueMode,
    enabled_features: frozenset[str] | None = None,
) -> ConversationPerformancePlan:
    """現在の発言と直近3回答から、短い演技方針を決定する。"""

    active_features = PERFORMANCE_FEATURES if enabled_features is None else enabled_features
    unknown_features = active_features - PERFORMANCE_FEATURES
    if unknown_features:
        raise ValueError(f"未対応の会話演技機能です: {', '.join(sorted(unknown_features))}")

    speeches = _recent_assistant_speeches(messages)
    return ConversationPerformancePlan(
        avoid_stock_acknowledgement="stock_acknowledgement" in active_features
        and bool(speeches and _starts_with_stock_acknowledgement(speeches[-1])),
        avoid_reused_motif="motif" in active_features and _has_reused_motif(speeches),
        prefer_plain_language="plain_language" in active_features
        and mode in {DialogueMode.CASUAL, DialogueMode.LISTENING}
        and _is_short_status_message(user_text),
        respect_user_decision="respect_user_decision" in active_features
        and bool(_USER_DECISION.search(user_text)),
        respect_correction="respect_correction" in active_features
        and _has_recent_correction(messages),
        avoid_unearned_agreement="unearned_agreement" in active_features
        and bool(_EXTREME_OR_HARMFUL_CLAIM.search(user_text)),
        allow_small_opinion="small_opinion" in active_features
        and mode is DialogueMode.CASUAL
        and bool(_ASSISTANT_PREFERENCE_QUESTION.search(user_text)),
    )


def build_performance_prompt(plan: ConversationPerformancePlan) -> str:
    """LLMへ渡す最大4項目の短い会話演技指示を返す。"""

    rules: list[str] = []
    if plan.avoid_stock_acknowledgement:
        rules.append("直前と同じ定型相づちだけで始めず、利用者の具体的な内容を短く受け止める。")
    if plan.avoid_reused_motif:
        rules.append("直近で使った比喩やモチーフを繰り返さず、今回は平易な言葉で答える。")
    if plan.prefer_plain_language:
        rules.append(
            "短い近況には、一文で具体的に返す。新しい比喩、情景説明、頼まれていない助言を足さない。"
        )
    if plan.respect_user_decision:
        rules.append("利用者が決めた方針を短く受け止める。次の行動や別案を提案しない。")
    if plan.respect_correction:
        rules.append(
            "利用者の訂正を最新の事実として扱う。以前の情報を言い切らず、自分の好み・比喩・新しい助言を足さない。"
        )
    if plan.avoid_unearned_agreement:
        rules.append("無条件に同意せず、必要なら理由を添えて穏やかに異なる見方を示す。")
    if plan.allow_small_opinion:
        rules.append("相手の決定を奪わない範囲で、自分の小さな所感を一つだけ添えてよい。")
    if not rules:
        return ""
    return "【今回の会話演技】\n- " + "\n- ".join(rules)


def _recent_assistant_speeches(messages: Sequence[StoredMessage]) -> tuple[str, ...]:
    return tuple(
        message.content
        for message in messages
        if message.role is MessageRole.ASSISTANT and message.content.strip()
    )[-3:]


def _starts_with_stock_acknowledgement(speech: str) -> bool:
    normalized = speech.lstrip(" \t\r\n、。！!？?")
    return normalized.startswith(_STOCK_ACKNOWLEDGEMENTS)


def _has_reused_motif(speeches: Sequence[str]) -> bool:
    if len(speeches) < 2:
        return False
    motif_counts = {motif: sum(motif in speech for speech in speeches) for motif in _MOTIF_WORDS}
    return any(count >= 2 for count in motif_counts.values())


def _is_short_status_message(user_text: str) -> bool:
    return len("".join(user_text.split())) <= 24


def _has_recent_correction(messages: Sequence[StoredMessage]) -> bool:
    recent_user_texts = [
        message.content for message in messages if message.role is MessageRole.USER
    ][-3:]
    return any(_CORRECTION_MARKER.search(text) for text in recent_user_texts)


__all__ = [
    "ConversationPerformancePlan",
    "PERFORMANCE_FEATURES",
    "build_conversation_performance_plan",
    "build_performance_prompt",
]
