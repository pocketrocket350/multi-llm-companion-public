"""保存済み回答だけから、短期の会話運営状態を組み立てる。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from multi_llm_companion.config import PersonaConfig
from multi_llm_companion.domain import (
    AssistantResponse,
    DialogueAct,
    DialogueMode,
    MessageRole,
    StoredMessage,
)
from multi_llm_companion.services.dialogue_performance import (
    build_conversation_performance_plan,
    build_performance_prompt,
)
from multi_llm_companion.services.persona import build_dialogue_mode_prompt

_QUESTION_END = re.compile(r"(?:[?？]|(?:です|ます|でしょう|だろう|かな|かも)か[。！!]?)\s*$")
_QUESTION_WORD = re.compile(
    r"(?:どう思う|どうする|どうしたい|何が|何を|どれ|いつ|どこ|教えて[くれま]?すか)"
)
_SUPPORT_REQUEST = re.compile(
    r"(?:どうすれば|どうしたら|助けて|手伝って|相談|アドバイス|提案して|考えてほしい|一緒に考えて)"
)
_FATIGUE = re.compile(r"(?:疲れた|疲れて|しんどい|くたくた|へとへと|つらい|眠い|休みたい)")
_TECHNICAL = re.compile(
    r"(?:エラー|バグ|コード|実装|設定|API|Python|JSON|YAML|SQL|関数|クラス|アルゴリズム|原因|仕組み)"
)
_TECHNICAL_QUESTION = re.compile(r"(?:なぜ|どうして|どうやって|方法|とは|教えて|説明|[?？])")
_PLANNING = re.compile(r"(?:計画|予定|段取り|手順|スケジュール|ロードマップ|目標|準備|進め方)")
_PLAYFUL = re.compile(r"(?:冗談|ジョーク|からか|ふざけ|笑わせ|ネタ|ｗ|www|笑[）)]?)", re.IGNORECASE)
_BOUNDARY = re.compile(
    r"(?:その話(?:は)?やめ|話したくない|聞かないで|触れないで|詮索しないで|距離を置|依存させ|恋人のように|もう話さない)"
)
_TOPIC_RESET = re.compile(
    r"(?:別の話|話題を?変え|その話(?:は)?終わり|この話(?:は)?終わり|もういい|やめよう|ここまで|忘れて)"
)
_CONTINUATION = re.compile(r"(?:それ|その続き|続き|さっき(?:の.{0,24})?話|前の話|今の話|もう少し)")


@dataclass(frozen=True, slots=True)
class DialogueState:
    """DBへ追加保存しない、会話ごとの短期運営状態。"""

    last_mode: DialogueMode = DialogueMode.CASUAL
    last_act: DialogueAct = DialogueAct.ANSWER
    current_topic: str = ""
    open_loop: str = ""
    recent_question_count: int = 0
    recent_suggestion_count: int = 0


class DialogueDirector:
    """現在発言とresponse_jsonから追加LLMなしで指示を作る。"""

    def __init__(
        self,
        persona: PersonaConfig,
        *,
        performance_features: frozenset[str] | None = None,
    ) -> None:
        self._persona = persona
        self._performance_features = performance_features

    def rebuild_state(self, messages: Sequence[StoredMessage]) -> DialogueState:
        """同一会話の切り詰め前履歴から状態を決定的に再構築する。"""

        responses: list[AssistantResponse] = []
        for message in messages:
            if message.role is not MessageRole.ASSISTANT:
                continue
            responses.append(message.response or AssistantResponse(speech=""))
        if not responses:
            return DialogueState()

        latest = responses[-1]
        return DialogueState(
            last_mode=latest.dialogue_mode,
            last_act=latest.dialogue_act,
            current_topic=latest.current_topic,
            open_loop=latest.open_loop,
            recent_question_count=sum(_is_question(item) for item in responses[-2:]),
            recent_suggestion_count=sum(
                item.dialogue_act in {DialogueAct.SUGGEST, DialogueAct.PLAN}
                for item in responses[-3:]
            ),
        )

    def select_mode(self, user_text: str) -> DialogueMode:
        """現在の発言を最優先に、7モードから1つだけ選ぶ。"""

        text = " ".join(user_text.split())
        if _BOUNDARY.search(text):
            return DialogueMode.BOUNDARY
        if _TECHNICAL.search(text) and _TECHNICAL_QUESTION.search(text):
            return DialogueMode.EXPLAINING
        if _PLANNING.search(text):
            return DialogueMode.PLANNING
        if _PLAYFUL.search(text):
            return DialogueMode.PLAYFUL
        if _SUPPORT_REQUEST.search(text):
            return DialogueMode.SUPPORTING
        if _FATIGUE.search(text):
            return DialogueMode.LISTENING
        return DialogueMode.CASUAL

    def build_prompt(self, user_text: str, messages: Sequence[StoredMessage]) -> str:
        """今回だけ使うモード、状態、反復抑制を整形する。"""

        state = self.rebuild_state(messages)
        return self.build_runtime_prompt(user_text, state, messages=messages)

    def build_proactive_prompt(self, messages: Sequence[StoredMessage]) -> str:
        """利用者入力なしの短い話しかけにも、直前の会話状態を適用する。"""

        state = self.rebuild_state(messages)
        sections = [
            build_dialogue_mode_prompt(self._persona, DialogueMode.CASUAL),
            _state_prompt(state, continues_topic=False),
            _proactive_cooldown_prompt(state),
        ]
        return "\n\n".join(sections)

    def build_runtime_prompt(
        self,
        user_text: str,
        state: DialogueState,
        *,
        messages: Sequence[StoredMessage] = (),
    ) -> str:
        mode = self.select_mode(user_text)
        resets_topic = bool(_TOPIC_RESET.search(user_text) or _BOUNDARY.search(user_text))
        continues_topic = bool(_CONTINUATION.search(user_text))
        effective = replace(
            state,
            current_topic="" if resets_topic else state.current_topic,
            open_loop="" if resets_topic else state.open_loop,
        )
        sections = [build_dialogue_mode_prompt(self._persona, mode)]
        sections.append(_state_prompt(effective, continues_topic=continues_topic))
        sections.append(_cooldown_prompt(user_text, effective, resets_topic=resets_topic))
        performance = build_performance_prompt(
            build_conversation_performance_plan(
                user_text,
                messages,
                mode=mode,
                enabled_features=self._performance_features,
            )
        )
        if performance:
            sections.append(performance)
        return "\n\n".join(sections)


def _is_question(response: AssistantResponse) -> int:
    if response.dialogue_act is DialogueAct.ASK:
        return 1
    speech = " ".join(response.speech.split())
    return int(bool(_QUESTION_END.search(speech) or _QUESTION_WORD.search(speech)))


def _state_prompt(state: DialogueState, *, continues_topic: bool) -> str:
    topic = state.current_topic or "なし"
    open_loop = state.open_loop or "なし"
    continuation = "継続表現あり" if continues_topic else "継続表現なし"
    return (
        "【直前のDialogueState】\n"
        f"last_mode: {state.last_mode.value}\n"
        f"last_act: {state.last_act.value}\n"
        f"current_topic: {topic}\n"
        f"open_loop: {open_loop}\n"
        f"recent_question_count: {state.recent_question_count}\n"
        f"recent_suggestion_count: {state.recent_suggestion_count}\n"
        f"参照条件: {continuation}。現在の利用者発言を常に最優先する。\n"
        "current_topicとopen_loopは短期の補助情報であり、確定した利用者情報として扱わない。"
    )


def _cooldown_prompt(user_text: str, state: DialogueState, *, resets_topic: bool) -> str:
    rules = [
        "毎回を質問終端にせず、open_loopは現在の流れに自然なときだけ拾う。",
        "呼び名は必要なときだけ使い、毎回名前を添えたり同じ励ましを繰り返したりしない。",
        "現在の短答・詳説・質問不要・提案不要の指定を優先し、明示的に切り替わるまで尊重する。",
        "保持していない過去の話や実体験を覚えていると装わず、曜日だけから明日などの日時を補わない。",
    ]
    if re.search(r"一文|一言|短く|簡潔", user_text):
        rules.append("今回は指定された短さを守る。一文の依頼なら挨拶を含めて一文だけにする。")
    elif re.search(r"詳しく|詳しい|具体例|コード例", user_text):
        rules.append("今回は短い相づちで済ませず、依頼された理由や具体例まで説明する。")
    if state.recent_question_count:
        rules.append("質問直後なので、回答に不要な質問を続けない。")
    if state.recent_question_count >= 2:
        rules.append("直近2回が質問傾向なので、今回は明確な必要性がない限り質問しない。")
    if state.recent_suggestion_count:
        rules.append("直近の提案と同じ行動を短期間に繰り返さない。")
    if _FATIGUE.search(user_text) and not _SUPPORT_REQUEST.search(user_text):
        rules.append("疲れを述べただけの利用者へ、毎回答える行動要求や助言を押しつけない。")
    if resets_topic:
        rules.append("利用者が閉じた話題を蒸し返さず、保存済みopen_loopも使わない。")
    return "【会話運営のクールダウン】\n- " + "\n- ".join(rules)


def _proactive_cooldown_prompt(state: DialogueState) -> str:
    """利用者がいない場面で、閉じた話や連続質問を避ける。"""

    rules = [
        "利用者からの新しい発言がない場面なので、返答や確認を強要しない。",
        "質問で終えることを避け、短い一言だけにする。",
    ]
    if state.last_act in {DialogueAct.CLOSE, DialogueAct.BOUNDARY}:
        rules.append("直前で閉じた話題や境界は、確認、言い換え、要約をせず、能動発話を始めない。")
    if state.recent_question_count:
        rules.append("直近に質問が続いているため、新しい質問を重ねない。")
    if state.recent_suggestion_count:
        rules.append("直近の提案と同じ行動を、こちらから繰り返し勧めない。")
    return "【能動会話のクールダウン】\n- " + "\n- ".join(rules)


__all__ = ["DialogueDirector", "DialogueState"]
