"""会話状態と、一つの発話意図から作る出力別の表現計画。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .models import AssistantResponse, Emotion


class ConversationState(StrEnum):
    """画面、音声、自然動作で共有する会話状態。"""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    REACTING = "reacting"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"

    @property
    def display_name(self) -> str:
        """画面へ表示する初心者向けの状態名。"""

        return {
            self.IDLE: "待機中",
            self.LISTENING: "聞き取り中",
            self.THINKING: "回答を作成中",
            self.REACTING: "姿勢を準備中",
            self.SPEAKING: "読み上げ中",
            self.INTERRUPTED: "音声を停止しました",
        }[self]


class SpeechAct(StrEnum):
    """発言の大まかな役割。動作判断の入力だけに使う。"""

    STATEMENT = "statement"
    QUESTION = "question"


class ExpressionOutputStatus(StrEnum):
    """各出力が今回の発話へ適用される状態。"""

    APPLIED = "applied"
    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ExpressionChannel:
    """一つの出力へ渡す、上限処理済みの感情と強度。"""

    emotion: Emotion
    intensity: float
    status: ExpressionOutputStatus = ExpressionOutputStatus.APPLIED


@dataclass(frozen=True, slots=True)
class ExpressionPlan:
    """音声と見た目を同じ発話意図から決定した結果。"""

    source_intensity: float
    voice: ExpressionChannel
    expression: ExpressionChannel
    gesture: ExpressionChannel
    gaze: ExpressionChannel
    natural_motion: ExpressionChannel
    corrections: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()

    @property
    def status_text(self) -> str:
        """適用値、無効、未対応を一行で判別できる表示を返す。"""

        return self._status_text(
            (
                ("音声", self.voice),
                ("表情", self.expression),
                ("動作", self.gesture),
                ("視線", self.gaze),
                ("自然動作", self.natural_motion),
            ),
            include_unsupported=True,
        )

    @property
    def visual_status_text(self) -> str:
        """PMX側へ、音声設定と混同しない見た目だけの状態を返す。"""

        return self._status_text(
            (
                ("表情", self.expression),
                ("動作", self.gesture),
                ("視線", self.gaze),
                ("自然動作", self.natural_motion),
            ),
            include_unsupported=False,
        )

    def _status_text(
        self,
        channels: tuple[tuple[str, ExpressionChannel], ...],
        *,
        include_unsupported: bool,
    ) -> str:
        """指定された出力の適用状態を共通書式へ整える。"""

        parts = []
        for label, channel in channels:
            if channel.status == ExpressionOutputStatus.APPLIED:
                value = f"{round(channel.intensity * 100)}%"
            elif channel.status == ExpressionOutputStatus.DISABLED:
                value = "無効"
            else:
                value = "未対応"
            parts.append(f"{label} {value}")
        if self.diagnostics:
            parts.append("診断あり")
        elif self.corrections:
            parts.append("安全補正済み")
        else:
            parts.append("矛盾なし")
        if include_unsupported and self.unsupported:
            parts.append("ボイス固有スタイル未対応")
        return " / ".join(parts)


@dataclass(frozen=True, slots=True)
class ExpressionIntent:
    """LLM固有の表記を除いた、一回分の音声・表情・動作・視線の意図。"""

    emotion: Emotion
    intensity: float
    gesture_intent: str
    gaze_intent: str
    speech_act: SpeechAct
    conversation_state: ConversationState
    outputs: ExpressionPlan | None = None


class ExpressionPolicy:
    """既存の共通回答を壊さず、安全で整合した表現計画へ変換する。"""

    def __init__(
        self,
        *,
        supported_gesture_intents: tuple[str, ...] = (),
        supported_gaze_intents: tuple[str, ...] = (),
        voice_expression_enabled: bool = False,
    ) -> None:
        self._supported_gestures = {
            "none",
            "wave",
            "nod",
            *(normalize_gesture_intent(value) for value in supported_gesture_intents),
        }
        self._supported_gazes = {
            "user",
            "forward",
            "left",
            "right",
            "up",
            "down",
            *(normalize_gaze_intent(value) for value in supported_gaze_intents),
        }
        self._voice_expression_enabled = voice_expression_enabled

    def normalize(
        self,
        response: AssistantResponse,
        *,
        conversation_state: ConversationState = ConversationState.REACTING,
    ) -> ExpressionIntent:
        """自由文字列を検証し、出力別の強度を一度だけ作る。"""

        corrections: list[str] = []
        gesture = normalize_gesture_intent(response.gesture)
        if gesture not in self._supported_gestures:
            gesture = "none"
            corrections.append("未対応ジェスチャーをnoneへ補正")
        gaze = normalize_gaze_intent(response.gaze)
        if gaze not in self._supported_gazes:
            gaze = "forward"
            corrections.append("未対応視線をforwardへ補正")
        speech_act = (
            SpeechAct.QUESTION
            if response.speech.rstrip().endswith(("?", "？"))
            else SpeechAct.STATEMENT
        )
        intensity = _clamp(response.emotion_intensity)
        outputs = build_expression_plan(
            response.emotion,
            intensity,
            gesture_intent=gesture,
            gaze_intent=gaze,
            voice_expression_enabled=self._voice_expression_enabled,
            corrections=tuple(corrections),
        )
        return ExpressionIntent(
            emotion=response.emotion,
            intensity=intensity,
            gesture_intent=gesture,
            gaze_intent=gaze,
            speech_act=speech_act,
            conversation_state=conversation_state,
            outputs=outputs,
        )

    def apply_emotion_state(
        self,
        intent: ExpressionIntent,
        emotion: Emotion,
        intensity: float,
    ) -> ExpressionIntent:
        """なだらかにした感情へ置き換え、全出力を同じ規則で再計算する。"""

        normalized_intensity = _clamp(intensity)
        outputs = build_expression_plan(
            emotion,
            normalized_intensity,
            gesture_intent=intent.gesture_intent,
            gaze_intent=intent.gaze_intent,
            voice_expression_enabled=self._voice_expression_enabled,
            corrections=intent.outputs.corrections if intent.outputs is not None else (),
        )
        return replace(
            intent,
            emotion=emotion,
            intensity=normalized_intensity,
            outputs=outputs,
        )


_OUTPUT_CAPS = {
    "voice": 0.55,
    "expression": 0.82,
    "gesture": 0.68,
    "gaze": 0.50,
    "natural_motion": 0.32,
}

_EMOTION_MULTIPLIERS: dict[Emotion, dict[str, float]] = {
    Emotion.NEUTRAL: {
        "voice": 0.0,
        "expression": 0.0,
        "gesture": 0.0,
        "gaze": 0.0,
        "natural_motion": 0.0,
    },
    Emotion.HAPPY: {
        "voice": 1.0,
        "expression": 1.0,
        "gesture": 0.90,
        "gaze": 0.70,
        "natural_motion": 0.80,
    },
    Emotion.SAD: {
        "voice": 0.70,
        "expression": 0.82,
        "gesture": 0.42,
        "gaze": 0.55,
        "natural_motion": 0.50,
    },
    Emotion.ANGRY: {
        "voice": 0.82,
        "expression": 0.76,
        "gesture": 0.62,
        "gaze": 0.58,
        "natural_motion": 0.58,
    },
    Emotion.EMBARRASSED: {
        "voice": 0.52,
        "expression": 0.78,
        "gesture": 0.38,
        "gaze": 0.46,
        "natural_motion": 0.44,
    },
    Emotion.SURPRISED: {
        "voice": 0.90,
        "expression": 0.92,
        "gesture": 0.72,
        "gaze": 0.82,
        "natural_motion": 0.68,
    },
}


def build_expression_plan(
    emotion: Emotion,
    intensity: float,
    *,
    gesture_intent: str = "none",
    gaze_intent: str = "forward",
    voice_expression_enabled: bool = False,
    corrections: tuple[str, ...] = (),
) -> ExpressionPlan:
    """感情強度を、曲線と上限が異なる五つの出力へ変換する。"""

    source = _clamp(intensity)
    multiplier = _EMOTION_MULTIPLIERS[emotion]
    voice = _channel(
        emotion,
        source**1.15,
        "voice",
        multiplier["voice"],
        status=(
            ExpressionOutputStatus.APPLIED
            if voice_expression_enabled
            else ExpressionOutputStatus.DISABLED
        ),
    )
    expression = _channel(emotion, source**0.85, "expression", multiplier["expression"])
    gesture = _channel(
        emotion,
        source**1.25,
        "gesture",
        multiplier["gesture"],
        status=(
            ExpressionOutputStatus.APPLIED
            if gesture_intent not in {"", "none"}
            else ExpressionOutputStatus.DISABLED
        ),
    )
    if gesture.status == ExpressionOutputStatus.APPLIED and gesture.intensity == 0.0:
        gesture = ExpressionChannel(emotion, _OUTPUT_CAPS["gesture"], gesture.status)
    gaze = _channel(
        emotion,
        source,
        "gaze",
        multiplier["gaze"],
        status=(
            ExpressionOutputStatus.APPLIED
            if gaze_intent not in {"", "forward"}
            else ExpressionOutputStatus.DISABLED
        ),
    )
    natural_motion = _channel(
        emotion,
        source**0.95,
        "natural_motion",
        multiplier["natural_motion"],
    )
    channels = {
        "voice": voice,
        "expression": expression,
        "gesture": gesture,
        "gaze": gaze,
        "natural_motion": natural_motion,
    }
    return ExpressionPlan(
        source_intensity=source,
        voice=voice,
        expression=expression,
        gesture=gesture,
        gaze=gaze,
        natural_motion=natural_motion,
        corrections=corrections,
        diagnostics=diagnose_expression_channels(channels),
        unsupported=("VoiSonaボイス固有style_weights",),
    )


def diagnose_expression_channels(
    channels: dict[str, ExpressionChannel],
    *,
    threshold: float = 0.35,
) -> tuple[str, ...]:
    """強い正負感情の衝突だけを検出し、弱い表現は誤検出しない。"""

    polarity = {
        Emotion.HAPPY: "positive",
        Emotion.EMBARRASSED: "positive",
        Emotion.SAD: "negative",
        Emotion.ANGRY: "negative",
    }
    strong = {
        name: polarity.get(channel.emotion)
        for name, channel in channels.items()
        if channel.status == ExpressionOutputStatus.APPLIED
        and channel.intensity >= threshold
        and polarity.get(channel.emotion) is not None
    }
    values = {value for value in strong.values() if value is not None}
    if len(values) <= 1:
        return ()
    names = "、".join(sorted(strong))
    return (f"強い感情方向が出力間で矛盾しています: {names}",)


def _channel(
    emotion: Emotion,
    curved_intensity: float,
    output: str,
    multiplier: float,
    *,
    status: ExpressionOutputStatus = ExpressionOutputStatus.APPLIED,
) -> ExpressionChannel:
    value = min(_OUTPUT_CAPS[output], curved_intensity * _OUTPUT_CAPS[output] * multiplier)
    if status != ExpressionOutputStatus.APPLIED:
        value = 0.0
    return ExpressionChannel(emotion, round(_clamp(value), 4), status)


def normalize_gesture_intent(value: str) -> str:
    """同じ意味の動作名を、第34段階の意味タグへ揃える。"""

    normalized = _normalize_tag(value)
    if normalized in {"", "none", "no_gesture", "なし"}:
        return "none"
    if (
        "wave" in normalized
        or normalized in {"greeting", "farewell", "hello", "goodbye"}
        or "手を振" in value
    ):
        return "wave"
    if (
        "nod" in normalized
        or normalized in {"agreement", "gentle_agreement", "small_agreement"}
        or "うなず" in value
    ):
        return "nod"
    return normalized


def normalize_gaze_intent(value: str) -> str:
    """視線の別表記を、共通の六方向へ揃える。"""

    normalized = _normalize_tag(value)
    aliases = {
        "": "forward",
        "front": "forward",
        "viewer": "user",
        "camera": "user",
        "利用者": "user",
        "正面": "forward",
        "左": "left",
        "右": "right",
        "上": "up",
        "下": "down",
    }
    return aliases.get(normalized, normalized)


def _normalize_tag(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
