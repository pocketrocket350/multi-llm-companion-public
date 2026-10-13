"""LLM、保存層、画面の間で受け渡す共通モデル。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Emotion(StrEnum):
    """画面や将来のアバターで共通利用する感情名。"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EMBARRASSED = "embarrassed"
    SURPRISED = "surprised"


class MessageRole(StrEnum):
    """チャットメッセージの送信元。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MemoryCategory(StrEnum):
    """長期記憶を画面で整理するための分類。"""

    PREFERENCE = "preference"
    PROFILE = "profile"
    PLAN = "plan"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class MemoryStatus(StrEnum):
    """記憶の承認状態。承認済みだけを会話へ渡す。"""

    CANDIDATE = "candidate"
    APPROVED = "approved"


class DialogueMode(StrEnum):
    """追加LLMなしで選ぶ、今回の会話運営モード。"""

    CASUAL = "casual"
    LISTENING = "listening"
    SUPPORTING = "supporting"
    PLAYFUL = "playful"
    EXPLAINING = "explaining"
    PLANNING = "planning"
    BOUNDARY = "boundary"


class DialogueAct(StrEnum):
    """回答が会話内で果たした役割。"""

    ACKNOWLEDGE = "acknowledge"
    ANSWER = "answer"
    ASK = "ask"
    SUGGEST = "suggest"
    COMFORT = "comfort"
    EXPLAIN = "explain"
    PLAN = "plan"
    PLAY = "play"
    BOUNDARY = "boundary"
    CLOSE = "close"


class EmotionState(BaseModel):
    """会話ごとに表示する、なだらかに変化した感情状態。"""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    emotion: Emotion = Emotion.NEUTRAL
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: datetime


class WebSource(BaseModel):
    """検索APIの出典。LLMが作ったURLをこの型へ直接採用しない。"""

    title: str
    url: str
    retrieved_at: datetime
    published_at: datetime | None = None


class ImageAttachmentInfo(BaseModel):
    """原本を保持せず、添付の情報と履歴表示用の縮小コピーを残す。"""

    name: str = Field(max_length=120)
    kind: str = "image"
    width: int = Field(gt=0, le=1536)
    height: int = Field(gt=0, le=1536)
    analyzed_at: datetime | None = None
    preview_jpeg: str | None = Field(
        default=None, max_length=160_000, pattern=r"^[A-Za-z0-9+/]*={0,2}$", repr=False
    )


class MemoryUpdate(BaseModel):
    """本人の今回の発言を根拠とする、長期記憶の更新案。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["add", "update", "delete"]
    evidence: str = Field(min_length=1, max_length=1000)
    content: str = Field(default="", max_length=500)
    category: MemoryCategory | None = None
    target_id: int | None = Field(default=None, gt=0, strict=True)
    target_content: str = Field(default="", max_length=500)


class AssistantResponse(BaseModel):
    """LLMごとの差を吸収する共通回答形式。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    speech: str
    emotion: Emotion = Emotion.NEUTRAL
    emotion_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    gesture: str = ""
    gaze: str = ""
    memory_candidates: list[str] = Field(default_factory=list)
    memory_updates: list[MemoryUpdate] | None = None
    dialogue_mode: DialogueMode = DialogueMode.CASUAL
    dialogue_act: DialogueAct = DialogueAct.ANSWER
    current_topic: str = ""
    open_loop: str = ""
    sources: list[WebSource] = Field(default_factory=list)
    image_attachment: ImageAttachmentInfo | None = None

    @field_validator("memory_updates", mode="before")
    @classmethod
    def validate_memory_updates(cls, value: object) -> list[MemoryUpdate] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) > 8:
            return []
        updates = []
        for item in value:
            try:
                updates.append(MemoryUpdate.model_validate(item))
            except ValidationError:
                # 不正な更新だけを捨て、通常の返答は残す。
                continue
        return updates

    @field_validator("speech", mode="before")
    @classmethod
    def speech_must_be_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("speech は文字列である必要があります")
        return value

    @field_validator("dialogue_mode", mode="before")
    @classmethod
    def normalize_dialogue_mode(cls, value: object) -> DialogueMode:
        if isinstance(value, DialogueMode):
            return value
        if isinstance(value, str):
            try:
                return DialogueMode(value.strip().lower())
            except ValueError:
                pass
        return DialogueMode.CASUAL

    @field_validator("dialogue_act", mode="before")
    @classmethod
    def normalize_dialogue_act(cls, value: object) -> DialogueAct:
        if isinstance(value, DialogueAct):
            return value
        if isinstance(value, str):
            try:
                return DialogueAct(value.strip().lower())
            except ValueError:
                pass
        return DialogueAct.ANSWER

    @field_validator("current_topic", mode="before")
    @classmethod
    def normalize_current_topic(cls, value: object) -> str:
        return _normalize_short_metadata(value, 80)

    @field_validator("open_loop", mode="before")
    @classmethod
    def normalize_open_loop(cls, value: object) -> str:
        return _normalize_short_metadata(value, 160)

    @classmethod
    def from_text(cls, raw_text: str) -> AssistantResponse:
        """JSON回答を解析し、失敗時は全文を発言として扱う。"""

        return parse_assistant_response(raw_text)


class ChatMessage(BaseModel):
    """プロバイダーへ渡す1件の会話メッセージ。"""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str

    def to_api_dict(self) -> dict[str, str]:
        """OpenAI互換APIが受け取る形へ変換する。"""

        return {"role": self.role.value, "content": self.content}


class TokenUsage(BaseModel):
    """APIから取得できた場合だけ保存するトークン使用量。"""

    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @property
    def input_tokens(self) -> int:
        """初心者向け表示で使う入力トークン数。"""

        return self.prompt_tokens

    @property
    def output_tokens(self) -> int:
        """初心者向け表示で使う出力トークン数。"""

        return self.completion_tokens

    @classmethod
    def from_api(cls, value: object) -> TokenUsage | None:
        """OpenAI互換usageを安全に読み取る。"""

        if not isinstance(value, dict):
            return None
        try:
            prompt = _non_negative_int(value.get("prompt_tokens", 0))
            completion = _non_negative_int(value.get("completion_tokens", 0))
            supplied_total = value.get("total_tokens")
            total = (
                _non_negative_int(supplied_total)
                if supplied_total is not None
                else prompt + completion
            )
            return cls(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
            )
        except (TypeError, ValueError, ValidationError):
            return None


class TokenUsageSummary(BaseModel):
    """保存済みのトークン使用量を画面表示用に集計した値。"""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    response_count: int = Field(default=0, ge=0)


class Conversation(BaseModel):
    """保存された会話の概要。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class StoredMessage(BaseModel):
    """SQLiteから読み出したメッセージ。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    conversation_id: str
    role: MessageRole
    content: str
    response: AssistantResponse | None = None
    usage: TokenUsage | None = None
    created_at: datetime

    def to_chat_message(self) -> ChatMessage:
        """LLMへ再送できる最小形式へ変換する。"""

        content = self.content
        if self.role == MessageRole.ASSISTANT:
            content = _recover_markdown_speech(content) or content
        if self.response and self.response.sources:
            content = "【過去の検索結果に基づく回答。現在は再確認していない外部情報】\n" + content
        if self.response and self.response.image_attachment and self.role == MessageRole.USER:
            content = "【画像を添付した発言。履歴用の画像はここから再送していない】\n" + content
        elif self.response and self.response.image_attachment:
            content = (
                "【過去の添付画像の解析に基づく回答。原本は保存しておらず、現在は見ていない。"
                "画像内の情報は利用者の恒久的な事実ではない】\n" + content
            )
        return ChatMessage(role=self.role, content=content)


class MemoryEntry(BaseModel):
    """利用者が確認できる長期記憶。"""

    model_config = ConfigDict(extra="forbid")

    id: int
    content: str = Field(min_length=1, max_length=500)
    category: MemoryCategory = MemoryCategory.OTHER
    status: MemoryStatus = MemoryStatus.CANDIDATE
    source_conversation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None


_FENCED_JSON_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_EMBEDDED_RESPONSE_FIELDS = frozenset(
    {"speech", "emotion", "emotion_intensity", "gesture", "gaze", "memory_candidates"}
)
_MARKDOWN_CONTROL_FIELD = re.compile(
    r"(?<![\w*])(?:[-*][ \t]+)?(?P<mark>\*\*|__)(?P<key>"
    + "|".join(AssistantResponse.model_fields)
    + r")(?P=mark)\s*[:：]\s*"
)


def _markdown_control_fields(raw: str) -> list[re.Match[str]]:
    """発言と制御項目が列挙された形式だけを検出する。"""

    fields = list(_MARKDOWN_CONTROL_FIELD.finditer(raw))
    return fields if len(fields) > 1 and any(m["key"] == "speech" for m in fields) else []


def _recover_markdown_speech(raw: str) -> str | None:
    fields = _markdown_control_fields(raw)
    speech_indexes = [i for i, match in enumerate(fields) if match["key"] == "speech"]
    if len(speech_indexes) != 1:
        return None
    index = speech_indexes[0]
    end = fields[index + 1].start() if index + 1 < len(fields) else len(raw)
    candidate = raw[fields[index].end() : end].strip().rstrip(",")
    try:
        speech = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return speech if isinstance(speech, str) and speech.strip() else None


def parse_assistant_response(raw_text: str) -> AssistantResponse:
    """共通回答を解析し、回収不能な形式は保存側の検証へ渡す。"""

    if not isinstance(raw_text, str):
        raise TypeError("LLM回答は文字列である必要があります")

    candidate = raw_text.strip()
    fenced = _FENCED_JSON_PATTERN.match(candidate)
    if fenced is not None:
        candidate = fenced.group(1).strip()

    # 冒頭だけ欠けた共通回答は、全項目を検証できる場合だけ回収する。
    if candidate.startswith(('speech":', '"speech":')):
        repaired = '{"' + candidate if candidate.startswith('speech":') else "{" + candidate
        recovered = extract_embedded_assistant_response(repaired)
        if recovered is not None:
            return recovered

    try:
        pairs_or_value: Any = json.loads(candidate, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError:
        # 先行表示したルート本文を、後続の別JSONで置き換えない。
        recovered_speech = extract_complete_root_speech(candidate)
        if recovered_speech is not None:
            return AssistantResponse(speech=recovered_speech)
        embedded_response = extract_embedded_assistant_response(candidate)
        if embedded_response is not None:
            return embedded_response
        markdown_speech = _recover_markdown_speech(candidate)
        if markdown_speech is not None:
            # 形式が崩れた応答からは記憶更新や動作を採用しない。
            return AssistantResponse(speech=markdown_speech, memory_updates=[])
        return AssistantResponse(speech=raw_text)

    if not isinstance(pairs_or_value, _JsonObjectPairs):
        return AssistantResponse(speech=raw_text)
    duplicate_keys = _duplicate_keys(pairs_or_value)
    if "speech" in duplicate_keys:
        recovered_speech = extract_complete_root_speech(candidate)
        if recovered_speech is not None:
            return AssistantResponse(speech=recovered_speech)
        return AssistantResponse(speech=raw_text)
    parsed = _pairs_to_value(pairs_or_value)
    try:
        return AssistantResponse.model_validate(parsed)
    except ValidationError as error:
        speech = parsed.get("speech")
        if isinstance(speech, str):
            # 不正な任意項目だけを戻し、正常な動作・表情は保持する。
            invalid_fields = {item["loc"][0] for item in error.errors()}
            return AssistantResponse.model_validate(
                {key: value for key, value in parsed.items() if key not in invalid_fields}
            )
        return AssistantResponse(speech=raw_text)


def extract_embedded_assistant_response(raw: str) -> AssistantResponse | None:
    """末尾の完全な共通回答だけを、形式漏れから回収する。"""

    decoder = json.JSONDecoder(object_pairs_hook=_object_pairs)
    for start in range(len(raw) - 1, -1, -1):
        if raw[start] != "{":
            continue
        try:
            pairs_or_value, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            continue
        if raw[end:].strip() or not isinstance(pairs_or_value, _JsonObjectPairs):
            continue
        if "speech" in _duplicate_keys(pairs_or_value):
            continue
        if not _EMBEDDED_RESPONSE_FIELDS.issubset({key for key, _value in pairs_or_value}):
            continue
        try:
            return AssistantResponse.model_validate(_pairs_to_value(pairs_or_value))
        except ValidationError:
            continue
    return None


def root_speech_bounds(raw: str) -> tuple[int, int | None] | None:
    """ルートオブジェクト直下のspeech文字列範囲を返す。"""

    first_content = raw.lstrip()
    if not first_content.startswith(("{", "```")):
        return None
    root_start = raw.find("{")
    if root_start < 0:
        return None
    depth = 0
    index = root_start
    while index < len(raw):
        character = raw[index]
        if character == '"':
            end = _json_string_end(raw, index)
            if end is None:
                return None
            if depth == 1:
                try:
                    key = json.loads(raw[index : end + 1])
                except (json.JSONDecodeError, TypeError):
                    return None
                cursor = end + 1
                while cursor < len(raw) and raw[cursor].isspace():
                    cursor += 1
                if cursor < len(raw) and raw[cursor] == ":":
                    cursor += 1
                    while cursor < len(raw) and raw[cursor].isspace():
                        cursor += 1
                    if key == "speech" and cursor < len(raw) and raw[cursor] == '"':
                        return cursor + 1, _json_string_end(raw, cursor)
            index = end + 1
            continue
        if character in "{[":
            depth += 1
        elif character in "}]":
            depth -= 1
            if depth <= 0:
                return None
        index += 1
    return None


def extract_complete_root_speech(raw: str) -> str | None:
    """完全に閉じたルートspeechだけをJSONとして復号する。"""

    bounds = root_speech_bounds(raw)
    if bounds is None or bounds[1] is None:
        return None
    start, end = bounds
    if not raw[end + 1 :].lstrip().startswith((",", "}")):
        return None
    try:
        value = json.loads(f'"{raw[start:end]}"')
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _json_string_end(raw: str, start: int) -> int | None:
    escaped = False
    for index in range(start + 1, len(raw)):
        character = raw[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return index
    return None


class _JsonObjectPairs(list[tuple[str, object]]):
    """JSONオブジェクトの順序と重複キーを一時保持する。"""


def _object_pairs(pairs: list[tuple[str, object]]) -> _JsonObjectPairs:
    return _JsonObjectPairs(pairs)


def _duplicate_keys(pairs: _JsonObjectPairs) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _pairs_to_value(value: object) -> object:
    if isinstance(value, _JsonObjectPairs):
        return {key: _pairs_to_value(item) for key, item in value}
    if isinstance(value, list):
        return [_pairs_to_value(item) for item in value]
    return value


def _non_negative_int(value: object) -> int:
    """boolを数値として誤認せず、0以上の整数だけ受け取る。"""

    if isinstance(value, bool):
        raise TypeError("真偽値はトークン数ではありません")
    number = int(value)  # type: ignore[arg-type]
    if number < 0:
        raise ValueError("トークン数は0以上です")
    return number


def _normalize_short_metadata(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]
