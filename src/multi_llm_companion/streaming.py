"""構造化JSONのうち、speechだけを可能な範囲で逐次取り出す。"""

from __future__ import annotations

from multi_llm_companion.domain import root_speech_bounds

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _partial_json_string(raw: str, start: int) -> str:
    """終端前でも、確定済みのJSON文字列部分だけを戻す。"""

    result: list[str] = []
    index = start
    while index < len(raw):
        character = raw[index]
        if character == '"':
            break
        if character != "\\":
            result.append(character)
            index += 1
            continue

        if index + 1 >= len(raw):
            break
        escaped = raw[index + 1]
        if escaped == "u":
            end = index + 6
            if end > len(raw):
                break
            digits = raw[index + 2 : end]
            try:
                codepoint = int(digits, 16)
                if 0xD800 <= codepoint <= 0xDBFF:
                    # 分割されたサロゲートペアは、後半が届くまで表示しない。
                    if end + 6 > len(raw) or raw[end : end + 2] != "\\u":
                        break
                    low = int(raw[end + 2 : end + 6], 16)
                    if not 0xDC00 <= low <= 0xDFFF:
                        break
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + low - 0xDC00
                    end += 6
                elif 0xDC00 <= codepoint <= 0xDFFF:
                    break
                result.append(chr(codepoint))
            except ValueError:
                break
            index = end
            continue
        decoded = _ESCAPES.get(escaped)
        if decoded is None:
            break
        result.append(decoded)
        index += 2
    return "".join(result)


class SpeechStreamFilter:
    """通常文はそのまま、JSONならspeechフィールドだけを流す。"""

    def __init__(self, *, require_structured: bool = False) -> None:
        self._raw = ""
        self._mode: str | None = None
        self._emitted_speech = ""
        self._require_structured = require_structured

    @property
    def raw_text(self) -> str:
        return self._raw

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._raw += chunk

        if self._mode is None:
            first_content = self._raw.lstrip()
            if not first_content:
                return ""
            if first_content[0] in {"{", "[", "`"}:
                self._mode = "json"
            elif self._require_structured:
                self._mode = "pending"
            else:
                self._mode = "plain"

        if self._mode == "plain":
            new_text = self._raw[len(self._emitted_speech) :]
            self._emitted_speech += new_text
            return new_text

        if self._mode == "pending":
            # 前置き内の例示を、本物の回答より先に読み上げない。
            return ""
        bounds = root_speech_bounds(self._raw)
        if bounds is None:
            return ""
        speech_start, _speech_end = bounds
        speech = _partial_json_string(self._raw, speech_start)
        new_text = speech[len(self._emitted_speech) :]
        self._emitted_speech += new_text
        return new_text

    def finish(self, speech: str) -> str:
        """形式違反時は、解析済みの最終本文だけを一度だけ返す。"""

        if not speech:
            return ""
        if speech.startswith(self._emitted_speech):
            remainder = speech[len(self._emitted_speech) :]
            self._emitted_speech = speech
            return remainder
        raise RuntimeError("回答形式が途中の本文と一致しないため、追加の読み上げを止めました。")


class SpeechSentenceBuffer:
    """逐次表示中の発話を、読み上げ可能な文へ分割する。"""

    _BOUNDARY = frozenset("。！？!?\n")
    _QUOTES = {"「": "」", "『": "』", "“": "”", '"': '"'}
    _CLOSING = frozenset("」』”）)]\"'")

    def __init__(self) -> None:
        self._buffer = ""

    def reset(self) -> None:
        """未送出の本文を破棄して、新しい発話へ切り替える。"""

        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """本文を追加し、終端記号まで確定した文だけを返す。"""

        if text:
            self._buffer += text
        sentences: list[str] = []
        start = 0
        quotes: list[str] = []
        for index, character in enumerate(self._buffer):
            if index < start:
                continue
            if quotes and character == quotes[-1]:
                quotes.pop()
            elif character in self._QUOTES:
                quotes.append(self._QUOTES[character])
            if character not in self._BOUNDARY:
                continue
            end = index + 1
            remaining_quotes = quotes.copy()
            while end < len(self._buffer) and (
                self._buffer[end] in self._BOUNDARY or self._buffer[end] in self._CLOSING
            ):
                if remaining_quotes and self._buffer[end] == remaining_quotes[-1]:
                    remaining_quotes.pop()
                end += 1
            # 作品名などの引用途中や、記号だけの断片を合成へ送らない。
            if remaining_quotes:
                continue
            sentence = self._buffer[start:end].strip()
            if any(character.isalnum() for character in sentence):
                sentences.append(sentence)
                start = end
                quotes = remaining_quotes
        if start:
            self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        """回答終了時に、終端記号のない残りを一文として確定する。"""

        sentence = self._buffer.strip()
        self._buffer = ""
        return sentence if any(character.isalnum() for character in sentence) else ""
