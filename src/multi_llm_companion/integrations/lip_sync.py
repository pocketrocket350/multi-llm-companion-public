"""VoiSonaの発音時刻、または日本語文から口形の時刻表を作る部品。"""

from __future__ import annotations

from dataclasses import dataclass

_MOUTH_SHAPES = frozenset({"neutral", "a", "i", "u", "e", "o"})
_PAUSE_CHARACTERS = frozenset("、。！？!?…・\n\r")


@dataclass(frozen=True, slots=True)
class LipSyncState:
    """画面へ渡す、現在の推定口形。"""

    is_active: bool = False
    mouth_shape: str = "neutral"
    is_estimated: bool = True
    is_timeline_active: bool = False


@dataclass(frozen=True, slots=True)
class _LipEvent:
    start: float
    end: float
    mouth_shape: str


class JapaneseLipSyncSchedule:
    """かなを母音へ寄せて、再生時刻ごとの口形を推定する。"""

    def __init__(
        self,
        events: tuple[_LipEvent, ...],
        start_at: float,
        *,
        is_estimated: bool = True,
    ) -> None:
        self._events = events
        self._start_at = start_at
        self._is_estimated = is_estimated

    @property
    def start_at(self) -> float:
        """再生開始として扱う時刻を返す。"""

        return self._start_at

    @property
    def end_at(self) -> float:
        """最後の音素または推定文字が終わる時刻を返す。"""

        return max((event.end for event in self._events), default=self._start_at)

    def rebased(self, start_at: float) -> JapaneseLipSyncSchedule:
        """音素間隔を保ったまま、実際の再生開始時刻へ移動する。"""

        difference = start_at - self._start_at
        return JapaneseLipSyncSchedule(
            tuple(
                _LipEvent(
                    event.start + difference,
                    event.end + difference,
                    event.mouth_shape,
                )
                for event in self._events
            ),
            start_at,
            is_estimated=self._is_estimated,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        start_at: float,
        character_duration: float,
        total_duration: float | None = None,
    ) -> JapaneseLipSyncSchedule:
        """文字列を、短い母音ごとの時刻表へ変換する。"""

        duration = max(0.04, character_duration)
        cursor = start_at
        previous_shape = "a"
        events: list[_LipEvent] = []
        for character in text:
            shape, length = _mouth_shape_for(character, previous_shape, duration)
            events.append(_LipEvent(cursor, cursor + length, shape))
            cursor += length
            if shape != "neutral":
                previous_shape = shape
        if total_duration is not None and cursor > start_at:
            scale = max(0.04, total_duration) / (cursor - start_at)
            events = [
                _LipEvent(
                    start_at + (event.start - start_at) * scale,
                    start_at + (event.end - start_at) * scale,
                    event.mouth_shape,
                )
                for event in events
            ]
        return cls(tuple(events), start_at)

    @classmethod
    def from_phoneme_timings(
        cls,
        timings: tuple[tuple[float, float, str], ...],
        *,
        start_at: float,
    ) -> JapaneseLipSyncSchedule | None:
        """VoiSonaが返した音素時刻を、画面用の口形時刻表へ変換する。"""

        events = [
            _LipEvent(start_at + start, start_at + end, _mouth_shape_for_phoneme(phoneme))
            for start, end, phoneme in timings
            if end > start >= 0.0
        ]
        if not events:
            return None
        return cls(tuple(events), start_at, is_estimated=False)

    def state_at(self, now: float) -> LipSyncState:
        """指定時刻の推定口形を返す。予定外では口を閉じる。"""

        if now < self._start_at:
            return LipSyncState()
        for event in self._events:
            if event.start <= now < event.end:
                return LipSyncState(
                    event.mouth_shape != "neutral",
                    event.mouth_shape,
                    self._is_estimated,
                    True,
                )
        return LipSyncState()

    def state_at_progress(self, progress: float) -> LipSyncState:
        """0から1の再生進捗を、予定表内の口形へ変換する。"""

        if self.end_at <= self._start_at:
            return LipSyncState()
        normalized = min(1.0, max(0.0, progress))
        if normalized >= 1.0:
            return LipSyncState()
        position = self._start_at + (self.end_at - self._start_at) * normalized
        return self.state_at(position)


def _mouth_shape_for(
    character: str,
    previous_shape: str,
    duration: float,
) -> tuple[str, float]:
    """1文字を口形へ寄せる。漢字など読みが分からない文字は開口で補う。"""

    if character in _PAUSE_CHARACTERS or character.isspace():
        return "neutral", duration * 2.2
    if character == "ー":
        return previous_shape, duration
    if character in {"っ", "ッ", "ん", "ン"}:
        return "neutral", duration * 0.7
    normalized = _to_hiragana(character)
    for vowel, characters in _VOWEL_CHARACTERS.items():
        if normalized in characters:
            return vowel, duration
    return "a", duration


def _mouth_shape_for_phoneme(value: str) -> str:
    """日本語の音素名を、少数の安全な口形へ寄せる。"""

    normalized = value.strip().lower()
    if normalized in _MOUTH_SHAPES:
        return normalized
    if normalized in {"sil", "pau", "sp", "cl", "n", "n'", "ng", "m"}:
        return "neutral"
    for shape in ("a", "i", "u", "e", "o"):
        if normalized.endswith(shape):
            return shape
    return "neutral"


def _to_hiragana(character: str) -> str:
    code = ord(character)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return character


_VOWEL_CHARACTERS = {
    "a": frozenset("あかさたなはまやらわがざだばぱぁゃゎ"),
    "i": frozenset("いきしちにひみりゐぎじぢびぴぃ"),
    "u": frozenset("うくすつぬふむゆるをぐずづぶぷぅゅ"),
    "e": frozenset("えけせてねへめれゑげぜでべぺぇ"),
    "o": frozenset("おこそとのほもよろをごぞどぼぽぉょ"),
}


def normalize_mouth_shape(value: str) -> str:
    """画面へ渡す口形を少数の安全な値へ限定する。"""

    return value if value in _MOUTH_SHAPES else "neutral"
