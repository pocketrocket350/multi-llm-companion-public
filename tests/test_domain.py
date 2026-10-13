import pytest

from multi_llm_companion.domain import DialogueAct, DialogueMode, Emotion, parse_assistant_response


@pytest.mark.parametrize("marker", ["**", "__"])
def test_markdown_response_recovers_only_speech_without_applying_controls(marker):
    raw = (
        f'こんばんは。 {marker}speech{marker}: "こんばんは。お話ししよう。" '
        f'{marker}emotion{marker}: "happy" {marker}gesture{marker}: "wave" '
        f'{marker}memory_updates{marker}: [{{"action":"forget","target_ids":[1]}}]'
    )
    response = parse_assistant_response(raw)
    assert response.speech == "こんばんは。お話ししよう。"
    assert response.emotion is Emotion.NEUTRAL
    assert response.gesture == ""
    assert response.memory_candidates == []
    assert response.memory_updates == []


def test_markdown_response_preserves_escaped_quotes_inside_speech():
    raw = '**speech**: "言葉の\\"speech\\"は発言という意味だよ。" **emotion**: "neutral"'
    assert parse_assistant_response(raw).speech == '言葉の"speech"は発言という意味だよ。'


@pytest.mark.parametrize(
    "raw",
    [
        "**大切**な話だね。",
        "speechという項目は発言を示します。",
        '説明: **speech**: "発言" というフィールドです。',
    ],
)
def test_plain_conversation_is_not_mistaken_for_markdown_controls(raw):
    assert parse_assistant_response(raw).speech == raw


def test_valid_json_becomes_common_response() -> None:
    response = parse_assistant_response(
        """{
            "speech": "こんにちは",
            "emotion": "happy",
            "emotion_intensity": 0.8,
            "gesture": "wave",
            "gaze": "user",
            "memory_candidates": ["猫が好き"]
        }"""
    )

    assert response.speech == "こんにちは"
    assert response.emotion is Emotion.HAPPY
    assert response.emotion_intensity == pytest.approx(0.8)
    assert response.memory_candidates == ["猫が好き"]


def test_invalid_json_uses_entire_answer_as_speech() -> None:
    raw = '{"speech": "途中で切れた回答"'

    response = parse_assistant_response(raw)

    assert response.speech == raw
    assert response.emotion is Emotion.NEUTRAL


def test_invalid_common_fields_also_fall_back_to_entire_answer() -> None:
    raw = '{"speech":"回答","emotion":"unknown"}'

    response = parse_assistant_response(raw)

    assert response.speech == "回答"
    assert response.emotion is Emotion.NEUTRAL
    assert response.dialogue_mode is DialogueMode.CASUAL


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        '"emotion":"unknown"',
        '"emotion_intensity":2',
        '"memory_candidates":"invalid"',
    ],
)
def test_valid_speech_survives_invalid_known_metadata(invalid_metadata: str) -> None:
    response = parse_assistant_response(f'{{"speech":"回収する回答",{invalid_metadata}}}')

    assert response.speech == "回収する回答"
    assert response.emotion is Emotion.NEUTRAL
    assert response.memory_candidates == []


def test_missing_or_non_text_speech_with_invalid_metadata_keeps_raw_fallback() -> None:
    for raw in (
        '{"emotion":"unknown"}',
        '{"speech":12,"emotion":"unknown"}',
    ):
        assert parse_assistant_response(raw).speech == raw


@pytest.mark.parametrize(
    "invalid_metadata",
    ['"emotion":"playful"', '"emotion_intensity":2', '"memory_candidates":"invalid"'],
)
def test_invalid_optional_field_preserves_valid_gesture(invalid_metadata: str) -> None:
    response = parse_assistant_response(
        '{"speech":"バキュン。","gesture":"playful_shoot","gaze":"user",' + invalid_metadata + "}"
    )
    assert response.gesture == "playful_shoot"
    assert response.gaze == "user"
    assert response.speech == "バキュン。"


def test_invalid_gesture_does_not_discard_valid_emotion() -> None:
    response = parse_assistant_response(
        '{"speech":"うれしい。","emotion":"happy","emotion_intensity":0.8,'
        '"gesture":null,"memory_candidates":[12]}'
    )
    assert response.gesture == ""
    assert response.memory_candidates == []
    assert response.emotion is Emotion.HAPPY
    assert response.emotion_intensity == pytest.approx(0.8)


def test_dialogue_metadata_is_optional_normalized_and_bounded() -> None:
    response = parse_assistant_response(
        '{"speech":"回答","dialogue_mode":"PLANNING","dialogue_act":"suggest",'
        '"current_topic":" 旅行  の 計画 ","open_loop":"' + "次" * 200 + '"}'
    )

    assert response.dialogue_mode is DialogueMode.PLANNING
    assert response.dialogue_act is DialogueAct.SUGGEST
    assert response.current_topic == "旅行 の 計画"
    assert len(response.open_loop) == 160


def test_unknown_dialogue_metadata_does_not_discard_valid_speech() -> None:
    response = parse_assistant_response(
        '{"speech":"残す回答","dialogue_mode":"future",'
        '"dialogue_act":{"invalid":true},"current_topic":12,"open_loop":null}'
    )

    assert response.speech == "残す回答"
    assert response.dialogue_mode is DialogueMode.CASUAL
    assert response.dialogue_act is DialogueAct.ANSWER
    assert response.current_topic == ""
    assert response.open_loop == ""


def test_complete_root_speech_survives_invalid_trailing_metadata() -> None:
    raw = '{"speech":"先読み成功。","dialogue_mode": ???}'

    response = parse_assistant_response(raw)

    assert response.speech == "先読み成功。"
    assert response.dialogue_mode is DialogueMode.CASUAL
    assert response.dialogue_act is DialogueAct.ANSWER
    assert response.current_topic == ""
    assert response.open_loop == ""


def test_trailing_complete_response_recovers_from_non_json_prefix() -> None:
    raw = (
        "わかった、同じ質問は続けないね。JSON? Wait must JSON object.\n"
        '{"speech":"わかった、同じ質問は続けないね。","emotion":"neutral",'
        '"emotion_intensity":0.2,"gesture":"nod","gaze":"user",'
        '"memory_candidates":[],"dialogue_mode":"casual",'
        '"dialogue_act":"acknowledge","current_topic":"質問の頻度","open_loop":""}'
    )

    response = parse_assistant_response(raw)

    assert response.speech == "わかった、同じ質問は続けないね。"
    assert response.dialogue_act is DialogueAct.ACKNOWLEDGE


def test_embedded_json_without_response_contract_keeps_raw_fallback() -> None:
    raw = '説明の末尾に {"speech":"例"} がある。'

    assert parse_assistant_response(raw).speech == raw


def test_missing_or_incomplete_root_speech_keeps_raw_fallback() -> None:
    for raw in (
        '{"speech":"途中"',
        '{"dialogue_mode": ???}',
        '[{"speech":"配列内"}, ???]',
    ):
        assert parse_assistant_response(raw).speech == raw


def test_duplicate_root_speech_uses_first_complete_value() -> None:
    response = parse_assistant_response('{"speech":"first.","speech":"second."}')

    assert response.speech == "first."
    assert response.emotion is Emotion.NEUTRAL
    assert response.dialogue_mode is DialogueMode.CASUAL


def test_missing_opening_brace_and_quote_in_response_are_recovered() -> None:
    raw = (
        'speech":"疲れたね。","emotion":"sad","emotion_intensity":0.4,'
        '"gesture":"reassure","gaze":"user","memory_candidates":[]}'
    )
    response = parse_assistant_response(raw)
    assert response.speech == "疲れたね。"
    assert response.emotion is Emotion.SAD


def test_complete_speech_as_last_field_survives_broken_metadata() -> None:
    assert parse_assistant_response('{"dialogue_mode":???,"speech":"回答。"}').speech == "回答。"


def test_first_root_speech_is_not_replaced_by_a_second_json_object() -> None:
    raw = (
        '{"speech":"先に確定した文。"}\n'
        '{"speech":"別の文。","emotion":"neutral","emotion_intensity":0,'
        '"gesture":"none","gaze":"user","memory_candidates":[]}'
    )
    assert parse_assistant_response(raw).speech == "先に確定した文。"
