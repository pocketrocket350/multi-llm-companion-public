import pytest

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.streaming import SpeechSentenceBuffer, SpeechStreamFilter


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_markdown_fields_never_reach_speech_or_stored_conversation(chunk_size):
    raw = (
        'こんばんは。 **speech**: "こんばんは。お話ししよう。" '
        '**emotion**: "happy" **gesture**: "wave" **memory_updates**: []'
    )
    repository = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repository, MemoryConfig())
    service = ChatService(
        provider=MockProvider(response_text=raw, chunk_size=chunk_size),
        repository=repository,
        memory_service=memory,
    )
    try:
        assert "".join(service.stream_speech("私は紅茶が好きです")) == "こんばんは。お話ししよう。"
        assert service.get_current_messages()[-1].content == "こんばんは。お話ししよう。"
        assert service.last_response.gesture == ""
        assert service.last_response.memory_updates == []
        assert memory.list_memories() == []
    finally:
        service.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "raw",
    [
        '**speech**: "閉じていない **emotion**: "happy"',
        '**speech**: 123 **emotion**: "happy"',
        '**speech**: "一つ目" **speech**: "二つ目" **emotion**: "happy"',
    ],
)
def test_broken_markdown_response_is_not_spoken_or_saved(raw, streaming):
    service = ChatService(
        provider=MockProvider(response_text=raw, chunk_size=1),
        repository=SQLiteConversationRepository(":memory:"),
    )
    emitted = []
    try:
        with pytest.raises(RuntimeError, match="回答形式"):
            if streaming:
                emitted.extend(service.stream_speech("確認"))
            else:
                service.send_message("確認")
        assert emitted == []
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()


def test_plain_text_is_streamed_unchanged() -> None:
    stream_filter = SpeechStreamFilter()

    assert stream_filter.feed("こん") == "こん"
    assert stream_filter.feed("にちは") == "にちは"
    assert stream_filter.raw_text == "こんにちは"


def test_only_speech_is_streamed_from_structured_json() -> None:
    stream_filter = SpeechStreamFilter()

    chunks = [
        '{"spe',
        'ech":"こん',
        'にちは\\n世界","emotion":"happy"}',
    ]

    assert "".join(stream_filter.feed(chunk) for chunk in chunks) == "こんにちは\n世界"


def test_json_prefix_is_not_displayed_while_waiting_for_speech() -> None:
    stream_filter = SpeechStreamFilter()

    assert stream_filter.feed('{"emotion":"neutral",') == ""
    assert stream_filter.feed('"speech":"回答') == "回答"


def test_nested_unknown_metadata_speech_is_never_streamed() -> None:
    stream_filter = SpeechStreamFilter()
    chunks = [
        '{"metadata":{"speech":"偽',
        'の発言"},"speech":"正',
        'しい発言","dialogue_mode":"casual"}',
    ]

    assert "".join(stream_filter.feed(chunk) for chunk in chunks) == "正しい発言"


def test_root_speech_streams_before_trailing_metadata_arrives() -> None:
    stream_filter = SpeechStreamFilter()

    assert stream_filter.feed('{"speech":"最初の一文。') == "最初の一文。"
    assert stream_filter.feed('","metadata":{"large":"') == ""


def test_markdown_fence_is_not_shown_when_json_is_streamed() -> None:
    stream_filter = SpeechStreamFilter()

    assert stream_filter.feed("```json\n{") == ""
    assert stream_filter.feed('"speech":"発言だけ') == "発言だけ"


def test_structured_mode_holds_plain_prefix_until_the_final_response() -> None:
    stream_filter = SpeechStreamFilter(require_structured=True)

    assert stream_filter.feed("JSON形式を考え中") == ""
    assert stream_filter.finish("自然な返答です。") == "自然な返答です。"


def test_structured_mode_waits_for_validation_after_a_leaked_prefix() -> None:
    stream_filter = SpeechStreamFilter(require_structured=True)

    assert stream_filter.feed("形式漏れ\n{") == ""
    assert stream_filter.feed('"speech":"自然な') == ""
    assert stream_filter.feed('返答。","emotion":"neutral"}') == ""
    assert stream_filter.finish("自然な返答。") == "自然な返答。"


def test_sentence_buffer_releases_completed_sentences_and_flushes_tail() -> None:
    buffer = SpeechSentenceBuffer()

    assert buffer.feed("最初") == []
    assert buffer.feed("です。次") == ["最初です。"]
    assert buffer.feed("です！") == ["次です！"]
    assert buffer.flush() == ""


def test_sentence_buffer_keeps_closing_quote_with_sentence() -> None:
    buffer = SpeechSentenceBuffer()

    assert buffer.feed("『雨です。』続き") == ["『雨です。』"]
    assert buffer.flush() == "続き"


@pytest.mark.parametrize("size", [1, 2, 3, 8, 64])
def test_sentence_buffer_keeps_punctuation_inside_a_title(size: int) -> None:
    text = "楽曲「粛聖!! ロリ神レクイエム☆」が話題です。続きも読みます。"
    buffer = SpeechSentenceBuffer()
    spoken = []
    for index in range(0, len(text), size):
        spoken.extend(buffer.feed(text[index : index + size]))
    if tail := buffer.flush():
        spoken.append(tail)
    assert spoken == ["楽曲「粛聖!! ロリ神レクイエム☆」が話題です。", "続きも読みます。"]


@pytest.mark.parametrize(
    "text", ["本当！？次です。", "はい!! 次です。", "!?はい。", "「『何!?』という本」です。"]
)
def test_sentence_buffer_never_emits_punctuation_only_at_any_chunk_boundary(text: str) -> None:
    for cut in range(len(text) + 1):
        buffer = SpeechSentenceBuffer()
        spoken = buffer.feed(text[:cut]) + buffer.feed(text[cut:])
        if tail := buffer.flush():
            spoken.append(tail)
        assert all(any(char.isalnum() for char in sentence) for sentence in spoken)
        assert "".join(char for char in "".join(spoken) if char.isalnum()) == "".join(
            char for char in text if char.isalnum()
        )


def test_sentence_buffer_punctuation_tail_and_reset_do_not_leak_into_next_turn() -> None:
    buffer = SpeechSentenceBuffer()
    assert buffer.feed("はい!") == ["はい!"]
    assert buffer.feed("!」") == []
    assert buffer.flush() == ""
    assert buffer.feed("「未完了") == []
    buffer.reset()
    assert buffer.feed("新しい文。") == ["新しい文。"]


def test_chat_stream_speech_matches_final_response_and_database_content() -> None:
    raw = (
        '{"speech":"逐次\u8868示です。","metadata":{"speech":"偽物"},'
        '"dialogue_mode":"explaining","dialogue_act":"answer"}'
    )
    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(response_text=raw, chunk_size=3)
    service = ChatService(provider=provider, repository=repository)

    streamed = "".join(service.stream_speech("確認"))
    response = provider.last_response
    messages = service.get_current_messages()

    assert response is not None
    assert streamed == response.speech == "逐次表示です。"
    assert messages[-1].content == response.speech
    assert messages[-1].response == response
    service.close()


def test_complete_speech_with_invalid_trailing_metadata_matches_saved_response() -> None:
    raw = '{"speech":"先読み成功。","dialogue_mode": ???}'
    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(response_text=raw, chunk_size=2)
    service = ChatService(provider=provider, repository=repository)

    streamed = "".join(service.stream_speech("部分JSON確認"))
    response = provider.last_response
    saved = service.get_current_messages()[-1]

    assert response is not None
    assert streamed == response.speech == saved.content == "先読み成功。"
    assert saved.response == response
    assert response.dialogue_mode.value == "casual"
    assert response.dialogue_act.value == "answer"
    service.close()


def test_streamed_prefix_leak_is_replaced_by_the_structured_response() -> None:
    raw = (
        "JSON? Wait.\n"
        '{"speech":"自然な返答です。","emotion":"neutral","emotion_intensity":0.2,'
        '"gesture":"none","gaze":"forward","memory_candidates":[]}'
    )
    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(response_text=raw, chunk_size=3)
    service = ChatService(provider=provider, repository=repository)

    streamed = "".join(service.stream_speech("構造確認"))
    response = provider.last_response

    assert response is not None
    assert streamed == response.speech == "自然な返答です。"
    assert service.get_current_messages()[-1].content == response.speech
    service.close()


def test_invalidated_stream_does_not_save_old_assistant_response() -> None:
    def responder(messages: object) -> str:
        last = list(messages)[-1].content
        return f'{{"speech":"{last}への回答"}}'

    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(responder=responder, chunk_size=1)
    service = ChatService(provider=provider, repository=repository)

    old_stream = service.stream_message("古い発話")
    next(old_stream)
    service._invalidate_active_stream()
    list(old_stream)
    list(service.stream_message("新しい発話"))

    assert [
        (message.role.value, message.content) for message in service.get_current_messages()
    ] == [
        ("user", "古い発話"),
        ("user", "新しい発話"),
        ("assistant", "新しい発話への回答"),
    ]
    service.close()


def test_invalidated_or_failed_stream_does_not_leave_memory_candidates() -> None:
    repository = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repository, MemoryConfig())
    provider = MockProvider(
        response_text='{"speech":"回答","memory_candidates":["ユーザーは紅茶が好き"]}',
        chunk_size=1,
    )
    service = ChatService(provider=provider, repository=repository, memory_service=memory)

    old_stream = service.stream_message("ユーザーは紅茶が好き")
    next(old_stream)
    service._invalidate_active_stream()
    list(old_stream)

    assert memory.last_candidates == []
    assert repository.list_memories() == []

    provider._responder = lambda _messages: (_ for _ in ()).throw(RuntimeError("失敗"))  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="失敗"):
        list(service.stream_message("ユーザーは緑茶が好き"))
    assert memory.last_candidates == []
    assert repository.list_memories() == []
    service.close()


def test_valid_speech_with_invalid_known_metadata_matches_saved_response() -> None:
    raw = '{"speech":"回答。","emotion":"unknown"}'
    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(response_text=raw, chunk_size=2)
    service = ChatService(provider=provider, repository=repository)

    streamed = "".join(service.stream_speech("既知metadata確認"))
    response = provider.last_response
    saved = service.get_current_messages()[-1]

    assert response is not None
    assert streamed == response.speech == saved.content == "回答。"
    assert saved.response == response
    assert response.emotion.value == "neutral"
    service.close()


def test_duplicate_root_speech_keeps_first_streamed_and_saved_value() -> None:
    raw = '{"speech":"first.","speech":"second."}'
    repository = SQLiteConversationRepository(":memory:")
    provider = MockProvider(response_text=raw, chunk_size=2)
    service = ChatService(provider=provider, repository=repository)

    streamed = "".join(service.stream_speech("重複speech確認"))
    response = provider.last_response
    saved = service.get_current_messages()[-1]

    assert response is not None
    assert streamed == response.speech == saved.content == "first."
    assert saved.response == response
    assert response.dialogue_mode.value == "casual"
    service.close()


def test_embedded_example_is_not_emitted_before_the_final_response() -> None:
    stream_filter = SpeechStreamFilter(require_structured=True)
    assert stream_filter.feed('形式の例: {"speech":"例です。"}') == ""
    assert stream_filter.feed('\n実際の回答: {"speech":"本当の返答。"}') == ""
    assert stream_filter.finish("本当の返答。") == "本当の返答。"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"dialogue_mode":???,"speech":"ok"}', "ok"),
        (
            'prefix {"speech":"ANSWER","dialogue_act":"answer"}',
            None,
        ),
    ],
)
def test_held_or_recovered_response_is_emitted_only_once(raw: str, expected: str | None) -> None:
    service = ChatService(
        provider=MockProvider(response_text=raw, chunk_size=1),
        repository=SQLiteConversationRepository(":memory:"),
    )
    try:
        if expected is None:
            emitted = []
            with pytest.raises(RuntimeError, match="回答形式"):
                emitted.extend(service.stream_speech("確認"))
            assert emitted == []
            assert len(service.get_current_messages()) == 1
            return
        assert "".join(service.stream_speech("確認")) == expected
        assert service.get_current_messages()[-1].content == expected
    finally:
        service.close()


def test_invalidated_pending_stream_never_flushes_old_response() -> None:
    service = ChatService(
        provider=MockProvider(response_text="通常の返答。", chunk_size=1),
        repository=SQLiteConversationRepository(":memory:"),
    )
    original = service.provider.stream_chat

    def invalidate_during_generation(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            service._invalidate_active_stream()
            yield chunk

    service.provider.stream_chat = invalidate_during_generation
    try:
        assert list(service.stream_speech("取り消す")) == []
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()


def test_malformed_response_is_not_saved_or_flushed_as_json() -> None:
    service = ChatService(
        provider=MockProvider(response_text='{"speech":"途中の文。"', chunk_size=1),
        repository=SQLiteConversationRepository(":memory:"),
    )
    emitted = []
    try:
        with pytest.raises(RuntimeError, match="回答形式"):
            emitted.extend(service.stream_speech("形式不良の検査"))
        assert "{" not in "".join(emitted)
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()


def test_structured_stream_stops_new_chunks_immediately_after_invalidation() -> None:
    service = ChatService(
        provider=MockProvider(response_text='{"speech":"古い返答です。"}', chunk_size=1),
        repository=SQLiteConversationRepository(":memory:"),
    )
    try:
        stream = service.stream_speech("確認")
        assert next(stream) == "古"
        service._invalidate_active_stream()
        assert list(stream) == []
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()
