"""研究応答の逐次表示・終端検証・取消・保存をHTTP Mockで確認する。"""

import json
import threading

import httpx
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from test_application import _runtime
from test_web_search import client_for, search_payload
from test_web_search_ui import wait_until

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import AssistantResponse, ChatMessage
from multi_llm_companion.providers import OpenAICompatibleProvider, ProviderResponseError
from multi_llm_companion.providers._research import _ResearchCall
from multi_llm_companion.services import ChatService, MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.streaming import SpeechStreamFilter
from multi_llm_companion.ui.chat_window import ChatWindow


def event(delta, finish=None):
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False
        )
        + "\n\n"
    ).encode()


def tool_delta(arguments, *, name=None, index=0):
    call = {"index": index, "function": {"arguments": arguments}}
    if name is not None:
        call.update(id="artificial-call", type="function")
        call["function"]["name"] = name
    return {"tool_calls": [call]}


def frames(parts, name="reply_to_user", finish="tool_calls", *, done=True):
    data = [event(tool_delta(part, name=name if i == 0 else None)) for i, part in enumerate(parts)]
    data.append(event({}, finish))
    data.append(
        b'data: {"choices":[],"usage":{"prompt_tokens":5,'
        b'"completion_tokens":3,"total_tokens":8}}\n\n'
    )
    if done:
        data.append(b"data: [DONE]\n\n")
    return data


class Body(httpx.SyncByteStream):
    def __init__(self, data, before=None):
        self.data = data
        self.before = before
        self.closed = False

    def __iter__(self):
        for index, part in enumerate(self.data):
            if self.before:
                self.before(index)
            yield part

    def close(self):
        self.closed = True


def response(data, before=None):
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=Body(data, before)
    )


def provider(handler):
    def guarded(request):
        payload = json.loads(request.content)
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert payload["model"] == "gpt-6-luna"
        assert payload["stream"] is True
        assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False
        assert {tool["function"]["name"] for tool in payload["tools"]} == {
            "reply_to_user",
            "request_web_search",
        }
        return handler(request)

    return OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="artificial",
        model="gpt-6-luna",
        transport=httpx.MockTransport(guarded),
    )


def test_reply_arguments_stream_before_metadata_and_validate_once():
    parts = ['{"speech":"先の文。', '後の文。","memory_updates":[],', '"emotion":"neutral"}']
    p = provider(lambda _: response(frames(parts)))
    stream = p._stream_research_reply([ChatMessage(role="user", content="人工入力")])
    first = next(stream)
    assert first == parts[0] and p.last_response is None and p.last_raw_response == ""
    rest = list(stream)
    speech = SpeechStreamFilter(require_structured=True)
    assert "".join(speech.feed(value) for value in [first, *rest[:-1]]) == "先の文。後の文。"
    assert all("memory_updates" not in value and "emotion" not in value for value in rest[:-1])
    assert isinstance(rest[-1], AssistantResponse)
    assert rest[-1].speech == "先の文。後の文。"
    assert p.last_response is rest[-1] and p.last_raw_response == "".join(parts)
    assert p.get_usage().total_tokens == 8


def test_name_and_arguments_can_arrive_in_separate_deltas():
    data = [
        event({"tool_calls": [{"index": 0, "function": {"arguments": '{"speech":"一文。"}'}}]}),
        event(tool_delta("", name="reply_to_user")),
        event({}, "tool_calls"),
        b"data: [DONE]\n\n",
    ]
    p = provider(lambda _: response(data))
    values = list(p._stream_research_reply([]))
    assert "".join(value for value in values if isinstance(value, str)) == '{"speech":"一文。"}'
    assert values[-1].speech == "一文。"


def test_search_arguments_are_held_and_return_only_validated_call():
    p = provider(
        lambda _: response(
            frames(['{"query":"内部の人工検索語",', '"detailed":false}'], name="request_web_search")
        )
    )
    values = list(p._stream_research_reply([]))
    assert values == [_ResearchCall("内部の人工検索語", False)]
    assert p.last_response is None and p.last_raw_response == ""
    assert p.get_usage().total_tokens == 8


@pytest.mark.parametrize("bad", ["unknown", "multiple", "wrong_index", "mixed_content"])
def test_invalid_envelope_never_emits_arguments(bad):
    delta = tool_delta('{"speech":"出してはいけない"}', name="reply_to_user")
    if bad == "unknown":
        delta["tool_calls"][0]["function"]["name"] = "delete_files"
    elif bad == "multiple":
        delta["tool_calls"].append(delta["tool_calls"][0].copy())
    elif bad == "wrong_index":
        delta["tool_calls"][0]["index"] = 1
    else:
        delta["content"] = "内部制御情報"
    p = provider(lambda _: response([event(delta), event({}, "tool_calls"), b"data: [DONE]\n\n"]))
    values = []
    with pytest.raises(ProviderResponseError):
        values.extend(p._stream_research_reply([]))
    assert values == [] and p.last_response is None


@pytest.mark.parametrize(
    "bad",
    [
        "length",
        "missing_finish",
        "missing_done",
        "invalid_json",
        "late_multiple",
        "late_name",
        "invalid_metadata",
    ],
)
def test_bad_completion_is_not_saved_after_partial_speech(bad):
    parts = ['{"speech":"先の文。', '後の文。"}']
    data = frames(parts)
    if bad == "length":
        data = frames(parts, finish="length")
    elif bad == "missing_finish":
        data = frames(parts, finish=None)
    elif bad == "missing_done":
        data = frames(parts, done=False)
    elif bad == "invalid_json":
        data = frames([parts[0], '後の文。"'])
    elif bad == "late_multiple":
        data.insert(1, event(tool_delta("{}", name="request_web_search", index=1)))
    elif bad == "late_name":
        data.insert(1, event(tool_delta("", name="request_web_search")))
    else:
        data = frames([parts[0], '後の文。","speech":123}'])
    service = ChatService(
        provider=provider(lambda _: response(data)),
        repository=SQLiteConversationRepository(":memory:"),
    )
    stream = service.stream_speech("人工入力")
    try:
        assert next(stream) == "先の文。"
        with pytest.raises(ProviderResponseError):
            list(stream)
        assert service.last_response is None
        assert [m.role.value for m in service.get_current_messages()] == ["user"]
    finally:
        service.close()


def test_stream_result_matches_storage_and_memory_is_applied_once(monkeypatch):
    repository = SQLiteConversationRepository(":memory:")
    memory = MemoryService(repository, MemoryConfig())
    calls = []
    original = memory.propose

    def propose(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(memory, "propose", propose)
    service = ChatService(
        provider=provider(
            lambda _: response(frames(['{"speech":"一文目。', '二文目。","memory_candidates":[]}']))
        ),
        repository=repository,
        memory_service=memory,
    )
    try:
        stream = service.stream_speech("人工入力")
        assert next(stream) == "一文目。"
        assert calls == [] and len(service.get_current_messages()) == 1
        result = "一文目。" + "".join(stream)
        assert result == service.last_response.speech == "一文目。二文目。"
        assert [m.content for m in service.get_current_messages()] == ["人工入力", result]
        assert calls == [True] and service.last_usage.total_tokens == 8
    finally:
        service.close()


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("kind", ["generation", "callback", "permission"])
def test_cancel_before_after_and_at_finish_drops_remaining_response(stage, kind):
    cancelled = [False]
    service = None

    def cancel(index):
        if index != stage:
            return
        if kind == "generation":
            service._invalidate_active_stream()
        elif kind == "permission":
            service.set_web_search_enabled(False)
        else:
            cancelled[0] = True

    service = ChatService(
        provider=provider(
            lambda _: response(frames(['{"speech":"一文目。', '二文目。"}']), cancel)
        ),
        repository=SQLiteConversationRepository(":memory:"),
    )
    try:
        values = list(service.stream_message("人工入力", cancelled=lambda: cancelled[0]))
        speech = SpeechStreamFilter(require_structured=True)
        visible = "".join(speech.feed(value) for value in values)
        assert "二文目" not in visible if stage < 2 else visible == "一文目。二文目。"
        assert service._stream_was_invalidated and service.last_response is None
        assert len(service.get_current_messages()) == 1
    finally:
        service.close()


def test_streamed_search_keeps_sources_and_never_yields_query():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "tools" in payload:
            assert payload["stream"] is True
            return response(
                frames(['{"query":"内部の人工検索語","detailed":false}'], name="request_web_search")
            )
        assert payload["stream"] is False
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"speech":"確認できたよ。[S1]"}'}}]}
        )

    p = OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="artificial",
        model="gpt-6-luna",
        transport=httpx.MockTransport(handler),
    )
    service = ChatService(provider=p, repository=SQLiteConversationRepository(":memory:"))
    service.web_search.client = client_for(
        lambda _: httpx.Response(200, json=search_payload()), model="gpt-6-luna"
    )
    service.web_search.set_enabled(True)
    try:
        assert "".join(service.stream_speech("人工資料を調べて")) == "確認できたよ。[S1]"
        assert len(requests) == 2 and len(service.last_response.sources) == 1
        assert len(service.get_current_messages()) == 2
    finally:
        service.close()


@pytest.mark.parametrize("ending", ["bad_finish", "cancel"])
def test_ui_partial_reply_is_removed_and_voice_stopped_on_failure_or_cancel(
    tmp_path, monkeypatch, ending
):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        lambda *_: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat),
    )
    arrived, release = threading.Event(), threading.Event()

    def gate(index):
        if index == 1:
            arrived.set()
            assert release.wait(5)

    data = frames(
        ['{"speech":"先の文。']
        if ending == "bad_finish"
        else ['{"speech":"先の文。', '後の文。"}'],
        finish="length" if ending == "bad_finish" else "tool_calls",
    )
    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(provider(lambda _: response(data, gate)))
    spoken, stops = [], []
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    backend._service.voice_output.stop = lambda: stops.append(True)
    window = ChatWindow(backend)
    try:
        window.show()
        app.processEvents()
        window.input_edit.setPlainText("人工入力")
        window.send_button.click()
        wait_until(lambda: arrived.is_set() and "先の文。" in window.chat_view.toPlainText())
        assert spoken == ["先の文。"] and len(backend._service.get_current_messages()) == 1
        before = len(stops)
        if ending == "cancel":
            window._cancel_web_turn()
        release.set()
        wait_until(lambda: window._thread is None)
        assert "先の文。" not in window.chat_view.toPlainText()
        assert "後の文。" not in window.chat_view.toPlainText()
        assert len(stops) > before and spoken == ["先の文。"]
        assert len(backend._service.get_current_messages()) == 1
        assert backend._service.last_response is None
    finally:
        release.set()
        wait_until(lambda: window._thread is None)
        window.close()
        backend.close()


def test_fragmented_function_name_holds_arguments_until_reply_is_known():
    data = [
        event(tool_delta('{"speech":', name="reply_")),
        event(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "to_user", "arguments": '"一文。"}'}}
                ]
            }
        ),
        event({}, "tool_calls"),
        b"data: [DONE]\n\n",
    ]
    p = provider(lambda _: response(data))
    values = list(p._stream_research_reply([]))
    assert "".join(value for value in values if isinstance(value, str)) == '{"speech":"一文。"}'
    assert values[-1].speech == "一文。"


def test_misnamed_search_arguments_are_not_emitted_before_late_name_change():
    data = [
        event(tool_delta('{"query":"内部の人工検索語","detailed":false}', name="reply_to_user")),
        event(tool_delta("", name="request_web_search")),
        event({}, "tool_calls"),
        b"data: [DONE]\n\n",
    ]
    p = provider(lambda _: response(data))
    values = []
    with pytest.raises(ProviderResponseError):
        values.extend(p._stream_research_reply([]))
    assert values == [] and p.last_response is None


def test_json_http_response_keeps_validated_reply_and_usage_without_retry():
    calls = []

    def handler(_):
        calls.append(True)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "reply_to_user",
                                        "arguments": '{"speech":"一文。"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 8},
            },
        )

    p = provider(handler)
    values = list(p._stream_research_reply([]))
    assert len(values) == 1 and values[0].speech == "一文。"
    assert p.get_usage().total_tokens == 8 and calls == [True]


def test_partial_network_failure_does_not_save_or_expose_transport_detail():
    def fail(index):
        if index == 1:
            raise httpx.ReadError("artificial-private-detail")

    service = ChatService(
        provider=provider(lambda _: response(frames(['{"speech":"先の文。', '後の文。"}']), fail)),
        repository=SQLiteConversationRepository(":memory:"),
    )
    try:
        stream = service.stream_speech("人工入力")
        assert next(stream) == "先の文。"
        with pytest.raises(RuntimeError) as error:
            list(stream)
        assert "artificial-private-detail" not in str(error.value)
        assert service.last_response is None and len(service.get_current_messages()) == 1
    finally:
        service.close()


@pytest.mark.parametrize(
    "parts,expected",
    [
        (['{"speech":"  一文。', ' 二文。  "}'], "  一文。 二文。  "),
        (['{"speech":"笑顔\\ud83d', '\\ude0a。"}'], "笑顔😊。"),
    ],
)
def test_stream_preserves_original_text_and_split_unicode(parts, expected):
    service = ChatService(
        provider=provider(lambda _: response(frames(parts))),
        repository=SQLiteConversationRepository(":memory:"),
    )
    try:
        assert "".join(service.stream_speech("人工入力")) == expected
        assert (
            service.last_response.speech == service.get_current_messages()[-1].content == expected
        )
    finally:
        service.close()
