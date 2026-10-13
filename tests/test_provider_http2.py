"""実HTTP/2フレームで、逐次応答の終了と共有接続を確認する。"""

import json
import ssl

import h2.config
import h2.connection
import h2.events
import httpcore
import pytest

from multi_llm_companion.domain import ChatMessage
from multi_llm_companion.providers import OpenAICompatibleProvider, ProviderConnectionError


class H2Stream(httpcore.MockStream):
    def __init__(self, *, http2, end_stream=True, fail=False):
        super().__init__([], http2=http2)
        self.peer = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        self.peer.initiate_connection()
        self._buffer.append(self.peer.data_to_send())
        self.requests = {}
        self.end_stream = end_stream
        self.fail = fail

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert ssl_context.check_hostname
        assert server_hostname == "example.invalid"
        return self

    def read(self, max_bytes, timeout=None):
        if self.fail and self.requests:
            raise httpcore.ReadError("synthetic transport failure")
        assert self._buffer, "DONE後にHTTPの終端を待ってはならない"
        return super().read(max_bytes, timeout)

    def write(self, buffer, timeout=None):
        for event in self.peer.receive_data(buffer):
            if isinstance(event, h2.events.RequestReceived):
                self.requests[event.stream_id] = bytearray()
            elif isinstance(event, h2.events.DataReceived):
                self.requests[event.stream_id].extend(event.data)
                self.peer.acknowledge_received_data(event.flow_controlled_length, event.stream_id)
            elif isinstance(event, h2.events.StreamEnded):
                payload = json.loads(self.requests[event.stream_id])
                text = payload["messages"][-1]["content"]
                raw = json.dumps({"speech": text})
                if "tools" in payload:
                    delta = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "reply",
                                "type": "function",
                                "function": {"name": "reply_to_user", "arguments": raw},
                            }
                        ]
                    }
                else:
                    delta = {"content": raw}
                self.peer.send_headers(
                    event.stream_id, [(":status", "200"), ("content-type", "text/event-stream")]
                )
                reason = "tool_calls" if "tools" in payload else "stop"
                content = (
                    "data: "
                    + json.dumps({"choices": [{"delta": delta, "finish_reason": reason}]})
                    + "\n\n"
                ).encode()
                self.peer.send_data(event.stream_id, content)
                self._buffer.append(self.peer.data_to_send())
                self.peer.send_data(
                    event.stream_id,
                    b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\ndata: [DONE]\n\n',
                    end_stream=self.end_stream,
                )
        if outgoing := self.peer.data_to_send():
            self._buffer.append(outgoing)


@pytest.fixture
def h2_backend(monkeypatch):
    class Backend(httpcore.NetworkBackend):
        def __init__(self):
            self.streams = []
            self.http2 = False
            self.end_stream = True
            self.fail = False

        def connect_tcp(self, *args, **kwargs):
            stream = H2Stream(http2=self.http2, end_stream=self.end_stream, fail=self.fail)
            self.streams.append(stream)
            return stream

    backend = Backend()

    class Pool(httpcore.ConnectionPool):
        def __init__(self, *args, **kwargs):
            backend.http2 = kwargs.get("http2", False)
            super().__init__(*args, network_backend=backend, **kwargs)

    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(httpcore, "ConnectionPool", Pool)
    return backend


def provider():
    return OpenAICompatibleProvider(
        name="test",
        base_url="https://example.invalid/v1",
        api_key="synthetic-test-only",
        model="test-model",
    )


@pytest.mark.parametrize("research", [False, True])
@pytest.mark.parametrize("end_stream", [False, True])
def test_http2_reuses_connection_without_waiting_after_done(h2_backend, research, end_stream):
    h2_backend.end_stream = end_stream
    client = provider()
    try:
        for text in ("first", "second", "third"):
            messages = [ChatMessage(role="user", content=text)]
            if research:
                assert list(client._stream_research_reply(messages))[-1].speech == text
            else:
                assert json.loads("".join(client.stream_chat(messages)))["speech"] == text
            assert client.last_response.speech == text
            assert client.get_usage().total_tokens == 5
        assert len(h2_backend.streams) == 1
        assert not h2_backend.streams[0]._closed
    finally:
        client._close()
    assert h2_backend.streams[0]._closed


@pytest.mark.parametrize("cancel_old", [False, True])
def test_http2_keeps_new_fork_response_separate(h2_backend, cancel_old):
    client = provider()
    fork = client._fork_request()
    old = fork.stream_chat([ChatMessage(role="user", content="old")])
    try:
        assert json.loads(next(old))["speech"] == "old"
        if cancel_old:
            old.close()
        new = client.stream_chat([ChatMessage(role="user", content="new")])
        assert json.loads("".join(new))["speech"] == "new"
        assert client.last_response.speech == "new"
        if cancel_old:
            assert fork.last_response is None
        else:
            list(old)
            assert fork.last_response.speech == "old"
        assert len(h2_backend.streams) == 1
    finally:
        old.close()
        client._close()
    assert h2_backend.streams[0]._closed


def test_http2_reconnects_after_transport_failure(h2_backend):
    client = provider()
    h2_backend.fail = True
    messages = [ChatMessage(role="user", content="retry")]
    try:
        with pytest.raises(ProviderConnectionError):
            list(client.stream_chat(messages))
        assert client.last_response is None
        h2_backend.fail = False
        assert json.loads("".join(client.stream_chat(messages)))["speech"] == "retry"
        assert len(h2_backend.streams) == 2
    finally:
        client._close()
    assert all(stream._closed for stream in h2_backend.streams)
