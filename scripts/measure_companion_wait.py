"""現行の会話経路をHTTP Mock・隔離無音で測る専用補助。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import wave
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
OLD = "旧能動発話です。"
TEXTS = {
    "normal": ("今日は静かな時間だね。", "うん、静かな時間だね。"),
    "search": ("人工資料の試験値を調べて。", "人工資料では試験値は七です。[S1]"),
    "interrupt": ("今はこちらの話を聞いて。", "新しい入力を受け取ったよ。"),
}
REQUIRED = (
    "ui_text_send",
    "llm_request_start",
    "provider_full_response",
    "ui_answer_commit",
    "ui_answer_paint",
    "sentence_ready",
    "stream_synthesis_requested",
    "stream_synthesis_completed",
    "mock_playback_start",
)
INTERVALS = {
    "input_to_request": ("ui_text_send", "llm_request_start"),
    "request_to_full": ("llm_request_start", "provider_full_response"),
    "full_to_display": ("provider_full_response", "ui_answer_commit"),
    "input_to_display": ("ui_text_send", "ui_answer_commit"),
    "input_to_paint": ("ui_text_send", "ui_answer_paint"),
    "input_to_sentence": ("ui_text_send", "sentence_ready"),
    "sentence_to_synthesis": ("sentence_ready", "stream_synthesis_requested"),
    "synthesis_to_data": ("stream_synthesis_requested", "stream_synthesis_completed"),
    "data_to_mock_playback": ("stream_synthesis_completed", "mock_playback_start"),
    "input_to_mock_playback": ("ui_text_send", "mock_playback_start"),
    "input_to_cancel": ("ui_text_send", "proactive_cancelled"),
    "cancel_to_old_http_end": ("proactive_cancelled", "old_http_end"),
    "old_http_end_to_new_request": ("old_http_end", "llm_request_start"),
    "old_http_end_to_worker_finish": ("old_http_end", "old_worker_finished"),
    "worker_finish_to_new_request": ("old_worker_finished", "llm_request_start"),
    "full_to_save_start": ("provider_full_response", "assistant_save_started"),
    "assistant_save": ("assistant_save_started", "assistant_save_completed"),
    "save_to_display": ("assistant_save_completed", "ui_answer_commit"),
}


def save(path, value):
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return None
    n = len(ordered)
    middle = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
    return {
        "n": n,
        "median_ms": middle,
        "p95_ms": ordered[math.ceil(n * 0.95) - 1],
        "max_ms": ordered[-1],
    }


def research_sse_blocks(answer):
    """同じ生成時刻の引数を、4個のSSE応答へ包む。"""
    choice = answer["choices"][0]
    call = choice["message"]["tool_calls"][0]
    arguments = call["function"]["arguments"]
    if call["function"]["name"] == "reply_to_user":
        end = arguments.rfind('"')
        middle = arguments.index(":") + 2 + (end - arguments.index(":")) // 2
        parts = [arguments[:middle], arguments[middle : end + 1], arguments[end + 1 :]]
    else:
        parts = [
            arguments[len(arguments) * i // 3 : len(arguments) * (i + 1) // 3] for i in range(3)
        ]
    blocks = []
    for index, part in enumerate(parts):
        delta = {"index": 0, "function": {"arguments": part}}
        if index == 0:
            delta.update(id=call["id"], type="function")
            delta["function"]["name"] = call["function"]["name"]
        data = {"choices": [{"index": 0, "delta": {"tool_calls": [delta]}, "finish_reason": None}]}
        blocks.append(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode())
    end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    usage = {"choices": [], "usage": answer["usage"]}
    blocks.append(
        (
            "data: " + json.dumps(end) + "\n\ndata: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n"
        ).encode()
    )
    return blocks


def artificial_wait_ms(events, input_ns, until_ns=None):
    """同時要求の人工待ちを重複せず、観測時刻までの区間和にする。"""
    intervals = []
    for event in events:
        if event["event"] != "artificial_http_wait":
            continue
        end = event.get("wait_finished_ns", event["monotonic_ns"])
        start = event.get("wait_started_ns", end - event["actual_ns"])
        begin, finish = max(start, input_ns), min(end, until_ns if until_ns is not None else end)
        if begin < finish:
            intervals.append((begin, finish))
    elapsed, previous_end = 0, input_ns
    for begin, finish in sorted(intervals):
        elapsed += max(0, finish - max(begin, previous_end))
        previous_end = max(previous_end, finish)
    return elapsed / 1e6


def summarize(cases):
    summary = {}
    for condition in TEXTS:
        rows = [row for row in cases if row["condition"] == condition and not row["warmup"]]
        valid = [row for row in rows if row["success"]]
        summary[condition] = {
            "success": len(valid),
            "failed": len(rows) - len(valid),
            "missing_cases": sum(bool(row["missing"]) for row in rows),
            "intervals": {
                key: stats(
                    [
                        row["intervals_ms"][key]
                        for row in valid
                        if row["intervals_ms"][key] is not None
                    ]
                )
                for key in INTERVALS
            },
            "http_artificial_wait_ms": stats([row["artificial_wait_ms"] for row in valid]),
            "http_artificial_wait_total_ms": stats(
                [row.get("artificial_wait_total_ms", row["artificial_wait_ms"]) for row in valid]
            ),
            "input_to_display_excluding_http_wait_ms": stats(
                [
                    row["intervals_ms"]["input_to_display"] - row["artificial_wait_ms"]
                    for row in valid
                ]
            ),
            "provider_fragment_unobserved": sum(
                row["events_ns"].get("provider_first_fragment") is None for row in rows
            ),
            "events_missing": {
                key: sum(row["events_ns"][key] is None for row in rows) for key in REQUIRED
            },
        }
    return summary


def plan(samples):
    return {
        "created_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "samples_per_condition": samples,
        "clock": "time.perf_counter_ns; monotonic high resolution; original trace clock kept separately",
        "warmup_per_condition": 1,
        "case_limit_seconds": 30,
        "series_limit_seconds": 2700,
        "stop_conditions": [
            "first missing required event",
            "first invalid case",
            "unexpected network",
            "STOP file",
            "independent deadline",
        ],
        "provider": {"class": "OpenAICompatibleProvider", "base_url": URL, "model": MODEL},
        "http_body": {
            "mode": "four JSON blocks before fix; four SSE blocks for stream=true research",
            "reply_sse": "partial speech at 300ms; sentence at 400ms; args close at 500ms; finish at 600ms",
            "chat_initial_ms": 300,
            "chat_block_interval_ms": 100,
            "search_initial_ms": 500,
            "search_block_interval_ms": 100,
            "old_initial_ms": 600,
            "old_block_interval_ms": 200,
        },
        "interrupt_after_old_request_ms": 200,
        "proactive_preparation": "isolated last user activity 31 minutes earlier; scheduler invoked once",
        "artificial_wait_accounting": "union of overlapping HTTP waits, clipped at input/display",
        "voice": {
            "class": "VoiSonaTalkVoiceOutput",
            "post_ms": 50,
            "poll_ms": 100,
            "mock_play_ms": 20,
            "wav_ms": 40,
            "monitor_interval_seconds": 0.05,
            "base_url": "http://127.0.0.1:32766/api/talk/v1",
            "player": "silent in-memory player; no sound device",
        },
        "inputs_and_replies": TEXTS,
        "cancelled_reply": OLD,
        "required_events": REQUIRED,
        "interrupt_required": [
            "proactive_cancelled",
            "old_http_end",
            "old_worker_finished",
            "old_result_cancelled",
        ],
        "null_events": ["real_voice_start", "speech_recognition"],
        "conditional_events": {
            "provider_first_fragment": "observed for reply SSE; null for held search/JSON"
        },
        "ui": "offscreen Qt 2D 1000x800; content commit and paint dispatch recorded separately",
        "p95_method": "nearest rank ceil(n*0.95); warmups excluded",
        "scope": "phases 7/9/10 current measurement; phase12 not restarted; no product fixes",
    }


def protected():
    paths = [
        ROOT / ".env",
        *ROOT.glob("config/*.yaml"),
        *ROOT.glob("local/*.yaml"),
        *ROOT.glob("data/*"),
        *ROOT.glob("src/**/*.py"),
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def worker(directory, samples, *, live=False, source_root=None, budget_directory=None):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MLC_LIP_SYNC_DIAGNOSTIC"] = "1"
    sys.path.insert(0, str(source_root or ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import httpx
    from PySide6.QtCore import QEvent, QObject, QSettings
    from PySide6.QtWidgets import QApplication

    import multi_llm_companion.application as application
    import multi_llm_companion.integrations.voisona_talk as voice_module
    import multi_llm_companion.ui.chat_window as ui
    from multi_llm_companion.config import (
        AppConfig,
        AudioConfig,
        AvatarConfig,
        PersonaConfig,
        ProviderConfig,
        ProvidersConfig,
        ResolvedProviderConfig,
        RuntimeConfig,
    )
    from multi_llm_companion.domain import AssistantResponse
    from multi_llm_companion.integrations.lip_sync_diagnostic import write_lip_sync_trace
    from multi_llm_companion.providers import OpenAICompatibleProvider
    from multi_llm_companion.services.proactive import ProactiveSettings
    from multi_llm_companion.services.web_search import OpenAIWebSearch, WebSearchSession

    if live:
        from scripts.evaluate_memory_continuity import ApiBudget, api_key

        key = api_key()
        budget = ApiBudget(budget_directory or directory, limit=380)
        real_transport = httpx.HTTPTransport(retries=0)
        TEXTS["search"] = (
            "Python公式のリストappendの説明を検索で確認して、出典付きで一文にまとめて。",
            "",
        )

    app = QApplication.instance() or QApplication([])
    lock = threading.RLock()
    context = {
        "case_id": "setup",
        "condition": "normal",
        "events": [],
        "window": None,
        "backend": None,
        "request_index": 0,
        "trace_write_ns": 0,
        "old_speech": None,
        "old_thread_id": None,
    }
    old_started = threading.Event()
    voice_texts = []
    blocked = []

    def record(event, source="probe", **fields):
        with lock:
            window, backend = context["window"], context["backend"]
            row = {
                "case_id": context["case_id"],
                "condition": context["condition"],
                "monotonic_ns": time.perf_counter_ns(),
                "event": event,
                "source": source,
                "ui_generation": window._stream_generation if window else None,
                "proactive_generation": window._proactive_turn_generation if window else None,
                "service_generation": backend._service._stream_generation if backend else None,
                "service_proactive_generation": backend._service._proactive_generation
                if backend
                else None,
                **fields,
            }
            context["events"].append(row)
            started = time.perf_counter_ns()
            write_lip_sync_trace(
                source,
                event,
                path=directory / "events.jsonl",
                **{k: v for k, v in row.items() if k not in {"source", "event"}},
            )
            context["trace_write_ns"] += time.perf_counter_ns() - started

    def trace(source, event, **fields):
        if "monotonic_ns" in fields:
            fields["original_monotonic_ns"] = fields.pop("monotonic_ns")
        record(event, source, **fields)

    def deny(*args, **kwargs):
        blocked.append("unexpected_network")
        record("unexpected_network_blocked")
        raise RuntimeError("計測外の通信を遮断しました。")

    class LiveBody(httpx.SyncByteStream):
        def __init__(self, response, kind, request_id):
            self.response, self.kind, self.request_id = response, kind, request_id

        def __iter__(self):
            try:
                for block in self.response.stream:
                    record("http_body_block", kind=self.kind, request_id=self.request_id)
                    yield block
            finally:
                record(
                    "old_http_end" if self.kind == "old" else "http_response_end",
                    kind=self.kind,
                    request_id=self.request_id,
                )

        def close(self):
            self.response.close()

    def live_http(request, kind, request_id):
        budget.check(request)
        try:
            response = real_transport.handle_request(request)
            if response.status_code >= 400:
                budget.failed = True
                save(budget.directory / "api-budget.json", budget.summary())
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=LiveBody(response, kind, request_id),
            )
        except Exception:
            budget.failed = True
            save(budget.directory / "api-budget.json", budget.summary())
            raise

    class Body(httpx.SyncByteStream):
        def __init__(self, payload, kind, *, streaming=False, request_id=None):
            self.data = json.dumps(payload, ensure_ascii=False).encode()
            self.kind = kind
            self.request_id = request_id
            window = context["window"]
            self.request_generation = window._stream_generation if window else None
            self.request_proactive_generation = (
                window._proactive_turn_generation if window else None
            )
            self.blocks = (
                research_sse_blocks(payload)
                if streaming
                else [
                    self.data[len(self.data) * i // 4 : len(self.data) * (i + 1) // 4]
                    for i in range(4)
                ]
            )

        def __iter__(self):
            initial, interval = (
                (0.6, 0.2)
                if self.kind == "old"
                else ((0.5, 0.1) if self.kind == "search" else (0.3, 0.1))
            )
            for index in range(4):
                start = time.perf_counter_ns()
                time.sleep(initial if index == 0 else interval)
                finished = time.perf_counter_ns()
                record(
                    "artificial_http_wait",
                    kind=self.kind,
                    actual_ns=finished - start,
                    wait_started_ns=start,
                    wait_finished_ns=finished,
                    planned_ms=round((initial if index == 0 else interval) * 1000),
                    request_id=self.request_id,
                    request_generation=self.request_generation,
                    request_proactive_generation=self.request_proactive_generation,
                )
                record(
                    "http_body_block",
                    kind=self.kind,
                    block=index,
                    request_id=self.request_id,
                    request_generation=self.request_generation,
                    request_proactive_generation=self.request_proactive_generation,
                )
                if index == 3:
                    record(
                        "old_http_end" if self.kind == "old" else "http_response_end",
                        kind=self.kind,
                        request_id=self.request_id,
                    )
                yield self.blocks[index]

    def tool(name, args):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "artificial-call",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    def chat_http(request):
        payload = json.loads(request.content)
        assert str(request.url) == URL + "/chat/completions"
        assert payload["model"] == MODEL
        assert "tools" in payload or not payload.get("stream", False)
        condition = context["condition"]
        is_old = condition == "interrupt" and context["request_index"] == 0
        context["request_index"] += 1
        request_id = f"{context['case_id']}-http-{context['request_index']}"
        kind = "old" if is_old else "decision" if "tools" in payload else "final"
        record(
            "old_request_start" if is_old else "llm_request_start",
            kind=kind,
            url=str(request.url),
            model=payload["model"],
            stream=payload.get("stream", False),
            tools_present="tools" in payload,
            tool_choice=payload.get("tool_choice"),
            request_id=request_id,
        )
        if is_old:
            context["old_thread_id"] = threading.get_ident()
            old_started.set()
        if live:
            return live_http(request, kind, request_id)
        if is_old:
            answer = {
                "choices": [
                    {"message": {"content": json.dumps({"speech": OLD})}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 101, "completion_tokens": 103, "total_tokens": 204},
            }
        elif "tools" in payload:
            assert payload["tool_choice"] == "required"
            assert {x["function"]["name"] for x in payload["tools"]} == {
                "reply_to_user",
                "request_web_search",
            }
            answer = (
                tool("request_web_search", {"query": "人工資料 試験値", "detailed": False})
                if condition == "search"
                else tool("reply_to_user", {"speech": TEXTS[condition][1]})
            )
        else:
            answer = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"speech": TEXTS[condition][1]}, ensure_ascii=False
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        streaming = payload.get("stream", False)
        return httpx.Response(
            200,
            stream=Body(answer, kind, streaming=streaming, request_id=request_id),
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
        )

    def search_http(request):
        assert str(request.url) == URL + "/responses"
        payload = json.loads(request.content)
        assert payload["model"] == MODEL and context["condition"] == "search"
        record("search_request_start", url=str(request.url), model=MODEL)
        if live:
            return live_http(request, "search", f"{context['case_id']}-search")
        answer = {
            "status": "completed",
            "output": [
                {"type": "web_search_call", "status": "completed", "action": {"type": "search"}},
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "人工資料の試験値は七です。[1]",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "start_index": 13,
                                    "end_index": 16,
                                    "url": "https://fixture.example/value",
                                    "title": "人工資料",
                                }
                            ],
                        }
                    ],
                },
            ],
        }
        return httpx.Response(200, stream=Body(answer, "search"))

    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 640)
    pending_wavs = {}
    playback_context = threading.local()

    def voice_http(request):
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "artificial-voice"}])
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "POST":
            payload = json.loads(request.content)
            assert OLD not in payload["text"]
            voice_texts.append(payload["text"])
            path = Path(payload["output_file_path"]).resolve()
            assert path.is_relative_to(directory.resolve())
            request_id = f"artificial-{len(pending_wavs)}"
            pending_wavs[request_id] = path
            start = time.perf_counter_ns()
            time.sleep(0.05)
            record("artificial_voice_post_wait", actual_ns=time.perf_counter_ns() - start)
            return httpx.Response(200, json={"uuid": request_id})
        request_id = request.url.path.rsplit("/", 1)[-1]
        start = time.perf_counter_ns()
        time.sleep(0.1)
        record("artificial_voice_poll_wait", actual_ns=time.perf_counter_ns() - start)
        pending_wavs[request_id].write_bytes(wav.getvalue())
        return httpx.Response(200, json={"state": "succeeded"})

    class Player:
        is_playing = False
        position_seconds = 0.0
        duration_seconds = 0.04
        lead_in_seconds = 0.0
        amplitude = 0.0

        def load(self, data):
            assert data == wav.getvalue()
            record("mock_player_loaded")

        def play(self):
            start = time.perf_counter_ns()
            time.sleep(0.02)
            record("artificial_mock_player_wait", actual_ns=time.perf_counter_ns() - start)
            record(
                "mock_playback_start",
                playback_text=getattr(playback_context, "text", None),
                voice_generation=getattr(playback_context, "generation", None),
            )

        def stop(self):
            self.is_playing = False

    class PaintProbe(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            if event.type() == QEvent.Type.Paint:
                record("ui_answer_paint")
                watched.removeEventFilter(self)
            return False

    paint_probe = PaintProbe()

    class ObservedVoice(voice_module.VoiSonaTalkVoiceOutput):
        def _play_stream_segment(self, generation, prepared, **kwargs):
            playback_context.text = prepared.text
            playback_context.generation = generation
            return super()._play_stream_segment(generation, prepared, **kwargs)

    class Window(ui.ChatWindow):
        def _render_messages(self):
            if any(OLD in message.content for message in self._messages):
                record("old_answer_render_attempt")
            super()._render_messages()
            if not hasattr(self, "_measured_commit"):
                self._measured_commit = False
            if (
                self._messages
                and self._messages[-1].content
                and (
                    self._messages[-1].role == "assistant"
                    if live
                    else TEXTS[context["condition"]][1].startswith(self._messages[-1].content)
                )
                and not self._measured_commit
            ):
                self._measured_commit = True
                record("ui_answer_commit")
                self.chat_view._rows[-1][1].text.viewport().installEventFilter(paint_probe)

        def _queue_stream_sentence(self, sentence):
            if sentence.strip():
                record("sentence_ready")
            return super()._queue_stream_sentence(sentence)

        def _cancel_pending_proactive_turn(self):
            active = self._proactive_thread is not None or self._proactive_turn_active
            result = super()._cancel_pending_proactive_turn()
            if active:
                record("proactive_cancelled")
            return result

        def _proactive_turn_finished(self):
            record("old_worker_finished")
            return super()._proactive_turn_finished()

        def _proactive_turn_succeeded(self, generation, value):
            prior = list(self._messages)
            stale = generation != self._proactive_turn_generation
            result = super()._proactive_turn_succeeded(generation, value)
            if stale and self._messages != prior:
                record("old_answer_render_attempt")
            elif stale:
                record("stale_proactive_result_suppressed")
            return result

        def _proactive_turn_cancelled(self, generation):
            record("old_result_cancelled", cancelled_worker_generation=generation)
            return super()._proactive_turn_cancelled(generation)

    def wait_for(predicate):
        until = context["case_started"] + (85 if live else 28)
        while not predicate():
            app.processEvents()
            if time.monotonic() > until or (directory / "STOP").exists():
                raise TimeoutError("ケース期限または停止要求")
            time.sleep(0.001)
        app.processEvents()

    def observed(name):
        return any(row["event"] == name for row in context["events"])

    class ObservedProvider(OpenAICompatibleProvider):
        """HTTPと要求の動作を保ち、分離した能動要求も同じ方法で観測する。"""

        def _chat_response(self, *args, **kwargs):
            answer = super()._chat_response(*args, **kwargs)
            if isinstance(answer, AssistantResponse):
                old = (
                    threading.get_ident() == context["old_thread_id"]
                    if live
                    else answer.speech == OLD
                )
                if old:
                    context["old_speech"] = answer.speech
                record("old_provider_full_response" if old else "provider_full_response")
            else:
                record("provider_search_decision")
            return answer

        def _stream_research_reply(self, *args, **kwargs):
            for value in super()._stream_research_reply(*args, **kwargs):
                record(
                    "provider_first_fragment"
                    if isinstance(value, str)
                    else "provider_full_response"
                    if isinstance(value, AssistantResponse)
                    else "provider_search_decision"
                )
                yield value

    before = protected()
    settings = QSettings("multi-llm-companion", "desktop-companion")

    def settings_hash():
        return hashlib.sha256(
            repr(sorted((key, str(settings.value(key))) for key in settings.allKeys())).encode()
        ).hexdigest()

    normal_settings_before = settings_hash()
    save(directory / "protected-before.json", before)
    cases = []
    try:
        with ExitStack() as stack:
            if not live:
                stack.enter_context(patch.object(httpx.HTTPTransport, "handle_request", deny))
                stack.enter_context(patch.object(socket.socket, "connect", deny))
            stack.enter_context(patch.object(ui, "write_lip_sync_trace", trace))
            stack.enter_context(patch.object(voice_module, "write_lip_sync_trace", trace))
            for condition in TEXTS:
                for index in range(samples + 1):
                    case_id = f"{condition}-{'warmup' if index == 0 else f'{index:02}'}"
                    run = directory / case_id
                    run.mkdir()
                    context.update(
                        case_id=case_id,
                        condition=condition,
                        events=[],
                        window=None,
                        backend=None,
                        request_index=0,
                        trace_write_ns=0,
                        old_speech=None,
                        old_thread_id=None,
                        case_started=time.monotonic(),
                    )
                    old_started.clear()
                    voice_texts.clear()
                    save(
                        directory / "case-state.json",
                        {
                            "case_id": case_id,
                            "started_monotonic": context["case_started"],
                            "active": True,
                        },
                    )
                    provider = ObservedProvider(
                        name="openai",
                        base_url=URL,
                        api_key=key if live else "artificial-only",
                        model=MODEL,
                        max_output_tokens=1536 if live else 1024,
                        timeout=25 if live else 60,
                        transport=httpx.MockTransport(chat_http),
                    )
                    voice = ObservedVoice(
                        base_url="http://127.0.0.1:32766/api/talk/v1",
                        username="artificial",
                        password="artificial",
                        transport=httpx.MockTransport(voice_http),
                        wav_player=Player(),
                        temp_root=run,
                        monitor_interval=0.05,
                    )
                    runtime = RuntimeConfig(
                        project_root=run,
                        app_config_path=run / "app.yaml",
                        app=AppConfig(active_provider="openai"),
                        providers=ProvidersConfig(
                            providers={
                                "openai": ProviderConfig(
                                    provider="openai_compatible",
                                    base_url_env="OPENAI_BASE_URL",
                                    api_key_env="OPENAI_API_KEY",
                                    model_env="OPENAI_MODEL",
                                )
                            }
                        ),
                        persona=PersonaConfig(name="人工案内役", system_prompt="短い人工会話。"),
                        active_provider=ResolvedProviderConfig(
                            "openai",
                            "openai_compatible",
                            URL,
                            "artificial-only",
                            MODEL,
                            0.7,
                            1024,
                            30,
                        ),
                        database_path=run / "conversation.db",
                        audio=AudioConfig(),
                        avatar=AvatarConfig(),
                    )
                    with (
                        patch.object(application, "build_provider", lambda _, p=provider: p),
                        patch.object(application, "build_voice_output", lambda _, v=voice: v),
                        patch.object(
                            ui,
                            "QSettings",
                            lambda *_, r=run: QSettings(
                                str(r / "window.ini"), QSettings.Format.IniFormat
                            ),
                        ),
                    ):
                        backend = application.ApplicationBackend(runtime)
                        context["backend"] = backend
                        add_message = backend._repository.add_message

                        def observe_save(
                            conversation_id, role, content, *args, _original=add_message, **kwargs
                        ):
                            if role == "assistant":
                                record(
                                    "assistant_save_started",
                                    old_answer=threading.get_ident() == context["old_thread_id"]
                                    if live
                                    else content == OLD,
                                )
                            result = _original(conversation_id, role, content, *args, **kwargs)
                            if role == "assistant":
                                record(
                                    "assistant_save_completed",
                                    old_answer=threading.get_ident() == context["old_thread_id"]
                                    if live
                                    else content == OLD,
                                )
                            return result

                        backend._repository.add_message = observe_save
                        backend._service.web_search = WebSearchSession(
                            OpenAIWebSearch(
                                api_key=key if live else "artificial-only",
                                model=MODEL,
                                transport=httpx.MockTransport(search_http),
                            )
                        )
                        backend._service.web_search.set_enabled(True)
                        research = backend._service._research_turn_reply

                        def research_call(
                            *args, _original=research, p=provider, b=backend, **kwargs
                        ):
                            record(
                                "research_flow_entered",
                                supports_research=p._supports_research,
                                uses_research_flow=b._uses_research_flow,
                            )
                            return _original(*args, **kwargs)

                        backend._service._research_turn_reply = research_call
                        if hasattr(backend._service, "_stream_research_turn_reply"):
                            streaming_research = backend._service._stream_research_turn_reply

                            def observe_research(
                                *args, _original=streaming_research, p=provider, b=backend, **kwargs
                            ):
                                record(
                                    "research_flow_entered",
                                    supports_research=p._supports_research,
                                    uses_research_flow=b._uses_research_flow,
                                )
                                yield from _original(*args, **kwargs)

                            backend._service._stream_research_turn_reply = observe_research
                        window = Window(backend)
                    context["window"] = window
                    window._proactive_timer.stop()
                    window.resize(1000, 800)
                    window.show()
                    app.processEvents()
                    error = None
                    try:
                        if condition == "interrupt":
                            conversation_id = backend.create_conversation("人工割り込み")
                            window._current_conversation_id = conversation_id
                            backend._service.load_conversation(conversation_id)
                            window._proactive_settings = lambda: ProactiveSettings(
                                enabled=True, read_aloud=True, quiet_start_hour=0, quiet_end_hour=0
                            )
                            window._proactive.note_user_activity(
                                datetime.now() - timedelta(minutes=31)
                            )
                            window._on_proactive_timer()
                            assert window._proactive_thread is not None
                            wait_for(old_started.is_set)
                            old_start = next(
                                row["monotonic_ns"]
                                for row in context["events"]
                                if row["event"] == "old_request_start"
                            )
                            wait_for(
                                lambda start=old_start: (
                                    time.perf_counter_ns() - start >= 200_000_000
                                )
                            )
                        record("ui_text_send")
                        window.input_edit.setPlainText(TEXTS[condition][0])
                        window._send_message()
                        wait_for(
                            lambda w=window, v=voice: (
                                observed("mock_playback_start")
                                and observed("ui_answer_paint")
                                and w._thread is None
                                and w._proactive_thread is None
                                and w._voice_thread is None
                                and w._pending_voice_text is None
                                and not v.is_speaking
                            )
                        )
                        assert backend._uses_research_flow and provider._supports_research
                        assert observed("research_flow_entered")
                        stored = backend._service.get_current_messages()
                        replies = [m.content for m in stored if m.role.value == "assistant"]
                        if live:
                            assert len(replies) == 1 and replies[0].strip()
                            assert not budget.failed
                        else:
                            assert replies == [TEXTS[condition][1]]
                        assert OLD not in window.chat_view.toPlainText() and OLD not in voice_texts
                        assert not observed("old_answer_render_attempt")
                        assert not any(
                            row["event"] == "assistant_save_started" and row["old_answer"]
                            for row in context["events"]
                        )
                        if not live:
                            assert backend._service.last_usage.total_tokens == 8
                        if condition == "search":
                            assert (
                                observed("search_request_start")
                                and backend._service.last_response.sources
                            )
                        else:
                            assert not observed("search_request_start")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        record("case_failure", error=error)
                    events = list(context["events"])
                    old_saved = any(
                        m.content == OLD for m in backend._service.get_current_messages()
                    )
                    if live:
                        old_saved = any(
                            row["event"] == "assistant_save_started" and row["old_answer"]
                            for row in events
                        )
                    event_times = {
                        key: next(
                            (row["monotonic_ns"] for row in events if row["event"] == key), None
                        )
                        for key in set(REQUIRED)
                        | {x for pair in INTERVALS.values() for x in pair}
                        | {"old_worker_finished", "old_result_cancelled"}
                    }
                    event_times.update(
                        provider_first_fragment=next(
                            (
                                row["monotonic_ns"]
                                for row in events
                                if row["event"] == "provider_first_fragment"
                            ),
                            None,
                        ),
                        real_voice_start=None,
                        speech_recognition=None,
                    )
                    required = list(REQUIRED) + (
                        [
                            "proactive_cancelled",
                            "old_http_end",
                            "old_worker_finished",
                            "old_result_cancelled",
                        ]
                        if condition == "interrupt"
                        else []
                    )
                    missing = [key for key in required if event_times[key] is None]
                    intervals = {
                        key: (event_times[b] - event_times[a]) / 1e6
                        if event_times[a] is not None and event_times[b] is not None
                        else None
                        for key, (a, b) in INTERVALS.items()
                    }
                    input_ns = event_times["ui_text_send"]
                    artificial_total = artificial_wait_ms(events, input_ns) if input_ns else None
                    artificial = (
                        artificial_wait_ms(events, input_ns, event_times["ui_answer_commit"])
                        if input_ns
                        else None
                    )
                    case = {
                        "case_id": case_id,
                        "condition": condition,
                        "warmup": index == 0,
                        "success": error is None and not missing,
                        "error": error,
                        "missing": missing,
                        "events_ns": event_times,
                        "intervals_ms": intervals,
                        "artificial_wait_ms": artificial,
                        "artificial_wait_total_ms": artificial_total,
                        "trace_write_total_ms": context["trace_write_ns"] / 1e6,
                        "cancelled_answer_displayed": observed("old_answer_render_attempt")
                        if live
                        else OLD in window.chat_view.toPlainText(),
                        "cancelled_answer_saved": old_saved,
                        "cancelled_answer_synthesized": (
                            context["old_speech"] in "".join(voice_texts)
                            if context["old_speech"]
                            and context["old_speech"] != (replies[0] if replies else None)
                            else None
                        )
                        if live
                        else OLD in voice_texts,
                        "cancelled_answer_render_attempts": sum(
                            row["event"] == "old_answer_render_attempt" for row in events
                        ),
                        "cancelled_answer_save_attempts": sum(
                            row["event"] == "assistant_save_started" and row["old_answer"]
                            for row in events
                        ),
                        "http_request_counts": {
                            kind: sum(
                                row["event"]
                                in {
                                    "llm_request_start",
                                    "old_request_start",
                                    "search_request_start",
                                }
                                and row.get("kind", "search") == kind
                                for row in events
                            )
                            for kind in ("old", "decision", "search", "final")
                        },
                        "research_entry_count": sum(
                            row["event"] == "research_flow_entered" for row in events
                        ),
                        "mode": "live" if live else "http_mock",
                        "usage": backend._service.last_usage.model_dump()
                        if backend._service.last_usage
                        else None,
                        "old_answer_returned": context["old_speech"] is not None,
                        "cancelled_answer_mock_played": (
                            context["old_speech"]
                            in "".join(
                                row.get("playback_text") or ""
                                for row in events
                                if row["event"] == "mock_playback_start"
                            )
                            if context["old_speech"]
                            and context["old_speech"] != (replies[0] if replies else None)
                            else None
                        )
                        if live
                        else any(
                            OLD in (row.get("playback_text") or "")
                            for row in events
                            if row["event"] == "mock_playback_start"
                        ),
                        "old_speech": context["old_speech"] if live else None,
                        "answer_speech": replies[0] if live and replies else None,
                    }
                    save(run / "result.json", case)
                    cases.append(case)
                    save(directory / "cases.json", cases)
                    save(directory / "summary.json", summarize(cases))
                    save(directory / "case-state.json", {"case_id": case_id, "active": False})
                    print(f"{case_id}: {'passed' if case['success'] else 'FAILED'}", flush=True)
                    if window._thread is not None or window._proactive_thread is not None:
                        raise RuntimeError(
                            "処理未終了のため保存して停止。監視プロセスが回収します。"
                        )
                    window.close()
                    window.deleteLater()
                    app.processEvents()
                    voice.close()
                    backend.close()
                    context.update(window=None, backend=None)
                    if not case["success"]:
                        raise RuntimeError("必須証拠の欠落または不成立で計測を停止。")
    finally:
        after = protected()
        save(
            directory / "protection.json",
            {
                "protected_files_unchanged": before == after,
                "normal_qsettings_unchanged": normal_settings_before == settings_hash(),
                "blocked_network_attempts": len(blocked),
                "external_http_requests": budget.requests if live else 0,
                "microphone_used": False,
                "real_audio_used": False,
                "pmx_vmd_loaded": False,
                "source_changes": [
                    key for key in set(before) | set(after) if before.get(key) != after.get(key)
                ],
            },
        )
        if live:
            real_transport.close()
    return 0


def supervise(directory, samples, *, live=False, source_root=None, budget_directory=None):
    directory.mkdir(parents=True, exist_ok=False)
    measurement_plan = plan(samples)
    if live:
        measurement_plan.update(
            mode="live",
            case_limit_seconds=90,
            deadline_change_reason="live search has decision, external search and final answer requests; 30 seconds can cut valid evidence",
            artificial_http_delay_ms=0,
            inputs={key: value[0] for key, value in TEXTS.items()},
            search_input="Python公式のリストappendの説明を検索で確認して、出典付きで一文にまとめて。",
            fixed_response=False,
            source_root=str(source_root or ROOT / "src"),
            send_scope="artificial persona and inputs; public Python documentation search",
            api_request_limit=380,
            api_budget_usd=8,
        )
    save(directory / "plan.json", measurement_plan)
    started = time.monotonic()
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.measure_companion_wait",
                "--worker",
                "--directory",
                str(directory),
                "--samples",
                str(samples),
                *(["--live"] if live else []),
                *(["--source-root", str(source_root)] if source_root else []),
                *(["--budget-directory", str(budget_directory)] if budget_directory else []),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        save(
            directory / "supervisor-start.json",
            {
                "worker_pid": process.pid,
                "started_monotonic": started,
                "series_deadline_monotonic": started + 2700,
                "case_limit_seconds": 90 if live else 30,
            },
        )
        reason = None
        while process.poll() is None:
            state_path = directory / "case-state.json"
            try:
                state = (
                    json.loads(state_path.read_text(encoding="utf-8"))
                    if state_path.exists()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                state = {}
            if (directory / "STOP").exists():
                reason = "STOP file"
            elif time.monotonic() - started > 2700:
                reason = "series deadline"
            elif state.get("active") and time.monotonic() - state["started_monotonic"] > (
                90 if live else 30
            ):
                reason = "case deadline"
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
            time.sleep(0.1)
    save(
        directory / "supervisor-end.json",
        {
            "exit_code": process.returncode,
            "stop_reason": reason,
            "elapsed_seconds": time.monotonic() - started,
            "worker_reaped": process.poll() is not None,
        },
    )
    return process.returncode or (1 if reason else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--budget-directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "local" / "diagnostics"):
        parser.error("結果はプロジェクトのlocal/diagnostics内に保存してください。")
    if args.samples < 1:
        parser.error("標本数は1以上です。")
    for path in (args.source_root, args.budget_directory):
        if path is not None and not path.resolve().is_relative_to(ROOT / "local" / "diagnostics"):
            parser.error("隔離コードと予算記録もlocal/diagnostics内です。")
    options = dict(
        live=args.live,
        source_root=args.source_root.resolve() if args.source_root else None,
        budget_directory=args.budget_directory.resolve() if args.budget_directory else None,
    )
    return (
        worker(directory, args.samples, **options)
        if args.worker
        else supervise(directory, args.samples, **options)
    )


if __name__ == "__main__":
    raise SystemExit(main())
