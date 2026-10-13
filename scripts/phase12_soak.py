"""隔離した実PMXの長時間検査。製品の設定・保存形式は変更しない。"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now() -> str:
    return datetime.now().astimezone().isoformat()


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.monotonic()
    for attempt in range(5):
        try:
            temporary.replace(path)
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            last_error = exc.winerror
            # Windowsの読取ハンドルが閉じるまで、計100msだけ待つ。
            time.sleep(0.01 * (attempt + 1))
        else:
            if attempt:
                append(
                    path.parent / "atomic-save-retries.jsonl",
                    {
                        "at": now(),
                        "file": path.name,
                        "retries": attempt,
                        "winerror": last_error,
                        "elapsed_ms": (time.monotonic() - started) * 1000,
                    },
                )
            return


def append(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes() -> dict:
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/phase12*.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def distribution(values) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
    }


def active_credit(wall: float, awake: float, *, visible: bool, frames: int) -> float:
    """停止・スリープ・表示停止が混ざった計測区間を完走に含めない。"""
    if not visible or frames < 1 or wall > 10 or abs(wall - awake) > 1:
        return 0.0
    return max(0.0, min(wall, awake))


def slope_per_hour(points) -> float | None:
    if len(points) < 2:
        return None
    origin = points[0][0]
    xs = [(x - origin) / 3600 for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if not denominator:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def awake_seconds() -> float:
    ticks = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(ticks)):
        raise ctypes.WinError()
    return ticks.value / 10_000_000


def resources() -> dict:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_ws",
                "ws",
                "peak_pool",
                "pool",
                "peak_nonpool",
                "nonpool",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    process = ctypes.c_void_p(-1)
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    handles = ctypes.c_ulong()
    if not ctypes.windll.kernel32.GetProcessHandleCount(process, ctypes.byref(handles)):
        raise ctypes.WinError()

    # NtQueryInformationProcessの基本情報にはスレッド数がないためToolhelpを使う。
    class ThreadEntry(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulong)
            for name in ("size", "usage", "id", "owner", "priority", "delta", "flags")
        ]

    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    handle = kernel.CreateToolhelp32Snapshot(4, 0)
    entry = ThreadEntry()
    entry.size = ctypes.sizeof(entry)
    count = 0
    try:
        found = kernel.Thread32First(ctypes.c_void_p(handle), ctypes.byref(entry))
        while found:
            count += entry.owner == os.getpid()
            found = kernel.Thread32Next(ctypes.c_void_p(handle), ctypes.byref(entry))
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))
    return {
        "working_set_mb": counters.ws / 2**20,
        "private_mb": counters.private / 2**20,
        "handles": handles.value,
        "threads": count,
        "cpu_seconds": time.process_time(),
        "gpu_percent": None,
        "gpu_reason": "GPU per-process counter unavailable in runner",
    }


def desktop_unlocked() -> bool:
    user = ctypes.windll.user32
    user.OpenInputDesktop.restype = ctypes.c_void_p
    handle = user.OpenInputDesktop(0, False, 0x0001)
    if not handle:
        return False
    try:
        name = ctypes.create_unicode_buffer(256)
        needed = ctypes.c_ulong()
        ok = user.GetUserObjectInformationW(
            ctypes.c_void_p(handle), 2, name, ctypes.sizeof(name), ctypes.byref(needed)
        )
        return bool(ok and name.value.lower() == "default")
    finally:
        user.CloseDesktop(ctypes.c_void_p(handle))


def protected_hashes(runtime) -> dict:
    paths = {
        ROOT / "data/conversations.db",
        *ROOT.glob("local/*.yaml"),
        *ROOT.glob("config/*.yaml"),
    }
    for key in ("persona_path", "audio_config_path", "memory_config_path", "avatar_config_path"):
        paths.add((ROOT / getattr(runtime.app, key)).resolve())
    paths.add(runtime.app_config_path)
    # 素材は一度だけ読み、通常更新中の再読込にしない。
    paths.update(ROOT.glob("local/models/mmd/**/*"))
    paths.update(ROOT.glob("local/**/*.vmd"))
    paths.update(ROOT.glob("local/ui/*"))
    return {str(p): digest(p) for p in paths if p.is_file()}


def normal_settings_hash() -> str:
    from PySide6.QtCore import QSettings

    settings = QSettings("multi-llm-companion", "desktop-companion")
    raw = repr(sorted((key, str(settings.value(key))) for key in settings.allKeys()))
    return hashlib.sha256(raw.encode()).hexdigest()


def voice_probe(directory: Path) -> int:
    import httpx

    from multi_llm_companion.config import load_runtime_config
    from multi_llm_companion.integrations import build_voice_output

    runtime = load_runtime_config(ROOT)
    config = runtime.audio.model_copy(deep=True)
    config.voice_output.enabled = True
    config.voice_output.timeout = 5
    voice = build_voice_output(config)
    result = {
        "at": now(),
        "paid_requests": 0,
        "local_get_requests": 0,
        "synthesis_requests": 0,
        "playback": False,
        "microphone": False,
    }
    original = httpx.HTTPTransport.handle_request
    endpoint = httpx.URL(voice._base_url + "/voices")

    def guard(transport, request):
        if (
            request.url.host != "127.0.0.1"
            or request.url.port != 32766
            or request.method != "GET"
            or request.url.path != endpoint.path
        ):
            raise RuntimeError("voice probe only permits local GET /voices")
        result["local_get_requests"] += 1
        return original(transport, request)

    try:
        with patch.object(httpx.HTTPTransport, "handle_request", guard):
            requested = voice._voice_name
            selected, _version = voice._resolve_voice()
        result.update(requested=requested, selected=selected)
        result["matched"] = selected == requested and "tanaka" in selected.lower()
        result["status"] = (
            "deferred_no_verified_per_app_mute" if result["matched"] else "deferred_voice_mismatch"
        )
    except Exception as exc:
        # 例外本文は接続資格情報を含む可能性があるため保存しない。
        result.update(status="deferred_local_api_unavailable", error_type=type(exc).__name__)
    save(directory / "voice-probe.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def supervise(args) -> int:
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "child",
        str(directory),
        "--warmup",
        str(args.warmup),
        "--load",
        str(args.load),
        "--idle",
        str(args.idle),
    ]
    stdout = (directory / "app.log").open("w", encoding="utf-8")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    save(
        directory / "launch.json",
        {
            "run_id": directory.name,
            "watchdog_pid": os.getpid(),
            "app_pid": process.pid,
            "command": command,
            "started_at": now(),
            "planned_seconds": args.warmup + args.load + args.idle,
            "deadline_seconds": 36000,
        },
    )
    reason = None
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            state = directory / "state.json"
            stale = time.time() - state.stat().st_mtime if state.exists() else elapsed
            if elapsed > 36000:
                reason = "hard_deadline_10_hours"
            elif stale > (180 if not state.exists() else 60):
                reason = "gui_heartbeat_timeout"
            if reason:
                save(
                    directory / "watchdog-failure.json",
                    {
                        "reason": reason,
                        "at": now(),
                        "elapsed_seconds": elapsed,
                        "app_pid": process.pid,
                    },
                )
                # dump_traceback_laterが先にPythonスタックを保存する。
                cleanup = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                save(
                    directory / "watchdog-cleanup.json",
                    {"returncode": cleanup.returncode, "output": cleanup.stdout + cleanup.stderr},
                )
                process.wait(timeout=20)
                break
            time.sleep(5)
    finally:
        stdout.close()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    success = process.returncode == 0 and result.get("status") == "passed" and not reason
    save(
        directory / "watchdog-result.json",
        {
            "at": now(),
            "exit_code": process.returncode,
            "status": "passed" if success else "failed",
            "reason": reason,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return 0 if success else 1


def child(args) -> int:
    directory = args.directory.resolve()
    stack_log = (directory / "stacks.log").open("w", encoding="utf-8")
    faulthandler.enable(stack_log)
    faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
    print("child imports", now(), flush=True)
    os.environ["QT_QPA_PLATFORM"] = "windows"
    import httpx
    import numpy as np
    import yaml
    from PySide6 import __version__ as qt_version
    from PySide6.QtCore import QSettings, Qt, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from multi_llm_companion.config import PersonaConfig, load_runtime_config
    from multi_llm_companion.integrations import NullSpeechRecognizer, NullVoiceOutput
    from multi_llm_companion.providers import MockProvider
    from multi_llm_companion.services.image_sharing import ImageSharingSession, OpenAIImageAnalysis
    from multi_llm_companion.services.web_search import OpenAIWebSearch, WebSearchSession
    from multi_llm_companion.ui.chat_window import ChatWindow
    from multi_llm_companion.ui.render_quality import configure_default_surface_format
    from scripts.phase9_acceptance_app import AcceptanceBackend, isolated_runtime

    print("imports complete", now(), flush=True)
    original = load_runtime_config(ROOT)
    before = protected_hashes(original)
    settings_before = normal_settings_hash()
    save(directory / "protected-before.json", before)
    code_before = source_hashes()
    audio = original.audio.model_copy(deep=True)
    audio.speech_recognition.enabled = False
    audio.voice_output.enabled = False
    persona = PersonaConfig(name="隔離検査", system_prompt="人工の試験会話です。")
    runtime = isolated_runtime(
        replace(original, audio=audio, persona=persona), directory / "runtime"
    )
    runtime.app.persona_path = str(directory / "runtime/persona.yaml")
    runtime = replace(
        runtime,
        active_provider=replace(
            runtime.active_provider,
            name="mock",
            provider="mock",
            api_key="",
            base_url="http://127.0.0.1",
            model="mock",
        ),
    )
    runtime.app.active_provider = "mock"
    for name, value in (("persona", persona), ("app", runtime.app)):
        (directory / f"runtime/{name}.yaml").write_text(
            yaml.safe_dump(value.model_dump(mode="json", by_alias=True), allow_unicode=True),
            encoding="utf-8",
        )
    os.chdir(directory)
    configure_default_surface_format(runtime.avatar.three_d.render_quality)
    app = QApplication([])
    result = {
        "run_id": directory.name,
        "started_at": now(),
        "status": "running",
        "errors": [],
        "external_requests": 0,
        "external_attempts_blocked": 0,
        "microphone": False,
        "real_audio": False,
        "audio_mode": "silent Mock",
        "phases": {},
        "cases": [],
    }
    limits = {
        "ui_p95_ms": 500,
        "ui_max_ms": 5000,
        "private_growth_mb": 512,
        "idle_private_growth_mb": 128,
        "idle_private_slope_mb_hour": 32,
        "handle_growth": 300,
        "thread_growth": 12,
        "queue_timeout_seconds": 20,
        "invalid_active_seconds": 300,
        "consecutive_bad_windows": 3,
    }
    save(
        directory / "configuration.json",
        {
            "python": sys.version,
            "pyside": qt_version,
            "source_hashes": code_before,
            "screen": [1100, 760],
            "dpi": app.primaryScreen().devicePixelRatio(),
            "avatar": runtime.avatar.model_dump(mode="json"),
            "provider": "Mock",
            "persona": "artificial",
            "warmup_seconds": args.warmup,
            "load_seconds": args.load,
            "idle_seconds": args.idle,
            "send_interval_seconds": 20,
            "send_cap": 240,
            "proactive_idle": False,
            "limits": limits,
            "limits_reason": "pilot-004: UI p95 25ms, heavy mean 15.5ms/33ms; absolute limits frozen before long run",
        },
    )
    timers = []
    window = backend = None
    exit_status = 1
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000003)

    def forbidden(*_args, **_kwargs):
        result["external_attempts_blocked"] += 1
        raise AssertionError("network disabled for isolated soak")

    class SilentVoice(NullVoiceOutput):
        def __init__(self):
            super().__init__("無音Mock")
            self.until = 0.0
            self.spoken = []

        @property
        def is_available(self):
            return True

        @property
        def is_speaking(self):
            return time.monotonic() < self.until

        @property
        def lip_sync_state(self):
            from multi_llm_companion.integrations.lip_sync import LipSyncState

            active = self.is_speaking
            return LipSyncState(
                is_active=active,
                mouth_shape="a" if active else "neutral",
                is_timeline_active=active,
            )

        def speak(self, text):
            assert all(token not in text for token in ("[S1]", "取得:", '"speech":'))
            self.spoken.append(len(text))
            self.until = time.monotonic() + 0.7

        def stop(self):
            self.until = 0.0

    voice = SilentVoice()
    provider = MockProvider(response_text='{"speech":"うん、検査の会話を続けよう。"}')
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(httpx.HTTPTransport, "handle_request", forbidden))
            stack.enter_context(
                patch.object(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
            )
            stack.enter_context(patch.object(socket.socket, "connect", forbidden))
            stack.enter_context(patch.object(socket.socket, "connect_ex", forbidden))
            stack.enter_context(
                patch("multi_llm_companion.application.build_provider", return_value=provider)
            )
            stack.enter_context(
                patch("multi_llm_companion.application.build_voice_output", return_value=voice)
            )
            stack.enter_context(
                patch(
                    "multi_llm_companion.application.build_speech_recognizer",
                    return_value=NullSpeechRecognizer(),
                )
            )
            stack.enter_context(
                patch(
                    "multi_llm_companion.ui.chat_window.QSettings",
                    lambda *_: QSettings(
                        str(directory / "runtime/window.ini"), QSettings.Format.IniFormat
                    ),
                )
            )
            backend = AcceptanceBackend(runtime)
            window = ChatWindow(backend)
            window.setWindowTitle(f"フェーズ12【隔離・無音検査】{directory.name}")
            window.resize(1100, 760)
            window.show()
            wrapper = window.pmx_preview_widget
            dense = wrapper._dense_view
            window._settings.setValue("proactive/enabled", False)
            frames = [0]
            dense.frameSwapped.connect(lambda: frames.__setitem__(0, frames[0] + 1))
            metrics = defaultdict(list)

            def trace(event, elapsed_seconds, *, started_at, period_ms):
                metrics[event + "_interval_ms"].append(elapsed_seconds * 1000)
                metrics[event + "_duration_ms"].append((time.perf_counter() - started_at) * 1000)

            wrapper._write_timer_trace = trace
            physics = dense._advance_physics

            def measure_physics(*a, **kw):
                start = time.perf_counter()
                try:
                    return physics(*a, **kw)
                finally:
                    metrics["physics_duration_ms"].append((time.perf_counter() - start) * 1000)

            dense._advance_physics = measure_physics
            image = QImage(80, 60, QImage.Format.Format_RGB32)
            image.fill(0xFFFF5544)
            image_path = directory / "runtime/fixture.png"
            image.save(str(image_path))

            def wait(predicate, seconds=20):
                end = time.monotonic() + seconds
                while not predicate():
                    QTest.qWait(20)
                    if time.monotonic() > end:
                        raise AssertionError("operation timeout")

            def idle():
                return (
                    not window._busy
                    and window._thread is None
                    and window._voice_thread is None
                    and window._proactive_thread is None
                    and not window._image_fetches
                    and window._pending_voice_text is None
                    and not voice.is_speaking
                )

            def send(text):
                window.input_edit.setPlainText(text)
                window.send_button.click()

            state = {
                "phase": "warmup",
                "active_seconds": 0.0,
                "excluded_seconds": 0.0,
                "phase_started_at": now(),
                "phase_wall_start": time.monotonic(),
                "requests": 0,
                "case_index": 0,
                "last_send": 0.0,
                "in_case": False,
            }
            last = {
                "mono": time.monotonic(),
                "awake": awake_seconds(),
                "frames": 0,
                "cpu": time.process_time(),
                "summary": time.monotonic(),
                "probe": time.monotonic(),
                "baseline": None,
                "idle_baseline": None,
            }
            bad_windows = defaultdict(int)
            rows = []

            def failure(exc):
                if result["status"] == "running":
                    result["status"] = "failed"
                    result["errors"].append(str(exc))
                    save(
                        directory / "failure.json", {"at": now(), "state": state, "error": str(exc)}
                    )
                    faulthandler.dump_traceback(file=stack_log)
                    app.quit()

            def probe():
                current = time.monotonic()
                metrics["ui_delay_ms"].append(max(0, (current - last["probe"] - 0.1) * 1000))
                last["probe"] = current

            def payload(text, annotations=None):
                return {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": annotations or [],
                                }
                            ],
                        }
                    ],
                }

            cases = (
                "short",
                "long",
                "search",
                "no_evidence",
                "search_timeout",
                "image",
                "image_error",
                "short",
                "cancel",
                "new_input",
                "llm_disconnect",
                "short",
            )

            def run_case(kind=None):
                kind = kind or cases[state["case_index"] % len(cases)]
                state["case_index"] += 1
                state["in_case"] = True
                start = time.monotonic()
                row = {"kind": kind, "at": now(), "phase": state["phase"]}
                try:
                    response = "検査の返答だよ。" + (
                        "読み返し用の長い文章だよ。" * 80 if kind == "long" else ""
                    )
                    provider._response_text = json.dumps({"speech": response}, ensure_ascii=False)
                    provider._responder = None
                    if kind in {"search", "no_evidence", "search_timeout"}:

                        def search_handler(_request):
                            if kind == "search_timeout":
                                raise httpx.ReadTimeout("fixture timeout")
                            annotations = (
                                [
                                    {
                                        "type": "url_citation",
                                        "start_index": 0,
                                        "end_index": 5,
                                        "url": "https://news.example/article",
                                        "title": "人工資料",
                                    }
                                ]
                                if kind == "search"
                                else []
                            )
                            data = payload("検査の資料。[1]", annotations)
                            data["output"].insert(
                                0,
                                {
                                    "type": "web_search_call",
                                    "status": "completed",
                                    "action": {"type": "search"},
                                },
                            )
                            return httpx.Response(200, json=data)

                        backend._service.web_search = WebSearchSession(
                            OpenAIWebSearch(
                                api_key="fixture",
                                model="gpt-6-luna",
                                transport=httpx.MockTransport(search_handler),
                            )
                        )
                        if kind == "search":
                            provider._response_text = '{"speech":"検査資料が見つかったよ。[S1]"}'
                        send("今日の科学ニュースを調べて")
                    elif kind in {"image", "image_error"}:
                        backend._service.image_sharing = ImageSharingSession(
                            OpenAIImageAnalysis(
                                api_key="fixture",
                                model="gpt-6-luna",
                                transport=httpx.MockTransport(
                                    lambda _: (
                                        httpx.Response(500)
                                        if kind == "image_error"
                                        else httpx.Response(200, json=payload("赤い四角形です。"))
                                    )
                                ),
                            )
                        )
                        window._attach_image(str(image_path))
                        send("画像には何がある？")
                    elif kind in {"cancel", "new_input"}:
                        started, release = threading.Event(), threading.Event()

                        def delayed(_request):
                            started.set()
                            assert release.wait(5), "fixture release timeout"
                            return httpx.Response(200, json=payload("破棄対象の遅い返答"))

                        backend._service.image_sharing = ImageSharingSession(
                            OpenAIImageAnalysis(
                                api_key="fixture",
                                model="gpt-6-luna",
                                transport=httpx.MockTransport(delayed),
                            )
                        )
                        provider._response_text = '{"speech":"破棄対象の遅い返答"}'
                        window._attach_image(str(image_path))
                        send("この画像を確認して")
                        try:
                            wait(started.is_set)
                            generation = window._stream_generation
                            if kind == "cancel":
                                window._stop_voice_output()
                            else:
                                window.input_edit.setPlainText("新しい入力を優先するよ")
                            assert window._stream_generation > generation
                        finally:
                            release.set()
                        wait(idle)
                        assert "破棄対象の遅い返答" not in window.chat_view.toPlainText()
                        assert (
                            not wrapper._voice_active
                            and window._continuous_voice_waiting_generation is None
                        )
                        window.input_edit.clear()
                    elif kind == "llm_disconnect":
                        from multi_llm_companion.providers import ProviderError

                        def disconnected(_messages):
                            raise ProviderError("検査用の接続断")

                        provider._responder = disconnected
                        send("接続断の検査")
                    else:
                        send("人工データの会話を続けよう")
                    state["requests"] += 1
                    wait(idle)
                    if kind in {"image_error", "llm_disconnect"}:
                        assert not window.error_label.isHidden()
                    if kind == "short":
                        assert window.error_label.isHidden()
                        assert "検査の返答だよ。" in window.chat_view.toPlainText()
                    if kind == "search":
                        assert "https://news.example/article" in window.chat_view.toHtml()
                    if kind == "long":
                        scroll = window.chat_view.verticalScrollBar()
                        scroll.setValue(scroll.maximum() // 3)
                        selected = scroll.value()
                        QTest.qWait(300)
                        assert abs(scroll.value() - selected) <= 2
                    assert window.input_edit.isEnabled()
                    row.update(status="passed", elapsed_seconds=time.monotonic() - start)
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                    failure(row["error"])
                finally:
                    provider._responder = None
                    state["in_case"] = False
                    result["cases"].append(row)
                    append(directory / "cases.jsonl", row)

            def finish_phase():
                result["phases"][state["phase"]] = {
                    "started_at": state["phase_started_at"],
                    "ended_at": now(),
                    "wall_seconds": time.monotonic() - state["phase_wall_start"],
                    "active_seconds": state["active_seconds"],
                    "excluded_seconds": state["excluded_seconds"],
                    "requests_total": state["requests"],
                    "status": "passed",
                }

            def transition(phase):
                finish_phase()
                state.update(
                    phase=phase,
                    active_seconds=0.0,
                    excluded_seconds=0.0,
                    phase_started_at=now(),
                    phase_wall_start=time.monotonic(),
                )
                window.grab().save(str(directory / (phase + "-start.png")))
                save(directory / "progress.json", result)

            def health():
                if result["status"] != "running":
                    return
                try:
                    faulthandler.cancel_dump_traceback_later()
                    faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
                    current, awake = time.monotonic(), awake_seconds()
                    delta, frame_delta = current - last["mono"], frames[0] - last["frames"]
                    unlocked = desktop_unlocked()
                    visible = window.isVisible() and not window.isMinimized() and unlocked
                    credit = active_credit(
                        delta, awake - last["awake"], visible=visible, frames=frame_delta
                    )
                    state["active_seconds"] += credit
                    state["excluded_seconds"] += delta - credit
                    resource = resources()
                    resource["cpu_percent_one_core"] = (
                        100 * (resource["cpu_seconds"] - last["cpu"]) / delta
                    )
                    last.update(
                        mono=current, awake=awake, frames=frames[0], cpu=resource["cpu_seconds"]
                    )
                    pending = sum(
                        (
                            window._thread is not None,
                            window._voice_thread is not None,
                            window._proactive_thread is not None,
                            bool(window._image_fetches),
                            window._pending_voice_text is not None,
                            voice.is_speaking,
                        )
                    )
                    record = {
                        "at": now(),
                        "monotonic": current,
                        "awake_seconds": awake,
                        **state,
                        **resource,
                        "frame_delta": frame_delta,
                        "frame_total": frames[0],
                        "visible": visible,
                        "desktop_unlocked": unlocked,
                        "exposed": window.windowHandle().isExposed(),
                        "pending": pending,
                        "audio_queue": int(voice.is_speaking),
                        "audio_kind": "Mock",
                        "external_requests": result["external_requests"],
                        "external_attempts_blocked": result["external_attempts_blocked"],
                    }
                    append(directory / "health.jsonl", record)
                    save(directory / "state.json", record)
                    assert result["external_attempts_blocked"] == 0, "unexpected network attempt"
                    assert state["excluded_seconds"] < limits["invalid_active_seconds"], (
                        "render/awake interval missing"
                    )
                    assert dense._available, "real PMX renderer unavailable"
                    assert not dense._cloth_state.error, "cloth physics error"
                    pose = dense._motion_mixer.current
                    assert all(
                        np.isfinite(v).all()
                        for v in (*pose.rotations.values(), *pose.translations.values())
                    ), "nonfinite pose"
                    if current - last["summary"] >= 30:
                        row = {
                            "at": now(),
                            "phase": state["phase"],
                            "active_seconds": state["active_seconds"],
                            "resource": resource,
                            "timings": {k: distribution(v) for k, v in metrics.items()},
                            "pending": pending,
                            "frames": frames[0],
                            "disk_bytes": {
                                p.name: p.stat().st_size
                                for p in (directory / "runtime").iterdir()
                                if p.is_file()
                            },
                        }
                        rows.append(row)
                        append(directory / "metrics.jsonl", row)
                        metrics.clear()
                        last["summary"] = current
                        if state["phase"] != "warmup":
                            base = last["baseline"]
                            assert (
                                resource["private_mb"] - base["private_mb"]
                                < limits["private_growth_mb"]
                            ), "private memory growth"
                            assert (
                                resource["handles"] - base["handles"] < limits["handle_growth"]
                            ), "handle growth"
                            assert (
                                resource["threads"] - base["threads"] < limits["thread_growth"]
                            ), "thread growth"
                            for event, period in (
                                ("pmx_lightweight_tick", wrapper._timer.interval()),
                                ("pmx_heavy_tick", wrapper._heavy_timer.interval()),
                            ):
                                timing = row["timings"].get(event + "_duration_ms")
                                assert timing, "timer stopped"
                                bad_windows[event] = (
                                    bad_windows[event] + 1 if timing["mean"] > period else 0
                                )
                                assert bad_windows[event] < 3, "sustained callback overrun"
                            ui = row["timings"]["ui_delay_ms"]
                            assert ui["max"] < limits["ui_max_ms"], "UI stall"
                            bad_windows["ui"] = (
                                bad_windows["ui"] + 1 if ui["p95"] > limits["ui_p95_ms"] else 0
                            )
                            assert bad_windows["ui"] < 3, "sustained UI delay"
                            if state["phase"] == "idle":
                                assert pending == 0, "idle queue not empty"
                                assert (
                                    resource["private_mb"] - last["idle_baseline"]["private_mb"]
                                    < limits["idle_private_growth_mb"]
                                ), "idle memory growth"
                                idle_rows = [r for r in rows if r["phase"] == "idle"]
                                if state["active_seconds"] >= 1800:
                                    growth = slope_per_hour(
                                        [
                                            (r["active_seconds"], r["resource"]["private_mb"])
                                            for r in idle_rows
                                        ]
                                    )
                                    assert growth < limits["idle_private_slope_mb_hour"], (
                                        "idle memory slope"
                                    )
                    if (directory / "STOP").exists():
                        raise RuntimeError("user_stop_requested")
                    if state["in_case"]:
                        return
                    target = {"warmup": args.warmup, "load": args.load, "idle": args.idle}[
                        state["phase"]
                    ]
                    if state["active_seconds"] >= target and idle():
                        if state["phase"] == "warmup":
                            last["baseline"] = resource
                            transition("load")
                        elif state["phase"] == "load":
                            window._stop_voice_output()
                            last["idle_baseline"] = resource
                            transition("idle")
                        else:
                            finish_phase()
                            # 待機後のUI操作と保存を、計測区間の外で確認する。
                            run_case("short")
                            assert result["status"] == "running"
                            window.history_button.click()
                            window.options_button.click()
                            assert window.options_dialog.isVisible()
                            window.options_dialog.close()
                            window._stop_voice_output()
                            wait(idle)
                            result["post_idle_operations"] = True
                            result["status"] = "passed"
                            app.quit()
                    elif (
                        state["phase"] != "idle"
                        and current - state["last_send"] >= 20
                        and idle()
                        and state["requests"] < 240
                    ):
                        state["last_send"] = current
                        run_case()
                except Exception:
                    failure(traceback.format_exc())

            for interval, callback in ((100, probe), (5000, health)):
                timer = QTimer()
                timer.setTimerType(Qt.TimerType.PreciseTimer)
                timer.setInterval(interval)
                timer.timeout.connect(callback)
                timer.start()
                timers.append(timer)
            app.exec()
            for timer in timers:
                timer.stop()
            window._stop_voice_output()
            wait(idle)
            result["close_accepted"] = window.close()
            QTest.qWait(150)
            backend.close()
            if result["status"] == "running":
                result["status"] = "interrupted"
            assert result["close_accepted"], "close refused"
            result["spoken_mock_segments"] = len(voice.spoken)
            result["physics"] = dense.physics_status_text
            result["timing_summary"] = {
                key: distribution([r["resource"][key] for r in rows])
                for key in (
                    "working_set_mb",
                    "private_mb",
                    "handles",
                    "threads",
                    "cpu_percent_one_core",
                )
            }
            result["phase_trends"] = {}
            for phase in ("warmup", "load", "idle"):
                selected = [row for row in rows if row["phase"] == phase]
                result["phase_trends"][phase] = {
                    "samples": len(selected),
                    "private_mb_per_hour": slope_per_hour(
                        [(row["active_seconds"], row["resource"]["private_mb"]) for row in selected]
                    ),
                    "working_set_mb_per_hour": slope_per_hour(
                        [
                            (row["active_seconds"], row["resource"]["working_set_mb"])
                            for row in selected
                        ]
                    ),
                }
    except Exception:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        faulthandler.cancel_dump_traceback_later()
        result["ended_at"] = now()
        result["protected_files_unchanged"] = before == protected_hashes(original)
        result["normal_qsettings_unchanged"] = settings_before == normal_settings_hash()
        result["source_unchanged"] = code_before == source_hashes()
        if not all(
            result[k]
            for k in ("protected_files_unchanged", "normal_qsettings_unchanged", "source_unchanged")
        ):
            result["status"] = "failed"
            result["errors"].append("protection or source hash mismatch")
        save(directory / "result.json", result)
        exit_status = 0 if result["status"] == "passed" else 1
    return exit_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "child", "voice-probe"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--warmup", type=float, default=300)
    parser.add_argument("--load", type=float, default=3600)
    parser.add_argument("--idle", type=float, default=28800)
    args = parser.parse_args()
    if args.mode == "voice-probe":
        return voice_probe(args.directory)
    return supervise(args) if args.mode == "supervise" else child(args)


if __name__ == "__main__":
    raise SystemExit(main())
