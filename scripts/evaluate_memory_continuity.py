"""人工記憶と60ターンを、現行ProviderのHTTP経路で隔離評価する。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import MemoryCategory, MemoryStatus
from multi_llm_companion.providers import OpenAICompatibleProvider
from multi_llm_companion.services.chat import ChatService
from multi_llm_companion.services.conversation_context import ConversationContextService
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from scripts.measure_companion_wait import protected, save, stats

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
FACTS = {
    "radio": "休日の楽しみは年代物のラジオを修理すること。",
    "garden": "庭では青いデルフィニウムを育てている。",
    "breakfast": "朝食はトーストを食べている。",
    "deleted": "楽器はオーボエを練習している。",
    "pending": "未承認の呼び名はクロマル。",
}
MIDDLE = "銀色のしおりをカナさんへ渡す約束をした。"
TAIL = "石の展示の合言葉は琥珀灯。"


def scenario():
    """期待値を回答の生成前に固定する。"""
    turns = [{"user": f"静かな雑談の第{i}回だよ。一文だけ返して。"} for i in range(1, 61)]
    turns[0] = {"user": "最初の話題は歩道の模様だよ。"}
    turns[19] = {"user": MIDDLE}
    turns[21] = {"user": "展示の感想を話すね。" * 35 + TAIL}
    turns[24] = {"user": "訂正: 待ち合わせは金曜ではなく木曜だよ。"}
    probes = {
        41: {"user": "カナさんに渡す約束のしおりはどんなもの？", "tokens": ["銀色"]},
        42: {"user": "石の展示の合言葉を教えて。", "tokens": ["琥珀灯"]},
        43: {
            "user": "休みの日の古い受信機いじり、私の楽しみを覚えている？",
            "fact": "radio",
            "tokens": ["ラジオ", "修理"],
        },
        44: {
            "user": "庭で育てている青い花の名前は？",
            "fact": "garden",
            "tokens": ["デルフィニウム"],
        },
        45: {
            "user": "訂正後の朝食は何だった？",
            "action": "correct",
            "fact": "breakfast",
            "tokens": ["ご飯"],
            "forbidden": ["トーストを食べ", "トーストだ"],
        },
        46: {
            "user": "保存済みの楽器の記憶はある？なければ分からないと言って。",
            "action": "delete",
            "forbidden": ["オーボエ"],
        },
        47: {
            "user": "承認された呼び名はある？なければ分からないと言って。",
            "forbidden": ["クロマル"],
        },
        48: {
            "user": "今の記憶は無効だから、庭の情報を覚えていると断言しないで。",
            "action": "disable",
            "forbidden": ["デルフィニウム"],
        },
        49: {
            "user": "庭で育てている青い花の名前をもう一度教えて。",
            "action": "enable",
            "fact": "garden",
            "tokens": ["デルフィニウム"],
        },
        50: {
            "user": "訂正した待ち合わせの曜日は？",
            "tokens": ["木曜"],
            "forbidden": ["金曜です", "金曜だ"],
        },
        55: {
            "user": "休暇の娯楽として無線装置を直すのは好きだった？",
            "fact": "radio",
            "tokens": ["ラジオ", "修理"],
        },
        60: {
            "user": "カナさんへのしおりと展示の合言葉を振り返って。",
            "tokens": ["銀色", "琥珀灯"],
        },
    }
    for number, probe in probes.items():
        turns[number - 1] = dict(probe, probe=True)
    return turns


class ApiBudget:
    """送信前の保守的予約と、失敗後の追加送信停止。"""

    def __init__(self, directory, limit=380):
        self.directory, self.limit = directory, limit
        self.requests = 0
        self.failed = False
        self.lock = threading.Lock()
        previous = directory / "api-budget.json"
        if previous.exists():
            ledger = json.loads(previous.read_text(encoding="utf-8"))
            self.requests = ledger["requests"]
            self.failed = ledger["failed"]
            self.limit = min(self.limit, ledger.get("max_requests", self.limit))

    def check(self, request):
        payload = json.loads(request.content)
        if (
            request.method != "POST"
            or str(request.url) not in {URL + "/chat/completions", URL + "/responses"}
            or payload.get("model") != MODEL
            or len(request.content) > 80_000
            or not 0
            < payload.get("max_completion_tokens", payload.get("max_output_tokens", 0))
            <= 1536
            or (
                str(request.url).endswith("/responses")
                and (payload.get("max_tool_calls") != 1 or len(request.content) > 10_000)
            )
        ):
            raise RuntimeError("許可した送信先・モデル・入力・出力枠の範囲外です。")
        with self.lock:
            if self.failed or self.requests >= self.limit or (self.directory / "STOP").exists():
                raise RuntimeError("APIの停止条件または要求上限に達しました。")
            self.requests += 1
            save(self.directory / "api-budget.json", self.summary())

    def summary(self):
        return {
            "requests": self.requests,
            "max_requests": self.limit,
            "reserved_usd": self.requests * 0.02,
            "budget_usd": min(8, self.limit * 0.02),
            "failed": self.failed,
        }


def api_key():
    """既存のキーだけを読み、値を出力・保存しない。"""
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("既存OPENAI_API_KEYがありません。")
    return key


def run(directory, live=False, budget_directory=None):
    directory.mkdir(parents=True, exist_ok=True)
    before = protected()
    save(directory / "protected-before.json", before)
    repo = SQLiteConversationRepository(directory / "artificial.db")
    ids = {}
    for name, content in FACTS.items():
        ids[name] = repo.create_memory(
            content, status=MemoryStatus.CANDIDATE if name == "pending" else MemoryStatus.APPROVED
        ).id
    for index in range(120):
        repo.create_memory(
            f"工作記録{index:03}では紙の箱の寸法を測定した。", status=MemoryStatus.APPROVED
        )
    memory = MemoryService(repo, MemoryConfig())
    prepare = memory.prepare_turn
    memory_timings = []

    def timed_prepare(user_text):
        started = time.perf_counter_ns()
        result = prepare(user_text)
        memory_timings.append((time.perf_counter_ns() - started) / 1e6)
        return result

    memory.prepare_turn = timed_prepare
    observations = []
    active = {"probe": {}, "number": 0}
    budget = ApiBudget(budget_directory or directory)
    transport = httpx.HTTPTransport(retries=0) if live else None

    def responder(request):
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"]
        observation = {
            "request_ns": time.perf_counter_ns(),
            "model": payload["model"],
            "stream": payload.get("stream"),
            "tools": bool(payload.get("tools")),
            "supports_research": provider._supports_research,
            "uses_research_flow": service._uses_research_flow,
            "prompt": prompt,
            "messages": payload["messages"],
        }
        observations.append(observation)
        if live:
            budget.check(request)
            try:
                response = transport.handle_request(request)
                if response.status_code >= 400:
                    budget.failed = True
                return response
            except Exception:
                budget.failed = True
                save(budget.directory / "api-budget.json", budget.summary())
                raise
        evidence = prompt + "\n" + "\n".join(item["content"] for item in payload["messages"][1:-1])
        probe = active["probe"]
        tokens = probe.get("tokens", [])
        speech = (
            "、".join(tokens) + "。"
            if tokens and all(token in evidence for token in tokens)
            else "今の参考情報では分からない。"
            if probe.get("probe")
            else "うん、静かな時間だね。"
        )
        arguments = {"speech": speech, "memory_updates": []}
        answer = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "artificial",
                                "type": "function",
                                "function": {
                                    "name": "reply_to_user",
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=answer)

    provider = OpenAICompatibleProvider(
        name="openai",
        base_url=URL,
        api_key=api_key() if live else "artificial-only",
        model=MODEL,
        max_output_tokens=1536,
        timeout=25,
        transport=httpx.MockTransport(responder),
    )
    service = ChatService(
        provider=provider,
        repository=repo,
        memory_service=memory,
        system_prompt="人工案内役。日本語で短く自然に返答する。既存の情報が見つからなければ正直に伝える。",
        conversation_context=ConversationContextService(
            max_recent_messages=12, summary_max_characters=900
        ),
    )
    conversation = service.start_new_conversation("人工60ターン")
    results = []
    error = None
    try:
        for number, probe in enumerate(scenario(), 1):
            save(
                directory / "case-state.json",
                {
                    "active": True,
                    "started_monotonic": time.monotonic(),
                    "case_id": f"memory-{number:02}",
                },
            )
            action = probe.get("action")
            if action == "correct":
                memory.update(ids["breakfast"], "朝食はご飯を食べている。", MemoryCategory.OTHER)
            elif action == "delete":
                memory.delete(ids["deleted"])
            elif action in {"disable", "enable"}:
                memory.set_enabled(action == "enable")
            active.update(probe=probe, number=number)
            started = time.perf_counter_ns()
            response = service.send_message(probe["user"])
            selected = memory.last_context_memories
            expected_id = ids.get(probe.get("fact"))
            usage = provider.get_usage()
            row = {
                "turn": number,
                "case_id": f"memory-{number:02}",
                "generation": service._stream_generation,
                "conversation_id": conversation.id,
                "input": probe["user"],
                "probe": bool(probe.get("probe")),
                "expected_tokens": probe.get("tokens", []),
                "forbidden": probe.get("forbidden", []),
                "expected_memory_id": expected_id,
                "selected_ids": [item.id for item in selected],
                "selected_characters": sum(len(item.content) for item in selected),
                "selected_count": len(selected),
                "candidate_recalled": expected_id in {item.id for item in selected}
                if expected_id
                else None,
                "speech": response.speech,
                "answer_tokens_ok": all(
                    token in response.speech for token in probe.get("tokens", [])
                ),
                "forbidden_reused": [
                    word for word in probe.get("forbidden", []) if word in response.speech
                ],
                "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
                "memory_selection_ms": memory_timings[-1],
                "usage": usage.model_dump() if usage else None,
                "provider_fragment_ns": None,
                "real_voice_start_ns": None,
                "saved_memory_ids": [item.id for item in memory.list_memories()],
                "request": observations[-1],
            }
            results.append(row)
            save(directory / f"turn-{number:02}.json", row)
            save(directory / "results.json", results)
            if budget.failed:
                raise RuntimeError("実API失敗により追加送信を停止しました。")
            print(f"turn {number}/60 recorded", flush=True)
        repo.close()
        repo = SQLiteConversationRepository(directory / "artificial.db")
        restored = repo.list_memories()
        restart = {
            "deleted_absent": ids["deleted"] not in {item.id for item in restored},
            "corrected_value": next(
                item.content for item in restored if item.id == ids["breakfast"]
            ),
            "pending_not_approved": next(
                item.status.value for item in restored if item.id == ids["pending"]
            ),
            "memory_count": len(restored),
        }
        save(directory / "restart.json", restart)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # 再開後も通常DBへ接続せず、この評価の接続だけを閉じる。
        service.close()
        repo.close()
        if transport:
            transport.close()
        rows = [row for row in results if row["probe"]]
        retrieval = [row for row in rows if row["expected_memory_id"]]
        save(
            directory / "summary.json",
            {
                "mode": "live" if live else "http_mock",
                "turns": len(results),
                "initial_memory_count": 125,
                "single_conversation_turns": len(results),
                "probe_count": len(rows),
                "candidate_success": sum(row["candidate_recalled"] for row in retrieval),
                "candidate_total": len(retrieval),
                "answer_token_success": sum(
                    row["answer_tokens_ok"] and not row["forbidden_reused"] for row in rows
                ),
                "answer_total": len(rows),
                "forbidden_reuse_count": sum(bool(row["forbidden_reused"]) for row in rows),
                "missing_turns": 60 - len(results),
                "elapsed": stats([row["elapsed_ms"] for row in results]),
                "memory_selection": stats([row["memory_selection_ms"] for row in results]),
                "max_selected_count": max((row["selected_count"] for row in results), default=0),
                "max_selected_characters": max(
                    (row["selected_characters"] for row in results), default=0
                ),
                "error": error,
                "budget": budget.summary(),
                "mock_answer_is_prompt_oracle": not live,
                "real_voice_measured": False,
            },
        )
        after = protected()
        save(
            directory / "protection.json",
            {
                "normal_files_unchanged": before == after,
                "source_or_normal_changes": [
                    key for key, value in before.items() if after.get(key) != value
                ],
                "normal_database_used": False,
                "synthetic_persona": True,
            },
        )
        save(directory / "case-state.json", {"active": False})
    return 1 if error else 0


def supervise(directory, live=False, source_root=None, budget_directory=None):
    directory.mkdir(parents=True, exist_ok=False)
    save(
        directory / "plan.json",
        {
            "mode": "live" if live else "http_mock",
            "turns": 60,
            "memories": 125,
            "scenario": scenario(),
            "mandatory": ["request", "selected_ids", "speech", "generation", "save", "restart"],
            "case_limit_seconds": 30,
            "series_limit_seconds": 2700,
            "stop": ["HTTP error", "request budget", "deadline", "STOP file"],
            "model": MODEL,
            "base_url": URL,
            "send_scope": "synthetic persona, artificial conversation and memories only",
            "real_audio": False,
            "normal_data": False,
        },
    )
    started = time.monotonic()
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        entry = [sys.executable, "-m", "scripts.evaluate_memory_continuity"]
        if source_root:
            entry = [
                sys.executable,
                "-c",
                "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('scripts.evaluate_memory_continuity',run_name='__main__')",
                str(source_root),
            ]
        process = subprocess.Popen(
            [
                *entry,
                "--worker",
                "--directory",
                str(directory),
                *(["--live"] if live else []),
                *(["--budget-directory", str(budget_directory)] if budget_directory else []),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        save(
            directory / "supervisor-start.json",
            {"worker_pid": process.pid, "deadline_monotonic": started + 2700},
        )
        reason = None
        while process.poll() is None:
            try:
                state = json.loads((directory / "case-state.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if (directory / "STOP").exists() or time.monotonic() - started > 2700:
                reason = "stop or series deadline"
            elif state.get("active") and time.monotonic() - state["started_monotonic"] > 30:
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
            "reason": reason,
            "elapsed_seconds": time.monotonic() - started,
            "reaped": process.poll() is not None,
        },
    )
    return process.returncode or (1 if reason else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--budget-directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "local" / "diagnostics"):
        parser.error("結果保存先はlocal/diagnostics内です。")
    for path in (args.source_root, args.budget_directory):
        if path is not None and not path.resolve().is_relative_to(ROOT / "local" / "diagnostics"):
            parser.error("隔離コードと予算記録もlocal/diagnostics内です。")
    budget_directory = args.budget_directory.resolve() if args.budget_directory else None
    return (
        run(directory, args.live, budget_directory)
        if args.worker
        else supervise(
            directory,
            args.live,
            args.source_root.resolve() if args.source_root else None,
            budget_directory,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
