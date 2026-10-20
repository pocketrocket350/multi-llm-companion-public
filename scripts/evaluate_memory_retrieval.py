"""人工記憶だけを使い、現在値と離れた表現の候補を限定評価する。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import MemoryCategory, MemoryStatus
from multi_llm_companion.providers import OpenAICompatibleProvider
from multi_llm_companion.services.chat import ChatService
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from scripts.evaluate_memory_continuity import MODEL, ROOT, URL, ApiBudget, api_key
from scripts.measure_companion_wait import protected, save, stats


def cases(correction_repeats=0):
    corrections = [
        ("朝食はトースト。", "朝食はご飯。", "訂正後の朝食は何だった？", "ご飯"),
        ("呼び名はミナ。", "呼び名はナツ。", "今の呼び名は？", "ナツ"),
        ("趣味は釣り。", "趣味は彫金。", "変更後の趣味を教えて。", "彫金"),
        ("仕事は営業。", "仕事は設計。", "今の仕事は何だった？", "設計"),
        ("朝食はトースト。", "朝食はご飯。", "訂正した今の朝食を覚えている？", "ご飯"),
        ("呼び名はミナ。", "呼び名はナツ。", "訂正した呼び方で呼んで。", "ナツ"),
        ("朝食はトースト。", "朝食はご飯。", "今の朝食と、それに変えた日を教えて。", "ご飯"),
        ("趣味は釣り。", "趣味は彫金。", "今の趣味と、それに変えた理由を教えて。", "彫金"),
    ]
    result = [
        dict(kind="correction", old=old, fact=fact, query=query, tokens=[token])
        for old, fact, query, token in corrections
    ]
    paraphrases = [
        ("休日の楽しみは年代物のラジオを修理すること。", "余暇の過ごし方は？", ["ラジオ"]),
        ("休日の楽しみは年代物のラジオを修理すること。", "暇なときの過ごし方は？", ["ラジオ"]),
        ("趣味はガラス細工。", "空き時間に楽しんでいることは？", ["ガラス"]),
        ("週末には野鳥の写真を撮るのが楽しみ。", "息抜きに何をするんだっけ？", ["野鳥"]),
        ("朝食はおかゆを食べている。", "起きて最初の食事は何だった？", ["おかゆ"]),
        ("仕事は図書館の司書。", "私の職業を覚えてる？", ["司書"]),
        ("住まいは松本。", "私の居住地はどこだった？", ["松本"]),
        ("休日は盆栽の手入れが楽しみ。", "自由な時間の好きな過ごし方は？", ["盆栽"]),
    ]
    result += [
        dict(kind="paraphrase", fact=fact, query=query, tokens=tokens)
        for fact, query, tokens in paraphrases
    ]
    result += [
        dict(
            kind=kind,
            fact="休日の楽しみはオーボエの練習。",
            query="私の余暇の過ごし方を覚えている？なければ分からないと言って。",
            tokens=[],
            forbidden=["オーボエ"],
        )
        for kind in ("absent", "deleted", "pending", "disabled")
    ]
    result += [dict(result[2], correction_repeat=index + 1) for index in range(correction_repeats)]
    return result


def run(directory, budget_directory, live=False, correction_repeats=0):
    directory.mkdir(parents=True, exist_ok=True)
    before = protected()
    save(directory / "protected-before.json", before)
    budget = ApiBudget(budget_directory)
    transport = httpx.HTTPTransport(retries=0) if live else None
    rows = []
    error = None
    evaluation_cases = cases(correction_repeats)
    try:
        for number, case in enumerate(evaluation_cases, 1):
            save(directory / "case-state.json", dict(active=True, started=time.monotonic()))
            repo = SQLiteConversationRepository(directory / f"case-{number:02}.db")
            target = repo.create_memory(
                case.get("old", case["fact"]),
                status=MemoryStatus.CANDIDATE
                if case["kind"] == "pending"
                else MemoryStatus.APPROVED,
            )
            for index in range(120):
                repo.create_memory(
                    f"工作記録{index:03}では紙の箱の寸法を測定した。",
                    status=MemoryStatus.APPROVED,
                )
            memory = MemoryService(repo, MemoryConfig())
            if case["kind"] == "correction":
                memory.update(target.id, case["fact"], MemoryCategory.OTHER)
            elif case["kind"] in {"absent", "deleted"}:
                memory.delete(target.id)
            elif case["kind"] == "disabled":
                memory.set_enabled(False)
            observations = []

            def send(request, observations=observations):
                observations.append(json.loads(request.content))
                budget.check(request)
                try:
                    response = transport.handle_request(request)
                    budget.failed = response.status_code >= 400
                    return response
                except Exception:
                    budget.failed = True
                    raise
                finally:
                    save(budget.directory / "api-budget.json", budget.summary())

            provider = OpenAICompatibleProvider(
                name="openai",
                base_url=URL,
                api_key=api_key() if live else "artificial-only",
                model=MODEL,
                max_output_tokens=1536,
                timeout=25,
                transport=httpx.MockTransport(send) if live else None,
            )
            service = ChatService(
                provider=provider,
                repository=repo,
                memory_service=memory,
                system_prompt="人工案内役。日本語で短く自然に返答する。情報がなければ正直に伝える。",
            )
            try:
                service.start_new_conversation("人工記憶の限定評価")
                timings = []
                for _ in range(20):
                    start = time.perf_counter_ns()
                    turn = memory.prepare_turn(case["query"])
                    timings.append((time.perf_counter_ns() - start) / 1e6)
                response = service.send_message(case["query"]) if live else None
                selected = memory.last_context_memories
                row = dict(
                    number=number,
                    case=case,
                    expected_id=target.id,
                    selected_ids=[item.id for item in selected],
                    selected_count=len(selected),
                    selected_characters=sum(len(item.content) for item in selected),
                    target_retrieved=target.id in {item.id for item in selected},
                    selection_ms=stats(timings),
                    speech=response.speech if response else None,
                    tokens_ok=all(token in response.speech for token in case["tokens"])
                    if response
                    else None,
                    forbidden_reused=[
                        token
                        for token in case.get("forbidden", [])
                        if response and token in response.speech
                    ],
                    prompt=turn.prompt,
                    requests=observations,
                    usage=provider.get_usage().model_dump() if provider.get_usage() else None,
                    supports_research=provider._supports_research,
                    uses_research_flow=service._uses_research_flow,
                )
                rows.append(row)
                save(directory / f"case-{number:02}.json", row)
                save(directory / "results.json", rows)
                if budget.failed:
                    raise RuntimeError("API failure stopped subsequent requests")
                print(f"case {number}/{len(evaluation_cases)} recorded", flush=True)
            finally:
                service.close()
                repo.close()
    except Exception as exc:
        error = type(exc).__name__
    finally:
        if transport:
            transport.close()
        save(
            directory / "summary.json",
            dict(
                completed=len(rows),
                missing=len(evaluation_cases) - len(rows),
                error=error,
                normal_files_unchanged=before == protected(),
                budget=budget.summary(),
                paraphrase_retrieved=sum(
                    row["target_retrieved"] for row in rows if row["case"]["kind"] == "paraphrase"
                ),
                forbidden_reuse=sum(bool(row["forbidden_reused"]) for row in rows),
                answer_tokens_passed=sum(bool(row["tokens_ok"]) for row in rows) if live else None,
                full_answer_quality="requires parent review; tokens alone are not quality acceptance",
            ),
        )
        save(directory / "case-state.json", dict(active=False))
    return 1 if error else 0


def supervise(args):
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=False)
    save(
        directory / "plan.json",
        dict(
            cases=cases(args.correction_repeats),
            live=args.live,
            case_limit_seconds=30,
            series_limit_seconds=900,
            model=MODEL,
            send_scope="artificial persona and memories only; no external search or audio",
        ),
    )
    command = [sys.executable, "-m", "scripts.evaluate_memory_retrieval"]
    if args.source_root:
        command = [
            sys.executable,
            "-c",
            "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('scripts.evaluate_memory_retrieval',run_name='__main__')",
            str(args.source_root),
        ]
    command += [
        "--worker",
        "--directory",
        str(directory),
        "--budget-directory",
        str(args.budget_directory),
        "--correction-repeats",
        str(args.correction_repeats),
    ]
    if args.live:
        command.append("--live")
    start = time.monotonic()
    reason = None
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        save(directory / "supervisor-start.json", dict(pid=process.pid, deadline=start + 900))
        while process.poll() is None:
            try:
                state = json.loads((directory / "case-state.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if (directory / "STOP").exists() or time.monotonic() - start > 900:
                reason = "stop or series deadline"
            if state.get("active") and time.monotonic() - state["started"] > 30:
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
        dict(
            exit_code=process.returncode,
            reason=reason,
            elapsed_seconds=time.monotonic() - start,
            reaped=process.poll() is not None,
        ),
    )
    return process.returncode or (1 if reason else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--budget-directory", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--correction-repeats", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.correction_repeats <= 30:
        parser.error("訂正ケースの追加反復は0〜30回です。")
    for name in ("directory", "budget_directory", "source_root"):
        value = getattr(args, name)
        if value is not None:
            value = value.resolve()
            if not value.is_relative_to(ROOT / "local/diagnostics"):
                parser.error("評価の保存先と隔離コードはlocal/diagnostics内です。")
            setattr(args, name, value)
    return (
        run(args.directory, args.budget_directory, args.live, args.correction_repeats)
        if args.worker
        else supervise(args)
    )


if __name__ == "__main__":
    raise SystemExit(main())
