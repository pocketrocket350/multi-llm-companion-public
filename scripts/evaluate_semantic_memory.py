"""承認された枠だけで、通常データを使わず意味による記憶を評価する。"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx

from multi_llm_companion.application import _engine_contract
from multi_llm_companion.config import MemoryConfig, load_runtime_config
from multi_llm_companion.domain import AssistantResponse
from multi_llm_companion.domain.models import WebSource
from multi_llm_companion.providers import build_provider
from multi_llm_companion.services import ChatService, DialogueDirector, MemoryService
from multi_llm_companion.services.persona import build_persona_prompt
from multi_llm_companion.storage import SQLiteConversationRepository


class SemanticEvaluationBudget:
    """最大16要求・出力1536・0.50米ドル。送信前に保守的な費用を予約する。"""

    def __init__(self, prior_attempts=0, *, request_limit=16):
        if not 1 <= request_limit <= 16 or not 0 <= prior_attempts <= request_limit:
            raise ValueError("要求上限は1〜16、既実行件数はその範囲内です。")
        self.requests = prior_attempts
        self.request_limit = request_limit
        self.failed = False
        self._lock = threading.Lock()

    def check(self, request: httpx.Request):
        if (
            str(request.url) != "https://api.openai.com/v1/chat/completions"
            or request.method != "POST"
            or len(request.content) > 80_000
        ):
            raise RuntimeError("評価で許可した送信先・要求サイズではありません。")
        data = json.loads(request.content)
        if (
            set(data) - {"model", "messages", "stream", "max_completion_tokens", "stream_options"}
            or data.get("model") != "gpt-6-luna"
            or data.get("stream") not in (True, False)
            or (data.get("stream") and data.get("stream_options") != {"include_usage": True})
            or (not data.get("stream") and "stream_options" in data)
            or not 0 < data.get("max_completion_tokens", 0) <= 1536
            or not isinstance(data.get("messages"), list)
            or not 1 <= len(data["messages"]) <= 30
            or any(
                not isinstance(item, dict)
                or set(item) != {"role", "content"}
                or item["role"] not in {"system", "user", "assistant"}
                or not isinstance(item["content"], str)
                for item in data["messages"]
            )
        ):
            raise RuntimeError("評価で許可したモデル・出力・テキスト形式ではありません。")
        # 入力を最大80000 byte+余裕4000 tokenと見積もっても1件0.03ドル以内。
        with self._lock:
            if self.failed or self.requests >= self.request_limit:
                raise RuntimeError("評価枠またはAPI失敗により追加送信を止めました。")
            self.requests += 1

    def summary(self):
        return {
            "requests": self.requests,
            "reserved_usd": self.requests * 0.03,
            "failed": self.failed,
            "max_requests": self.request_limit,
            "budget_usd": min(0.50, self.request_limit * 0.03),
        }


CASES = [
    ("hobby", "モトジムカーナの話をしよう。私は見るより自分で走る方だよ。"),
    ("routine", "休日は古いラジオを直して過ごしてる。朝食はトーストとヨーグルトにしてる。"),
    ("correction", "朝食のトーストはやめて、最近はご飯に変えたよ。ヨーグルトはそのまま。"),
    ("paraphrase", "休みの日は昔のラジオの修理を楽しんでるんだ。"),
    ("refusal", "火曜日は折り紙をしているけど、このことは覚えないでね。"),
    ("third_party", "弟は陶芸をしてるって。私はしていないよ。"),
    ("fiction", "架空の人物の設定として『私は宇宙飛行士です』という台詞を考えて。"),
    ("temporary", "今日は少し眠いなあ。"),
    ("external", "その記事の著者は毎朝ヨガをするそうだね。"),
    ("after_external", "この記事も面白いね。私は夜になるとベランダで星を眺めてるよ。"),
    ("recall", "休みの朝は何をしようかな。私に合いそうなことを一つ話そう。"),
    ("forget", "朝食について覚えていることは忘れてね。"),
    ("short_answer", "走る方だよ。"),
    ("experience", "春に初めて完走できたんだ。あれは今でも大切な思い出だよ。"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="別途利用者が承認した場合だけ指定")
    parser.add_argument(
        "--prior-attempts", type=int, default=0, help="同じ承認枠で既に試みた要求数"
    )
    args = parser.parse_args()
    if not args.live:
        print("通信なし。--live は今回の評価枠の承認後だけ指定してください。")
        return 0
    root = Path(__file__).resolve().parents[1]
    directory = root / "local/diagnostics/semantic-memory-20260925" / ("live-" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    runtime = load_runtime_config(root)
    config = replace(
        runtime.active_provider,
        name="openai",
        provider="openai_compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-6-luna",
        max_output_tokens=1536,
    )
    repo = SQLiteConversationRepository(directory / "evaluation.db")
    memory = MemoryService(repo, MemoryConfig())
    service = ChatService(
        provider=build_provider(config),
        repository=repo,
        memory_service=memory,
        system_prompt=build_persona_prompt(runtime.persona),
        engine_contract=_engine_contract(runtime),
        dialogue_director=DialogueDirector(runtime.persona),
    )
    budget = SemanticEvaluationBudget(args.prior_attempts)
    send = httpx.Client.send
    results = []

    def guarded_send(client, request, *values, **kwargs):
        budget.check(request)
        try:
            response = send(client, request, *values, **kwargs)
            if response.status_code >= 400:
                budget.failed = True
            return response
        except Exception:
            budget.failed = True
            raise

    def save():
        (directory / "result.json").write_text(
            json.dumps(
                {"budget": budget.summary(), "turns": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        with patch.object(httpx.Client, "send", guarded_send):
            for name, text in CASES:
                if name in {"external", "recall", "short_answer"}:
                    conversation = service.start_new_conversation()
                    if name == "external":
                        repo.add_message(
                            conversation.id,
                            "assistant",
                            "記事の著者は毎朝ヨガをする。",
                            response=AssistantResponse(
                                speech="記事の著者は毎朝ヨガをする。",
                                sources=[
                                    WebSource(
                                        title="検査用の架空の記事",
                                        url="https://example.com/test",
                                        retrieved_at=datetime.now(UTC),
                                    )
                                ],
                            ),
                        )
                    elif name == "short_answer":
                        repo.add_message(conversation.id, "user", "モトジムカーナの話をしよう")
                        repo.add_message(
                            conversation.id, "assistant", "見る方と走る方、どちらが楽しい？"
                        )
                before = [item.model_dump(mode="json") for item in memory.list_memories()]
                response = service.send_message(text)
                results.append(
                    {
                        "case": name,
                        "input": text,
                        "response": response.model_dump(mode="json"),
                        "before": before,
                        "after": [item.model_dump(mode="json") for item in memory.list_memories()],
                        "usage": service.last_usage.model_dump() if service.last_usage else None,
                    }
                )
                save()
                print(
                    f"{name}: saved={len(memory.list_memories())}, requests={budget.requests}",
                    flush=True,
                )
    finally:
        save()
        service.close()
    print(f"結果: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
