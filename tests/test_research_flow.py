"""検索判断のHTTP応答を再生し、分野を限定しない会話→実行の接続を検査する。"""

import json

import httpx
import pytest
from test_web_search import client_for, search_payload

from multi_llm_companion.providers import OpenAICompatibleProvider
from multi_llm_companion.services import ChatService
from multi_llm_companion.services.web_search import WebSearchSession
from multi_llm_companion.storage import SQLiteConversationRepository


def reply(speech):
    return {
        "choices": [
            {"message": {"content": json.dumps({"speech": speech})}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def call(query):
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixture",
                            "type": "function",
                            "function": {
                                "name": "request_web_search",
                                "arguments": json.dumps({"query": query, "detailed": False}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def tool_reply(speech):
    data = call("unused")
    data["choices"][0]["message"]["tool_calls"][0]["function"] = {
        "name": "reply_to_user",
        "arguments": json.dumps(
            {"speech": speech, "dialogue_act": "ask", "current_topic": "製品の確認"}
        ),
    }
    return data


def test_explicit_conversation_action_keeps_normal_response_contract(flow):
    service, http, searches, answers = flow
    answers.append(tool_reply("どの製品か教えてね。"))
    chunks = list(service.stream_message("この製品を確かめたい"))
    assert http[0]["tool_choice"] == "required"
    assert {tool["function"]["name"] for tool in http[0]["tools"]} == {
        "reply_to_user",
        "request_web_search",
    }
    assert not searches
    assert service.last_response.speech == "どの製品か教えてね。"
    assert service.last_response.current_topic == "製品の確認"
    assert "reply_to_user" not in "".join(chunks)
    assert service.provider.last_response == service.last_response


def test_legacy_explicit_prepared_search_still_runs_without_decision_request(flow):
    service, http, searches, answers = flow
    text = "前に指定した条件でお願い"
    service.web_search.approve(text, "製品A 公表仕様", detailed=False)
    answers.append(reply("仕様を確認したよ。[S1]"))
    list(service.stream_message(text))
    assert len(searches) == len(http) == 1
    assert "tools" not in http[0]
    assert service.web_search._pending is None


@pytest.fixture
def flow():
    http, searches, answers = [], [], []

    def chat(request):
        data = json.loads(request.content)
        http.append(data)
        answer = answers.pop(0)
        if callable(answer):
            answer = answer(data)
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)

    provider = OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key="fixture",
        model="gpt-6-luna",
        transport=httpx.MockTransport(chat),
    )
    service = ChatService(provider=provider, repository=SQLiteConversationRepository(":memory:"))

    def search(request):
        searches.append(json.loads(request.content))
        return httpx.Response(200, json=search_payload())

    service.web_search = WebSearchSession(client_for(search))
    yield service, http, searches, answers
    service.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "question,clarification,condition,query",
    [
        (
            "降水量を例年と比べて",
            "どの地域と期間を比べたい？",
            "千代田区の今年9月28日までで",
            "東京 千代田区 2026年9月1〜28日 降水量 同期間の平年値 比較",
        ),
        (
            "このバイクの重さをメーカーの資料で確かめたい",
            "車種と年式を教えてね。",
            "CB125Rの2026年モデル",
            "ホンダ CB125R 2026年モデル メーカー公表 車両重量",
        ),
        (
            "博物館って今度の休み開いてる？",
            "どの博物館で、何日に行く予定？",
            "東京国立博物館、10月12日",
            "東京国立博物館 2026年10月12日 開館日 公式",
        ),
        ("あの人物の経歴を確かめたい", "誰のことかな？", "南方熊楠です", "南方熊楠 経歴 一次資料"),
        (
            "この設定の変え方、公式の説明を見たい",
            "どのアプリの設定かな？",
            "Windows11の表示倍率",
            "Windows 11 表示倍率 変更 公式説明",
        ),
    ],
)
def test_context_reply_executes_one_search_and_finishes_without_extra_input(
    flow, stream, question, clarification, condition, query
):
    service, http, searches, answers = flow
    answers.extend([reply(clarification), call(query), reply("資料ではこう説明されているよ。[S1]")])

    def send(text):
        return list(service.stream_message(text)) if stream else service.send_message(text)

    send(question)
    assert not searches
    send(condition)
    assert len(searches) == 1 and len(http) == 3
    assert json.loads(searches[0]["input"])["検索文"] == query
    assert [m["content"] for m in http[1]["messages"][-3:]] == [question, clarification, condition]
    assert "tools" in http[0] and "tools" in http[1] and "tools" not in http[2]
    assert http[1]["parallel_tool_calls"] is False
    assert http[1]["reasoning_effort"] == "none"
    assert service.last_usage.total_tokens == 12 + 52 + 8
    assert service.last_response.sources
    saved = service.get_current_messages()
    assert len(saved) == 4
    assert "request_web_search" not in saved[-1].response.model_dump_json()


@pytest.mark.parametrize(
    "text",
    [
        "やっぱり調べなくていいよ",
        "今日は疲れたな",
        "『検索して』を英語にして",
        "前の資料の意味を教えて",
    ],
)
def test_regular_reply_does_not_invent_background_search(flow, text):
    service, http, searches, answers = flow
    answers.append(reply("うん、話を聞くよ。"))
    list(service.stream_message(text))
    assert len(http) == 1 and not searches
    assert service.last_usage.total_tokens == 8
    assert service.last_response.speech == "うん、話を聞くよ。"


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("off", "オフ"),
        ("unsupported", "接続先やモデル"),
        ("secret", "そのまま検索へ送れない"),
        ("long", "短く教えて"),
    ],
)
def test_blocked_search_is_a_spoken_response_without_external_search(flow, failure, reason):
    service, http, searches, answers = flow
    query = "製品仕様の確認"
    if failure == "off":
        service.set_web_search_enabled(False)
    if failure == "unsupported":
        service.web_search.client = None
    if failure == "secret":
        query = "api_key=secret 製品仕様"
    if failure == "long":
        query = "長" * 401
    answers.append(call(query))
    list(service.stream_message("それでおねがい"))
    assert len(http) == 1 and not searches
    assert reason in service.last_response.speech
    assert service.last_usage.total_tokens == 12
    assert not service.last_response.memory_candidates


@pytest.mark.parametrize(
    "status,reason", [(401, "拒否"), (429, "上限"), (500, "応答を処理"), (400, "接続先やモデル")]
)
def test_search_http_failure_becomes_dialogue_then_regular_chat(flow, status, reason):
    service, http, _, answers = flow
    answers.extend([call("製品の仕様"), reply("うん、また話そう。")])
    requests = []
    service.web_search.client = client_for(lambda r: requests.append(r) or httpx.Response(status))
    list(service.stream_message("この仕様、公式情報だとどうなってる？"))
    assert len(requests) == 1 and len(http) == 1
    assert reason in service.last_response.speech
    assert service.last_usage.total_tokens == 12
    list(service.stream_message("ありがとう"))
    assert len(requests) == 1 and len(http) == 2
    assert service.last_response.speech == "うん、また話そう。"


@pytest.mark.parametrize("stage", ["decision", "search", "final"])
@pytest.mark.parametrize("cancel_kind", ["stop", "disable", "callback"])
def test_cancel_at_each_stage_never_applies_old_answer(flow, stage, cancel_kind):
    service, http, _, answers = flow
    cancelled = [False]

    def cancel():
        if cancel_kind == "disable":
            service.set_web_search_enabled(False)
        elif cancel_kind == "stop":
            service._invalidate_active_stream()
        else:
            cancelled[0] = True

    def decision(_):
        if stage == "decision":
            cancel()
        return call("製品仕様")

    def search(_):
        if stage == "search":
            cancel()
        return httpx.Response(200, json=search_payload())

    def final(_):
        cancel()
        return reply("古い返事。[S1]")

    answers.extend([decision, final])
    service.web_search.client = client_for(search)
    chunks = list(
        service.stream_message("メーカーの情報を確認したい", cancelled=lambda: cancelled[0])
    )
    assert chunks == [] and service._stream_was_invalidated
    assert service.last_response is None
    assert len(service.get_current_messages()) == 1
    assert len(http) == (2 if stage == "final" else 1)


@pytest.mark.parametrize(
    "bad",
    ["multiple", "truncated", "unknown", "invalid_json", "invalid_function", "invalid_choice"],
)
def test_invalid_tool_request_never_runs_or_leaks(flow, bad):
    service, http, searches, answers = flow
    data = call("機密の内部制御語")
    choice = data["choices"][0]
    calls = choice["message"]["tool_calls"]
    if bad == "multiple":
        calls.append(calls[0].copy())
    if bad == "truncated":
        choice["finish_reason"] = "length"
    if bad == "unknown":
        calls[0]["function"]["name"] = "delete_files"
    if bad == "invalid_json":
        calls[0]["function"]["arguments"] = "{"
    if bad == "invalid_function":
        calls[0]["function"] = None
    if bad == "invalid_choice":
        data["choices"] = []
    answers.append(data)
    list(service.stream_message("これを確かめたい"))
    assert not searches and len(http) == 1
    assert "機密" not in service.last_response.speech
    assert "検索は始められていない" in service.last_response.speech
