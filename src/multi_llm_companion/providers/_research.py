"""会話中の検索呼出し。保存する回答形式には含めない。"""

import json
from dataclasses import dataclass

from multi_llm_companion.domain import (
    AssistantResponse,
    parse_assistant_response,
    root_speech_bounds,
)

from .errors import ProviderResponseError


@dataclass(frozen=True)
class _ResearchCall:
    query: str
    detailed: bool = False


_REPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "reply_to_user",
        "description": (
            "通常の会話、必要な確認質問、できないことの説明を利用者へ返す。"
            "調べ物の対象を特定できない場合もこちらを選んで質問する。"
            "まだ検索を実行していないので、調べた結果や検索中という予告は返さない。"
        ),
        "parameters": AssistantResponse.model_json_schema(),
        "strict": False,
    },
}


_RESEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "request_web_search",
        "description": (
            "会話から対象を特定できた調べ物だけを実行する。分野を限定しない。"
            "対象を識別する情報が会話にないときは呼ばず、まず利用者へ質問する。"
            "未特定の対象そのものを検索しても解決できない。"
            "依頼、続きの条件回答、訂正、検索提案への同意を会話の意味で判断する。"
            "調べると約束するだけの返事の代わりに、この関数を呼ぶ。"
            "対象が特定できないときだけ、必要な質問を通常の返事で行う。"
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "利用者が指定した対象・目的・条件・期間・比較相手を含む400文字以内の検索文。"
                        "省略された条件は同じ話題の会話から引き継ぐ。話題変更や取消を尊重する。"
                        "人格指示、保存記憶、秘密情報を含めない。場所や時期を勝手に補わない。"
                    ),
                },
                "detailed": {"type": "boolean", "description": "詳しい説明を依頼されたか"},
            },
            "required": ["query", "detailed"],
            "additionalProperties": False,
        },
    },
}

_RESEARCH_RULES = """
【今回の会話と調べ物の実行】
通常の会話・確認質問・できないことの説明はreply_to_userを呼び、引数をいつもの回答JSONにする。調べ物を実行するときはrequest_web_searchを呼ぶ。どちらか一つを選び、分野や特定の言葉ではなく、利用者の意図と会話の流れで判断する。
最初に、会話に出た情報だけで検索対象を特定できるかを判断する。対象が未特定なら検索より先に必要な質問を返す。検索は足りない対象情報を利用者に代わって決める手段ではない。
検索文を単独で読んだ第三者が、何について調べるか分かる必要がある。まだ何を指すか分からない指示語・対象名の省略をそのまま検索文へ渡さない。会話から明確に解決できる省略は補う。
広い分野の依頼と、対象が分からない依頼を区別する。利用者が広く知りたい場合は不要に絞り込む質問をせず、その範囲で調べる。
利用者が情報を調べる・確認する・最新の情報で比較することを求め、必要な対象が分かるなら、request_web_searchをこの返答で呼ぶ。
「調べるね」「比べるよ」「結果が届いていない」と発言するだけでは検索は始まらない。実行の代わりに予告を返してはいけない。
必要な対象・条件だけが不明なら、一つの自然な質問を返す。既に会話で指定された内容は聞き直さず、条件が揃ったら追加の「お願い」や実行確認を求めず呼び出す。
短い条件回答、訂正、直前の提案への同意も、元の依頼の目的や比較条件を含めて解釈する。新しい話題、断り、引用、雑談から勝手に調べ物を再開しない。
検索が不要な知識説明や既に得た資料の説明は普通に返す。検索済みと装わず、検索しないまま後で結果を届けると約束しない。
公開Web情報の調査だけを扱う。ログイン、購入、ファイル操作、動画・文書・カメラ取得などは実行できない。求められた場合は、できない範囲と理由を自分の口調で伝える。
検索許可がオフ・未対応でも、調べ物の依頼を受けたら同じ関数でアプリへ渡せる。アプリは検索を実行せず、実際の理由を会話へ返す。
関数とは別に回答本文を返さない。request_web_searchの引数は発言・記憶ではない。
""".strip()


def _read_research_call(data: dict) -> _ResearchCall | AssistantResponse | None:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("choices"), list)
        or not data["choices"]
    ):
        raise ProviderResponseError("会話の応答を読み取れませんでした。")
    choice = data["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ProviderResponseError("会話の応答を読み取れませんでした。")
    calls = choice["message"].get("tool_calls")
    if not calls:
        if choice.get("finish_reason") == "length":
            raise ProviderResponseError("返事の生成が上限に達しました。")
        return None
    if (
        not isinstance(calls, list)
        or len(calls) != 1
        or not isinstance(calls[0], dict)
        or choice.get("finish_reason") != "tool_calls"
    ):
        raise ProviderResponseError("検索の実行指示を確定できませんでした。")
    call = calls[0]
    function = call.get("function", {})
    if not isinstance(function, dict):
        raise ProviderResponseError("会話の実行指示を読み取れませんでした。")
    if call.get("type") == "function" and function.get("name") == "reply_to_user":
        args = json.loads(function["arguments"])
        if (
            not isinstance(args, dict)
            or not isinstance(args.get("speech"), str)
            or not args["speech"].strip()
        ):
            raise ProviderResponseError("返事の本文を読み取れませんでした。")
        return parse_assistant_response(function["arguments"])
    if call.get("type") != "function" or function.get("name") != "request_web_search":
        raise ProviderResponseError("実行を許可していない指示が返されました。")
    args = json.loads(function["arguments"])
    if (
        not isinstance(args, dict)
        or set(args) != {"query", "detailed"}
        or not isinstance(args["query"], str)
        or not args["query"].strip()
        or type(args["detailed"]) is not bool
    ):
        raise ProviderResponseError("検索の対象を読み取れませんでした。")
    return _ResearchCall(args["query"].strip(), args["detailed"])


class _ResearchStream:
    """内部操作を再構成し、通常返答の引数だけを逐次渡す。"""

    def __init__(self):
        self._call = None
        self._content = []
        self._finish_reason = None
        self._emitted = 0

    def feed(self, data):
        if not isinstance(data, dict) or "error" in data:
            raise ValueError("研究応答のストリームが不正です")
        choices = data.get("choices")
        if choices == []:
            return ""
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("研究応答の選択肢を確認できません")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ValueError("研究応答の選択肢が不正です")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValueError("研究応答の断片が不正です")
        if self._finish_reason is not None and delta:
            raise ValueError("終了後に研究応答の断片が届きました")
        if (reason := choice.get("finish_reason")) is not None:
            if self._finish_reason is not None:
                raise ValueError("研究応答の終端が重複しています")
            self._finish_reason = reason
        calls, content = delta.get("tool_calls"), delta.get("content")
        if content:
            if calls or self._call is not None or not isinstance(content, str):
                raise ValueError("内部操作と回答本文が混在しています")
            # 操作のない互換応答は、検証完了まで保持する。
            self._content.append(content)
        if not calls:
            return ""
        if self._content or not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("研究応答の内部操作は一つだけです")
        call = calls[0]
        if not isinstance(call, dict) or type(call.get("index")) is not int or call["index"] != 0:
            raise ValueError("研究応答の内部操作番号が不正です")
        if self._call is None:
            self._call = {"id": None, "type": None, "function": {"name": "", "arguments": ""}}
        for key in ("id", "type"):
            value = call.get(key)
            if value is not None:
                if not isinstance(value, str) or (self._call[key] not in {None, value}):
                    raise ValueError("研究応答の内部操作が途中で変わりました")
                self._call[key] = value
        if self._call["type"] not in {None, "function"}:
            raise ValueError("未許可の内部操作です")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError("研究応答の関数を確認できません")
        current = self._call["function"]
        if (name := function.get("name")) is not None:
            if not isinstance(name, str) or not name:
                raise ValueError("研究応答の関数名が不正です")
            if name != current["name"]:
                candidate = current["name"] + name
                if not any(
                    allowed.startswith(candidate)
                    for allowed in ("reply_to_user", "request_web_search")
                ):
                    raise ValueError("未許可の関数名です")
                current["name"] = candidate
        arguments = function.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise ValueError("研究応答の引数が不正です")
            current["arguments"] += arguments
        if self._call["type"] != "function" or current["name"] != "reply_to_user":
            return ""
        # speechが確認できるまで保持し、誤った通常返答に付いた検索引数も流さない。
        if self._emitted == 0 and root_speech_bounds(current["arguments"]) is None:
            return ""
        chunk = current["arguments"][self._emitted :]
        self._emitted = len(current["arguments"])
        return chunk

    def completed(self):
        message = {"content": "".join(self._content)}
        if self._call is not None:
            message["tool_calls"] = [self._call]
        return {"choices": [{"message": message, "finish_reason": self._finish_reason}]}
