"""OpenAI Chat Completions互換APIの共通実装。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import httpx

from multi_llm_companion.domain import (
    AssistantResponse,
    ChatMessage,
    TokenUsage,
    parse_assistant_response,
)

from .base import ChatProvider
from .errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
    _EmptyProviderResponse,
)


class _SharedHttpClient:
    """能動要求と通常要求で接続を共有し、利用中の切断を避ける。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._client: httpx.Client | None = None
        self._users = 0
        self._close_pending = False

    @contextmanager
    def use(self, create: Callable[[], httpx.Client]) -> Iterator[httpx.Client]:
        with self._lock:
            if self._client is None:
                self._client = create()
            client = self._client
            self._users += 1
        try:
            yield client
        finally:
            with self._lock:
                self._users -= 1
                if self._users == 0 and self._close_pending:
                    self._close_idle()

    def close(self) -> None:
        with self._lock:
            self._close_pending = True
            if self._users == 0:
                self._close_idle()

    def _close_idle(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._close_pending = False


class OpenAICompatibleProvider(ChatProvider):
    """Qwen、DeepSeek、OpenAIなどの互換APIへ接続する。"""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        normalized_model = model.strip()
        if not normalized_base_url:
            raise ProviderConfigurationError("ベースURLが設定されていません。")
        if not api_key:
            raise ProviderConfigurationError("APIキーが環境変数に設定されていません。")
        if not normalized_model:
            raise ProviderConfigurationError("モデル名が設定されていません。")
        if not 0.0 <= temperature <= 2.0:
            raise ProviderConfigurationError("温度は0から2の範囲で設定してください。")
        if max_output_tokens <= 0:
            raise ProviderConfigurationError("最大出力数は1以上にしてください。")
        if timeout <= 0:
            raise ProviderConfigurationError("タイムアウトは0より大きくしてください。")

        super().__init__(name=name, model=normalized_model)
        self._base_url = normalized_base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._transport = transport
        self._shared_client = _SharedHttpClient()

    def __repr__(self) -> str:
        """デバッグ表示にもAPIキーを含めない。"""

        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"base_url={self._base_url!r}, model={self.model!r})"
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AssistantResponse:
        return self._chat_response(messages, system_prompt=system_prompt)

    @property
    def _supports_research(self) -> bool:
        endpoint = urlsplit(self._base_url)
        return (
            endpoint.scheme == "https"
            and endpoint.hostname == "api.openai.com"
            and (not self.model.lower().startswith("gpt-6") or self.model.lower() == "gpt-6-luna")
        )

    def _research_reply(self, messages, *, system_prompt=None):
        return self._chat_response(messages, system_prompt=system_prompt, research=True)

    def _stream_research_reply(self, messages, *, system_prompt=None):
        from multi_llm_companion.streaming import SpeechStreamFilter

        from ._research import (
            _REPLY_TOOL,
            _RESEARCH_RULES,
            _RESEARCH_TOOL,
            _read_research_call,
            _ResearchStream,
        )

        self._begin_request()
        payload = self._payload(
            messages, (system_prompt or "") + "\n\n" + _RESEARCH_RULES, stream=True
        )
        payload.update(
            tools=[_REPLY_TOOL, _RESEARCH_TOOL], tool_choice="required", parallel_tool_calls=False
        )
        preview = SpeechStreamFilter(require_structured=True)
        emitted = ""
        try:
            with (
                self._client() as client,
                client.stream("POST", self._chat_completions_url, json=payload) as response,
            ):
                self._raise_for_status(response)
                if "application/json" in response.headers.get("content-type", ""):
                    # JSONを返す接続先も、追加要求なしで従来通り検証する。
                    response.read()
                    data = response.json()
                    if not isinstance(data, dict) or "error" in data:
                        raise ValueError("研究応答がオブジェクトではありません")
                    self._last_usage = TokenUsage.from_api(data.get("usage"))
                else:
                    stream = _ResearchStream()
                    finished = False
                    for line in response.iter_lines():
                        if not line.strip().startswith("data:"):
                            continue
                        raw = line.strip()[5:].strip()
                        if raw == "[DONE]":
                            finished = True
                            break
                        data = json.loads(raw)
                        chunk = stream.feed(data)
                        if (usage := TokenUsage.from_api(data.get("usage"))) is not None:
                            self._last_usage = usage
                        if chunk:
                            speech = preview.feed(chunk)
                            if speech:
                                prefix = '{"speech":"' if not emitted else ""
                                emitted += speech
                                # 制御項目は流さず、元の本文だけを既存JSON抽出へ渡す。
                                yield prefix + json.dumps(speech, ensure_ascii=False)[1:-1]
                    if not finished:
                        raise ValueError("研究応答の終端が届きませんでした")
                    data = stream.completed()
            decision = _read_research_call(data)
            if decision is None:
                _validate_finish_reason(data["choices"][0].get("finish_reason"))
                raw = _extract_message_content(data)
                if not raw.strip():
                    raise _EmptyProviderResponse("LLMから回答本文が返されませんでした。")
                decision = parse_assistant_response(raw)
            elif isinstance(decision, AssistantResponse):
                raw = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            if isinstance(decision, AssistantResponse):
                # 先行した本文と最終検証結果が違う場合は、保存へ進めない。
                if emitted and emitted != decision.speech:
                    raise ValueError("逐次本文と最終回答が一致しません")
                self._last_raw_response = raw
                self._last_response = decision
                if emitted:
                    yield '"}'
            yield decision
        except httpx.TimeoutException as exc:
            raise self._connection_error("LLMの回答待ちがタイムアウトしました。") from exc
        except httpx.RequestError as exc:
            raise self._connection_error("LLMとの通信が途中で切れました。") from exc
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            error = ProviderResponseError(
                "LLMの逐次回答を検証できなかったため、保存と読み上げを止めました。"
            )
            self._last_error = str(error)
            raise error from exc
        except ProviderResponseError as exc:
            self._last_error = str(exc)
            raise

    def _chat_response(self, messages, *, system_prompt=None, research=False):
        from ._research import _REPLY_TOOL, _RESEARCH_RULES, _RESEARCH_TOOL, _read_research_call

        self._begin_request()
        if research:
            system_prompt = (system_prompt or "") + "\n\n" + _RESEARCH_RULES
        payload = self._payload(messages, system_prompt, stream=False)
        if research:
            payload.update(
                tools=[_REPLY_TOOL, _RESEARCH_TOOL],
                tool_choice="required",
                parallel_tool_calls=False,
            )
        try:
            with self._client() as client:
                response = client.post(self._chat_completions_url, json=payload)
                self._raise_for_status(response)
                data = response.json()
                if isinstance(data, dict) and "error" in data:
                    raise ValueError("APIエラーが返されました")
            self._last_usage = TokenUsage.from_api(
                data.get("usage") if isinstance(data, dict) else None
            )
            if research and (call := _read_research_call(data)) is not None:
                if isinstance(call, AssistantResponse):
                    self._last_raw_response = data["choices"][0]["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                    self._last_response = call
                return call
            raw_text = _extract_message_content(data)
            if not raw_text.strip():
                error = _EmptyProviderResponse(
                    "LLMから回答本文が返されませんでした。モデル設定を確認してください。"
                )
                self._last_error = str(error)
                raise error
            _validate_finish_reason(data["choices"][0].get("finish_reason"))
            parsed = parse_assistant_response(raw_text)
            self._last_raw_response = raw_text
            self._last_response = parsed
            return parsed
        except httpx.TimeoutException as exc:
            raise self._connection_error(
                "LLMへの接続がタイムアウトしました。設定の秒数や通信環境を確認してください。"
            ) from exc
        except httpx.RequestError as exc:
            raise self._connection_error(
                "LLMへ接続できませんでした。ベースURLと通信環境を確認してください。"
            ) from exc
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            error = ProviderResponseError(
                "LLMから読み取れない応答が返されました。モデル設定を確認してください。"
            )
            self._last_error = str(error)
            raise error from exc

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        self._begin_request()
        payload = self._payload(messages, system_prompt, stream=True)
        chunks: list[str] = []
        finished = False
        finish_reason = None
        try:
            with self._client() as client:
                active_payload = payload
                for attempt in range(2):
                    retry_without_usage_option = False
                    with client.stream(
                        "POST",
                        self._chat_completions_url,
                        json=active_payload,
                    ) as response:
                        if (
                            attempt == 0
                            and response.status_code in {400, 422}
                            and "stream_options" in active_payload
                        ):
                            retry_without_usage_option = True
                        else:
                            self._raise_for_status(response)
                            for line in response.iter_lines():
                                content, usage, finished, reason = _parse_sse_line(line)
                                if usage is not None:
                                    self._last_usage = usage
                                if finish_reason is not None and content:
                                    raise ValueError("回答終了後に本文が届きました")
                                if reason is not None:
                                    finish_reason = reason
                                    if chunks or content.strip():
                                        _validate_finish_reason(reason)
                                if content:
                                    chunks.append(content)
                                    yield content
                                if finished:
                                    break
                    if retry_without_usage_option:
                        active_payload = dict(active_payload)
                        active_payload.pop("stream_options", None)
                    else:
                        break
            if not finished:
                raise ValueError("回答の終端が届きませんでした")
            if any(chunk.strip() for chunk in chunks):
                _validate_finish_reason(finish_reason)
        except httpx.TimeoutException as exc:
            raise self._connection_error(
                "LLMの回答待ちがタイムアウトしました。設定の秒数や通信環境を確認してください。"
            ) from exc
        except httpx.RequestError as exc:
            raise self._connection_error(
                "LLMとの通信が途中で切れました。通信環境を確認して再送してください。"
            ) from exc
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            error = ProviderResponseError(
                "LLMの逐次回答を読み取れませんでした。プロバイダー設定を確認してください。"
            )
            self._last_error = str(error)
            raise error from exc

        raw_text = "".join(chunks)
        if not raw_text.strip():
            error = _EmptyProviderResponse(
                "LLMから回答本文が返されませんでした。モデル設定を確認してください。"
            )
            self._last_error = str(error)
            raise error
        self._last_raw_response = raw_text
        self._last_response = parse_assistant_response(raw_text)

    def health_check(self) -> bool:
        self._last_error = None
        try:
            with self._client() as client:
                response = client.get(self._models_url)
                if response.status_code in {404, 405}:
                    # モデル一覧を持たない互換APIでも、接続先への到達は確認できた。
                    return True
                self._raise_for_status(response)
            return True
        except (
            httpx.HTTPError,
            ProviderAuthenticationError,
            ProviderResponseError,
        ):
            if self._last_error is None:
                self._last_error = "LLMへ接続できませんでした。設定と通信環境を確認してください。"
            return False

    @property
    def _chat_completions_url(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"

    @property
    def _models_url(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            prefix = self._base_url[: -len("/chat/completions")]
            return f"{prefix}/models"
        return f"{self._base_url}/models"

    def _client(self) -> AbstractContextManager[httpx.Client]:
        return self._shared_client.use(self._new_client)

    def _close(self) -> None:
        self._shared_client.close()

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
            transport=self._transport,
            # 逐次応答を閉じても、対応先ではTLS接続を次の要求へ再利用する。
            http2=True,
            # 読み上げや利用者の返答を待つ間も接続を維持する。
            limits=httpx.Limits(keepalive_expiry=60.0),
        )

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        system_prompt: str | None,
        *,
        stream: bool,
    ) -> dict[str, object]:
        prepared = self.prepare_messages(messages, system_prompt)
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.to_api_dict() for message in prepared],
            "stream": stream,
        }
        if self._uses_openai_reasoning_parameters:
            # GPT-5/6系の出力上限を使い、既定の推論と非互換な温度指定を省く。
            payload["max_completion_tokens"] = self._max_output_tokens
            if self.model.lower() == "gpt-6-luna":
                # 短い会話の出力枠を回答本文へ使う。
                payload["reasoning_effort"] = "none"
        else:
            payload["temperature"] = self._temperature
            payload["max_tokens"] = self._max_output_tokens
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @property
    def _uses_openai_reasoning_parameters(self) -> bool:
        """OpenAI公式のGPT-5/6系へ対応するChat Completions形式を使う。"""

        endpoint = urlsplit(self._base_url)
        return (
            endpoint.scheme == "https"
            and endpoint.hostname == "api.openai.com"
            and self.model.lower().startswith(("gpt-5", "gpt-6"))
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status in {401, 403}:
            error = ProviderAuthenticationError(
                "APIキーが拒否されました。環境変数の設定を確認してください。"
            )
            self._last_error = str(error)
            raise error
        if status == 429:
            error = ProviderResponseError(
                "LLMの利用上限に達したか、混み合っています。時間を置いて再試行してください。"
            )
            self._last_error = str(error)
            raise error
        if status == 400:
            error = ProviderResponseError(_safe_bad_request_message(response))
            self._last_error = str(error)
            raise error
        if status >= 400:
            error = ProviderResponseError(
                f"LLM APIがエラーを返しました（HTTP {status}）。設定を確認してください。"
            )
            self._last_error = str(error)
            raise error

    def _connection_error(self, message: str) -> ProviderConnectionError:
        error = ProviderConnectionError(message)
        self._last_error = message
        return error


def _extract_message_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("応答がオブジェクトではありません")
    if "error" in data:
        raise ValueError("APIエラーが返されました")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choicesがありません")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choiceがオブジェクトではありません")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("messageがありません")
    if "content" in message and message["content"] is None and not message.get("refusal"):
        return ""
    return _content_to_text(message.get("content"))


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise ValueError("回答本文が文字列ではありません")


def _safe_bad_request_message(response: httpx.Response) -> str:
    """APIの詳細を画面へ出さず、対応方法だけを示す。"""

    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            if code in {"unsupported_parameter", "invalid_parameter"}:
                return (
                    "モデルが受け付けない設定が含まれています。モデル名とアプリを確認してください。"
                )
    return "LLM APIがリクエストを受け付けませんでした。モデル名と設定を確認してください。"


def _validate_finish_reason(reason: object) -> None:
    """終了理由を省略する互換APIを保ち、明示された異常終了は拒否する。"""

    if reason is not None and reason != "stop":
        raise ValueError("回答が正常に完了しませんでした")


def _parse_sse_line(line: str) -> tuple[str, TokenUsage | None, bool, object]:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return "", None, False, None
    if stripped.startswith("data:"):
        stripped = stripped[5:].strip()
    else:
        return "", None, False, None
    if stripped == "[DONE]":
        return "", None, True, None

    data: Any = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("ストリーム応答がオブジェクトではありません")
    if "error" in data:
        # 応答本文には秘密情報が混ざる可能性があるため、内容は例外へ含めない。
        raise ValueError("ストリーム内でAPIエラーが返されました")
    usage = TokenUsage.from_api(data.get("usage"))
    choices = data.get("choices")
    if not choices:
        return "", usage, False, None
    if not isinstance(choices, list) or not isinstance(choices[0], dict):
        raise ValueError("ストリームのchoicesが不正です")
    delta = choices[0].get("delta")
    reason = choices[0].get("finish_reason")
    if not isinstance(delta, dict):
        return "", usage, False, reason
    content = delta.get("content")
    if content is None:
        return "", usage, False, reason
    return _content_to_text(content), usage, False, reason
