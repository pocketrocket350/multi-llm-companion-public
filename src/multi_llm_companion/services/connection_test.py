"""設定や会話を保存せず、固定文だけでLLM接続を確認する。"""

from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from multi_llm_companion.config import ResolvedProviderConfig
from multi_llm_companion.domain import ChatMessage
from multi_llm_companion.environment import EnvironmentSettingsError, validate_connection_url
from multi_llm_companion.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _extract_message_content,
)


@dataclass(frozen=True)
class ConnectionTestResult:
    success: bool
    message: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 別のURLへ認証情報を転送しない。
        return None


def _http_failure(status: int) -> ConnectionTestResult:
    if status == 401:
        reason = "APIキーが拒否されました。キーをコピーし直し、有効期限を確認してください。"
    elif status == 403:
        reason = "利用が許可されていません。APIキーの権限とモデルの利用権限を確認してください。"
    elif status == 404:
        reason = "接続先URLまたはモデル名が見つかりません。提供元の案内と照合してください。"
    elif status == 405:
        reason = "このURLは会話用の送信に対応していません。APIの接続先URLを確認してください。"
    elif status in {400, 422}:
        reason = "要求が受け付けられません。モデル名と、そのモデルで使える設定を確認してください。"
    elif status == 429:
        reason = (
            "利用上限に達したか、混み合っています。残高・利用枠を確認し、時間を置いてください。"
        )
    elif status >= 500:
        reason = "接続先のサービスでエラーが発生しています。時間を置いて再試行してください。"
    elif 300 <= status < 400:
        reason = "別のURLへの転送が返されました。転送せず停止しました。正式なAPIのURLを入力してください。"
    else:
        reason = "接続先が要求を受け付けませんでした。URL・モデル名・利用条件を確認してください。"
    return ConnectionTestResult(False, f"接続テスト失敗（HTTP {status}）: {reason}")


def probe_llm_connection(config: ResolvedProviderConfig) -> ConnectionTestResult:
    """一時的な接続で指定モデルの応答を確認し、詳細を表示しない。"""
    if config.provider != "openai_compatible":
        return ConnectionTestResult(False, "この接続先は実接続テストの対象外です。")
    try:
        validate_connection_url(config.base_url, "接続先URL")
        if not config.api_key.strip() or not config.model.strip():
            return ConnectionTestResult(False, "APIキーとモデル名を入力してください。")
        if any(not 32 <= ord(char) <= 126 for char in config.api_key):
            return ConnectionTestResult(
                False, "APIキーの文字を確認してください。改行を入れずにコピーし直してください。"
            )
        provider = OpenAICompatibleProvider(
            name="connection-test",
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=min(config.max_output_tokens, 1024),
            timeout=min(config.timeout, 15.0),
        )
        payload = provider._payload(
            [ChatMessage(role="user", content="Reply with OK.")], None, stream=False
        )
        request = Request(
            provider._chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # 専用の標準HTTP接続は、ヘッダーや例外詳細をデバッグログへ流さない。
        opener = build_opener(HTTPHandler(debuglevel=0), HTTPSHandler(debuglevel=0), _NoRedirect())
        with opener.open(request, timeout=min(config.timeout, 15.0)) as response:
            if not 200 <= response.status < 300:
                return _http_failure(response.status)
            data = response.read(65537)
        if len(data) > 65536:
            return ConnectionTestResult(
                False, "接続先の応答が大きすぎます。会話用APIのURLを確認してください。"
            )
        if not _extract_message_content(json.loads(data)).strip():
            raise ValueError
        return ConnectionTestResult(
            True,
            "接続テスト成功: 指定したモデルから確認用の応答を受信しました。設定はまだ保存していません。",
        )
    except HTTPError as exc:
        exc.close()
        return _http_failure(exc.code)
    except EnvironmentSettingsError as exc:
        return ConnectionTestResult(False, str(exc))
    except (TimeoutError, URLError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(reason, TimeoutError):
            message = "時間内に応答がありませんでした。接続先の状態を確認し、再試行してください。"
        elif isinstance(reason, ssl.SSLError):
            message = (
                "安全な接続の証明書を確認できません。URL・PCの日時・通信環境を確認してください。"
            )
        else:
            message = "接続先へ接続できません。URL・インターネット接続・プロキシの設定を確認してください。"
        return ConnectionTestResult(False, message)
    except (ValueError, KeyError, TypeError):
        return ConnectionTestResult(
            False,
            "会話用APIの応答を読み取れませんでした。URL・モデル名・APIの互換性を確認してください。",
        )
    except Exception:
        return ConnectionTestResult(
            False, "接続テストを完了できませんでした。入力内容と通信環境を確認してください。"
        )
