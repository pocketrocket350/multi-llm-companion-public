"""保存前のVoiSona Talk接続を、固定文の短い試聴で確認する。"""

from __future__ import annotations

import base64
import json
import re
import ssl
import tempfile
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

from multi_llm_companion.environment import EnvironmentSettingsError, VoiceValues, validate_voice
from multi_llm_companion.integrations.voisona_talk import VoiSonaTalkVoiceOutput
from multi_llm_companion.integrations.wav_playback import create_wav_player
from multi_llm_companion.services.connection_test import ConnectionTestResult, _NoRedirect

PREVIEW_TEXT = "こんにちは。音声のテストです。"
_SYNTHESIS_TIMEOUT = 30.0
_PLAYBACK_TIMEOUT = 20.0


class _ProbeFailure(RuntimeError):
    """表示してよい固定の日本語だけを運ぶ。"""


class _Cancelled(Exception):
    pass


@dataclass(frozen=True)
class VoiceListResult:
    success: bool
    message: str
    voices: tuple[str, ...] = field(default=(), repr=False)


def _http_failure(status: int) -> str:
    if status == 401:
        reason = "ユーザー名またはAPI用パスワードが拒否されました。VoiSona Talk側のAPI設定と合わせてください。"
    elif status == 403:
        reason = "利用が許可されていません。REST APIの有効化とアクセス権限を確認してください。"
    elif status in {404, 405}:
        reason = "APIの接続先が見つかりません。URL・ポートとREST APIの設定を確認してください。"
    elif status in {400, 422}:
        reason = (
            "音声の要求が受け付けられません。ボイス名・言語とVoiSona Talkの状態を確認してください。"
        )
    elif status == 429:
        reason = "要求が混み合っています。VoiSona Talkの処理が終わってから再試行してください。"
    elif status >= 500:
        reason = "VoiSona Talk側でエラーが発生しています。本体の状態を確認してください。"
    elif 300 <= status < 400:
        reason = "別URLへの転送が返されたため停止しました。正式なREST APIのURLを入力してください。"
    else:
        reason = "要求が受け付けられません。REST APIの設定を確認してください。"
    return f"VoiSona Talk接続テスト失敗（HTTP {status}）: {reason}"


class _VoiceApi:
    def __init__(self, values: VoiceValues, timeout: float) -> None:
        # 一覧取得には、まだ選んでいないボイス名を要求しない。
        validate_voice(replace(values, enabled=True, voice_name="一覧取得"))
        if ":" in values.username:
            raise _ProbeFailure("API用ユーザー名に「:」は使えません。入力内容を確認してください。")
        self._authorization = base64.b64encode(
            f"{values.username}:{values.password}".encode()
        ).decode("ascii")
        self._base_url = values.base_url.rstrip("/")
        self._timeout = timeout
        self._opener = build_opener(
            HTTPHandler(debuglevel=0), HTTPSHandler(debuglevel=0), _NoRedirect()
        )

    def request(self, method, suffix, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(
            self._base_url + suffix,
            data=data,
            headers={
                "Authorization": f"Basic {self._authorization}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with self._opener.open(
                req, timeout=min(self._timeout, 3.0 if method == "DELETE" else 15.0)
            ) as response:
                if not 200 <= response.status < 300:
                    raise _ProbeFailure(_http_failure(response.status))
                if method == "DELETE":
                    return None
                raw = response.read(65537)
            if len(raw) > 65536:
                raise _ProbeFailure(
                    "VoiSona Talkの応答が大きすぎます。REST APIのURLを確認してください。"
                )
            return json.loads(raw)
        except HTTPError as exc:
            exc.close()
            raise _ProbeFailure(_http_failure(exc.code)) from None


def _network_failure(exc: TimeoutError | URLError) -> str:
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, TimeoutError):
        return (
            "VoiSona Talkから時間内に応答がありません。起動状態とREST APIの設定を確認してください。"
        )
    if isinstance(reason, ssl.SSLError):
        return "安全な接続の証明書を確認できません。URL・PCの日時・通信環境を確認してください。"
    return "VoiSona Talkへ接続できません。本体の起動・REST API・URL・ポート・通信環境を確認してください。"


def fetch_voisona_voices(values: VoiceValues, *, timeout: float, cancel: Event) -> VoiceListResult:
    """入力中の認証で一覧だけを取得し、合成・保存は行わない。"""
    try:
        api = _VoiceApi(values, timeout)
        if cancel.is_set():
            raise _Cancelled
        items = VoiSonaTalkVoiceOutput._voice_items(api.request("GET", "/voices"))
        if cancel.is_set():
            raise _Cancelled
        forbidden = (values.password, api._authorization, f"Basic {api._authorization}")
        voices = []
        for item in items:
            name = item.get("voice_name")
            if not isinstance(name, str):
                continue
            # 名前に紛れた認証情報や制御文字を画面へ反射しない。
            if any(secret and secret in name for secret in forbidden):
                continue
            if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in name):
                continue
            if name and name == name.strip() and len(name) <= 256 and name not in voices:
                voices.append(name)
        if not voices:
            return VoiceListResult(
                False,
                "表示できるボイス名がありません。VoiSona Talkのボイス導入とREST APIの設定を確認してください。",
            )
        return VoiceListResult(
            True, f"ボイスを{len(voices)}件取得しました。一覧から選んでください。", tuple(voices)
        )
    except _Cancelled:
        return VoiceListResult(
            False, "ボイス一覧の取得を中止しました。入力内容は変更していません。"
        )
    except (EnvironmentSettingsError, _ProbeFailure) as exc:
        return VoiceListResult(False, str(exc))
    except (TimeoutError, URLError) as exc:
        return VoiceListResult(False, _network_failure(exc))
    except Exception:
        return VoiceListResult(
            False, "ボイス一覧を取得できませんでした。REST APIのURLと本体の状態を確認してください。"
        )


def probe_voisona_connection(
    values: VoiceValues,
    *,
    language: str,
    timeout: float,
    playback_preroll: float,
    cancel: Event,
    report: Callable[[str], None],
) -> ConnectionTestResult:
    """会話や認証情報を保存せず、接続・合成・再生を順に確認する。"""
    player = None
    phase = "connection"
    cleanup_incomplete = False
    try:
        validate_voice(replace(values, enabled=True))
        api = _VoiceApi(values, timeout)
        request = api.request

        def check_cancelled() -> None:
            if cancel.is_set():
                raise _Cancelled

        check_cancelled()
        report("VoiSona Talkへ接続し、認証とボイス名を確認しています。")
        voices = VoiSonaTalkVoiceOutput._voice_items(request("GET", "/voices"))
        check_cancelled()
        if not voices:
            raise _ProbeFailure(
                "利用できるボイスがありません。VoiSona Talkでボイスの導入を確認してください。"
            )
        selected = next(
            (item for item in voices if item.get("voice_name") == values.voice_name), None
        )
        if selected is None:
            raise _ProbeFailure(
                "指定したボイス名が見つかりません。API上のボイス名を確認してください。別の声は使いません。"
            )
        phase = "synthesis"
        report("接続確認成功。固定文の音声を合成しています。まだ再生していません。")
        with tempfile.TemporaryDirectory(prefix="multi-llm-companion-preview-") as directory:
            wav_path = Path(directory) / "preview.wav"
            request_uuid = None
            try:
                check_cancelled()
                payload = {
                    "text": PREVIEW_TEXT,
                    "language": language,
                    "voice_name": values.voice_name,
                    "force_enqueue": True,
                    "destination": "file",
                    "output_file_path": str(wav_path.resolve()),
                    "can_overwrite_file": True,
                }
                version = str(selected.get("voice_version", "")).strip()
                if version:
                    payload["voice_version"] = version
                result = request("POST", "/speech-syntheses", payload)
                found = VoiSonaTalkVoiceOutput._extract_uuid(result)
                if not found or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", found):
                    raise _ProbeFailure(
                        "音声合成の受付番号を読み取れません。REST APIの対応状況を確認してください。"
                    )
                request_uuid = found
                deadline = time.monotonic() + _SYNTHESIS_TIMEOUT
                while True:
                    check_cancelled()
                    if time.monotonic() >= deadline:
                        raise _ProbeFailure(
                            "音声合成が時間内に完了しませんでした。VoiSona Talkの状態を確認してください。"
                        )
                    result = request("GET", f"/speech-syntheses/{request_uuid}")
                    if VoiSonaTalkVoiceOutput._is_request_failed(result):
                        raise _ProbeFailure(
                            "VoiSona Talkで音声合成に失敗しました。ボイス・言語・本体の状態を確認してください。"
                        )
                    if VoiSonaTalkVoiceOutput._is_request_completed(result):
                        break
                    cancel.wait(0.1)
                check_cancelled()
                phase = "file"
                if wav_path.stat().st_size > 4 * 1024 * 1024:
                    raise _ProbeFailure(
                        "試聴音声が大きすぎるため再生しません。VoiSona Talkの設定を確認してください。"
                    )
                wav_data = wav_path.read_bytes()
                check_cancelled()
                phase = "playback"
                player = create_wav_player(lead_in_seconds=playback_preroll)
                player.load(wav_data)
                if not 0 < player.duration_seconds <= 15.0:
                    raise _ProbeFailure(
                        "試聴音声が空か長すぎるため再生しません。VoiSona Talkの設定を確認してください。"
                    )
                check_cancelled()
                player.play()
                report("音声を受信しました。試聴を再生中です。音が聞こえることを確認してください。")
                deadline = time.monotonic() + _PLAYBACK_TIMEOUT
                while player.is_playing:
                    if time.monotonic() >= deadline:
                        raise _ProbeFailure(
                            "試聴の再生が時間内に終わらないため停止しました。Windowsの音声出力を確認してください。"
                        )
                    if cancel.wait(0.05):
                        raise _Cancelled
                check_cancelled()
            finally:
                try:
                    if player is not None:
                        player.stop()
                finally:
                    if request_uuid is not None:
                        # 自分の試聴要求だけを片付ける。失敗しても秘密値は表示しない。
                        try:
                            request("DELETE", f"/speech-syntheses/{request_uuid}")
                        except Exception:
                            cleanup_incomplete = True
            phase = "cleanup"
        if cleanup_incomplete:
            return ConnectionTestResult(
                False,
                "試聴の再生は終了しましたが、合成要求の後片付けを確認できませんでした。VoiSona Talk側の状態を確認してください。設定は保存していません。",
            )
        return ConnectionTestResult(
            True,
            "VoiSona Talkの接続確認と試聴の再生が終了しました。音が聞こえたことを確認してください。設定はまだ保存していません。",
        )
    except _Cancelled:
        return ConnectionTestResult(
            False, "VoiSona Talkの接続テスト・試聴を停止しました。設定は保存していません。"
        )
    except (EnvironmentSettingsError, _ProbeFailure) as exc:
        return ConnectionTestResult(False, str(exc))
    except (TimeoutError, URLError) as exc:
        return ConnectionTestResult(False, _network_failure(exc))
    except Exception:
        message = {
            "file": "合成した音声ファイルを読み取れません。VoiSona Talkと同じPCで実行し、一時保存先への書き込みを確認してください。",
            "playback": "音声は取得できましたが、再生できません。Windowsの音声出力機器・音量を確認してください。",
            "cleanup": "試聴用の一時音声を片付けられませんでした。保存先の権限を確認してください。",
        }.get(
            phase,
            "VoiSona Talkの応答を確認できません。REST APIのURLと本体の状態を確認してください。",
        )
        return ConnectionTestResult(False, message)
