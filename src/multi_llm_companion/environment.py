"""画面入力の接続情報を、Windows利用者ごとに保護して保存する。"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


class EnvironmentSettingsError(RuntimeError):
    """秘密値を含めずに表示できる環境設定エラー。"""


@dataclass(frozen=True, repr=False)
class ConnectionValues:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass(frozen=True, repr=False)
class VoiceValues:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:32766/api/talk/v1"
    username: str = ""
    password: str = ""
    voice_name: str = ""


@dataclass(frozen=True, repr=False)
class EnvironmentSettings:
    use_environment: bool = True
    providers: dict[str, ConnectionValues] = field(default_factory=dict)
    voice: VoiceValues = field(default_factory=VoiceValues)


def validate_voice(values: VoiceValues) -> None:
    if not values.enabled:
        return
    missing = [
        label
        for label, value in (
            ("接続先URL", values.base_url),
            ("ユーザー名", values.username),
            ("パスワード", values.password),
            ("ボイス名", values.voice_name),
        )
        if not value.strip()
    ]
    if missing:
        raise EnvironmentSettingsError(
            "VoiSona Talkの" + "、".join(missing) + "を入力してください。"
        )
    validate_connection_url(values.base_url, "VoiSona Talkの接続先URL")


def validate_connection_url(value: str, label: str) -> None:
    """URL内への認証情報の埋め込みも防ぐ。"""
    try:
        url = urlsplit(value)
        valid = (
            url.scheme in {"http", "https"}
            and bool(url.hostname)
            and url.port != 0
            and not url.username
            and not url.password
            and not url.query
            and not url.fragment
            and not any(char.isspace() for char in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise EnvironmentSettingsError(
            f"{label}には http:// または https:// で始まる接続先を入力してください。"
            "ユーザー名・パスワード・?以降の文字列は含めないでください。"
        )


class _DataBlob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _protect(data: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise EnvironmentSettingsError("画面入力の保存にはWindowsが必要です。")
    buffer = ctypes.create_string_buffer(data)
    source = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    function.restype = ctypes.c_int
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    try:
        # 利用者単位の保護を使い、Windowsの確認画面は要求しない。
        if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
            raise EnvironmentSettingsError("Windowsによる接続情報の保護・読み出しに失敗しました。")
        return ctypes.string_at(output.data, output.size)
    finally:
        ctypes.memset(buffer, 0, len(buffer))
        if output.data:
            ctypes.memset(output.data, 0, output.size)
            kernel32.LocalFree(output.data)


class EnvironmentStore:
    """暗号化済みの一時ファイルだけを使い、方式と値を一括保存する。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> EnvironmentSettings:
        try:
            encrypted = self.path.read_bytes()
        except FileNotFoundError:
            return EnvironmentSettings()
        except OSError:
            raise EnvironmentSettingsError(
                "環境設定を読み出せません。保存先の権限を確認してください。"
            ) from None
        try:
            raw = json.loads(_protect(encrypted, decrypt=True))
            if not isinstance(raw, dict) or set(raw) != {"use_environment", "providers", "voice"}:
                raise ValueError
            if type(raw["use_environment"]) is not bool or not isinstance(raw["providers"], dict):
                raise ValueError
            providers = {}
            for name, values in raw["providers"].items():
                if (
                    not isinstance(values, dict)
                    or set(values) != {"base_url", "api_key", "model"}
                    or not all(isinstance(value, str) for value in values.values())
                ):
                    raise ValueError
                providers[name] = ConnectionValues(**values)
            voice = raw["voice"]
            if not isinstance(voice, dict) or set(voice) != {
                "enabled",
                "base_url",
                "username",
                "password",
                "voice_name",
            }:
                raise ValueError
            if type(voice["enabled"]) is not bool or not all(
                isinstance(value, str) for key, value in voice.items() if key != "enabled"
            ):
                raise ValueError
            return EnvironmentSettings(raw["use_environment"], providers, VoiceValues(**voice))
        except (EnvironmentSettingsError, OSError, ValueError, TypeError, KeyError):
            raise EnvironmentSettingsError(
                "保存した環境設定を読み出せません。同じWindows利用者で開いてください。"
                "直らない場合は、この画面で再入力して保存するか、環境変数方式へ戻してください。"
            ) from None

    def save(self, settings: EnvironmentSettings) -> None:
        temporary: Path | None = None
        try:
            encrypted = _protect(json.dumps(asdict(settings), ensure_ascii=False).encode("utf-8"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, EnvironmentSettingsError):
            raise EnvironmentSettingsError(
                "環境設定を保存できませんでした。変更は反映されていません。"
                "Windowsの保護機能と保存先の空き容量・権限を確認してください。"
            ) from None
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)


def environment_store(project_root: Path) -> EnvironmentStore:
    # コピーや復元先へ認証情報を持ち越さないよう、アプリの場所ごとに分ける。
    identity = os.path.normcase(str(project_root.resolve())).encode("utf-8")
    folder = hashlib.sha256(identity).hexdigest()
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise EnvironmentSettingsError("Windowsの個人用保存先が見つかりません。")
    return EnvironmentStore(
        Path(local_app_data) / "multi-llm-companion" / "connections" / folder / "settings.dpapi"
    )
