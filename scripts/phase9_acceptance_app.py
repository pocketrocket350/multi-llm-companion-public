"""通常データを触らず、既存の音声・PMXでフェーズ9を実機確認する。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import yaml
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from multi_llm_companion.application import ApplicationBackend, ApplicationError
from multi_llm_companion.config import RuntimeConfig, load_runtime_config
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.providers.errors import ProviderResponseError
from multi_llm_companion.ui import chat_window
from multi_llm_companion.ui.render_quality import configure_default_surface_format

_ROOT = Path(__file__).resolve().parents[1]


def isolated_runtime(runtime: RuntimeConfig, directory: Path) -> RuntimeConfig:
    """既存形式の設定だけを新規フォルダーへ複製し、空のDBを使う。"""

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    app = runtime.app.model_copy(
        deep=True,
        update={
            "database_path": str(directory / "conversations.db"),
            "audio_config_path": str(directory / "audio.yaml"),
            "memory_config_path": str(directory / "memory.yaml"),
            "avatar_config_path": str(directory / "avatar.yaml"),
        },
    )
    # 検証側だけ空の記憶・自動承認オンとする。音声と素材の設定は変えない。
    memory = runtime.memory.model_copy(
        deep=True, update={"enabled": True, "auto_approve_candidates": True}
    )
    audio = runtime.audio.model_copy(deep=True)
    avatar = runtime.avatar.model_copy(deep=True)
    for name, config in (("app", app), ("memory", memory), ("audio", audio), ("avatar", avatar)):
        (directory / f"{name}.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return replace(
        runtime,
        app_config_path=directory / "app.yaml",
        app=app,
        database_path=directory / "conversations.db",
        memory=memory,
        audio=audio,
        avatar=avatar,
    )


class AcceptanceBackend(ApplicationBackend):
    """隔離設定を経由しない素材への書き込みを止める。"""

    def set_motion_enabled(self, motion_id: str, enabled: bool) -> None:
        raise ApplicationError("隔離検証では元のモーションカタログを変更しません。")

    def generate_confirmation_vmd(self) -> str:
        raise ApplicationError("隔離検証ではモデルの隣にVMDを書き出しません。")


class AcceptanceWindow(chat_window.ChatWindow):
    """隔離検証用のウィンドウ。"""


def create_window(backend: ApplicationBackend, directory: Path, *, replay: bool = False):
    """QSettingsも通常のWindows設定から分離する。"""

    def settings(_organization, _application):
        return QSettings(str(directory / "window.ini"), QSettings.Format.IniFormat)

    with patch.object(chat_window, "QSettings", settings):
        window = AcceptanceWindow(backend)
    mode = "記録応答の再生・API送信なし" if replay else "実API会話"
    window.setWindowTitle(f"【フェーズ9・隔離検証】{mode} - {directory.name}")
    window.statusBar().showMessage(
        "検証専用の空のDB・設定です。通常の記憶・会話・設定は変更しません。"
    )
    # 受入済みD・素材調整・接続設定は今回の確認対象から外す。
    for button in (
        window.proactive_button,
        window.motion_library_button,
        window.expression_library_button,
        window.settings_button,
    ):
        button.clicked.disconnect()
        button.hide()
    return window


def _replay_provider(path: Path) -> MockProvider:
    saved = json.loads(path.read_text(encoding="utf-8"))
    responses = {item["user"]: item["raw_response"] for item in saved["trace"]}

    def respond(messages):
        question = next(item.content for item in reversed(messages) if item.role == "user")
        if question not in responses:
            raise ProviderResponseError("記録にない入力です。再生モードは外部APIへ送信しません。")
        return responses[question]

    return MockProvider(responder=respond)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true", help="保存済み実LLM応答をAPIなしで再生")
    args = parser.parse_args()
    directory = (
        _ROOT
        / "local/diagnostics/phase9-acceptance"
        / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
    )
    runtime = isolated_runtime(load_runtime_config(_ROOT), directory)
    # 任意の診断ログも検証フォルダー内へ閉じ込める。
    os.chdir(directory)
    configure_default_surface_format(runtime.avatar.three_d.render_quality)
    app = QApplication.instance() or QApplication(sys.argv)
    if args.replay:
        provider = _replay_provider(
            _ROOT / "local/diagnostics/phase9-readiness-20260922/after-result.json"
        )
        with patch("multi_llm_companion.application.build_provider", return_value=provider):
            backend = AcceptanceBackend(runtime)
    else:
        backend = AcceptanceBackend(runtime)
    try:
        window = create_window(backend, directory, replay=args.replay)
        window.show()
        print(f"隔離データ保存先: {directory}", flush=True)
        return app.exec()
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
