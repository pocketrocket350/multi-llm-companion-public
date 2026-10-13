"""通常データを分離し、要求数と予算を制限して検索会話を確認する。"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from PySide6.QtWidgets import QApplication, QMessageBox

from multi_llm_companion.config import load_runtime_config
from multi_llm_companion.ui.render_quality import configure_default_surface_format
from scripts.phase9_acceptance_app import AcceptanceBackend, create_window, isolated_runtime

NOTICE = (
    "OpenAI公式APIのgpt-5.6-lunaで検索会話を確認します。\n"
    "上限: API要求8件、検索ツール3回、各出力1024トークン、予算0.50米ドル。\n"
    "失敗後の自動再試行は禁止し、検索1要求に0.10、会話1要求に0.02米ドルを予約します。\n"
    "検索へは確認した検索文と日時、会話へは人格・この隔離会話・検索結果を送ります。\n"
    "通常の会話DB・保存記憶・音声やPMX素材は外部へ送信しません。\n"
    "通常データと別の空のDB・設定で起動します。音声は現在のローカルVoiSonaです。\n"
    "この条件で検証用アプリを開きますか？"
)


class EvaluationBudget:
    """失敗要求も数え、送信前に上限を検査する検証専用ガード。"""

    def __init__(self) -> None:
        self.requests = 0
        self.searches = 0
        self.reserved_cents = 0
        self.failed = False
        self._lock = threading.Lock()

    def check(self, request: httpx.Request) -> None:
        url = request.url
        if url.host == "127.0.0.1" and url.port == 32766 and url.scheme == "http":
            return
        if (
            url.scheme != "https"
            or url.host != "api.openai.com"
            or url.port not in {None, 443}
            or url.path not in {"/v1/responses", "/v1/chat/completions"}
            or request.method != "POST"
            or len(request.content) > 40_000
        ):
            raise RuntimeError("検証で許可した送信先・要求サイズではありません。")
        data = json.loads(request.content)
        search = url.path == "/v1/responses"
        output = data.get("max_output_tokens" if search else "max_completion_tokens", 0)
        if data.get("model") != "gpt-5.6-luna" or not 0 < output <= 1024:
            raise RuntimeError("検証で許可したモデル・出力上限ではありません。")
        if search and (
            data.get("max_tool_calls") != 1
            or data.get("store") is not False
            or data.get("tools")
            != [{"type": "web_search", "search_context_size": "low", "external_web_access": True}]
        ):
            raise RuntimeError("検証の検索ツール上限を確認できません。")
        reserve = 10 if search else 2
        with self._lock:
            if (
                self.failed
                or self.requests >= 8
                or (search and self.searches >= 3)
                or self.reserved_cents + reserve > 50
            ):
                raise RuntimeError("検証上限または直前のAPI失敗により追加送信を止めました。")
            self.requests += 1
            self.searches += int(search)
            self.reserved_cents += reserve

    def summary(self) -> dict:
        return {
            "api_requests": self.requests,
            "search_requests": self.searches,
            "reserved_usd": self.reserved_cents / 100,
            "api_failure_stopped_further_requests": self.failed,
        }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication(sys.argv)
    if (
        QMessageBox.question(
            None,
            "新フェーズ10・隔離検証の送信範囲",
            NOTICE,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        != QMessageBox.StandardButton.Yes
    ):
        return 0
    directory = (
        root
        / "local/diagnostics/phase10-acceptance"
        / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
    )
    runtime = isolated_runtime(load_runtime_config(root), directory)
    os.chdir(directory)
    configure_default_surface_format(runtime.avatar.three_d.render_quality)
    budget = EvaluationBudget()
    original_send = httpx.Client.send

    def guarded_send(client, request, *args, **kwargs):
        budget.check(request)
        external = request.url.host == "api.openai.com"
        try:
            response = original_send(client, request, *args, **kwargs)
            if external and response.status_code >= 400:
                budget.failed = True
            return response
        except Exception:
            if external:
                budget.failed = True
            raise

    with patch.object(httpx.Client, "send", guarded_send):
        backend = AcceptanceBackend(runtime)
        try:
            window = create_window(backend, directory)
            window.setWindowTitle("【新フェーズ10・隔離検証】現在情報・Web検索")
            window.show()
            print(f"隔離データ保存先: {directory}", flush=True)
            return app.exec()
        finally:
            backend.close()
            (directory / "budget.json").write_text(
                json.dumps(budget.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
