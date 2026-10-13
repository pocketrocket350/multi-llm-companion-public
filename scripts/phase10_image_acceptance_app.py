"""画像共有の隔離確認。起動時に送信範囲と新しい予算を確認する。"""

from __future__ import annotations

import base64
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
    "OpenAI公式のgpt-5.6-lunaへ選んだ静止画像と質問を送り、通常の会話と合わせて返答します。\n"
    "上限: 合計6要求、画像解析2回、検索1回、各出力1024トークン、予算0.50米ドル。\n"
    "検索1要求は0.10米ドル、他は0.02米ドルを予約し、API失敗後は追加送信を止めます。\n"
    "解析へ渡す画像はプレビューどおりの縮小JPEGです。原本の付加情報とパスは送りません。\n"
    "返答には人格・この隔離会話・解析結果を使います。通常の会話・記憶・音声・PMXは送信しません。\n"
    "空の隔離DB・設定・QSettingsを作り、画像原本は保存しません。\n"
    "この条件で有料APIを使う確認アプリを開きますか？"
)


class ImageEvaluationBudget:
    def __init__(self) -> None:
        self.requests = 0
        self.images = 0
        self.searches = 0
        self.reserved_cents = 0
        self.failed = False
        self._lock = threading.Lock()

    def check(self, request: httpx.Request) -> str:
        url = request.url
        if url.scheme == "http" and url.host == "127.0.0.1" and url.port == 32766:
            return "local-voice"
        if (
            url.scheme != "https"
            or url.host != "api.openai.com"
            or url.port not in {None, 443}
            or url.path not in {"/v1/responses", "/v1/chat/completions"}
            or request.method != "POST"
            or len(request.content) > 2_800_000
        ):
            raise RuntimeError("画像確認で許可した送信先・要求サイズではありません。")
        data = json.loads(request.content)
        responses = url.path.endswith("responses")
        output = data.get("max_output_tokens" if responses else "max_completion_tokens")
        if data.get("model") != "gpt-5.6-luna" or type(output) is not int or not 0 < output <= 1024:
            raise RuntimeError("画像確認で許可したモデル・出力上限ではありません。")
        kind = "chat"
        if responses:
            if data.get("store") is not False:
                raise RuntimeError("解析の保存無効を確認できません。")
            if data.get("tools"):
                if data.get("max_tool_calls") != 1 or data["tools"] != [
                    {
                        "type": "web_search",
                        "search_context_size": "low",
                        "external_web_access": True,
                    }
                ]:
                    raise RuntimeError("検索上限を確認できません。")
                kind = "search"
            else:
                inputs = data.get("input")
                if (
                    not isinstance(inputs, list)
                    or len(inputs) != 1
                    or inputs[0].get("role") != "user"
                ):
                    raise RuntimeError("画像の送信範囲を確認できません。")
                parts = inputs[0].get("content", [])
                if (
                    len(parts) != 2
                    or parts[0].get("type") != "input_text"
                    or not isinstance(parts[0].get("text"), str)
                    or not 0 < len(parts[0]["text"]) <= 4000
                    or parts[1].get("type") != "input_image"
                    or parts[1].get("detail") != "high"
                    or not parts[1].get("image_url", "").startswith("data:image/jpeg;base64,")
                ):
                    raise RuntimeError("画像1枚と質問以外の送信を止めました。")
                raw = base64.b64decode(parts[1]["image_url"].split(",", 1)[1], validate=True)
                if not raw.startswith(b"\xff\xd8") or not 0 < len(raw) <= 2_000_000:
                    raise RuntimeError("画像の容量・形式を確認できません。")
                kind = "image"
        if kind != "image" and len(request.content) > 40_000:
            raise RuntimeError("会話・検索の送信サイズが上限を超えています。")
        reserve = 10 if kind == "search" else 2
        with self._lock:
            if (
                self.failed
                or self.requests >= 6
                or (kind == "image" and self.images >= 2)
                or (kind == "search" and self.searches >= 1)
                or self.reserved_cents + reserve > 50
            ):
                raise RuntimeError("確認上限または直前のAPI失敗により追加送信を止めました。")
            self.requests += 1
            self.images += int(kind == "image")
            self.searches += int(kind == "search")
            self.reserved_cents += reserve
        return kind

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "images": self.images,
            "searches": self.searches,
            "reserved_usd": self.reserved_cents / 100,
            "failed": self.failed,
        }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication(sys.argv)
    if (
        QMessageBox.question(
            None,
            "新フェーズ10・画像確認の送信範囲",
            NOTICE,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        != QMessageBox.StandardButton.Yes
    ):
        return 0
    directory = (
        root
        / "local/diagnostics/phase10-image-acceptance"
        / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6])
    )
    runtime = isolated_runtime(load_runtime_config(root), directory)
    os.chdir(directory)
    configure_default_surface_format(runtime.avatar.three_d.render_quality)
    budget = ImageEvaluationBudget()
    original = httpx.Client.send

    def guarded_send(client, request, *args, **kwargs):
        kind = budget.check(request)
        try:
            response = original(client, request, *args, **kwargs)
            if kind != "local-voice" and response.status_code >= 400:
                budget.failed = True
            return response
        except Exception:
            if kind != "local-voice":
                budget.failed = True
            raise

    with patch.object(httpx.Client, "send", guarded_send):
        backend = AcceptanceBackend(runtime)
        try:
            window = create_window(backend, directory)
            window.setWindowTitle("【新フェーズ10・隔離確認】画像共有・ニュース")
            window.show()
            return app.exec()
        finally:
            backend.close()
            (directory / "budget.json").write_text(
                json.dumps(budget.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    raise SystemExit(main())
