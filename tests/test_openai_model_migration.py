"""モデル切替を実バックエンド・画面と通信なしのHTTP応答で確認する。"""

import json
from dataclasses import replace

import httpx
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
from test_application import _runtime
from test_image_sharing import analysis_payload
from test_image_sharing import picture as picture
from test_research_flow import call as research_call
from test_web_search import search_payload
from test_web_search_ui import wait_until

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.config import ProvidersConfig
from multi_llm_companion.ui.chat_window import ChatWindow, QFileDialog


@pytest.mark.parametrize(
    "model,base_url,provider_kind,available",
    [
        ("gpt-5.6-luna", "https://api.openai.com/v1", "openai_compatible", True),
        ("gpt-6-luna", "https://api.openai.com/v1/", "openai_compatible", True),
        ("configured-model", "https://api.openai.com/v1", "openai_compatible", True),
        ("gpt-6-luna", "https://example.invalid/v1", "openai_compatible", False),
        ("gpt-6-luna", "https://api.openai.com.example.invalid/v1", "openai_compatible", False),
        ("gpt-6-luna", "http://api.openai.com/v1", "openai_compatible", False),
        ("gpt-6-luna", "https://api.openai.com/v1", "mock", False),
    ],
)
def test_external_clients_follow_configured_model_and_preserve_permission(
    tmp_path, monkeypatch, model, base_url, provider_kind, available
):
    def no_request(*_args, **_kwargs):
        pytest.fail("接続の設定だけで外部通信してはいけません")

    monkeypatch.setattr(httpx.Client, "send", no_request)
    runtime = _runtime(tmp_path)
    backend = ApplicationBackend(runtime)
    try:
        backend.set_web_search_enabled(False)
        backend.set_image_sharing_enabled(False)
        backend._configure_web_search(
            replace(runtime.active_provider, model=model, base_url=base_url, provider=provider_kind)
        )
        assert backend.web_search_available() is available
        assert backend.image_sharing_available() is available
        assert not backend.web_search_enabled()
        assert not backend.image_sharing_enabled()
        if available:
            assert backend._service.web_search.client.model == model
            assert backend._service.image_sharing.client.model == model
        backend._configure_web_search(runtime.active_provider)
        assert not backend.web_search_available()
        assert not backend.image_sharing_available()
        assert not backend.web_search_enabled()
        assert not backend.image_sharing_enabled()
    finally:
        backend.close()


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-6-luna"])
@pytest.mark.parametrize(
    "search_text", ["今日の科学ニュースを一つ調べて", "明日の千代田区の天気は？"]
)
def test_window_uses_configured_model_for_chat_search_and_image(
    tmp_path, monkeypatch, picture, model, search_text
):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        lambda *_: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_: pytest.fail("追加の確認ダイアログは不要です")
    )
    requests = []
    replies = iter(["こんにちは。", "記事を見つけたよ。[S1]", "赤い四角が見えるね。"])

    def handle(request):
        assert request.url.host == "api.openai.com"
        payload = json.loads(request.content)
        assert payload["model"] == model
        requests.append((request.url.path, payload))
        if request.url.path == "/v1/responses":
            assert payload["store"] is False
            assert payload["max_output_tokens"] == 1024
            return httpx.Response(
                200, json=search_payload() if "tools" in payload else analysis_payload()
            )
        assert request.url.path == "/v1/chat/completions"
        assert "temperature" not in payload and "max_tokens" not in payload
        assert payload["max_completion_tokens"] == 128
        if "tools" in payload and payload["messages"][-1]["content"] == search_text:
            return httpx.Response(200, json=research_call(search_text))
        raw = json.dumps({"speech": next(replies), "emotion": "neutral", "gesture": "none"})
        if not payload["stream"]:
            return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})
        event = json.dumps({"choices": [{"delta": {"content": raw}}]})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {event}\n\ndata: [DONE]\n\n",
        )

    # 通常アプリと同じ生成経路で、HTTP送信だけをモックへ差し替える。
    client_type = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client_type(**{**kwargs, "transport": httpx.MockTransport(handle)}),
    )
    runtime = _runtime(tmp_path)
    runtime = replace(
        runtime,
        app=runtime.app.model_copy(update={"active_provider": "openai"}),
        providers=ProvidersConfig(
            providers={
                "openai": runtime.providers.providers["mock"].model_copy(
                    update={"provider": "openai_compatible"}
                )
            }
        ),
        active_provider=replace(
            runtime.active_provider,
            name="openai",
            provider="openai_compatible",
            base_url="https://api.openai.com/v1",
            api_key="test-only",
            model=model,
        ),
    )
    backend = ApplicationBackend(runtime)
    window = ChatWindow(backend)
    try:
        window.show()
        app.processEvents()
        assert backend.status().model_name == model
        assert backend.web_search_available() and backend.image_sharing_available()
        for prompt in ["こんにちは", search_text]:
            window.input_edit.setPlainText(prompt)
            window.send_button.click()
            wait_until(lambda: window._thread is None and not window._busy)
            assert not window.error_label.text(), window.error_label.text()
        assert 'href="https://news.example/article"' in window.chat_view.toHtml()
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(picture), ""))
        window.attach_image_button.click()
        assert window._attached_image is not None
        window.input_edit.setPlainText("この画像は何色？")
        window.send_button.click()
        wait_until(lambda: window._thread is None and not window._busy)
        assert not window.error_label.text(), window.error_label.text()
        assert "赤い四角が見えるね。" in window.chat_view.toPlainText()
        assert [path for path, _payload in requests] == [
            "/v1/chat/completions",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/chat/completions",
        ]
        messages = backend.load_messages(window._current_conversation_id)
        assert len(messages) == 6
        assert messages[3].sources[0].url == "https://news.example/article"
        assert messages[-1].image_attachment.name == "fixture.png"
        assert backend.list_memories() == []
    finally:
        window.close()
        backend.close()
