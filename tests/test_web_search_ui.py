import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import httpx
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from test_application import _runtime
from test_research_flow import call as research_call
from test_research_flow import reply as research_reply
from test_web_search import client_for, search_payload

from multi_llm_companion.application import ApplicationBackend
from multi_llm_companion.providers import MockProvider, OpenAICompatibleProvider
from multi_llm_companion.services.web_search import WebSearchSession
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.models import UiMessage
from multi_llm_companion.ui.workers import StreamWorker


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        QApplication.processEvents()
        QTest.qWait(10)
        assert time.monotonic() < deadline, "Qt処理が期限内に完了しませんでした"


@pytest.mark.parametrize("result_kind", ["source", "unavailable", "failure"])
def test_semantic_research_runs_without_ui_word_matching_and_speaks_final(
    screen, monkeypatch, result_kind
):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    spoken, requests = [], []
    final = (
        "確かめられたのはこの部分だよ。[S1]"
        if result_kind == "source"
        else "探してみたけれど、確かな資料は見つからなかったよ。"
    )
    replies = [
        research_reply("どの製品のことかな？"),
        research_call("CB125R 2026年モデル メーカー公表 車両重量"),
        research_reply(final),
    ]

    def chat(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    backend._service.set_provider(
        OpenAICompatibleProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="fixture",
            model="gpt-6-luna",
            transport=httpx.MockTransport(chat),
        )
    )
    monkeypatch.setattr(window, "_web_query", lambda _: pytest.fail("旧UIの言葉判定を通っている"))

    def search(_):
        started.set()
        assert release.wait(5)
        if result_kind == "failure":
            return httpx.Response(500)
        data = search_payload()
        if result_kind == "unavailable":
            data["output"][1]["content"][0]["annotations"] = []
        return httpx.Response(200, json=data)

    backend._service.web_search.client = client_for(search)
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window.input_edit.setPlainText("メーカーの仕様ではどうなっているのかな？")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "どの製品" in window.chat_view.toPlainText() and not started.is_set()
    spoken.clear()
    window.input_edit.setPlainText("CB125Rの2026年式")
    window.send_button.click()
    try:
        wait_until(started.is_set)
        wait_until(lambda: "Webで調べています" in window.web_activity_label.text())
        assert window.web_cancel_button.isVisible()
        assert not spoken
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    text = window.chat_view.toPlainText()
    expected = "応答を処理できなく" if result_kind == "failure" else final.split("[S1]")[0]
    assert expected in text and expected in "".join(spoken)
    assert window.error_label.isHidden()
    assert not window.web_activity_label.isVisible()
    assert "request_web_search" not in text
    assert all("[S1]" not in part and "メーカー公表" not in part for part in spoken)
    assert len(requests) == (2 if result_kind == "failure" else 3)
    assert ('href="https://news.example/article"' in window.chat_view.toHtml()) == (
        result_kind == "source"
    )


@pytest.mark.parametrize("action", ["stop", "draft", "permission", "microphone"])
def test_semantic_research_cancel_does_not_deliver_late_speech(screen, action):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    calls, spoken = [], []

    def chat(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=research_call("南方熊楠 経歴 一次資料"))

    backend._service.set_provider(
        OpenAICompatibleProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="fixture",
            model="gpt-6-luna",
            transport=httpx.MockTransport(chat),
        )
    )

    def search(_):
        started.set()
        assert release.wait(5)
        return httpx.Response(200, json=search_payload())

    backend._service.web_search.client = client_for(search)
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window.input_edit.setPlainText("南方熊楠の経歴を確かめたい")
    window.send_button.click()
    try:
        wait_until(started.is_set)
        if action == "stop":
            window.web_cancel_button.click()
        elif action == "draft":
            window.input_edit.setPlainText("やっぱり別の話をしよう")
        elif action == "microphone":
            window._toggle_microphone()
        else:
            window._toggle_web_permission(False)
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not spoken and len(calls) == 1
    assert not [m for m in backend._service.get_current_messages() if m.role.value == "assistant"]


@pytest.fixture
def screen(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        "multi_llm_companion.ui.chat_window.QSettings",
        lambda *_: QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat),
    )
    backend = ApplicationBackend(_runtime(tmp_path))
    backend._service.set_provider(MockProvider(response_text='{"speech":"晴れの記事だね。[S1]"}'))
    backend._service.web_search = WebSearchSession(
        client_for(lambda _: httpx.Response(200, json=search_payload()))
    )
    window = ChatWindow(backend)

    def unexpected_dialog(*_args, **_kwargs):
        pytest.fail("検索に追加の確認ダイアログは不要です")

    monkeypatch.setattr(QDialog, "exec", unexpected_dialog)
    monkeypatch.setattr(QMessageBox, "question", unexpected_dialog)
    window.show()
    yield window, backend
    wait_until(lambda: window._thread is None and not window._busy)
    window.close()
    backend.close()
    app.processEvents()


def test_real_window_shows_linked_sources_and_history_but_does_not_speak_labels(screen):
    window, backend = screen
    spoken = []
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window.input_edit.setPlainText("今日のニュースを一つ調べて、一緒に話そう")
    window.send_button.click()
    assert window._web_turn_active
    assert not window.web_cancel_button.isHidden()
    wait_until(lambda: window._thread is None and not window._busy)
    rendered = window.chat_view.toHtml()
    assert 'href="https://news.example/article"' in rendered
    assert "公開日時: 不明" in window.chat_view.toPlainText()
    assert window.web_activity_label.isHidden()
    assert spoken and all("[S1]" not in text for text in spoken)
    messages = backend.load_messages(window._current_conversation_id)
    assert messages[-1].sources[0].title == "天気の記事"
    assert backend.list_memories() == []
    assert not window._settings.contains("web_search/enabled")


@pytest.mark.parametrize(
    "previous,proposal,reply",
    [
        (
            "観測史上のデータを調べて",
            "千代田区（東京）の2026年9月1〜28日の降水量を過去の同期間と比べて調べればいい？",
            "それでお願い",
        ),
        (
            "お願い",
            "まだ検索結果が届いていないから順位は出せないよ。"
            "調べる条件は「東京の2026年9月1〜28日の降水量を、過去の同期間と比較」で合ってる。",
            "お願い",
        ),
    ],
)
def test_search_acknowledgement_is_replaced_by_result_without_another_input(
    screen, previous, proposal, reply
):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    requests, spoken, prompts = [], [], []

    def delayed(request):
        requests.append(json.loads(request.content))
        started.set()
        assert release.wait(5)
        return httpx.Response(200, json=search_payload())

    def respond(messages):
        prompts.append(messages[0].content)
        return '{"speech":"観測値の比較結果だよ。[S1]"}'

    backend._service.set_provider(MockProvider(responder=respond))
    backend._service.web_search.client = client_for(delayed)
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window._messages = [
        UiMessage("user", previous),
        UiMessage("assistant", proposal),
    ]
    window.input_edit.setPlainText(reply)
    window.send_button.click()
    try:
        wait_until(started.is_set)
        assert "調べているよ" in window.chat_view.toPlainText()
        assert not window.web_cancel_button.isHidden()
        assert not spoken
        assert not [
            m for m in backend._service.get_current_messages() if m.role.value == "assistant"
        ]
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(requests) == len(prompts) == 1
    assert "東京" in json.loads(requests[0]["input"])["検索文"]
    assert "観測値の比較結果だよ。" in window.chat_view.toPlainText()
    assert "調べているよ" not in window.chat_view.toPlainText()
    assert 'href="https://news.example/article"' in window.chat_view.toHtml()
    assert spoken and all("調べているよ" not in text and "[S1]" not in text for text in spoken)
    saved = backend.load_messages(window._current_conversation_id)
    assert [m.content for m in saved if m.role == "assistant"] == ["観測値の比較結果だよ。[S1]"]
    assert "検索は今回分が完了" in prompts[0]
    assert window._web_query("お願い") == "お願い"


def test_non_search_turn_does_not_claim_a_background_job_is_pending(screen):
    window, backend = screen
    prompts = []
    backend._service.set_provider(
        MockProvider(
            responder=lambda messages: (
                prompts.append(messages[0].content) or '{"speech":"お話ししよう。"}'
            )
        )
    )
    window.input_edit.setPlainText("こんにちは")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "バックグラウンドで検索を続けていない" in prompts[0]


@pytest.mark.parametrize("when", ["before_run", "at_backend_entry"])
def test_cancel_before_search_worker_starts_does_not_send_or_save(screen, monkeypatch, when):
    _window, backend = screen
    backend._service.web_search.client = client_for(lambda _: pytest.fail("取消後の送信禁止"))
    conversation = backend.create_conversation("開始前取消")
    backend.approve_web_search("ニュースを調べて", "ニュースを調べて", False, "")
    worker = StreamWorker(backend, conversation, "ニュースを調べて", 1)
    cancelled, succeeded, failed = [], [], []
    worker.cancelled.connect(cancelled.append)
    worker.succeeded.connect(lambda *_: succeeded.append(True))
    worker.failed.connect(lambda *_: failed.append(True))
    if when == "before_run":
        worker.cancel()
    else:
        original = backend.stream_message

        def enter(*args, **kwargs):
            worker.cancel()
            backend._invalidate_active_stream()
            return original(*args, **kwargs)

        monkeypatch.setattr(backend, "stream_message", enter)
    worker.run()
    assert cancelled == [1] and not succeeded and not failed
    assert backend.load_messages(conversation) == []
    assert backend._service.web_search._pending is None


@pytest.mark.parametrize("action", ["cancel", "disable", "input", "microphone", "stop"])
@pytest.mark.parametrize("has_evidence", [True, False])
def test_window_discards_late_search_after_each_user_boundary(screen, action, has_evidence):
    window, backend = screen
    started, release = threading.Event(), threading.Event()
    generated, spoken = [], []

    def delayed(_request):
        started.set()
        assert release.wait(5)
        payload = search_payload()
        if not has_evidence:
            payload["output"][1]["content"][0]["annotations"] = []
        return httpx.Response(200, json=payload)

    def respond(_messages):
        generated.append(True)
        return '{"speech":"遅着返答[S1]"}'

    backend._service.set_provider(MockProvider(responder=respond))
    backend._service.web_search.client = client_for(delayed)
    backend._service.voice_output.start_stream = spoken.append
    window.input_edit.setPlainText("ニュースを調べて")
    window.send_button.click()
    try:
        wait_until(started.is_set)
        assert "調べているよ" in window.chat_view.toPlainText()
        if action == "cancel":
            window.web_cancel_button.click()
        elif action == "disable":
            window._toggle_web_permission(False)
        elif action == "input":
            assert not window.input_edit.isReadOnly()
            QTest.keyClicks(window.input_edit, "new topic")
            assert window.input_edit.toPlainText() == "new topic"
        elif action == "microphone":
            window._toggle_microphone()
        else:
            window._stop_voice_output()
    finally:
        release.set()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not generated and not spoken
    assert "遅着返答" not in window.chat_view.toPlainText()
    assert "調べているよ" not in window.chat_view.toPlainText()
    assert not [
        message
        for message in backend.load_messages(window._current_conversation_id)
        if message.role == "assistant"
    ]
    assert backend.list_memories() == []


def test_weather_asks_region_in_chat_then_searches_without_a_special_form(screen):
    window, backend = screen
    requests, prompts = [], []
    backend._service.web_search.client = client_for(
        lambda request: (
            requests.append(json.loads(request.content))
            or httpx.Response(200, json=search_payload())
        )
    )
    backend._service.set_provider(
        MockProvider(
            responder=lambda messages: (
                prompts.append(messages[0].content)
                or (
                    '{"speech":"どこの天気を調べようか？"}'
                    if not requests
                    else '{"speech":"予報を確認したよ。[S1]"}'
                )
            )
        )
    )
    assert backend.web_search_enabled()
    window.input_edit.setPlainText("今日の天気")
    assert not hasattr(window, "search_notice")
    assert not hasattr(window, "search_region_edit")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not requests
    assert not window.error_label.text()
    assert "普段の口調の短い会話" in prompts[0]
    assert "どこの天気を調べようか？" in window.chat_view.toPlainText()
    window.input_edit.setPlainText("東京都千代田区だよ")
    window.send_button.click()
    assert window._web_turn_active
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(requests) == 1
    assert json.loads(requests[0]["input"])["検索文"] == "今日の東京都千代田区の天気を調べて"
    assert 'href="https://news.example/article"' in window.chat_view.toHtml()
    assert not window.error_label.text()
    assert backend.list_memories() == []


def test_weather_question_and_followup_use_search_with_no_extra_dialog(screen):
    window, backend = screen
    requests = []
    backend._service.web_search.client = client_for(
        lambda request: (
            requests.append(json.loads(request.content))
            or httpx.Response(200, json=search_payload())
        ),
        model="gpt-6-luna",
    )
    for text in ["明日の千代田区の天気は？", "ネットで確認できないの？"]:
        window.input_edit.setPlainText(text)
        assert not hasattr(window, "search_notice")
        assert not hasattr(window, "search_region_edit")
        window.send_button.click()
        assert window._web_turn_active
        wait_until(lambda: window._thread is None and not window._busy)
        assert not window.error_label.text()
        assert 'href="https://news.example/article"' in window.chat_view.toHtml()
    assert len(requests) == 2
    assert all(p["model"] == "gpt-6-luna" for p in requests)
    assert json.loads(requests[0]["input"])["検索文"] == "明日の千代田区の天気は？"
    assert json.loads(requests[1]["input"])["検索文"] == "明日の千代田区の天気を調べて"
    assert backend.list_memories() == []


@pytest.mark.parametrize("next_step", ["other_topic", "new_conversation", "disable", "proactive"])
def test_weather_clarification_respects_topic_changes_and_permission(screen, next_step):
    window, backend = screen
    backend._service.web_search.client = client_for(lambda _: pytest.fail("検索は禁止"))
    backend._service.set_provider(MockProvider(response_text='{"speech":"どこの天気が知りたい？"}'))
    window.input_edit.setPlainText("今日の天気")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    if next_step == "other_topic":
        window.input_edit.setPlainText("紅茶を飲もう")
        window.send_button.click()
        wait_until(lambda: window._thread is None and not window._busy)
    elif next_step == "new_conversation":
        window._start_new_conversation()
    elif next_step == "proactive":
        window._messages.append(UiMessage(role="assistant", content="一息つこうか。"))
    else:
        window._toggle_web_permission(False)
    window.input_edit.setPlainText("千代田区だよ")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert not window._web_turn_active
    if next_step == "disable":
        assert "オフ" in window.error_label.text()
    else:
        assert not window.error_label.text()


def test_automatic_preference_memory_appears_in_ui_after_a_search(screen):
    window, backend = screen
    window.input_edit.setPlainText("ニュースを調べて")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    backend._service.set_provider(MockProvider(response_text='{"speech":"知ってるよ。"}'))
    window.input_edit.setPlainText(
        "私はモトジムカーナが大好きです。\nモトジムカーナって知っていますか？"
    )
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    memories = backend.list_memories()
    assert len(memories) == 1
    assert memories[0].content == "ユーザーはモトジムカーナが大好き"


def test_natural_hobby_after_weather_appears_in_memory_manager(screen):
    from multi_llm_companion.ui.memory_dialog import MemoryDialog

    window, backend = screen
    for index, text in enumerate(
        [
            "今日の東京都千代田区の天気を教えて",
            "モトジムカーナって知っていますか？",
            "うん。モトジムカーナが趣味です",
        ]
    ):
        if index == 1:
            backend._service.set_provider(MockProvider(response_text='{"speech":"うん。"}'))
        window.input_edit.setPlainText(text)
        window.send_button.click()
        wait_until(lambda: window._thread is None and not window._busy)
        assert not window.error_label.text()
    memories = backend.list_memories()
    assert [m.content for m in memories] == ["ユーザーの趣味はモトジムカーナ"]
    dialog = MemoryDialog(backend, window)
    dialog.show()
    QApplication.processEvents()
    assert "モトジムカーナ" in dialog.memory_list.item(0).text()
    dialog.close()
    dialog.deleteLater()


def test_news_request_from_screenshot_searches_and_displays_sources(screen):
    window, backend = screen
    requests = []
    backend._service.web_search.client = client_for(
        lambda request: (
            requests.append(json.loads(request.content))
            or httpx.Response(200, json=search_payload())
        ),
        model="gpt-6-luna",
    )
    text = "今日のニュースを一つ教えて。\n感想も付け加えて"
    window.input_edit.setPlainText(text)
    assert not hasattr(window, "search_notice")
    assert not hasattr(window, "search_region_edit")
    window.send_button.click()
    assert window._web_turn_active
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(requests) == 1
    assert json.loads(requests[0]["input"])["検索文"] == text
    assert 'href="https://news.example/article"' in window.chat_view.toHtml()
    assert not window.error_label.text()
    assert backend.list_memories() == []


def test_weather_kanji_polite_request_from_screenshot_starts_search(screen):
    window, backend = screen
    requests = []
    backend._service.web_search.client = client_for(
        lambda request: (
            requests.append(json.loads(request.content))
            or httpx.Response(200, json=search_payload())
        ),
        model="gpt-6-luna",
    )
    text = "今日の東京都千代田区の天気を教えて下さい"
    window.input_edit.setPlainText(text)
    assert not hasattr(window, "search_notice")
    assert not hasattr(window, "search_region_edit")
    window.send_button.click()
    assert window._web_turn_active
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(requests) == 1
    assert requests[0]["model"] == "gpt-6-luna"
    assert json.loads(requests[0]["input"])["検索文"] == text
    assert 'href="https://news.example/article"' in window.chat_view.toHtml()
    assert not window.error_label.text()
    assert backend.list_memories() == []


def test_topicless_network_request_does_not_send_or_guess(screen):
    window, backend = screen
    backend._service.web_search.client = client_for(lambda _: pytest.fail("送信禁止"))
    window.input_edit.setPlainText("ネットで確認できないの？")
    window.send_button.click()
    assert "検索したい内容" in window.error_label.text()
    assert window._thread is None


def test_weather_small_talk_does_not_search_or_request_region(screen):
    window, backend = screen
    backend._service.web_search.client = client_for(lambda _: pytest.fail("送信禁止"))
    backend._service.set_provider(MockProvider(response_text='{"speech":"気持ちがいいね。"}'))
    window.input_edit.setPlainText("今日はいい天気")
    assert not hasattr(window, "search_notice")
    assert not hasattr(window, "search_region_edit")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "気持ちがいいね。" in window.chat_view.toPlainText()


def test_capability_question_stays_in_ordinary_chat_and_knows_search_is_available(screen):
    window, backend = screen
    prompts = []
    backend._service.web_search.client = client_for(lambda _: pytest.fail("送信禁止"))
    backend._service.set_provider(
        MockProvider(
            responder=lambda messages: (
                prompts.append(messages[0].content) or '{"speech":"検索できるよ。"}'
            )
        )
    )
    window.input_edit.setPlainText("ウェブ検索はできるはずだけど？")
    assert not hasattr(window, "search_notice")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "検索経路が利用できる" in prompts[0]
    assert "Web検索は未実施" in prompts[0]
    assert "検索できるよ。" in window.chat_view.toPlainText()


@pytest.mark.parametrize("detailed", [False, True])
def test_natural_detail_request_without_dialog_or_detail_checkbox(screen, detailed):
    window, backend = screen
    requests = []
    prompts = []
    backend._service.set_provider(
        MockProvider(
            responder=lambda messages: (
                prompts.append(messages[0].content) or '{"speech":"一緒に考えてみよう。[S1]"}'
            )
        )
    )
    backend._service.web_search.client = client_for(
        lambda request: (
            requests.append(json.loads(request.content))
            or httpx.Response(200, json=search_payload())
        )
    )
    assert not hasattr(window, "web_detailed_action")
    assert "OpenAI" in window.web_permission_checkbox.toolTip()
    text = "ニュースを調べて" + ("詳しく説明して" if detailed else "")
    window.input_edit.setPlainText(text)
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(requests) == 1
    assert text in requests[0]["input"]
    assert ("理由まで詳しく" in prompts[0]) == bool(detailed)
    assert ("2〜3文" in prompts[0]) != bool(detailed)


def test_permission_settings_and_ordinary_chat_never_search_on_their_own(screen):
    window, backend = screen
    requests = []
    backend._service.web_search.client = client_for(lambda request: requests.append(request))
    window._open_options()
    assert window.web_permission_checkbox.isVisible()
    assert not hasattr(window, "conversation_menu_button")
    assert backend.web_search_enabled()
    window.input_edit.setPlainText("こんにちは")
    assert not hasattr(window, "search_notice")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    window.web_permission_checkbox.click()
    assert not backend.web_search_enabled()
    window.input_edit.setPlainText("ニュースを調べて")
    window.send_button.click()
    assert "オフ" in window.error_label.text()
    assert window._thread is None
    window.web_permission_checkbox.click()
    assert backend.web_search_enabled()
    assert not requests


@pytest.mark.parametrize("text", ["api_key=secret を調べて", "調べて" + "あ" * 400])
def test_rejected_search_never_sends_without_dialog(screen, text):
    window, backend = screen
    requests = []
    backend._service.web_search.client = client_for(lambda request: requests.append(request))
    window.input_edit.setPlainText(text)
    window.send_button.click()
    assert window.error_label.isVisible()
    assert not requests and window._thread is None


def test_model_generated_sources_are_not_accepted_in_regular_chat(screen):
    _window, backend = screen
    fake = {
        "title": "偽出典",
        "url": "https://fake.example",
        "retrieved_at": "2026-09-22T00:00:00+09:00",
    }
    backend._service.set_provider(
        MockProvider(response_text=json.dumps({"speech": "普通の返答", "sources": [fake]}))
    )
    conversation = backend.create_conversation("通常")
    outcome = backend.stream_message(conversation, "こんにちは", lambda _: None)
    assert outcome.sources == ()
    assert backend.load_messages(conversation)[-1].sources == ()


def test_quoted_external_fact_does_not_become_memory_but_user_fact_does(screen):
    _window, backend = screen
    backend._service.set_provider(
        MockProvider(
            response_text='{"speech":"そう書いてあるんだね。","memory_candidates":["ユーザーは青が好き"]}'
        )
    )
    conversation = backend.create_conversation("引用")
    backend.stream_message(
        conversation,
        "この記事に「ユーザーは青が好き」と書いてあったので続きを話そう",
        lambda _: None,
    )
    assert backend.list_memories() == []
    backend.stream_message(conversation, "私は青が好き。覚えておいて", lambda _: None)
    assert len(backend.list_memories()) == 1


@pytest.mark.parametrize("failed_turn", [False, True, "no_evidence"])
@pytest.mark.parametrize(
    "text",
    [
        "今年の9月の降水量と降水日は例年に比べてどうだったか比較して",
        "今年の9月の降水量と降水日は例年に比べてどうだったか調べて比較して",
        "今年の9月の降水量と降水日を調べて",
    ],
)
def test_observation_request_reuses_discussed_place_through_real_provider(
    screen, text, failed_turn
):
    window, backend = screen
    searches, completions, spoken = [], [], []

    def search(request):
        searches.append(json.loads(request.content))
        return httpx.Response(200, json=search_payload())

    def complete(request):
        payload = json.loads(request.content)
        completions.append(payload)
        if "tools" in payload:
            assert payload["stream"] is True
            assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False
            assert "東京の2026年9月1〜28日" in str(payload["messages"])
            return httpx.Response(200, json=research_call("東京の2026年9月1〜28日 " + text))
        assert payload["stream"] is False, "検索後の最終回答経路を維持する"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"speech":"資料から確認できた範囲を伝えるね。[S1]"}'}}
                ]
            },
        )

    backend._service.web_search.client = client_for(search)
    backend._service.set_provider(
        OpenAICompatibleProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-only",
            model="gpt-6-luna",
            transport=httpx.MockTransport(complete),
        )
    )
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window._messages = [
        UiMessage("user", "今年の値は9月28日までの暫定値で良いよ"),
        UiMessage(
            "assistant", "東京の2026年9月1〜28日の降水量と降水日数を平年と比べればいいんだね。"
        ),
    ]
    if failed_turn == "no_evidence":
        window._messages.extend(
            [
                UiMessage("user", "今年の9月の降水量と降水日を調べて"),
                UiMessage("assistant", "今回の検索では確かな資料を見つけられなかったよ。"),
            ]
        )
    elif failed_turn:
        window._messages.append(UiMessage("user", "それを調べて教えて下さい"))
    window._current_conversation_id = backend.create_conversation("観測資料の確認")
    for item in window._messages:
        backend._service._repository.add_message(
            window._current_conversation_id, item.role, item.content
        )
    window.input_edit.setPlainText(text)
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(searches) == 1 and len(completions) == 2
    query = json.loads(searches[0]["input"])["検索文"]
    assert "東京" in query and text in query
    assert "28日" in query
    assert "暫定値で良いよ" not in query
    assert window.error_label.isHidden()
    assert "資料から確認できた範囲" in window.chat_view.toPlainText()
    assert spoken and all("[S1]" not in s for s in spoken)


@pytest.mark.parametrize(
    "intervening",
    [
        [UiMessage("user", "お茶を飲もう"), UiMessage("assistant", "休憩しよう。")],
        [UiMessage("user", "お茶を飲もう"), UiMessage("assistant", "東京都の天気を話していたよ。")],
        [
            UiMessage("user", "旅行の予定は9月1〜28日だよ"),
            UiMessage("assistant", "東京都の天気を話していたよ。"),
        ],
        [UiMessage("user", "お茶を飲もう")],
        [
            UiMessage("user", "東京都と大阪府の降水量だよ"),
            UiMessage("assistant", "どちらを調べようか？"),
        ],
    ],
)
def test_weather_context_does_not_cross_another_topic_or_ambiguous_places(screen, intervening):
    window, _ = screen
    window._messages = [
        UiMessage("user", "東京の降水量を調べて"),
        UiMessage("assistant", "東京の降水量だね。"),
        *intervening,
    ]
    text = "今年の9月の降水量を調べて"
    assert window._web_query(text) == text


def test_interrupted_real_response_shape_becomes_spoken_final_reply_without_another_request(screen):
    window, backend = screen
    spoken, calls = [], []
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {"type": "reasoning"},
            {"type": "web_search_call", "status": "completed"},
            {"type": "web_search_call", "status": "searching"},
        ],
        "usage": {"input_tokens": 13680, "output_tokens": 884, "total_tokens": 14564},
    }
    backend._service.web_search.client = client_for(
        lambda request: calls.append(request) or httpx.Response(200, json=payload)
    )
    backend._service.set_provider(MockProvider(responder=lambda _: pytest.fail("追加生成禁止")))
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window.input_edit.setPlainText("東京の今年の9月の降水量を調べて")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(calls) == 1
    assert spoken and "最後まで終えられなかった" in "".join(spoken)
    assert "処理の上限に達して" in "".join(spoken)
    assert "調べているよ" not in window.chat_view.toPlainText()
    assert window.error_label.isHidden()
    saved = backend.load_messages(window._current_conversation_id)
    assert saved[-1].content == "".join(spoken) and saved[-1].sources == ()
    assert backend._service.last_usage.total_tokens == 14564
    assert backend.list_memories() == []


def test_empty_completion_after_no_evidence_still_gives_a_spoken_reply(screen):
    window, backend = screen
    spoken, calls = [], []
    payload = search_payload()
    payload["output"][1]["content"][0]["annotations"] = []
    backend._service.web_search.client = client_for(lambda _: httpx.Response(200, json=payload))

    def empty(request):
        calls.append(json.loads(request.content))
        if "tools" in calls[-1]:
            return httpx.Response(200, json=research_call("東京 2026年9月 降水量 降水日数"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    backend._service.set_provider(
        OpenAICompatibleProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-only",
            model="gpt-6-luna",
            transport=httpx.MockTransport(empty),
        )
    )
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window.input_edit.setPlainText("東京の今年の9月の降水量と降水日を調べて")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert len(calls) == 2
    assert window.error_label.isHidden()
    assert "見つけられなかった" in window.chat_view.toPlainText()
    assert spoken and "見つけられなかった" in "".join(spoken)
    saved = backend.load_messages(window._current_conversation_id)
    assert saved[-1].role == "assistant" and not saved[-1].sources
    assert backend._service.last_usage.total_tokens == 12 + 30 + payload["usage"]["total_tokens"]


def test_no_evidence_is_spoken_and_saved_as_conversation_after_reference_request(screen):
    window, backend = screen
    spoken, requests = [], []
    speech = (
        "探してみたけれど、今回の検索では東京の降水量を比較できる確かな資料を見つけられなかったよ。"
    )

    def no_evidence(request):
        requests.append(json.loads(request.content))
        data = search_payload()
        data["output"][1]["content"][0]["annotations"] = []
        return httpx.Response(200, json=data)

    backend._service.web_search.client = client_for(no_evidence)
    backend._service.set_provider(MockProvider(response_text=json.dumps({"speech": speech})))
    backend._service.voice_output.start_stream = spoken.append
    backend._service.voice_output.enqueue_stream = spoken.append
    window._messages = [
        UiMessage("user", "今年の値は9月28日までの暫定値で良いよ"),
        UiMessage(
            "assistant", "東京の2026年9月1〜28日の降水量と降水日数を平年と比べればいいんだね。"
        ),
    ]
    window.input_edit.setPlainText("それを調べて教えて下さい")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy and bool(spoken))
    assert len(requests) == 1
    assert "東京の2026年9月1〜28日" in json.loads(requests[0]["input"])["検索文"]
    assert speech in window.chat_view.toPlainText()
    assert "".join(spoken) == speech
    assert window.error_label.isHidden()
    assert window.web_activity_label.isHidden() and window.web_cancel_button.isHidden()
    assert "調べているよ" not in window.chat_view.toPlainText()
    saved = backend.load_messages(window._current_conversation_id)
    assert [m.content for m in saved if m.role == "assistant"] == [speech]
    assert saved[-1].sources == ()
    assert backend.list_memories() == []
    backend._service.set_provider(MockProvider(response_text='{"speech":"お茶にしよう。"}'))
    window.input_edit.setPlainText("お茶を飲もう")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "お茶にしよう" in window.chat_view.toPlainText()
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["denied", "timeout"])
def test_search_failure_shows_reason_then_returns_to_normal_chat(screen, failure):
    window, backend = screen

    def fail(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("private-error", request=request)
        data = search_payload()
        data["output"][1]["content"][0]["annotations"] = []
        return httpx.Response(403 if failure == "denied" else 200, json=data)

    backend._service.web_search.client = client_for(fail)
    window.input_edit.setPlainText("ニュースを調べて")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert window.error_label.isHidden()
    final = backend.load_messages(window._current_conversation_id)[-1]
    assert final.role == "assistant"
    assert ("拒否" if failure == "denied" else "時間内") in final.content
    assert "private-error" not in final.content
    assert window.web_activity_label.isHidden()
    assert window.web_cancel_button.isHidden()
    assert "調べているよ" not in window.chat_view.toPlainText()
    backend._service.set_provider(MockProvider(response_text='{"speech":"お茶にしよう。"}'))
    window.input_edit.setPlainText("お茶を飲もう")
    window.send_button.click()
    wait_until(lambda: window._thread is None and not window._busy)
    assert "お茶にしよう" in window.chat_view.toPlainText()
    assert window.error_label.isHidden()
