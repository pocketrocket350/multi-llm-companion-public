from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from multi_llm_companion.providers import MockProvider
from multi_llm_companion.services import ChatService
from multi_llm_companion.services.chat import _current_time_context
from multi_llm_companion.services.web_search import OpenAIWebSearch
from multi_llm_companion.storage import SQLiteConversationRepository


@pytest.mark.parametrize(
    ("instant", "weekday"),
    [
        (datetime(2026, 9, 22, 0, 0, tzinfo=timezone(timedelta(hours=9))), "火"),
        (datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC), "木"),
        (datetime(2027, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-8))), "金"),
        (datetime(2026, 9, 22, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=45))), "火"),
    ],
)
def test_current_time_keeps_local_date_offset_and_weekday(instant, weekday) -> None:
    with patch("multi_llm_companion.services.chat.datetime") as clock:
        clock.now.return_value.astimezone.return_value = instant
        context = _current_time_context()
    assert instant.isoformat(timespec="seconds") in context
    assert f"{weekday}曜日" in context
    assert "居住地・現在地・ニュース・天気の根拠ではない" in context
    assert "Web検索を行ったことも意味しない" in context


def test_every_chat_path_refreshes_clock_without_saving_system_time() -> None:
    captured = []

    def respond(messages):
        captured.append(messages[0].content)
        return '{"speech":"会話を続けます。"}'

    service = ChatService(
        provider=MockProvider(responder=respond),
        repository=SQLiteConversationRepository(":memory:"),
        system_prompt="受入済みの人格",
        engine_contract="既存の表現契約",
    )
    before = datetime(2026, 9, 21, 23, 59, 59, tzinfo=timezone(timedelta(hours=9)))
    after = before + timedelta(seconds=1)
    try:
        with patch("multi_llm_companion.services.chat.datetime") as clock:
            clock.now.return_value.astimezone.side_effect = [before, after, after]
            service.send_message("今何時？")
            list(service.stream_speech("日付は変わった？"))
            service.generate_proactive_turn()
        assert before.isoformat() in captured[0]
        assert after.isoformat() in captured[1]
        assert after.isoformat() in captured[2]
        assert all("受入済みの人格" in prompt for prompt in captured)
        assert all("既存の表現契約" in prompt for prompt in captured)
        assert all(
            "【このターンの現在日時】" not in m.content for m in service.get_current_messages()
        )
    finally:
        service.close()


@pytest.mark.parametrize("enabled,available", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("stream", [False, True])
def test_capability_uses_actual_setting_separately_from_search_execution(
    enabled, available, stream
):
    captured = []
    service = ChatService(
        provider=MockProvider(
            responder=lambda messages: (
                captured.append(messages[0].content) or '{"speech":"通常の返答"}'
            )
        ),
        repository=SQLiteConversationRepository(":memory:"),
    )
    if available:
        service.web_search.client = OpenAIWebSearch(api_key="unused", model="gpt-6-luna")
    service.set_web_search_enabled(enabled)
    try:
        if stream:
            list(service.stream_speech("Web検索はできる？"))
        else:
            service.send_message("Web検索はできる？")
        prompt = captured[0]
        assert "Web検索は未実施" in prompt
        if not enabled:
            assert "Web検索は設定でオフ" in prompt
        elif not available:
            assert "検索経路を利用できない" in prompt
        else:
            assert "検索経路が利用できる" in prompt
            assert "検索結果が届くまで保証しない" in prompt
        service.generate_proactive_turn()
        assert "【アプリのWeb検索の状態】" not in captured[-1]
        assert "能動会話ではWeb検索・カメラ・外部ツールを使用していない" in captured[-1]
    finally:
        service.close()
