"""計測の欠測・95%点・人工待ち分離を検査する。"""

import json

from scripts.measure_companion_wait import (
    INTERVALS,
    REQUIRED,
    artificial_wait_ms,
    plan,
    research_sse_blocks,
    stats,
    summarize,
)


def test_wait_stats_uses_nearest_rank_and_median():
    result = stats(list(range(1, 21)))
    assert result == {"n": 20, "median_ms": 10.5, "p95_ms": 19, "max_ms": 20}
    assert stats([]) is None


def test_wait_summary_excludes_warmup_and_failed_samples():
    def case(warmup=False, success=True):
        return {
            "condition": "normal",
            "warmup": warmup,
            "success": success,
            "missing": [] if success else ["sentence_ready"],
            "events_ns": {key: 1 if success else None for key in REQUIRED},
            "intervals_ms": dict.fromkeys(INTERVALS, 700),
            "artificial_wait_ms": 600,
        }

    result = summarize([case(True), case(), case(success=False)])["normal"]
    assert result["success"] == result["failed"] == result["missing_cases"] == 1
    assert result["intervals"]["input_to_display"]["n"] == 1
    assert result["input_to_display_excluding_http_wait_ms"]["median_ms"] == 100
    assert result["events_missing"]["sentence_ready"] == 1
    assert result["provider_fragment_unobserved"] == 2


def test_wait_plan_preserves_official_research_route_and_deadlines():
    result = plan(20)
    assert result["provider"]["base_url"] == "https://api.openai.com/v1"
    assert result["provider"]["model"] == "gpt-6-luna"
    assert result["case_limit_seconds"] == 30
    assert result["series_limit_seconds"] == 2700
    assert "provider_first_fragment" in result["conditional_events"]
    assert result["samples_per_condition"] == 20


def test_wait_clips_old_and_future_waits_to_display_time():
    events = [
        {"event": "artificial_http_wait", "monotonic_ns": end, "actual_ns": end - start}
        for start, end in [(0, 600_000_000), (600_000_000, 700_000_000)]
    ]
    assert artificial_wait_ms(events, 200_000_000) == 500
    assert artificial_wait_ms(events, 200_000_000, 650_000_000) == 450


def test_wait_merges_overlapping_old_and_new_http_waits():
    events = [
        {"event": "artificial_http_wait", "monotonic_ns": end, "actual_ns": end - start}
        for start, end in [(0, 600_000_000), (300_000_000, 900_000_000)]
    ]
    assert artificial_wait_ms(events, 200_000_000) == 700
    assert artificial_wait_ms(events, 200_000_000, 500_000_000) == 300


def test_wait_sse_preserves_reply_and_exposes_sentence_before_finish():
    from multi_llm_companion.streaming import SpeechStreamFilter

    arguments = json.dumps({"speech": "うん、静かな時間だね。"}, ensure_ascii=False)
    answer = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "artificial",
                            "function": {"name": "reply_to_user", "arguments": arguments},
                        }
                    ]
                }
            }
        ],
        "usage": {"total_tokens": 8},
    }
    blocks = research_sse_blocks(answer)
    stream_filter = SpeechStreamFilter(require_structured=True)
    parts = [
        json.loads(block.decode().split("\n")[0][6:])["choices"][0]["delta"]["tool_calls"][0][
            "function"
        ]["arguments"]
        for block in blocks[:3]
    ]
    first = stream_filter.feed(parts[0])
    second = stream_filter.feed(parts[1])
    assert first and "。" not in first
    assert first + second == "うん、静かな時間だね。"
    assert "".join(parts) == arguments
    assert b"[DONE]" in blocks[3]
