from __future__ import annotations

import pytest

from multi_llm_companion.config import MemoryConfig
from multi_llm_companion.domain import MemoryCategory, MemoryStatus
from multi_llm_companion.services.memory import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from scripts.evaluate_memory_retrieval import cases, run


@pytest.fixture
def repository():
    with SQLiteConversationRepository(":memory:") as repo:
        yield repo


def distractors(repository, count=120, long=False):
    for index in range(count):
        content = f"工作記録{index:03}では紙の箱の寸法を測定した。"
        repository.create_memory(content * (15 if long else 1), status=MemoryStatus.APPROVED)


@pytest.mark.parametrize(
    ("fact", "query"),
    [(case["fact"], case["query"]) for case in cases() if case["kind"] == "paraphrase"],
)
def test_old_memory_reached_with_distant_wording(repository, fact, query):
    target = repository.create_memory(fact, status=MemoryStatus.APPROVED)
    distractors(repository)
    service = MemoryService(repository, MemoryConfig())
    turn = service.prepare_turn(query)
    assert target.id in {item.id for item in turn.memories}
    assert len(turn.memories) <= 40
    assert sum(len(item.content) for item in turn.memories) <= 10000
    assert len({item.id for item in turn.memories}) == len(turn.memories)


@pytest.mark.parametrize("kind", ["pending", "deleted", "expired", "sensitive", "disabled"])
def test_topic_retrieval_preserves_memory_boundaries(repository, kind):
    target = repository.create_memory(
        "休日の楽しみはラジオ修理。" + ("APIキー: artificial" if kind == "sensitive" else ""),
        status=MemoryStatus.CANDIDATE if kind == "pending" else MemoryStatus.APPROVED,
    )
    distractors(repository)
    service = MemoryService(repository, MemoryConfig(max_memory_age_days=1))
    if kind == "deleted":
        service.delete(target.id)
    elif kind == "disabled":
        service.set_enabled(False)
    elif kind == "expired":
        with repository._connection:
            repository._connection.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ?",
                ("2001-01-01T00:00:00+00:00", target.id),
            )
    turn = service.prepare_turn("余暇の過ごし方は？")
    assert target.id not in {item.id for item in turn.memories}
    assert "ラジオ" not in turn.prompt


def test_topic_candidates_keep_content_budget_and_current_value(repository):
    target = repository.create_memory("休日は釣り。", status=MemoryStatus.APPROVED)
    distractors(repository, long=True)
    service = MemoryService(repository, MemoryConfig())
    service.update(target.id, "休日は彫金。", MemoryCategory.PREFERENCE)
    turn = service.prepare_turn("余暇の過ごし方は？")
    assert any(item.content == "休日は彫金。" for item in turn.memories)
    assert "釣り" not in turn.prompt
    assert sum(len(item.content) for item in turn.memories) <= 10000


def test_topic_supplement_does_not_evict_specific_lexical_match(repository):
    target = repository.create_memory("休日はラジオ修理。", status=MemoryStatus.APPROVED)
    for index in range(60):
        repository.create_memory(f"休日の楽しみは工作{index:03}。", status=MemoryStatus.APPROVED)
    service = MemoryService(repository, MemoryConfig())
    assert target.id in {item.id for item in service.prepare_turn("休日のラジオ修理は？").memories}


def test_unknown_topic_does_not_select_old_leisure_memory(repository):
    target = repository.create_memory("休日はラジオ修理。", status=MemoryStatus.APPROVED)
    distractors(repository)
    service = MemoryService(repository, MemoryConfig())
    assert target.id not in {item.id for item in service.prepare_turn("天気予報は？").memories}


def test_repeated_correction_evaluation_keeps_original_controls_and_counts(tmp_path, monkeypatch):
    import json

    assert cases(2)[:20] == cases()
    assert [case["correction_repeat"] for case in cases(2)[20:]] == [1, 2]
    assert [case["kind"] for case in cases(2)[16:20]] == [
        "absent",
        "deleted",
        "pending",
        "disabled",
    ]
    repeated = cases(2)[20:]
    monkeypatch.setattr("scripts.evaluate_memory_retrieval.cases", lambda repeats: repeated)
    assert run(tmp_path, tmp_path, correction_repeats=2) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert summary["completed"] == len(rows) == 2
    assert summary["missing"] == summary["budget"]["requests"] == 0
    assert all(row["target_retrieved"] for row in rows)
