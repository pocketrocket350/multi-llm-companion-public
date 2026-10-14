"""完走時間の水増し防止と、人工データだけのバックアップ復元を確認する。"""

import importlib.util
import json
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import yaml
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage

from multi_llm_companion.config import (
    AppConfig,
    AudioConfig,
    AvatarConfig,
    MemoryConfig,
    PersonaConfig,
    ProviderConfig,
    ProvidersConfig,
    load_runtime_config,
)
from multi_llm_companion.domain import AssistantResponse, MemoryStatus
from multi_llm_companion.domain.models import ImageAttachmentInfo, WebSource
from multi_llm_companion.services import MemoryService
from multi_llm_companion.storage import SQLiteConversationRepository
from multi_llm_companion.ui.image_attachment import prepare_image

SPEC = importlib.util.spec_from_file_location(
    "phase12_soak", Path(__file__).parents[1] / "scripts/phase12_soak.py"
)
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_atomic_save_recovers_a_transient_windows_reader(tmp_path, monkeypatch, winerror):
    path = tmp_path / "state.json"
    soak.save(path, {"phase": "load"})
    replace = Path.replace
    attempts = []
    delays = []

    def held_reader(temporary, target):
        attempts.append(target)
        if len(attempts) == 1:
            assert json.loads(path.read_text(encoding="utf-8")) == {"phase": "load"}
            raise PermissionError(13, "reader holds file", str(target), winerror)
        return replace(temporary, target)

    monkeypatch.setattr(Path, "replace", held_reader)
    monkeypatch.setattr(soak.time, "sleep", delays.append)
    soak.save(path, {"phase": "idle"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"phase": "idle"}
    assert len(attempts) == 2 and sum(delays) <= 0.15
    retry = json.loads((tmp_path / "atomic-save-retries.jsonl").read_text(encoding="utf-8"))
    assert retry["winerror"] == winerror and retry["retries"] == 1


def test_atomic_save_persistent_denial_fails_without_losing_old_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    soak.save(path, {"version": "old"})
    attempts = []
    delays = []

    def denied(temporary, target):
        attempts.append(target)
        raise PermissionError(13, "persistent denial", str(target), 5)

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(soak.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        soak.save(path, {"version": "new"})
    assert len(attempts) <= 5 and sum(delays) <= 0.15
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}
    assert json.loads(path.with_suffix(".json.tmp").read_text(encoding="utf-8")) == {
        "version": "new"
    }


@pytest.mark.parametrize("error", [PermissionError("unrecognized denial"), OSError("disk error")])
def test_atomic_save_does_not_retry_unrelated_errors(tmp_path, monkeypatch, error):
    path = tmp_path / "state.json"
    delays = []

    def denied(temporary, target):
        raise error

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(soak.time, "sleep", delays.append)
    with pytest.raises(type(error)):
        soak.save(path, {"phase": "idle"})
    assert not delays


@pytest.mark.parametrize(
    ("wall", "awake", "visible", "frames", "expected"),
    [
        (5, 5, True, 100, 5),
        (5, 0, True, 100, 0),
        (5, 5, False, 100, 0),
        (5, 5, True, 0, 0),
        (60, 60, True, 100, 0),
        (5, 4.9, True, 100, 4.9),
    ],
)
def test_sleep_stall_and_missing_frames_do_not_count(wall, awake, visible, frames, expected):
    assert soak.active_credit(wall, awake, visible=visible, frames=frames) == expected


def test_missing_metrics_are_not_reported_as_zero():
    assert soak.distribution([]) is None
    assert soak.distribution([1, 2, 3, 100])["max"] == 100


def test_resource_slope_uses_elapsed_hours_and_missing_data_is_unknown():
    assert soak.slope_per_hour([]) is None
    assert soak.slope_per_hour([(100, 30)]) is None
    assert soak.slope_per_hour([(100, 30), (1900, 40), (3700, 50)]) == pytest.approx(20)


def test_sqlite_backup_restores_settings_images_sources_and_corrected_memories(tmp_path):
    source = tmp_path / "artificial"
    destination = tmp_path / "empty-restore"
    source.mkdir()
    destination.mkdir()
    db = source / "conversations.db"
    repository = SQLiteConversationRepository(db)
    first = repository.create_conversation("人工会話A")
    second = repository.create_conversation("人工会話B")
    picture = QImage(80, 60, QImage.Format.Format_RGB32)
    picture.fill(0xFFCC4422)
    picture.save(str(source / "fixture.png"))
    prepared = prepare_image(str(source / "fixture.png"))
    attachment = ImageAttachmentInfo(
        name="fixture.png", width=80, height=60, preview_jpeg=prepared.preview_jpeg
    )
    repository.add_message(
        first.id,
        "user",
        "この人工画像を見て",
        response=AssistantResponse(speech="", image_attachment=attachment),
    )
    response = AssistantResponse(
        speech="人工資料の返答",
        sources=[
            WebSource(
                title="人工出典",
                url="https://example.invalid/source",
                retrieved_at=datetime.now().astimezone(),
            )
        ],
    )
    repository.add_message(first.id, "assistant", response.speech, response=response)
    repository.add_message(second.id, "user", "別の人工会話")
    memory = MemoryService(repository, MemoryConfig(auto_approve_candidates=True))
    old = repository.create_memory("ユーザーの朝食はトースト", status=MemoryStatus.APPROVED)
    deleted = repository.create_memory("ユーザーは紅茶が好き", status=MemoryStatus.APPROVED)
    text = "朝食はご飯に変えたよ"
    memory.propose(
        text,
        AssistantResponse(
            speech="わかったよ",
            memory_updates=[
                {
                    "action": "update",
                    "evidence": text,
                    "target_id": old.id,
                    "target_content": old.content,
                    "content": "ユーザーの朝食はご飯",
                }
            ],
        ),
        first.id,
        turn=memory.prepare_turn(text),
    )
    text = "紅茶が好きなことを忘れて"
    memory.propose(
        text,
        AssistantResponse(
            speech="わかったよ",
            memory_updates=[
                {
                    "action": "delete",
                    "evidence": text,
                    "target_id": deleted.id,
                    "target_content": deleted.content,
                }
            ],
        ),
        first.id,
        turn=memory.prepare_turn(text),
    )
    assert [entry.content for entry in memory.list_memories()] == ["ユーザーの朝食はご飯"]
    persona = PersonaConfig(name="人工人格", system_prompt="人工の検査です")
    config_models = {
        "persona": persona,
        "audio": AudioConfig(),
        "avatar": AvatarConfig(),
        "memory": MemoryConfig(),
        "providers": ProvidersConfig(
            providers={
                "mock": ProviderConfig(
                    provider="mock", base_url_env="", api_key_env="", model_env=""
                )
            }
        ),
        "app": AppConfig(
            database_path="conversations.db",
            persona_path="persona.yaml",
            providers_config_path="providers.yaml",
            audio_config_path="audio.yaml",
            avatar_config_path="avatar.yaml",
            memory_config_path="memory.yaml",
        ),
    }
    for name, model in config_models.items():
        (source / (name + ".yaml")).write_text(
            yaml.safe_dump(model.model_dump(by_alias=True), allow_unicode=True), encoding="utf-8"
        )
    settings = QSettings(str(source / "window.ini"), QSettings.Format.IniFormat)
    settings.setValue("proactive/enabled", False)
    settings.setValue("window/size", "1100x760")
    settings.sync()
    # WAL接続を残したままSQLiteのオンラインbackupで整合したスナップショットを作る。
    with sqlite3.connect(db) as reader, sqlite3.connect(destination / "conversations.db") as target:
        reader.backup(target)
        assert target.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert target.execute("PRAGMA foreign_key_check").fetchall() == []
    for name in (*(name + ".yaml" for name in config_models), "window.ini"):
        shutil.copy2(source / name, destination / name)
    restored_runtime = load_runtime_config(destination, destination / "app.yaml")
    assert restored_runtime.database_path == destination / "conversations.db"
    assert restored_runtime.persona == persona
    assert not restored_runtime.audio.speech_recognition.enabled
    assert restored_runtime.active_provider.provider == "mock"
    restored = SQLiteConversationRepository(destination / "conversations.db")
    try:
        assert restored.list_conversations() == repository.list_conversations()
        assert restored.list_messages(first.id) == repository.list_messages(first.id)
        assert restored.list_messages(second.id) == repository.list_messages(second.id)
        assert restored.list_memories() == repository.list_memories()
        assert restored.list_memories()[0].id == old.id
        restored_settings = QSettings(str(destination / "window.ini"), QSettings.Format.IniFormat)
        assert restored_settings.value("window/size") == "1100x760"
        assert (
            PersonaConfig.model_validate(
                yaml.safe_load((destination / "persona.yaml").read_text(encoding="utf-8"))
            )
            == persona
        )
        restored.delete_conversation(first.id)
        assert restored.list_messages(first.id) == []
        assert restored.list_messages(second.id)
        assert repository.list_messages(first.id)
    finally:
        restored.close()
        repository.close()
    (tmp_path / "recovery-result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "conversations": 2,
                "image_preview": True,
                "sources": True,
                "memory_correction_and_deletion": True,
                "sqlite_online_backup": True,
                "settings_and_persona": True,
                "normal_data_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("fault", ["timeout", "device_lost"])
def test_local_voice_fault_stops_and_next_mock_synthesis_recovers(tmp_path, fault):
    from test_audio_integrations import FakeWavPlayer, _wav_bytes

    from multi_llm_companion.integrations import VoiSonaTalkVoiceOutput

    broken = True

    class FaultPlayer(FakeWavPlayer):
        def play(self):
            if broken and fault == "device_lost":
                raise RuntimeError("fixture device unavailable")
            super().play()

    player = FaultPlayer(finish_on_play=True)

    def handler(request):
        if broken and fault == "timeout":
            raise httpx.ReadTimeout("fixture timeout")
        if request.url.path.endswith("/voices"):
            return httpx.Response(200, json=[{"voice_name": "fixture"}])
        if request.method == "POST":
            Path(json.loads(request.content)["output_file_path"]).write_bytes(_wav_bytes(0.02))
            return httpx.Response(200, json={"uuid": "fixture"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"state": "succeeded"})

    voice = VoiSonaTalkVoiceOutput(
        base_url="http://127.0.0.1:32766/api/talk/v1",
        username="fixture",
        password="fixture",
        voice_name="fixture",
        transport=httpx.MockTransport(handler),
        wav_player=player,
        temp_root=tmp_path,
        monitor_interval=0.05,
    )

    def settle():
        end = time.monotonic() + 3
        while voice.is_speaking and time.monotonic() < end:
            time.sleep(0.01)
        assert not voice.is_speaking

    try:
        voice.start_stream("障害の検査。")
        voice.finish_stream()
        settle()
        assert voice.last_error
        voice.stop()
        assert not player.is_playing and not voice.lip_sync_state.is_active
        broken = False
        voice.start_stream("復帰の検査。")
        voice.finish_stream()
        settle()
        assert voice.last_error is None
        assert not player.is_playing
    finally:
        voice.close()
