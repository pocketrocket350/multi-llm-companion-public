from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.application import ApplicationError
from multi_llm_companion.config import (
    AppConfig,
    AvatarConfig,
    PersonaConfig,
    ProviderConfig,
    ProvidersConfig,
    ResolvedProviderConfig,
    RuntimeConfig,
)
from multi_llm_companion.ui.memory_dialog import MemoryDialog
from scripts.phase9_acceptance_app import AcceptanceBackend, create_window, isolated_runtime


def _runtime(root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        project_root=root,
        app_config_path=root / "local/app.yaml",
        app=AppConfig(),
        persona=PersonaConfig(name="検証用", system_prompt="短く返す"),
        providers=ProvidersConfig(
            providers={
                "mock": ProviderConfig(
                    provider="mock", base_url_env="", api_key_env="", model_env=""
                )
            }
        ),
        active_provider=ResolvedProviderConfig(
            name="mock",
            provider="mock",
            base_url="",
            api_key="",
            model="mock-model",
            temperature=0.0,
            max_output_tokens=1024,
            timeout=5,
        ),
        database_path=root / "data/conversations.db",
        avatar=AvatarConfig(),
    )


def test_isolated_app_uses_own_database_settings_and_preserves_source(tmp_path) -> None:
    original = _runtime(tmp_path)
    directory = tmp_path / "isolated"
    runtime = isolated_runtime(original, directory)
    assert original.database_path == tmp_path / "data/conversations.db"
    assert runtime.project_root == original.project_root
    assert runtime.audio == original.audio
    assert runtime.avatar == original.avatar
    assert runtime.database_path.is_relative_to(directory)
    for path in (
        runtime.app_config_path,
        runtime.app.audio_config_path,
        runtime.app.memory_config_path,
        runtime.app.avatar_config_path,
    ):
        assert Path(path).is_relative_to(directory)
        assert Path(path).exists()
    with pytest.raises(FileExistsError):
        isolated_runtime(original, directory)
    app = QApplication.instance() or QApplication([])
    backend = AcceptanceBackend(runtime)
    window = create_window(backend, directory)
    try:
        assert str(directory) in window._settings.fileName().replace("/", os.sep)
        assert "隔離検証" in window.windowTitle()
        window._update_status_labels()
        assert not window._proactive_settings().enabled
        assert backend.list_memories() == []
        conversation = backend.create_conversation("隔離検証")
        backend.stream_message(conversation, "私をナツと呼んで", lambda _: None)
        assert len(backend.list_memories()) == 1
        backend.set_memory_enabled(False)
        assert not backend.memory_settings().enabled
        with pytest.raises(ApplicationError, match="カタログ"):
            backend.set_motion_enabled("any", False)
        with pytest.raises(ApplicationError, match="VMD"):
            backend.generate_confirmation_vmd()
        assert not original.database_path.exists()
        assert not original.app_config_path.exists()
    finally:
        window.close()
        backend.close()
        app.processEvents()


def test_memory_dialog_updates_real_isolated_backend_and_not_source(tmp_path) -> None:
    original = _runtime(tmp_path)
    runtime = isolated_runtime(original, tmp_path / "isolated")
    app = QApplication.instance() or QApplication([])
    backend = AcceptanceBackend(runtime)
    conversation = backend.create_conversation("隔離検証")
    backend.stream_message(conversation, "私をナツと呼んで", lambda _: None)
    dialog = MemoryDialog(backend)
    try:
        dialog.memory_list.setCurrentRow(0)
        assert dialog.status_label.text() == "承認済み"
        dialog.content_edit.setPlainText("ユーザーの呼び方はハルと呼んでほしい")
        dialog.save_button.click()
        assert backend.list_memories()[0].content == "ユーザーの呼び方はハルと呼んでほしい"
        dialog.enabled_check.setChecked(False)
        assert not backend.memory_settings().enabled
        dialog.enabled_check.setChecked(True)
        assert backend.memory_settings().enabled
        dialog.delete_button.click()
        assert not backend.list_memories()
        dialog.auto_approve_check.setChecked(False)
        backend.stream_message(conversation, "私をナツと呼んで", lambda _: None)
        dialog._reload()
        dialog.memory_list.setCurrentRow(0)
        assert dialog.status_label.text() == "確認待ち"
        dialog.approve_button.click()
        assert backend.list_memories()[0].status == "approved"
        assert not original.database_path.exists()
        assert not original.app_config_path.exists()
    finally:
        dialog.close()
        backend.close()
        app.processEvents()
