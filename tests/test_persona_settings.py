from __future__ import annotations

from dataclasses import replace

import pytest
import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit
from test_application import _runtime
from test_ui import FakeBackend
from test_ui import isolate_window_settings as isolate_window_settings

from multi_llm_companion.application import ApplicationBackend, ApplicationError
from multi_llm_companion.config import ConfigError, PersonaConfig, save_persona_config
from multi_llm_companion.providers import MockProvider
from multi_llm_companion.ui.chat_window import ChatWindow
from multi_llm_companion.ui.persona_dialog import PersonaDialog

SOURCE = """# 設定の出典を残す
character:
  name: 案内役
  first_person: 私
  speaking_style: 穏やかに話す
  values:
    - 正直さ
behavior_policy:
  response_length: 短め
# 未編集の資料
official_facts: [設定の資料]
system_prompt_appendix: |
  以前の追加指示
dialogue_examples:
  - user: こんにちは
    assistant: こんにちは。
dialogue_modes:
  casual:
    do: 以前の場面指示
"""


def _persona() -> PersonaConfig:
    return PersonaConfig.model_validate(yaml.safe_load(SOURCE))


def _put(dialog, path, text):
    editor = dialog.editors[path]
    if isinstance(editor, QLineEdit):
        editor.setText(text)
    else:
        editor.setPlainText(text)


def test_persona_save_preserves_other_sections_and_restart(tmp_path):
    path = tmp_path / "persona.yaml"
    path.write_text(SOURCE, encoding="utf-8")
    old = _persona()
    changed = old.model_copy(deep=True)
    changed.character.speaking_style = "新しい口調"
    save_persona_config(path, changed, expected=old)
    saved = path.read_text(encoding="utf-8")
    assert saved.startswith("# 設定の出典を残す\n")
    assert SOURCE[SOURCE.index("behavior_policy:") :] == saved[saved.index("behavior_policy:") :]
    assert PersonaConfig.model_validate(yaml.safe_load(saved)) == changed
    before = path.read_bytes()
    save_persona_config(path, changed, expected=changed)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "source",
    [
        SOURCE,
        "{character: {name: 案内役}, behavior_policy: {}}",
        "---\ncharacter: {name: 案内役}\nbehavior_policy: {}\n...\n",
    ],
)
def test_persona_save_adds_sections_and_handles_block_and_flow(tmp_path, source):
    path = tmp_path / "persona.yaml"
    path.write_text(source, encoding="utf-8")
    old = PersonaConfig.model_validate(yaml.safe_load(source))
    new = old.model_copy(deep=True)
    new.system_prompt_appendix = "新しい複数行の指示\n次の行"
    new.official_facts = ["確認した事実", "もう一つ"]
    save_persona_config(path, new, expected=old)
    assert PersonaConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) == new


def test_persona_save_rejects_external_change_and_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "persona.yaml"
    path.write_text(SOURCE, encoding="utf-8")
    old = _persona()
    new = old.model_copy(update={"system_prompt_appendix": "新しい指示"})

    def fail(*_args):
        raise PermissionError("検証用")

    monkeypatch.setattr("multi_llm_companion.config.os.replace", fail)
    with pytest.raises(ConfigError, match="保存できません"):
        save_persona_config(path, new, expected=old)
    assert path.read_text(encoding="utf-8") == SOURCE
    assert list(tmp_path.glob("*.tmp")) == []
    external = SOURCE.replace("以前の追加指示", "別の編集")
    path.write_text(external, encoding="utf-8")
    with pytest.raises(ConfigError, match="別の場所"):
        save_persona_config(path, new, expected=old)
    assert path.read_text(encoding="utf-8") == external


def test_legacy_editor_and_storage_keep_original_format(tmp_path):
    QApplication.instance() or QApplication([])
    path = tmp_path / "persona.yaml"
    path.write_text("# 旧設定\nname: 案内役\nsystem_prompt: 以前の指示\n", encoding="utf-8")
    persona = PersonaConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    dialog = PersonaDialog(
        persona, lambda value: save_persona_config(path, value, expected=persona)
    )
    assert dialog.section_combo.count() == 1
    _put(dialog, ("system_prompt",), "更新した指示")
    dialog.save_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw == {"name": "案内役", "system_prompt": "更新した指示"}
    dialog.deleteLater()


def test_persona_editor_cancel_and_validation_preserve_original():
    QApplication.instance() or QApplication([])
    persona = _persona()
    calls = []
    dialog = PersonaDialog(persona, calls.append)
    assert dialog._value() == persona
    _put(dialog, ("character", "name"), "  ")
    dialog.save_button.click()
    assert not calls and not dialog.error_label.isHidden()
    _put(dialog, ("character", "name"), "変更案")
    dialog.reject()
    assert not calls and persona.character.name == "案内役"
    dialog.deleteLater()


def test_unedited_multiline_list_item_is_not_split():
    QApplication.instance() or QApplication([])
    persona = _persona()
    persona.character.values = ["一つの項目の\n続き"]
    dialog = PersonaDialog(persona, lambda _: None)
    assert dialog._value() == persona
    dialog.deleteLater()


def test_persona_editor_failed_save_preserves_draft():
    QApplication.instance() or QApplication([])
    persona = _persona()

    def fail(value):
        raise ApplicationError("保存先に書き込めません")

    dialog = PersonaDialog(persona, fail)
    _put(dialog, ("system_prompt_appendix",), "保存していない編集")
    dialog.save_button.click()
    assert "保存先" in dialog.error_label.text()
    assert dialog._value().system_prompt_appendix == "保存していない編集"
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_persona_editor_all_sections_and_small_screen():
    app = QApplication.instance() or QApplication([])
    saved = []
    dialog = PersonaDialog(_persona(), saved.append)
    _put(dialog, ("character", "values"), "正直さ\n好奇心")
    _put(dialog, ("behavior_policy", "initiative"), "自然に話題を出す")
    _put(dialog, ("style_anchors",), "短い文\n自然な言葉")
    _put(dialog, ("dialogue_modes", "listening", "do"), "静かに聞く")
    _put(dialog, ("system_prompt_appendix",), "新しい指示")
    dialog._add_example("おつかれ", "おつかれさま。")
    dialog.resize(500, 430)
    dialog.show()
    app.processEvents()
    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    for index in range(dialog.section_combo.count()):
        dialog.section_combo.setCurrentIndex(index)
        app.processEvents()
        scroll = dialog.pages.currentWidget()
        assert scroll.horizontalScrollBar().maximum() == 0
        assert dialog.rect().contains(
            dialog.save_button.mapTo(dialog, dialog.save_button.rect().center())
        )
    dialog.save_button.click()
    value = saved[0]
    assert value.character.values == ["正直さ", "好奇心"]
    assert value.behavior_policy.initiative == "自然に話題を出す"
    assert value.style_anchors == ["短い文", "自然な言葉"]
    assert value.dialogue_modes["listening"].do == "静かに聞く"
    assert value.dialogue_modes["casual"].do == "以前の場面指示"
    assert value.dialogue_examples[-1].assistant == "おつかれさま。"
    assert value.system_prompt_appendix == "新しい指示"
    dialog.deleteLater()


def test_backend_updates_normal_and_proactive_persona_without_losing_conversation(tmp_path):
    runtime = replace(_runtime(tmp_path), persona=_persona())
    path = tmp_path / "local/persona.yaml"
    path.write_text(SOURCE, encoding="utf-8")
    backend = ApplicationBackend(runtime)
    seen = []

    def respond(messages):
        seen.append(messages)
        return '{"speech":"確認しました。"}'

    backend._service.set_provider(MockProvider(responder=respond))
    try:
        conversation = backend.create_conversation("継続する会話")
        backend.stream_message(conversation, "こんにちは", lambda _: None)
        before = backend.load_messages(conversation)
        memory = backend._service.memory_service
        engine = backend._service._engine_contract
        new = backend.persona_settings()
        new.system_prompt_appendix = "新しい人格の指示"
        new.dialogue_modes["casual"].do = "新しい場面方針"
        backend.change_persona_settings(new)
        assert backend.load_messages(conversation) == before
        assert backend._service.memory_service is memory
        assert backend._service._engine_contract == engine
        backend.stream_message(conversation, "その後どう？", lambda _: None)
        prompt = seen[-1][0].content
        assert "新しい人格の指示" in prompt and "新しい場面方針" in prompt
        assert "以前の追加指示" not in prompt and "以前の場面指示" not in prompt
        backend.generate_proactive_turn(conversation)
        assert "新しい人格の指示" in seen[-1][0].content
        assert "新しい場面方針" in seen[-1][0].content
        reloaded = PersonaConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert reloaded == backend.persona_settings()
    finally:
        backend.close()


def test_backend_busy_or_save_error_keeps_runtime_and_file(tmp_path, monkeypatch):
    runtime = replace(_runtime(tmp_path), persona=_persona())
    path = tmp_path / "local/persona.yaml"
    path.write_text(SOURCE, encoding="utf-8")
    backend = ApplicationBackend(runtime)
    new = backend.persona_settings()
    new.system_prompt_appendix = "変更案"
    try:
        backend._service._send_lock.acquire()
        with pytest.raises(ApplicationError, match="処理が終わって"):
            backend.change_persona_settings(new)
        backend._service._send_lock.release()

        def fail(*_args, **_kwargs):
            raise ConfigError("検証用の保存失敗")

        monkeypatch.setattr("multi_llm_companion.application.save_persona_config", fail)
        with pytest.raises(ApplicationError, match="保存失敗"):
            backend.change_persona_settings(new)
        assert path.read_text(encoding="utf-8") == SOURCE
        assert backend.persona_settings() == runtime.persona
        assert "変更案" not in backend._service._system_prompt
    finally:
        if backend._service._send_lock.locked():
            backend._service._send_lock.release()
        backend.close()


def test_settings_entry_suspends_proactive_and_does_not_save_on_cancel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    backend.persona_settings = _persona
    calls = []
    backend.change_persona_settings = calls.append
    window = ChatWindow(backend)
    window._set_busy(True)
    assert not window.persona_button.isEnabled()
    window._set_busy(False)
    backend.listening = True
    window._sync_activity_controls()
    assert not window.persona_button.isEnabled()
    backend.listening = False
    window._sync_activity_controls()
    checked = []

    def close_editor():
        dialog = app.activeModalWidget()
        assert isinstance(dialog, PersonaDialog)
        assert not window._proactive_timer.isActive()
        _put(dialog, ("system_prompt_appendix",), "キャンセルする編集")
        checked.append(True)
        dialog.reject()

    QTimer.singleShot(0, close_editor)
    window.persona_button.click()
    assert checked and calls == [] and window._proactive_timer.isActive()
    window.close()
