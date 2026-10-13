"""既存の人格設定を、日本語の項目から編集する。"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multi_llm_companion.config import PersonaConfig

from ._dialog_layout import _fit_dialog


class PersonaDialog(QDialog):
    def __init__(
        self,
        persona: PersonaConfig,
        save: Callable[[PersonaConfig], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("人格の設定")
        self._original = persona.model_copy(deep=True)
        self._save = save
        self.editors: dict[tuple[str, ...], QLineEdit | QTextEdit] = {}
        self._list_fields: set[tuple[str, ...]] = set()
        self._initial: dict[tuple[str, ...], str | list[str]] = {}
        self.examples: QTableWidget | None = None
        root = QVBoxLayout(self)
        note = QLabel(
            "話し方や性格、会話の方針を編集します。「保存」で次の返答から反映し、"
            "再起動後も維持します。会話時に、設定中のLLMへ人格の内容も送ります。"
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.section_combo = QComboBox()
        self.section_combo.setAccessibleName("編集する人格の項目")
        root.addWidget(self.section_combo)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.section_combo.currentIndexChanged.connect(self.pages.setCurrentIndex)
        if persona.is_legacy:
            page = self._page("名前・人格")
            self._field(page, ("name",), "名前", persona.name, single=True)
            self._field(page, ("system_prompt",), "人格・話し方の指示", persona.system_prompt)
        else:
            self._structured_fields(persona)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #9B2434;")
        self.error_label.hide()
        root.addWidget(self.error_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("キャンセル")
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        _fit_dialog(self, 780, 680)

    def _page(self, title: str) -> QVBoxLayout:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        scroll.setWidget(page)
        self.pages.addWidget(scroll)
        self.section_combo.addItem(title)
        return layout

    def _field(
        self,
        layout: QVBoxLayout,
        path: tuple[str, ...],
        label: str,
        value: str | list[str],
        *,
        single: bool = False,
    ) -> None:
        self._initial[path] = value
        if isinstance(value, list):
            self._list_fields.add(path)
            text = "\n".join(value)
            label += "（1行に1項目）"
        else:
            text = value
        caption = QLabel(label)
        caption.setWordWrap(True)
        layout.addWidget(caption)
        editor = QLineEdit(text) if single else QTextEdit()
        if isinstance(editor, QTextEdit):
            editor.setAcceptRichText(False)
            editor.setPlainText(text)
            editor.setMinimumHeight(105)
            editor.setTabChangesFocus(True)
        editor.setAccessibleName(label)
        caption.setBuddy(editor)
        layout.addWidget(editor)
        self.editors[path] = editor

    def _structured_fields(self, persona: PersonaConfig) -> None:
        character = persona.character
        policy = persona.behavior_policy
        assert character is not None and policy is not None
        page = self._page("話し方・性格")
        for key, label in (
            ("name", "名前"),
            ("first_person", "一人称"),
            ("speaking_style", "口調"),
            ("values", "大切にすること"),
            ("preferences", "好み"),
            ("character_notes", "性格・キャラクターの設定"),
        ):
            self._field(
                page,
                ("character", key),
                label,
                getattr(character, key),
                single=key in {"name", "first_person"},
            )
        page = self._page("会話の方針")
        for key, label in (
            ("response_length", "返答の長さ"),
            ("question_frequency", "質問する頻度"),
            ("initiative", "話題や提案の出し方"),
            ("disagreement_style", "反対意見の伝え方"),
            ("user_autonomy", "利用者の意思の尊重"),
            ("intimacy_boundary", "親しさと距離感"),
            ("technical_guidance", "詳しい説明・技術的な質問への対応"),
            ("memory_usage", "記憶の使い方"),
        ):
            self._field(page, ("behavior_policy", key), label, getattr(policy, key))
        page = self._page("背景・表現")
        for key, label in (
            ("official_facts", "公式に確認した事実"),
            ("interpretations", "このアプリでの解釈"),
            ("forbidden_inferences", "推測で付け加えないこと"),
            ("style_anchors", "話し方の基準"),
            ("anti_patterns", "避ける表現"),
        ):
            self._field(page, (key,), label, getattr(persona, key))
        page = self._page("会話例")
        note = QLabel("話し方の参考にする例です。実際の会話履歴や記憶には登録しません。")
        note.setWordWrap(True)
        page.addWidget(note)
        self.examples = QTableWidget(0, 2)
        self.examples.setHorizontalHeaderLabels(["利用者の言葉", "雨衣の返答例"])
        self.examples.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.examples.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.examples.setMinimumHeight(240)
        self.examples.setAccessibleName("会話例")
        for example in persona.dialogue_examples:
            self._add_example(example.user, example.assistant)
        page.addWidget(self.examples, 1)
        row = QHBoxLayout()
        add = QPushButton("例を追加")
        remove = QPushButton("選んだ例を削除")
        add.clicked.connect(lambda: self._add_example())
        remove.clicked.connect(self._remove_examples)
        row.addWidget(add)
        row.addWidget(remove)
        page.addLayout(row)
        page = self._page("場面ごとの話し方")
        for mode, title in (
            ("casual", "雑談"),
            ("listening", "話を聞く"),
            ("supporting", "相談を受ける"),
            ("playful", "冗談・遊び"),
            ("explaining", "説明する"),
            ("planning", "計画を考える"),
            ("boundary", "距離や境界を伝える"),
        ):
            values = persona.dialogue_modes.get(mode)
            for key, label in (
                ("applies_when", "使う場面"),
                ("do", "行うこと"),
                ("avoid", "避けること"),
            ):
                self._field(
                    page,
                    ("dialogue_modes", mode, key),
                    f"{title}：{label}",
                    getattr(values, key) if values else "",
                )
        page = self._page("追加指示")
        self._field(
            page,
            ("system_prompt_appendix",),
            "その他の人格・会話の指示",
            persona.system_prompt_appendix,
        )

    def _add_example(self, user: str = "", assistant: str = "") -> None:
        assert self.examples is not None
        row = self.examples.rowCount()
        self.examples.insertRow(row)
        for column, text in enumerate((user, assistant)):
            self.examples.setItem(row, column, QTableWidgetItem(text))

    def _remove_examples(self) -> None:
        assert self.examples is not None
        rows = {index.row() for index in self.examples.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.examples.removeRow(row)

    def _value(self) -> PersonaConfig:
        data = self._original.model_dump(by_alias=True, exclude_none=True)
        for path, editor in self.editors.items():
            text = editor.text() if isinstance(editor, QLineEdit) else editor.toPlainText()
            initial = self._initial[path]
            if text == ("\n".join(initial) if isinstance(initial, list) else initial):
                continue
            value = (
                [line.strip() for line in text.splitlines() if line.strip()]
                if path in self._list_fields
                else text.strip()
            )
            if value == self._initial[path]:
                continue
            target = data
            for key in path[:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = value
        if self.examples is not None:
            examples = []
            for row in range(self.examples.rowCount()):
                user, assistant = (
                    self.examples.item(row, column).text().strip() for column in (0, 1)
                )
                if user or assistant:
                    examples.append({"user": user, "assistant": assistant})
            data["dialogue_examples"] = examples
        return PersonaConfig.model_validate(data)

    def _save_and_close(self) -> None:
        try:
            persona = self._value()
            if not persona.name.strip():
                raise ValueError("名前を入力してください。")
        except (ValidationError, ValueError):
            self.error_label.setText(
                "名前と人格の指示を確認してください。空のままでは保存できません。"
            )
            self.error_label.show()
            return
        try:
            self._save(persona)
        except RuntimeError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            return
        self.accept()
