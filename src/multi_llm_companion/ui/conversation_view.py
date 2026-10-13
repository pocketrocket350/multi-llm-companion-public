"""既存の本文・出典を、選択可能なQtの吹き出しで表示する。"""

import base64
import math
from datetime import datetime

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize, Qt, QTimer
from PySide6.QtGui import QFont, QImageReader, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class _MessageBubble(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.role = ""
        self.body = ""
        self._fitted_content = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 8)
        layout.setSpacing(5)
        self.speaker = QLabel()
        self.speaker.setStyleSheet("color: #1A267C; font-size: 13px;")
        layout.addWidget(self.speaker)
        self.frame = QFrame()
        self.frame.setObjectName("messageBubble")
        inner = QVBoxLayout(self.frame)
        inner.setContentsMargins(14, 9, 14, 9)
        self.image_label = QLabel()
        self.image_label.setAccessibleName("送った画像の縮小コピー")
        self.image_label.hide()
        self._image_data = None
        self._image = QPixmap()
        self._scaled_image_size = None
        inner.addWidget(self.image_label)
        self.text = QTextBrowser()
        self.text.setFrameShape(QFrame.Shape.NoFrame)
        self.text.setOpenExternalLinks(True)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text.setStyleSheet(
            "QTextBrowser {background: transparent; color: #182436; border: none;"
            " padding: 0; margin: 0; font-size: 16px;}"
        )
        self.text.document().setDefaultFont(QFont("Yu Gothic UI", 12))
        self.text.document().setDocumentMargin(0)
        self.text.document().setDefaultStyleSheet(
            "a { color: #244CB3; } small { font-size: 11px; }"
        )
        inner.addWidget(self.text)
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #576A7D; font-size: 11px;")
        self.time_label.setFixedWidth(36)
        self.time_label.hide()
        self.body_layout = QHBoxLayout()
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        self.body_layout.addWidget(self.frame)
        self.body_layout.addWidget(self.time_label, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(self.body_layout)

    def update_content(self, role: str, speaker: str, body: str) -> None:
        if self.role != role:
            self.role = role
            color = "#8DE055" if role == "user" else "#FFFFFF"
            self.frame.setStyleSheet(
                f"QFrame#messageBubble {{background: {color}; border-radius: 14px;}}"
            )
            self.body_layout.removeWidget(self.time_label)
            self.body_layout.insertWidget(
                0 if role == "user" else 1, self.time_label, 0, Qt.AlignmentFlag.AlignBottom
            )
            self.time_label.setAlignment(
                Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
            )
        self.speaker.setText(speaker)
        if self.body != body:
            self.body = body
            self.text.setHtml(body)

    def set_image(self, data: str | None) -> None:
        if data == self._image_data:
            return
        self._image_data = data
        self._image = QPixmap()
        self._scaled_image_size = None
        if data:
            try:
                buffer = QBuffer()
                buffer.setData(QByteArray(base64.b64decode(data, validate=True)))
                buffer.open(QIODevice.OpenModeFlag.ReadOnly)
                reader = QImageReader(buffer)
                size = reader.size()
                if (
                    bytes(reader.format()) == b"jpeg"
                    and size.isValid()
                    and max(size.width(), size.height()) <= 320
                ):
                    self._image = QPixmap.fromImage(reader.read())
            except ValueError:
                pass
        self.image_label.setVisible(not self._image.isNull())

    def fit_width(self, width: int) -> None:
        maximum = max(150, width)
        clock_width = 0 if self.time_label.isHidden() else 42
        content = (
            maximum,
            self.role,
            self.body,
            self._image_data,
            clock_width,
            self.speaker.text(),
            self.text.font().key(),
        )
        # 不変の履歴をストリームの各断片で再レイアウトしない。
        if content == self._fitted_content:
            return
        # 短文は必要な幅に収め、長文・URLは同じ吹き出しの中で折り返す。
        document = self.text.document()
        document.setTextWidth(maximum - 28 - clock_width)
        natural = math.ceil(document.idealWidth()) + 28
        if not self._image.isNull():
            natural = max(natural, min(240, self._image.width()) + 28)
        self.setFixedWidth(min(maximum, max(150, natural + clock_width)))
        document.setTextWidth(self.width() - 28 - clock_width)
        self.text.setFixedHeight(math.ceil(document.size().height()) + 4)
        image_height = 0
        if not self._image.isNull():
            size = self._image.size().scaled(
                QSize(
                    min(240, self.width() - 28 - clock_width, self._image.width()),
                    min(180, self._image.height()),
                ),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            if size != self._scaled_image_size:
                self._scaled_image_size = size
                self.image_label.setFixedSize(size)
                self.image_label.setPixmap(
                    self._image.scaled(
                        size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            image_height = size.height() + self.frame.layout().spacing()
        self.frame.setFixedHeight(self.text.height() + 18 + image_height)
        self.setFixedHeight(self.frame.height() + self.speaker.sizeHint().height() + 17)
        self._fitted_content = content


class ConversationView(QScrollArea):
    """更新中の最後の発話だけを描き直し、履歴のウィジェットを再利用する。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setObjectName("conversationView")
        self._content = QWidget()
        self._content.setObjectName("transcriptContent")
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(22, 20, 22, 20)
        self._layout.setSpacing(12)
        self._empty = QLabel("新しい会話です。下の欄に文章を入力してください。")
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet("color: #576A7D; padding: 12px;")
        self._layout.addWidget(self._empty)
        self._layout.addStretch()
        self._rows: list[tuple[QWidget, _MessageBubble, QHBoxLayout]] = []
        self._row_content: list[tuple | None] = []
        self._date_labels: list[QLabel] = []
        self._follow_tail = True
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._scroll_to_end)
        self.setWidget(self._content)
        self.verticalScrollBar().rangeChanged.connect(lambda *_: self._scroll_to_end())
        self.verticalScrollBar().valueChanged.connect(self._scroll_position_changed)

    def set_messages(
        self,
        messages: list[tuple[str, str, str]],
        *,
        timestamps: list[datetime | None] | None = None,
        images: list[str | None] | None = None,
    ) -> None:
        if not self._rows or not messages:
            self._follow_tail = True
        previous_date = None
        for index, (role, speaker, body) in enumerate(messages):
            if index == len(self._rows):
                row = QWidget()
                outer = QVBoxLayout(row)
                outer.setContentsMargins(0, 0, 0, 0)
                outer.setSpacing(12)
                date_label = QLabel()
                date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                date_label.setStyleSheet("color: #576A7D; font-size: 12px; padding: 6px;")
                outer.addWidget(date_label)
                layout = QHBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                bubble = _MessageBubble(row)
                layout.addWidget(bubble)
                outer.addLayout(layout)
                self._layout.insertWidget(self._layout.count() - 1, row)
                self._rows.append((row, bubble, layout))
                self._row_content.append(None)
                self._date_labels.append(date_label)
            image = images[index] if images is not None else None
            timestamp = timestamps[index] if timestamps is not None else None
            local = timestamp.astimezone() if timestamp is not None else None
            show_date = local is not None and local.date() != previous_date
            content = (role, speaker, body, image, timestamp, show_date)
            previous_date = local.date() if local is not None else None
            # 本文だけでなく日時・画像・日付境界が同じ行も再設定を避ける。
            if self._row_content[index] == content:
                continue
            self._row_content[index] = content
            _row, bubble, layout = self._rows[index]
            layout.setAlignment(
                Qt.AlignmentFlag.AlignRight if role == "user" else Qt.AlignmentFlag.AlignLeft
            )
            bubble.update_content(role, speaker, body)
            bubble.set_image(image)
            date_label = self._date_labels[index]
            date_label.setVisible(show_date)
            if local is not None:
                date_label.setText(f"{local:%Y/%m/%d}（{'月火水木金土日'[local.weekday()]}）")
                bubble.time_label.setText(f"{local:%H:%M}")
                bubble.time_label.setToolTip(local.isoformat(timespec="seconds"))
            bubble.time_label.setVisible(local is not None)
        while len(self._rows) > len(messages):
            row, _bubble, _layout = self._rows.pop()
            self._row_content.pop()
            self._date_labels.pop()
            self._layout.removeWidget(row)
            row.deleteLater()
        self._empty.setVisible(not messages)
        self._fit_messages()
        self._scroll_timer.start(0)

    def _fit_messages(self) -> None:
        width = max(150, round((self.viewport().width() - 44) * 0.88))
        for index, (row, bubble, _layout) in enumerate(self._rows):
            bubble.fit_width(width)
            label = self._date_labels[index]
            date_height = 0 if label.isHidden() else label.sizeHint().height() + 12
            row.setFixedHeight(bubble.height() + date_height)

    def _scroll_to_end(self) -> None:
        if self._follow_tail:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def _scroll_position_changed(self, value: int) -> None:
        # 読み返し中は更新で引き戻さず、末尾へ戻ったときだけ追随する。
        self._follow_tail = value >= self.verticalScrollBar().maximum() - 2

    def _follow_latest(self) -> None:
        self._follow_tail = True
        self._scroll_timer.start(0)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_messages()

    def toPlainText(self) -> str:  # noqa: N802
        if not self._rows:
            return self._empty.text()
        lines = []
        for index, (_row, bubble, _layout) in enumerate(self._rows):
            date = self._date_labels[index]
            if not date.isHidden():
                lines.append(date.text())
            lines.extend((bubble.speaker.text(), bubble.text.toPlainText()))
            if not bubble.time_label.isHidden():
                lines.append(bubble.time_label.text())
        return "\n".join(lines)

    def toHtml(self) -> str:  # noqa: N802
        return "\n".join(bubble.body for _row, bubble, _layout in self._rows)
