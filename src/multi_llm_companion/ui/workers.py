"""時間のかかるLLM通信を画面の外で動かす。"""

from __future__ import annotations

import threading

from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

from multi_llm_companion.application import (
    ProactiveTurnInvalidatedError,
    StreamTurnInvalidatedError,
)
from multi_llm_companion.services.image_sharing import PreparedImage
from multi_llm_companion.ui.models import ChatBackend, SendOutcome


class StreamWorker(QObject):
    """LLM通信中も画面が固まらないようにする作業役。"""

    chunk_received = Signal(int, str)
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)
    finished = Signal()

    def __init__(
        self,
        backend: ChatBackend,
        conversation_id: str,
        user_text: str,
        generation: int,
        image: PreparedImage | None = None,
    ) -> None:
        super().__init__()
        self._owner_thread = self.thread()
        self._backend = backend
        self._conversation_id = conversation_id
        self._user_text = user_text
        self._generation = generation
        self._image = image
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            outcome: SendOutcome = self._backend.stream_message(
                self._conversation_id,
                self._user_text,
                lambda chunk: self.chunk_received.emit(self._generation, chunk),
                cancelled=self._cancel_requested.is_set,
                **({"image": self._image} if self._image is not None else {}),
            )
            self.succeeded.emit(self._generation, outcome)
        except StreamTurnInvalidatedError:
            self.cancelled.emit(self._generation)
        except Exception as exc:  # 画面には安全な日本語だけを渡す
            message = str(exc).strip() or "LLMとの通信中に不明なエラーが発生しました。"
            self.failed.emit(self._generation, message)
        finally:
            self._image = None
            # Qtの接続解除とGUI更新がGILを挟んで待ち合わないよう、作成元で破棄する。
            self.moveToThread(self._owner_thread)
            self.finished.emit()


class ProactiveWorker(QObject):
    """利用者操作を止めずに、能動発話だけを画面外で生成する。"""

    succeeded = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)
    finished = Signal()

    def __init__(
        self,
        backend: ChatBackend,
        conversation_id: str,
        generation: int,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._conversation_id = conversation_id
        self._generation = generation
        self._cancel_requested = threading.Event()

    def cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        try:
            if self._cancel_requested.is_set():
                raise ProactiveTurnInvalidatedError()
            generate = getattr(self._backend, "_generate_proactive_turn", None)
            outcome = (
                generate(self._conversation_id, cancelled=self._cancel_requested.is_set)
                if callable(generate)
                else self._backend.generate_proactive_turn(self._conversation_id)
            )
            if self._cancel_requested.is_set():
                raise ProactiveTurnInvalidatedError()
            self.succeeded.emit(self._generation, outcome)
        except ProactiveTurnInvalidatedError:
            self.cancelled.emit(self._generation)
        except Exception as exc:  # 画面には安全な日本語だけを渡す
            message = str(exc).strip() or "能動発話の生成中に不明なエラーが発生しました。"
            self.failed.emit(self._generation, message)
        finally:
            # 並行する通常workerの回収後も、存続中のGUIスレッドへ戻す。
            self.moveToThread(QCoreApplication.instance().thread())
            self.finished.emit()


class VoiceStartWorker(QObject):
    """姿勢準備後のVoiSona受付を、画面スレッドの外で行う。"""

    completed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        backend: ChatBackend,
        text: str,
        generation: int,
        *,
        proactive: bool = False,
    ) -> None:
        super().__init__()
        self._owner_thread = self.thread()
        self._backend = backend
        self._text = text
        self._generation = generation
        self._proactive = proactive
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """開始前と受付中の両方で、古い要求を無効化する。"""

        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            if self._cancel_event.is_set():
                return
            guarded_start = getattr(self._backend, "_start_voice_output", None)
            if callable(guarded_start):
                result = guarded_start(
                    self._text,
                    reject_invalidated_stream=not self._proactive,
                    cancel_event=self._cancel_event,
                )
            else:
                start = (
                    self._backend.start_proactive_voice_output
                    if self._proactive
                    else self._backend.start_voice_output
                )
                result = start(self._text)
            self.completed.emit((self._generation, result))
        except Exception:  # 画面へ内部情報を露出しない
            self.completed.emit(
                (
                    self._generation,
                    "音声出力を開始できませんでした。設定を確認してください。",
                )
            )
        finally:
            self.moveToThread(self._owner_thread)
            self.finished.emit()
