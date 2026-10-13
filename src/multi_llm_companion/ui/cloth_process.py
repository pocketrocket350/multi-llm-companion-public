"""布の計算世界だけを専用プロセスへ置く、アプリ内の接続。"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from multiprocessing import Pipe

import numpy as np
from panda3d.core import TransformState

from multi_llm_companion.ui.independent_cloth import ClothNonFiniteError, SkirtClothSimulation
from multi_llm_companion.ui.models import UiAvatarModelPreview
from multi_llm_companion.ui.pmx_pose import BoneTransform

_RESPONSE_TIMEOUT = 5.0


class ProcessClothSimulation:
    """一つの姿勢ワーカーから順番に使う、別プロセスの布計算。"""

    @classmethod
    def create(
        cls, preview: UiAvatarModelPreview, points: np.ndarray, transforms: dict[int, BoneTransform]
    ) -> ProcessClothSimulation | None:
        names = {b.name for b in preview.bones}
        if "下半身" not in names or not any(name.startswith("Skirt_") for name in names):
            return None
        self = cls()
        try:
            self._connection.send((preview, points, transforms))
            metadata = self._receive(timeout=20.0)
            if metadata is None:
                self.close()
                return None
            self.vertex_indexes, self.bone_indexes, self.render_bone_indexes = metadata
            self.vertex_indexes.setflags(write=False)
            self._positions = points[self.vertex_indexes].copy()
            self._time = None
            self._pending_time = None
            return self
        except BaseException:
            self.close()
            raise

    def __init__(self):
        parent, child = Pipe(duplex=True)
        self._connection = parent
        self._process = None
        self._closed = False
        handle = child.fileno()
        options = {}
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {"handle_list": [handle]}
            options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
            os.set_handle_inheritable(handle, True)
        else:
            options["pass_fds"] = (handle,)
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "multi_llm_companion.ui.cloth_process", str(handle)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                close_fds=True,
                **options,
            )
        except BaseException:
            parent.close()
            raise
        finally:
            if os.name == "nt":
                os.set_handle_inheritable(handle, False)
            child.close()

    def _receive(self, *, timeout=_RESPONSE_TIMEOUT):
        if not self._connection.poll(timeout):
            raise RuntimeError("布計算プロセスから応答がありません。")
        try:
            status, payload = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("布計算プロセスが応答前に終了しました。") from exc
        if status == "nonfinite":
            raise ClothNonFiniteError(f"布計算プロセスで数値失敗しました: {payload}")
        if status == "error":
            raise RuntimeError(f"布計算プロセスで失敗しました: {payload}")
        return payload

    def _begin_advance(self, seconds: float, transforms: dict[int, BoneTransform]) -> None:
        if self._closed:
            raise RuntimeError("終了した布計算プロセスは更新できません。")
        if self._time is not None and seconds <= self._time:
            return
        if self._pending_time is not None:
            if seconds != self._pending_time:
                raise RuntimeError("布計算の応答前に別の姿勢を送信できません。")
            return
        try:
            self._connection.send((seconds, {i: transforms[i] for i in self.bone_indexes}))
            self._pending_time = seconds
        except (EOFError, OSError) as exc:
            self.close()
            raise RuntimeError(f"布の姿勢送信を停止しました: {exc}") from exc

    def advance(self, seconds: float, transforms: dict[int, BoneTransform]) -> np.ndarray:
        if self._time is not None and seconds <= self._time and not self._closed:
            return self._positions
        self._begin_advance(seconds, transforms)
        if self._pending_time is None:
            return self._positions
        try:
            self._positions = self._receive()
            self._time = self._pending_time
            self._pending_time = None
            return self._positions
        except ClothNonFiniteError:
            self.close()
            raise
        except (RuntimeError, EOFError, OSError) as exc:
            self.close()
            raise RuntimeError(f"布の更新を停止しました: {exc}") from exc

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None and self._process.poll() is None:
                with suppress(EOFError, OSError):
                    self._connection.send(None)
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            self._connection.close()


def _serve(connection):
    cloth = None
    try:
        preview, points, transforms = connection.recv()
        cloth = SkirtClothSimulation.create(preview, points, transforms)
        del preview, points, transforms
        if cloth is None:
            connection.send(("ok", None))
            return
        connection.send(
            ("ok", (cloth.vertex_indexes, cloth.bone_indexes, cloth.render_bone_indexes))
        )
        while True:
            request = connection.recv()
            if request is None:
                break
            seconds, transforms = request
            try:
                positions = cloth.advance(seconds, transforms)
            except ClothNonFiniteError as exc:
                connection.send(("nonfinite", str(exc)))
                return
            # ShowBaseのない子プロセスでも、不要になった変換状態を毎更新回収する。
            TransformState.garbage_collect()
            connection.send(("ok", positions))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if cloth is not None:
            cloth.close()
        connection.close()


if __name__ == "__main__":
    if os.name == "nt":
        from multiprocessing.connection import PipeConnection as Connection
    else:
        from multiprocessing.connection import Connection
    _serve(Connection(int(sys.argv[1])))
