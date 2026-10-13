"""PMX描画で、Python起動時と別ビューのGC固定を区別する。"""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from multi_llm_companion.ui import dense_pmx_widget as renderer_module


@pytest.mark.parametrize(
    ("baseline", "frozen", "expected"),
    [(0, 0, True), (377, 377, True), (377, 5000, False), (0, 5000, False)],
)
def test_runtime_frozen_objects_do_not_block_pmx_gc_guard(monkeypatch, baseline, frozen, expected):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    renderer = renderer_module.DensePmxWidget()
    renderer._preview = SimpleNamespace(dense_textured_triangles=(None,) * 50_000)
    renderer._physics = SimpleNamespace(is_initialized=True)
    calls = []
    monkeypatch.setattr(renderer_module, "_INITIAL_GC_FREEZE_COUNT", baseline)
    monkeypatch.setattr(renderer_module.gc, "get_freeze_count", lambda: frozen)
    monkeypatch.setattr(renderer_module.gc, "collect", lambda: 0)
    monkeypatch.setattr(renderer_module.gc, "freeze", lambda: calls.append("freeze"))
    monkeypatch.setattr(renderer_module.gc, "unfreeze", lambda: calls.append("unfreeze"))
    try:
        renderer._freeze_long_lived_pmx_graph()
        renderer._freeze_long_lived_pmx_graph()
        assert renderer._gc_freeze_owned is expected
        renderer._release_gc_freeze()
        assert calls == (["freeze", "unfreeze"] if expected else [])
    finally:
        renderer._physics = SimpleNamespace(reset=lambda _preview: None)
        renderer.shutdown()
