"""口元ポーズの手首方向を実素材なしで検証する。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "author_cheek_idle",
    Path(__file__).resolve().parents[1] / "tools/motion-converter/author_cheek_idle.py",
)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)


@pytest.mark.parametrize("palm_sign", [-1, 1])
def test_hand_orientation_constrains_finger_and_palm(palm_sign: int) -> None:
    finger = np.array((1.0, 0.0, 0.0))
    palm = np.array((0.0, -1.0, 0.0))
    target_finger = np.array((-0.45, 1.0, 0.1))
    target_palm = np.array((-0.6, 0.0, 1.0)) * palm_sign
    q = author.orient_hand(finger, palm, target_finger, target_palm)
    expected_finger = target_finger / np.linalg.norm(target_finger)
    expected_palm = target_palm - expected_finger * (target_palm @ expected_finger)
    expected_palm /= np.linalg.norm(expected_palm)
    assert author.rotate(q, finger) == pytest.approx(expected_finger)
    assert author.rotate(q, palm) == pytest.approx(expected_palm)
    assert np.linalg.norm(q) == pytest.approx(1.0)
