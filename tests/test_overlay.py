import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mac"))

import overlay  # noqa: E402


def make_landmarks():
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]


def test_draw_hand_skeleton_draws_on_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = make_landmarks()

    overlay.draw_hand_skeleton(frame, landmarks, (255, 255, 255))

    assert np.count_nonzero(frame) > 0
