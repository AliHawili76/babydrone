"""Tests for the log analysis script. Made up a tiny CSV by hand and worked
out what the confusion matrix/accuracy SHOULD be, then checked the script
actually gets those numbers. Same tempfile trick as test_mock_dac.py."""

import sys
import os
import csv
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from analyze_gesture_log import (  # noqa: E402
    compute_hand_metrics, load_logs, RIGHT_LABELS, LEFT_LABELS,
)


def _write_temp_log(rows):
    """rows are (target_right, right_stable, target_left, left_stable) tuples."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.writer(f)
    writer.writerow(["target_right_gesture", "right_stable", "target_left_gesture", "left_stable"])
    writer.writerows(rows)
    f.close()
    return f.name


# worked this out by hand: right hand ends up with 4 tagged rows (3 right, 1
# wrong), left hand with 3 tagged rows (2 right, 1 wrong). the row with NONE
# for both hands shouldn't count toward either one.
_ROWS = [
    ("THROTTLE_UP", "THROTTLE_UP", "NONE", "DIRECTION_NEUTRAL"),
    ("THROTTLE_UP", "THROTTLE_UP", "ROLL_LEFT", "ROLL_LEFT"),
    ("THROTTLE_UP", "THROTTLE_DOWN", "ROLL_LEFT", "ROLL_RIGHT"),
    ("THROTTLE_DOWN", "THROTTLE_DOWN", "ROLL_RIGHT", "ROLL_RIGHT"),
    ("NONE", "THROTTLE_UP", "NONE", "DIRECTION_NEUTRAL"),
]


def test_compute_hand_metrics_right_hand_known_example():
    path = _write_temp_log(_ROWS)
    try:
        df = load_logs([path])
        metrics = compute_hand_metrics(df, "target_right_gesture", "right_stable", RIGHT_LABELS)
    finally:
        os.unlink(path)

    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.75

    up_idx = RIGHT_LABELS.index("THROTTLE_UP")
    down_idx = RIGHT_LABELS.index("THROTTLE_DOWN")
    cm = metrics["confusion_matrix"]
    assert cm[up_idx][up_idx] == 2      # UP correctly predicted UP twice
    assert cm[up_idx][down_idx] == 1    # UP mistakenly predicted DOWN once
    assert cm[down_idx][down_idx] == 1  # DOWN correctly predicted DOWN once
    assert cm.sum() == 4


def test_compute_hand_metrics_left_hand_known_example():
    path = _write_temp_log(_ROWS)
    try:
        df = load_logs([path])
        metrics = compute_hand_metrics(df, "target_left_gesture", "left_stable", LEFT_LABELS)
    finally:
        os.unlink(path)

    assert metrics["n"] == 3
    assert abs(metrics["accuracy"] - (2 / 3)) < 1e-9

    left_idx = LEFT_LABELS.index("ROLL_LEFT")
    right_idx = LEFT_LABELS.index("ROLL_RIGHT")
    cm = metrics["confusion_matrix"]
    assert cm[left_idx][left_idx] == 1    # LEFT correctly predicted LEFT once
    assert cm[left_idx][right_idx] == 1   # LEFT mistakenly predicted RIGHT once
    assert cm[right_idx][right_idx] == 1  # RIGHT correctly predicted RIGHT once
    assert cm.sum() == 3


def test_compute_hand_metrics_returns_none_when_no_ground_truth():
    path = _write_temp_log([("NONE", "THROTTLE_UP", "NONE", "DIRECTION_NEUTRAL")])
    try:
        df = load_logs([path])
        metrics = compute_hand_metrics(df, "target_right_gesture", "right_stable", RIGHT_LABELS)
    finally:
        os.unlink(path)

    assert metrics is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
