"""Tests for TargetChangeTracker — checks the frames-since-change counter
resets on a switch and counts up correctly in between."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mac"))

from target_tracker import TargetChangeTracker  # noqa: E402


def test_starts_at_zero_on_first_frame():
    tracker = TargetChangeTracker()
    assert tracker.update("THROTTLE_UP", "NONE") == 0


def test_counts_up_while_targets_stay_the_same():
    tracker = TargetChangeTracker()
    tracker.update("THROTTLE_UP", "NONE")
    assert tracker.update("THROTTLE_UP", "NONE") == 1
    assert tracker.update("THROTTLE_UP", "NONE") == 2
    assert tracker.update("THROTTLE_UP", "NONE") == 3


def test_resets_to_zero_when_right_target_changes():
    tracker = TargetChangeTracker()
    tracker.update("THROTTLE_UP", "NONE")
    tracker.update("THROTTLE_UP", "NONE")
    assert tracker.update("THROTTLE_DOWN", "NONE") == 0
    assert tracker.update("THROTTLE_DOWN", "NONE") == 1


def test_resets_to_zero_when_left_target_changes_even_if_right_is_unchanged():
    tracker = TargetChangeTracker()
    tracker.update("THROTTLE_UP", "NONE")
    tracker.update("THROTTLE_UP", "NONE")
    assert tracker.update("THROTTLE_UP", "ROLL_LEFT") == 0
    assert tracker.update("THROTTLE_UP", "ROLL_LEFT") == 1


def test_switching_back_and_forth_resets_each_time():
    tracker = TargetChangeTracker()
    tracker.update("THROTTLE_UP", "NONE")
    tracker.update("THROTTLE_UP", "NONE")
    tracker.update("THROTTLE_DOWN", "NONE")  # switch #1, back to 0
    assert tracker.update("THROTTLE_UP", "NONE") == 0  # switch #2, back to 0 again
    assert tracker.update("THROTTLE_UP", "NONE") == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
