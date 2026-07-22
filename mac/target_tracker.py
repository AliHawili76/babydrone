"""Tracks how many frames it's been since the ground-truth target gesture
(set by the keyboard controls in main.py) last changed. Used to filter out
the first ~1s of frames right after a switch, since the tester is probably
still moving into the new gesture rather than holding it cleanly."""


class TargetChangeTracker:
    def __init__(self):
        self._prev_right = None
        self._prev_left = None
        self.frames_since_change = 0

    def update(self, target_right, target_left):
        """Call once per frame with the current target_right/target_left.
        Returns 0 on the frame either one changes, then counts up by 1
        each frame after that until the next change."""
        if target_right != self._prev_right or target_left != self._prev_left:
            self.frames_since_change = 0
        else:
            self.frames_since_change += 1
        self._prev_right = target_right
        self._prev_left = target_left
        return self.frames_since_change
