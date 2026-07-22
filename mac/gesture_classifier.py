"""
Gesture classification stuff.

These functions just take a list of hand landmarks and spit out a state
string (like "THROTTLE_UP"). Kept them pure (no camera/mediapipe calls
inside) so I could actually unit test this without a webcam. Landmarks
just need .x/.y, so tests fake them with SimpleNamespace instead of real
MediaPipe objects.
"""

import math
import time
from dataclasses import dataclass

# --- Right hand states ---
THROTTLE_UP = "THROTTLE_UP"
THROTTLE_DOWN = "THROTTLE_DOWN"
THROTTLE_HOLD = "THROTTLE_HOLD"
RIGHT_UNKNOWN = "RIGHT_UNKNOWN"
RIGHT_MISSING = "RIGHT_MISSING"

# --- Left hand states ---
ROLL_LEFT = "ROLL_LEFT"
ROLL_RIGHT = "ROLL_RIGHT"
PITCH_FORWARD = "PITCH_FORWARD"
PITCH_BACKWARD = "PITCH_BACKWARD"
DIRECTION_NEUTRAL = "DIRECTION_NEUTRAL"
LEFT_UNKNOWN = "LEFT_UNKNOWN"
LEFT_MISSING = "LEFT_MISSING"

# base -> tip landmark indices for the 4 main fingers (borrowed the numbering
# from the reference repo so I didn't have to relearn the hand landmark map)
_FINGER_BASE_TIP = {
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2


def _angle_from_vertical(dx, dy):
    """0 = pointing straight up, 180 = pointing straight down."""
    return math.degrees(math.atan2(abs(dx), -dy))


def _is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    # tip above pip above mcp (smaller y = higher up on screen) = finger is out straight
    tip, pip, mcp = landmarks[tip_idx], landmarks[pip_idx], landmarks[mcp_idx]
    return tip.y < pip.y < mcp.y


# pip indices for the 4 main fingers, needed for the extended/folded checks above
_FINGER_PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
_FINGER_MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_FINGER_TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}


def _finger_openness(landmarks):
    """Fist-or-not check that doesn't care which way the hand is facing.

    Basically: how far are the fingertips from the wrist compared to how
    far the knuckles are? A curled-up fist has tips close to the wrist no
    matter which way it's rotated. I tried just checking "tip above pip
    above mcp" first but that only catches fingers pointing up and thinks
    a fingers-down pose is a fist too, so this ratio thing was the fix.
    """
    wrist = landmarks[WRIST]
    tip_total, mcp_total = 0.0, 0.0
    for base_idx, tip_idx in _FINGER_BASE_TIP.values():
        mcp, tip = landmarks[base_idx], landmarks[tip_idx]
        tip_total += math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        mcp_total += math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
    return tip_total / max(mcp_total, 1e-6)


def classify_right_hand(landmarks, angle_threshold_deg=45.0, openness_threshold=1.3):
    """Right hand controls throttle: up/down/hold. Thumb doesn't count here."""
    up_count, down_count = 0, 0

    for base_idx, tip_idx in _FINGER_BASE_TIP.values():
        base, tip = landmarks[base_idx], landmarks[tip_idx]
        dx, dy = tip.x - base.x, tip.y - base.y
        ang = _angle_from_vertical(dx, dy)
        if ang <= angle_threshold_deg:
            up_count += 1
        elif ang >= (180 - angle_threshold_deg):
            down_count += 1

    # fist check wins over everything else
    if _finger_openness(landmarks) < openness_threshold:
        return THROTTLE_HOLD
    if up_count == 4:
        return THROTTLE_UP
    if down_count == 4:
        return THROTTLE_DOWN
    return RIGHT_UNKNOWN


def classify_left_hand(landmarks, thumb_angle_threshold_deg=45.0):
    """Left hand controls roll/pitch. Checked in this order:
    1. open palm -> neutral
    2. just index finger out -> forward
    3. index + middle out -> backward
    4. everything folded except thumb, thumb sticking out sideways -> roll
    5. anything else -> neutral (safe default)
    """
    index_ext = _is_finger_extended(landmarks, _FINGER_TIP["index"], _FINGER_PIP["index"], _FINGER_MCP["index"])
    middle_ext = _is_finger_extended(landmarks, _FINGER_TIP["middle"], _FINGER_PIP["middle"], _FINGER_MCP["middle"])
    ring_ext = _is_finger_extended(landmarks, _FINGER_TIP["ring"], _FINGER_PIP["ring"], _FINGER_MCP["ring"])
    pinky_ext = _is_finger_extended(landmarks, _FINGER_TIP["pinky"], _FINGER_PIP["pinky"], _FINGER_MCP["pinky"])

    wrist = landmarks[WRIST]
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    thumb_dx = thumb_tip.x - wrist.x
    thumb_dy = thumb_tip.y - wrist.y
    # thumb "extended" = tip is noticeably farther from the knuckle than the middle joint is
    thumb_extended = math.hypot(thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y) > \
        math.hypot(landmarks[THUMB_IP].x - thumb_mcp.x, landmarks[THUMB_IP].y - thumb_mcp.y) * 0.8

    # 1. open palm -> neutral
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return DIRECTION_NEUTRAL

    # 2. just the index finger -> forward
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return PITCH_FORWARD

    # 3. index + middle -> backward
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return PITCH_BACKWARD

    # 4. everything folded but the thumb sticking out sideways -> roll
    all_main_folded = not index_ext and not middle_ext and not ring_ext and not pinky_ext
    if all_main_folded and thumb_extended:
        thumb_angle = math.degrees(math.atan2(abs(thumb_dy), abs(thumb_dx)))
        if thumb_angle <= thumb_angle_threshold_deg:
            return ROLL_LEFT if thumb_dx < 0 else ROLL_RIGHT

    # 5. couldn't match anything -> just say neutral, safer than guessing wrong
    return DIRECTION_NEUTRAL


def right_hand_diagnostics(landmarks, angle_threshold_deg=45.0, openness_threshold=1.3):
    """Debug/logging helper for the right hand — doesn't affect classification
    or control at all, just reports how close the hand was to each threshold
    so I can plot it later. Not used anywhere in the actual flight logic.

    margin_deg is basically "how far from flipping". classify_right_hand
    needs all 4 fingers to agree before it calls it UP or DOWN, so I take
    the worst (smallest) margin out of the 4 fingers toward whichever side
    has more votes. Positive = comfortably past the threshold, negative =
    at least one finger hasn't crossed it yet.
    """
    angles = {}
    up_count, down_count = 0, 0
    for name, (base_idx, tip_idx) in _FINGER_BASE_TIP.items():
        base, tip = landmarks[base_idx], landmarks[tip_idx]
        dx, dy = tip.x - base.x, tip.y - base.y
        ang = _angle_from_vertical(dx, dy)
        angles[name] = ang
        if ang <= angle_threshold_deg:
            up_count += 1
        elif ang >= (180 - angle_threshold_deg):
            down_count += 1

    if up_count >= down_count:
        margin = angle_threshold_deg - max(angles.values())
    else:
        margin = min(angles.values()) - (180 - angle_threshold_deg)

    openness = _finger_openness(landmarks)

    return {
        "finger_angles_deg": angles,
        "up_count": up_count,
        "down_count": down_count,
        "margin_deg": margin,
        "openness_ratio": openness,
        "openness_margin": openness_threshold - openness,
    }


def left_hand_diagnostics(landmarks, thumb_angle_threshold_deg=45.0):
    """Same idea as right_hand_diagnostics but for the left hand's roll
    check — logging/debug only, doesn't touch classify_left_hand's output.

    thumb_margin_deg = threshold minus the thumb's angle from horizontal.
    Positive means it's solidly in "roll" territory, negative means it's
    tilted too far toward vertical to register.
    """
    index_ext = _is_finger_extended(landmarks, _FINGER_TIP["index"], _FINGER_PIP["index"], _FINGER_MCP["index"])
    middle_ext = _is_finger_extended(landmarks, _FINGER_TIP["middle"], _FINGER_PIP["middle"], _FINGER_MCP["middle"])
    ring_ext = _is_finger_extended(landmarks, _FINGER_TIP["ring"], _FINGER_PIP["ring"], _FINGER_MCP["ring"])
    pinky_ext = _is_finger_extended(landmarks, _FINGER_TIP["pinky"], _FINGER_PIP["pinky"], _FINGER_MCP["pinky"])

    wrist = landmarks[WRIST]
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    thumb_dx = thumb_tip.x - wrist.x
    thumb_dy = thumb_tip.y - wrist.y
    thumb_extended = math.hypot(thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y) > \
        math.hypot(landmarks[THUMB_IP].x - thumb_mcp.x, landmarks[THUMB_IP].y - thumb_mcp.y) * 0.8
    thumb_angle = math.degrees(math.atan2(abs(thumb_dy), abs(thumb_dx)))

    return {
        "index_extended": index_ext,
        "middle_extended": middle_ext,
        "ring_extended": ring_ext,
        "pinky_extended": pinky_ext,
        "all_main_folded": not index_ext and not middle_ext and not ring_ext and not pinky_ext,
        "thumb_extended": thumb_extended,
        "thumb_angle_from_horizontal_deg": thumb_angle,
        "thumb_margin_deg": thumb_angle_threshold_deg - thumb_angle,
    }


@dataclass
class DebouncedState:
    stable: str
    candidate: str = None
    candidate_since: float = 0.0


class GestureDebouncer:
    """Stops the gesture from flickering — a new reading only "counts" once
    it's stayed the same for debounce_ms. Without this the raw classifier
    output jitters between states basically every frame."""

    def __init__(self, initial_state, debounce_ms=125):
        self.debounce_s = debounce_ms / 1000.0
        self._state = DebouncedState(stable=initial_state)

    def update(self, raw_gesture, now=None):
        now = now if now is not None else time.monotonic()
        s = self._state
        if raw_gesture == s.stable:
            # already what we think it is, nothing to do
            s.candidate = None
            s.candidate_since = 0.0
            return s.stable

        if raw_gesture != s.candidate:
            # new candidate reading, start the clock on it
            s.candidate = raw_gesture
            s.candidate_since = now
            return s.stable

        # same candidate as last time, check if it's been long enough yet
        if now - s.candidate_since >= self.debounce_s:
            s.stable = raw_gesture
            s.candidate = None
            s.candidate_since = 0.0

        return s.stable

    @property
    def stable(self):
        return self._state.stable
