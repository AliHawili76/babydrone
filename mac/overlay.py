"""Draws the HUD overlay on the preview window (gauges, badges, status text)."""

import cv2

try:
    from mediapipe.solutions.hands import HAND_CONNECTIONS
except Exception:  # pragma: no cover - fallback for older/newer MediaPipe builds
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

GREEN = (110, 220, 120)
RED = (80, 80, 250)
BLUE = (255, 180, 90)
GRAY = (180, 180, 180)
ORANGE = (70, 160, 255)
PANEL_FILL = (24, 24, 28)
PANEL_BORDER = (84, 90, 98)
TEXT_COLOR = (236, 240, 244)
TRACK_COLOR = (48, 48, 54)

SHOW_LANDMARKS = False

_THROTTLE_COLOR = {"THROTTLE_UP": GREEN, "THROTTLE_DOWN": RED}
_LEFT_COLOR = {
    "ROLL_LEFT": BLUE, "ROLL_RIGHT": BLUE,
    "PITCH_FORWARD": BLUE, "PITCH_BACKWARD": BLUE,
}


def _draw_rounded_panel(frame, x0, y0, x1, y1, fill, border, alpha=0.78, radius=14):
    """Draws a see-through rounded rectangle, used for the HUD panel and badges.

    Note: radius has to be well under half of (y1 - y0) or the corners
    start overlapping and it just looks like a solid bar. Learned this the
    hard way on the small badges before I made the radius smaller for those.
    """
    overlay = frame.copy()

    cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), fill, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), fill, -1, cv2.LINE_AA)
    cv2.ellipse(overlay, (x0 + radius, y0 + radius), (radius, radius), 180, 270, 360, fill, -1)
    cv2.ellipse(overlay, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, fill, -1)
    cv2.ellipse(overlay, (x0 + radius, y1 - radius), (radius, radius), 90, 180, 270, fill, -1)
    cv2.ellipse(overlay, (x1 - radius, y1 - radius), (radius, radius), 0, 90, 180, fill, -1)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x0 + radius, y0), (x1 - radius, y1), border, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0 + radius), (x1, y1 - radius), border, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (x0 + radius, y0 + radius), (radius, radius), 180, 270, 360, border, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, border, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (x0 + radius, y1 - radius), (radius, radius), 90, 180, 270, border, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 - radius, y1 - radius), (radius, radius), 0, 90, 180, border, 1, cv2.LINE_AA)


def _draw_bar_track(frame, x, y, width, height):
    """Just the empty background/border for a gauge bar, shared by both bar types below."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), TRACK_COLOR, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_BORDER, 1, cv2.LINE_AA)


def _draw_percent_bar(frame, x, y, value, width, height, color):
    """0-1 value bar (throttle) that fills left to right."""
    _draw_bar_track(frame, x, y, width, height)
    value = max(0.0, min(1.0, value))
    fill_w = int(round(value * width))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1, cv2.LINE_AA)


def _draw_centered_bar(frame, x, y, value, width, height, color):
    """-1 to 1 value bar (pitch/roll) that fills out from the middle instead of the left edge."""
    _draw_bar_track(frame, x, y, width, height)
    center_x = x + width // 2
    cv2.line(frame, (center_x, y), (center_x, y + height), (120, 120, 128), 1, cv2.LINE_AA)

    value = max(-1.0, min(1.0, value))
    fill_w = int(round(abs(value) * (width // 2)))
    if value >= 0:
        cv2.rectangle(frame, (center_x, y), (center_x + fill_w, y + height), color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (center_x - fill_w, y), (center_x, y + height), color, -1, cv2.LINE_AA)


def _draw_badge(frame, x, y, text, color, border):
    """Little pill-shaped badge, makes state stuff easier to spot at a glance."""
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
    pad_x = 12
    pad_y = 8
    x0 = x - pad_x
    y0 = y - text_h - pad_y
    x1 = x + text_w + pad_x
    y1 = y + pad_y
    badge_radius = min(14, max(4, (y1 - y0) // 3))
    _draw_rounded_panel(frame, x0, y0, x1, y1, fill=color, border=border, alpha=0.95, radius=badge_radius)
    cv2.putText(frame, text, (x0 + pad_x // 2, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_hand_skeleton(frame, landmarks, color):
    """Draws the little skeleton lines/dots over a hand, using MediaPipe's connection list."""
    if not landmarks or len(landmarks) < 21:
        return

    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        if not hasattr(landmark, "x") or not hasattr(landmark, "y"):
            return
        points.append((int(landmark.x * width), int(landmark.y * height)))

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)


def draw_overlay(frame, *, right_hand_present, left_hand_present,
                  right_raw, right_stable, left_raw, left_stable,
                  throttle, pitch, roll, state, pi_connected,
                  status_age_s, fps, right_raw_label=None, left_raw_label=None,
                  round_trip_s=None):
    """Draws the whole HUD panel (dark box + gauges + badges + text).

    This works in two passes: first I build up a list of "rows" (text,
    gauge, or badge) and walk through them once just to figure out where
    everything goes and how big the panel needs to be. THEN I draw the
    panel background sized to fit, and THEN draw the rows on top in the
    same positions. Doing it this way (one shared y-cursor moving down for
    every row, no matter what kind it is) means rows can't ever overlap
    and the panel is never too small — had an annoying bug earlier where
    stuff was overlapping before I switched to this approach.
    """
    rows = []

    def text_row(text, color=TEXT_COLOR, font_scale=0.55):
        rows.append({"kind": "text", "text": text, "color": color, "font_scale": font_scale})

    def gauge_row(label, value, color, mode):
        rows.append({"kind": "gauge", "label": label, "value": value, "color": color, "mode": mode})

    def badge_row(text, fill, border):
        rows.append({"kind": "badge", "text": text, "fill": fill, "border": border})

    # ---- what goes in the panel, top to bottom ----
    right_color = GREEN if right_hand_present else ORANGE
    right_status = f"Right: {'detected' if right_hand_present else 'missing'}"
    if right_hand_present and right_raw_label is not None:
        # if mediapipe's own guess disagrees with the role we assigned, that's
        # continuity correction kicking in — worth seeing live during testing
        right_status += f" (raw: {right_raw_label})" if right_raw_label == "Right" \
            else f" (raw: {right_raw_label}, corrected)"
    text_row(right_status, right_color)
    text_row(f"raw={right_raw}  stable={right_stable}", _THROTTLE_COLOR.get(right_stable, GRAY))

    left_color = GREEN if left_hand_present else ORANGE
    left_status = f"Left: {'detected' if left_hand_present else 'missing'}"
    if left_hand_present and left_raw_label is not None:
        left_status += f" (raw: {left_raw_label})" if left_raw_label == "Left" \
            else f" (raw: {left_raw_label}, corrected)"
    text_row(left_status, left_color)
    text_row(f"raw={left_raw}  stable={left_stable}", _LEFT_COLOR.get(left_stable, GRAY))

    gauge_row("Throttle", throttle, GREEN, mode="percent")
    gauge_row("Pitch", pitch, BLUE, mode="centered")
    gauge_row("Roll", roll, BLUE, mode="centered")

    state_color = {
        "ARMED": GREEN,
        "DISARMED": GRAY,
        "FAILSAFE": ORANGE,
        "EMERGENCY_STOP": RED,
    }.get(state, GRAY)
    badge_row(f"STATE: {state}", state_color, (255, 255, 255))
    badge_row(f"PI: {'CONNECTED' if pi_connected else 'NO STATUS'}", GREEN if pi_connected else RED, (255, 255, 255))

    if round_trip_s is not None:
        text_row(f"Round-trip: {round_trip_s * 1000:.0f} ms", GRAY, font_scale=0.45)

    if status_age_s is not None:
        text_row(f"Last PI status: {status_age_s * 1000:.0f} ms ago", GRAY, font_scale=0.45)
    text_row(f"FPS: {fps:.1f}", GRAY, font_scale=0.5)

    # ---- pass 1: figure out where each row goes and how big everything is ----
    margin_x = 24
    margin_top = 24
    bottom_margin = 16
    right_margin = 16
    text_row_h = 24
    gauge_label_h = 18
    gauge_bar_h = 12
    gauge_bar_gap = 8   # space between the bar and the value text under it
    gauge_value_h = 24
    bar_width = 140
    badge_gap = 8       # extra space after a badge so the next row isn't crammed in

    y = margin_top
    max_content_right = margin_x
    layout = []  # (row, y_top) pairs, remembering where pass 1 put each row

    for row in rows:
        layout.append((row, y))
        if row["kind"] == "text":
            (tw, _), _ = cv2.getTextSize(row["text"], cv2.FONT_HERSHEY_DUPLEX, row["font_scale"], 1)
            max_content_right = max(max_content_right, margin_x + tw)
            y += text_row_h
        elif row["kind"] == "gauge":
            value_text = f"{row['value'] * 100:.0f}%" if row["mode"] == "percent" else f"{row['value']:+.2f}"
            (label_w, _), _ = cv2.getTextSize(row["label"], cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            (value_w, _), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            max_content_right = max(max_content_right, margin_x + bar_width, margin_x + label_w, margin_x + value_w)
            row["_value_text"] = value_text
            y += gauge_label_h + gauge_bar_h + gauge_bar_gap + gauge_value_h
        elif row["kind"] == "badge":
            (text_w, text_h), _ = cv2.getTextSize(row["text"], cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            badge_w = text_w + 24
            row["_text_h"] = text_h
            max_content_right = max(max_content_right, margin_x + badge_w)
            y += text_h + 16 + badge_gap

    content_bottom = y

    # ---- now we know how big the panel needs to be, so draw it ----
    panel_x0, panel_y0 = 12, 12
    panel_x1 = max_content_right + right_margin
    panel_y1 = content_bottom + bottom_margin
    _draw_rounded_panel(frame, panel_x0, panel_y0, panel_x1, panel_y1, PANEL_FILL, PANEL_BORDER, alpha=0.78)

    # ---- pass 2: actually draw everything at the positions from pass 1 ----
    for row, y_top in layout:
        if row["kind"] == "text":
            baseline_y = y_top + 16
            cv2.putText(frame, row["text"], (margin_x, baseline_y), cv2.FONT_HERSHEY_DUPLEX,
                        row["font_scale"], row["color"], 1, cv2.LINE_AA)
        elif row["kind"] == "gauge":
            label_y = y_top + 14
            cv2.putText(frame, row["label"], (margin_x, label_y), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            bar_y = y_top + gauge_label_h
            if row["mode"] == "percent":
                _draw_percent_bar(frame, margin_x, bar_y, row["value"], bar_width, gauge_bar_h, row["color"])
            else:
                _draw_centered_bar(frame, margin_x, bar_y, row["value"], bar_width, gauge_bar_h, row["color"])
            value_y = bar_y + gauge_bar_h + gauge_bar_gap + 14
            cv2.putText(frame, row["_value_text"], (margin_x, value_y), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, row["color"], 1, cv2.LINE_AA)
        elif row["kind"] == "badge":
            baseline_y = y_top + row["_text_h"] + 8
            _draw_badge(frame, margin_x + 12, baseline_y, row["text"], row["fill"], row["border"])
