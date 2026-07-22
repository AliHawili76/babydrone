"""Loads calibration.json and converts -1..1 / 0..1 commands into actual DAC codes."""

import json


def map_bipolar_axis(command: float, negative_code: int, centre_code: int, positive_code: int) -> int:
    """command is -1.0..1.0. Maps it in two pieces (negative side and positive
    side separately around centre) so it still works if the ranges aren't
    symmetric or the wiring makes "positive" the smaller code number."""
    command = max(-1.0, min(1.0, command))
    if command >= 0:
        return round(centre_code + command * (positive_code - centre_code))
    return round(centre_code + (-command) * (negative_code - centre_code))


def map_throttle(command: float, minimum_code: int, maximum_code: int) -> int:
    # throttle is simpler than pitch/roll, just 0..1 straight to min..max, no centre point
    command = max(0.0, min(1.0, command))
    return round(minimum_code + command * (maximum_code - minimum_code))


class Calibration:
    def __init__(self, path):
        self.path = path
        self.data = self._load()

    def _load(self):
        with open(self.path) as f:
            return json.load(f)

    def reload(self):
        self.data = self._load()

    def is_complete(self):
        """Just checks nobody left a null in there — i.e. did we actually fill in real values yet."""
        for axis in self.data.values():
            for v in axis.values():
                if v is None:
                    return False
        return True

    def throttle_code(self, command: float) -> int:
        t = self.data["throttle"]
        if t["minimum_code"] is None or t["maximum_code"] is None:
            raise ValueError("throttle calibration incomplete")
        return map_throttle(command, t["minimum_code"], t["maximum_code"])

    def pitch_code(self, command: float) -> int:
        p = self.data["pitch"]
        if None in (p["backward_code"], p["centre_code"], p["forward_code"]):
            raise ValueError("pitch calibration incomplete")
        return map_bipolar_axis(command, p["backward_code"], p["centre_code"], p["forward_code"])

    def roll_code(self, command: float) -> int:
        r = self.data["roll"]
        if None in (r["left_code"], r["centre_code"], r["right_code"]):
            raise ValueError("roll calibration incomplete")
        return map_bipolar_axis(command, r["left_code"], r["centre_code"], r["right_code"])
