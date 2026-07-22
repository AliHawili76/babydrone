"""
Listens for UDP packets from the Mac and checks they're actually valid
before trusting them.

validate_packet() is just a plain function (dict in, (ok, reason, packet)
out) — kept it separate from the socket stuff so it's easy to unit test
without needing an actual network connection.
"""

import json
import math
import socket

REQUIRED_FIELDS = {
    "version": int, "session": str, "sequence": int, "mac_time_ms": int,
    "armed": bool, "emergency_stop": bool, "throttle": (int, float),
    "pitch": (int, float), "roll": (int, float),
    "right_gesture": str, "left_gesture": str,
}
SUPPORTED_VERSION = 1


def validate_packet(packet: dict, *, last_sequence=None, current_session=None,
                     is_armed=False):
    """Returns (ok, reason, packet). If ok, the returned packet has its
    numeric fields clamped into valid range just in case."""
    if not isinstance(packet, dict):
        return False, "not a JSON object", None

    # basic shape check first — make sure every field we need is there and is the right type
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in packet:
            return False, f"missing field: {field}", None
        if not isinstance(packet[field], expected_type):
            return False, f"wrong type for field: {field}", None

    if packet["version"] != SUPPORTED_VERSION:
        return False, f"unsupported version: {packet['version']}", None

    # catch NaN/inf before they sneak into a range check below (NaN comparisons are always False,
    # so without this a NaN throttle would silently pass the 0..1 range check)
    for field in ("throttle", "pitch", "roll"):
        v = packet[field]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False, f"NaN/inf value in field: {field}", None

    if not (0.0 <= packet["throttle"] <= 1.0):
        return False, "throttle out of range", None
    if not (-1.0 <= packet["pitch"] <= 1.0):
        return False, "pitch out of range", None
    if not (-1.0 <= packet["roll"] <= 1.0):
        return False, "roll out of range", None

    # reject replayed/out-of-order packets
    if last_sequence is not None and packet["sequence"] <= last_sequence:
        return False, "old or duplicate sequence number", None

    # while armed, a sudden session change is suspicious (could be a stale/rogue sender) — block it
    if is_armed and current_session is not None and packet["session"] != current_session:
        return False, "unexpected session change while armed", None

    packet = dict(packet)
    packet["throttle"] = max(0.0, min(1.0, packet["throttle"]))
    packet["pitch"] = max(-1.0, min(1.0, packet["pitch"]))
    packet["roll"] = max(-1.0, min(1.0, packet["roll"]))
    return True, "ok", packet


class UdpReceiver:
    def __init__(self, listen_ip, listen_port, allowed_source_ip=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((listen_ip, listen_port))
        self.allowed_source_ip = allowed_source_ip
        self.last_sender_addr = None

    def poll(self):
        """Grabs everything currently waiting on the socket. Returns a list
        of (packet_or_None, addr, error) tuples."""
        results = []
        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                if self.allowed_source_ip and addr[0] != self.allowed_source_ip:
                    results.append((None, addr, "unauthorized source IP"))
                    continue
                try:
                    packet = json.loads(data.decode("utf-8"))
                    results.append((packet, addr, None))
                    self.last_sender_addr = addr
                except (json.JSONDecodeError, UnicodeDecodeError):
                    results.append((None, addr, "invalid JSON"))
        except BlockingIOError:
            pass
        return results

    def send_status(self, status: dict):
        if self.last_sender_addr is None:
            return
        data = json.dumps(status).encode("utf-8")
        try:
            self.sock.sendto(data, self.last_sender_addr)
        except OSError:
            pass

    def close(self):
        self.sock.close()
