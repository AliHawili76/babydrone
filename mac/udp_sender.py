"""
Sends control packets to the Pi over UDP and reads back its status packets.
Just using the built-in socket module, nothing fancy.
"""

import json
import socket
import time
import uuid


class UdpLink:
    # how we keep the round-trip tracking dict from growing forever: drop
    # anything older than this many seconds, or once we've got more than
    # this many entries sitting around unmatched (e.g. if the pi stops
    # responding, we don't want this to just leak memory)
    _ROUND_TRIP_MAX_AGE_S = 5.0
    _ROUND_TRIP_MAX_ENTRIES = 200

    def __init__(self, pi_host, pi_port):
        self.pi_addr = (pi_host, pi_port)
        self.session = uuid.uuid4().hex[:8]
        self.sequence = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._last_status = None
        self._last_status_time = 0.0
        self._sent_times = {}  # sequence number -> when we sent it, so we can measure round-trip later
        self._last_round_trip_s = None

    def send(self, armed, emergency_stop, throttle, pitch, roll,
              right_gesture, left_gesture):
        packet = {
            "version": 1,
            "session": self.session,
            "sequence": self.sequence,
            "mac_time_ms": int(time.monotonic() * 1000),
            "armed": bool(armed),
            "emergency_stop": bool(emergency_stop),
            "throttle": float(throttle),
            "pitch": float(pitch),
            "roll": float(roll),
            "right_gesture": right_gesture,
            "left_gesture": left_gesture,
        }
        now = time.monotonic()
        self._sent_times[self.sequence] = now
        self._prune_sent_times(now)
        self.sequence += 1
        data = json.dumps(packet).encode("utf-8")
        try:
            self.sock.sendto(data, self.pi_addr)
        except OSError:
            pass  # network hiccup, just drop it — the pi's watchdog handles stale/missing packets
        return packet

    def _prune_sent_times(self, now):
        # first toss anything too old to matter anymore
        cutoff = now - self._ROUND_TRIP_MAX_AGE_S
        stale = [seq for seq, t in self._sent_times.items() if t < cutoff]
        for seq in stale:
            del self._sent_times[seq]
        # then if we're still over the cap, trim off the oldest until we're not
        overflow = len(self._sent_times) - self._ROUND_TRIP_MAX_ENTRIES
        if overflow > 0:
            for seq in sorted(self._sent_times)[:overflow]:
                del self._sent_times[seq]

    def poll_status(self):
        """Non-blocking check for a status packet from the pi. Returns the
        latest one we've seen, or None if nothing's ever come in (if
        nothing NEW arrived this call, we just return the last one again).

        Mac and Pi clocks aren't synced, so I can't just diff timestamps
        across the two machines to get latency. Instead, when a status
        packet echoes back a sequence_received number, I look up when I
        sent that sequence (all on the Mac's own clock) and that gap is
        the round trip time.
        """
        try:
            while True:
                data, _addr = self.sock.recvfrom(4096)
                try:
                    status = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                self._last_status = status
                self._last_status_time = time.monotonic()
                seq_received = status.get("sequence_received") if isinstance(status, dict) else None
                if seq_received is not None and seq_received in self._sent_times:
                    send_time = self._sent_times.pop(seq_received)
                    self._last_round_trip_s = time.monotonic() - send_time
        except BlockingIOError:
            pass  # nothing (more) to read right now, that's fine
        return self._last_status

    def get_last_round_trip_s(self):
        return self._last_round_trip_s

    def status_age_s(self):
        if self._last_status is None:
            return None
        return time.monotonic() - self._last_status_time

    def close(self):
        self.sock.close()
