"""Tests for UdpLink's round-trip time tracking. Uses a real loopback UDP
socket to stand in for the pi (rather than mocking), so this exercises the
actual send()/poll_status() code path end to end."""

import sys
import os
import json
import socket
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mac"))

from udp_sender import UdpLink  # noqa: E402


def _make_fake_pi():
    """A plain UDP socket bound to localhost, standing in for the pi so
    send() has somewhere real to deliver to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    return sock, sock.getsockname()[1]


def test_round_trip_measured_from_matching_status_packet():
    fake_pi, fake_pi_port = _make_fake_pi()
    link = UdpLink("127.0.0.1", fake_pi_port)
    try:
        assert link.get_last_round_trip_s() is None  # nothing sent/received yet

        packet = link.send(
            armed=False, emergency_stop=False, throttle=0.0, pitch=0.0, roll=0.0,
            right_gesture="THROTTLE_HOLD", left_gesture="DIRECTION_NEUTRAL",
        )
        sender_port = link.sock.getsockname()[1]

        # pretend the pi got it instantly and echoed the sequence back
        status = json.dumps({"sequence_received": packet["sequence"]}).encode("utf-8")
        fake_pi.sendto(status, ("127.0.0.1", sender_port))
        time.sleep(0.05)  # give the loopback packet a moment to actually arrive

        link.poll_status()
        round_trip = link.get_last_round_trip_s()

        assert round_trip is not None
        assert 0.0 <= round_trip < 1.0  # should be near-instant on loopback
    finally:
        link.close()
        fake_pi.close()


def test_round_trip_ignores_status_with_unmatched_sequence():
    fake_pi, fake_pi_port = _make_fake_pi()
    link = UdpLink("127.0.0.1", fake_pi_port)
    try:
        link.send(
            armed=False, emergency_stop=False, throttle=0.0, pitch=0.0, roll=0.0,
            right_gesture="THROTTLE_HOLD", left_gesture="DIRECTION_NEUTRAL",
        )
        sender_port = link.sock.getsockname()[1]

        # a sequence number that was never sent shouldn't produce a round-trip reading
        status = json.dumps({"sequence_received": 9999}).encode("utf-8")
        fake_pi.sendto(status, ("127.0.0.1", sender_port))
        time.sleep(0.05)

        link.poll_status()
        assert link.get_last_round_trip_s() is None
    finally:
        link.close()
        fake_pi.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
