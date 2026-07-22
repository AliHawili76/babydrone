"""
The watchdog/failsafe logic — this is the Pi's own safety net, it doesn't
just trust whatever the Mac tells it. If packets stop arriving (timeout_ms),
pitch/roll snap back to centre right away and throttle ramps down slowly
instead of cutting instantly. Getting OUT of failsafe needs several good
packets in a row, not just one — don't want it flapping in and out.
"""

DISARMED = "DISARMED"
ARMED = "ARMED"
FAILSAFE = "FAILSAFE"
EMERGENCY_STOP = "EMERGENCY_STOP"


class Watchdog:
    def __init__(self, timeout_s=0.5, failsafe_descent_rate=0.5,
                 required_consecutive_valid=5, i2c_fail_threshold=5):
        self.timeout_s = timeout_s
        self.failsafe_descent_rate = failsafe_descent_rate
        self.required_consecutive_valid = required_consecutive_valid
        self.i2c_fail_threshold = i2c_fail_threshold

        self.state = DISARMED
        self.last_valid_time = None
        self.consecutive_valid = 0
        self.failsafe_throttle = 0.0
        self.i2c_fail_count = 0

    def on_valid_packet(self, now, armed, emergency_stop):
        self.last_valid_time = now

        if emergency_stop:
            self.state = EMERGENCY_STOP
            self.consecutive_valid = 0
            return

        if self.state == FAILSAFE:
            # need required_consecutive_valid good packets in a row before we trust it again
            self.consecutive_valid += 1
            if armed and self.consecutive_valid >= self.required_consecutive_valid:
                self.state = ARMED
                self.consecutive_valid = 0
        elif self.state == EMERGENCY_STOP:
            # only way out of e-stop is an explicit disarm from the mac side
            if not armed:
                self.state = DISARMED
        else:
            self.state = ARMED if armed else DISARMED
            self.consecutive_valid += 1

    def tick(self, now, dt, current_throttle_target):
        """Call this every loop. Returns (state, throttle_override) — the
        override is None unless we're in FAILSAFE, in which case use THAT
        instead of whatever throttle came in the packet, and force pitch/roll
        to 0 no matter what."""
        if self.last_valid_time is None:
            return self.state, None

        age = now - self.last_valid_time
        if age > self.timeout_s and self.state == ARMED:
            self.state = FAILSAFE
            self.failsafe_throttle = current_throttle_target
            self.consecutive_valid = 0

        if self.state == FAILSAFE:
            self.failsafe_throttle = max(0.0, self.failsafe_throttle - self.failsafe_descent_rate * dt)
            return self.state, self.failsafe_throttle

        return self.state, None

    def record_i2c_failure(self):
        """Returns True once we've hit the failure threshold — that's the
        signal to the caller to go to a safe state (centre pitch/roll, cut
        throttle, disarm)."""
        self.i2c_fail_count += 1
        if self.i2c_fail_count >= self.i2c_fail_threshold:
            self.state = DISARMED
            return True
        return False

    def record_i2c_success(self):
        self.i2c_fail_count = 0
