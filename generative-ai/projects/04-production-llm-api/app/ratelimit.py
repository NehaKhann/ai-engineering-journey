"""A token-bucket rate limiter, one bucket per API key."""

import threading
import time


class RateLimiter:
    """Each key holds up to `per_minute` tokens and regains them steadily. A request costs one token.

    This allows short bursts (a full bucket) while capping the long-run rate, which is what a
    provider's rate limit usually does. `clock` is injectable for tests.
    """

    def __init__(self, per_minute, clock=time.monotonic):
        self.capacity = per_minute
        self.refill_per_second = per_minute / 60
        self.clock = clock
        self._buckets = {}  # key -> (tokens, last_update_time)
        self._lock = threading.Lock()

    def allow(self, key):
        """Return (allowed, seconds_until_a_token_is_available)."""
        with self._lock:
            now = self.clock()
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1 - tokens) / self.refill_per_second
