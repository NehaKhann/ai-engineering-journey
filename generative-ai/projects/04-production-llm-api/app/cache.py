"""An in-memory cache that evicts the least recently used entry and expires old ones (Module 11)."""

import threading
import time
from collections import OrderedDict


class TTLCache:
    """Least-recently-used cache whose entries also expire after `ttl_seconds`.

    `clock` is injectable so tests can move time forward without sleeping.
    """

    def __init__(self, capacity, ttl_seconds, clock=time.monotonic):
        self.capacity, self.ttl, self.clock = capacity, ttl_seconds, clock
        self._items = OrderedDict()
        self._lock = threading.Lock()  # the server handles requests on several threads

    def get(self, key):
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            value, stored_at = entry
            if self.clock() - stored_at > self.ttl:
                del self._items[key]  # expired
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key, value):
        with self._lock:
            self._items[key] = (value, self.clock())
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def __len__(self):
        return len(self._items)
