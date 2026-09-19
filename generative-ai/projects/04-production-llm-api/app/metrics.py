"""Counters and latency percentiles: the numbers a dashboard would show (Module 11)."""

import threading

import numpy as np


class Metrics:
    def __init__(self, price_in_per_million, price_out_per_million):
        self.price_in, self.price_out = price_in_per_million / 1e6, price_out_per_million / 1e6
        self._lock = threading.Lock()
        self.requests = self.errors = self.cache_hits = self.rate_limited = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.latencies = []

    def record(self, latency_s, cached=False, prompt_tokens=0, completion_tokens=0, error=False):
        with self._lock:
            self.requests += 1
            self.errors += error
            self.cache_hits += cached
            self.latencies.append(latency_s)
            if not cached and not error:  # a cache hit costs nothing, a failed call bills nothing
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens

    def record_rate_limited(self):
        with self._lock:
            self.rate_limited += 1

    def snapshot(self):
        with self._lock:
            latencies = np.array(self.latencies) if self.latencies else np.array([0.0])
            return {
                "requests": self.requests,
                "errors": self.errors,
                "rate_limited": self.rate_limited,
                "cache_hit_rate": self.cache_hits / self.requests if self.requests else 0.0,
                "latency_ms_p50": float(np.percentile(latencies, 50) * 1000),
                "latency_ms_p95": float(np.percentile(latencies, 95) * 1000),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "estimated_cost_usd": self.prompt_tokens * self.price_in + self.completion_tokens * self.price_out,
            }
