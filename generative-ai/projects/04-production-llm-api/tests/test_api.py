"""API tests. A fake model and a fake clock make every behavior exact and instant.

Run from the project folder:  python -m unittest discover -s tests -v
"""

import json
import logging
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends import FakeBackend  # noqa: E402
from app.cache import TTLCache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.ratelimit import RateLimiter  # noqa: E402
from app.server import create_app  # noqa: E402

KEY = "test-key-alpha"
OTHER_KEY = "test-key-beta"
CHAT = {"messages": [{"role": "user", "content": "What is a token?"}]}


class Clock:
    """A clock the test controls."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_client(backend=None, clock=None, **overrides):
    settings = Settings(api_keys=(KEY, OTHER_KEY), **overrides)
    backend = backend or FakeBackend("A token is a small chunk of text.")
    clock = clock or Clock()
    return TestClient(create_app(settings, backend, clock)), backend, clock


def post(client, body=CHAT, key=KEY, path="/v1/chat"):
    return client.post(path, json=body, headers={"X-API-Key": key} if key else {})


class UnitTests(unittest.TestCase):
    def test_cache_evicts_least_recently_used(self):
        cache = TTLCache(2, 100, Clock())
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # "a" is now the most recently used
        cache.put("c", 3)  # evicts "b"
        self.assertEqual((cache.get("a"), cache.get("b"), cache.get("c")), (1, None, 3))

    def test_cache_entries_expire(self):
        clock = Clock()
        cache = TTLCache(10, 60, clock)
        cache.put("a", 1)
        clock.advance(59)
        self.assertEqual(cache.get("a"), 1)
        clock.advance(2)
        self.assertIsNone(cache.get("a"))

    def test_token_bucket_allows_a_burst_then_refills(self):
        clock = Clock()
        limiter = RateLimiter(per_minute=3, clock=clock)
        self.assertTrue(all(limiter.allow("k")[0] for _ in range(3)))
        allowed, wait = limiter.allow("k")
        self.assertFalse(allowed)
        self.assertAlmostEqual(wait, 20.0)  # 3 per minute is one token every 20 seconds
        clock.advance(20)
        self.assertTrue(limiter.allow("k")[0])


class AuthTests(unittest.TestCase):
    def test_health_needs_no_key(self):
        client, _, _ = make_client()
        self.assertEqual(client.get("/health").json(), {"status": "ok"})

    def test_missing_and_wrong_keys_are_rejected_before_the_model_runs(self):
        client, backend, _ = make_client()
        self.assertEqual(post(client, key=None).status_code, 401)
        self.assertEqual(post(client, key="wrong").status_code, 401)
        self.assertEqual(backend.calls, 0)

    def test_valid_key_works(self):
        client, _, _ = make_client()
        response = post(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "A token is a small chunk of text.")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.client, self.backend, _ = make_client(max_prompt_chars=50, max_new_tokens_cap=10)

    def test_bad_input_is_rejected_without_calling_the_model(self):
        cases = [
            ({"messages": []}, 422),
            ({"messages": [{"role": "hacker", "content": "hi"}]}, 422),
            ({"messages": [{"role": "user", "content": ""}]}, 422),
            ({"messages": [{"role": "user", "content": "x" * 51}]}, 413),
            ({"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 11}, 422),
            ({"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 0}, 422),
        ]
        for body, expected in cases:
            self.assertEqual(post(self.client, body).status_code, expected, body)
        self.assertEqual(self.backend.calls, 0)


class CacheBehaviorTests(unittest.TestCase):
    def test_repeat_request_is_served_from_cache(self):
        client, backend, _ = make_client()
        first, second = post(client).json(), post(client).json()
        self.assertEqual((first["cached"], second["cached"]), (False, True))
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(backend.calls, 1)

    def test_different_parameters_are_different_cache_entries(self):
        client, backend, _ = make_client()
        post(client, {**CHAT, "max_new_tokens": 5})
        post(client, {**CHAT, "max_new_tokens": 6})
        self.assertEqual(backend.calls, 2)

    def test_one_customers_cache_is_never_served_to_another(self):
        client, backend, _ = make_client()
        post(client, key=KEY)
        other = post(client, key=OTHER_KEY).json()
        self.assertFalse(other["cached"])
        self.assertEqual(backend.calls, 2)

    def test_cache_expires_after_the_ttl(self):
        client, backend, clock = make_client(cache_ttl_seconds=60)
        post(client)
        clock.advance(61)
        self.assertFalse(post(client).json()["cached"])
        self.assertEqual(backend.calls, 2)


class RateLimitTests(unittest.TestCase):
    def test_excess_requests_get_429_with_retry_after(self):
        client, backend, clock = make_client(rate_limit_per_minute=3)
        codes = [post(client, {"messages": [{"role": "user", "content": f"q{i}"}]}).status_code for i in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        blocked = post(client, {"messages": [{"role": "user", "content": "q9"}]})
        self.assertEqual(blocked.headers["Retry-After"], "20")
        clock.advance(20)
        self.assertEqual(post(client, {"messages": [{"role": "user", "content": "q10"}]}).status_code, 200)

    def test_each_key_has_its_own_limit(self):
        client, _, _ = make_client(rate_limit_per_minute=1)
        self.assertEqual(post(client, key=KEY).status_code, 200)
        self.assertEqual(post(client, {"messages": [{"role": "user", "content": "again"}]}, key=KEY).status_code, 429)
        self.assertEqual(post(client, key=OTHER_KEY).status_code, 200)


class ErrorHandlingTests(unittest.TestCase):
    def test_backend_failure_returns_a_generic_502_and_leaks_nothing(self):
        client, _, _ = make_client(backend=FakeBackend(fail=True))
        with self.assertLogs("llm_api", level="ERROR"):  # the detail is logged on the server...
            response = post(client)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("exploded", response.text)  # ...but never shown to the caller
        self.assertNotIn("Traceback", response.text)


class StreamingTests(unittest.TestCase):
    def test_stream_sends_tokens_then_a_done_event(self):
        client, _, _ = make_client(backend=FakeBackend("one two three"))
        response = post(client, path="/v1/chat/stream")
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual("".join(e["token"] for e in events if "token" in e).strip(), "one two three")
        self.assertTrue(events[-1]["done"])

    def test_stream_requires_a_key(self):
        client, _, _ = make_client()
        self.assertEqual(post(client, key=None, path="/v1/chat/stream").status_code, 401)


class ObservabilityTests(unittest.TestCase):
    def test_metrics_report_traffic_cache_hits_and_errors(self):
        client, _, _ = make_client()
        post(client)
        post(client)  # cache hit
        post(client, key="wrong")  # rejected before it is counted as model traffic
        metrics = client.get("/metrics", headers={"X-API-Key": KEY}).json()
        self.assertEqual(metrics["requests"], 2)
        self.assertEqual(metrics["cache_hit_rate"], 0.5)
        self.assertGreater(metrics["completion_tokens"], 0)  # the cache hit added no tokens

    def test_metrics_need_a_key(self):
        client, _, _ = make_client()
        self.assertEqual(client.get("/metrics").status_code, 401)

    def test_every_response_has_a_request_id_and_the_key_is_never_logged(self):
        client, _, _ = make_client()
        with self.assertLogs("llm_api", level="INFO") as captured:
            first, second = post(client), post(client, {"messages": [{"role": "user", "content": "other"}]})
        self.assertNotEqual(first.headers["X-Request-ID"], second.headers["X-Request-ID"])
        self.assertNotIn(KEY, "\n".join(captured.output))


if __name__ == "__main__":
    logging.disable(logging.NOTSET)
    unittest.main()
