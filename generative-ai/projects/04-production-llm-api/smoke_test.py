"""Send real requests through the real API to the real local model, and measure what happens.

Run from this folder:  python smoke_test.py
Most checks use FastAPI's test client. Streaming uses a real local server, because the test client
buffers a streamed response and so cannot show when the first token arrives.
"""

import json
import logging
import socket
import threading
import time

import httpx
import uvicorn
from fastapi.testclient import TestClient

from app.backends import LocalBackend
from app.config import Settings
from app.server import create_app

KEY = "smoke-test-key"
HEADERS = {"X-API-Key": KEY}
BODY = {"messages": [{"role": "user", "content": "Explain in two sentences what an API is."}], "max_new_tokens": 48}


class ThreadedServer(uvicorn.Server):
    def install_signal_handlers(self):  # signal handlers only work on the main thread
        pass


def start_live_server(app):
    """Run the app on a real local port so streaming can be measured over a real connection."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = ThreadedServer(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, port


def main():
    logging.disable(logging.CRITICAL)  # keep the output readable
    print("Loading the model...")
    app = create_app(Settings(api_keys=(KEY,), rate_limit_per_minute=1000), LocalBackend())
    client = TestClient(app)

    print("\n1. The same request twice (the second should come from the cache)")
    timings = []
    for attempt in ("first ", "second"):
        start = time.perf_counter()
        data = client.post("/v1/chat", json=BODY, headers=HEADERS).json()
        timings.append(time.perf_counter() - start)
        print(f"   {attempt}: {timings[-1] * 1000:>8.1f} ms   cached={data['cached']}   tokens={data['usage']}")
    print(f"   The cached answer was {timings[0] / timings[1]:,.0f}x faster.")
    print(f"   Reply: {data['text'][:100]!r}")

    print("\n2. Streaming over a real connection: time to the first token vs the whole reply")
    server, port = start_live_server(app)
    start = time.perf_counter()
    first, pieces = None, []
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/stream", json=BODY, headers=HEADERS, timeout=120) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if "token" in event:
                    first = first if first is not None else time.perf_counter() - start
                    pieces.append(event["token"])
    total = time.perf_counter() - start
    server.should_exit = True
    print(f"   first token after {first * 1000:.0f} ms, complete after {total * 1000:.0f} ms ({len(pieces)} chunks)")

    print("\n3. Rejected requests never reach the model")
    for label, kwargs in [("no API key", dict(json=BODY)), ("wrong API key", dict(json=BODY, headers={"X-API-Key": "nope"}))]:
        start = time.perf_counter()
        status = client.post("/v1/chat", **kwargs).status_code
        print(f"   {label:<14} -> {status} in {(time.perf_counter() - start) * 1000:.1f} ms")

    print("\n4. Metrics after this session")
    for name, value in client.get("/metrics", headers=HEADERS).json().items():
        print(f"   {name:<20}{value:.4f}" if isinstance(value, float) else f"   {name:<20}{value}")


if __name__ == "__main__":
    main()
