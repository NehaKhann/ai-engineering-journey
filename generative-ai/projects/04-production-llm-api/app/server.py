"""The API itself. `create_app` takes its settings, model backend and clock as arguments, so tests can swap in fakes."""

import hashlib
import hmac
import json
import logging
import math
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache import TTLCache
from .metrics import Metrics
from .ratelimit import RateLimiter

log = logging.getLogger("llm_api")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_new_tokens: int | None = Field(default=None, ge=1)


def key_id(api_key):
    """A short, non-reversible label for a key, safe to put in logs. The key itself is never logged."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]


def create_app(settings, backend, clock=time.monotonic):
    app = FastAPI(title="Production LLM API")
    cache = TTLCache(settings.cache_size, settings.cache_ttl_seconds, clock)
    limiter = RateLimiter(settings.rate_limit_per_minute, clock)
    metrics = Metrics(settings.price_per_million_input, settings.price_per_million_output)

    # ---- request id and structured access log -------------------------------------------------
    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        supplied = request.headers.get("x-api-key")
        log.info(json.dumps({
            "request_id": request_id, "method": request.method, "path": request.url.path,
            "status": response.status_code, "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "client": key_id(supplied) if supplied else None,
        }))
        return response

    # ---- authentication and rate limiting -----------------------------------------------------
    def authenticate(x_api_key: str | None = Header(default=None)):
        if x_api_key is None:
            raise HTTPException(status_code=401, detail="Missing API key. Send it in the X-API-Key header.")
        # compare_digest takes the same time whether the first or last character is wrong
        if not any(hmac.compare_digest(x_api_key.encode(), valid.encode()) for valid in settings.api_keys):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        return x_api_key

    def rate_limited(api_key: str = Depends(authenticate)):
        allowed, wait = limiter.allow(key_id(api_key))
        if not allowed:
            metrics.record_rate_limited()
            raise HTTPException(status_code=429, detail="Rate limit exceeded.", headers={"Retry-After": str(math.ceil(wait))})
        return api_key

    # ---- shared request checks ----------------------------------------------------------------
    def prepare(body: ChatRequest):
        """Validate limits that the schema cannot express. Returns (messages, max_new_tokens)."""
        if sum(len(m.content) for m in body.messages) > settings.max_prompt_chars:
            raise HTTPException(status_code=413, detail=f"Prompt too long. The limit is {settings.max_prompt_chars} characters.")
        max_new = body.max_new_tokens or settings.default_max_new_tokens
        if max_new > settings.max_new_tokens_cap:
            raise HTTPException(status_code=422, detail=f"max_new_tokens must be at most {settings.max_new_tokens_cap}.")
        return [m.model_dump() for m in body.messages], max_new

    # ---- routes -------------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def get_metrics(api_key: str = Depends(authenticate)):
        return metrics.snapshot()

    @app.post("/v1/chat")
    def chat(body: ChatRequest, api_key: str = Depends(rate_limited)):
        start = clock()
        messages, max_new = prepare(body)
        # The cache key includes WHO is asking, so one customer's cached answer is never served to another.
        cache_key = hashlib.sha256(json.dumps([key_id(api_key), messages, max_new], sort_keys=True).encode()).hexdigest()

        cached = cache.get(cache_key)
        if cached is not None:
            metrics.record(clock() - start, cached=True)
            return {**cached, "cached": True}

        try:
            result = backend.generate(messages, max_new)
        except Exception:
            log.exception("backend failure")  # full detail goes to the server log only
            metrics.record(clock() - start, error=True)
            raise HTTPException(status_code=502, detail="The model backend failed. Please try again.")

        payload = {
            "id": uuid.uuid4().hex[:12],
            "text": result.text,
            "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
            "cached": False,
        }
        cache.put(cache_key, payload)
        metrics.record(clock() - start, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens)
        return payload

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatRequest, api_key: str = Depends(rate_limited)):
        messages, max_new = prepare(body)

        def events():
            start, chunks = clock(), 0
            try:
                for piece in backend.stream(messages, max_new):
                    chunks += 1
                    yield f"data: {json.dumps({'token': piece})}\n\n"
            except Exception:
                log.exception("backend failure while streaming")
                metrics.record(clock() - start, error=True)
                yield f"data: {json.dumps({'error': 'The model backend failed.'})}\n\n"
                return
            prompt_words = sum(len(m["content"].split()) for m in messages)
            metrics.record(clock() - start, prompt_tokens=prompt_words, completion_tokens=chunks)  # approximate counts
            yield f"data: {json.dumps({'done': True, 'usage': {'approx_completion_tokens': chunks}})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
