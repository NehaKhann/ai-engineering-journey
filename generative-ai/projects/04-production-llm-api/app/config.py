"""Settings, read from environment variables so the same code runs locally and in production."""

import os
from dataclasses import dataclass, field


def _split(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    api_keys: tuple = ()  # who may call the API. Never hard-code keys in source.
    rate_limit_per_minute: int = 30  # requests per key per minute
    cache_size: int = 256  # most cached answers to keep
    cache_ttl_seconds: float = 300.0  # how long an answer stays valid
    max_new_tokens_cap: int = 256  # the most tokens any request may ask for
    default_max_new_tokens: int = 64
    max_prompt_chars: int = 4000  # reject huge inputs before they cost money
    price_per_million_input: float = 2.00  # illustrative dollars, for cost estimates
    price_per_million_output: float = 10.00
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        return cls(
            api_keys=_split(os.getenv("LLM_API_KEYS", "")),
            rate_limit_per_minute=int(os.getenv("LLM_RATE_LIMIT_PER_MINUTE", "30")),
            cache_size=int(os.getenv("LLM_CACHE_SIZE", "256")),
            cache_ttl_seconds=float(os.getenv("LLM_CACHE_TTL_SECONDS", "300")),
            max_new_tokens_cap=int(os.getenv("LLM_MAX_NEW_TOKENS_CAP", "256")),
        )
