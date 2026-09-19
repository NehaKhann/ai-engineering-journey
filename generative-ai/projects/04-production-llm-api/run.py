"""Start the API server:  python run.py

Configure it with environment variables (see app/config.py), for example:
    $env:LLM_API_KEYS = "my-secret-key"
    python run.py
"""

import logging
import os

import uvicorn

from app.backends import LocalBackend
from app.config import Settings
from app.server import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings.from_env()
    if not settings.api_keys:
        # Fail closed: a server with no keys configured must not start open to everyone.
        raise SystemExit("Set LLM_API_KEYS (comma-separated) before starting the server.")
    app = create_app(settings, LocalBackend())
    uvicorn.run(app, host=os.getenv("LLM_HOST", "127.0.0.1"), port=int(os.getenv("LLM_PORT", "8000")))


if __name__ == "__main__":
    main()
