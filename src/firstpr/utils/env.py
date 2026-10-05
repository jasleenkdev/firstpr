"""Secrets from `.env` (never from code or configs). Values are never logged."""

import os

from dotenv import load_dotenv

from firstpr.utils.io import REPO_ROOT

_loaded = False


def require_env(name: str) -> str:
    """Value of an environment variable, loading `<repo>/.env` once; raises if it is unset."""
    global _loaded
    if not _loaded:
        load_dotenv(REPO_ROOT / ".env", override=False)
        _loaded = True
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set (add it to .env, see .env.example)")
    return value
