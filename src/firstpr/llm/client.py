"""LLM client with an on-disk cache: the same (model, prompt, options) is never sent twice.

Backends (free only): a local Ollama server (default) or Groq's free tier (OpenAI-compatible
API, key from the GROQ_API_KEY environment variable). Responses are appended to one JSONL file
per model under `data/llm_cache/`, keyed by a sha256 of the request. Prompts must never contain
user names, emails or profile text (CLAUDE.md privacy rules): callers pass item text only.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

from firstpr.utils.io import REPO_ROOT
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

CACHE_DIR = REPO_ROOT / "data" / "llm_cache"


class LLMClient:
    def __init__(
        self,
        model: str,
        backend: str = "ollama",
        cache_dir: str | Path = CACHE_DIR,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self.model, self.backend, self.host, self.timeout = model, backend, host, timeout
        safe = model.replace("/", "_").replace(":", "_")
        self.cache_path = Path(cache_dir) / f"{backend}_{safe}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec
        self.calls = 0  # uncached requests made by this client

    def key(self, prompt: str, options: dict[str, Any]) -> str:
        payload = json.dumps([self.backend, self.model, prompt, options], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def cached(self, prompt: str, options: dict[str, Any]) -> str | None:
        rec = self._cache.get(self.key(prompt, options))
        return None if rec is None else rec["response"]

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """Deterministic generation (callers pass temperature 0); cached."""
        k = self.key(prompt, options)
        if k in self._cache:
            return self._cache[k]["response"]
        t0 = time.time()
        response = self._request(prompt, options)
        rec = {
            "key": k,
            "model": self.model,
            "prompt": prompt,
            "options": options,
            "response": response,
            "seconds": round(time.time() - t0, 3),
        }
        with self._lock:
            self._cache[k] = rec
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.calls += 1
        return response

    def _request(self, prompt: str, options: dict[str, Any]) -> str:
        if self.backend == "ollama":
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": options},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["response"]
        if self.backend == "groq":
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
            for attempt in range(8):  # free tier: back off on 429
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": options.get("temperature", 0),
                        "max_tokens": options.get("num_predict", 256),
                    },
                    timeout=self.timeout,
                )
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("retry-after", 2**attempt)))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            raise RuntimeError("groq: rate limited after retries")
        raise ValueError(f"unknown backend {self.backend!r}")
