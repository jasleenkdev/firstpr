"""Serving data: static artifacts (model, vectors, repo metadata) + dynamic data (open issues, LLM
features, fresh repos) refreshed daily into a Hugging Face dataset.

Dynamic files are re-fetched lazily: at most once per `refresh_seconds`, in a background
thread, only when the dataset revision changed. Until then (or if the Hub is unreachable) the
bundled snapshot is used, so a refresh failure never takes the API down.
"""

import gzip
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .encoder import SASRecEncoder

DYNAMIC_FILES = ("issues.json.gz", "repo_features.json", "fresh.json.gz", "manifest.json")


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


@dataclass
class Catalog:
    repos: list[dict[str, Any]]
    item_vecs: np.ndarray  # [n, 64] adapter(text) for every repo
    text_emb: np.ndarray  # [n, 384] L2-normalised bge embeddings
    encoder: SASRecEncoder
    topics: list[dict[str, Any]]
    topic_emb: np.ndarray
    costar: dict[int, list[tuple[int, int]]]
    meta: dict[str, Any]
    index: dict[int, int] = field(default_factory=dict)
    issues: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    features: dict[int, dict[str, Any]] = field(default_factory=dict)
    fresh: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, static_dir: str | Path, dynamic_dir: str | Path) -> "Catalog":
        s = Path(static_dir)
        text = np.load(s / "text_emb.npy").astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True) + 1e-12
        cat = cls(
            repos=_read_json(s / "repos.json"),
            item_vecs=np.load(s / "item_vecs.npy"),
            text_emb=text,
            encoder=SASRecEncoder.load(s / "model.npz"),
            topics=_read_json(s / "topics.json"),
            topic_emb=np.load(s / "topic_emb.npy"),
            costar={
                int(k): [tuple(x) for x in v] for k, v in _read_json(s / "costar.json").items()
            },
            meta=_read_json(s / "meta.json"),
        )
        cat.index = {r["id"]: i for i, r in enumerate(cat.repos)}
        cat.load_dynamic(dynamic_dir)
        return cat

    def load_dynamic(self, dynamic_dir: str | Path) -> None:
        d = Path(dynamic_dir)
        issues: dict[int, list[dict[str, Any]]] = {}
        for iss in _read_json(d / "issues.json.gz"):
            issues.setdefault(int(iss["repo_id"]), []).append(iss)
        self.issues = issues
        self.features = {int(k): v for k, v in _read_json(d / "repo_features.json").items()}
        self.fresh = _read_json(d / "fresh.json.gz")
        self.manifest = _read_json(d / "manifest.json")

    def stats(self) -> dict[str, Any]:
        return {
            "repos": len(self.repos),
            "repos_with_issues": sum(1 for r in self.repos if self.issues.get(r["id"])),
            "open_issues": sum(len(v) for v in self.issues.values()),
            "repos_with_llm_features": len(self.features),
            "fresh_repos": len(self.fresh),
            "data_updated": self.manifest.get("updated_at"),
        }


class DynamicRefresher:
    """Pulls newer dynamic files from a (private) HF dataset into a writable dir."""

    def __init__(
        self,
        catalog: Catalog,
        dataset: str | None,
        token: str | None,
        dest: str | Path,
        refresh_seconds: float = 3600.0,
    ) -> None:
        self.catalog, self.dataset, self.token = catalog, dataset, token
        self.dest = Path(dest)
        self.refresh_seconds = refresh_seconds
        self.last_check = 0.0
        self.revision: str | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def maybe_refresh(self, blocking: bool = False) -> None:
        """Check for a new dataset revision at most every `refresh_seconds`. Serverless
        platforms may freeze a process after the response, so callers on a path the user waits
        for anyway (page load) pass blocking=True; others start a background thread."""
        if not self.dataset or time.time() - self.last_check < self.refresh_seconds:
            return
        if not self._lock.acquire(blocking=False):
            return
        self.last_check = time.time()
        if blocking:
            self._refresh()
        else:
            threading.Thread(target=self._refresh, daemon=True).start()

    def _refresh(self) -> None:
        try:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            base = f"https://huggingface.co/api/datasets/{self.dataset}"
            r = requests.get(base, headers=headers, timeout=5)
            r.raise_for_status()
            sha = r.json().get("sha")
            if not sha or sha == self.revision:
                return
            self.dest.mkdir(parents=True, exist_ok=True)
            for name in DYNAMIC_FILES:
                url = f"https://huggingface.co/datasets/{self.dataset}/resolve/{sha}/{name}"
                f = requests.get(url, headers=headers, timeout=15)
                f.raise_for_status()
                (self.dest / name).write_bytes(f.content)
            self.catalog.load_dynamic(self.dest)
            self.revision, self.last_error = sha, None
        except Exception as e:  # noqa: BLE001 - keep serving the current data
            self.last_error = type(e).__name__
        finally:
            self._lock.release()


def env_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("FIRSTPR_ARTIFACTS", Path(__file__).resolve().parent / "artifacts"))
    return root / "static", root / "dynamic"
