"""GitHub GraphQL client: every response cached on disk (sqlite), rate limits respected.

Repos are addressed by numeric id (GH Archive `repo.id`) through global node ids, so renamed
repos resolve correctly. No query asks for a person (issue authors, assignee logins, ...): only
repo metadata, README text, labels and issue text / counts.
"""

import base64
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from firstpr.utils.env import require_env
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

URL = "https://api.github.com/graphql"
README_PATHS = [
    "README.md",
    "readme.md",
    "Readme.md",
    "README.rst",
    "README",
    "README.markdown",
    "README.txt",
]


def _msgpack_uint(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    if n < 0x100:
        return b"\xcc" + n.to_bytes(1, "big")
    if n < 0x10000:
        return b"\xcd" + n.to_bytes(2, "big")
    if n < 0x100000000:
        return b"\xce" + n.to_bytes(4, "big")
    return b"\xcf" + n.to_bytes(8, "big")


def repo_node_id(database_id: int) -> str:
    """Global node id of a repository: 'R_' + base64url(msgpack([0, database_id]))."""
    raw = b"\x92\x00" + _msgpack_uint(int(database_id))
    return "R_" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class GraphQLClient:
    def __init__(self, cache_path: str | Path, min_remaining: int = 50, workers: int = 3) -> None:
        """`workers` requests in flight: large batches take ~15 s of server time each, and
        GitHub's secondary limit allows ~60 s of server time per minute."""
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self._lock = threading.Lock()
        self.min_remaining = min_remaining
        self.workers = workers
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"bearer {require_env('GITHUB_TOKEN')}"
        self.calls = 0

    def run(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Cached GraphQL call. Responses with partial errors (e.g. NOT_FOUND for a deleted
        repo) are cached too; transport errors are retried and never cached."""
        key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
        with self._lock:
            row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return json.loads(row[0])
        body = self._post({"query": query, "variables": variables or {}})
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(body)))
            self.db.commit()
            self.calls += 1
        return body

    def run_many(
        self, requests_: list[tuple[str, dict[str, Any] | None]], what: str
    ) -> Iterator[dict[str, Any]]:
        """`run` over many requests with `workers` in flight; results in input order."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, body in enumerate(pool.map(lambda qv: self.run(*qv), requests_)):
                if i % 100 == 0:
                    log.info("%s: %d / %d requests", what, i + 1, len(requests_))
                yield body

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(8):
            try:
                r = self.session.post(URL, json=payload, timeout=120)
            except requests.RequestException as e:
                log.warning("graphql transport error (%s), retry %d", type(e).__name__, attempt)
                time.sleep(2**attempt)
                continue
            if r.status_code in (403, 429):  # secondary rate limit
                wait = float(r.headers.get("retry-after", 60))
                log.warning("graphql %d: waiting %.0fs", r.status_code, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                log.warning("graphql %d, retry %d", r.status_code, attempt)
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            body = r.json()
            self._respect_limit(body)
            if body.get("data") is None:
                raise RuntimeError(f"graphql error: {json.dumps(body.get('errors'))[:500]}")
            return body
        raise RuntimeError("graphql: giving up after retries")

    def _respect_limit(self, body: dict[str, Any]) -> None:
        rl = (body.get("data") or {}).get("rateLimit")
        if rl and rl["remaining"] < self.min_remaining:
            reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
            wait = max(0.0, (reset - datetime.now(UTC)).total_seconds()) + 5
            log.info("graphql budget low (%d left): sleeping %.0fs", rl["remaining"], wait)
            time.sleep(wait)

    def _nodes(
        self,
        repo_ids: list[int],
        fields: str,
        batch: int,
        parse: Callable[[dict[str, Any]], dict[str, Any]],
        what: str,
    ) -> pd.DataFrame:
        query = (
            "query($ids: [ID!]!) { rateLimit { cost remaining resetAt } "
            f"nodes(ids: $ids) {{ ... on Repository {{ databaseId {fields} }} }} }}"
        )
        rows = []
        reqs = [
            (query, {"ids": [repo_node_id(r) for r in chunk]})
            for chunk in _chunks([int(r) for r in repo_ids], batch)
        ]
        for body in self.run_many(reqs, what):
            for node in body["data"]["nodes"]:
                if node and node.get("databaseId") is not None:
                    rows.append({"repo_id": int(node["databaseId"]), **parse(node)})
        log.info("%s: %d of %d repos resolved", what, len(rows), len(repo_ids))
        return pd.DataFrame(rows)

    def _aliased(
        self,
        items: list[tuple[int, list[str]]],
        batch: int,
        field: Callable[[list[str]], str],
        what: str,
    ) -> Iterator[tuple[list[tuple[int, list[str]]], dict[str, Any]]]:
        """One query per `batch` repos, each repo an alias `r<j>` with its own field (per-repo
        label lists); yields (chunk, response) in input order."""
        chunks = list(_chunks(items, batch))
        reqs: list[tuple[str, dict[str, Any] | None]] = []
        for chunk in chunks:
            parts = [
                f'r{j}: node(id: "{repo_node_id(rid)}") {{ ... on Repository {{ databaseId '
                f"{field(labels)} }} }}"
                for j, (rid, labels) in enumerate(chunk)
            ]
            reqs.append(
                ("query { rateLimit { cost remaining resetAt } " + " ".join(parts) + "}", None)
            )
        yield from zip(chunks, self.run_many(reqs, what), strict=True)

    def repo_labels(self, repo_ids: list[int], batch: int = 50) -> pd.DataFrame:
        """name, created_at, is_fork, is_archived, stars, label names (first 100)."""
        fields = (
            "nameWithOwner createdAt isFork isArchived stargazerCount "
            "labels(first: 100) { nodes { name } }"
        )

        def parse(n: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": n["nameWithOwner"],
                "created_at": n["createdAt"],
                "is_fork": n["isFork"],
                "is_archived": n["isArchived"],
                "stars_now": n["stargazerCount"],
                "labels": [x["name"] for x in (n.get("labels") or {}).get("nodes", [])],
            }

        return self._nodes(repo_ids, fields, batch, parse, "repo labels")

    def labelled_issue_dates(
        self, repo_labels: dict[int, list[str]], since: str, batch: int = 25, per_repo: int = 20
    ) -> dict[int, list[str]]:
        """Creation dates of the oldest `per_repo` issues updated since `since` that carry any
        of the repo's given labels (an issue created in a window is updated after its start)."""
        out: dict[int, list[str]] = {}

        def field(labels: list[str]) -> str:
            return (
                f"issues(first: {per_repo}, orderBy: {{field: CREATED_AT, direction: ASC}}, "
                f'filterBy: {{since: "{since}T00:00:00Z", labels: {json.dumps(labels)}}}) '
                "{ nodes { createdAt } }"
            )

        for chunk, body in self._aliased(
            list(repo_labels.items()), batch, field, "labelled issues"
        ):
            for j, (rid, _) in enumerate(chunk):
                node = body["data"].get(f"r{j}") or {}
                out[rid] = [n["createdAt"] for n in (node.get("issues") or {}).get("nodes", [])]
        return out

    def repo_details(self, repo_ids: list[int], cutoff: str, batch: int = 20) -> pd.DataFrame:
        """Metadata (current) + README text as of the last commit before `cutoff`."""
        readme = " ".join(
            f'f{k}: file(path: "{p}") {{ object {{ ... on Blob {{ text isBinary }} }} }}'
            for k, p in enumerate(README_PATHS)
        )
        fields = (
            "nameWithOwner description homepageUrl createdAt pushedAt isFork isArchived "
            "stargazerCount forkCount licenseInfo { spdxId } primaryLanguage { name } "
            "languages(first: 5, orderBy: {field: SIZE, direction: DESC}) { nodes { name } } "
            "repositoryTopics(first: 10) { nodes { topic { name } } } "
            "defaultBranchRef { target { ... on Commit { "
            f'history(first: 1, until: "{cutoff}T00:00:00Z") {{ nodes {{ oid committedDate '
            f"{readme} }} }} }} }} }}"
        )

        def parse(n: dict[str, Any]) -> dict[str, Any]:
            hist = ((n.get("defaultBranchRef") or {}).get("target") or {}).get("history") or {}
            commit = (hist.get("nodes") or [None])[0] or {}
            text = ""
            for k in range(len(README_PATHS)):
                obj = (commit.get(f"f{k}") or {}).get("object") or {}
                if obj.get("text") and not obj.get("isBinary"):
                    text = obj["text"]
                    break
            return {
                "name": n["nameWithOwner"],
                "description": n.get("description") or "",
                "created_at": n["createdAt"],
                "pushed_at": n.get("pushedAt"),
                "is_fork": n["isFork"],
                "is_archived": n["isArchived"],
                "stars_now": n["stargazerCount"],
                "forks_now": n["forkCount"],
                "license": (n.get("licenseInfo") or {}).get("spdxId"),
                "language": (n.get("primaryLanguage") or {}).get("name"),
                "languages": [x["name"] for x in (n.get("languages") or {}).get("nodes", [])],
                "topics": [
                    x["topic"]["name"] for x in (n.get("repositoryTopics") or {}).get("nodes", [])
                ],
                "readme_commit_date": commit.get("committedDate"),
                "readme": text,
            }

        return self._nodes(repo_ids, fields, batch, parse, "repo details")

    def open_issues(
        self, repo_labels: dict[int, list[str]], batch: int = 10, per_repo: int = 20
    ) -> pd.DataFrame:
        """Currently open issues carrying any of the repo's beginner labels (newest first)."""
        rows = []

        def field(labels: list[str]) -> str:
            return (
                f"issues(first: {per_repo}, states: OPEN, "
                "orderBy: {field: CREATED_AT, direction: DESC}, "
                f"filterBy: {{labels: {json.dumps(labels)}}}) "
                "{ nodes { number title body createdAt updatedAt "
                "labels(first: 10) { nodes { name } } comments { totalCount } "
                "assignees { totalCount } } }"
            )

        for chunk, body in self._aliased(list(repo_labels.items()), batch, field, "open issues"):
            for j, (rid, _) in enumerate(chunk):
                node = body["data"].get(f"r{j}") or {}
                for iss in (node.get("issues") or {}).get("nodes", []):
                    rows.append(
                        {
                            "repo_id": rid,
                            "number": iss["number"],
                            "title": iss["title"],
                            "body": iss.get("body") or "",
                            "created_at": iss["createdAt"],
                            "updated_at": iss["updatedAt"],
                            "labels": [x["name"] for x in iss["labels"]["nodes"]],
                            "n_comments": iss["comments"]["totalCount"],
                            "n_assignees": iss["assignees"]["totalCount"],
                        }
                    )
        return pd.DataFrame(rows)
