"""Public stars of a GitHub user, for the "I have GitHub" path.

Privacy: the username is validated, used once for the API call and immediately replaced by a
salted hash (cache key); it is never stored, logged, returned or put in a prompt. Repos owned by
the user are dropped from the stars (their names contain the username).
"""

import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
API = "https://api.github.com"


class InvalidUsername(ValueError):
    pass


class UserNotFound(LookupError):
    pass


class GitHubUnavailable(RuntimeError):
    pass


def hash_username(username: str, salt: bytes) -> str:
    """Salted HMAC of the lower-cased login (logins are case-insensitive)."""
    return hmac.new(salt, username.lower().encode(), hashlib.sha256).hexdigest()[:16]


@dataclass
class Star:
    repo_id: int
    full_name: str
    language: str | None
    topics: list[str]
    starred_at: str


def fetch_stars(
    username: str, token: str | None, max_pages: int = 2, timeout: float = 8.0
) -> list[Star]:
    """Most recent stars (newest first, up to 100 * max_pages), the user's own repos removed."""
    if not USERNAME.match(username or ""):
        raise InvalidUsername("not a valid GitHub username")
    headers = {"Accept": "application/vnd.github.star+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    out: list[Star] = []
    owner = username.lower()
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(
                f"{API}/users/{username}/starred",
                params={"per_page": 100, "page": page, "sort": "created", "direction": "desc"},
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise GitHubUnavailable(type(e).__name__) from None
        if r.status_code == 404:
            raise UserNotFound("no such GitHub user")
        if r.status_code in (403, 429) or r.status_code >= 500:
            raise GitHubUnavailable(f"GitHub API returned {r.status_code}")
        r.raise_for_status()
        batch: list[dict[str, Any]] = r.json()
        for item in batch:
            repo = item.get("repo", item)
            if (repo.get("owner") or {}).get("login", "").lower() == owner:
                continue  # own repo: its name contains the username
            out.append(
                Star(
                    repo_id=int(repo["id"]),
                    full_name=repo["full_name"],
                    language=repo.get("language"),
                    topics=list(repo.get("topics") or []),
                    starred_at=item.get("starred_at", ""),
                )
            )
        if len(batch) < 100:
            break
    return out


class StarCache:
    """In-memory cache keyed by the username hash (TTL), so repeated requests skip GitHub."""

    def __init__(self, ttl: float = 3600.0, max_items: int = 512) -> None:
        self.ttl, self.max_items = ttl, max_items
        self._d: dict[str, tuple[float, list[Star]]] = {}

    def get(self, key: str) -> list[Star] | None:
        hit = self._d.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        return None

    def put(self, key: str, stars: list[Star]) -> None:
        if len(self._d) >= self.max_items:
            self._d.pop(next(iter(self._d)))
        self._d[key] = (time.time(), stars)
