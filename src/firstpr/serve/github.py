"""Public stars of a GitHub user, for the "I have GitHub" path.

Privacy: the username is validated, used once for the API call and immediately replaced by a
salted hash (cache key); it is never stored, logged, returned or put in a prompt. Repos owned by
the user are dropped from the stars (their names contain the username).
"""

import hashlib
import hmac
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
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


class GitHubRateLimited(GitHubUnavailable):
    pass


class GitHubQuota:
    """Remaining requests of the shared server token, from GitHub's X-RateLimit-* headers.
    `low()` is true while fewer than `min_remaining` requests are left before the reset, or
    after GitHub refused a request (403 / 429) until its reset time. Per process."""

    def __init__(self, min_remaining: int = 300) -> None:
        self.min_remaining = min_remaining
        self.remaining: int | None = None
        self.reset_at = 0.0

    def update(self, headers: Any) -> None:
        try:
            self.remaining = int(headers["x-ratelimit-remaining"])
            self.reset_at = float(headers["x-ratelimit-reset"])
        except (KeyError, TypeError, ValueError):
            pass

    def exhaust(self, headers: Any) -> None:
        self.remaining = 0
        retry = headers.get("retry-after") if headers is not None else None
        reset = headers.get("x-ratelimit-reset") if headers is not None else None
        self.reset_at = float(reset) if reset else time.time() + float(retry or 60)

    def low(self) -> bool:
        if time.time() >= self.reset_at:
            return False
        return self.remaining is not None and self.remaining < self.min_remaining


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
    username: str,
    token: str | None,
    quota: GitHubQuota | None = None,
    max_pages: int = 2,
    timeout: float = 8.0,
) -> list[Star]:
    """Most recent stars (newest first, up to 100 * max_pages), the user's own repos removed."""
    if not USERNAME.match(username or ""):
        raise InvalidUsername("not a valid GitHub username")
    headers = {"Accept": "application/vnd.github.star+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get_page(page: int) -> list[dict[str, Any]]:
        try:
            r = requests.get(
                f"{API}/users/{username}/starred",
                params={"per_page": 100, "page": page, "sort": "created", "direction": "desc"},
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise GitHubUnavailable(type(e).__name__) from None
        if quota is not None:
            quota.update(r.headers)
        if r.status_code == 404:
            raise UserNotFound("no such GitHub user")
        if r.status_code in (403, 429):
            if quota is not None:
                quota.exhaust(r.headers)
            raise GitHubRateLimited(f"GitHub API returned {r.status_code}")
        if r.status_code >= 500:
            raise GitHubUnavailable(f"GitHub API returned {r.status_code}")
        r.raise_for_status()
        return r.json()

    # pages are fetched in parallel (one round trip instead of max_pages); an empty extra page
    # for users with few stars costs one cheap request
    with ThreadPoolExecutor(max_workers=max_pages) as pool:
        pages = list(pool.map(get_page, range(1, max_pages + 1)))
    out: list[Star] = []
    owner = username.lower()
    for batch in pages:
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


def star_activity(
    stars: list[Star], days: int = 182, today: date | None = None
) -> list[dict[str, Any]]:
    """Stars per day over the last `days` days (date + count only, no repo names), for the
    activity grid on the results page."""
    end = today or datetime.now(UTC).date()
    counts: dict[str, int] = {}
    for s in stars:
        try:
            d = date.fromisoformat((s.starred_at or "")[:10])
        except ValueError:
            continue
        if 0 <= (end - d).days < days:
            counts[d.isoformat()] = counts.get(d.isoformat(), 0) + 1
    return [{"date": d, "count": c} for d, c in sorted(counts.items())]
