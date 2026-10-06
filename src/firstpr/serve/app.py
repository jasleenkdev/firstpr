"""FirstPR API (FastAPI). Configuration from environment variables:

ID_SALT (required)       salt for hashing usernames
GITHUB_TOKEN             server token for the public-stars request (5,000 requests/h)
GROQ_API_KEY             explanations (template fallback without it)
EXPLAIN_MODEL            default qwen/qwen3.8-27b (its own free-tier quota)
FIRSTPR_DATASET, HF_TOKEN  dataset with the daily-refreshed dynamic files
ALLOWED_ORIGINS          comma-separated CORS origins; ALLOWED_ORIGIN_REGEX optional
FIRSTPR_ARTIFACTS        artifact root (static/ + dynamic/ snapshot)

Logs carry stage timings and counts only: never a username, a hash of one, or a star list.
"""

import json
import logging
import os
import time
from collections import Counter, defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import ranking
from .catalog import Catalog, DynamicRefresher, env_paths
from .explain import Explainer, collect_facts
from .github import (
    USERNAME,
    GitHubUnavailable,
    InvalidUsername,
    StarCache,
    UserNotFound,
    fetch_stars,
    hash_username,
)

log = logging.getLogger("firstpr.api")


class GitHubRequest(BaseModel):
    username: str = Field(min_length=1, max_length=39)
    hours: float = Field(default=3.0, ge=0, le=60)


class OnboardingRequest(BaseModel):
    languages: list[str] = Field(default_factory=list, max_length=10)
    interests: list[str] = Field(default_factory=list, max_length=10)
    hours: float = Field(default=3.0, ge=0, le=60)


class ExplainRequest(BaseModel):
    repo_id: int
    issue_number: int | None = None
    co_starred: list[str] = Field(default_factory=list, max_length=5)
    skills: list[str] = Field(default_factory=list, max_length=10)


class RateLimiter:
    """Sliding window per client key (requests per minute); keys are not logged or stored."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.time(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        if len(self.hits) > 10_000:
            self.hits.clear()
        return True


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def create_app(catalog: Catalog | None = None, star_fetcher: Any = fetch_stars) -> FastAPI:
    t_load = time.perf_counter()
    static, dynamic = env_paths()
    cat = catalog or Catalog.load(static, dynamic)
    load_ms = _ms(t_load)
    salt = os.environ.get("ID_SALT", "").encode()
    if not salt:
        raise RuntimeError("ID_SALT must be set")
    refresher = DynamicRefresher(
        cat, os.environ.get("FIRSTPR_DATASET"), os.environ.get("HF_TOKEN"), "/tmp/firstpr-dynamic"
    )
    explainer = Explainer(
        os.environ.get("GROQ_API_KEY"), os.environ.get("EXPLAIN_MODEL", "qwen/qwen3.8-27b")
    )
    stars_cache = StarCache()
    limiter = RateLimiter(int(os.environ.get("RATE_PER_MINUTE", "30")))

    app = FastAPI(title="FirstPR API", docs_url=None, redoc_url=None)
    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def guard(request: Request) -> None:
        refresher.maybe_refresh()
        client = request.headers.get(
            "x-forwarded-for", request.client.host if request.client else ""
        )
        if not limiter.allow(client.split(",")[0].strip()):
            raise HTTPException(429, "too many requests, try again in a minute")

    def record(path: str, timings: dict[str, float], **counts: Any) -> None:
        log.info(json.dumps({"path": path, "timings_ms": timings, **counts}))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": cat.meta.get("model"), "load_ms": load_ms, **cat.stats()}

    @app.get("/options")
    def options() -> dict[str, Any]:
        langs = Counter(
            r["language"] for r in cat.repos if r["language"] and cat.issues.get(r["id"])
        )
        return {
            "languages": [x for x, _ in langs.most_common(16)],
            "interests": [{"id": t["id"], "label": t["label"]} for t in cat.topics],
            "hours": [1, 3, 5, 8, 12],
        }

    @app.post("/recommend/github")
    def recommend_github(body: GitHubRequest, request: Request) -> dict[str, Any]:
        guard(request)
        t0 = time.perf_counter()
        if not USERNAME.match(body.username):
            raise HTTPException(422, "that is not a valid GitHub username")
        try:
            key = hash_username(body.username, salt)  # the raw username is not kept past here
            stars = stars_cache.get(key)
            cached = stars is not None
            if stars is None:
                stars = star_fetcher(body.username, os.environ.get("GITHUB_TOKEN"))
                stars_cache.put(key, stars)
        except InvalidUsername:
            raise HTTPException(422, "that is not a valid GitHub username") from None
        except UserNotFound:
            raise HTTPException(404, "no GitHub user with that name") from None
        except GitHubUnavailable:
            raise HTTPException(
                503, "GitHub is not answering right now, try again shortly"
            ) from None
        t_fetch = _ms(t0)
        t1 = time.perf_counter()
        out = ranking.recommend_github(cat, stars, body.hours, exclude_owner=body.username.lower())
        t_rank = _ms(t1)
        timings = {
            "fetch_stars": t_fetch,
            "retrieve_and_rank": t_rank,
            "total": _ms(t0),
            "stars_cached": cached,
            **out.pop("_timings", {}),
        }
        record("github", timings, n_stars=len(stars), n_repos=len(out["repos"]))
        return {**out, "timings_ms": timings, "data_updated": cat.manifest.get("updated_at")}

    @app.post("/recommend/onboarding")
    def recommend_onboarding(body: OnboardingRequest, request: Request) -> dict[str, Any]:
        guard(request)
        t0 = time.perf_counter()
        known = {t["id"] for t in cat.topics}
        interests = [i for i in body.interests if i in known]
        if not interests and not body.languages:
            raise HTTPException(422, "pick at least one language or interest")
        out = ranking.recommend_onboarding(cat, body.languages, interests, body.hours)
        timings = {"retrieve_and_rank": _ms(t0), "total": _ms(t0)}
        record("onboarding", timings, n_repos=len(out["repos"]))
        return {**out, "timings_ms": timings, "data_updated": cat.manifest.get("updated_at")}

    @app.post("/explain")
    def explain(body: ExplainRequest, request: Request) -> dict[str, Any]:
        guard(request)
        t0 = time.perf_counter()
        facts = collect_facts(cat, body.repo_id, body.issue_number, body.co_starred, body.skills)
        if facts is None:
            raise HTTPException(404, "unknown repository")
        out = explainer.explain(facts)
        timings = {"explain": _ms(t0)}
        record("explain", timings, source=out["source"])
        return {**out, "timings_ms": timings}

    return app
