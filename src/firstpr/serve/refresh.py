"""Daily refresh of the API's dynamic data (GitHub GraphQL + Groq; no BigQuery).

Outputs (`<out>/dynamic/`, uploaded to the HF dataset by `scripts/refresh_serving.py`):
- issues.json.gz       open beginner-labelled issues of catalog repos, with difficulty / skills
                       (LLM when available, label heuristic otherwise) and a scrubbed snippet
- repo_features.json   LLM README summary / skills / domain per repo (accumulates)
- issue_features.json  LLM difficulty / skills per issue, keyed "repo_id#number" (accumulates)
- fresh.json.gz        new repos (created <= `fresh_max_age_days` ago) with beginner issues opened
                       in the last `fresh_issue_days` days (GraphQL search), for the cold-start
                       "new projects" section
- manifest.json        timestamp and counts
No query asks for people (no issue authors, assignee logins or commenters).
"""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from firstpr.github import features as F
from firstpr.github.api import GraphQLClient
from firstpr.github.privacy import scrub_text
from firstpr.github.scope import beginner_pattern
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

EASY = (
    "good first issue",
    "first-timers-only",
    "first timers only",
    "beginner",
    "easy",
    "starter",
    "difficulty: easy",
    "level: easy",
    "good-first-issue",
    "good first contribution",
)
HARD = ("hard", "difficulty: hard", "complex", "advanced", "level: hard")
MEDIUM = ("medium", "intermediate", "difficulty: medium", "level: medium")
SKILL_WORDS = {
    "documentation": "documentation",
    "docs": "documentation",
    "test": "testing",
    "tests": "testing",
    "testing": "testing",
    "ui": "frontend",
    "ux": "frontend",
    "frontend": "frontend",
    "css": "css",
    "backend": "backend",
    "api": "api",
    "ci": "ci",
    "translation": "translation",
    "i18n": "translation",
    "accessibility": "accessibility",
    "a11y": "accessibility",
    "performance": "performance",
    "security": "security",
    "design": "design",
    "refactor": "refactoring",
    "typescript": "typescript",
    "javascript": "javascript",
    "python": "python",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "docker": "docker",
    "sql": "sql",
}


def label_difficulty(labels: list[str]) -> str | None:
    low = [x.lower() for x in labels]
    if any(h in x for x in low for h in HARD):
        return "hard"
    if any(m in x for x in low for m in MEDIUM):
        return "medium"
    if any(e in x for x in low for e in EASY):
        return "easy"
    return None


def label_colors(labels: list[str], colors: dict[str, Any] | None) -> dict[str, str]:
    """GitHub label colours (6-digit hex without '#') for the kept labels; unknown ones omitted."""
    colors = colors or {}
    return {
        lab: f"#{colors[lab].lower()}"
        for lab in labels
        if isinstance(colors.get(lab), str) and len(colors[lab]) == 6
    }


def label_skills(labels: list[str]) -> list[str]:
    out = []
    for lab in labels:
        for w in lab.lower().replace(":", " ").replace("/", " ").replace("-", " ").split():
            if w in SKILL_WORDS and SKILL_WORDS[w] not in out:
                out.append(SKILL_WORDS[w])
    return out


def issue_record(repo_id: int, iss: dict[str, Any], llm: dict[str, Any] | None) -> dict[str, Any]:
    labels = list(iss["labels"])
    diff = (llm or {}).get("difficulty")
    return {
        "repo_id": int(repo_id),
        "number": int(iss["number"]),
        "title": scrub_text(iss["title"], 200),
        "labels": labels[:8],
        "label_colors": label_colors(labels[:8], iss.get("label_colors")),
        "created_at": iss["created_at"],
        "updated_at": iss["updated_at"],
        "n_comments": int(iss["n_comments"]),
        "n_assignees": int(iss["n_assignees"]),
        "difficulty": diff or label_difficulty(labels),
        "difficulty_source": "llm" if diff else ("labels" if label_difficulty(labels) else None),
        "skills": list((llm or {}).get("skills") or label_skills(labels))[:4],
        "snippet": scrub_text(iss.get("body") or "", 300),
    }


def catalog_labels(
    gql: GraphQLClient, repo_ids: list[int], labels_cfg: list[str]
) -> dict[int, list[str]]:
    """Beginner-style label names defined on each repo (exact names, for the issue filter)."""
    meta = gql.repo_labels(repo_ids)
    pattern = beginner_pattern(labels_cfg)
    out = {}
    for rid, names in zip(meta["repo_id"], meta["labels"], strict=True):
        hits = [n for n in names if pattern.search(n)]
        if hits:
            out[int(rid)] = hits
    return out


FRESH_QUERY = """query($q: String!, $after: String) {
  rateLimit { cost remaining resetAt }
  search(query: $q, type: ISSUE, first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on Issue {
      number title body createdAt updatedAt
      labels(first: 10) { nodes { name color } } comments { totalCount } assignees { totalCount }
      repository { databaseId nameWithOwner description createdAt stargazerCount isFork
        isArchived primaryLanguage { name }
        repositoryTopics(first: 8) { nodes { topic { name } } } }
    } }
  }
}"""


def fresh_repos(
    gql: GraphQLClient,
    now: datetime,
    issue_days: int,
    max_age_days: int,
    min_stars: int,
    labels: tuple[str, ...] = ("good first issue", "help wanted"),
    max_pages: int = 5,
) -> list[dict[str, Any]]:
    since = (now - timedelta(days=issue_days)).strftime("%Y-%m-%d")
    born = now - timedelta(days=max_age_days)
    repos: dict[int, dict[str, Any]] = {}
    for label in labels:
        q = f'is:issue is:open label:"{label}" created:>={since} sort:created-desc'
        after = None
        for _ in range(max_pages):
            body = gql.run(FRESH_QUERY, {"q": q, "after": after})
            s = body["data"]["search"]
            for n in s["nodes"]:
                r = (n or {}).get("repository")
                if not r or r["isFork"] or r["isArchived"] or r["stargazerCount"] < min_stars:
                    continue
                if datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) < born:
                    continue
                rid = int(r["databaseId"])
                rec = repos.setdefault(
                    rid,
                    {
                        "id": rid,
                        "name": r["nameWithOwner"],
                        "description": scrub_text(r.get("description") or "", 300),
                        "language": (r.get("primaryLanguage") or {}).get("name"),
                        "topics": [t["topic"]["name"] for t in r["repositoryTopics"]["nodes"]],
                        "stars": r["stargazerCount"],
                        "created_at": r["createdAt"],
                        "issues": [],
                    },
                )
                iss = {
                    "number": n["number"],
                    "title": n["title"],
                    "body": n.get("body") or "",
                    "created_at": n["createdAt"],
                    "updated_at": n["updatedAt"],
                    "labels": [x["name"] for x in n["labels"]["nodes"]],
                    "label_colors": {x["name"]: x.get("color") for x in n["labels"]["nodes"]},
                    "n_comments": n["comments"]["totalCount"],
                    "n_assignees": n["assignees"]["totalCount"],
                }
                if all(x["number"] != iss["number"] for x in rec["issues"]):
                    rec["issues"].append(issue_record(rid, iss, None))
            if not s["pageInfo"]["hasNextPage"]:
                break
            after = s["pageInfo"]["endCursor"]
    log.info("fresh pool: %d new repos with recent beginner issues", len(repos))
    return sorted(repos.values(), key=lambda r: -r["stars"])


def write_dynamic(
    out: Path,
    issues: list[dict],
    repo_feats: dict,
    issue_feats: dict,
    fresh: list[dict],
    now: datetime,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "issues.json.gz", "wt") as f:
        json.dump(issues, f)
    with gzip.open(out / "fresh.json.gz", "wt") as f:
        json.dump(fresh, f)
    (out / "repo_features.json").write_text(json.dumps(repo_feats))
    (out / "issue_features.json").write_text(json.dumps(issue_feats))
    manifest = {
        "updated_at": now.isoformat(timespec="seconds"),
        "open_issues": len(issues),
        "repos_with_issues": len({i["repo_id"] for i in issues}),
        "repos_with_llm_features": len(repo_feats),
        "issues_with_llm_features": len(issue_feats),
        "fresh_repos": len(fresh),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("dynamic data: %s", manifest)


def features_from_phase5(gdir: Path) -> tuple[dict, dict]:
    """Seed the accumulating feature files with phase-5 Groq outputs."""
    fdir = gdir / "features"
    rf = pd.read_parquet(fdir / "repo_features.parquet")
    repo_feats = {
        str(r.repo_id): {"summary": r.summary, "skills": list(r.skills), "domain": r.domain}
        for r in rf.itertuples()
        if r.parsed and r.summary
    }
    isf = pd.read_parquet(fdir / "issue_features.parquet")
    rated = pd.read_parquet(fdir / "issues_rated.parquet")[["issue_row", "repo_id", "number"]]
    isf = isf.merge(rated, on="issue_row")
    issue_feats = {
        f"{r.repo_id}#{r.number}": {"difficulty": r.difficulty, "skills": list(r.skills)}
        for r in isf.itertuples()
        if r.difficulty
    }
    return repo_feats, issue_feats


def backfill_llm(
    repos: list[dict],
    issues: list[dict],
    repo_feats: dict,
    issue_feats: dict,
    readmes: dict[int, str],
    lc: dict[str, Any],
    deadline: datetime,
) -> tuple[int, int]:
    """New Groq features for repos / issues without them (priority order given), until the
    daily limit or the deadline. Returns (new repo features, new issue features)."""
    from firstpr.llm.client import LLMClient

    opts = lc["options"]
    todo_r = [r for r in repos if str(r["id"]) not in repo_feats and readmes.get(r["id"])]
    prompts = [(r["id"], F.repo_prompt(readmes[r["id"]])) for r in todo_r[: lc["max_repos"]]]
    got = F.run_calls(LLMClient(lc["repo_model"], backend="groq"), prompts, opts, deadline=deadline)
    for _, row in F.repo_features(got).iterrows():
        if row["parsed"] and row["summary"]:
            repo_feats[str(row["repo_id"])] = {
                "summary": row["summary"],
                "skills": row["skills"],
                "domain": row["domain"],
            }
    by_repo: dict[int, list[dict]] = {}
    for iss in issues:
        if f"{iss['repo_id']}#{iss['number']}" not in issue_feats:
            by_repo.setdefault(iss["repo_id"], []).append(iss)
    names = {r["id"]: (r["name"].split("/")[-1], r.get("language") or "") for r in repos}
    batches, iprompts = {}, []
    for rid in [r["id"] for r in repos if r["id"] in by_repo][: lc["max_repos"]]:
        chunk = by_repo[rid][: lc["issues_per_call"]]
        blocks = [
            F.issue_block(
                j, x["title"], x["labels"], x.get("snippet", ""), lc["issue_body_max_chars"]
            )
            for j, x in enumerate(chunk, 1)
        ]
        batches[len(batches)] = [f"{rid}#{x['number']}" for x in chunk]
        iprompts.append((len(iprompts), F.issue_prompt(names[rid][0], names[rid][1], blocks)))
    igot = F.run_calls(
        LLMClient(lc["issue_model"], backend="groq"), iprompts, opts, deadline=deadline
    )
    n_issue = 0
    for _, row in F.issue_features(igot, batches).iterrows():
        if row["difficulty"]:
            issue_feats[row["issue_row"]] = {
                "difficulty": row["difficulty"],
                "skills": row["skills"],
            }
            n_issue += 1
    return len(got), n_issue
