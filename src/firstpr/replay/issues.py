"""Issue-level replay: which open beginner issue did the newcomer's first PR close?

Links come from GitHub GraphQL `closingIssuesReferences` (PRs that say "fixes #N" or are linked
in the UI). PR authors are numeric user ids, salted-hashed in memory with the GH Archive salt to
match cohort users; nothing raw is written (the GraphQL cache is in memory only).

State at PR time, where the data allows:
- candidate issues = issues with the repo's beginner labels created before the PR and not closed
  before it (createdAt / closedAt are exact);
- labels are *current* labels (label history is not fetched) -> a label added after the PR leaks;
- assignees and comment counts are *current* -> leak (the newcomer is often assigned afterwards)
  -> not used; LLM difficulty exists only for currently open issues -> not used.
Features used: label difficulty, skill overlap (issue labels + repo language vs the languages of
the user's pre-PR repos), issue age at PR time.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from firstpr.github.api import GraphQLClient, repo_node_id
from firstpr.github.privacy import hash_ids
from firstpr.serve.ranking import difficulty_match
from firstpr.serve.refresh import label_difficulty, label_skills

PR_QUERY = """query($id: ID!, $after: String) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on Repository {
    pullRequests(first: 50, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number createdAt author { ... on User { databaseId } }
              closingIssuesReferences(first: 3) { nodes { number } } } } } } }"""

ISSUE_QUERY = """query($id: ID!, $after: String, $labels: [String!]) {
  rateLimit { cost remaining resetAt }
  node(id: $id) { ... on Repository {
    issues(first: 100, after: $after, filterBy: {labels: $labels},
           orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number createdAt closedAt labels(first: 10) { nodes { name } } } } } } }"""


def _ts(s: str | None) -> float | None:
    return pd.Timestamp(s).timestamp() if s else None


def linked_prs(
    gql: GraphQLClient, repo_id: int, since: float, salt: bytes, max_pages: int = 8
) -> list[dict[str, Any]]:
    """PRs created since `since` that close >= 1 issue; author = salted hash of the user id."""
    out, after = [], None
    for _ in range(max_pages):
        body = gql.run(PR_QUERY, {"id": repo_node_id(repo_id), "after": after})
        conn = ((body["data"].get("node") or {}).get("pullRequests")) or {}
        oldest = None
        for n in conn.get("nodes", []):
            oldest = _ts(n["createdAt"])
            refs = [x["number"] for x in n["closingIssuesReferences"]["nodes"]]
            uid = (n.get("author") or {}).get("databaseId")
            if refs and uid is not None and oldest >= since:
                out.append(
                    {
                        "repo_id": repo_id,
                        "pr": n["number"],
                        "created": oldest,
                        "user": hash_ids([uid], salt)[0],
                        "issues": refs,
                    }
                )
        if not conn.get("pageInfo", {}).get("hasNextPage") or (
            oldest is not None and oldest < since
        ):
            break
        after = conn["pageInfo"]["endCursor"]
    return out


def candidate_issues(
    gql: GraphQLClient, repo_id: int, labels: list[str], before: float, max_pages: int = 5
) -> list[dict[str, Any]]:
    """Issues carrying any of `labels`, newest first, back to one year before `before`."""
    out, after = [], None
    for _ in range(max_pages):
        body = gql.run(ISSUE_QUERY, {"id": repo_node_id(repo_id), "after": after, "labels": labels})
        conn = ((body["data"].get("node") or {}).get("issues")) or {}
        oldest = None
        for n in conn.get("nodes", []):
            oldest = _ts(n["createdAt"])
            out.append(
                {
                    "number": n["number"],
                    "created": oldest,
                    "closed": _ts(n["closedAt"]),
                    "labels": [x["name"] for x in n["labels"]["nodes"]],
                }
            )
        if not conn.get("pageInfo", {}).get("hasNextPage") or (
            oldest and oldest < before - 365 * 86400
        ):
            break
        after = conn["pageInfo"]["endCursor"]
    return out


def collect(
    gql: GraphQLClient,
    cases: pd.DataFrame,
    labels: dict[int, list[str]],
    since: float,
    salt: bytes,
    workers: int = 3,
) -> list[dict[str, Any]]:
    """For each repo in `cases` (columns user, repo_id, pr_ts): the cohort user's linked PR and
    the candidate issues open at that PR's time."""

    def one(repo_id: int) -> list[dict[str, Any]]:
        prs = linked_prs(gql, repo_id, since, salt)
        want = cases.loc[cases["repo_id"] == repo_id]
        matched = []
        for row in want.itertuples():
            mine = [
                p
                for p in prs
                if p["user"] == row.user and abs(p["created"] - row.pr_ts) < 2 * 86400
            ]
            if not mine:
                continue
            pr = min(mine, key=lambda p: p["created"])
            matched.append(
                {
                    "user": row.user,
                    "repo_id": repo_id,
                    "pr_ts": pr["created"],
                    "linked": pr["issues"],
                }
            )
        if not matched:
            return []
        cands = candidate_issues(gql, repo_id, labels[repo_id], min(m["pr_ts"] for m in matched))
        for m in matched:
            m["candidates"] = [
                c
                for c in cands
                if c["created"] < m["pr_ts"] and (c["closed"] is None or c["closed"] > m["pr_ts"])
            ]
        return matched

    with ThreadPoolExecutor(max_workers=workers) as pool:
        repos = sorted(int(r) for r in cases["repo_id"].unique())
        return [m for part in pool.map(one, repos) for m in part]


def score_issues(
    cands: list[dict[str, Any]],
    pr_ts: float,
    user_skills: set[str],
    repo_lang: str,
    w: dict[str, float],
    target: str = "easy",
) -> np.ndarray:
    """Phase-6 issue score restricted to features known at PR time (see module docstring)."""
    out = []
    for c in cands:
        skills = set(label_skills(c["labels"])) | ({repo_lang.lower()} if repo_lang else set())
        skill = len(skills & user_skills) / max(1, min(len(skills), 4))
        age = max(0.0, (pr_ts - c["created"]) / 86400)
        out.append(
            w["diff"] * difficulty_match(label_difficulty(c["labels"]), target)
            + w["skill"] * min(1.0, skill)
            + w["recent"] * math.exp(-age / w["half_life"])
        )
    return np.array(out)
