"""Retrieval and issue ranking.

Repo retrieval is driven by star signals (phase 5: the models predict stars; contribution
targets were too sparse to separate models):
- GitHub path: text-SASRec over the user's catalog stars (inductive: any catalog repo can be in
  the history or the output), blended with a text profile when few stars are in the catalog
  (the fallback ladder: >= 3 catalog stars -> model only; 1-2 -> blend; 0 -> text profile).
- Onboarding path: cosine between repo text embeddings and the chosen interests' embeddings,
  plus language and skill overlap.
Issue ranking is driven by contribution signals: difficulty match (hours per week -> target
difficulty), skill match, recency, unclaimed, and the repo's external-contributor activity.
"New projects" (cold start): repos created recently with fresh beginner issues from the daily
GitHub search pool, ranked by keyword overlap with the profile (collaborative models cannot
reach them: phase 5, cold NDCG ~0).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .catalog import Catalog
from .github import Star

DIFFICULTIES = ("easy", "medium", "hard")
N_REPOS, N_ISSUES, N_NEW = 10, 3, 3
MAX_STARRED, MAX_PER_OWNER = 3, 2
# issue score weights (hand-set: no labelled data for issue choice yet; phase 7 replay tunes them)
W_DIFF, W_SKILL, W_RECENT, W_FREE, W_REPO = 0.35, 0.25, 0.2, 0.1, 0.1
CLAIMED, BUSY = 0.3, 0.1  # penalties: someone is assigned (mostly taken); > 10 comments
TOKEN = re.compile(r"[a-z0-9+#.]+")


def tokens(text: str) -> set[str]:
    return {t.strip(".") for t in TOKEN.findall((text or "").lower()) if len(t.strip(".")) > 1}


def target_difficulty(hours: float) -> str:
    """Hours per week -> the difficulty a first contribution should aim at."""
    if hours <= 3:
        return "easy"
    if hours <= 8:
        return "medium"
    return "hard"


def difficulty_match(issue: str | None, target: str) -> float:
    if issue not in DIFFICULTIES:
        return 0.5
    gap = abs(DIFFICULTIES.index(issue) - DIFFICULTIES.index(target))
    # one level easier than the target is fine for a first contribution; harder is not
    if DIFFICULTIES.index(issue) < DIFFICULTIES.index(target):
        return 1.0 - 0.25 * gap
    return 1.0 - 0.6 * gap


def _age_days(ts: str | None, now: datetime) -> float:
    if not ts:
        return 365.0
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return max(0.0, (now - t).total_seconds() / 86400)


@dataclass
class Profile:
    skills: set[str] = field(default_factory=set)  # lower-case languages / topics / tools
    languages: list[str] = field(default_factory=list)  # display order
    hours: float = 5.0
    interest_ids: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return target_difficulty(self.hours)


def issue_skills(issue: dict[str, Any], repo: dict[str, Any]) -> set[str]:
    s = {x.lower() for x in issue.get("skills") or []}
    if repo.get("language"):
        s.add(repo["language"].lower())
    return s


def score_issue(
    issue: dict[str, Any],
    repo: dict[str, Any],
    profile: Profile,
    repo_activity: float,
    now: datetime,
) -> tuple[float, dict[str, Any]]:
    skills = issue_skills(issue, repo)
    overlap = sorted(skills & profile.skills)
    skill = len(overlap) / max(1, min(len(skills), 4))
    recent = math.exp(-_age_days(issue.get("updated_at") or issue.get("created_at"), now) / 30)
    free = 1.0 if not issue.get("n_assignees") else 0.0
    diff = difficulty_match(issue.get("difficulty"), profile.target)
    penalty = (CLAIMED if issue.get("n_assignees") else 0.0) + (
        BUSY if (issue.get("n_comments") or 0) > 10 else 0.0
    )
    score = (
        W_DIFF * diff
        + W_SKILL * min(1.0, skill)
        + W_RECENT * recent
        + W_FREE * free
        + W_REPO * repo_activity
        - penalty
    )
    return score, {"matched_skills": overlap, "difficulty_match": round(diff, 2)}


def pick_issues(
    cat: Catalog,
    repo: dict[str, Any],
    profile: Profile,
    now: datetime,
    issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pool = issues if issues is not None else cat.issues.get(repo["id"], [])
    act = math.log1p(repo.get("n_contrib", 0)) / math.log1p(500)
    scored = []
    for iss in pool:
        s, why = score_issue(iss, repo, profile, min(1.0, act), now)
        scored.append((s, iss, why))
    scored.sort(key=lambda x: (-x[0], -int(x[1]["number"])))
    out = []
    for s, iss, why in scored[:N_ISSUES]:
        out.append(
            {
                "number": iss["number"],
                "title": iss["title"],
                "url": f"https://github.com/{repo['name']}/issues/{iss['number']}",
                "labels": iss.get("labels", [])[:5],
                "difficulty": iss.get("difficulty"),
                "difficulty_source": iss.get("difficulty_source"),
                "skills": iss.get("skills", [])[:4],
                "created_at": iss.get("created_at"),
                "n_comments": iss.get("n_comments", 0),
                "score": round(s, 3),
                **why,
            }
        )
    return out


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def candidates(cat: Catalog) -> np.ndarray:
    """Catalog indices of repos that currently have open beginner issues."""
    return np.array([i for i, r in enumerate(cat.repos) if cat.issues.get(r["id"])], dtype=int)


def interest_vector(cat: Catalog, interest_ids: list[str]) -> np.ndarray | None:
    idx = [i for i, t in enumerate(cat.topics) if t["id"] in set(interest_ids)]
    if not idx:
        return None
    v = cat.topic_emb[idx].mean(0)
    return v / (np.linalg.norm(v) + 1e-12)


def infer_interests(cat: Catalog, words: set[str]) -> list[str]:
    """Interest ids whose keywords appear among the user's star topics / languages."""
    hits = Counter()
    for t in cat.topics:
        hits[t["id"]] = len(words & set(t["kw"]))
    return [k for k, v in hits.most_common(3) if v > 0]


def repo_skills(cat: Catalog, repo: dict[str, Any]) -> set[str]:
    f = cat.features.get(repo["id"], {})
    s = {x.lower() for x in f.get("skills", [])}
    s |= {x.lower() for x in repo.get("languages", [])[:3]}
    s |= {x.lower() for x in repo.get("topics", [])}
    return s


def _select(
    cat: Catalog, order: np.ndarray, starred: set[int], profile: Profile, now: datetime
) -> list[tuple[int, list[dict[str, Any]]]]:
    picked, n_starred, owners = [], 0, Counter()
    for i in order:
        repo = cat.repos[int(i)]
        owner = repo["name"].split("/")[0].lower()
        if owners[owner] >= MAX_PER_OWNER:
            continue
        if repo["id"] in starred:
            if n_starred >= MAX_STARRED:
                continue
            n_starred += 1
        issues = pick_issues(cat, repo, profile, now)
        if not issues:
            continue
        owners[owner] += 1
        picked.append((int(i), issues))
        if len(picked) == N_REPOS:
            break
    return picked


def _repo_card(
    cat: Catalog, i: int, issues: list[dict[str, Any]], score: float, signals: dict[str, Any]
) -> dict[str, Any]:
    r = cat.repos[i]
    f = cat.features.get(r["id"], {})
    return {
        "id": r["id"],
        "name": r["name"],
        "url": f"https://github.com/{r['name']}",
        "description": r["description"],
        "summary": f.get("summary") or r["description"],
        "summary_source": "llm" if f.get("summary") else "description",
        "language": r["language"],
        "topics": r["topics"][:5],
        "stars": r["stars"],
        "skills": (f.get("skills") or [x for x in [r["language"], *r["topics"][:3]] if x])[:6],
        "score": round(float(score), 3),
        "signals": signals,
        "issues": issues,
    }


def recommend_github(
    cat: Catalog, stars: list[Star], hours: float, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    ordered = sorted(stars, key=lambda s: s.starred_at)  # oldest -> newest
    hist = [cat.index[s.repo_id] for s in ordered if s.repo_id in cat.index]
    lang_counts = Counter(s.language for s in stars if s.language)
    words = {t.lower() for s in stars for t in s.topics} | {x.lower() for x in lang_counts}
    profile = Profile(
        skills=words,
        languages=[x for x, _ in lang_counts.most_common(5)],
        hours=hours,
        interest_ids=infer_interests(cat, words),
    )
    cand = candidates(cat)
    text_vec = None
    parts = []
    if hist:
        parts.append(cat.text_emb[hist[-50:]].mean(0))
    iv = interest_vector(cat, profile.interest_ids)
    if iv is not None:
        parts.append(iv)
    if parts:
        text_vec = np.mean(parts, axis=0)
        text_vec /= np.linalg.norm(text_vec) + 1e-12
    w_model = min(1.0, len(hist) / 3)
    score = np.zeros(len(cand))
    if hist:
        u = cat.encoder.encode(np.array(hist), cat.item_vecs)
        score += w_model * _z(cat.item_vecs[cand] @ u)
    if text_vec is not None:
        score += (1 - w_model) * _z(cat.text_emb[cand] @ text_vec)
    langs = {x.lower() for x in profile.languages}
    score += 0.15 * np.array([(cat.repos[i]["language"] or "").lower() in langs for i in cand])
    mode = "model" if w_model == 1 else ("blend" if hist else "text_profile")
    starred = {s.repo_id for s in stars}
    order = cand[np.argsort(-score)]
    picked = _select(cat, order, starred, profile, now)
    pos = {int(c): k for k, c in enumerate(cand)}
    hist_ids = {cat.repos[i]["id"] for i in hist}
    repos = []
    for i, issues in picked:
        r = cat.repos[i]
        costarred = [
            cat.repos[cat.index[n]]["name"]
            for n, _ in cat.costar.get(r["id"], [])
            if n in hist_ids and n in cat.index
        ][:3]
        signals = {
            "starred_by_you": r["id"] in starred,
            "co_starred_with": costarred,
            "matched_skills": sorted(repo_skills(cat, r) & profile.skills)[:5],
            "retrieval": mode,
        }
        repos.append(_repo_card(cat, i, issues, score[pos[i]], signals))
    return {
        "repos": repos,
        "new_projects": new_projects(cat, profile, now),
        "profile": {
            "languages": profile.languages,
            "interests": profile.interest_ids,
            "target_difficulty": profile.target,
            "stars_seen": len(stars),
            "stars_in_catalog": len(hist),
            "retrieval": mode,
        },
    }


def recommend_onboarding(
    cat: Catalog,
    languages: list[str],
    interests: list[str],
    hours: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    kw = {w for t in cat.topics if t["id"] in set(interests) for w in t["kw"]}
    profile = Profile(
        skills={x.lower() for x in languages} | kw,
        languages=languages,
        hours=hours,
        interest_ids=interests,
    )
    cand = candidates(cat)
    iv = interest_vector(cat, interests)
    score = np.zeros(len(cand))
    if iv is not None:
        score += 0.6 * _z(cat.text_emb[cand] @ iv)
    langs = {x.lower() for x in languages}
    if langs:
        score += 0.6 * np.array([(cat.repos[i]["language"] or "").lower() in langs for i in cand])
    skill = np.array([len(repo_skills(cat, cat.repos[i]) & profile.skills) for i in cand])
    score += 0.2 * np.minimum(skill, 3) / 3
    score += 0.1 * _z(np.log1p([cat.repos[i]["n_contrib"] for i in cand]))
    order = cand[np.argsort(-score)]
    picked = _select(cat, order, set(), profile, now)
    pos = {int(c): k for k, c in enumerate(cand)}
    repos = []
    for i, issues in picked:
        r = cat.repos[i]
        signals = {
            "starred_by_you": False,
            "co_starred_with": [],
            "matched_skills": sorted(repo_skills(cat, r) & profile.skills)[:5],
            "matched_interests": [
                t["label"]
                for t in cat.topics
                if t["id"] in set(interests) and set(t["kw"]) & repo_skills(cat, r)
            ],
            "retrieval": "onboarding",
        }
        repos.append(_repo_card(cat, i, issues, score[pos[i]], signals))
    return {
        "repos": repos,
        "new_projects": new_projects(cat, profile, now),
        "profile": {
            "languages": languages,
            "interests": interests,
            "target_difficulty": profile.target,
            "retrieval": "onboarding",
        },
    }


def new_projects(cat: Catalog, profile: Profile, now: datetime) -> list[dict[str, Any]]:
    """Cold start: recently created repos with fresh beginner issues (daily search pool)."""
    scored = []
    for f in cat.fresh:
        words = tokens(f.get("description", "")) | {t.lower() for t in f.get("topics", [])}
        if f.get("language"):
            words.add(f["language"].lower())
        overlap = sorted(words & profile.skills)
        lang = (f.get("language") or "").lower() in {x.lower() for x in profile.languages}
        s = len(overlap) + 2 * lang + math.exp(-_age_days(f.get("created_at"), now) / 60)
        if overlap or lang:
            scored.append((s, f, overlap))
    scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
    out = []
    for s, f, overlap in scored[:N_NEW]:
        issues = pick_issues(cat, f, profile, now, issues=f.get("issues", []))
        if issues:
            out.append(
                {
                    "id": f["id"],
                    "name": f["name"],
                    "url": f"https://github.com/{f['name']}",
                    "description": f.get("description", ""),
                    "language": f.get("language"),
                    "stars": f.get("stars", 0),
                    "created_at": f.get("created_at"),
                    "score": round(s, 3),
                    "signals": {"matched_skills": overlap[:5], "retrieval": "new_project_keywords"},
                    "issues": issues,
                }
            )
    return out
