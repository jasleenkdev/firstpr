"""LLM features for FirstPR (KAR-style feature engineering, Lin et al. TOIS 2025: "LLM as feature
engineer"): README summary + skills per repo, difficulty + skills per open beginner issue.

Prompts are grounded: they hold only scrubbed repo / issue text (no URLs, emails, @mentions, no
owner names) and ask for facts stated in it; issue ratings must quote their evidence, so
grounding can be checked mechanically (`grounding_checks`) as well as by hand.
Groq's free tier allows ~1,000 requests and ~200k tokens per model per day: `run_calls` sleeps
until the limit resets and resumes; every response is cached, so a restart repeats nothing.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from firstpr.github.privacy import scrub_text
from firstpr.llm.client import LLMClient, RateLimitedError
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

REPO_PROMPT = """You describe open-source repositories for students looking for a first \
contribution. Use ONLY facts stated in the repository text below. Do not guess; leave out \
anything the text does not say.

Repository text:
<<<
{text}
>>>

Return a JSON object with exactly these keys:
"summary": one or two plain sentences on what the project does,
"skills": a list of up to 6 programming languages, frameworks or tools a contributor needs, \
each named in the text,
"domain": a topic of 1 to 3 words."""

ISSUE_PROMPT = """You rate GitHub issues for students looking for a first contribution. Use \
ONLY the repository facts and issue text below; do not guess beyond them.

Repository: {repo} (main language: {language})
{issues}

Return a JSON object {{"issues": [...]}} with one entry per issue, each with exactly these keys:
"id": the number in brackets,
"difficulty": "easy", "medium" or "hard" for a student new to this project,
"skills": a list of up to 4 skills the issue needs, each named in the issue or repository facts,
"evidence": up to 15 words copied exactly from the issue text that justify the difficulty."""


def repo_prompt(text: str) -> str:
    return REPO_PROMPT.format(text=text)


def issue_block(i: int, title: str, labels: list[str], body: str, max_chars: int) -> str:
    return (
        f"[{i}] Title: {scrub_text(title)}\n"
        f"Labels: {', '.join(labels)}\n"
        f"Text: {scrub_text(body, max_chars) or '(empty)'}"
    )


def issue_prompt(repo: str, language: str, blocks: list[str]) -> str:
    return ISSUE_PROMPT.format(
        repo=repo, language=language or "unknown", issues="\n\n".join(blocks)
    )


def parse_json(response: str) -> dict[str, Any] | None:
    """First JSON object in a response (models sometimes wrap it in prose or fences)."""
    m = re.search(r"\{.*\}", response, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def run_calls(
    client: LLMClient,
    prompts: list[tuple[Any, str]],
    options: dict[str, Any],
    on_progress: Callable[[int], None] | None = None,
    max_sleep: float = 6 * 3600,
    deadline: datetime | None = None,
) -> dict[Any, str]:
    """Responses for (key, prompt) pairs; cached prompts cost nothing. On a daily limit the
    loop sleeps until reset (up to `max_sleep` per wait) and resumes; it returns early (partial)
    when the wait would pass `deadline`."""
    out: dict[Any, str] = {}
    todo = [(k, p) for k, p in prompts if client.cached(p, options) is None]
    log.info("llm: %d prompts, %d not cached", len(prompts), len(todo))
    for k, p in prompts:
        if (hit := client.cached(p, options)) is not None:
            out[k] = hit
    for n, (k, p) in enumerate(todo, 1):
        while True:
            try:
                out[k] = client.generate(p, options)
                break
            except RateLimitedError as e:
                wait = min(e.wait_seconds + 30, max_sleep)
                if deadline and datetime.now() + timedelta(seconds=wait) > deadline:
                    log.info(
                        "llm limit: reset is past the deadline, stopping (%d new calls)", n - 1
                    )
                    return out
                log.info("llm daily limit after %d new calls: sleeping %.0f min", n - 1, wait / 60)
                time.sleep(wait)
        if on_progress and n % 25 == 0:
            on_progress(n)
    return out


def repo_features(responses: dict[int, str]) -> pd.DataFrame:
    rows = []
    for rid, resp in responses.items():
        obj = parse_json(resp) or {}
        skills = obj.get("skills") if isinstance(obj.get("skills"), list) else []
        rows.append(
            {
                "repo_id": rid,
                "summary": str(obj.get("summary", "")).strip(),
                "skills": [str(s).strip() for s in skills][:6],
                "domain": str(obj.get("domain", "")).strip(),
                "parsed": bool(obj),
            }
        )
    return pd.DataFrame(rows)


def issue_features(responses: dict[int, str], batches: dict[int, list[int]]) -> pd.DataFrame:
    """responses / batches keyed by batch id; batches map to the issue row ids, in prompt order
    (prompt ids are 1-based positions)."""
    rows = []
    for bid, resp in responses.items():
        obj = parse_json(resp) or {}
        entries = obj.get("issues") if isinstance(obj.get("issues"), list) else []
        by_id = {}
        for e in entries:
            if isinstance(e, dict):
                try:
                    by_id[int(e.get("id"))] = e
                except (TypeError, ValueError):
                    continue
        for pos, row_id in enumerate(batches[bid], 1):
            e = by_id.get(pos, {})
            diff = str(e.get("difficulty", "")).lower().strip()
            skills = e.get("skills") if isinstance(e.get("skills"), list) else []
            rows.append(
                {
                    "issue_row": row_id,
                    "difficulty": diff if diff in ("easy", "medium", "hard") else None,
                    "skills": [str(s).strip() for s in skills][:4],
                    "evidence": str(e.get("evidence", "")).strip(),
                }
            )
    return pd.DataFrame(rows)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#.]+", " ", s.lower()).split())


def skill_grounded(skill: str, source: str) -> bool:
    """A skill counts as grounded when its normalised name occurs in the normalised source."""
    s = _norm(skill)
    return bool(s) and s in _norm(source)


def evidence_grounded(evidence: str, source: str) -> bool:
    """Evidence must be copied: every 4-word window of it occurs in the source (tolerates
    trimmed punctuation and an elided middle)."""
    words = _norm(evidence).split()
    if not words:
        return False
    src = _norm(source)
    if len(words) < 4:
        return " ".join(words) in src
    windows = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return sum(w in src for w in windows) / len(windows) >= 0.5
