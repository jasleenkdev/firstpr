"""'Why this?' explanations, grounded in real signals only.

Facts are assembled server-side from the catalog (README summary / description, labels, the
issue title, skill overlap, co-star patterns from GH Archive): never from free text the client
sends. Client-supplied signals are re-validated against the catalog. The LLM may only rephrase
the listed facts; if Groq is rate-limited, down or not configured, a template built from the same
facts is returned. Prompts contain repo / issue facts only, never a username.
"""

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any

import requests

from .catalog import Catalog

PROMPT = """You explain to a student why an open-source repository was recommended for their \
first contribution. Use ONLY the facts below; do not add any other claim. Write 2 or 3 short \
sentences in plain English, addressed to the student as "you"; never refer to yourself (no "I" \
or "we"). Mention the most specific facts (shared stars, matching skills, the issue).

Facts:
{facts}"""


def collect_facts(
    cat: Catalog, repo_id: int, issue_number: int | None, co_starred: list[str], skills: list[str]
) -> list[str] | None:
    i = cat.index.get(repo_id)
    fresh = next((f for f in cat.fresh if f["id"] == repo_id), None)
    if i is None and fresh is None:
        return None
    repo = cat.repos[i] if i is not None else fresh
    feats = cat.features.get(repo_id, {})
    lang = repo.get("language") or "unknown"
    facts = [f"Repository: {repo['name'].split('/')[-1]} (main language: {lang})."]
    if feats.get("summary"):
        facts.append(f"README summary: {feats['summary']}")
    elif repo.get("description"):
        facts.append(f"Description: {repo['description']}")
    # co-star claims only for real co-star neighbours of this repo (client input is not trusted)
    neigh = {
        cat.repos[cat.index[n]]["name"] for n, _ in cat.costar.get(repo_id, []) if n in cat.index
    }
    shared = [n for n in co_starred if n in neigh][:3]
    if shared:
        facts.append(
            "People who starred "
            + ", ".join(shared)
            + " (which you starred) also starred this repository."
        )
    repo_sk = {s.lower() for s in feats.get("skills", [])} | {(repo.get("language") or "").lower()}
    repo_sk |= {t.lower() for t in repo.get("topics", [])}
    match = [s for s in skills if s.lower() in repo_sk][:5]
    if match:
        facts.append("Skills you have that the project uses: " + ", ".join(match) + ".")
    pool = cat.issues.get(repo_id, []) if i is not None else (fresh or {}).get("issues", [])
    issue = next((x for x in pool if x["number"] == issue_number), None)
    if issue:
        labels = ", ".join(issue.get("labels", [])[:4])
        facts.append(f'Suggested issue: "{issue["title"]}" (labels: {labels}).')
        if issue.get("difficulty"):
            facts.append(f"Estimated difficulty: {issue['difficulty']}.")
    return facts


REPO_NAME = re.compile(r"\b[\w.-]+/[\w.-]+\b")


def names_supported(text: str, facts: list[str]) -> bool:
    """Every owner/repo name in the text appears verbatim in the facts (phase-7 check found
    a mangled co-starred repo name in an LLM explanation)."""
    joined = " ".join(facts).lower()
    return all(n.lower() in joined for n in REPO_NAME.findall(text))


def template(facts: list[str]) -> str:
    return " ".join(facts[1:4]) if len(facts) > 1 else facts[0]


class Explainer:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout: float = 8.0,
        cache_size: int = 2048,
        daily_cap: int = 400,
    ) -> None:
        """`daily_cap`: LLM calls per UTC day (per process); beyond it, template explanations."""
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.daily_cap, self.count, self.day = daily_cap, 0, ""
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = cache_size
        self.blocked_until = 0.0  # after a 429, skip Groq until then

    def explain(self, facts: list[str]) -> dict[str, Any]:
        key = hashlib.sha256(json.dumps(facts).encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return {"text": self.cache[key], "source": "cache", "facts": facts}
        text, source = None, "template"
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self.day:
            self.day, self.count = today, 0
        if self.api_key and time.time() >= self.blocked_until and self.count < self.daily_cap:
            self.count += 1
            text = self._groq(PROMPT.format(facts="\n".join(f"- {f}" for f in facts)))
            if text and not names_supported(text, facts):
                text = None  # an unverifiable repo name: fall back to the template
            source = "llm" if text else "template"
        if not text:
            return {"text": template(facts), "source": "template", "facts": facts}
        self.cache[key] = text
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return {"text": text, "source": source, "facts": facts}

    def _groq(self, prompt: str) -> str | None:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 160,
                    "reasoning_effort": "none",
                },
                timeout=self.timeout,
            )
        except requests.RequestException:
            return None
        if r.status_code == 429:
            self.blocked_until = time.time() + float(r.headers.get("retry-after", 60))
            return None
        if r.status_code != 200:
            return None
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text or None
