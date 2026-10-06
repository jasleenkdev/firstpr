"""Onboarding-path check with student personas (configs/phase7_personas.yaml).

1. Each persona goes through the live onboarding endpoint (top 10 repos + their first issue).
2. An LLM judge (Groq, cached) rates each recommendation 0 / 1 / 2 for that persona, using only
   the shown repo facts and the persona's languages, interests and hours.
3. A blind sample (no judge scores) goes to notes/phase7-persona-labels.csv for hand labels;
   `scripts/persona_agreement.py` compares them with the judge once filled in.
Judge results -> results/analysis/persona_judge.csv (+ summary rows).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from firstpr.github.features import parse_json
from firstpr.llm.client import LLMClient
from firstpr.utils.io import load_yaml

JUDGE = """You check recommendations of open-source projects for a student's first contribution.
Student: {bio}. Languages: {langs}. Interests: {interests}. Time: {hours} hours per week.

Rate each recommendation for THIS student using only the facts shown:
2 = good fit (uses a language they know or clearly matches an interest, and the issue looks
doable in their time); 1 = partly fits; 0 = poor fit.

{items}

Return JSON {{"ratings": [{{"id": <number>, "score": 0|1|2, "reason": "<= 12 words"}}]}} with one
entry per recommendation."""

RUBRIC = "my_label: 2 = good fit, 1 = partly fits, 0 = poor fit (same rubric as the judge)"


def item_text(i: int, r: dict) -> str:
    iss = r["issues"][0] if r["issues"] else {}
    diff = iss.get("difficulty") or "unknown"
    return (
        f"[{i}] {r['name']} ({r.get('language') or 'unknown language'}): "
        f"{(r.get('summary') or r.get('description') or '')[:220]}\n"
        f'    First issue: "{iss.get("title", "")}" (difficulty: {diff})'
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://firstpr-api.vercel.app")
    p.add_argument("--n-label", type=int, default=60)
    p.add_argument("--judge", default="qwen/qwen3.8-27b", help="Groq model (own daily quota)")
    a = p.parse_args()
    personas = load_yaml("configs/phase7_personas.yaml")["personas"]
    topics = {
        t["id"]: t["label"]
        for t in requests.get(f"{a.api}/options", timeout=90).json()["interests"]
    }
    judge = LLMClient(a.judge, backend="groq")
    effort = "low" if "gpt-oss" in a.judge else "none"
    opts = {"temperature": 0, "num_predict": 900, "reasoning_effort": effort, "json": True}
    rows = []
    for pr in personas:
        body = {"languages": pr["languages"], "interests": pr["interests"], "hours": pr["hours"]}
        recs = requests.post(f"{a.api}/recommend/onboarding", json=body, timeout=60).json()["repos"]
        time.sleep(3.5)  # stay under the API's per-IP limit
        prompt = JUDGE.format(
            bio=pr["bio"],
            langs=", ".join(pr["languages"]) or "none yet",
            interests=", ".join(topics[t] for t in pr["interests"]),
            hours=pr["hours"],
            items="\n".join(item_text(i, r) for i, r in enumerate(recs, 1)),
        )
        ratings = {
            int(x.get("id", -1)): x
            for x in (parse_json(judge.generate(prompt, opts)) or {}).get("ratings", [])
            if isinstance(x, dict) and str(x.get("id", "")).isdigit()
        }
        for i, r in enumerate(recs, 1):
            iss = r["issues"][0] if r["issues"] else {}
            j = ratings.get(i, {})
            rows.append(
                {
                    "persona_id": pr["id"],
                    "persona": pr["bio"],
                    "languages": "; ".join(pr["languages"]),
                    "interests": "; ".join(topics[t] for t in pr["interests"]),
                    "hours": pr["hours"],
                    "rank": i,
                    "repo": r["name"],
                    "repo_url": r["url"],
                    "language": r.get("language") or "",
                    "description": (r.get("summary") or r.get("description") or "")[:300],
                    "issue_title": iss.get("title", ""),
                    "issue_url": iss.get("url", ""),
                    "issue_difficulty": iss.get("difficulty") or "",
                    "judge_score": j.get("score"),
                    "judge_reason": j.get("reason", ""),
                }
            )
    df = pd.DataFrame(rows)
    df["row_id"] = [f"r{i:03d}" for i in range(len(df))]
    df.to_csv("results/analysis/persona_judge.csv", index=False)
    s = df["judge_score"].astype(float)
    print(
        f"{len(personas)} personas, {len(df)} recommendations; judge mean {s.mean():.2f}, "
        f"good (2) {np.mean(s == 2):.2f}, at least partly (>=1) {np.mean(s >= 1):.2f}, "
        f"unrated {s.isna().sum()}"
    )
    print(df.groupby("persona_id")["judge_score"].mean().round(2).to_dict())
    rng = np.random.default_rng(0)
    sample = df.iloc[np.sort(rng.choice(len(df), min(a.n_label, len(df)), replace=False))]
    blind = sample.drop(columns=["judge_score", "judge_reason"]).assign(my_label="", my_note="")
    cols = [
        "row_id",
        "persona_id",
        "persona",
        "languages",
        "interests",
        "hours",
        "rank",
        "repo",
        "repo_url",
        "language",
        "description",
        "issue_title",
        "issue_url",
        "issue_difficulty",
        "my_label",
        "my_note",
    ]
    Path("notes").mkdir(exist_ok=True)
    blind[cols].to_csv("notes/phase7-persona-labels.csv", index=False)
    Path("notes/phase7-persona-labels.README.txt").write_text(
        RUBRIC + "\nFill my_label for every row; leave the other columns as they are. Judge scores "
        "are kept out of this file on purpose.\n"
    )
    print(json.dumps({"label_rows": len(blind)}))


if __name__ == "__main__":
    main()
