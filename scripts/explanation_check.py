"""Faithfulness of live "Why this?" explanations.

Collects 50 explanations from the live API (40 from persona onboarding results, 10 from the
GitHub path for public accounts that own catalog repos, chosen at runtime and never stored) and
checks automatically that every checkable term in the text is supported by the facts the API
returned with it: languages / skills / interests, difficulty words, repo names, quoted issue
titles, numbers, and any "starred" claim (needs a co-star fact).
-> results/analysis/explanation_faithfulness.csv, notes/phase7-explanation-labels.csv (manual).
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from firstpr.utils.env import require_env
from firstpr.utils.io import load_yaml

GH = {"Authorization": f"Bearer {require_env('GITHUB_TOKEN')}"}
DIFFICULTY = ["easy", "medium", "hard"]


def vocabulary(api: str) -> set[str]:
    opts = requests.get(f"{api}/options", timeout=90).json()
    words = {x.lower() for x in opts["languages"]} | {t["label"].lower() for t in opts["interests"]}
    words |= {
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "c++",
        "c#",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "dart",
        "shell",
        "html",
        "css",
        "sql",
        "documentation",
        "testing",
        "docker",
        "kubernetes",
        "react",
        "pytorch",
        "machine learning",
        "translation",
    }
    return words


def norm(x: str) -> str:
    """Lower case, hyphens / underscores as spaces, collapsed whitespace."""
    return " ".join(re.sub(r"[-_]", " ", x.lower()).split())


def check(text: str, facts: list[str], vocab: set[str]) -> list[str]:
    """Checkable terms in `text` that the facts do not contain (after `norm`; punctuation the
    writer put inside quotes is ignored)."""
    t, f = norm(text), norm(" ".join(facts))
    vocab = {norm(w) for w in vocab}
    missing = []
    for w in sorted(vocab):
        if re.search(rf"(?<![\w+#]){re.escape(w)}(?![\w+#])", t) and w not in f:
            missing.append(w)
    missing += [d for d in DIFFICULTY if re.search(rf"\b{d}\b", t) and d not in f]
    missing += [n for n in re.findall(r"\b[\w.-]+/[\w.-]+\b", text) if norm(n) not in f]
    quotes = [q.strip(" .,;:!?") for q in re.findall(r'"([^"]{6,})"', text)]
    missing += [q for q in quotes if norm(q) not in f]
    missing += [n for n in re.findall(r"\b\d+\b", text) if n not in f]
    if re.search(r"\bstarred\b", t) and "starred" not in f:
        missing.append("starred-claim")
    return sorted(set(missing))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://firstpr-api.vercel.app")
    p.add_argument("--catalog", default="data/serving/static/repos.json")
    p.add_argument("--recheck", action="store_true", help="re-run the check on saved outputs")
    a = p.parse_args()
    vocab = vocabulary(a.api)
    if a.recheck:
        df = pd.read_csv("results/analysis/explanation_faithfulness.csv")
        miss = [
            check(e, f.split(" | "), vocab)
            for e, f in zip(df["explanation"], df["facts"], strict=True)
        ]
        df["auto_faithful"] = [not m for m in miss]
        df["unsupported_terms"] = ["; ".join(m) for m in miss]
        df.to_csv("results/analysis/explanation_faithfulness.csv", index=False)
        labels = Path("notes/phase7-explanation-labels.csv")
        df.assign(my_faithful="", my_note="").to_csv(labels, index=False)
        print(df.groupby(["path", "source"])["auto_faithful"].agg(["count", "mean"]).round(3))
        print("overall auto-faithful:", round(df["auto_faithful"].mean(), 3), "of", len(df))
        for r in df.loc[~df["auto_faithful"]].itertuples():
            print(r.id, r.path, r.source, "|", r.unsupported_terms)
        return
    personas = load_yaml("configs/phase7_personas.yaml")["personas"]
    requests_ = []
    for pr in personas[:20]:
        body = {"languages": pr["languages"], "interests": pr["interests"], "hours": pr["hours"]}
        recs = requests.post(f"{a.api}/recommend/onboarding", json=body, timeout=60).json()["repos"]
        for r in recs[:2]:
            requests_.append(
                ("onboarding", r, [*pr["languages"], *r["signals"].get("matched_skills", [])])
            )
        time.sleep(3.5)
    owners = list(
        dict.fromkeys(x["name"].split("/")[0] for x in json.loads(Path(a.catalog).read_text()))
    )
    n_gh = 0
    for login in owners:
        if n_gh == 10:
            break
        t = requests.get(f"https://api.github.com/users/{login}", headers=GH, timeout=10)
        if not (t.ok and t.json().get("type") == "User"):
            continue
        out = requests.post(
            f"{a.api}/recommend/github", json={"username": login, "hours": 3}, timeout=60
        ).json()
        time.sleep(3.5)
        withco = [r for r in out.get("repos", []) if r["signals"].get("co_starred_with")]
        if withco:
            r = withco[0]
            requests_.append(("github", r, r["signals"].get("matched_skills", [])))
            n_gh += 1
    rows = []
    for i, (path, r, skills) in enumerate(requests_):
        body = {
            "repo_id": r["id"],
            "issue_number": r["issues"][0]["number"] if r["issues"] else None,
            "co_starred": r["signals"].get("co_starred_with", []),
            "skills": skills[:10],
        }
        e = requests.post(f"{a.api}/explain", json=body, timeout=60).json()
        time.sleep(6.5)  # explain limit: 10 per minute per IP
        missing = check(e["text"], e["facts"], vocab)
        rows.append(
            {
                "id": f"e{i:02d}",
                "path": path,
                "repo": r["name"],
                "source": e["source"],
                "facts": " | ".join(e["facts"]),
                "explanation": e["text"],
                "auto_faithful": not missing,
                "unsupported_terms": "; ".join(missing),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("results/analysis/explanation_faithfulness.csv", index=False)
    print(df.groupby(["path", "source"])["auto_faithful"].agg(["count", "mean"]).round(3))
    print("overall auto-faithful:", round(df["auto_faithful"].mean(), 3), "of", len(df))
    df.assign(my_faithful="", my_note="").to_csv("notes/phase7-explanation-labels.csv", index=False)


if __name__ == "__main__":
    main()
