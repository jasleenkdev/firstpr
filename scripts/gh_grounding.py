"""Grounding of the Groq features.

Automatic, over all outputs: JSON parse rate, share of skills named in the source text, share of
issue evidence quotes copied from the issue, difficulty distribution
-> results/analysis/llm_grounding_github.csv.
Manual: a seeded sample (half repos, half issues) with source and output side by side
-> data/github/features/grounding_sample.md (local: holds README / issue text) for a by-hand
verdict, recorded in results/analysis/llm_grounding_manual_github.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.github import features as F
from firstpr.github.build import item_text
from firstpr.github.privacy import scrub_text
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--llm-config", default="configs/llm/github_features.yaml")
    a = p.parse_args()
    cfg, lc = load_yaml(a.config), load_yaml(a.llm_config)
    gdir = Path(cfg["github_dir"])
    fdir = gdir / "features"
    repos = pd.read_parquet(fdir / "repo_features.parquet")
    details = pd.read_parquet(gdir / "details.parquet")
    src = {
        int(r.repo_id): item_text(pd.Series(r._asdict()), lc["readme_max_chars"], 300)
        for r in details.itertuples()
    }
    lang = dict(zip(details["repo_id"], details["language"].fillna(""), strict=True))
    repos["source"] = repos["repo_id"].map(src)
    repo_skill = [
        F.skill_grounded(s, f"{t} {lang.get(r, '')}")
        for r, t, sk in zip(repos["repo_id"], repos["source"], repos["skills"], strict=True)
        for s in sk
    ]

    iss = pd.read_parquet(fdir / "issue_features.parquet")
    rated = pd.read_parquet(fdir / "issues_rated.parquet")
    full = pd.read_parquet(gdir / "issues.parquet")
    rated = rated.merge(full[["repo_id", "number", "body"]], on=["repo_id", "number"], how="left")
    iss = iss.merge(rated, on="issue_row", how="left")
    iss["source"] = [
        f"{scrub_text(t)} {', '.join(lb)} {scrub_text(b or '', lc['issue_body_max_chars'])}"
        for t, lb, b in zip(iss["title"], iss["labels"], iss["body"], strict=True)
    ]
    issue_skill = [
        F.skill_grounded(s, f"{t} {lang.get(r, '')}")
        for r, t, sk in zip(iss["repo_id"], iss["source"], iss["skills"], strict=True)
        for s in sk
    ]
    evid = [F.evidence_grounded(e, t) for e, t in zip(iss["evidence"], iss["source"], strict=True)]
    auto = pd.DataFrame(
        [
            {"kind": "repo", "metric": "outputs", "value": len(repos)},
            {"kind": "repo", "metric": "parse_rate", "value": repos["parsed"].mean()},
            {"kind": "repo", "metric": "skills_in_source", "value": np.mean(repo_skill)},
            {"kind": "issue", "metric": "outputs", "value": len(iss)},
            {
                "kind": "issue",
                "metric": "difficulty_valid",
                "value": iss["difficulty"].notna().mean(),
            },
            {"kind": "issue", "metric": "skills_in_source", "value": np.mean(issue_skill)},
            {"kind": "issue", "metric": "evidence_copied", "value": np.mean(evid)},
            *[
                {"kind": "issue", "metric": f"difficulty_{d}", "value": v}
                for d, v in iss["difficulty"].value_counts(normalize=True).items()
            ],
        ]
    )
    out = Path("results/analysis")
    out.mkdir(parents=True, exist_ok=True)
    auto.to_csv(out / "llm_grounding_github.csv", index=False, float_format="%.4f")
    print(auto.to_string(index=False))

    rng = np.random.default_rng(lc["seed"])
    half = lc["grounding_sample"] // 2
    rs = repos.iloc[rng.choice(len(repos), min(half, len(repos)), replace=False)]
    isample = iss.iloc[rng.choice(len(iss), min(half, len(iss)), replace=False)]
    lines = ["# Grounding sample (local only)\n"]
    for r in rs.itertuples():
        lines += [
            f"## repo {r.repo_id}\n",
            f"SOURCE:\n{r.source}\n",
            f"OUTPUT: summary={r.summary!r} skills={list(r.skills)} domain={r.domain!r}\n",
        ]
    for r in isample.itertuples():
        lines += [
            f"## issue row {r.issue_row} (repo {r.repo_id})\n",
            f"SOURCE:\n{r.source}\n",
            f"OUTPUT: difficulty={r.difficulty} skills={list(r.skills)} evidence={r.evidence!r}\n",
        ]
    (fdir / "grounding_sample.md").write_text("\n".join(lines))
    print(f"\nsample written: {len(rs)} repos, {len(isample)} issues")


if __name__ == "__main__":
    main()
