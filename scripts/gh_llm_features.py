"""Groq features for beginner repos (summary, skills, domain) and their open beginner issues
(difficulty, skills, evidence). Repo and issue calls use different models, so each runs in its
own thread against its own daily quota; both sleep through limits and stop at --deadline.
Outputs: <github_dir>/features/{repo,issue}_features.parquet (rewritten every 25 new calls)."""

import argparse
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from firstpr.github import features as F
from firstpr.github.build import item_text
from firstpr.llm.client import LLMClient
from firstpr.utils.io import load_yaml
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def priority_repos(gdir: Path, max_repos: int) -> pd.DataFrame:
    repo_set = pd.read_parquet(gdir / "scope" / "repo_set.parquet")
    beg = pd.read_parquet(gdir / "scope" / "beginner.parquet")[["repo_id", "n_contrib"]]
    issues = pd.read_parquet(gdir / "issues.parquet")
    df = repo_set.loc[repo_set["group"] == "beginner"].merge(beg, on="repo_id")
    df = df.loc[df["repo_id"].isin(issues["repo_id"])]
    df = df.sort_values(["n_contrib", "repo_id"], ascending=[False, True]).head(max_repos)
    details = pd.read_parquet(gdir / "details.parquet").drop(columns="name")
    return df.merge(details, on="repo_id", how="left")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--llm-config", default="configs/llm/github_features.yaml")
    p.add_argument("--deadline", help="stop starting new calls after this local time (ISO)")
    a = p.parse_args()
    cfg, lc = load_yaml(a.config), load_yaml(a.llm_config)
    gdir = Path(cfg["github_dir"])
    out = gdir / "features"
    out.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(a.deadline) if a.deadline else None
    opts = lc["options"]

    repos = priority_repos(gdir, lc["max_repos"])
    repo_prompts = [
        (
            int(r.repo_id),
            F.repo_prompt(item_text(pd.Series(r._asdict()), lc["readme_max_chars"], 300)),
        )
        for r in repos.itertuples()
    ]

    issues = pd.read_parquet(gdir / "issues.parquet")
    issues = issues.loc[issues["repo_id"].isin(repos["repo_id"])]
    issues = issues.sort_values(["repo_id", "created_at"], ascending=[True, False])
    issues = issues.groupby("repo_id").head(lc["issues_per_repo"]).reset_index(drop=True)
    issues["issue_row"] = issues.index
    order = {rid: i for i, rid in enumerate(repos["repo_id"])}  # same priority as repos
    issues = issues.sort_values(by="repo_id", key=lambda s: s.map(order), kind="stable")
    lang = dict(zip(repos["repo_id"], repos["language"].fillna(""), strict=True))
    issue_prompts, batches = [], {}
    for rid, part in issues.groupby("repo_id", sort=False):
        rows = list(part.itertuples())
        for k in range(0, len(rows), lc["issues_per_call"]):
            chunk = rows[k : k + lc["issues_per_call"]]
            blocks = [
                F.issue_block(j, r.title, list(r.labels), r.body, lc["issue_body_max_chars"])
                for j, r in enumerate(chunk, 1)
            ]
            bid = len(batches)
            batches[bid] = [r.issue_row for r in chunk]
            name = repos.loc[repos["repo_id"] == rid, "name"].iloc[0].split("/")[-1]
            issue_prompts.append((bid, F.issue_prompt(name, lang.get(rid, ""), blocks)))
    issues.drop(columns=["body"]).to_parquet(out / "issues_rated.parquet", index=False)
    log.info(
        "%d repo prompts, %d issue prompts (%d issues)",
        len(repo_prompts),
        len(issue_prompts),
        len(issues),
    )

    def worker(model: str, prompts: list, write) -> None:
        client = LLMClient(model, backend=lc["backend"])
        done: dict = {}
        for i in range(0, len(prompts), 25):  # deadline checked between blocks of 25
            if deadline and datetime.now() > deadline:
                log.info("%s: deadline reached after %d prompts", model, i)
                break
            done.update(F.run_calls(client, prompts[i : i + 25], opts, deadline=deadline))
            write(done)
        write(done)

    def write_repos(done: dict) -> None:
        F.repo_features(done).to_parquet(out / "repo_features.parquet", index=False)

    def write_issues(done: dict) -> None:
        F.issue_features(done, batches).to_parquet(out / "issue_features.parquet", index=False)

    threads = [
        threading.Thread(target=worker, args=(lc["repo_model"], repo_prompts, write_repos)),
        threading.Thread(target=worker, args=(lc["issue_model"], issue_prompts, write_issues)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
