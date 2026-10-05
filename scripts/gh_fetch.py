"""GitHub GraphQL fetches for the repo set (cached): repo details with the README as of the
train cutoff -> details.parquet; currently open beginner-labelled issues of beginner repos ->
issues.parquet."""

import argparse
from pathlib import Path

import pandas as pd

from firstpr.github.api import GraphQLClient
from firstpr.github.scope import beginner_pattern
from firstpr.utils.io import load_yaml
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument(
        "--what", nargs="+", choices=["details", "issues"], default=["details", "issues"]
    )
    a = p.parse_args()
    cfg = load_yaml(a.config)
    gdir = Path(cfg["github_dir"])
    repo_set = pd.read_parquet(gdir / "scope" / "repo_set.parquet")
    gql = GraphQLClient(gdir / "api_cache.sqlite")

    if "details" in a.what and not (gdir / "details.parquet").exists():
        det = gql.repo_details(repo_set["repo_id"].tolist(), cfg["windows"]["val_start"])
        det.to_parquet(gdir / "details.parquet", index=False)
        log.info("details: %d repos, %d with a README", len(det), int((det["readme"] != "").sum()))

    if "issues" in a.what and not (gdir / "issues.parquet").exists():
        beg = pd.read_parquet(gdir / "scope" / "beginner.parquet")
        beg = beg.loc[beg["repo_id"].isin(repo_set["repo_id"])]
        # labels as defined on the repo (exact names); repos without any are skipped
        pattern = beginner_pattern(cfg["beginner_labels"])
        labels = {
            int(r): [n for n in names if pattern.search(n)]
            for r, names in zip(beg["repo_id"], beg["labels"], strict=True)
        }
        labels = {r: v for r, v in labels.items() if v}
        issues = gql.open_issues(labels)
        issues.to_parquet(gdir / "issues.parquet", index=False)
        log.info("open beginner issues: %d in %d repos", len(issues), issues["repo_id"].nunique())


if __name__ == "__main__":
    main()
