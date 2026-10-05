"""Repo set for the GitHub benchmark: BigQuery activity -> beginner check -> broad -> cold ->
repo_set.parquet (see firstpr.github.scope). Each step is skipped when its output exists."""

import argparse
from pathlib import Path

from firstpr.github import scope
from firstpr.github.api import GraphQLClient
from firstpr.github.bigquery import BigQueryRunner
from firstpr.utils.disk import check_disk
from firstpr.utils.io import load_yaml

STEPS = ["activity", "beginner", "broad", "cold", "finalize"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    a = p.parse_args()
    cfg = load_yaml(a.config)
    gdir = Path(cfg["github_dir"])
    check_disk(gdir, cfg["disk"]["max_data_gb"], cfg["disk"]["warn_free_gb"])
    bqc = cfg["bigquery"]
    bq = None
    gql = None
    for step in a.steps:
        if step in ("activity", "broad") and bq is None:
            bq = BigQueryRunner(
                gdir / "bq_ledger.jsonl", bqc["max_gb_per_query"], bqc["monthly_budget_gb"]
            )
        if step in ("beginner", "cold") and gql is None:
            gql = GraphQLClient(gdir / "api_cache.sqlite")
        if step == "activity":
            scope.step_activity(cfg, bq)
        elif step == "beginner":
            scope.step_beginner(cfg, gql)
        elif step == "broad":
            scope.step_broad(cfg, bq)
        elif step == "cold":
            scope.step_cold(cfg, gql)
        else:
            scope.finalize(cfg)


if __name__ == "__main__":
    main()
