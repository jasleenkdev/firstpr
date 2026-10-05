"""Daily batch ingestion of GH Archive events for the repo set (idempotent per day).

Backfill: --start 2025-03-01 --end 2025-11-01. Daily use: --start <yesterday> --end <today>.
"""

import argparse
from pathlib import Path

import pandas as pd

from firstpr.github.ingest import ingest_range
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--start", help="first day (YYYY-MM-DD); default: train_start")
    p.add_argument("--end", help="day after the last (YYYY-MM-DD); default: test_end")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cfg = load_yaml(a.config)
    w = cfg["windows"]
    repo_set = pd.read_parquet(Path(cfg["github_dir"]) / "scope" / "repo_set.parquet")
    ingest_range(
        a.start or w["train_start"],
        a.end or w["test_end"],
        [int(r) for r in repo_set["repo_id"]],
        cfg,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
