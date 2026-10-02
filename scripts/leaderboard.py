"""Aggregate test runs: main datasets -> results/leaderboard.csv, data variants (e.g. other
tie-break seeds) -> results/robustness_tiebreak.csv. Prints markdown tables."""

import argparse
from pathlib import Path

from firstpr.eval.leaderboard import aggregate, to_markdown
from firstpr.models.registry import MODELS
from firstpr.train.runner import load_runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    results = Path(p.parse_args().results_dir)

    df = aggregate(load_runs(results, "test"), model_order=list(MODELS))
    if df.empty:
        raise SystemExit("no test runs found")
    is_variant = df["dataset"].str.contains("_tb")
    for name, part in (
        ("leaderboard.csv", df[~is_variant]),
        ("robustness_tiebreak.csv", df[is_variant]),
    ):
        if part.empty:
            continue
        part.to_csv(results / name, index=False, float_format="%.6g")
        print(f"\n## {name}\n")
        print(to_markdown(part))


if __name__ == "__main__":
    main()
