"""Aggregate test runs: main datasets -> results/leaderboard.csv, data variants (e.g. other
tie-break seeds) -> results/robustness_tiebreak.csv. Prints markdown tables."""

import argparse
from pathlib import Path

from firstpr.eval import bootstrap
from firstpr.eval.leaderboard import aggregate, select_final_runs, to_markdown
from firstpr.models.registry import MODELS
from firstpr.train.runner import load_runs
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    results = Path(p.parse_args().results_dir)

    runs = load_runs(results, "test")
    df = aggregate(runs, model_order=list(MODELS))
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
    bootstrap_report(results, runs)


def bootstrap_report(results: Path, runs: list) -> None:
    """Paired bootstrap CIs for every model pair (main datasets) -> results/bootstrap_cis.csv."""
    cfg = load_yaml("configs/eval/default.yaml")["bootstrap"]
    final = {k: v for k, v in select_final_runs(runs).items() if "_tb" not in k[0]}
    cis = bootstrap.pairwise_cis(final, cfg["metric"], cfg["n_resamples"], cfg["ci"], cfg["seed"])
    if cis.empty:
        print("\n(no per-user files yet: bootstrap CIs skipped)")
        return
    cis.to_csv(results / "bootstrap_cis.csv", index=False, float_format="%.6g")
    print(f"\n## bootstrap_cis.csv (paired, {cfg['n_resamples']} resamples over users)\n")
    print(bootstrap.to_markdown(cis, pairs=[("mf_bpr", "itemknn"), ("itemknn", "popularity")]))


if __name__ == "__main__":
    main()
