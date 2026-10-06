"""Replay report from data/replay/ (scripts/replay.py):
- results/analysis/replay_repo.csv: hit@20, NDCG@20, near-misses per model and cohort, for all
  queries and for "new repo" queries (target not touched before the PR);
- results/analysis/replay_slices.csv: the same by prior history (0, 1-5, 6-20, 20+ events);
- results/analysis/replay_cis.csv: paired bootstrap CIs (NDCG@20) for model pairs;
- results/analysis/replay_ladder.csv: serving fallback ladder chosen on Sep, reported on Oct.
Per-user metrics are averaged over seeds before aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.eval.bootstrap import paired_bootstrap
from firstpr.replay.evaluate import history_bin
from firstpr.replay.models import MODELS
from firstpr.replay.run import LADDER_BETA, LADDER_K, MONTHS, SEEDS, SERVING_NOW

OUT = Path("data/replay")
RES = Path("results/analysis")
NAMES = {
    "popularity": "Popularity",
    "lightgcn": "LightGCN (fold-in)",
    "sasrec": "SASRec",
    "sasrec_text_mlp_warm": "text-SASRec (serving architecture)",
}
PAIRS = [
    ("sasrec_text_mlp_warm", "popularity"),
    ("lightgcn", "popularity"),
    ("sasrec", "popularity"),
    ("sasrec_text_mlp_warm", "lightgcn"),
    ("sasrec_text_mlp_warm", "sasrec"),
    ("lightgcn", "sasrec"),
]


def per_user(model: str, month: str) -> dict[str, np.ndarray]:
    seeds = [0] if model == "popularity" else SEEDS
    runs = [np.load(OUT / f"{model}_s{s}_{month}.npz") for s in seeds]
    return {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0].files}


def ladder_user(month: str, k: int, b: float) -> np.ndarray:
    return np.mean(
        [np.load(OUT / f"ladder_s{s}_{month}.npz")[f"k{k}_b{b}_ndcg"] for s in SEEDS], axis=0
    )


def main() -> None:
    rows, slices, cis = [], [], []
    for split, month in MONTHS.items():
        q = pd.read_parquet(OUT / f"queries_{month}.parquet")
        new = ~q["target_in_profile"].to_numpy()
        bins = np.array([history_bin(n) for n in q["n_history"]])
        res = {m: per_user(m, month) for m in MODELS}
        for m, r in res.items():
            for subset, mask in [("all", np.ones(len(q), bool)), ("new repo", new)]:
                rows.append(
                    {
                        "cohort": month,
                        "role": split,
                        "model": NAMES[m],
                        "queries": subset,
                        "n": int(mask.sum()),
                        "hit@20": r["hit"][mask].mean(),
                        "ndcg@20": r["ndcg"][mask].mean(),
                        "near_miss_language": r["near_lang"][mask].mean(),
                        "near_miss_org": r["near_org"][mask].mean(),
                    }
                )
            for b in ["0", "1-5", "6-20", "20+"]:
                mask = bins == b
                slices.append(
                    {
                        "cohort": month,
                        "model": NAMES[m],
                        "history": b,
                        "n": int(mask.sum()),
                        "hit@20": r["hit"][mask].mean(),
                        "ndcg@20": r["ndcg"][mask].mean(),
                        "hit@20 new repo": r["hit"][mask & new].mean()
                        if (mask & new).any()
                        else np.nan,
                    }
                )
        for a, b in PAIRS:
            for subset, mask in [("all", np.ones(len(q), bool)), ("new repo", new)]:
                ci = paired_bootstrap(res[a]["ndcg"][mask], res[b]["ndcg"][mask], 1000, 0.95, 0)
                cis.append({"cohort": month, "queries": subset, "a": NAMES[a], "b": NAMES[b], **ci})
    pd.DataFrame(rows).to_csv(RES / "replay_repo.csv", index=False, float_format="%.4f")
    pd.DataFrame(slices).to_csv(RES / "replay_slices.csv", index=False, float_format="%.4f")
    pd.DataFrame(cis).to_csv(RES / "replay_cis.csv", index=False, float_format="%.4f")

    tune, rep = MONTHS["tune"], MONTHS["report"]
    grid = [(k, b) for k in LADDER_K for b in LADDER_BETA]
    sep = {c: ladder_user(tune, *c).mean() for c in grid}
    best = max(grid, key=lambda c: (sep[c], -c[0]))
    lad = []
    for name, c in [
        ("phase-6 serving (k=3, beta=0.15)", SERVING_NOW),
        (f"tuned on Sep (k={best[0]}, beta={best[1]})", best),
    ]:
        lad.append(
            {"ladder": name, "sep_ndcg@20": sep[c], "oct_ndcg@20": ladder_user(rep, *c).mean()}
        )
    raw = per_user("sasrec_text_mlp_warm", rep)["ndcg"]
    lad.append(
        {
            "ladder": "text-SASRec alone (no ladder)",
            "sep_ndcg@20": per_user("sasrec_text_mlp_warm", tune)["ndcg"].mean(),
            "oct_ndcg@20": raw.mean(),
        }
    )
    ci = paired_bootstrap(ladder_user(rep, *best), ladder_user(rep, *SERVING_NOW), 1000, 0.95, 0)
    lad.append(
        {
            "ladder": "Oct: tuned - phase-6 (95% CI)",
            "oct_ndcg@20": ci["diff"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
        }
    )
    pd.DataFrame(lad).to_csv(RES / "replay_ladder.csv", index=False, float_format="%.4f")
    (OUT / "ladder_tuned.txt").write_text(f"{best[0]} {best[1]}\n")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(
        pd.DataFrame(cis)
        .query("cohort == @rep")[["queries", "a", "b", "diff", "ci_low", "ci_high"]]
        .round(4)
        .to_string(index=False)
    )
    print(pd.DataFrame(lad).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
