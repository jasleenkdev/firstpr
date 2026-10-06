"""Issue-level replay (firstpr.replay.issues).

--fetch: seeded sample of beginner target repos per cohort -> GraphQL PR -> closing-issue links
         for cohort users + candidate issues open at PR time -> data/replay/issue_cases_<m>.json
--report: tune the issue weights on the Sep cohort only, then report Oct once against the
         phase-6 weights, newest-first and random -> results/analysis/replay_issues.csv
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.eval.bootstrap import paired_bootstrap
from firstpr.github.api import GraphQLClient
from firstpr.github.privacy import load_salt
from firstpr.github.scope import beginner_pattern
from firstpr.replay.issues import collect, score_issues
from firstpr.utils.io import load_yaml
from firstpr.utils.logging import get_logger

log = get_logger(__name__)
MONTHS = {"tune": "2025-09", "report": "2025-10"}
PHASE6 = {"diff": 0.35, "skill": 0.25, "recent": 0.2, "half_life": 30.0}
GRID = {
    "diff": [0.0, 0.2, 0.35, 0.5],
    "skill": [0.0, 0.25, 0.5],
    "recent": [0.0, 0.2, 0.4],
    "half_life": [7.0, 30.0, 90.0],
}


def fetch(cfg: dict, out: Path, n_repos: int, seed: int) -> None:
    gdir = Path(cfg["github_dir"])
    im = pd.read_parquet(Path(cfg["processed_dir"]) / "item_map.parquet")
    beg = pd.read_parquet(gdir / "scope" / "beginner.parquet")
    pattern = beginner_pattern(cfg["beginner_labels"])
    labels = {
        int(r): [n for n in names if pattern.search(n)]
        for r, names in zip(beg["repo_id"], beg["labels"], strict=True)
    }
    gql = GraphQLClient(None)  # in-memory cache: responses hold raw user ids
    salt = load_salt()
    for month in MONTHS.values():
        f = out / f"issue_cases_{month}.json"
        if f.exists():
            continue
        q = pd.read_parquet(out / f"queries_{month}.parquet").merge(
            im[["item", "raw_id", "group"]], left_on="target", right_on="item"
        )
        q = q.loc[(q["group"] == "beginner") & q["raw_id"].map(lambda r: bool(labels.get(r)))]
        repos = np.random.default_rng(seed).permutation(np.sort(q["raw_id"].unique()))[:n_repos]
        cases = q.loc[q["raw_id"].isin(repos), ["user", "raw_id", "pr_ts", "profile"]].rename(
            columns={"raw_id": "repo_id"}
        )
        since = pd.Timestamp(f"{month}-01", tz="UTC").timestamp()
        got = collect(gql, cases, labels, since, salt)
        prof = dict(zip(cases["user"], cases["profile"], strict=True))
        for g in got:
            g["profile"] = [int(x) for x in prof[g["user"]]]
        f.write_text(json.dumps({"n_repos": len(repos), "n_pairs": len(cases), "cases": got}))
        log.info(
            "%s: %d repos, %d cohort pairs, %d with a linked PR",
            month,
            len(repos),
            len(cases),
            len(got),
        )


def evaluate(
    cases: list[dict], w: dict, lang_of_item: np.ndarray, repo_lang: dict
) -> dict[str, np.ndarray]:
    rr, hit1, hit3, n_c = [], [], [], []
    for c in cases:
        cands = c["candidates"]
        pos = [i for i, x in enumerate(cands) if x["number"] in c["linked"]]
        skills = {str(x).lower() for x in lang_of_item[c["profile"]] if x}
        if w == "random":
            r = (len(cands) + 1) / 2  # expected rank of one target among n
            rr.append(np.mean([1 / k for k in range(1, len(cands) + 1)]))
            hit1.append(1 / len(cands))
            hit3.append(min(3, len(cands)) / len(cands))
            n_c.append(len(cands))
            continue
        if w == "newest":
            s = np.array([x["created"] for x in cands])
        else:
            s = score_issues(cands, c["pr_ts"], skills, repo_lang.get(c["repo_id"], ""), w)
        order = np.argsort(-s, kind="stable")
        r = 1 + min(int(np.where(order == p)[0][0]) for p in pos)
        rr.append(1 / r)
        hit1.append(float(r <= 1))
        hit3.append(float(r <= 3))
        n_c.append(len(cands))
    return {
        "mrr": np.array(rr),
        "hit@1": np.array(hit1),
        "hit@3": np.array(hit3),
        "n_cand": np.array(n_c),
    }


def report(cfg: dict, out: Path) -> None:
    im = pd.read_parquet(Path(cfg["processed_dir"]) / "item_map.parquet").sort_values("item")
    det = pd.read_parquet(Path(cfg["github_dir"]) / "details.parquet").set_index("repo_id")[
        "language"
    ]
    lang_of_item = np.array(
        [det.get(r) if isinstance(det.get(r), str) else "" for r in im["raw_id"]], dtype=object
    )
    repo_lang = {int(k): v for k, v in det.items() if isinstance(v, str)}
    data = {}
    for split, month in MONTHS.items():
        blob = json.loads((out / f"issue_cases_{month}.json").read_text())
        # evaluable: the linked issue was a beginner-labelled issue open at PR time, among >= 2
        cases = [
            c
            for c in blob["cases"]
            if len(c["candidates"]) >= 2
            and any(x["number"] in c["linked"] for x in c["candidates"])
        ]
        data[split] = (blob, cases)
        log.info(
            "%s: %d sampled repos, %d pairs, %d linked PRs, %d evaluable",
            month,
            blob["n_repos"],
            blob["n_pairs"],
            len(blob["cases"]),
            len(cases),
        )
    tune = data["tune"][1]
    grid = [dict(zip(GRID, v, strict=True)) for v in itertools.product(*GRID.values())]
    grid = [g for g in grid if g["diff"] + g["skill"] + g["recent"] > 0]
    scored = [
        (evaluate(tune, g, lang_of_item, repo_lang)["mrr"].mean(), i) for i, g in enumerate(grid)
    ]
    best = grid[max(scored)[1]]
    log.info(
        "tuned on Sep: %s (MRR %.4f; phase-6 weights %.4f)",
        best,
        max(scored)[0],
        evaluate(tune, PHASE6, lang_of_item, repo_lang)["mrr"].mean(),
    )
    rows, per = [], {}
    for split in ("tune", "report"):
        blob, cases = data[split]
        for name, w in [
            ("random", "random"),
            ("newest first", "newest"),
            ("phase-6 weights", PHASE6),
            ("tuned on Sep", best),
        ]:
            m = evaluate(cases, w, lang_of_item, repo_lang)
            per[(split, name)] = m
            rows.append(
                {
                    "cohort": MONTHS[split],
                    "ranker": name,
                    "n_cases": len(cases),
                    "mean_candidates": m["n_cand"].mean(),
                    **{k: m[k].mean() for k in ("mrr", "hit@1", "hit@3")},
                    "sampled_repos": blob["n_repos"],
                    "cohort_pairs": blob["n_pairs"],
                    "linked_prs": len(blob["cases"]),
                }
            )
    for split in ("tune", "report"):
        for a, b in [
            ("tuned on Sep", "phase-6 weights"),
            ("phase-6 weights", "newest first"),
            ("tuned on Sep", "newest first"),
        ]:
            ci = paired_bootstrap(per[(split, a)]["mrr"], per[(split, b)]["mrr"], 1000, 0.95, 0)
            rows.append({"cohort": MONTHS[split], "ranker": f"{a} - {b} (MRR, 95% CI)", **ci})
    df = pd.DataFrame(rows)
    df.to_csv("results/analysis/replay_issues.csv", index=False, float_format="%.4f")
    (out / "issue_weights_tuned.json").write_text(json.dumps(best))
    print(df.round(4).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--out", default="data/replay")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--n-repos", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = load_yaml(a.config)
    if a.fetch:
        fetch(cfg, Path(a.out), a.n_repos, a.seed)
    if a.report:
        report(cfg, Path(a.out))


if __name__ == "__main__":
    main()
