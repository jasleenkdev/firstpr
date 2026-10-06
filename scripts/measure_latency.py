"""p50 / p95 latency per stage on the deployed API -> results/analysis/latency_phase6.csv.

Stages come from the API's own `timings_ms` (fetch_stars, retrieve_and_rank, explain) plus the
client-observed round trip. GitHub-path requests use public personal accounts that own catalog
repos, chosen at runtime; first request per account = cold star fetch, second = cached. Only
aggregate timings are written: no usernames.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True)
    p.add_argument("--n-users", type=int, default=20)
    p.add_argument("--n-onboarding", type=int, default=30)
    p.add_argument("--catalog", default="data/serving/static/repos.json")
    a = p.parse_args()
    api = a.api.rstrip("/")
    rng = random.Random(0)
    rows: list[dict] = []

    def post(path: str, body: dict) -> tuple[dict, float]:
        t0 = time.perf_counter()
        r = requests.post(f"{api}{path}", json=body, timeout=60)
        rt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        return r.json(), rt

    t0 = time.perf_counter()
    health = requests.get(f"{api}/health", timeout=90).json()
    rows.append({"stage": "health (first request)", "ms": (time.perf_counter() - t0) * 1000})
    rows.append({"stage": "server load_ms (process start)", "ms": health.get("load_ms", np.nan)})

    repos = json.loads(Path(a.catalog).read_text())
    owners = list(
        dict.fromkeys(r["name"].split("/")[0] for r in sorted(repos, key=lambda r: -r["stars"]))
    )
    users = []
    for login in owners:
        t = requests.get(f"https://api.github.com/users/{login}", timeout=10)
        if t.ok and t.json().get("type") == "User":
            users.append(login)
        if len(users) == a.n_users:
            break
    for u in users:
        for attempt in ("cold", "cached"):
            out, rt = post("/recommend/github", {"username": u, "hours": rng.choice([1, 3, 5, 8])})
            tm = out["timings_ms"]
            rows += [
                {"stage": f"github: fetch_stars ({attempt})", "ms": tm["fetch_stars"]},
                {"stage": "github: retrieve_and_rank", "ms": tm["retrieve_and_rank"]},
                {"stage": f"github: server total ({attempt})", "ms": tm["total"]},
                {"stage": f"github: client round trip ({attempt})", "ms": rt},
            ]
            if attempt == "cold" and out["repos"]:
                r0 = out["repos"][0]
                ex, ert = post(
                    "/explain",
                    {
                        "repo_id": r0["id"],
                        "issue_number": r0["issues"][0]["number"],
                        "co_starred": r0["signals"].get("co_starred_with", []),
                        "skills": r0["signals"].get("matched_skills", []),
                    },
                )
                rows += [
                    {"stage": f"explain ({ex['source']})", "ms": ex["timings_ms"]["explain"]},
                    {"stage": "explain: client round trip", "ms": ert},
                ]
        time.sleep(1.5)  # stay well under the API's per-client rate limit
    opts = requests.get(f"{api}/options", timeout=30).json()
    for _ in range(a.n_onboarding):
        body = {
            "languages": rng.sample(opts["languages"][:10], 2),
            "interests": rng.sample([t["id"] for t in opts["interests"]], 2),
            "hours": rng.choice([1, 3, 5, 8]),
        }
        out, rt = post("/recommend/onboarding", body)
        rows += [
            {
                "stage": "onboarding: retrieve_and_rank",
                "ms": out["timings_ms"]["retrieve_and_rank"],
            },
            {"stage": "onboarding: client round trip", "ms": rt},
        ]
        time.sleep(1.5)
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("stage")["ms"]
        .agg(n="count", p50=lambda x: np.percentile(x, 50), p95=lambda x: np.percentile(x, 95))
        .reset_index()
    )
    out = Path("results/analysis/latency_phase6.csv")
    agg.to_csv(out, index=False, float_format="%.1f")
    print(agg.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
