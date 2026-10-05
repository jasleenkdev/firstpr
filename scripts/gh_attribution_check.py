"""Check the payload-free attribution on Oct 2025, where payload is affordable.

The benchmark keeps PR / issue events of actors who had not pushed to the repo by the end of the
window (non-maintainers) and treats them as contributions by that actor. Payload says what each
event was: PR `action` (`opened` = the actor authored it; after GitHub's Oct-2025 trim nothing
else identifies the author) and, for issues, whether the actor is the issue author.

precision = share of kept (actor, repo) pairs with at least one authored event in the month
(a lower bound: a PR opened in September and closed by its author in October counts as a miss).
-> results/analysis/attribution_check_github.csv
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from firstpr.github.ingest import ingest_payload_check
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2025-10-31")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    gdir = Path(cfg["github_dir"])
    repo_set = pd.read_parquet(gdir / "scope" / "repo_set.parquet")
    pay = ingest_payload_check(a.start, a.end, [int(r) for r in repo_set["repo_id"]], cfg)
    pay = pay.drop(columns=["labels"])
    con = duckdb.connect()
    con.register("pay", pay)
    events = str(gdir / "events" / "*.parquet")
    end = cfg["windows"]["test_end"]
    df = con.execute(
        f"""
        WITH push AS (
          SELECT DISTINCT "user", repo_id FROM read_parquet('{events}')
          WHERE kind = 'push' AND first_at < TIMESTAMP '{end}'
        ),
        ev AS (
          SELECT p.*, (m."user" IS NOT NULL) AS maintainer,
                 CASE WHEN type = 'PullRequestEvent' THEN action = 'opened'
                      ELSE action = 'opened' AND actor_is_issue_author END AS authored
          FROM pay p LEFT JOIN push m USING ("user", repo_id)
        ),
        pairs AS (
          SELECT type, "user", repo_id, ANY_VALUE(maintainer) AS maintainer,
                 BOOL_OR(authored) AS authored, COUNT(*) AS n_events
          FROM ev GROUP BY ALL
        )
        SELECT type, maintainer, COUNT(*) AS pairs, AVG(authored::INT) AS share_authored,
               SUM(n_events) AS events
        FROM pairs GROUP BY ALL ORDER BY type, maintainer
        """
    ).df()
    actions = (
        pay.loc[pay["type"] == "PullRequestEvent", "action"].value_counts(normalize=True).round(4)
    )
    out = Path("results/analysis/attribution_check_github.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(df.to_string(index=False))
    print("\nPR actions:", actions.to_dict())


if __name__ == "__main__":
    main()
