"""Judge vs hand labels on the persona sample (run after filling notes/phase7-persona-labels.csv):
exact agreement, Cohen's kappa (unweighted and quadratic-weighted), and the share of good / at
least partly relevant recommendations by each rater -> results/analysis/persona_agreement.csv."""

import numpy as np
import pandas as pd


def kappa(a: np.ndarray, b: np.ndarray, k: int = 3, weighted: bool = False) -> float:
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) ** 2) / (k - 1) ** 2 if weighted else (i != j).astype(float)
    return float(1 - (w * obs).sum() / (w * exp).sum())


def main() -> None:
    mine = pd.read_csv("notes/phase7-persona-labels.csv")
    judge = pd.read_csv("results/analysis/persona_judge.csv")[["row_id", "judge_score"]]
    df = mine.merge(judge, on="row_id")
    df = df.loc[df["my_label"].notna() & df["judge_score"].notna()]
    a, b = df["my_label"].astype(int).to_numpy(), df["judge_score"].astype(int).to_numpy()
    out = pd.DataFrame(
        [
            {
                "n": len(df),
                "exact_agreement": float((a == b).mean()),
                "kappa": kappa(a, b),
                "kappa_quadratic": kappa(a, b, weighted=True),
                "mine_good": float((a == 2).mean()),
                "judge_good": float((b == 2).mean()),
                "mine_at_least_partly": float((a >= 1).mean()),
                "judge_at_least_partly": float((b >= 1).mean()),
            }
        ]
    )
    out.to_csv("results/analysis/persona_agreement.csv", index=False, float_format="%.3f")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
