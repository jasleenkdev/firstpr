"""First-contribution replay (phase 7): cohorts, leakage-safe profiles, models trained on the
phase-5 train split, per-user metrics -> data/replay/ (then scripts/replay_report.py)."""

import argparse
from pathlib import Path

from firstpr.replay.run import run
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--out", default="data/replay")
    a = p.parse_args()
    run(load_yaml(a.config), Path(a.out))


if __name__ == "__main__":
    main()
