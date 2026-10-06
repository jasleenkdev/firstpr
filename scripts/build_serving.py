"""Train the serving model on all GitHub interactions and write the API's static artifacts
(firstpr.serve.build) to data/serving/static/."""

import argparse
from pathlib import Path

from firstpr.serve.build import build_static
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--out", default="data/serving")
    p.add_argument("--epochs", type=int, default=24, help="median best epoch of the tuned runs")
    a = p.parse_args()
    build_static(load_yaml(a.config), Path(a.out), a.epochs)


if __name__ == "__main__":
    main()
