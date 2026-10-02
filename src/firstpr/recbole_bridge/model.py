"""Run a RecBole model in the isolated `recbole_env` and expose its scores as a Recommender.

fit(): export the split (cached per dataset), write a job JSON, run
`uv run --project recbole_env python recbole_env/run_model.py job.json` in a subprocess, load
the score matrix it writes (FirstPR id space, train-only scoring context). score() then indexes
that matrix. RecBole early-stops on its own NDCG@20 over our val split; that is part of
training only -- all reported metrics come from the FirstPR evaluator.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.recbole_bridge.export import export_benchmark
from firstpr.utils.io import REPO_ROOT, config_hash
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

ENV_DIR = REPO_ROOT / "recbole_env"


class RecBoleModel(Recommender):
    def __init__(self, recbole_model: str, work_dir: str | Path | None = None) -> None:
        self.recbole_model = recbole_model
        self.name = f"recbole_{recbole_model.lower()}"
        self.work_dir = Path(work_dir) if work_dir else REPO_ROOT / "checkpoints" / "recbole"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        rb_config = {k: v for k, v in config.items() if k != "seed"}
        data_dir = self.work_dir / "data"
        export_benchmark(data, data_dir, data.name)
        out_dir = (
            self.work_dir / "runs" / f"{self.name}_{data.name}_{config_hash(rb_config)}_s{seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "model": self.recbole_model,
            "dataset": data.name,
            "data_path": str(data_dir),
            "out_dir": str(out_dir),
            "seed": seed,
            "config": rb_config,
            "n_users": data.n_users,
            "n_items": data.n_items,
        }
        job_path = out_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2))
        t0 = time.time()
        proc = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(ENV_DIR),
                "python",
                str(ENV_DIR / "run_model.py"),
                str(job_path),
            ],
            capture_output=True,
            text=True,
            cwd=out_dir,
            env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        )
        (out_dir / "recbole_stdout.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
            raise RuntimeError(f"RecBole {self.recbole_model} failed:\n{tail}")
        self.scores_ = np.load(out_dir / "scores.npy")
        meta = json.loads((out_dir / "meta.json").read_text())
        log.info("RecBole %s done in %.0fs: %s", self.recbole_model, time.time() - t0, meta)
        return {**meta, "subprocess_seconds": time.time() - t0, "work_dir": str(out_dir)}

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return self.scores_[user_ids]
