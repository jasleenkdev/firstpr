"""Experiment runner: config -> fit -> evaluate -> one JSON record per (config, seed).

- mode "val": expand the model's `search` grid, run every config once (tune seed) on validation,
  save the best config (by the primary metric) to results/best/<model>_<dataset>.yaml.
- mode "test": load that best config (or the fixed `params` if there is nothing to tune), run every
  final seed on test. Test is touched once per final config: re-running needs force=True.
"""

import itertools
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.models.registry import build_model
from firstpr.utils.io import config_hash, git_info, load_json, load_yaml, save_json, save_yaml
from firstpr.utils.logging import get_logger
from firstpr.utils.seed import set_seed

log = get_logger(__name__)


def expand_grid(params: dict[str, Any], search: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of the `search` values, each merged over the default `params`."""
    if not search:
        return [dict(params)]
    keys = sorted(search)
    return [
        {**params, **dict(zip(keys, values, strict=True))}
        for values in itertools.product(*(search[k] for k in keys))
    ]


def run_single(
    model_name: str,
    data: InteractionData,
    evaluator: Evaluator,
    params: dict[str, Any],
    seed: int,
    mode: str,
    primary_metric: str,
    dataset: str,
    results_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model = build_model(model_name)
    chash = config_hash(params)

    def val_fn(m: Any) -> float:
        return evaluator.evaluate(m, "val")["overall"][primary_metric]

    t0 = time.time()
    fit_info = model.fit(data, {**params, "seed": seed}, val_fn=val_fn)
    train_time = time.time() - t0
    t0 = time.time()
    result = evaluator.evaluate(model, mode)  # type: ignore[arg-type]
    inference_time = time.time() - t0

    record = {
        "model": model_name,
        "dataset": dataset,
        "mode": mode,
        "seed": seed,
        "config": params,
        "config_hash": chash,
        "git": git_info(),
        "data_stats_hash": config_hash(
            {k: v for k, v in data.stats.items() if k != "config"} or {"n": data.n_users}
        ),
        "train_time_s": train_time,
        "inference_time_s": inference_time,
        "fit_info": fit_info,
        "metrics": result,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = results_dir / "runs" / dataset / model_name / f"{mode}_{chash}_s{seed}_{stamp}.json"
    save_json(record, path)
    log.info(
        "%s %s seed=%d %s=%.5f train=%.1fs -> %s",
        model_name,
        mode,
        seed,
        primary_metric,
        result["overall"][primary_metric],
        train_time,
        path,
    )
    return record


def existing_test_runs(results_dir: Path, dataset: str, model_name: str, chash: str) -> list[Path]:
    return sorted((results_dir / "runs" / dataset / model_name).glob(f"test_{chash}_s*.json"))


def existing_val_run(
    results_dir: Path, dataset: str, model_name: str, chash: str, seed: int
) -> dict[str, Any] | None:
    """Latest finished validation run for this exact config and seed, if any."""
    paths = sorted(
        (results_dir / "runs" / dataset / model_name).glob(f"val_{chash}_s{seed}_*.json")
    )
    return load_json(paths[-1]) if paths else None


def best_config_path(results_dir: Path, model_name: str, dataset: str) -> Path:
    return results_dir / "best" / f"{model_name}_{dataset}.yaml"


def tune(
    model_name: str,
    data: InteractionData,
    model_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    dataset: str,
    results_dir: Path,
    reuse: bool = True,
) -> dict[str, Any]:
    evaluator = Evaluator(data, k=eval_cfg["k"], batch_size=eval_cfg["batch_size"])
    primary = eval_cfg["primary_metric"]
    grid = expand_grid(model_cfg.get("params", {}), model_cfg.get("search", {}))
    seed = eval_cfg["tune_seed"]
    log.info("tuning %s on %s: %d configs", model_name, dataset, len(grid))
    records = []
    for p in grid:
        cached = None
        if reuse:
            cached = existing_val_run(results_dir, dataset, model_name, config_hash(p), seed)
        if cached is not None:  # extending a grid does not re-run configs already finished
            log.info("reusing val run for %s config %s", model_name, cached["config_hash"])
            records.append(cached)
        else:
            records.append(
                run_single(
                    model_name, data, evaluator, p, seed, "val", primary, dataset, results_dir
                )
            )
    best = max(records, key=lambda r: r["metrics"]["overall"][primary])
    out = {
        "params": best["config"],
        "selected_by": f"val {primary}",
        "val_score": best["metrics"]["overall"][primary],
        "n_configs": len(grid),
        "grid": [
            {"config": r["config"], primary: r["metrics"]["overall"][primary]} for r in records
        ],
    }
    save_yaml(out, best_config_path(results_dir, model_name, dataset))
    return out


def final(
    model_name: str,
    data: InteractionData,
    model_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    dataset: str,
    results_dir: Path,
    force: bool = False,
    tuned_from: str | None = None,
) -> list[dict[str, Any]]:
    """3-seed test runs with the tuned config. `tuned_from` lets a data variant (e.g. another
    tie-break seed) reuse the config tuned on the main dataset, without re-tuning.

    Ablations: a model config with `tuned_from_model: <name>` takes that model's tuned params and
    applies its `overrides` (e.g. sasrec_bce = sasrec's tuned config with loss: bce)."""
    source = model_cfg.get("tuned_from_model", model_name)
    best_path = best_config_path(results_dir, source, tuned_from or dataset)
    if best_path.exists():
        params = load_yaml(best_path)["params"]
    elif model_cfg.get("search") or source != model_name:
        raise FileNotFoundError(f"{best_path} missing: run tuning (mode val) first")
    else:
        params = dict(model_cfg.get("params", {}))
    params = {**params, **model_cfg.get("overrides", {})}

    chash = config_hash(params)
    existing = existing_test_runs(results_dir, dataset, model_name, chash)
    if existing and not force:
        raise RuntimeError(
            f"test runs already exist for {model_name} config {chash} on {dataset} "
            f"({len(existing)} files); test is touched once per final config. Use force=True."
        )
    evaluator = Evaluator(data, k=eval_cfg["k"], batch_size=eval_cfg["batch_size"])
    return [
        run_single(
            model_name,
            data,
            evaluator,
            params,
            s,
            "test",
            eval_cfg["primary_metric"],
            dataset,
            results_dir,
        )
        for s in eval_cfg["seeds"]
    ]


def load_runs(results_dir: Path, mode: str = "test") -> list[dict[str, Any]]:
    return [load_json(p) for p in sorted((results_dir / "runs").rglob(f"{mode}_*.json"))]
