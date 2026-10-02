"""Generic epoch loop with early stopping on a validation metric (higher is better)."""

import copy
import math
import time
from collections.abc import Callable
from typing import Any

import torch

from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def train_with_early_stopping(
    module: torch.nn.Module,
    run_epoch: Callable[[int], float],
    val_fn: Callable[[], float] | None,
    max_epochs: int,
    patience: int,
    eval_every: int = 1,
) -> dict[str, Any]:
    """Run `run_epoch(epoch) -> mean loss` until the val metric stops improving.

    Keeps the best weights in memory and restores them at the end. Stops after `patience`
    evaluations without improvement, or on a non-finite loss (reported, not hidden).
    """
    best_val, best_epoch, best_state = -math.inf, 0, None
    bad_evals, history = 0, []
    stopped = "max_epochs"
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        loss = run_epoch(epoch)
        if not math.isfinite(loss):
            stopped = "non_finite_loss"
            log.warning("epoch %d: loss is %s, stopping", epoch, loss)
            break
        record: dict[str, float] = {"epoch": epoch, "loss": loss}
        if val_fn is not None and epoch % eval_every == 0:
            val = val_fn()
            record["val"] = val
            if val > best_val:
                best_val, best_epoch, bad_evals = val, epoch, 0
                best_state = copy.deepcopy(module.state_dict())
            else:
                bad_evals += 1
            log.info(
                "epoch %d loss %.4f val %.5f (best %.5f @ %d)",
                epoch,
                loss,
                val,
                best_val,
                best_epoch,
            )
            if bad_evals >= patience:
                stopped = "early_stopping"
                break
        history.append(record)
    if best_state is not None:
        module.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val": best_val if best_state is not None else None,
        "stopped": stopped,
        "train_loop_seconds": time.time() - t0,
        "history": history,
    }
