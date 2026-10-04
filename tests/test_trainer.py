import torch

from firstpr.train.trainer import train_with_early_stopping


def _run(vals: list[float], patience: int, min_epochs: int = 0) -> dict:
    module = torch.nn.Linear(1, 1)
    it = iter(vals)
    return train_with_early_stopping(
        module, lambda e: 1.0, lambda: next(it), len(vals), patience, min_epochs=min_epochs
    )


def test_early_stopping_on_plateau():
    info = _run([0.1, 0.1, 0.1, 0.1, 0.5, 0.6], patience=2)
    assert info["stopped"] == "early_stopping"
    assert info["epochs_run"] == 3 and info["best_epoch"] == 1


def test_min_epochs_lets_a_model_leave_the_plateau():
    info = _run([0.1, 0.1, 0.1, 0.1, 0.5, 0.6, 0.6, 0.6, 0.6], patience=2, min_epochs=5)
    assert info["best_epoch"] == 6 and info["best_val"] == 0.6
    assert info["stopped"] == "early_stopping" and info["epochs_run"] == 8
