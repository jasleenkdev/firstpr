"""The one interface every recommender implements."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from firstpr.data.dataset import InteractionData

# Called by models that early-stop: returns the validation primary metric (higher is better),
# computed by the shared Evaluator so models never compute metrics themselves.
ValFn = Callable[["Recommender"], float]


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        """Train on `data.train` only. Returns fit info for the run record (e.g. epochs)."""

    @abstractmethod
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Scores for every item: float array [len(user_ids), n_items], higher = better."""
