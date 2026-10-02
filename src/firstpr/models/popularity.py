"""Most-popular baseline: every user gets items ranked by train interaction count."""

from typing import Any

import numpy as np

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn


class Popularity(Recommender):
    name = "popularity"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        self.scores_ = data.item_popularity.astype(np.float32)
        return {}

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.scores_, (len(user_ids), len(self.scores_)))
