"""Model name (as used in configs/models/<name>.yaml) -> class."""

from firstpr.models.base import Recommender
from firstpr.models.itemknn import ItemKNN
from firstpr.models.popularity import Popularity

MODELS: dict[str, type[Recommender]] = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
}


def build_model(name: str) -> Recommender:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]()
