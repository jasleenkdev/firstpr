"""Model name (as used in configs/models/<name>.yaml) -> constructor."""

from collections.abc import Callable

from firstpr.models.base import Recommender
from firstpr.models.itemknn import ItemKNN
from firstpr.models.lightgcn import LightGCN
from firstpr.models.mf_bpr import MFBPR
from firstpr.models.ncf import NCF
from firstpr.models.popularity import Popularity
from firstpr.models.sasrec import SASRec
from firstpr.models.two_tower import TwoTower
from firstpr.recbole_bridge.model import RecBoleModel

MODELS: dict[str, Callable[[], Recommender]] = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "mf_bpr": MFBPR,
    "ncf_gmf": lambda: NCF("gmf"),
    "ncf_mlp": lambda: NCF("mlp"),
    "ncf_neumf": lambda: NCF("neumf"),
    "two_tower": TwoTower,
    "two_tower_nologq": TwoTower,  # ablation: same model, logq: false (tuned separately)
    "sasrec": SASRec,
    "sasrec_bce": SASRec,  # ablation: sasrec's tuned config with the paper's BCE loss
    "sasrec_shuffled": SASRec,  # ablation: sasrec's tuned config on shuffled histories
    "recbole_bpr": lambda: RecBoleModel("BPR"),  # via recbole_bridge (isolated env)
    "lightgcn": LightGCN,
    "lightgcn_l1": LightGCN,  # layer ablation: lightgcn's tuned config with n_layers overridden
    "lightgcn_l2": LightGCN,
    "lightgcn_l4": LightGCN,
}


def build_model(name: str) -> Recommender:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]()
