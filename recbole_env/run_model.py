"""Train one RecBole model on an exported FirstPR split and dump full score matrices.

Runs inside the isolated recbole_env (see pyproject.toml). Input: a job JSON written by
firstpr.recbole_bridge. Output: scores.npy [n_users, n_items] in FirstPR id space, where the
scoring context is the user's *train* data only (same information every FirstPR model gets),
plus meta.json with training info. RecBole's own validation metric is used only for early
stopping; every reported metric comes from the FirstPR evaluator.
"""

import json
import sys
import time

import numpy as np
import recbole.trainer.trainer as rb_trainer
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed


def guard_early_stopping(min_evals: int) -> None:
    """Do not let RecBole stop before `min_evals` validations (same rule as FirstPR's trainer
    `min_epochs`: LightGCN sits on a popularity plateau for the first epochs at small lr)."""
    original = rb_trainer.early_stopping
    calls = {"n": 0}

    def guarded(*args, **kwargs):
        calls["n"] += 1
        best, cur_step, stop, update = original(*args, **kwargs)
        return best, cur_step, stop and calls["n"] >= min_evals, update

    rb_trainer.early_stopping = guarded


def main(job_path: str) -> None:
    job = json.load(open(job_path))
    min_epochs = int(job["config"].pop("min_epochs", 0))
    if min_epochs:
        guard_early_stopping(min_epochs // int(job["config"].get("eval_step", 1)))
    cfg = {
        "data_path": job["data_path"],
        "benchmark_filename": ["train", "valid", "test"],
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "eval_args": {"order": "TO", "mode": "full", "group_by": "user"},
        "metrics": ["NDCG", "Recall"],
        "topk": [20],
        "valid_metric": "NDCG@20",
        "device": "cpu",
        "use_gpu": False,
        "show_progress": False,
        "checkpoint_dir": job["out_dir"],
        "seed": job["seed"],
        "reproducibility": True,
        "log_wandb": False,
        **job["config"],
    }
    config = Config(model=job["model"], dataset=job["dataset"], config_dict=cfg)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    if config["MODEL_TYPE"].name == "SEQUENTIAL":
        # RecBole 1.2.1 benchmark mode needs pre-augmented item_id_list files for sequential
        # models; FirstPR uses its own SASRec instead (see the decision log)
        raise SystemExit("sequential RecBole models are not supported by this bridge")
    model_cls = get_model(config["model"])
    model = model_cls(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    t0 = time.time()
    best_valid, _ = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    train_time = time.time() - t0
    state = torch.load(trainer.saved_model_file, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # RecBole internal ids (1..n, 0 = padding) -> FirstPR ids (the exported tokens)
    uid_tokens = dataset.field2id_token["user_id"]
    iid_tokens = dataset.field2id_token["item_id"]
    n_users, n_items = job["n_users"], job["n_items"]
    item_ours = np.array([int(t) for t in iid_tokens[1:]])
    user_ours = np.array([int(t) for t in uid_tokens[1:]])

    scores = np.zeros((n_users, n_items), dtype=np.float32)
    inter_cls = train_data._dataset.inter_feat.__class__
    with torch.no_grad():
        for start in range(1, len(uid_tokens), 1024):
            uids = np.arange(start, min(start + 1024, len(uid_tokens)))
            inter = inter_cls({"user_id": torch.from_numpy(uids)})
            s = model.full_sort_predict(inter).view(len(uids), -1).numpy()
            scores[np.ix_(user_ours[uids - 1], item_ours)] = s[:, 1:]

    np.save(f"{job['out_dir']}/scores.npy", scores)
    json.dump(
        {
            "recbole_best_valid_ndcg@20": float(best_valid),
            "train_loop_seconds": train_time,
            "recbole_model_type": config["MODEL_TYPE"].name,
        },
        open(f"{job['out_dir']}/meta.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main(sys.argv[1])
