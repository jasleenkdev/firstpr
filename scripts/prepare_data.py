"""Download -> implicit conversion -> k-core -> id remap -> chronological split -> stats
(+ item text for datasets that have it)."""

import argparse
import json
from pathlib import Path

from firstpr.data.amazon import item_table, load_amazon_ratings
from firstpr.data.dataset import InteractionData
from firstpr.data.download import download, stream_jsonl
from firstpr.data.preprocess import k_core, load_ml1m_ratings, remap_ids, to_implicit
from firstpr.data.split import chronological_split
from firstpr.data.stats import compute_stats, tie_stats
from firstpr.utils.io import load_yaml, save_json, save_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/ml1m.yaml")
    parser.add_argument(
        "--tie-break-seed",
        type=int,
        default=None,
        help="override the config's tie-break seed; output goes to <processed_dir>_tb<seed>",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.tie_break_seed is not None and args.tie_break_seed != cfg["split"]["tie_break_seed"]:
        cfg["split"]["tie_break_seed"] = args.tie_break_seed
        cfg["processed_dir"] = f"{cfg['processed_dir']}_tb{args.tie_break_seed}"
        cfg["name"] = f"{cfg['name']}_tb{args.tie_break_seed}"

    dl, pp = cfg["download"], cfg["preprocess"]
    source = cfg.get("source", "ml1m")
    if source == "ml1m":
        archive = download(dl["url"], dl["raw_dir"], dl["md5"])
        ratings = load_ml1m_ratings(archive, dl["ratings_member"])
    elif source == "amazon2023":
        base = f"{dl['base_url']}/{dl['revision']}"
        csv = download(
            f"{base}/{dl['ratings_path']}", dl["raw_dir"], expected_sha256=dl["ratings_sha256"]
        )
        ratings = load_amazon_ratings(csv, pp.get("min_date"))
    else:
        raise ValueError(f"unknown source {source!r}")

    df = k_core(to_implicit(ratings, pp["rating_threshold"]), pp["k_core"])
    df, user_map, item_map = remap_ids(df)

    sp = cfg["split"]
    split = chronological_split(df, sp["val"], sp["test"], sp["tie_break_seed"])

    out = Path(cfg["processed_dir"])
    for name, frame in [("train", split.train), ("val", split.val), ("test", split.test)]:
        save_parquet(frame, out / f"{name}.parquet")
    save_parquet(user_map, out / "user_map.parquet")
    save_parquet(item_map, out / "item_map.parquet")
    if source == "amazon2023":  # metadata streamed (1.1 GB), only our items' text is kept
        wanted = set(item_map["raw_id"])
        meta = stream_jsonl(
            f"{base}/{dl['meta_path']}", dl["meta_sha256"], lambda r: r.get("parent_asin") in wanted
        )
        it = cfg["item_text"]
        items = item_table(meta, item_map, it["max_features"], it["max_description_chars"])
        save_parquet(items, out / "items.parquet")

    data = InteractionData.from_frames(
        split.train,
        split.val,
        split.test,
        n_users=len(user_map),
        n_items=len(item_map),
        head_fraction=cfg["head_fraction"],
        name=cfg["name"],
    )
    stats = compute_stats(data, split.n_short_users, tie_stats(split.train, split.val, split.test))
    stats["raw_ratings"] = len(ratings)
    stats["config"] = cfg
    save_json(stats, out / "stats.json")
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
