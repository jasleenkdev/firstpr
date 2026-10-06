"""Daily refresh of the API's dynamic data (firstpr.serve.refresh): open beginner issues, fresh
repos, LLM feature backfill within the Groq free tier -> <out>/dynamic/ (+ upload to the HF
dataset). Runs locally (--source local) or in GitHub Actions (--source hub)."""

import argparse
import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from firstpr.github.api import GraphQLClient
from firstpr.github.build import item_text
from firstpr.serve import refresh as R
from firstpr.utils.env import require_env
from firstpr.utils.io import load_yaml
from firstpr.utils.logging import get_logger

log = get_logger(__name__)
CATALOG_FILES = ("catalog/repos.json", "catalog/catalog_text.json.gz")
FEATURE_FILES = ("repo_features.json", "issue_features.json")


def load_local(cfg: dict, out: Path) -> tuple[list, dict, dict, dict]:
    repos = json.loads((out / "static" / "repos.json").read_text())
    details = pd.read_parquet(Path(cfg["github_dir"]) / "details.parquet").set_index("repo_id")
    it = cfg["item_text"]
    texts = {
        r["id"]: item_text(
            details.loc[r["id"]], it["readme_max_chars"], it["description_max_chars"]
        )
        for r in repos
    }
    dyn = out / "dynamic"
    if (dyn / "repo_features.json").exists():
        rf = json.loads((dyn / "repo_features.json").read_text())
        isf = json.loads((dyn / "issue_features.json").read_text())
    else:
        rf, isf = R.features_from_phase5(Path(cfg["github_dir"]))
    return repos, texts, rf, isf


def load_hub(dataset: str, token: str, work: Path) -> tuple[list, dict, dict, dict]:
    from huggingface_hub import hf_hub_download

    def get(name: str) -> Path:
        return Path(
            hf_hub_download(dataset, name, repo_type="dataset", token=token, local_dir=work)
        )

    repos = json.loads(get(CATALOG_FILES[0]).read_text())
    with gzip.open(get(CATALOG_FILES[1]), "rt") as f:
        texts = {int(k): v for k, v in json.load(f).items()}
    rf = json.loads(get(FEATURE_FILES[0]).read_text())
    isf = json.loads(get(FEATURE_FILES[1]).read_text())
    return repos, texts, rf, isf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/github.yaml")
    p.add_argument("--llm-config", default="configs/llm/github_features.yaml")
    p.add_argument("--out", default="data/serving")
    p.add_argument("--source", choices=["local", "hub"], default="local")
    p.add_argument("--llm-minutes", type=float, default=0.0, help="Groq backfill time budget")
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    cfg, lc = load_yaml(a.config), load_yaml(a.llm_config)
    sc = load_yaml("configs/serving.yaml")["refresh"]
    out = Path(a.out)
    now = datetime.now(UTC)
    dataset = f"{require_env('HF_USERNAME')}/{sc['dataset_name']}"
    if a.source == "local":
        repos, texts, repo_feats, issue_feats = load_local(cfg, out)
    else:
        repos, texts, repo_feats, issue_feats = load_hub(
            dataset, require_env("HF_TOKEN"), out / "hub"
        )

    gql = GraphQLClient(None)
    labels = R.catalog_labels(gql, [r["id"] for r in repos], cfg["beginner_labels"])
    raw = gql.open_issues(labels, per_repo=sc["issues_per_repo"])
    rows = raw.to_dict("records")

    if a.llm_minutes > 0:
        by_contrib = sorted(
            [r for r in repos if r["id"] in labels], key=lambda r: (-r["n_contrib"], r["id"])
        )
        snippets = [R.issue_record(x["repo_id"], x, None) for x in rows]
        n_r, n_i = R.backfill_llm(
            by_contrib,
            snippets,
            repo_feats,
            issue_feats,
            texts,
            lc,
            deadline=datetime.now() + timedelta(minutes=a.llm_minutes),
        )
        log.info("llm backfill: %d repo features, %d issue features", n_r, n_i)

    issues = [
        R.issue_record(x["repo_id"], x, issue_feats.get(f"{x['repo_id']}#{x['number']}"))
        for x in rows
    ]
    fresh = R.fresh_repos(
        gql, now, sc["fresh_issue_days"], sc["fresh_max_age_days"], sc["fresh_min_stars"]
    )
    catalog_ids = {r["id"] for r in repos}
    fresh = [f for f in fresh if f["id"] not in catalog_ids][: sc["fresh_max_repos"]]
    R.write_dynamic(out / "dynamic", issues, repo_feats, issue_feats, fresh, now)

    if a.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=require_env("HF_TOKEN"))
        if a.source == "local":  # the catalog the scheduled job needs, written once per build
            cdir = out / "catalog"
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "repos.json").write_text(json.dumps(repos))
            with gzip.open(cdir / "catalog_text.json.gz", "wt") as f:
                json.dump({str(k): v for k, v in texts.items()}, f)
            api.upload_folder(
                folder_path=str(cdir),
                path_in_repo="catalog",
                repo_id=dataset,
                repo_type="dataset",
                commit_message="catalog",
            )
        api.upload_folder(
            folder_path=str(out / "dynamic"),
            repo_id=dataset,
            repo_type="dataset",
            commit_message=f"refresh {now:%Y-%m-%d}",
        )
        log.info("uploaded dynamic data to the dataset")


if __name__ == "__main__":
    main()
