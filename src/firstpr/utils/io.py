"""Small IO helpers: yaml / json / parquet and git metadata for run records."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=True)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict (key order independent)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def git_info() -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = _run("rev-parse", "HEAD") or "no-commit"
    dirty = bool(_run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}
