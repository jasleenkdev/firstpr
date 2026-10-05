"""Disk guards for large downloads: free-space warning and a cap on the size of a data dir."""

import shutil
from pathlib import Path

from firstpr.utils.io import REPO_ROOT
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

GB = 1e9


def dir_size(path: str | Path) -> int:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def check_disk(data_dir: str | Path, max_data_gb: float, warn_free_gb: float) -> None:
    """Warn when free space is below `warn_free_gb`; raise when `data_dir` exceeds its cap."""
    free = shutil.disk_usage(REPO_ROOT).free
    if free < warn_free_gb * GB:
        log.warning("free disk space %.1f GB is below %.0f GB", free / GB, warn_free_gb)
    used = dir_size(data_dir)
    if used > max_data_gb * GB:
        raise RuntimeError(f"{data_dir} holds {used / GB:.1f} GB > cap {max_data_gb} GB")
