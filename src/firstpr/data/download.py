"""Download a dataset archive and verify its checksum."""

import hashlib
from pathlib import Path

import requests

from firstpr.utils.logging import get_logger

log = get_logger(__name__)

_CHUNK = 1 << 20


def md5sum(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, raw_dir: str | Path, expected_md5: str) -> Path:
    """Download `url` into `raw_dir` unless a file with the right checksum is already there."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / url.rsplit("/", 1)[-1]

    if dest.exists() and md5sum(dest) == expected_md5:
        log.info("found %s with valid checksum, skipping download", dest)
        return dest

    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                f.write(chunk)

    actual = md5sum(tmp)
    if actual != expected_md5:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {expected_md5}, got {actual}")
    tmp.rename(dest)
    log.info("saved %s (md5 ok)", dest)
    return dest
