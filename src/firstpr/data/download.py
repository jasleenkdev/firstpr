"""Download a dataset file and verify its checksum (md5 or sha256); stream large JSONL files."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from firstpr.utils.logging import get_logger

log = get_logger(__name__)

_CHUNK = 1 << 20


def file_digest(path: str | Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def md5sum(path: str | Path) -> str:
    return file_digest(path, "md5")


def download(
    url: str,
    raw_dir: str | Path,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download `url` into `raw_dir` unless a file with the right checksum is already there."""
    if (expected_md5 is None) == (expected_sha256 is None):
        raise ValueError("give exactly one of expected_md5 / expected_sha256")
    algorithm, expected = ("md5", expected_md5) if expected_md5 else ("sha256", expected_sha256)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / url.rsplit("/", 1)[-1]

    if dest.exists() and file_digest(dest, algorithm) == expected:
        log.info("found %s with valid checksum, skipping download", dest)
        return dest

    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                f.write(chunk)

    actual = file_digest(tmp, algorithm)
    if actual != expected:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {expected}, got {actual}")
    tmp.rename(dest)
    log.info("saved %s (%s ok)", dest, algorithm)
    return dest


def stream_jsonl(
    url: str, expected_sha256: str, keep: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    """Stream a (large) JSONL file, return the records `keep` accepts, without storing the file.

    The sha256 of the full byte stream is checked at the end; on a mismatch nothing is returned.
    """
    h = hashlib.sha256()
    kept, n, buf = [], 0, b""
    log.info("streaming %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=_CHUNK):
            h.update(chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip():
                    n += 1
                    rec = json.loads(line)
                    if keep(rec):
                        kept.append(rec)
    if buf.strip():
        n += 1
        rec = json.loads(buf)
        if keep(rec):
            kept.append(rec)
    if h.hexdigest() != expected_sha256:
        raise ValueError(f"checksum mismatch for {url}: expected {expected_sha256}")
    log.info("streamed %d records (sha256 ok), kept %d", n, len(kept))
    return kept
