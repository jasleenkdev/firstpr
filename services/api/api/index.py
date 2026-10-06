"""Serverless entry point: the FirstPR API (ASGI app from the bundled `serve` package)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FIRSTPR_ARTIFACTS", str(ROOT / "artifacts"))

from serve.app import create_app  # noqa: E402

app = create_app()
