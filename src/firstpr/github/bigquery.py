"""BigQuery with a byte budget (sandbox: 1 TB of query scans per month, no billing).

Every query is dry-run first; it runs only if its scan fits the per-query cap and the month's
remaining budget, with `maximum_bytes_billed` as a server-side guard. Bytes are appended to a
ledger (`data/github/bq_ledger.jsonl`) that the monthly budget is computed from. Results are
downloaded straight into Arrow (no tables are created: sandbox tables expire after 60 days).
The project id comes from `.env` and is never logged.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery

from firstpr.utils.env import require_env
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

GB = 1e9


def _param(name: str, value: Any) -> bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter:
    if isinstance(value, list | tuple):
        kind = "INT64" if value and isinstance(value[0], int) else "STRING"
        return bigquery.ArrayQueryParameter(name, kind, list(value))
    kind = "INT64" if isinstance(value, int) else "STRING"
    return bigquery.ScalarQueryParameter(name, kind, value)


class BudgetExceededError(RuntimeError):
    pass


class BigQueryRunner:
    def __init__(
        self,
        ledger_path: str | Path,
        max_gb_per_query: float,
        monthly_budget_gb: float,
    ) -> None:
        self.client = bigquery.Client(project=require_env("GCP_PROJECT_ID"))
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb_per_query * GB)
        self.monthly_budget = int(monthly_budget_gb * GB)

    def month_used(self) -> int:
        """Bytes billed by queries in the current UTC month, from the ledger."""
        if not self.ledger_path.exists():
            return 0
        month = datetime.now(UTC).strftime("%Y-%m")
        used = 0
        for line in self.ledger_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["at"].startswith(month):
                used += int(rec.get("bytes_billed") or rec["bytes_dry_run"])
        return used

    def _config(self, params: dict[str, Any] | None, dry: bool) -> bigquery.QueryJobConfig:
        cfg = bigquery.QueryJobConfig(
            query_parameters=[_param(k, v) for k, v in (params or {}).items()],
            dry_run=dry,
            use_query_cache=not dry,
        )
        if not dry:
            cfg.maximum_bytes_billed = self.max_bytes
        return cfg

    def _submit(self, sql: str, config: bigquery.QueryJobConfig) -> bigquery.QueryJob:
        """Submit a job; API errors are re-raised with the project id redacted."""
        try:
            return self.client.query(sql, job_config=config)
        except GoogleAPIError as e:
            raise RuntimeError(str(e).replace(self.client.project, "<project>")) from None

    def dry_run(self, sql: str, params: dict[str, Any] | None = None) -> int:
        job = self._submit(sql, self._config(params, dry=True))
        return int(job.total_bytes_processed or 0)

    def query(self, sql: str, label: str, params: dict[str, Any] | None = None) -> pa.Table:
        nbytes = self.dry_run(sql, params)
        used = self.month_used()
        log.info(
            "bq %s: dry run %.2f GB (month so far %.1f / %.0f GB)",
            label,
            nbytes / GB,
            used / GB,
            self.monthly_budget / GB,
        )
        if nbytes > self.max_bytes:
            raise BudgetExceededError(f"{label}: {nbytes / GB:.1f} GB > per-query cap")
        if used + nbytes > self.monthly_budget:
            raise BudgetExceededError(f"{label}: would exceed the monthly budget")
        t0 = time.time()
        job = self._submit(sql, self._config(params, dry=False))
        try:
            table = job.result().to_arrow(create_bqstorage_client=True)
        except GoogleAPIError as e:
            raise RuntimeError(str(e).replace(self.client.project, "<project>")) from None
        rec = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "bytes_dry_run": nbytes,
            "bytes_billed": int(job.total_bytes_billed or 0),
            "cache_hit": bool(job.cache_hit),
            "rows": table.num_rows,
            "seconds": round(time.time() - t0, 1),
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(
            "bq %s: billed %.2f GB, %d rows, %.0fs",
            label,
            rec["bytes_billed"] / GB,
            rec["rows"],
            rec["seconds"],
        )
        return table
