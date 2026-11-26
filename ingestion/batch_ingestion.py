"""Batch data ingestion from local synthetic/raw sources into Raw Parquet & S3 data lake."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger
from utils.storage import S3DataLakeClient

logger = get_logger(__name__)

EXPECTED_FILES = [
    "travelers.csv",
    "campaigns.csv",
    "ad_impressions.csv",
    "ad_inventory.csv",
    "bidding_data.csv",
    "targeting_signals.csv",
    "experiment_events.csv",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_batch_ingestion(
    config: PlatformConfig | None = None,
    force: bool = False,
) -> dict[str, dict[str, str | int]]:
    """Ingest synthetic CSV files into raw landing zone and mirror to S3/local data lake idempotently."""
    cfg = config or get_config()
    src_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)
    raw_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "_ingestion_manifest.json"

    prev_manifest: dict[str, dict[str, str | int]] = {}
    if manifest_path.exists():
        prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    s3_client = S3DataLakeClient(cfg)
    new_manifest: dict[str, dict[str, str | int]] = {}

    for filename in EXPECTED_FILES:
        src_path = src_dir / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Missing expected source file: {src_path}")

        checksum = _file_sha256(src_path)
        prev_entry = prev_manifest.get(filename)
        raw_target = raw_dir / filename

        if (
            not force
            and prev_entry
            and prev_entry.get("sha256") == checksum
            and raw_target.exists()
        ):
            logger.info("Skipping %s (unchanged checksum %s)", filename, checksum[:12])
            new_manifest[filename] = prev_entry
            continue

        # Validate readability and count rows before landing
        row_count = sum(1 for _ in open(src_path, "rb")) - 1
        df_head = pd.read_csv(src_path, nrows=5)
        raw_target.write_bytes(src_path.read_bytes())
        s3_uri = s3_client.upload_path(raw_target, f"raw/{filename.replace('.csv', '')}")

        new_manifest[filename] = {
            "sha256": checksum,
            "rows": row_count,
            "columns": len(df_head.columns),
            "s3_uri": s3_uri,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Ingested %s (%d rows) -> %s", filename, row_count, s3_uri)

    manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    return new_manifest


if __name__ == "__main__":
    run_batch_ingestion()
