"""End-to-end PySpark Batch ETL Orchestrator (Extract -> Transform -> Aggregate -> Load).

Supports idempotent & incremental processing via watermark tracking, writing outputs to:
- Partitioned Parquet (local processed zone)
- S3 Data Lake (with local S3 fallback)
- PostgreSQL (with local SQLite fallback)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pyspark.sql import functions as F

from configs.settings import PlatformConfig, get_config, get_logger
from ingestion.batch_ingestion import run_batch_ingestion
from spark.aggregations import compute_business_aggregations
from spark.extract import extract_all_raw
from spark.spark_session import create_spark_session
from spark.transform import transform_all
from utils.storage import DatabaseClient, S3DataLakeClient

logger = get_logger(__name__)


def run_etl_pipeline(
    config: PlatformConfig | None = None,
    incremental: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Execute the complete PySpark batch ETL pipeline."""
    cfg = config or get_config()
    processed_dir = cfg.resolve_path(cfg.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    watermark_file = processed_dir / "_etl_watermark.json"

    prev_watermark: str | None = None
    if incremental and watermark_file.exists() and not force:
        prev_watermark = json.loads(watermark_file.read_text(encoding="utf-8")).get(
            "max_impression_timestamp"
        )
        logger.info("Incremental mode enabled; previous watermark=%s", prev_watermark)

    # Step 1: Ensure raw data is landed idempotently
    run_batch_ingestion(cfg, force=force)

    # Step 2: Initialize Spark and Extract
    spark = create_spark_session(cfg)
    try:
        raw_datasets = extract_all_raw(spark, cfg)

        if incremental and prev_watermark:
            raw_datasets["ad_impressions"] = raw_datasets["ad_impressions"].filter(
                F.col("timestamp") > F.lit(prev_watermark)
            )

        # Step 3: Transform
        transformed, dq_reports = transform_all(raw_datasets)

        # Step 4: Business Aggregations
        aggregations = compute_business_aggregations(transformed)

        # Step 5: Load to Partitioned Parquet & S3 Data Lake
        s3_client = S3DataLakeClient(cfg)
        parquet_paths: dict[str, str] = {}

        # Write transformed datasets
        for name, df in transformed.items():
            out_path = processed_dir / "parquet" / name
            writer = df.write.mode("overwrite")
            if name == "ad_impressions_enriched":
                writer = writer.partitionBy("year", "month")
            elif name == "ad_inventory":
                writer = writer.partitionBy("country")
            writer.parquet(str(out_path))
            s3_client.upload_path(out_path, f"processed/{name}")
            parquet_paths[name] = str(out_path)
            logger.info("Saved Parquet '%s' -> %s", name, out_path)

        # Write aggregation tables to Parquet, S3, and Relational DB
        db_client = DatabaseClient(cfg)
        agg_counts: dict[str, int] = {}
        for name, df in aggregations.items():
            out_path = processed_dir / "aggregations" / name
            df.write.mode("overwrite").parquet(str(out_path))
            s3_client.upload_path(out_path, f"aggregations/{name}")
            pdf = df.toPandas()
            agg_counts[name] = db_client.write_dataframe(pdf, f"agg_{name}", if_exists="replace")

        # Update watermark idempotently
        max_ts_row = transformed["ad_impressions_enriched"].agg(F.max("timestamp")).collect()[0][0]
        watermark_payload = {
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
            "max_impression_timestamp": str(max_ts_row) if max_ts_row else prev_watermark,
            "db_backend": db_client.backend,
            "dq_summary": [vars(r) for r in dq_reports],
            "aggregation_row_counts": agg_counts,
        }
        watermark_file.write_text(json.dumps(watermark_payload, indent=2), encoding="utf-8")
        logger.info("ETL pipeline completed successfully (db=%s).", db_client.backend)
        return watermark_payload
    finally:
        spark.stop()


if __name__ == "__main__":
    run_etl_pipeline()
