"""Integration tests for PySpark batch ETL, business aggregations, and Structured Streaming."""

from __future__ import annotations

from pathlib import Path

from configs.settings import PlatformConfig
from spark.run_etl import run_etl_pipeline
from spark.streaming import run_streaming_pipeline


def test_pyspark_batch_etl_and_aggregations(
    test_config: PlatformConfig,
    synthetic_dataset_paths: dict[str, Path],
) -> None:
    summary = run_etl_pipeline(config=test_config, force=True)
    assert "aggregation_row_counts" in summary
    counts = summary["aggregation_row_counts"]

    expected_aggs = [
        "daily_impressions",
        "monthly_impressions",
        "campaign_revenue",
        "audience_reach_by_country",
        "inventory_by_placement",
        "campaign_ctr",
        "campaign_cvr",
        "avg_bidding_price",
        "audience_engagement_by_segment",
    ]
    for agg_name in expected_aggs:
        assert counts.get(agg_name, 0) > 0, f"Empty aggregation: {agg_name}"

    processed_dir = Path(test_config.paths.processed_data_dir)
    assert (processed_dir / "parquet" / "ad_impressions_enriched").exists()
    assert (processed_dir / "_etl_watermark.json").exists()


def test_spark_structured_streaming_pipeline(
    test_config: PlatformConfig,
    synthetic_dataset_paths: dict[str, Path],
) -> None:
    res = run_streaming_pipeline(config=test_config)
    assert res["watermark_delay"] == test_config.spark.watermark_delay
    assert Path(res["logged_batches_path"]).exists()
