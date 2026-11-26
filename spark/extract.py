"""Explicit PySpark schemas and extraction helpers for CSV, Parquet, and S3."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)

TRAVELERS_SCHEMA = StructType(
    [
        StructField("traveler_id", StringType(), False),
        StructField("age_group", StringType(), True),
        StructField("country", StringType(), True),
        StructField("city", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("travel_interest", StringType(), True),
        StructField("booking_frequency", IntegerType(), True),
        StructField("loyalty_segment", StringType(), True),
    ]
)

CAMPAIGNS_SCHEMA = StructType(
    [
        StructField("campaign_id", StringType(), False),
        StructField("advertiser_id", StringType(), True),
        StructField("campaign_name", StringType(), True),
        StructField("start_date", StringType(), True),
        StructField("end_date", StringType(), True),
        StructField("budget", DoubleType(), True),
        StructField("target_country", StringType(), True),
        StructField("target_age_group", StringType(), True),
        StructField("campaign_objective", StringType(), True),
        StructField("advertising_channel", StringType(), True),
    ]
)

AD_IMPRESSIONS_SCHEMA = StructType(
    [
        StructField("impression_id", StringType(), False),
        StructField("campaign_id", StringType(), False),
        StructField("traveler_id", StringType(), False),
        StructField("timestamp", StringType(), False),
        StructField("placement", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("impression_cost", DoubleType(), True),
        StructField("clicked", IntegerType(), True),
        StructField("converted", IntegerType(), True),
    ]
)

AD_INVENTORY_SCHEMA = StructType(
    [
        StructField("inventory_id", StringType(), False),
        StructField("placement", StringType(), False),
        StructField("country", StringType(), False),
        StructField("date", StringType(), False),
        StructField("available_impressions", IntegerType(), True),
        StructField("booked_impressions", IntegerType(), True),
        StructField("remaining_impressions", IntegerType(), True),
    ]
)

BIDDING_DATA_SCHEMA = StructType(
    [
        StructField("bid_id", StringType(), False),
        StructField("campaign_id", StringType(), False),
        StructField("placement", StringType(), True),
        StructField("bid_amount", DoubleType(), True),
        StructField("auction_timestamp", StringType(), True),
        StructField("advertiser_id", StringType(), True),
        StructField("auction_result", StringType(), True),
    ]
)

TARGETING_SIGNALS_SCHEMA = StructType(
    [
        StructField("traveler_id", StringType(), False),
        StructField("destination_interest", StringType(), True),
        StructField("travel_intent", DoubleType(), True),
        StructField("search_frequency", IntegerType(), True),
        StructField("booking_probability", DoubleType(), True),
        StructField("engagement_score", DoubleType(), True),
        StructField("signal_timestamp", StringType(), True),
    ]
)

DATASET_SCHEMAS: dict[str, StructType] = {
    "travelers": TRAVELERS_SCHEMA,
    "campaigns": CAMPAIGNS_SCHEMA,
    "ad_impressions": AD_IMPRESSIONS_SCHEMA,
    "ad_inventory": AD_INVENTORY_SCHEMA,
    "bidding_data": BIDDING_DATA_SCHEMA,
    "targeting_signals": TARGETING_SIGNALS_SCHEMA,
}


def read_dataset(
    spark: SparkSession,
    path: str | Path,
    schema: StructType | None = None,
    file_format: str = "csv",
) -> DataFrame:
    """Read a CSV or Parquet dataset with an explicit schema."""
    path_str = str(path)
    reader = spark.read
    if schema is not None:
        reader = reader.schema(schema)
    if file_format == "csv":
        df = reader.option("header", "true").option("mode", "PERMISSIVE").csv(path_str)
    elif file_format == "parquet":
        df = reader.parquet(path_str)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return df


def extract_all_raw(
    spark: SparkSession,
    config: PlatformConfig | None = None,
    source_dir: Path | None = None,
) -> dict[str, DataFrame]:
    """Extract all six raw advertising datasets with explicit schemas."""
    cfg = config or get_config()
    raw_dir = source_dir or cfg.resolve_path(cfg.paths.raw_data_dir)
    if not (raw_dir / "travelers.csv").exists():
        raw_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)

    datasets: dict[str, DataFrame] = {}
    for name, schema in DATASET_SCHEMAS.items():
        csv_path = raw_dir / f"{name}.csv"
        parquet_path = raw_dir / f"{name}.parquet"
        if parquet_path.exists():
            df = read_dataset(spark, parquet_path, schema=None, file_format="parquet")
        elif csv_path.exists():
            df = read_dataset(spark, csv_path, schema=schema, file_format="csv")
        else:
            raise FileNotFoundError(f"Neither {csv_path} nor {parquet_path} exists.")
        datasets[name] = df
        logger.info("Extracted '%s' from %s", name, raw_dir)

    return datasets
