"""Spark Structured Streaming Pipeline with Watermarks, Window Aggregations, and Real-Time Feature Scoring."""

from __future__ import annotations

import json
import socket
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from configs.settings import PlatformConfig, get_config, get_logger
from ingestion.kafka_producer import AdEventKafkaProducer
from spark.spark_session import create_spark_session

logger = get_logger(__name__)

STREAM_IMPRESSION_SCHEMA = StructType(
    [
        StructField("impression_id", StringType(), True),
        StructField("campaign_id", StringType(), True),
        StructField("traveler_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("placement", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("impression_cost", DoubleType(), True),
        StructField("clicked", IntegerType(), True),
        StructField("converted", IntegerType(), True),
    ]
)


def _is_kafka_reachable(bootstrap_servers: str) -> bool:
    host_port = bootstrap_servers.split(",")[0].strip()
    host, _, port_str = host_port.partition(":")
    port = int(port_str or 9092)
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def build_windowed_stream_aggregations(
    stream_df: DataFrame, watermark: str = "10 minutes"
) -> DataFrame:
    """Apply event-time parsing, watermark for late data, and 5-minute tumbling window aggregations."""
    with_ts = (
        stream_df.withColumn("event_timestamp", F.to_timestamp("timestamp"))
        .filter(F.col("event_timestamp").isNotNull())
        .withWatermark("event_timestamp", watermark)
    )
    agg_df = with_ts.groupBy(
        F.window("event_timestamp", "5 minutes"),
        F.col("placement"),
        F.col("device_type"),
    ).agg(
        F.count("impression_id").alias("window_impressions"),
        F.approx_count_distinct("traveler_id").alias("approx_unique_reach"),
        F.sum("clicked").alias("window_clicks"),
        F.sum("converted").alias("window_conversions"),
        F.round(F.sum("impression_cost"), 4).alias("window_spend"),
    )
    return agg_df.select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "placement",
        "device_type",
        "window_impressions",
        "approx_unique_reach",
        "window_clicks",
        "window_conversions",
        "window_spend",
        F.round(
            F.when(
                F.col("window_impressions") > 0,
                F.col("window_clicks") / F.col("window_impressions"),
            ).otherwise(0.0),
            6,
        ).alias("realtime_ctr"),
    )


def run_streaming_pipeline(
    config: PlatformConfig | None = None,
    max_microbatches: int = 1,
) -> dict[str, Any]:
    """Execute Spark Structured Streaming using Kafka or local micro-batch JSON landing fallback."""
    cfg = config or get_config()
    processed_dir = cfg.resolve_path(cfg.paths.processed_data_dir)
    out_dir = processed_dir / "streaming_predictions"
    ckpt_dir = processed_dir / "streaming_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sample streaming events exist if landing directory is empty
    stream_landing = cfg.resolve_path("data/processed/stream_landing") / cfg.kafka.impressions_topic
    if not stream_landing.exists() or not list(stream_landing.glob("*.json")):
        producer = AdEventKafkaProducer(cfg)
        producer.replay_from_csv(limit=300, batch_size=100)

    spark = create_spark_session(cfg, app_name="ExpediaAdStructuredStreaming")
    try:
        kafka_up = _is_kafka_reachable(cfg.kafka.bootstrap_servers)
        if kafka_up:
            logger.info("Reading stream from Kafka topic '%s'", cfg.kafka.impressions_topic)
            raw_stream = (
                spark.readStream.format("kafka")
                .option("kafka.bootstrap.servers", cfg.kafka.bootstrap_servers)
                .option("subscribe", cfg.kafka.impressions_topic)
                .option("startingOffsets", "earliest")
                .load()
                .select(
                    F.from_json(F.col("value").cast("string"), STREAM_IMPRESSION_SCHEMA).alias(
                        "data"
                    )
                )
                .select("data.*")
            )
            source_mode = "kafka"
        else:
            logger.info(
                "Kafka broker unavailable; running Structured Streaming file source at %s",
                stream_landing,
            )
            raw_stream = (
                spark.readStream.schema(STREAM_IMPRESSION_SCHEMA)
                .option("maxFilesPerTrigger", 2)
                .json(str(stream_landing))
            )
            source_mode = "local_file_stream"

        windowed = build_windowed_stream_aggregations(
            raw_stream, watermark=cfg.spark.watermark_delay
        )

        collected_rows: list[dict[str, Any]] = []

        def _process_microbatch(batch_df: DataFrame, batch_id: int) -> None:
            if batch_df.isEmpty():
                return
            pdf = batch_df.toPandas()
            pdf["batch_id"] = batch_id
            # Real-time CTR anomaly / predicted CTR scoring log
            pdf["predicted_ctr_baseline"] = 0.042
            pdf["ctr_lift_vs_baseline"] = (
                pdf["realtime_ctr"] - pdf["predicted_ctr_baseline"]
            ).round(6)
            out_file = out_dir / f"stream_batch_{batch_id}.parquet"
            pdf.to_parquet(out_file, index=False)
            collected_rows.extend(pdf.head(20).to_dict(orient="records"))
            logger.info(
                "Processed streaming micro-batch %d (%d window rows) -> %s",
                batch_id,
                len(pdf),
                out_file,
            )

        query = (
            windowed.writeStream.outputMode("update")
            .foreachBatch(_process_microbatch)
            .option("checkpointLocation", str(ckpt_dir))
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination(timeout=60)

        summary = {
            "source_mode": source_mode,
            "watermark_delay": cfg.spark.watermark_delay,
            "window_duration": cfg.spark.window_duration,
            "logged_batches_path": str(out_dir),
            "sample_records": len(collected_rows),
        }
        (out_dir / "_streaming_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary
    finally:
        spark.stop()


if __name__ == "__main__":
    run_streaming_pipeline()
