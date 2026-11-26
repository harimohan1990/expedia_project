"""Kafka event producer for real-time advertising impressions and targeting signals."""

from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


def _broker_reachable(bootstrap_servers: str) -> bool:
    host_port = bootstrap_servers.split(",")[0].strip()
    host, _, port_str = host_port.partition(":")
    port = int(port_str or 9092)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class AdEventKafkaProducer:
    """Publishes ad impression and targeting signal events to Kafka (with local JSONL fallback)."""

    def __init__(self, config: PlatformConfig | None = None) -> None:
        self.config = config or get_config()
        self.producer = self._connect()
        self.fallback_dir = self.config.resolve_path("data/processed/stream_landing")
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> Any | None:
        if not _broker_reachable(self.config.kafka.bootstrap_servers):
            logger.info("Kafka broker unavailable; using local stream JSONL landing fallback")
            return None
        try:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                request_timeout_ms=2000,
                max_block_ms=2000,
            )
            return producer
        except Exception as exc:
            logger.info(
                "Kafka broker unavailable (%s); using local stream JSONL landing fallback", exc
            )
            return None

    def publish_batch(
        self,
        events: list[dict[str, Any]],
        topic: str,
        key_field: str = "traveler_id",
    ) -> int:
        if self.producer is not None:
            for ev in events:
                key = str(ev.get(key_field, ""))
                self.producer.send(topic, key=key, value=ev)
            self.producer.flush()
            return len(events)

        # Fallback: write micro-batch JSONL file for Spark Structured Streaming file source
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        out_dir = self.fallback_dir / topic
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"events_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        return len(events)

    def replay_from_csv(
        self,
        impressions_csv: Path | None = None,
        signals_csv: Path | None = None,
        limit: int = 500,
        batch_size: int = 100,
        sleep_seconds: float = 0.0,
    ) -> dict[str, int]:
        """Replay sample events from CSVs into the streaming topics."""
        raw_dir = self.config.resolve_path(self.config.paths.raw_data_dir)
        imp_path = impressions_csv or (raw_dir / "ad_impressions.csv")
        sig_path = signals_csv or (raw_dir / "targeting_signals.csv")

        counts = {"impressions": 0, "targeting_signals": 0}
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path, nrows=limit)
            records = imp_df.to_dict(orient="records")
            for i in range(0, len(records), batch_size):
                counts["impressions"] += self.publish_batch(
                    records[i : i + batch_size],
                    topic=self.config.kafka.impressions_topic,
                    key_field="campaign_id",
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        if sig_path.exists():
            sig_df = pd.read_csv(sig_path, nrows=limit)
            records = sig_df.to_dict(orient="records")
            for i in range(0, len(records), batch_size):
                counts["targeting_signals"] += self.publish_batch(
                    records[i : i + batch_size],
                    topic=self.config.kafka.targeting_signals_topic,
                    key_field="traveler_id",
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        logger.info("Published streaming replay events: %s", counts)
        return counts


if __name__ == "__main__":
    producer = AdEventKafkaProducer()
    producer.replay_from_csv(limit=200)
