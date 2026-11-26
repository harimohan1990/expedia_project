"""Centralized configuration and structured logging utilities."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    synthetic_data_dir: str = "data/synthetic"
    processed_data_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: str = "mlruns"


class DataGenerationConfig(BaseModel):
    num_travelers: int = 100_000
    num_campaigns: int = 10_000
    num_impressions: int = 1_000_000
    num_inventory_days: int = 1_400
    num_bids: int = 500_000
    num_targeting_signals: int = 200_000
    start_date: str = "2022-01-01"
    campaign_start_date: str = "2024-07-01"
    campaign_end_date: str = "2025-11-30"
    countries: list[str] = Field(default_factory=list)
    placements: list[str] = Field(default_factory=list)


class SparkConfig(BaseModel):
    app_name: str = "ExpediaAdIntelligenceETL"
    master: str = "local[*]"
    driver_memory: str = "4g"
    shuffle_partitions: int = 16
    sql_adaptive_enabled: bool = True
    watermark_delay: str = "10 minutes"
    window_duration: str = "5 minutes"


class KafkaConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    impressions_topic: str = "ad_impressions_stream"
    targeting_signals_topic: str = "targeting_signals_stream"
    predictions_topic: str = "realtime_predictions"
    consumer_group: str = "expedia-ad-streaming-group"


class DatabaseConfig(BaseModel):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "expedia_ads"
    postgres_user: str = "adtech"
    postgres_password: str = "adtech_dev_password"
    sqlite_fallback_path: str = "data/processed/expedia_ads.db"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    cache_ttl_seconds: int = 300


class AWSConfig(BaseModel):
    region: str = "us-west-2"
    s3_bucket: str = "expedia-ad-intelligence-datalake"
    s3_local_fallback_dir: str = "data/s3_datalake"
    cloudwatch_log_group: str = "/expedia/ad-intelligence"
    sagemaker_role_arn: str = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"


class MLConfig(BaseModel):
    mlflow_experiment_reach: str = "audience-reach-forecasting"
    mlflow_experiment_inventory: str = "inventory-forecasting"
    mlflow_experiment_performance: str = "campaign-performance-prediction"
    test_size_days: int = 60
    inventory_horizons: list[int] = Field(default_factory=lambda: [7, 14, 30])
    random_state: int = 42
    confidence_level: float = 0.95


class MonitoringConfig(BaseModel):
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    mape_degradation_threshold: float = 0.20
    rmse_degradation_ratio: float = 1.25
    latency_p95_ms_threshold: float = 250.0
    min_r2_threshold: float = 0.65


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"


class ProjectInfo(BaseModel):
    name: str = "Expedia Travel Advertising Intelligence & Forecasting Platform"
    version: str = "1.0.0"
    seed: int = 42


class PlatformConfig(BaseSettings):
    """Unified platform configuration loaded from YAML + environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="EXPEDIA_AD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectInfo = Field(default_factory=ProjectInfo)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data_generation: DataGenerationConfig = Field(default_factory=DataGenerationConfig)
    spark: SparkConfig = Field(default_factory=SparkConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    def resolve_path(self, relative_or_abs: str | Path) -> Path:
        path = Path(relative_or_abs)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> PlatformConfig:
    """Load YAML configuration with environment variable overrides."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    raw_cfg: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

    # Support standard DB/Redis/MLflow env overrides cleanly
    if os.getenv("POSTGRES_HOST"):
        raw_cfg.setdefault("database", {})["postgres_host"] = os.environ["POSTGRES_HOST"]
    if os.getenv("POSTGRES_PORT"):
        raw_cfg.setdefault("database", {})["postgres_port"] = int(os.environ["POSTGRES_PORT"])
    if os.getenv("POSTGRES_DB"):
        raw_cfg.setdefault("database", {})["postgres_db"] = os.environ["POSTGRES_DB"]
    if os.getenv("POSTGRES_USER"):
        raw_cfg.setdefault("database", {})["postgres_user"] = os.environ["POSTGRES_USER"]
    if os.getenv("POSTGRES_PASSWORD"):
        raw_cfg.setdefault("database", {})["postgres_password"] = os.environ["POSTGRES_PASSWORD"]
    if os.getenv("REDIS_HOST"):
        raw_cfg.setdefault("redis", {})["host"] = os.environ["REDIS_HOST"]
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        raw_cfg.setdefault("kafka", {})["bootstrap_servers"] = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    if os.getenv("MLFLOW_TRACKING_URI"):
        raw_cfg.setdefault("paths", {})["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

    return PlatformConfig(**raw_cfg)


def get_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
