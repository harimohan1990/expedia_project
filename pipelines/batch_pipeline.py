"""End-to-end Batch ML Pipeline orchestrating ETL -> Features -> Training -> Registration -> DB Storage."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger
from features.build_all_features import build_all
from models.audience_reach.train import train_audience_reach_models
from models.campaign_performance.train import train_campaign_performance_models
from models.inventory_forecasting.train import train_inventory_models
from spark.run_etl import run_etl_pipeline
from utils.storage import DatabaseClient

logger = get_logger(__name__)


def run_batch_ml_pipeline(
    config: PlatformConfig | None = None,
    skip_etl: bool = False,
) -> dict[str, Any]:
    """Execute the daily batch ML pipeline and persist predictions to PostgreSQL/SQLite."""
    cfg = config or get_config()
    summary: dict[str, Any] = {}

    if not skip_etl:
        logger.info("Step 1/4: Running PySpark ETL...")
        summary["etl"] = run_etl_pipeline(cfg)

    logger.info("Step 2/4: Building ML feature tables...")
    summary["features"] = build_all(cfg)

    logger.info("Step 3/4: Training & evaluating all three forecasting models...")
    summary["audience_reach"] = train_audience_reach_models(cfg)
    summary["inventory_forecasting"] = train_inventory_models(cfg)
    summary["campaign_performance"] = train_campaign_performance_models(cfg)

    logger.info("Step 4/4: Persisting forecasts to relational database...")
    db = DatabaseClient(cfg)
    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)

    for fname, table in [
        ("audience_reach_forecasts.csv", "pred_audience_reach"),
        ("inventory_forecasts_30d.csv", "pred_inventory_forecasts"),
        ("campaign_performance_predictions.csv", "pred_campaign_performance"),
    ]:
        fpath = reports_dir / fname
        if fpath.exists():
            db.write_dataframe(pd.read_csv(fpath), table, if_exists="replace")

    logger.info("Batch ML Pipeline complete.")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_batch_ml_pipeline(skip_etl=True), indent=2, default=str))
