"""Inference helper for Audience Reach Forecasting."""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from configs.settings import get_config


@lru_cache(maxsize=1)
def _load_reach_bundle() -> dict[str, Any]:
    cfg = get_config()
    model_path = (
        cfg.resolve_path(cfg.paths.models_dir) / cfg.ml.mlflow_experiment_reach / "best_model.pkl"
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_audience_reach(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict unique traveler reach and 95% confidence intervals for campaign requests."""
    bundle = _load_reach_bundle()
    pipe = bundle["pipeline"]
    resid_std = float(bundle.get("residual_std", 15.0))
    margin = 1.96 * resid_std

    df = pd.DataFrame(records)
    # Derive engineered ratios if not supplied
    df["budget_per_day"] = df["campaign_budget"] / df["campaign_duration"].clip(lower=1)
    df["budget_to_audience_ratio"] = df["campaign_budget"] / df["target_audience_size"].clip(
        lower=1
    )
    if "country_audience_size" not in df.columns:
        df["country_audience_size"] = df["target_audience_size"] * 4
    if "start_month" not in df.columns:
        df["start_month"] = pd.to_datetime(df["forecast_date"]).dt.month
    if "start_dow" not in df.columns:
        df["start_dow"] = pd.to_datetime(df["forecast_date"]).dt.dayofweek
    if "advertising_channel" not in df.columns:
        df["advertising_channel"] = "search_ads"
    if "campaign_objective" not in df.columns:
        df["campaign_objective"] = "bookings"

    preds = np.maximum(0.0, pipe.predict(df[bundle["features"]]))
    results = []
    for row, pred in zip(records, preds, strict=False):
        lo = max(0, int(round(pred - margin)))
        hi = max(lo, int(round(pred + margin)))
        results.append(
            {
                "campaign_id": row["campaign_id"],
                "forecast_date": str(row["forecast_date"]),
                "predicted_unique_reach": int(round(pred)),
                "ci_lower": lo,
                "ci_upper": hi,
                "confidence_interval": f"[{lo}, {hi}]",
            }
        )
    return results
