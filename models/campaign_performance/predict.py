"""Inference helper for Pre-launch Campaign Performance Prediction."""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from configs.settings import get_config


@lru_cache(maxsize=1)
def _load_campaign_bundle() -> dict[str, Any]:
    cfg = get_config()
    model_path = (
        cfg.resolve_path(cfg.paths.models_dir)
        / cfg.ml.mlflow_experiment_performance
        / "best_model.pkl"
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_campaign_performance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict pre-launch campaign impressions, clicks, CTR, conversions, CVR, and revenue."""
    bundle = _load_campaign_bundle()
    target_models: dict[str, Any] = bundle["target_models"]
    clf = bundle["classifier"]

    df = pd.DataFrame(records)
    df["daily_budget"] = df["budget"] / df["campaign_duration_days"].clip(lower=1)
    if "start_month" not in df.columns:
        df["start_month"] = 6
    if "start_quarter" not in df.columns:
        df["start_quarter"] = 2
    if "start_dow" not in df.columns:
        df["start_dow"] = 2
    if "target_segment_size" not in df.columns:
        df["target_segment_size"] = 5000
    if "target_avg_booking_freq" not in df.columns:
        df["target_avg_booking_freq"] = 3.2
    if "planned_avg_bid" not in df.columns:
        df["planned_avg_bid"] = 18.0
    if "channel_hist_ctr" not in df.columns:
        df["channel_hist_ctr"] = 0.045
    if "channel_hist_cvr" not in df.columns:
        df["channel_hist_cvr"] = 0.012
    if "advertiser_hist_campaigns" not in df.columns:
        df["advertiser_hist_campaigns"] = 3
    if "advertiser_hist_roas" not in df.columns:
        df["advertiser_hist_roas"] = 1.5

    X = df[bundle["features"]]
    pred_dict = {
        target: np.maximum(0.0, model.predict(X)) for target, model in target_models.items()
    }
    high_perf_probs = clf.predict_proba(X)[:, 1]

    outputs: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        exp_imp = int(round(pred_dict["impressions"][i]))
        exp_clicks = int(round(pred_dict["clicks"][i]))
        exp_ctr = float(np.clip(pred_dict["ctr"][i], 0.0, 1.0))
        exp_conv = int(round(pred_dict["conversions"][i]))
        exp_cvr = float(np.clip(pred_dict["conversion_rate"][i], 0.0, 1.0))
        est_rev = round(float(pred_dict["campaign_revenue"][i]), 2)
        outputs.append(
            {
                "campaign_id": rec.get("campaign_id", f"NEW_CMP_{i+1}"),
                "expected_impressions": exp_imp,
                "expected_clicks": exp_clicks,
                "expected_ctr": round(exp_ctr, 6),
                "expected_conversions": exp_conv,
                "expected_conversion_rate": round(exp_cvr, 6),
                "estimated_campaign_revenue": est_rev,
                "high_performance_probability": round(float(high_perf_probs[i]), 4),
            }
        )
    return outputs
