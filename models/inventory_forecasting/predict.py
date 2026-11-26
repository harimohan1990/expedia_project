"""Inference helper for 7, 14, and 30-day Inventory Forecasting."""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from configs.settings import get_config


@lru_cache(maxsize=1)
def _load_inventory_bundle() -> dict[str, Any]:
    cfg = get_config()
    model_path = (
        cfg.resolve_path(cfg.paths.models_dir)
        / cfg.ml.mlflow_experiment_inventory
        / "best_model.pkl"
    )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_inventory(
    country: str,
    placement: str,
    horizon_days: int = 14,
) -> list[dict[str, Any]]:
    """Forecast available impressions for a specific country & placement over `horizon_days`."""
    bundle = _load_inventory_bundle()
    pipe = bundle["pipeline"]
    hist_all: pd.DataFrame = bundle["latest_history"]
    hist = hist_all[
        (hist_all["country"] == country.upper()) & (hist_all["placement"] == placement.lower())
    ].sort_values("date")

    if hist.empty:
        raise ValueError(
            f"No historical inventory found for country={country}, placement={placement}"
        )

    last_date = pd.to_datetime(hist["date"].max())
    avail_buf = list(hist["available_impressions"].to_numpy(dtype=float))
    booked_buf = list(hist["booked_impressions"].to_numpy(dtype=float))
    served_buf = list(hist["historical_impressions_served"].to_numpy(dtype=float))
    last_demand = float(hist["advertiser_demand"].iloc[-1])

    out: list[dict[str, Any]] = []
    for step in range(1, horizon_days + 1):
        f_date = last_date + pd.Timedelta(days=step)
        dow = int(f_date.dayofweek)
        doy = int(f_date.dayofyear)
        row = {
            "country": country.upper(),
            "placement": placement.lower(),
            "avail_lag_1": avail_buf[-1],
            "avail_lag_7": avail_buf[-min(7, len(avail_buf))],
            "avail_lag_14": avail_buf[-min(14, len(avail_buf))],
            "avail_lag_28": avail_buf[-min(28, len(avail_buf))],
            "avail_lag_30": avail_buf[-min(30, len(avail_buf))],
            "booked_lag_7": booked_buf[-min(7, len(booked_buf))],
            "booked_lag_14": booked_buf[-min(14, len(booked_buf))],
            "avail_roll_mean_7": float(np.mean(avail_buf[-7:])),
            "avail_roll_mean_28": float(np.mean(avail_buf[-28:])),
            "avail_roll_std_14": float(np.std(avail_buf[-14:])),
            "booked_roll_mean_7": float(np.mean(booked_buf[-7:])),
            "booked_roll_mean_28": float(np.mean(booked_buf[-28:])),
            "booking_trends": float(np.mean(booked_buf[-7:]) / max(1.0, np.mean(booked_buf[-28:]))),
            "served_lag_7": served_buf[-min(7, len(served_buf))],
            "advertiser_demand": last_demand,
            "day_of_week": dow,
            "month": int(f_date.month),
            "is_weekend": int(dow >= 5),
            "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
            "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
            "sin_dow": float(np.sin(2 * np.pi * dow / 7.0)),
            "cos_dow": float(np.cos(2 * np.pi * dow / 7.0)),
        }
        pred = float(max(0.0, pipe.predict(pd.DataFrame([row]))[0]))
        avail_buf.append(pred)
        booked_buf.append(booked_buf[-min(7, len(booked_buf))])
        served_buf.append(served_buf[-min(7, len(served_buf))])
        out.append(
            {
                "date": f_date.strftime("%Y-%m-%d"),
                "placement": placement.lower(),
                "country": country.upper(),
                "predicted_available_impressions": int(round(pred)),
            }
        )
    return out
