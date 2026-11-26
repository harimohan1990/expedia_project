"""Unit and integration tests for evaluation metrics, time-based validation, and ML models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from configs.settings import PlatformConfig
from evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from evaluation.validation import time_based_split
from features.build_all_features import build_all
from models.audience_reach.train import train_audience_reach_models
from models.campaign_performance.train import train_campaign_performance_models
from models.inventory_forecasting.train import train_inventory_models


def test_regression_and_classification_metrics() -> None:
    yt = np.array([100.0, 200.0, 300.0, 400.0])
    yp = np.array([110.0, 190.0, 310.0, 390.0])
    reg = compute_regression_metrics(yt, yp)
    assert reg["mae"] == 10.0
    assert reg["rmse"] == 10.0
    assert 0.0 < reg["wape"] < 0.05
    assert reg["r2"] > 0.98

    yc = np.array([0, 0, 1, 1])
    yprob = np.array([0.1, 0.4, 0.6, 0.9])
    cls = compute_classification_metrics(yc, yprob)
    assert cls["precision"] == 1.0
    assert cls["recall"] == 1.0
    assert cls["f1"] == 1.0
    assert cls["roc_auc"] == 1.0


def test_time_based_split_is_chronological() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="D")[::-1],
            "val": range(10),
        }
    )
    train_df, test_df = time_based_split(df, date_col="date", test_fraction=0.2)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert train_df["date"].max() < test_df["date"].min()


def test_all_three_models_train_on_scaled_data(
    test_config: PlatformConfig,
    synthetic_dataset_paths: dict[str, Path],
) -> None:
    feat_counts = build_all(test_config)
    assert all(v > 0 for v in feat_counts.values())

    reach_res = train_audience_reach_models(test_config)
    assert reach_res["best_model"] in {"linear_regression", "random_forest", "xgboost", "lightgbm"}
    assert reach_res["forecast_rows"] > 0

    inv_res = train_inventory_models(test_config)
    assert inv_res["best_model"] in {"arima", "sarima", "prophet", "xgboost", "lightgbm"}
    assert inv_res["forecast_rows"] > 0

    camp_res = train_campaign_performance_models(test_config)
    assert "best_classifier" in camp_res
    assert "best_regressor" in camp_res
    assert len(camp_res["target_metrics"]) == 6
