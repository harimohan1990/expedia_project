"""Training & evaluation pipeline for Model 1: Audience Reach Forecasting."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from configs.settings import PlatformConfig, get_config, get_logger
from evaluation.metrics import compute_regression_metrics
from evaluation.model_comparison import CandidateResult, ModelEvaluator
from evaluation.validation import time_based_split
from features.audience_features import build_audience_reach_features

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "campaign_budget",
    "campaign_duration",
    "target_audience_size",
    "country_audience_size",
    "frequency_cap",
    "historical_unique_reach",
    "historical_impressions",
    "historical_engagement",
    "budget_per_day",
    "budget_to_audience_ratio",
    "start_month",
    "start_dow",
]

CATEGORICAL_FEATURES = [
    "country",
    "advertising_placement",
    "advertising_channel",
    "campaign_objective",
    "device_type",
]

TARGET_COL = "actual_unique_reach"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_audience_reach_models(config: PlatformConfig | None = None) -> dict[str, Any]:
    """Train LinearRegression, RandomForest, XGBoost, and LightGBM with time-based validation."""
    cfg = config or get_config()
    feat_path = (
        cfg.resolve_path(cfg.paths.processed_data_dir)
        / "features"
        / "audience_reach_features.parquet"
    )
    df = pd.read_parquet(feat_path) if feat_path.exists() else build_audience_reach_features(cfg)

    train_df, test_df = time_based_split(df, date_col="start_date", test_fraction=0.2)
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    seed = cfg.ml.random_state
    estimators = {
        "linear_regression": (
            LinearRegression(),
            {"model": "LinearRegression"},
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=180, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=seed
            ),
            {"model": "RandomForestRegressor", "n_estimators": 180, "max_depth": 12},
        ),
        "xgboost": (
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                n_jobs=-1,
            ),
            {"model": "XGBRegressor", "n_estimators": 300, "max_depth": 6, "lr": 0.05},
        ),
        "lightgbm": (
            LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            {"model": "LGBMRegressor", "n_estimators": 300, "max_depth": 8, "lr": 0.05},
        ),
    }

    candidates: list[CandidateResult] = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for name, (estimator, params) in estimators.items():
        pipe = Pipeline([("preprocessor", _build_preprocessor()), ("regressor", estimator)])
        pipe.fit(X_train, y_train)
        preds = np.maximum(0.0, pipe.predict(X_test))
        predictions_by_model[name] = preds
        metrics = compute_regression_metrics(y_test, preds)
        # Keep residual std for calibrated 95% confidence intervals
        resid_std = float(np.std(y_test - preds))
        bundle = {
            "pipeline": pipe,
            "residual_std": resid_std,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        }
        candidates.append(
            CandidateResult(
                model_name=name,
                task_type="regression",
                metrics=metrics,
                params=params,
                model_obj=bundle,
            )
        )
        logger.info("Audience Reach [%s]: %s", name, metrics)

    evaluator = ModelEvaluator(cfg.ml.mlflow_experiment_reach, cfg)
    best_cand, comparison_df = evaluator.log_and_compare(
        candidates=candidates,
        primary_metric="rmse",
        higher_is_better=False,
        registered_model_name="audience_reach_forecaster",
    )

    # Save test forecasts with confidence intervals
    best_preds = predictions_by_model[best_cand.model_name]
    resid_std = float(best_cand.model_obj["residual_std"])
    margin = 1.96 * resid_std
    lower = np.maximum(0.0, np.round(best_preds - margin)).astype(int)
    upper = np.maximum(lower, np.round(best_preds + margin)).astype(int)

    forecast_out = pd.DataFrame(
        {
            "campaign_id": test_df["campaign_id"].values,
            "forecast_date": test_df["forecast_date"].values,
            "country": test_df["country"].values,
            "advertising_placement": test_df["advertising_placement"].values,
            "actual_unique_reach": y_test.astype(int),
            "predicted_unique_reach": np.round(best_preds).astype(int),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_interval": [f"[{lo}, {hi}]" for lo, hi in zip(lower, upper, strict=False)],
        }
    )
    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    forecast_out.to_csv(reports_dir / "audience_reach_forecasts.csv", index=False)

    return {
        "best_model": best_cand.model_name,
        "best_metrics": best_cand.metrics,
        "comparison": comparison_df.to_dict(orient="records"),
        "forecast_rows": len(forecast_out),
    }


if __name__ == "__main__":
    print(json.dumps(train_audience_reach_models(), indent=2))
