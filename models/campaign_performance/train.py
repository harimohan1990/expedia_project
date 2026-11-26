"""Training & evaluation pipeline for Model 3: Pre-launch Campaign Performance Prediction.

Trains both classification (high-performing campaign flag) and multi-target regression models
for:
- expected_impressions
- expected_clicks
- expected_ctr
- expected_conversions
- expected_conversion_rate
- estimated_campaign_revenue

Algorithms compared: Logistic Regression, Random Forest, XGBoost, CatBoost.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from configs.settings import PlatformConfig, get_config, get_logger
from evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from evaluation.model_comparison import CandidateResult, ModelEvaluator
from evaluation.validation import time_based_split
from features.campaign_features import build_campaign_performance_features

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "budget",
    "campaign_duration_days",
    "daily_budget",
    "start_month",
    "start_quarter",
    "start_dow",
    "target_segment_size",
    "target_avg_booking_freq",
    "planned_avg_bid",
    "channel_hist_ctr",
    "channel_hist_cvr",
    "advertiser_hist_campaigns",
    "advertiser_hist_roas",
]

CATEGORICAL_FEATURES = [
    "target_country",
    "target_age_group",
    "campaign_objective",
    "advertising_channel",
]

REGRESSION_TARGETS = [
    "impressions",
    "clicks",
    "ctr",
    "conversions",
    "conversion_rate",
    "campaign_revenue",
]


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


def train_campaign_performance_models(config: PlatformConfig | None = None) -> dict[str, Any]:
    """Train classification & regression models for pre-launch campaign performance prediction."""
    cfg = config or get_config()
    feat_path = (
        cfg.resolve_path(cfg.paths.processed_data_dir)
        / "features"
        / "campaign_performance_features.parquet"
    )
    df = (
        pd.read_parquet(feat_path)
        if feat_path.exists()
        else build_campaign_performance_features(cfg)
    )

    train_df, test_df = time_based_split(df, date_col="start_date", test_fraction=0.2)
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    seed = cfg.ml.random_state

    # Part A: Classification (is_high_performing) comparing Logistic Regression, RF, XGBoost, CatBoost
    y_cls_train = train_df["is_high_performing"].to_numpy(dtype=int)
    y_cls_test = test_df["is_high_performing"].to_numpy(dtype=int)

    cls_specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=seed),
            {"model": "LogisticRegression"},
        ),
        "random_forest_cls": (
            RandomForestClassifier(n_estimators=180, max_depth=10, random_state=seed, n_jobs=-1),
            {"model": "RandomForestClassifier", "n_estimators": 180},
        ),
        "xgboost_cls": (
            XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
            {"model": "XGBClassifier", "n_estimators": 220, "max_depth": 5},
        ),
        "catboost_cls": (
            CatBoostClassifier(
                iterations=240,
                depth=6,
                learning_rate=0.05,
                random_seed=seed,
                verbose=0,
            ),
            {"model": "CatBoostClassifier", "iterations": 240, "depth": 6},
        ),
    }

    cls_candidates: list[CandidateResult] = []
    best_cls_probs: dict[str, np.ndarray] = {}
    for name, (est, params) in cls_specs.items():
        pipe = Pipeline([("pre", _build_preprocessor()), ("clf", est)])
        pipe.fit(X_train, y_cls_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        best_cls_probs[name] = probs
        metrics = compute_classification_metrics(y_cls_test, probs)
        cls_candidates.append(
            CandidateResult(
                model_name=name,
                task_type="classification",
                metrics=metrics,
                params=params,
                model_obj=pipe,
            )
        )
        logger.info("Campaign Performance Cls [%s]: %s", name, metrics)

    # Part B: Regression across all 6 performance metrics (using Ridge, RF, XGBoost, CatBoost)
    reg_specs = {
        "ridge_regression": (
            Ridge(alpha=1.0),
            {"model": "Ridge"},
        ),
        "random_forest_reg": (
            RandomForestRegressor(n_estimators=160, max_depth=10, random_state=seed, n_jobs=-1),
            {"model": "RandomForestRegressor", "n_estimators": 160},
        ),
        "xgboost_reg": (
            XGBRegressor(
                n_estimators=220, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=-1
            ),
            {"model": "XGBRegressor", "n_estimators": 220},
        ),
        "catboost_reg": (
            CatBoostRegressor(
                iterations=240, depth=6, learning_rate=0.05, random_seed=seed, verbose=0
            ),
            {"model": "CatBoostRegressor", "iterations": 240},
        ),
    }

    # Evaluate all 4 regressors on primary revenue target + train best regressor family across all 6 targets
    reg_candidates: list[CandidateResult] = []
    rev_preds_map: dict[str, np.ndarray] = {}
    y_rev_train = train_df["campaign_revenue"].to_numpy(dtype=float)
    y_rev_test = test_df["campaign_revenue"].to_numpy(dtype=float)

    for name, (est, params) in reg_specs.items():
        pipe = Pipeline([("pre", _build_preprocessor()), ("reg", est)])
        pipe.fit(X_train, y_rev_train)
        preds = np.maximum(0.0, pipe.predict(X_test))
        rev_preds_map[name] = preds
        metrics = compute_regression_metrics(y_rev_test, preds)
        reg_candidates.append(
            CandidateResult(
                model_name=name,
                task_type="regression",
                metrics=metrics,
                params=params,
                model_obj=pipe,
            )
        )
        logger.info("Campaign Revenue Reg [%s]: %s", name, metrics)

    # Train one multi-target suite using the top-performing regressor architecture (and bundle with best classifier)
    evaluator_cls = ModelEvaluator(f"{cfg.ml.mlflow_experiment_performance}-cls", cfg)
    best_cls, cls_comp = evaluator_cls.log_and_compare(
        candidates=cls_candidates,
        primary_metric="roc_auc",
        higher_is_better=True,
        registered_model_name="campaign_high_perf_classifier",
    )

    # Train per-target regressors with XGBoost & CatBoost comparison on campaign_revenue
    evaluator_reg = ModelEvaluator(cfg.ml.mlflow_experiment_performance, cfg)

    # Fit full 6-target suite using the winning regressor type
    best_reg_name_prelim = min(reg_candidates, key=lambda c: c.metrics["rmse"]).model_name
    target_models: dict[str, Pipeline] = {}
    target_metrics: dict[str, dict[str, float]] = {}
    test_preds_df = pd.DataFrame({"campaign_id": test_df["campaign_id"].values})

    for target in REGRESSION_TARGETS:
        if best_reg_name_prelim == "catboost_reg":
            est = CatBoostRegressor(
                iterations=240, depth=6, learning_rate=0.05, random_seed=seed, verbose=0
            )
        elif best_reg_name_prelim == "xgboost_reg":
            est = XGBRegressor(
                n_estimators=220, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=-1
            )
        elif best_reg_name_prelim == "random_forest_reg":
            est = RandomForestRegressor(
                n_estimators=160, max_depth=10, random_state=seed, n_jobs=-1
            )
        else:
            est = Ridge(alpha=1.0)

        pipe = Pipeline([("pre", _build_preprocessor()), ("reg", est)])
        pipe.fit(X_train, train_df[target].to_numpy(dtype=float))
        preds = np.maximum(0.0, pipe.predict(X_test))
        if target in ("ctr", "conversion_rate"):
            preds = np.clip(preds, 0.0, 1.0)
        target_models[target] = pipe
        target_metrics[target] = compute_regression_metrics(
            test_df[target].to_numpy(dtype=float), preds
        )
        test_preds_df[f"actual_{target}"] = test_df[target].values
        test_preds_df[f"predicted_{target}"] = preds

    test_preds_df["high_performing_prob"] = best_cls_probs[best_cls.model_name]

    # Attach complete multi-target bundle onto winning revenue candidate before logging
    for cand in reg_candidates:
        if cand.model_name == best_reg_name_prelim:
            cand.model_obj = {
                "target_models": target_models,
                "classifier": best_cls.model_obj,
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "target_metrics": target_metrics,
            }

    best_reg, reg_comp = evaluator_reg.log_and_compare(
        candidates=reg_candidates,
        primary_metric="rmse",
        higher_is_better=False,
        registered_model_name="campaign_performance_predictor",
    )

    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    test_preds_df.to_csv(reports_dir / "campaign_performance_predictions.csv", index=False)
    (reports_dir / "campaign_target_metrics.json").write_text(
        json.dumps(target_metrics, indent=2), encoding="utf-8"
    )

    return {
        "best_classifier": best_cls.model_name,
        "best_classifier_metrics": best_cls.metrics,
        "best_regressor": best_reg.model_name,
        "best_regressor_metrics": best_reg.metrics,
        "target_metrics": target_metrics,
        "classification_comparison": cls_comp.to_dict(orient="records"),
        "regression_comparison": reg_comp.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(json.dumps(train_campaign_performance_models(), indent=2))
