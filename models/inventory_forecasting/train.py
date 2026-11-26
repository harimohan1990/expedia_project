"""Training, comparison, and 7/14/30-day forecasting for Model 2: Inventory Forecasting.

Compares five algorithms on an identical temporal holdout validation set:
- ARIMA
- SARIMA
- Prophet
- XGBoost
- LightGBM
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from configs.settings import PlatformConfig, get_config, get_logger
from evaluation.metrics import compute_regression_metrics
from evaluation.model_comparison import CandidateResult, ModelEvaluator
from features.inventory_features import build_inventory_features

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "avail_lag_1",
    "avail_lag_7",
    "avail_lag_14",
    "avail_lag_28",
    "avail_lag_30",
    "booked_lag_7",
    "booked_lag_14",
    "avail_roll_mean_7",
    "avail_roll_mean_28",
    "avail_roll_std_14",
    "booked_roll_mean_7",
    "booked_roll_mean_28",
    "booking_trends",
    "served_lag_7",
    "advertiser_demand",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
]
CATEGORICAL_FEATURES = ["country", "placement"]
TARGET_COL = "available_impressions"


def _evaluate_ts_models_on_panel(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    benchmark_pairs: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    """Run ARIMA, SARIMA, and Prophet per (country, placement) benchmark series over identical holdout dates."""
    results: dict[str, dict[str, Any]] = {
        "arima": {"y_true": [], "y_pred": [], "models": {}},
        "sarima": {"y_true": [], "y_pred": [], "models": {}},
        "prophet": {"y_true": [], "y_pred": [], "models": {}},
    }

    for country, placement in benchmark_pairs:
        sub = df[(df["country"] == country) & (df["placement"] == placement)].sort_values("date")
        train_s = sub[sub["date"] < cutoff_date]
        test_s = sub[sub["date"] >= cutoff_date]
        if len(train_s) < 60 or len(test_s) == 0:
            continue

        y_tr = train_s[TARGET_COL].to_numpy(dtype=float)
        y_te = test_s[TARGET_COL].to_numpy(dtype=float)
        horizon = len(test_s)

        # 1. ARIMA(2,1,2)
        try:
            arima_fit = ARIMA(y_tr, order=(2, 1, 2)).fit()
            arima_pred = np.maximum(0.0, arima_fit.forecast(steps=horizon))
        except Exception:
            arima_pred = np.repeat(np.mean(y_tr[-14:]), horizon)
        results["arima"]["y_true"].extend(y_te)
        results["arima"]["y_pred"].extend(arima_pred)
        results["arima"]["models"][f"{country}_{placement}"] = {"last_train": float(y_tr[-1])}

        # 2. SARIMA(1,1,1)x(1,0,1,7) for weekly travel seasonality
        try:
            sarima_fit = SARIMAX(
                y_tr[-365:],
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            sarima_pred = np.maximum(0.0, sarima_fit.forecast(steps=horizon))
        except Exception:
            sarima_pred = np.repeat(np.mean(y_tr[-7:]), horizon)
        results["sarima"]["y_true"].extend(y_te)
        results["sarima"]["y_pred"].extend(sarima_pred)
        results["sarima"]["models"][f"{country}_{placement}"] = {
            "last_7_mean": float(np.mean(y_tr[-7:]))
        }

        # 3. Prophet
        try:
            p_train = pd.DataFrame({"ds": train_s["date"].values, "y": y_tr})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(p_train)
            future = pd.DataFrame({"ds": test_s["date"].values})
            p_pred = np.maximum(0.0, m.predict(future)["yhat"].to_numpy())
        except Exception:
            p_pred = np.repeat(np.mean(y_tr[-14:]), horizon)
        results["prophet"]["y_true"].extend(y_te)
        results["prophet"]["y_pred"].extend(p_pred)
        results["prophet"]["models"][f"{country}_{placement}"] = {"series_len": len(y_tr)}

    return results


def _generate_recursive_forecasts(
    df: pd.DataFrame,
    model_pipe: Pipeline,
    horizons: list[int],
) -> pd.DataFrame:
    """Generate recursive 1..max(horizons) day forecasts for every (country, placement) series."""
    max_h = max(horizons)
    last_date = df["date"].max()
    forecast_rows: list[dict[str, Any]] = []

    for (country, placement), grp_df in df.groupby(["country", "placement"]):
        hist = grp_df.sort_values("date").tail(45).copy()
        avail_buf = list(hist["available_impressions"].to_numpy(dtype=float))
        booked_buf = list(hist["booked_impressions"].to_numpy(dtype=float))
        served_buf = list(hist["historical_impressions_served"].to_numpy(dtype=float))
        last_demand = float(hist["advertiser_demand"].iloc[-1])

        for step in range(1, max_h + 1):
            f_date = last_date + pd.Timedelta(days=step)
            dow = int(f_date.dayofweek)
            doy = int(f_date.dayofyear)
            row = {
                "country": country,
                "placement": placement,
                "avail_lag_1": avail_buf[-1],
                "avail_lag_7": avail_buf[-7],
                "avail_lag_14": avail_buf[-14],
                "avail_lag_28": avail_buf[-28],
                "avail_lag_30": avail_buf[-30],
                "booked_lag_7": booked_buf[-7],
                "booked_lag_14": booked_buf[-14],
                "avail_roll_mean_7": float(np.mean(avail_buf[-7:])),
                "avail_roll_mean_28": float(np.mean(avail_buf[-28:])),
                "avail_roll_std_14": float(np.std(avail_buf[-14:])),
                "booked_roll_mean_7": float(np.mean(booked_buf[-7:])),
                "booked_roll_mean_28": float(np.mean(booked_buf[-28:])),
                "booking_trends": float(
                    np.mean(booked_buf[-7:]) / max(1.0, np.mean(booked_buf[-28:]))
                ),
                "served_lag_7": served_buf[-7],
                "advertiser_demand": last_demand,
                "day_of_week": dow,
                "month": int(f_date.month),
                "is_weekend": int(dow >= 5),
                "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
                "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
                "sin_dow": float(np.sin(2 * np.pi * dow / 7.0)),
                "cos_dow": float(np.cos(2 * np.pi * dow / 7.0)),
            }
            pred_avail = float(max(0.0, model_pipe.predict(pd.DataFrame([row]))[0]))
            avail_buf.append(pred_avail)
            booked_buf.append(booked_buf[-7])
            served_buf.append(served_buf[-7])

            horizon_tag = 7 if step <= 7 else (14 if step <= 14 else 30)
            forecast_rows.append(
                {
                    "date": f_date.strftime("%Y-%m-%d"),
                    "placement": placement,
                    "country": country,
                    "horizon_days": horizon_tag,
                    "step_day": step,
                    "predicted_available_impressions": int(round(pred_avail)),
                }
            )

    return pd.DataFrame(forecast_rows)


def train_inventory_models(config: PlatformConfig | None = None) -> dict[str, Any]:
    """Train and compare ARIMA, SARIMA, Prophet, XGBoost, and LightGBM on 30-day temporal holdout."""
    cfg = config or get_config()
    feat_path = (
        cfg.resolve_path(cfg.paths.processed_data_dir) / "features" / "inventory_features.parquet"
    )
    df = pd.read_parquet(feat_path) if feat_path.exists() else build_inventory_features(cfg)
    df["date"] = pd.to_datetime(df["date"])

    # Use last 30 days as strict temporal holdout across all 72 (country, placement) series
    cutoff_date = df["date"].max() - pd.Timedelta(days=29)
    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    # Representative 12 (country, placement) series for classical TS models (fair benchmark slice)
    top_countries = ["US", "UK", "CA", "DE", "JP", "AU"]
    top_placements = ["homepage_hero", "search_results_top"]
    benchmark_pairs = [(c, p) for c in top_countries for p in top_placements]
    bench_mask = test_df.apply(lambda r: (r["country"], r["placement"]) in benchmark_pairs, axis=1)

    logger.info(
        "Evaluating classical time-series models (ARIMA, SARIMA, Prophet) on 30-day holdout..."
    )
    ts_eval = _evaluate_ts_models_on_panel(df, cutoff_date, benchmark_pairs)

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    def _make_pre() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

    seed = cfg.ml.random_state
    gbdt_specs = {
        "xgboost": (
            XGBRegressor(
                n_estimators=260,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                n_jobs=-1,
            ),
            {"model": "XGBRegressor", "n_estimators": 260, "max_depth": 6},
        ),
        "lightgbm": (
            LGBMRegressor(
                n_estimators=260,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
            {"model": "LGBMRegressor", "n_estimators": 260, "max_depth": 8},
        ),
    }

    candidates: list[CandidateResult] = []
    # Classical TS candidates evaluated on benchmark slice
    for ts_name in ["arima", "sarima", "prophet"]:
        yt = np.array(ts_eval[ts_name]["y_true"], dtype=float)
        yp = np.array(ts_eval[ts_name]["y_pred"], dtype=float)
        metrics = compute_regression_metrics(yt, yp)
        candidates.append(
            CandidateResult(
                model_name=ts_name,
                task_type="regression",
                metrics=metrics,
                params={"model": ts_name.upper(), "eval_series": len(benchmark_pairs)},
                model_obj={"type": ts_name, "series_summaries": ts_eval[ts_name]["models"]},
            )
        )
        logger.info("Inventory [%s] (benchmark): %s", ts_name, metrics)

    # GBDT global models evaluated both on benchmark slice (apples-to-apples) and full test
    gbdt_pipelines: dict[str, Pipeline] = {}
    for name, (est, params) in gbdt_specs.items():
        pipe = Pipeline([("pre", _make_pre()), ("reg", est)])
        pipe.fit(X_train, y_train)
        preds_all = np.maximum(0.0, pipe.predict(X_test))
        metrics_all = compute_regression_metrics(y_test, preds_all)
        metrics_bench = compute_regression_metrics(y_test[bench_mask], preds_all[bench_mask])
        gbdt_pipelines[name] = pipe

        bundle = {
            "pipeline": pipe,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "full_test_metrics": metrics_all,
            "latest_history": df.sort_values("date").groupby(["country", "placement"]).tail(45),
        }
        candidates.append(
            CandidateResult(
                model_name=name,
                task_type="regression",
                metrics=metrics_bench,
                params={**params, "full_test_mape": metrics_all["mape"]},
                model_obj=bundle,
            )
        )
        logger.info("Inventory [%s] (benchmark=%s | full=%s)", name, metrics_bench, metrics_all)

    evaluator = ModelEvaluator(cfg.ml.mlflow_experiment_inventory, cfg)
    best_cand, comparison_df = evaluator.log_and_compare(
        candidates=candidates,
        primary_metric="mape",
        higher_is_better=False,
        registered_model_name="inventory_forecaster",
    )

    # Use best GBDT pipeline (or best overall if GBDT won) to emit 7, 14, and 30-day forecasts for ALL 72 series
    best_pipe = (
        best_cand.model_obj["pipeline"]
        if "pipeline" in best_cand.model_obj
        else gbdt_pipelines["lightgbm"]
    )
    forecast_df = _generate_recursive_forecasts(df, best_pipe, cfg.ml.inventory_horizons)
    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(reports_dir / "inventory_forecasts_30d.csv", index=False)

    # Also save actual vs predicted holdout for dashboard visualization
    holdout_vis = test_df[["date", "country", "placement", "available_impressions"]].copy()
    holdout_vis["predicted_available_impressions"] = np.round(
        np.maximum(0.0, best_pipe.predict(X_test))
    ).astype(int)
    holdout_vis["date"] = holdout_vis["date"].dt.strftime("%Y-%m-%d")
    holdout_vis.to_csv(reports_dir / "inventory_holdout_predictions.csv", index=False)

    return {
        "best_model": best_cand.model_name,
        "best_metrics": best_cand.metrics,
        "comparison": comparison_df.to_dict(orient="records"),
        "forecast_rows": len(forecast_df),
    }


if __name__ == "__main__":
    print(json.dumps(train_inventory_models(), indent=2))
