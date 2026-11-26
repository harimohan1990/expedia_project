"""End-to-end monitoring runner for data drift, model drift, and alert checks across all 3 models."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger
from monitoring.data_drift import run_data_drift_check
from monitoring.model_drift import evaluate_model_drift

logger = get_logger(__name__)


def run_all_monitoring(config: PlatformConfig | None = None) -> dict[str, Any]:
    """Run data drift and model drift evaluations comparing early vs recent windows."""
    cfg = config or get_config()
    feat_dir = cfg.resolve_path(cfg.paths.processed_data_dir) / "features"
    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    models_dir = cfg.resolve_path(cfg.paths.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    monitoring_summary: dict[str, Any] = {"data_drift": {}, "model_drift": {}}

    # 1. Audience Reach data & prediction drift
    reach_feat_path = feat_dir / "audience_reach_features.parquet"
    reach_pred_path = reports_dir / "audience_reach_forecasts.csv"
    if reach_feat_path.exists():
        reach_df = pd.read_parquet(reach_feat_path).sort_values("start_date")
        mid = len(reach_df) // 2
        monitoring_summary["data_drift"]["audience_reach"] = run_data_drift_check(
            reference_df=reach_df.iloc[:mid],
            current_df=reach_df.iloc[mid:],
            model_name="audience_reach",
            numeric_cols=[
                "campaign_budget",
                "campaign_duration",
                "target_audience_size",
                "historical_unique_reach",
                "historical_engagement",
            ],
            expected_schema=["campaign_id", "campaign_budget", "country", "advertising_placement"],
            config=cfg,
        )
    if reach_pred_path.exists():
        rp = pd.read_csv(reach_pred_path)
        mid_p = len(rp) // 2
        eval_sum = json.loads(
            (models_dir / cfg.ml.mlflow_experiment_reach / "evaluation_summary.json").read_text()
        )
        monitoring_summary["model_drift"]["audience_reach"] = evaluate_model_drift(
            model_name="audience_reach",
            y_ref_pred=rp["predicted_unique_reach"].iloc[:mid_p].to_numpy(),
            y_cur_true=rp["actual_unique_reach"].iloc[mid_p:].to_numpy(),
            y_cur_pred=rp["predicted_unique_reach"].iloc[mid_p:].to_numpy(),
            baseline_rmse=eval_sum["best_metrics"]["rmse"],
            baseline_r2=eval_sum["best_metrics"]["r2"],
            config=cfg,
        )

    # 2. Inventory data & prediction drift
    inv_feat_path = feat_dir / "inventory_features.parquet"
    inv_pred_path = reports_dir / "inventory_holdout_predictions.csv"
    if inv_feat_path.exists():
        inv_df = pd.read_parquet(inv_feat_path).sort_values("date")
        mid = int(len(inv_df) * 0.8)
        monitoring_summary["data_drift"]["inventory_forecasting"] = run_data_drift_check(
            reference_df=inv_df.iloc[:mid],
            current_df=inv_df.iloc[mid:],
            model_name="inventory_forecasting",
            numeric_cols=[
                "avail_lag_1",
                "avail_lag_7",
                "booked_roll_mean_7",
                "advertiser_demand",
                "booking_trends",
            ],
            expected_schema=["country", "placement", "date", "available_impressions"],
            config=cfg,
        )
    if inv_pred_path.exists():
        ip = pd.read_csv(inv_pred_path)
        mid_p = len(ip) // 2
        eval_sum = json.loads(
            (
                models_dir / cfg.ml.mlflow_experiment_inventory / "evaluation_summary.json"
            ).read_text()
        )
        monitoring_summary["model_drift"]["inventory_forecasting"] = evaluate_model_drift(
            model_name="inventory_forecasting",
            y_ref_pred=ip["predicted_available_impressions"].iloc[:mid_p].to_numpy(),
            y_cur_true=ip["available_impressions"].iloc[mid_p:].to_numpy(),
            y_cur_pred=ip["predicted_available_impressions"].iloc[mid_p:].to_numpy(),
            baseline_rmse=eval_sum["best_metrics"]["rmse"],
            config=cfg,
        )

    # 3. Campaign Performance data & prediction drift
    camp_feat_path = feat_dir / "campaign_performance_features.parquet"
    camp_pred_path = reports_dir / "campaign_performance_predictions.csv"
    if camp_feat_path.exists():
        camp_df = pd.read_parquet(camp_feat_path).sort_values("start_date")
        mid = len(camp_df) // 2
        monitoring_summary["data_drift"]["campaign_performance"] = run_data_drift_check(
            reference_df=camp_df.iloc[:mid],
            current_df=camp_df.iloc[mid:],
            model_name="campaign_performance",
            numeric_cols=[
                "budget",
                "campaign_duration_days",
                "planned_avg_bid",
                "channel_hist_ctr",
            ],
            expected_schema=["campaign_id", "budget", "target_country", "advertising_channel"],
            config=cfg,
        )
    if camp_pred_path.exists():
        cp = pd.read_csv(camp_pred_path)
        mid_p = len(cp) // 2
        eval_sum = json.loads(
            (
                models_dir / cfg.ml.mlflow_experiment_performance / "evaluation_summary.json"
            ).read_text()
        )
        monitoring_summary["model_drift"]["campaign_performance"] = evaluate_model_drift(
            model_name="campaign_performance",
            y_ref_pred=cp["predicted_campaign_revenue"].iloc[:mid_p].to_numpy(),
            y_cur_true=cp["actual_campaign_revenue"].iloc[mid_p:].to_numpy(),
            y_cur_pred=cp["predicted_campaign_revenue"].iloc[mid_p:].to_numpy(),
            baseline_rmse=eval_sum["best_metrics"]["rmse"],
            baseline_r2=eval_sum["best_metrics"]["r2"],
            config=cfg,
        )

    out_path = reports_dir / "monitoring_report.json"
    out_path.write_text(json.dumps(monitoring_summary, indent=2, default=str), encoding="utf-8")
    logger.info("Monitoring report saved -> %s", out_path)
    return monitoring_summary


if __name__ == "__main__":
    print(json.dumps(run_all_monitoring(), indent=2, default=str))
