"""Model prediction drift, performance degradation alerting, and CloudWatch logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np

from configs.settings import PlatformConfig, get_config, get_logger
from evaluation.metrics import compute_regression_metrics
from monitoring.data_drift import compute_psi
from monitoring.prometheus.metrics import DRIFT_ALERTS_TRIGGERED, MODEL_PERFORMANCE_GAUGE

logger = get_logger(__name__)


def emit_cloudwatch_alert(alert: dict[str, Any], config: PlatformConfig | None = None) -> str:
    """Send alert event to AWS CloudWatch Logs if configured, else log to local alert ledger."""
    cfg = config or get_config()
    try:
        session = boto3.Session(region_name=cfg.aws.region)
        if session.get_credentials() is not None:
            cw = session.client("logs")
            cw.put_log_events(
                logGroupName=cfg.aws.cloudwatch_log_group,
                logStreamName="model-monitoring-alerts",
                logEvents=[
                    {
                        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                        "message": json.dumps(alert),
                    }
                ],
            )
            return "cloudwatch"
    except Exception:
        pass

    alert_log = cfg.resolve_path(cfg.paths.reports_dir) / "alerts_history.jsonl"
    alert_log.parent.mkdir(parents=True, exist_ok=True)
    with open(alert_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")
    return "local_alert_log"


def evaluate_model_drift(
    model_name: str,
    y_ref_pred: np.ndarray,
    y_cur_true: np.ndarray,
    y_cur_pred: np.ndarray,
    baseline_rmse: float,
    baseline_r2: float | None = None,
    config: PlatformConfig | None = None,
) -> dict[str, Any]:
    """Detect prediction distribution drift and forecast accuracy degradation."""
    cfg = config or get_config()
    pred_psi = compute_psi(np.asarray(y_ref_pred, dtype=float), np.asarray(y_cur_pred, dtype=float))
    cur_metrics = compute_regression_metrics(y_cur_true, y_cur_pred)

    for metric_name, val in cur_metrics.items():
        MODEL_PERFORMANCE_GAUGE.labels(model_name=model_name, metric=metric_name).set(val)

    alerts: list[dict[str, Any]] = []
    if pred_psi >= cfg.monitoring.psi_warning_threshold:
        severity = "critical" if pred_psi >= cfg.monitoring.psi_critical_threshold else "warning"
        DRIFT_ALERTS_TRIGGERED.labels(severity=severity, alert_type="prediction_drift").inc()
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "alert_type": "prediction_drift",
            "severity": severity,
            "detail": f"Prediction PSI={pred_psi:.4f} exceeded threshold",
        }
        emit_cloudwatch_alert(alert, cfg)
        alerts.append(alert)

    if (
        baseline_rmse > 0
        and cur_metrics["rmse"] > baseline_rmse * cfg.monitoring.rmse_degradation_ratio
    ):
        DRIFT_ALERTS_TRIGGERED.labels(
            severity="critical", alert_type="performance_degradation"
        ).inc()
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "alert_type": "performance_degradation",
            "severity": "critical",
            "detail": f"Current RMSE={cur_metrics['rmse']:.2f} > {cfg.monitoring.rmse_degradation_ratio}x baseline ({baseline_rmse:.2f})",
        }
        emit_cloudwatch_alert(alert, cfg)
        alerts.append(alert)

    if baseline_r2 is not None and cur_metrics["r2"] < cfg.monitoring.min_r2_threshold:
        DRIFT_ALERTS_TRIGGERED.labels(severity="warning", alert_type="low_r2").inc()
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "alert_type": "low_r2",
            "severity": "warning",
            "detail": f"Current R2={cur_metrics['r2']:.4f} below min threshold {cfg.monitoring.min_r2_threshold}",
        }
        emit_cloudwatch_alert(alert, cfg)
        alerts.append(alert)

    return {
        "model_name": model_name,
        "prediction_psi": round(pred_psi, 4),
        "current_metrics": cur_metrics,
        "baseline_rmse": round(float(baseline_rmse), 4),
        "alerts_triggered": alerts,
    }
