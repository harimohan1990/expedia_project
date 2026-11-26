"""Data quality, schema validation, and feature drift detection using Evidently AI & PSI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from configs.settings import PlatformConfig, get_config, get_logger
from monitoring.prometheus.metrics import DRIFT_ALERTS_TRIGGERED, FEATURE_DRIFT_PSI

logger = get_logger(__name__)


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and current numeric distributions."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_counts / max(1, len(ref)), 1e-4, 1.0)
    cur_pct = np.clip(cur_counts / max(1, len(cur)), 1e-4, 1.0)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_data_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model_name: str,
    numeric_cols: list[str],
    expected_schema: list[str] | None = None,
    config: PlatformConfig | None = None,
) -> dict[str, Any]:
    """Check schema compliance, missing values, PSI, KS tests, and run Evidently DataDriftPreset."""
    cfg = config or get_config()
    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Schema check
    missing_cols = []
    if expected_schema:
        missing_cols = [c for c in expected_schema if c not in current_df.columns]

    # 2. Missing value ratios
    null_rates = {
        col: round(float(current_df[col].isna().mean()), 6)
        for col in numeric_cols
        if col in current_df.columns
    }
    max_null_rate = max(null_rates.values()) if null_rates else 0.0

    # 3. Feature drift (PSI + two-sample KS test)
    feature_drift: dict[str, dict[str, Any]] = {}
    drifted_features: list[str] = []
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].to_numpy(dtype=float)
        cur_vals = current_df[col].to_numpy(dtype=float)
        psi = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_p = stats.ks_2samp(ref_vals[~np.isnan(ref_vals)], cur_vals[~np.isnan(cur_vals)])
        FEATURE_DRIFT_PSI.labels(model_name=model_name, feature=col).set(psi)

        status = "ok"
        if psi >= cfg.monitoring.psi_critical_threshold:
            status = "critical"
            drifted_features.append(col)
            DRIFT_ALERTS_TRIGGERED.labels(severity="critical", alert_type="feature_drift").inc()
        elif psi >= cfg.monitoring.psi_warning_threshold:
            status = "warning"
            drifted_features.append(col)
            DRIFT_ALERTS_TRIGGERED.labels(severity="warning", alert_type="feature_drift").inc()

        feature_drift[col] = {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 6),
            "status": status,
        }

    # 4. Evidently AI report generation (supports both Evidently 0.4.x and 0.7.x APIs)
    evidently_html_path = reports_dir / f"evidently_data_drift_{model_name}.html"
    evidently_status = "generated"
    try:
        cols_present = [
            c for c in numeric_cols if c in reference_df.columns and c in current_df.columns
        ]
        ref_sub = reference_df[cols_present].head(5000)
        cur_sub = current_df[cols_present].head(5000)
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            rep = Report([DataDriftPreset()])
            snapshot = rep.run(current_data=cur_sub, reference_data=ref_sub)
            snapshot.save_html(str(evidently_html_path))
        except Exception:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref_sub, current_data=cur_sub)
            rep.save_html(str(evidently_html_path))
    except Exception as exc:
        evidently_status = f"fallback ({exc})"

    return {
        "model_name": model_name,
        "schema_valid": len(missing_cols) == 0,
        "missing_columns": missing_cols,
        "max_null_rate": max_null_rate,
        "null_rates": null_rates,
        "drifted_features": drifted_features,
        "feature_drift": feature_drift,
        "evidently_report_status": evidently_status,
        "evidently_html_path": str(evidently_html_path),
    }
