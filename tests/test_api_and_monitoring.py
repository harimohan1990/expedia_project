"""Tests for FastAPI endpoints, Pydantic validation, caching, and drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from monitoring.data_drift import compute_psi, run_data_drift_check
from monitoring.model_drift import evaluate_model_drift


def test_compute_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(50, 10, size=1000)
    assert compute_psi(a, a) < 0.01

    shifted = rng.normal(80, 10, size=1000)
    assert compute_psi(a, shifted) > 0.25


def test_monitoring_data_and_model_drift(tmp_path) -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {"budget": rng.normal(10000, 2000, size=300), "duration": rng.integers(7, 60, size=300)}
    )
    cur = pd.DataFrame(
        {"budget": rng.normal(25000, 2000, size=300), "duration": rng.integers(7, 60, size=300)}
    )
    res = run_data_drift_check(
        ref, cur, "test_model", ["budget", "duration"], ["budget", "duration"]
    )
    assert res["schema_valid"] is True
    assert "budget" in res["drifted_features"]

    drift_res = evaluate_model_drift(
        model_name="test_model",
        y_ref_pred=np.linspace(10, 100, 100),
        y_cur_true=np.linspace(10, 100, 100),
        y_cur_pred=np.linspace(10, 100, 100) + 50.0,
        baseline_rmse=5.0,
    )
    assert len(drift_res["alerts_triggered"]) >= 1


def test_fastapi_health_and_metrics_endpoints() -> None:
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"healthy", "degraded"}
    assert "models_loaded" in body

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "expedia_ad_prediction_requests_total" in metrics_resp.text
