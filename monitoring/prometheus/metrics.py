"""Prometheus metrics definitions for API latency, throughput, drift, and model performance."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_REQUESTS = Counter(
    "expedia_ad_prediction_requests_total",
    "Total number of prediction requests received",
    ["endpoint"],
)

PREDICTION_CACHE_HITS = Counter(
    "expedia_ad_prediction_cache_hits_total",
    "Total number of Redis/in-memory cache hits",
    ["endpoint"],
)

PREDICTION_ERRORS = Counter(
    "expedia_ad_prediction_errors_total",
    "Total number of prediction errors",
    ["endpoint"],
)

PREDICTION_LATENCY = Histogram(
    "expedia_ad_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

FEATURE_DRIFT_PSI = Gauge(
    "expedia_ad_feature_psi",
    "Population Stability Index for monitored ML features",
    ["model_name", "feature"],
)

MODEL_PERFORMANCE_GAUGE = Gauge(
    "expedia_ad_model_metric",
    "Latest validation/production model performance metric",
    ["model_name", "metric"],
)

DRIFT_ALERTS_TRIGGERED = Counter(
    "expedia_ad_drift_alerts_total",
    "Total number of data or model drift alerts triggered",
    ["severity", "alert_type"],
)
