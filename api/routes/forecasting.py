"""API v1 endpoints for forecasting, campaign performance, targeting recommendations, and health."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.models import (
    AudienceReachRequest,
    AudienceReachResponse,
    CampaignPerformanceRequest,
    CampaignPerformanceResponse,
    HealthResponse,
    InventoryForecastRequest,
    InventoryForecastResponse,
    ModelMetadata,
    ModelsListResponse,
    SegmentRecommendation,
    TargetingRecommendationRequest,
    TargetingRecommendationResponse,
)
from configs.settings import get_config, get_logger
from models.audience_reach.predict import predict_audience_reach
from models.campaign_performance.predict import predict_campaign_performance
from models.inventory_forecasting.predict import predict_inventory
from monitoring.prometheus.metrics import (
    PREDICTION_CACHE_HITS,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)
from utils.storage import CacheClient

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Advertising Intelligence API"])
_cache = CacheClient()


def _cache_key(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _read_model_summary(experiment_name: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.resolve_path(cfg.paths.models_dir) / experiment_name / "evaluation_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    cfg = get_config()
    models_dir = cfg.resolve_path(cfg.paths.models_dir)
    loaded = {
        "audience_reach": (models_dir / cfg.ml.mlflow_experiment_reach / "best_model.pkl").exists(),
        "inventory_forecasting": (
            models_dir / cfg.ml.mlflow_experiment_inventory / "best_model.pkl"
        ).exists(),
        "campaign_performance": (
            models_dir / cfg.ml.mlflow_experiment_performance / "best_model.pkl"
        ).exists(),
    }
    status = "healthy" if all(loaded.values()) else "degraded"
    return HealthResponse(
        status=status,
        version=cfg.project.version,
        cache_backend=_cache.backend,
        models_loaded=loaded,
        details={"mlflow_tracking_uri": cfg.paths.mlflow_tracking_uri},
    )


@router.get("/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    cfg = get_config()
    experiments = [
        ("audience_reach_forecaster", cfg.ml.mlflow_experiment_reach),
        ("inventory_forecaster", cfg.ml.mlflow_experiment_inventory),
        ("campaign_performance_predictor", cfg.ml.mlflow_experiment_performance),
        ("campaign_high_perf_classifier", f"{cfg.ml.mlflow_experiment_performance}-cls"),
    ]
    items: list[ModelMetadata] = []
    for model_name, exp in experiments:
        summary = _read_model_summary(exp)
        if summary:
            items.append(
                ModelMetadata(
                    model_name=model_name,
                    experiment_name=exp,
                    best_algorithm=summary.get("best_model", "unknown"),
                    metrics=summary.get("best_metrics", {}),
                    version=str(summary.get("best_run_id", "local-v1"))[:8],
                )
            )
    return ModelsListResponse(models=items)


@router.post("/forecast/audience-reach", response_model=AudienceReachResponse)
def forecast_audience_reach(req: AudienceReachRequest) -> AudienceReachResponse:
    t0 = time.perf_counter()
    endpoint = "forecast/audience-reach"
    PREDICTION_REQUESTS.labels(endpoint=endpoint).inc()
    try:
        records = [item.model_dump() for item in req.items]
        key = _cache_key("reach", records)
        cached_val = _cache.get_json(key)
        if cached_val is not None:
            PREDICTION_CACHE_HITS.labels(endpoint=endpoint).inc()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
            return AudienceReachResponse(
                model_version=cached_val["version"],
                cached=True,
                latency_ms=latency_ms,
                predictions=cached_val["predictions"],
            )

        preds = predict_audience_reach(records)
        summary = _read_model_summary(get_config().ml.mlflow_experiment_reach)
        version = (
            f"{summary.get('best_model', 'v1')}-{str(summary.get('best_run_id', 'local'))[:8]}"
        )
        _cache.set_json(key, {"version": version, "predictions": preds})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        return AudienceReachResponse(
            model_version=version,
            cached=False,
            latency_ms=latency_ms,
            predictions=preds,
        )
    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        logger.exception("Audience reach prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/forecast/inventory", response_model=InventoryForecastResponse)
def forecast_inventory(req: InventoryForecastRequest) -> InventoryForecastResponse:
    t0 = time.perf_counter()
    endpoint = "forecast/inventory"
    PREDICTION_REQUESTS.labels(endpoint=endpoint).inc()
    try:
        items = [i.model_dump() for i in req.items]
        key = _cache_key("inventory", items)
        cached_val = _cache.get_json(key)
        if cached_val is not None:
            PREDICTION_CACHE_HITS.labels(endpoint=endpoint).inc()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
            return InventoryForecastResponse(
                model_version=cached_val["version"],
                cached=True,
                latency_ms=latency_ms,
                forecasts=cached_val["forecasts"],
            )

        all_forecasts: list[dict[str, Any]] = []
        for item in items:
            all_forecasts.extend(
                predict_inventory(
                    country=item["country"],
                    placement=item["placement"],
                    horizon_days=item["horizon_days"],
                )
            )
        summary = _read_model_summary(get_config().ml.mlflow_experiment_inventory)
        version = (
            f"{summary.get('best_model', 'v1')}-{str(summary.get('best_run_id', 'local'))[:8]}"
        )
        _cache.set_json(key, {"version": version, "forecasts": all_forecasts})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        return InventoryForecastResponse(
            model_version=version,
            cached=False,
            latency_ms=latency_ms,
            forecasts=all_forecasts,
        )
    except ValueError as exc:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        logger.exception("Inventory forecast failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict/campaign-performance", response_model=CampaignPerformanceResponse)
def campaign_performance(req: CampaignPerformanceRequest) -> CampaignPerformanceResponse:
    t0 = time.perf_counter()
    endpoint = "predict/campaign-performance"
    PREDICTION_REQUESTS.labels(endpoint=endpoint).inc()
    try:
        records = [item.model_dump() for item in req.items]
        key = _cache_key("campaign", records)
        cached_val = _cache.get_json(key)
        if cached_val is not None:
            PREDICTION_CACHE_HITS.labels(endpoint=endpoint).inc()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
            return CampaignPerformanceResponse(
                model_version=cached_val["version"],
                cached=True,
                latency_ms=latency_ms,
                predictions=cached_val["predictions"],
            )

        preds = predict_campaign_performance(records)
        summary = _read_model_summary(get_config().ml.mlflow_experiment_performance)
        version = (
            f"{summary.get('best_model', 'v1')}-{str(summary.get('best_run_id', 'local'))[:8]}"
        )
        _cache.set_json(key, {"version": version, "predictions": preds})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        return CampaignPerformanceResponse(
            model_version=version,
            cached=False,
            latency_ms=latency_ms,
            predictions=preds,
        )
    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        logger.exception("Campaign performance prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/targeting/recommendations", response_model=TargetingRecommendationResponse)
def targeting_recommendations(
    req: TargetingRecommendationRequest,
) -> TargetingRecommendationResponse:
    t0 = time.perf_counter()
    endpoint = "targeting/recommendations"
    PREDICTION_REQUESTS.labels(endpoint=endpoint).inc()
    try:
        cfg = get_config()
        seg_path = cfg.resolve_path(cfg.paths.reports_dir) / "uplift_segments.csv"
        if not seg_path.exists():
            raise HTTPException(
                status_code=404, detail="Causal uplift segments report not generated yet."
            )

        seg_df = pd.read_csv(seg_path)
        positive = seg_df[seg_df["mean_predicted_uplift"] > 0].head(req.top_k).copy()
        if positive.empty:
            positive = seg_df.head(req.top_k).copy()

        weights = positive["mean_predicted_uplift"].clip(lower=0.001)
        weights = weights / weights.sum()

        recs: list[SegmentRecommendation] = []
        for (_, row), w in zip(positive.iterrows(), weights, strict=False):
            recs.append(
                SegmentRecommendation(
                    loyalty_segment=str(row["loyalty_segment"]),
                    device_type=str(row["device_type"]),
                    recommended_allocation_pct=round(float(w * 100.0), 2),
                    predicted_uplift_cate=round(float(row["mean_predicted_uplift"]), 6),
                    expected_cvr_lift=round(float(row["observed_segment_lift"]), 6),
                    rationale=(
                        f"High causal uplift segment (intent={row['avg_travel_intent']:.2f}); "
                        f"allocate ${req.budget * w:,.0f} under ML targeting."
                    ),
                )
            )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        return TargetingRecommendationResponse(
            country=req.country.upper(),
            campaign_objective=req.campaign_objective,
            total_budget=req.budget,
            latency_ms=latency_ms,
            recommendations=recs,
        )
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc()
        logger.exception("Targeting recommendation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
