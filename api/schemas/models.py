"""Pydantic v2 request and response schemas for all FastAPI endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AudienceReachItem(BaseModel):
    campaign_id: str = Field(..., json_schema_extra={"example": "CMP_000101"})
    forecast_date: str = Field(..., json_schema_extra={"example": "2025-11-15"})
    country: str = Field(..., json_schema_extra={"example": "US"})
    advertising_placement: str = Field(..., json_schema_extra={"example": "search_results_top"})
    device_type: str = Field(..., json_schema_extra={"example": "mobile"})
    campaign_budget: float = Field(..., gt=0, json_schema_extra={"example": 25000.0})
    campaign_duration: int = Field(..., ge=1, le=365, json_schema_extra={"example": 30})
    target_audience_size: int = Field(..., ge=1, json_schema_extra={"example": 8500})
    frequency_cap: int = Field(8, ge=1, le=50, json_schema_extra={"example": 8})
    historical_unique_reach: float = Field(1800.0, ge=0)
    historical_impressions: float = Field(4500.0, ge=0)
    historical_engagement: float = Field(52.0, ge=0, le=100)


class AudienceReachRequest(BaseModel):
    items: list[AudienceReachItem] = Field(..., min_length=1, max_length=500)


class AudienceReachPrediction(BaseModel):
    campaign_id: str
    forecast_date: str
    predicted_unique_reach: int
    ci_lower: int
    ci_upper: int
    confidence_interval: str


class AudienceReachResponse(BaseModel):
    model_version: str
    cached: bool
    latency_ms: float
    predictions: list[AudienceReachPrediction]


class InventoryForecastItem(BaseModel):
    country: str = Field(..., json_schema_extra={"example": "US"})
    placement: str = Field(..., json_schema_extra={"example": "homepage_hero"})
    horizon_days: int = Field(14, ge=1, le=90, json_schema_extra={"example": 14})


class InventoryForecastRequest(BaseModel):
    items: list[InventoryForecastItem] = Field(..., min_length=1, max_length=100)


class InventoryForecastPoint(BaseModel):
    date: str
    placement: str
    country: str
    predicted_available_impressions: int


class InventoryForecastResponse(BaseModel):
    model_version: str
    cached: bool
    latency_ms: float
    forecasts: list[InventoryForecastPoint]


class CampaignPerformanceItem(BaseModel):
    campaign_id: str = Field(default="NEW_CMP_001", json_schema_extra={"example": "NEW_CMP_001"})
    budget: float = Field(..., gt=0, json_schema_extra={"example": 35000.0})
    campaign_duration_days: int = Field(..., ge=1, le=365, json_schema_extra={"example": 30})
    target_country: str = Field(..., json_schema_extra={"example": "US"})
    target_age_group: str = Field(..., json_schema_extra={"example": "25-34"})
    campaign_objective: str = Field(..., json_schema_extra={"example": "bookings"})
    advertising_channel: str = Field(..., json_schema_extra={"example": "search_ads"})
    planned_avg_bid: float = Field(20.0, gt=0)


class CampaignPerformanceRequest(BaseModel):
    items: list[CampaignPerformanceItem] = Field(..., min_length=1, max_length=500)


class CampaignPerformancePrediction(BaseModel):
    campaign_id: str
    expected_impressions: int
    expected_clicks: int
    expected_ctr: float
    expected_conversions: int
    expected_conversion_rate: float
    estimated_campaign_revenue: float
    high_performance_probability: float


class CampaignPerformanceResponse(BaseModel):
    model_version: str
    cached: bool
    latency_ms: float
    predictions: list[CampaignPerformancePrediction]


class TargetingRecommendationRequest(BaseModel):
    country: str = Field(..., json_schema_extra={"example": "US"})
    campaign_objective: str = Field("bookings", json_schema_extra={"example": "bookings"})
    budget: float = Field(..., gt=0, json_schema_extra={"example": 50000.0})
    top_k: int = Field(5, ge=1, le=20)


class SegmentRecommendation(BaseModel):
    loyalty_segment: str
    device_type: str
    recommended_allocation_pct: float
    predicted_uplift_cate: float
    expected_cvr_lift: float
    rationale: str


class TargetingRecommendationResponse(BaseModel):
    country: str
    campaign_objective: str
    total_budget: float
    latency_ms: float
    recommendations: list[SegmentRecommendation]


class ModelMetadata(BaseModel):
    model_name: str
    experiment_name: str
    best_algorithm: str
    metrics: dict[str, float]
    version: str


class ModelsListResponse(BaseModel):
    models: list[ModelMetadata]


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_backend: str
    models_loaded: dict[str, bool]
    details: dict[str, Any]
