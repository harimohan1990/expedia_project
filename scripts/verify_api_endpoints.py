"""Verify all FastAPI endpoints end-to-end against trained models using FastAPI TestClient."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from api.main import app


def verify_all_endpoints() -> dict[str, dict]:
    client = TestClient(app)
    results: dict[str, dict] = {}

    # 1. Health
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200, r_health.text
    results["health"] = r_health.json()

    # 2. Models list
    r_models = client.get("/api/v1/models")
    assert r_models.status_code == 200, r_models.text
    results["models"] = r_models.json()

    # 3. Audience Reach Forecast (batch of 2)
    r_reach = client.post(
        "/api/v1/forecast/audience-reach",
        json={
            "items": [
                {
                    "campaign_id": "TEST_REACH_1",
                    "forecast_date": "2025-12-01",
                    "country": "US",
                    "advertising_placement": "search_results_top",
                    "device_type": "mobile",
                    "campaign_budget": 30000.0,
                    "campaign_duration": 30,
                    "target_audience_size": 10000,
                    "frequency_cap": 8,
                    "historical_unique_reach": 2100.0,
                    "historical_impressions": 5000.0,
                    "historical_engagement": 54.0,
                },
                {
                    "campaign_id": "TEST_REACH_2",
                    "forecast_date": "2025-12-15",
                    "country": "UK",
                    "advertising_placement": "homepage_hero",
                    "device_type": "desktop",
                    "campaign_budget": 12000.0,
                    "campaign_duration": 14,
                    "target_audience_size": 4500,
                },
            ]
        },
    )
    assert r_reach.status_code == 200, r_reach.text
    results["audience_reach"] = r_reach.json()

    # 4. Inventory Forecast (14 days)
    r_inv = client.post(
        "/api/v1/forecast/inventory",
        json={"items": [{"country": "US", "placement": "homepage_hero", "horizon_days": 14}]},
    )
    assert r_inv.status_code == 200, r_inv.text
    results["inventory"] = {
        "model_version": r_inv.json()["model_version"],
        "forecast_count": len(r_inv.json()["forecasts"]),
        "sample_first_2": r_inv.json()["forecasts"][:2],
    }

    # 5. Campaign Performance
    r_perf = client.post(
        "/api/v1/predict/campaign-performance",
        json={
            "items": [
                {
                    "campaign_id": "TEST_CMP_1",
                    "budget": 45000.0,
                    "campaign_duration_days": 30,
                    "target_country": "US",
                    "target_age_group": "25-34",
                    "campaign_objective": "bookings",
                    "advertising_channel": "search_ads",
                    "planned_avg_bid": 22.0,
                }
            ]
        },
    )
    assert r_perf.status_code == 200, r_perf.text
    results["campaign_performance"] = r_perf.json()

    # 6. Targeting Recommendations
    r_targ = client.post(
        "/api/v1/targeting/recommendations",
        json={"country": "US", "campaign_objective": "bookings", "budget": 50000.0, "top_k": 5},
    )
    assert r_targ.status_code == 200, r_targ.text
    results["targeting_recommendations"] = r_targ.json()

    # 7. Prometheus Metrics
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "expedia_ad_prediction_requests_total" in r_metrics.text

    return results


if __name__ == "__main__":
    print(json.dumps(verify_all_endpoints(), indent=2))
