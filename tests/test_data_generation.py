"""Tests for synthetic data schemas, row counts, and referential relationships."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_generated_files_exist_and_have_expected_schemas(
    synthetic_dataset_paths: dict[str, Path],
) -> None:
    expected_cols = {
        "travelers": {
            "traveler_id",
            "age_group",
            "country",
            "city",
            "device_type",
            "travel_interest",
            "booking_frequency",
            "loyalty_segment",
        },
        "campaigns": {
            "campaign_id",
            "advertiser_id",
            "campaign_name",
            "start_date",
            "end_date",
            "budget",
            "target_country",
            "target_age_group",
            "campaign_objective",
            "advertising_channel",
        },
        "ad_impressions": {
            "impression_id",
            "campaign_id",
            "traveler_id",
            "timestamp",
            "placement",
            "device_type",
            "impression_cost",
            "clicked",
            "converted",
        },
        "ad_inventory": {
            "inventory_id",
            "placement",
            "country",
            "date",
            "available_impressions",
            "booked_impressions",
            "remaining_impressions",
        },
        "bidding_data": {
            "bid_id",
            "campaign_id",
            "placement",
            "bid_amount",
            "auction_timestamp",
            "advertiser_id",
            "auction_result",
        },
        "targeting_signals": {
            "traveler_id",
            "destination_interest",
            "travel_intent",
            "search_frequency",
            "booking_probability",
            "engagement_score",
            "signal_timestamp",
        },
    }
    for name, cols in expected_cols.items():
        path = synthetic_dataset_paths[name]
        assert path.exists(), f"Missing {path}"
        df = pd.read_csv(path, nrows=50)
        assert cols.issubset(set(df.columns)), f"Schema mismatch for {name}: {df.columns}"


def test_referential_integrity_and_value_bounds(
    synthetic_dataset_paths: dict[str, Path],
) -> None:
    travelers = pd.read_csv(synthetic_dataset_paths["travelers"])
    campaigns = pd.read_csv(synthetic_dataset_paths["campaigns"])
    impressions = pd.read_csv(synthetic_dataset_paths["ad_impressions"])
    inventory = pd.read_csv(synthetic_dataset_paths["ad_inventory"])

    assert set(impressions["traveler_id"]).issubset(set(travelers["traveler_id"]))
    assert set(impressions["campaign_id"]).issubset(set(campaigns["campaign_id"]))
    assert (inventory["available_impressions"] >= inventory["booked_impressions"]).all()
    assert (
        inventory["remaining_impressions"]
        == inventory["available_impressions"] - inventory["booked_impressions"]
    ).all()
    assert impressions["clicked"].isin([0, 1]).all()
    assert impressions["converted"].isin([0, 1]).all()
