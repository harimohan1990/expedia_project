"""Feature engineering for Model 1: Audience Reach Forecasting (leak-free, time-aware)."""

from __future__ import annotations

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)

OBJECTIVE_FREQ_CAP = {
    "awareness": 6,
    "traffic": 8,
    "bookings": 10,
    "retargeting": 12,
}

CHANNEL_PRIMARY_PLACEMENT = {
    "search_ads": "search_results_top",
    "display_banner": "homepage_hero",
    "sponsored_listing": "hotel_detail_sidebar",
    "mobile_native": "mobile_app_feed",
    "video_pre_roll": "homepage_hero",
    "email_newsletter": "flight_confirmation",
}


def build_audience_reach_features(config: PlatformConfig | None = None) -> pd.DataFrame:
    """Build time-aware campaign-level feature table for Audience Reach Forecasting."""
    cfg = config or get_config()
    processed_dir = cfg.resolve_path(cfg.paths.processed_data_dir)
    raw_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    if not (raw_dir / "campaigns.csv").exists():
        raw_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)

    campaigns = pd.read_csv(raw_dir / "campaigns.csv")
    travelers = pd.read_csv(raw_dir / "travelers.csv")
    signals = pd.read_csv(raw_dir / "targeting_signals.csv")

    # Prefer PySpark-processed impressions if available, else raw
    imp_parquet = processed_dir / "parquet" / "ad_impressions_enriched"
    if imp_parquet.exists():
        impressions = pd.read_parquet(
            imp_parquet,
            columns=[
                "campaign_id",
                "traveler_id",
                "placement",
                "device_type",
                "timestamp",
                "engagement_score",
            ],
        )
    else:
        impressions = pd.read_csv(
            raw_dir / "ad_impressions.csv",
            usecols=["campaign_id", "traveler_id", "placement", "device_type", "timestamp"],
        )
        trav_sig = signals.groupby("traveler_id", as_index=False)["engagement_score"].mean()
        impressions = impressions.merge(trav_sig, on="traveler_id", how="left")
        impressions["engagement_score"] = impressions["engagement_score"].fillna(42.0)

    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
    campaigns["campaign_duration"] = (campaigns["end_date"] - campaigns["start_date"]).dt.days + 1

    # Target audience pool sizes from travelers dimension
    country_sizes = (
        travelers.groupby("country")["traveler_id"].nunique().rename("country_audience_size")
    )
    seg_sizes = (
        travelers.groupby(["country", "age_group"])["traveler_id"]
        .nunique()
        .rename("target_audience_size")
    )
    dom_device = (
        travelers.groupby(["country", "age_group"])["device_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "mobile")
        .rename("device_type")
    )

    # Observed campaign outcomes (strictly targets, plus dominant placement)
    camp_obs = impressions.groupby("campaign_id", as_index=False).agg(
        actual_unique_reach=("traveler_id", "nunique"),
        actual_impressions=("traveler_id", "count"),
        actual_avg_engagement=("engagement_score", "mean"),
        observed_placement=(
            "placement",
            lambda s: s.mode().iloc[0] if not s.mode().empty else "homepage_hero",
        ),
    )

    df = campaigns.merge(camp_obs, on="campaign_id", how="inner")
    df = df.merge(country_sizes, left_on="target_country", right_index=True, how="left")
    df = df.merge(
        seg_sizes, left_on=["target_country", "target_age_group"], right_index=True, how="left"
    )
    df = df.merge(
        dom_device, left_on=["target_country", "target_age_group"], right_index=True, how="left"
    )

    df["country"] = df["target_country"]
    df["advertising_placement"] = (
        df["advertising_channel"].map(CHANNEL_PRIMARY_PLACEMENT).fillna(df["observed_placement"])
    )
    df["frequency_cap"] = df["campaign_objective"].map(OBJECTIVE_FREQ_CAP).fillna(8)
    df["campaign_budget"] = df["budget"]

    # Sort chronologically and compute strictly causal expanding historical priors per (country, channel)
    df = df.sort_values(["start_date", "campaign_id"]).reset_index(drop=True)
    grp = df.groupby(["country", "advertising_channel"])
    df["historical_unique_reach"] = grp["actual_unique_reach"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["historical_impressions"] = grp["actual_impressions"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["historical_engagement"] = grp["actual_avg_engagement"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Cold-start fallbacks computed from global prior (only initial rows)
    df["historical_unique_reach"] = df["historical_unique_reach"].fillna(
        df["target_audience_size"] * 0.025
    )
    df["historical_impressions"] = df["historical_impressions"].fillna(
        df["historical_unique_reach"] * 2.2
    )
    df["historical_engagement"] = df["historical_engagement"].fillna(45.0)

    # Engineered saturation & budget intensity features
    df["budget_per_day"] = df["campaign_budget"] / df["campaign_duration"].clip(lower=1)
    df["budget_to_audience_ratio"] = df["campaign_budget"] / df["target_audience_size"].clip(
        lower=1
    )
    df["start_month"] = df["start_date"].dt.month
    df["start_dow"] = df["start_date"].dt.dayofweek
    df["forecast_date"] = df["end_date"].dt.strftime("%Y-%m-%d")

    feature_cols = [
        "campaign_id",
        "start_date",
        "forecast_date",
        "country",
        "advertising_placement",
        "advertising_channel",
        "campaign_objective",
        "device_type",
        "campaign_budget",
        "campaign_duration",
        "target_audience_size",
        "country_audience_size",
        "frequency_cap",
        "historical_unique_reach",
        "historical_impressions",
        "historical_engagement",
        "budget_per_day",
        "budget_to_audience_ratio",
        "start_month",
        "start_dow",
        "actual_unique_reach",
    ]
    out_df = df[feature_cols].copy()

    out_path = processed_dir / "features" / "audience_reach_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Built audience reach features (%d rows) -> %s", len(out_df), out_path)
    return out_df
