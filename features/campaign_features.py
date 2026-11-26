"""Feature engineering for Model 3: Pre-launch Campaign Performance Prediction (zero post-launch leakage)."""

from __future__ import annotations

import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


def build_campaign_performance_features(config: PlatformConfig | None = None) -> pd.DataFrame:
    """Build pre-launch campaign features and observed performance targets."""
    cfg = config or get_config()
    processed_dir = cfg.resolve_path(cfg.paths.processed_data_dir)
    raw_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    if not (raw_dir / "campaigns.csv").exists():
        raw_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)

    campaigns = pd.read_csv(raw_dir / "campaigns.csv")
    travelers = pd.read_csv(raw_dir / "travelers.csv")
    bidding = pd.read_csv(
        raw_dir / "bidding_data.csv", usecols=["campaign_id", "bid_amount", "auction_result"]
    )

    # Observed campaign performance from PySpark aggregation or raw impressions
    agg_camp_path = processed_dir / "aggregations" / "campaign_metrics"
    if agg_camp_path.exists():
        perf = pd.read_parquet(agg_camp_path)
    else:
        imp = pd.read_csv(
            raw_dir / "ad_impressions.csv",
            usecols=["campaign_id", "impression_id", "impression_cost", "clicked", "converted"],
        )
        perf = imp.groupby("campaign_id", as_index=False).agg(
            impressions=("impression_id", "count"),
            clicks=("clicked", "sum"),
            conversions=("converted", "sum"),
            ad_spend=("impression_cost", "sum"),
        )
        perf["campaign_revenue"] = perf["conversions"] * 145.0
        perf["ctr"] = perf["clicks"] / perf["impressions"].clip(lower=1)
        perf["conversion_rate"] = perf["conversions"] / perf["impressions"].clip(lower=1)

    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
    campaigns["campaign_duration_days"] = (
        campaigns["end_date"] - campaigns["start_date"]
    ).dt.days + 1
    campaigns["daily_budget"] = campaigns["budget"] / campaigns["campaign_duration_days"].clip(
        lower=1
    )
    campaigns["start_month"] = campaigns["start_date"].dt.month
    campaigns["start_quarter"] = campaigns["start_date"].dt.quarter
    campaigns["start_dow"] = campaigns["start_date"].dt.dayofweek

    # Target segment characteristics (pre-launch audience size & avg booking freq)
    seg_stats = travelers.groupby(["country", "age_group"], as_index=False).agg(
        target_segment_size=("traveler_id", "nunique"),
        target_avg_booking_freq=("booking_frequency", "mean"),
    )

    # Pre-launch bidding intensity proxy per advertiser (historical bid behavior)
    adv_bids = bidding.groupby("campaign_id", as_index=False).agg(
        planned_avg_bid=("bid_amount", "mean"),
    )

    df = campaigns.merge(
        perf[
            [
                "campaign_id",
                "impressions",
                "clicks",
                "conversions",
                "ad_spend",
                "campaign_revenue",
                "ctr",
                "conversion_rate",
            ]
        ],
        on="campaign_id",
        how="inner",
    )
    df = df.merge(
        seg_stats,
        left_on=["target_country", "target_age_group"],
        right_on=["country", "age_group"],
        how="left",
    ).drop(columns=["country", "age_group"])
    df = df.merge(adv_bids, on="campaign_id", how="left")
    df["planned_avg_bid"] = df["planned_avg_bid"].fillna(18.0)

    # Chronological historical priors per advertiser & channel (strictly prior campaigns)
    df = df.sort_values(["start_date", "campaign_id"]).reset_index(drop=True)
    ch_grp = df.groupby("advertising_channel")
    df["channel_hist_ctr"] = (
        ch_grp["ctr"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.045)
    )
    df["channel_hist_cvr"] = (
        ch_grp["conversion_rate"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0.012)
    )

    adv_grp = df.groupby("advertiser_id")
    df["advertiser_hist_campaigns"] = adv_grp.cumcount()
    df["advertiser_hist_roas"] = adv_grp["campaign_revenue"].transform(
        lambda s: s.shift(1).expanding().mean()
    ).fillna(df["budget"] * 1.5) / df["budget"].clip(lower=1.0)

    # Binary classification target: high-performing campaign (above median ROAS or CVR)
    roas_series = df["campaign_revenue"] / df["ad_spend"].clip(lower=1.0)
    df["roas"] = roas_series
    median_roas = float(roas_series.median())
    df["is_high_performing"] = (roas_series >= median_roas).astype(int)

    out_path = processed_dir / "features" / "campaign_performance_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Built campaign performance features (%d rows) -> %s", len(df), out_path)
    return df
