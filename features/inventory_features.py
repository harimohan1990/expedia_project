"""Feature engineering for Model 2: Inventory Forecasting (7, 14, and 30-day horizons)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


def build_inventory_features(config: PlatformConfig | None = None) -> pd.DataFrame:
    """Build lagged time-series, seasonality, and advertiser-demand features for Inventory Forecasting."""
    cfg = config or get_config()
    processed_dir = cfg.resolve_path(cfg.paths.processed_data_dir)
    raw_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    if not (raw_dir / "ad_inventory.csv").exists():
        raw_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)

    inv_parquet = processed_dir / "parquet" / "ad_inventory"
    if inv_parquet.exists():
        inv = pd.read_parquet(inv_parquet)
    else:
        inv = pd.read_csv(raw_dir / "ad_inventory.csv")

    campaigns = pd.read_csv(raw_dir / "campaigns.csv")
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
    campaigns["daily_budget"] = campaigns["budget"] / (
        (campaigns["end_date"] - campaigns["start_date"]).dt.days + 1
    ).clip(lower=1)

    inv["date"] = pd.to_datetime(inv["date"])
    inv = inv.sort_values(["country", "placement", "date"]).reset_index(drop=True)

    # Served impressions proxy derived from historical impressions per country/date if available
    imp_parquet = processed_dir / "aggregations" / "daily_impressions"
    if imp_parquet.exists():
        daily_imp = pd.read_parquet(imp_parquet)
        daily_imp["date"] = pd.to_datetime(daily_imp["event_date"])
        inv = inv.merge(
            daily_imp[["date", "total_impressions"]].rename(
                columns={"total_impressions": "platform_impressions_served"}
            ),
            on="date",
            how="left",
        )
    else:
        inv["platform_impressions_served"] = np.nan

    inv["historical_impressions_served"] = inv["platform_impressions_served"].fillna(
        inv["booked_impressions"] * 0.92
    )

    # Compute advertiser demand signal per (country, date) from active campaigns
    # Vectorized expansion using daily campaign budget active on each date
    camp_daily = (
        campaigns.groupby(["target_country", "start_date"])["daily_budget"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "target_country": "country",
                "start_date": "date",
                "daily_budget": "new_campaign_demand",
            }
        )
    )
    inv = inv.merge(camp_daily, on=["country", "date"], how="left")
    inv["new_campaign_demand"] = inv["new_campaign_demand"].fillna(0.0)
    inv["advertiser_demand"] = inv.groupby(["country", "placement"])[
        "new_campaign_demand"
    ].transform(lambda s: s.rolling(14, min_periods=1).mean())

    # Calendar & cyclical seasonality features
    inv["day_of_week"] = inv["date"].dt.dayofweek
    inv["month"] = inv["date"].dt.month
    inv["day_of_year"] = inv["date"].dt.dayofyear
    inv["is_weekend"] = (inv["day_of_week"] >= 5).astype(int)
    inv["sin_doy"] = np.sin(2 * np.pi * inv["day_of_year"] / 365.25)
    inv["cos_doy"] = np.cos(2 * np.pi * inv["day_of_year"] / 365.25)
    inv["sin_dow"] = np.sin(2 * np.pi * inv["day_of_week"] / 7.0)
    inv["cos_dow"] = np.cos(2 * np.pi * inv["day_of_week"] / 7.0)

    # Strictly causal lagged inventory & booking trends (lag >= 1 day, shifted before rolling)
    grp = inv.groupby(["country", "placement"])
    for lag in [1, 7, 14, 28, 30]:
        inv[f"avail_lag_{lag}"] = grp["available_impressions"].shift(lag)
        inv[f"booked_lag_{lag}"] = grp["booked_impressions"].shift(lag)

    inv["avail_roll_mean_7"] = grp["available_impressions"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    inv["avail_roll_mean_28"] = grp["available_impressions"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    inv["avail_roll_std_14"] = (
        grp["available_impressions"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).std())
        .fillna(0.0)
    )
    inv["booked_roll_mean_7"] = grp["booked_impressions"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    inv["booked_roll_mean_28"] = grp["booked_impressions"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    inv["booking_trends"] = (
        inv["booked_roll_mean_7"] / inv["booked_roll_mean_28"].clip(lower=1.0)
    ).fillna(1.0)
    inv["served_lag_7"] = grp["historical_impressions_served"].shift(7)

    # Multi-horizon future targets for direct 7, 14, 30 day validation if desired
    inv["target_7d_ahead"] = grp["available_impressions"].shift(-7)
    inv["target_14d_ahead"] = grp["available_impressions"].shift(-14)
    inv["target_30d_ahead"] = grp["available_impressions"].shift(-30)

    # Drop initial warmup rows where 30-day lag is missing
    inv_clean = inv.dropna(subset=["avail_lag_30", "booked_lag_30"]).reset_index(drop=True)

    out_path = processed_dir / "features" / "inventory_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inv_clean.to_parquet(out_path, index=False)
    logger.info("Built inventory forecasting features (%d rows) -> %s", len(inv_clean), out_path)
    return inv_clean
