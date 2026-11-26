"""Synthetic Travel Advertising Dataset Generator.

Generates six relational, realistic datasets with reproducible seeds:
- travelers.csv (100,000+ rows)
- campaigns.csv (10,000+ rows)
- ad_impressions.csv (1,000,000+ rows)
- ad_inventory.csv (100,000+ rows)
- bidding_data.csv (500,000+ rows)
- targeting_signals.csv (200,000+ rows)
Plus an experimentation log (`experiment_events.csv`) for causal inference evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)

COUNTRY_WEIGHTS = {
    "US": 0.30,
    "UK": 0.12,
    "CA": 0.08,
    "DE": 0.08,
    "FR": 0.07,
    "JP": 0.06,
    "AU": 0.06,
    "IN": 0.07,
    "BR": 0.05,
    "MX": 0.04,
    "ES": 0.04,
    "IT": 0.03,
}

COUNTRY_CITIES = {
    "US": ["New York", "Los Angeles", "Chicago", "Miami", "Seattle", "Austin"],
    "UK": ["London", "Manchester", "Edinburgh", "Birmingham"],
    "CA": ["Toronto", "Vancouver", "Montreal", "Calgary"],
    "DE": ["Berlin", "Munich", "Frankfurt", "Hamburg"],
    "FR": ["Paris", "Lyon", "Nice", "Marseille"],
    "JP": ["Tokyo", "Osaka", "Kyoto", "Sapporo"],
    "AU": ["Sydney", "Melbourne", "Brisbane", "Perth"],
    "IN": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad"],
    "BR": ["Sao Paulo", "Rio de Janeiro", "Brasilia"],
    "MX": ["Mexico City", "Cancun", "Guadalajara"],
    "ES": ["Madrid", "Barcelona", "Seville"],
    "IT": ["Rome", "Milan", "Florence", "Venice"],
}

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_WEIGHTS = [0.14, 0.28, 0.25, 0.17, 0.10, 0.06]

DEVICES = ["mobile", "desktop", "tablet", "app"]
INTERESTS = ["beach", "luxury", "adventure", "business", "family", "city_break", "eco_travel"]
LOYALTY_SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]
LOYALTY_WEIGHTS = [0.45, 0.30, 0.17, 0.08]

OBJECTIVES = ["awareness", "traffic", "bookings", "retargeting"]
CHANNELS = [
    "search_ads",
    "display_banner",
    "sponsored_listing",
    "mobile_native",
    "video_pre_roll",
    "email_newsletter",
]
PLACEMENTS = [
    "homepage_hero",
    "search_results_top",
    "hotel_detail_sidebar",
    "checkout_upsell",
    "mobile_app_feed",
    "flight_confirmation",
]

PLACEMENT_CPM_BASE = {
    "homepage_hero": 18.0,
    "search_results_top": 24.0,
    "hotel_detail_sidebar": 14.0,
    "checkout_upsell": 28.0,
    "mobile_app_feed": 12.0,
    "flight_confirmation": 20.0,
}

PLACEMENT_CTR_EFFECT = {
    "homepage_hero": 0.15,
    "search_results_top": 0.75,
    "hotel_detail_sidebar": 0.25,
    "checkout_upsell": 0.90,
    "mobile_app_feed": 0.10,
    "flight_confirmation": 0.45,
}

PLACEMENT_CAPACITY_MULT = {
    "homepage_hero": 1.4,
    "search_results_top": 1.8,
    "hotel_detail_sidebar": 1.2,
    "checkout_upsell": 0.6,
    "mobile_app_feed": 1.5,
    "flight_confirmation": 0.8,
}

COUNTRY_MARKET_MULT = {
    "US": 3.0,
    "UK": 1.4,
    "CA": 1.0,
    "DE": 1.1,
    "FR": 1.0,
    "JP": 1.1,
    "AU": 0.9,
    "IN": 1.2,
    "BR": 0.8,
    "MX": 0.7,
    "ES": 0.75,
    "IT": 0.7,
}


def _seasonal_factor(day_of_year: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Annual travel demand wave (summer + December peaks) + weekend effect."""
    annual = 1.0 + 0.22 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)
    holiday = 0.18 * np.exp(-0.5 * ((day_of_year - 355) / 14.0) ** 2)
    weekly = np.where(dow >= 5, 1.12, 0.96)
    return annual + holiday * 1.0 + (weekly - 1.0)


def generate_travelers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    countries = list(COUNTRY_WEIGHTS.keys())
    c_probs = np.array(list(COUNTRY_WEIGHTS.values()))
    country_arr = rng.choice(countries, size=n, p=c_probs)

    # Vectorized city assignment per country
    city_arr = np.empty(n, dtype=object)
    for c, cities in COUNTRY_CITIES.items():
        mask = country_arr == c
        cnt = int(mask.sum())
        if cnt > 0:
            city_arr[mask] = rng.choice(cities, size=cnt)

    age_arr = rng.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS)
    loyalty_arr = rng.choice(LOYALTY_SEGMENTS, size=n, p=LOYALTY_WEIGHTS)

    # Device correlates with age group
    device_arr = np.empty(n, dtype=object)
    young_mask = np.isin(age_arr, ["18-24", "25-34"])
    mid_mask = np.isin(age_arr, ["35-44", "45-54"])
    old_mask = ~(young_mask | mid_mask)
    device_arr[young_mask] = rng.choice(
        DEVICES, size=int(young_mask.sum()), p=[0.42, 0.20, 0.06, 0.32]
    )
    device_arr[mid_mask] = rng.choice(DEVICES, size=int(mid_mask.sum()), p=[0.32, 0.36, 0.12, 0.20])
    device_arr[old_mask] = rng.choice(DEVICES, size=int(old_mask.sum()), p=[0.22, 0.48, 0.20, 0.10])

    interest_arr = rng.choice(INTERESTS, size=n, p=[0.22, 0.12, 0.14, 0.16, 0.15, 0.14, 0.07])

    loyalty_lambda = np.where(
        loyalty_arr == "Platinum",
        9.5,
        np.where(loyalty_arr == "Gold", 5.5, np.where(loyalty_arr == "Silver", 2.8, 1.1)),
    )
    booking_freq = np.clip(rng.poisson(loyalty_lambda) + 1, 1, 30)

    return pd.DataFrame(
        {
            "traveler_id": [f"TRV_{i:07d}" for i in range(1, n + 1)],
            "age_group": age_arr,
            "country": country_arr,
            "city": city_arr,
            "device_type": device_arr,
            "travel_interest": interest_arr,
            "booking_frequency": booking_freq,
            "loyalty_segment": loyalty_arr,
        }
    )


def generate_campaigns(
    rng: np.random.Generator, n: int, start_str: str, end_str: str
) -> pd.DataFrame:
    countries = list(COUNTRY_WEIGHTS.keys())
    c_probs = np.array(list(COUNTRY_WEIGHTS.values()))
    advertiser_ids = [f"ADV_{i:04d}" for i in rng.integers(1, 501, size=n)]
    target_country = rng.choice(countries, size=n, p=c_probs)
    target_age = rng.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS)
    objective = rng.choice(OBJECTIVES, size=n, p=[0.25, 0.25, 0.35, 0.15])
    channel = rng.choice(CHANNELS, size=n, p=[0.28, 0.20, 0.22, 0.14, 0.10, 0.06])

    base_start = pd.Timestamp(start_str)
    base_end = pd.Timestamp(end_str)
    total_days = (base_end - base_start).days - 30
    offsets = rng.integers(0, max(total_days, 1), size=n)
    durations = rng.integers(7, 75, size=n)

    start_dates = pd.to_datetime(base_start) + pd.to_timedelta(offsets, unit="D")
    end_dates = start_dates + pd.to_timedelta(durations, unit="D")

    # Log-normal budget ($1,200 - $200,000) scaled by objective
    obj_mult = np.where(objective == "awareness", 1.35, np.where(objective == "bookings", 1.2, 0.9))
    budgets = np.round(
        np.clip(rng.lognormal(mean=9.4, sigma=0.75, size=n) * obj_mult, 1200, 220000), 2
    )

    ad_themes = [
        "Summer Escape",
        "City Deal",
        "Resort Promo",
        "Flight Saver",
        "Loyalty Boost",
        "Weekend Stay",
    ]
    themes = rng.choice(ad_themes, size=n)
    names = [
        f"{tc}_{th}_{i:05d}"
        for i, (tc, th) in enumerate(zip(target_country, themes, strict=False), start=1)
    ]

    return pd.DataFrame(
        {
            "campaign_id": [f"CMP_{i:06d}" for i in range(1, n + 1)],
            "advertiser_id": advertiser_ids,
            "campaign_name": names,
            "start_date": start_dates.strftime("%Y-%m-%d"),
            "end_date": end_dates.strftime("%Y-%m-%d"),
            "budget": budgets,
            "target_country": target_country,
            "target_age_group": target_age,
            "campaign_objective": objective,
            "advertising_channel": channel,
        }
    )


def generate_ad_inventory(rng: np.random.Generator, start_str: str, num_days: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_str, periods=num_days, freq="D")
    countries = list(COUNTRY_WEIGHTS.keys())

    rows = []
    inv_idx = 1
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    trend = 1.0 + 0.00028 * np.arange(num_days)
    season = _seasonal_factor(doy, dow)

    for country in countries:
        c_mult = COUNTRY_MARKET_MULT[country]
        for placement in PLACEMENTS:
            p_mult = PLACEMENT_CAPACITY_MULT[placement]
            base_avail = 45_000 * c_mult * p_mult
            noise = rng.normal(1.0, 0.045, size=num_days)
            avail = np.maximum(5_000, np.round(base_avail * trend * season * noise)).astype(int)

            # Demand/utilization rises during peak seasons and high-intent placements
            util_base = 0.48 + (
                0.12 if placement in ("search_results_top", "checkout_upsell") else 0.0
            )
            util_rate = np.clip(
                util_base + 0.14 * (season - 1.0) + rng.normal(0.0, 0.04, size=num_days),
                0.15,
                0.94,
            )
            booked = np.minimum(avail - 100, np.round(avail * util_rate)).astype(int)
            remaining = avail - booked

            df_slice = pd.DataFrame(
                {
                    "inventory_id": [f"INV_{i:07d}" for i in range(inv_idx, inv_idx + num_days)],
                    "placement": placement,
                    "country": country,
                    "date": dates.strftime("%Y-%m-%d"),
                    "available_impressions": avail,
                    "booked_impressions": booked,
                    "remaining_impressions": remaining,
                }
            )
            inv_idx += num_days
            rows.append(df_slice)

    return pd.concat(rows, ignore_index=True)


def generate_targeting_signals(
    rng: np.random.Generator, travelers_df: pd.DataFrame, n: int, end_str: str
) -> pd.DataFrame:
    n_trav = len(travelers_df)
    trav_indices = rng.integers(0, n_trav, size=n)
    trav_ids = travelers_df["traveler_id"].to_numpy()[trav_indices]
    booking_freq = travelers_df["booking_frequency"].to_numpy()[trav_indices]
    loyalty = travelers_df["loyalty_segment"].to_numpy()[trav_indices]

    destinations = [
        "Cancun",
        "Paris",
        "Tokyo",
        "London",
        "New York",
        "Maui",
        "Rome",
        "Barcelona",
        "Bali",
        "Sydney",
        "Dubai",
        "Vancouver",
    ]
    dest_arr = rng.choice(destinations, size=n)

    loyalty_boost = np.where(
        loyalty == "Platinum",
        0.25,
        np.where(loyalty == "Gold", 0.16, np.where(loyalty == "Silver", 0.08, 0.0)),
    )
    search_freq = np.clip(rng.poisson(2.0 + booking_freq * 0.45) + 1, 1, 45)
    travel_intent = np.round(
        np.clip(
            0.25 + 0.015 * search_freq + loyalty_boost + rng.normal(0, 0.12, size=n), 0.05, 0.99
        ),
        4,
    )
    booking_prob = np.round(
        np.clip(
            0.08 + 0.55 * travel_intent + 0.012 * booking_freq + rng.normal(0, 0.06, size=n),
            0.01,
            0.96,
        ),
        4,
    )
    engagement_score = np.round(
        np.clip(
            20.0 + 55.0 * travel_intent + 1.5 * search_freq + rng.normal(0, 8.0, size=n), 5.0, 99.5
        ),
        2,
    )

    end_ts = pd.Timestamp(end_str)
    start_ts = end_ts - pd.Timedelta(days=180)
    sec_range = int((end_ts - start_ts).total_seconds())
    sig_ts = start_ts + pd.to_timedelta(rng.integers(0, sec_range, size=n), unit="s")

    return pd.DataFrame(
        {
            "traveler_id": trav_ids,
            "destination_interest": dest_arr,
            "travel_intent": travel_intent,
            "search_frequency": search_freq,
            "booking_probability": booking_prob,
            "engagement_score": engagement_score,
            "signal_timestamp": sig_ts.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def generate_ad_impressions_and_experiments(
    rng: np.random.Generator,
    travelers_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    n_impressions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 1M+ impression events with realistic targeting, reach saturation, CTR, and CVR."""
    n_camp = len(campaigns_df)
    n_trav = len(travelers_df)

    # Pre-index traveler latest intent & engagement
    trav_sig = (
        signals_df.sort_values("signal_timestamp")
        .groupby("traveler_id", as_index=False)[
            ["travel_intent", "booking_probability", "engagement_score"]
        ]
        .last()
    )
    trav_merged = travelers_df.merge(trav_sig, on="traveler_id", how="left")
    trav_merged["travel_intent"] = trav_merged["travel_intent"].fillna(0.35)
    trav_merged["booking_probability"] = trav_merged["booking_probability"].fillna(0.18)
    trav_merged["engagement_score"] = trav_merged["engagement_score"].fillna(42.0)

    # Group traveler indices by country for realistic audience pool sampling
    trav_country = trav_merged["country"].to_numpy()
    country_to_idx: dict[str, np.ndarray] = {
        c: np.where(trav_country == c)[0] for c in COUNTRY_WEIGHTS
    }

    # Campaign weights proportional to budget^(0.85)
    budgets = campaigns_df["budget"].to_numpy()
    camp_probs = np.power(budgets, 0.85)
    camp_probs /= camp_probs.sum()
    camp_idx = rng.choice(n_camp, size=n_impressions, p=camp_probs)

    camp_ids = campaigns_df["campaign_id"].to_numpy()[camp_idx]
    camp_start = pd.to_datetime(campaigns_df["start_date"]).to_numpy()[camp_idx]
    camp_end = pd.to_datetime(campaigns_df["end_date"]).to_numpy()[camp_idx]
    camp_target_country = campaigns_df["target_country"].to_numpy()[camp_idx]
    camp_target_age = campaigns_df["target_age_group"].to_numpy()[camp_idx]
    camp_objective = campaigns_df["campaign_objective"].to_numpy()[camp_idx]
    camp_channel = campaigns_df["advertising_channel"].to_numpy()[camp_idx]

    # Sample timestamps uniformly inside campaign active window
    span_sec = np.maximum(
        ((camp_end - camp_start) / np.timedelta64(1, "s")).astype(np.int64),
        86400,
    )
    rand_sec = (rng.random(n_impressions) * span_sec).astype(np.int64)
    imp_ts = pd.to_datetime(camp_start) + pd.to_timedelta(rand_sec, unit="s")

    # Assign placement correlated with channel
    placement_arr = np.empty(n_impressions, dtype=object)
    is_search = camp_channel == "search_ads"
    is_mobile = camp_channel == "mobile_native"
    is_other = ~(is_search | is_mobile)
    placement_arr[is_search] = rng.choice(
        PLACEMENTS, size=int(is_search.sum()), p=[0.10, 0.55, 0.15, 0.08, 0.05, 0.07]
    )
    placement_arr[is_mobile] = rng.choice(
        PLACEMENTS, size=int(is_mobile.sum()), p=[0.10, 0.15, 0.10, 0.10, 0.45, 0.10]
    )
    placement_arr[is_other] = rng.choice(
        PLACEMENTS, size=int(is_other.sum()), p=[0.28, 0.18, 0.22, 0.12, 0.10, 0.10]
    )

    # Sample traveler IDs: ~76% from target country pool (creates realistic frequency cap & reach saturation)
    trav_idx = rng.integers(0, n_trav, size=n_impressions)
    in_target_draw = rng.random(n_impressions) < 0.76
    for c, pool in country_to_idx.items():
        mask = in_target_draw & (camp_target_country == c)
        cnt = int(mask.sum())
        if cnt > 0 and len(pool) > 0:
            # Power-law selection inside pool to emulate frequency concentration on active shoppers
            raw_pos = np.floor(np.power(rng.random(cnt), 1.35) * len(pool)).astype(np.int64)
            trav_idx[mask] = pool[np.clip(raw_pos, 0, len(pool) - 1)]

    trav_ids = trav_merged["traveler_id"].to_numpy()[trav_idx]
    trav_device = trav_merged["device_type"].to_numpy()[trav_idx]
    trav_age = trav_merged["age_group"].to_numpy()[trav_idx]
    trav_country_sel = trav_merged["country"].to_numpy()[trav_idx]
    trav_intent = trav_merged["travel_intent"].to_numpy()[trav_idx]
    trav_bprob = trav_merged["booking_probability"].to_numpy()[trav_idx]
    trav_loyalty = trav_merged["loyalty_segment"].to_numpy()[trav_idx]

    # Impression cost ($0.004 to $0.055)
    base_cpm = np.array([PLACEMENT_CPM_BASE[p] for p in placement_arr])
    country_cpm_mult = np.where(np.isin(trav_country_sel, ["US", "UK", "AU", "CH"]), 1.25, 0.90)
    imp_cost = np.round(
        np.clip(
            (base_cpm / 1000.0) * country_cpm_mult * rng.lognormal(0.0, 0.18, size=n_impressions),
            0.003,
            0.085,
        ),
        4,
    )

    # CTR logistic model (learnable relationship with placement, match, intent, channel, loyalty)
    p_eff = np.array([PLACEMENT_CTR_EFFECT[p] for p in placement_arr])
    match_country = (trav_country_sel == camp_target_country).astype(float)
    match_age = (trav_age == camp_target_age).astype(float)
    obj_eff = np.where(
        camp_objective == "traffic", 0.35, np.where(camp_objective == "retargeting", 0.55, 0.10)
    )
    loyalty_eff = np.where(
        trav_loyalty == "Platinum", 0.45, np.where(trav_loyalty == "Gold", 0.25, 0.0)
    )

    # Assign ML-targeting flag for a clean subset of campaigns (used for causal analysis too)
    # Campaigns with ID mod 2 == 0 after 2025-02-01 or randomized at traveler level
    logit_ctr = (
        -3.55
        + p_eff
        + 0.45 * match_country
        + 0.35 * match_age
        + 1.35 * trav_intent
        + obj_eff
        + loyalty_eff
    )
    click_prob = 1.0 / (1.0 + np.exp(-logit_ctr))
    clicked = rng.binomial(1, click_prob)

    # Conversion conditional on click (and tiny view-through probability)
    obj_conv_eff = np.where(
        camp_objective == "bookings", 0.65, np.where(camp_objective == "retargeting", 0.50, -0.35)
    )
    logit_cvr = -2.15 + 2.4 * trav_bprob + 0.55 * match_country + obj_conv_eff + 0.40 * loyalty_eff
    conv_prob_if_click = 1.0 / (1.0 + np.exp(-logit_cvr))
    conv_prob = np.where(clicked == 1, conv_prob_if_click, 0.0015)
    converted = rng.binomial(1, conv_prob)

    impressions_df = pd.DataFrame(
        {
            "impression_id": [f"IMP_{i:08d}" for i in range(1, n_impressions + 1)],
            "campaign_id": camp_ids,
            "traveler_id": trav_ids,
            "timestamp": imp_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "placement": placement_arr,
            "device_type": trav_device,
            "impression_cost": imp_cost,
            "clicked": clicked,
            "converted": converted,
        }
    )

    # Build clean Causal Inference Experimentation dataset (120,000 user-campaign observations)
    # Includes: randomized A/B test group, pre/post rollout period for DiD, and observational propensity features
    n_exp = 120_000
    exp_trav_idx = rng.integers(0, n_trav, size=n_exp)
    exp_camp_idx = rng.integers(0, n_camp, size=n_exp)
    exp_trav_id = trav_merged["traveler_id"].to_numpy()[exp_trav_idx]
    exp_intent = trav_merged["travel_intent"].to_numpy()[exp_trav_idx]
    exp_bprob = trav_merged["booking_probability"].to_numpy()[exp_trav_idx]
    exp_bfreq = trav_merged["booking_frequency"].to_numpy()[exp_trav_idx]
    exp_country = trav_merged["country"].to_numpy()[exp_trav_idx]
    exp_loyalty = trav_merged["loyalty_segment"].to_numpy()[exp_trav_idx]
    exp_device = trav_merged["device_type"].to_numpy()[exp_trav_idx]
    exp_budget = campaigns_df["budget"].to_numpy()[exp_camp_idx]
    exp_camp_id = campaigns_df["campaign_id"].to_numpy()[exp_camp_idx]
    exp_channel = campaigns_df["advertising_channel"].to_numpy()[exp_camp_idx]

    # Cluster randomization by traveler_id hash (consistent user-level assignment for A/B test)
    ab_group = np.where((exp_trav_idx % 2) == 1, "treatment", "control")
    ab_treated = (ab_group == "treatment").astype(int)

    # Heterogeneous treatment effect (Uplift): highest for mid/high intent mobile/app users ("Persuadables")
    # Small/zero for low-intent "Lost Causes" and slight negative for already-saturated Platinum "Sleeping Dogs"
    tau_heterogeneous = (
        0.012
        + 0.028 * np.exp(-((exp_intent - 0.62) ** 2) / 0.08)
        + 0.010 * np.isin(exp_device, ["mobile", "app"]).astype(float)
        - 0.015 * (exp_loyalty == "Platinum").astype(float)
    )
    baseline_cvr = np.clip(
        0.025 + 0.09 * exp_bprob + 0.002 * exp_bfreq + 0.015 * (exp_loyalty == "Gold"), 0.005, 0.35
    )
    ab_conv_prob = np.clip(baseline_cvr + ab_treated * tau_heterogeneous, 0.002, 0.60)
    ab_converted = rng.binomial(1, ab_conv_prob)

    # Difference-in-Differences panel indicator (pre vs post ML rollout, treatment vs comparison markets)
    did_post = rng.binomial(1, 0.5, size=n_exp)
    did_treat_market = np.isin(exp_country, ["US", "UK", "CA", "DE", "AU", "JP"]).astype(int)
    did_conv_prob = np.clip(
        0.032
        + 0.012 * did_treat_market
        + 0.008 * did_post
        + 0.019 * (did_post * did_treat_market)
        + 0.04 * exp_intent,
        0.005,
        0.55,
    )
    did_converted = rng.binomial(1, did_conv_prob)

    # Observational adoption with confounding (larger budgets & higher intent more likely to adopt ML targeting)
    obs_propensity_logit = (
        -1.1 + 0.35 * np.log1p(exp_budget / 10000.0) + 1.4 * exp_intent + 0.06 * exp_bfreq
    )
    obs_propensity = 1.0 / (1.0 + np.exp(-obs_propensity_logit))
    obs_treated = rng.binomial(1, obs_propensity)
    obs_conv_prob = np.clip(baseline_cvr + 0.021 * obs_treated, 0.002, 0.60)
    obs_converted = rng.binomial(1, obs_conv_prob)

    revenue_per_conv = np.round(rng.lognormal(mean=4.8, sigma=0.55, size=n_exp), 2)

    experiment_df = pd.DataFrame(
        {
            "event_id": [f"EXP_{i:07d}" for i in range(1, n_exp + 1)],
            "traveler_id": exp_trav_id,
            "campaign_id": exp_camp_id,
            "country": exp_country,
            "device_type": exp_device,
            "loyalty_segment": exp_loyalty,
            "advertising_channel": exp_channel,
            "campaign_budget": exp_budget,
            "travel_intent": exp_intent,
            "booking_probability": exp_bprob,
            "booking_frequency": exp_bfreq,
            "ab_group": ab_group,
            "ab_treated": ab_treated,
            "ab_converted": ab_converted,
            "did_post": did_post,
            "did_treat_market": did_treat_market,
            "did_converted": did_converted,
            "obs_treated": obs_treated,
            "obs_converted": obs_converted,
            "conversion_revenue": np.where(ab_converted == 1, revenue_per_conv, 0.0),
        }
    )

    return impressions_df, experiment_df


def generate_bidding_data(
    rng: np.random.Generator, campaigns_df: pd.DataFrame, n_bids: int
) -> pd.DataFrame:
    n_camp = len(campaigns_df)
    camp_idx = rng.integers(0, n_camp, size=n_bids)
    camp_ids = campaigns_df["campaign_id"].to_numpy()[camp_idx]
    adv_ids = campaigns_df["advertiser_id"].to_numpy()[camp_idx]
    camp_start = pd.to_datetime(campaigns_df["start_date"]).to_numpy()[camp_idx]
    camp_end = pd.to_datetime(campaigns_df["end_date"]).to_numpy()[camp_idx]

    placement_arr = rng.choice(PLACEMENTS, size=n_bids)
    base_cpm = np.array([PLACEMENT_CPM_BASE[p] for p in placement_arr])
    bid_amount = np.round(np.clip(base_cpm * rng.lognormal(0.0, 0.35, size=n_bids), 2.5, 95.0), 2)

    floor_price = base_cpm * 0.55
    clearing_price = base_cpm * rng.lognormal(0.0, 0.25, size=n_bids)

    auction_result = np.where(
        bid_amount < floor_price,
        "floor_rejected",
        np.where(bid_amount >= clearing_price, "won", "lost"),
    )

    span_sec = np.maximum(
        ((camp_end - camp_start) / np.timedelta64(1, "s")).astype(np.int64), 86400
    )
    auction_ts = pd.to_datetime(camp_start) + pd.to_timedelta(
        (rng.random(n_bids) * span_sec).astype(np.int64), unit="s"
    )

    return pd.DataFrame(
        {
            "bid_id": [f"BID_{i:07d}" for i in range(1, n_bids + 1)],
            "campaign_id": camp_ids,
            "placement": placement_arr,
            "bid_amount": bid_amount,
            "auction_timestamp": auction_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "advertiser_id": adv_ids,
            "auction_result": auction_result,
        }
    )


def generate_all_datasets(
    config: PlatformConfig | None = None, scale: float = 1.0
) -> dict[str, Path]:
    """Generate all synthetic datasets and save them to data/synthetic."""
    cfg = config or get_config()
    rng = np.random.default_rng(cfg.project.seed)
    out_dir = cfg.resolve_path(cfg.paths.synthetic_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trav = max(1000, int(cfg.data_generation.num_travelers * scale))
    n_camp = max(200, int(cfg.data_generation.num_campaigns * scale))
    n_imp = max(5000, int(cfg.data_generation.num_impressions * scale))
    n_days = max(180, int(cfg.data_generation.num_inventory_days))
    n_bids = max(2000, int(cfg.data_generation.num_bids * scale))
    n_sig = max(2000, int(cfg.data_generation.num_targeting_signals * scale))

    logger.info("Generating travelers (%d rows)...", n_trav)
    travelers_df = generate_travelers(rng, n_trav)

    logger.info("Generating campaigns (%d rows)...", n_camp)
    campaigns_df = generate_campaigns(
        rng, n_camp, cfg.data_generation.campaign_start_date, cfg.data_generation.campaign_end_date
    )

    logger.info("Generating ad inventory (%d days x 72 country-placements)...", n_days)
    inventory_df = generate_ad_inventory(rng, cfg.data_generation.start_date, n_days)

    logger.info("Generating targeting signals (%d rows)...", n_sig)
    signals_df = generate_targeting_signals(
        rng, travelers_df, n_sig, cfg.data_generation.campaign_end_date
    )

    logger.info("Generating ad impressions (%d rows) & causal experiment dataset...", n_imp)
    impressions_df, experiment_df = generate_ad_impressions_and_experiments(
        rng, travelers_df, campaigns_df, signals_df, n_imp
    )

    logger.info("Generating bidding data (%d rows)...", n_bids)
    bidding_df = generate_bidding_data(rng, campaigns_df, n_bids)

    outputs = {
        "travelers": out_dir / "travelers.csv",
        "campaigns": out_dir / "campaigns.csv",
        "ad_inventory": out_dir / "ad_inventory.csv",
        "targeting_signals": out_dir / "targeting_signals.csv",
        "ad_impressions": out_dir / "ad_impressions.csv",
        "bidding_data": out_dir / "bidding_data.csv",
        "experiment_events": out_dir / "experiment_events.csv",
    }
    dfs = {
        "travelers": travelers_df,
        "campaigns": campaigns_df,
        "ad_inventory": inventory_df,
        "targeting_signals": signals_df,
        "ad_impressions": impressions_df,
        "bidding_data": bidding_df,
        "experiment_events": experiment_df,
    }
    for name, df in dfs.items():
        df.to_csv(outputs[name], index=False)
        logger.info("Saved %s (%d rows) -> %s", name, len(df), outputs[name])

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Expedia advertising datasets.")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Dataset size multiplier (1.0 = full >=1M rows)."
    )
    args = parser.parse_args()
    generate_all_datasets(scale=args.scale)


if __name__ == "__main__":
    main()
