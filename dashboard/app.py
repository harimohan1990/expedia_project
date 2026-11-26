"""Interactive Streamlit Dashboard for Expedia Travel Advertising Intelligence & Forecasting Platform.

Includes all 8 required sections:
1. Advertising Overview
2. Audience Reach Forecast
3. Inventory Forecast
4. Campaign Performance
5. Audience Segmentation
6. Causal Experiment Results
7. Model Evaluation
8. Model Monitoring
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.settings import get_config

st.set_page_config(
    page_title="Expedia Travel Ad Intelligence & Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = get_config()
PROCESSED_DIR = cfg.resolve_path(cfg.paths.processed_data_dir)
REPORTS_DIR = cfg.resolve_path(cfg.paths.reports_dir)
MODELS_DIR = cfg.resolve_path(cfg.paths.models_dir)


@st.cache_data(ttl=300)
def load_parquet_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_json_safe(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    st.title("✈️ Expedia Travel Advertising Intelligence & Forecasting Platform")
    st.caption(
        "Production ML Platform: PySpark ETL • Audience Reach • 7/14/30d Inventory Forecasting • "
        "Pre-Launch Campaign Prediction • Causal Inference • Evidently & MLflow Observability"
    )

    section = st.sidebar.radio(
        "Navigation",
        [
            "1. Advertising Overview",
            "2. Audience Reach Forecast",
            "3. Inventory Forecast",
            "4. Campaign Performance",
            "5. Audience Segmentation",
            "6. Causal Experiment Results",
            "7. Model Evaluation",
            "8. Model Monitoring",
        ],
    )

    daily_imp = load_parquet_safe(PROCESSED_DIR / "aggregations" / "daily_impressions")
    reach_by_country = load_parquet_safe(
        PROCESSED_DIR / "aggregations" / "audience_reach_by_country"
    )
    camp_metrics = load_parquet_safe(PROCESSED_DIR / "aggregations" / "campaign_metrics")
    seg_engagement = load_parquet_safe(
        PROCESSED_DIR / "aggregations" / "audience_engagement_by_segment"
    )
    inv_by_placement = load_parquet_safe(PROCESSED_DIR / "aggregations" / "inventory_by_placement")
    avg_bids = load_parquet_safe(PROCESSED_DIR / "aggregations" / "avg_bidding_price")

    countries = (
        sorted(reach_by_country["country"].dropna().unique().tolist())
        if not reach_by_country.empty
        else ["US"]
    )
    placements = (
        sorted(inv_by_placement["placement"].dropna().unique().tolist())
        if not inv_by_placement.empty
        else ["homepage_hero"]
    )

    selected_countries = st.sidebar.multiselect(
        "Filter Countries", countries, default=countries[:5]
    )
    selected_placements = st.sidebar.multiselect(
        "Filter Placements", placements, default=placements
    )
    forecast_horizon = st.sidebar.selectbox(
        "Inventory Forecast Horizon (Days)", [7, 14, 30], index=1
    )

    if section == "1. Advertising Overview":
        st.header("📊 Advertising Intelligence Overview")
        if not daily_imp.empty:
            total_imp = int(daily_imp["total_impressions"].sum())
            total_clicks = int(daily_imp["total_clicks"].sum())
            total_conv = int(daily_imp["total_conversions"].sum())
            total_spend = float(daily_imp["total_ad_spend"].sum())
            total_rev = float(daily_imp["attributed_revenue"].sum())
            overall_ctr = total_clicks / max(1, total_imp)
            overall_roas = total_rev / max(1.0, total_spend)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Impressions", f"{total_imp:,}")
            c2.metric("Overall CTR", f"{overall_ctr:.2%}")
            c3.metric("Total Conversions", f"{total_conv:,}")
            c4.metric("Ad Spend", f"${total_spend:,.0f}")
            c5.metric("Attributed ROAS", f"{overall_roas:.2f}x")

            daily_imp["event_date"] = pd.to_datetime(daily_imp["event_date"])
            fig_trend = px.line(
                daily_imp.sort_values("event_date"),
                x="event_date",
                y=["total_impressions", "unique_travelers"],
                title="Daily Advertising Impressions & Unique Travelers Reached",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if not reach_by_country.empty:
                sub_c = reach_by_country[
                    reach_by_country["country"].isin(selected_countries or countries)
                ]
                fig_c = px.bar(
                    sub_c,
                    x="country",
                    y="unique_reach",
                    color="avg_frequency",
                    title="Unique Traveler Reach by Country",
                )
                st.plotly_chart(fig_c, use_container_width=True)
        with col_b:
            if not avg_bids.empty:
                fig_b = px.bar(
                    avg_bids,
                    x="placement",
                    y="avg_bid_amount",
                    color="auction_result",
                    barmode="group",
                    title="Average CPM Bid Price by Placement & Auction Result",
                )
                st.plotly_chart(fig_b, use_container_width=True)

    elif section == "2. Audience Reach Forecast":
        st.header("🎯 Audience Reach Forecasting (Model 1)")
        reach_df = load_csv_safe(REPORTS_DIR / "audience_reach_forecasts.csv")
        if reach_df.empty:
            st.warning("Run `make train` to generate Audience Reach forecasts.")
            return
        if selected_countries:
            reach_df = reach_df[reach_df["country"].isin(selected_countries)]
        if selected_placements:
            reach_df = reach_df[reach_df["advertising_placement"].isin(selected_placements)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecasted Campaigns", f"{len(reach_df):,}")
        c2.metric("Avg Actual Reach", f"{reach_df['actual_unique_reach'].mean():,.0f}")
        c3.metric("Avg Predicted Reach", f"{reach_df['predicted_unique_reach'].mean():,.0f}")

        fig = px.scatter(
            reach_df.head(1000),
            x="actual_unique_reach",
            y="predicted_unique_reach",
            color="country",
            hover_data=["campaign_id", "confidence_interval"],
            title="Actual vs. Predicted Campaign Unique Reach (with 95% CI)",
        )
        max_val = max(
            reach_df["actual_unique_reach"].max(), reach_df["predicted_unique_reach"].max()
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Ideal y=x",
                line=dict(dash="dash", color="gray"),
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            reach_df[
                [
                    "campaign_id",
                    "forecast_date",
                    "country",
                    "advertising_placement",
                    "actual_unique_reach",
                    "predicted_unique_reach",
                    "confidence_interval",
                ]
            ].head(50),
            use_container_width=True,
        )

    elif section == "3. Inventory Forecast":
        st.header("📦 Advertising Inventory Availability Forecast (Model 2)")
        inv_future = load_csv_safe(REPORTS_DIR / "inventory_forecasts_30d.csv")
        inv_holdout = load_csv_safe(REPORTS_DIR / "inventory_holdout_predictions.csv")
        if inv_future.empty:
            st.warning("Run `make train` to generate 7/14/30-day Inventory forecasts.")
            return

        c_sel = st.selectbox("Select Country", sorted(inv_future["country"].unique()), index=0)
        p_sel = st.selectbox("Select Placement", sorted(inv_future["placement"].unique()), index=0)

        hold_sub = inv_holdout[
            (inv_holdout["country"] == c_sel) & (inv_holdout["placement"] == p_sel)
        ].copy()
        fut_sub = inv_future[
            (inv_future["country"] == c_sel)
            & (inv_future["placement"] == p_sel)
            & (inv_future["step_day"] <= forecast_horizon)
        ].copy()

        fig = go.Figure()
        if not hold_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=hold_sub["date"],
                    y=hold_sub["available_impressions"],
                    mode="lines+markers",
                    name="Actual (Validation)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hold_sub["date"],
                    y=hold_sub["predicted_available_impressions"],
                    mode="lines",
                    name="Predicted (Validation)",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=fut_sub["date"],
                y=fut_sub["predicted_available_impressions"],
                mode="lines+markers",
                name=f"Future Forecast ({forecast_horizon}d)",
                line=dict(dash="dot", width=3),
            )
        )
        fig.update_layout(
            title=f"Available Impressions: {c_sel} — {p_sel} ({forecast_horizon}-Day Horizon)",
            xaxis_title="Date",
            yaxis_title="Impressions",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            fut_sub[["date", "placement", "country", "predicted_available_impressions"]],
            use_container_width=True,
        )

    elif section == "4. Campaign Performance":
        st.header("🚀 Pre-Launch Campaign Performance Prediction (Model 3)")
        cp_df = load_csv_safe(REPORTS_DIR / "campaign_performance_predictions.csv")
        target_metrics = load_json_safe(REPORTS_DIR / "campaign_target_metrics.json")
        if cp_df.empty:
            st.warning("Run `make train` to generate Campaign Performance predictions.")
            return

        if target_metrics:
            cols = st.columns(len(target_metrics))
            for col, (tname, m) in zip(cols, target_metrics.items(), strict=False):
                col.metric(
                    tname.replace("_", " ").title(), f"R²={m['r2']:.3f}", f"MAE={m['mae']:.2f}"
                )

        fig_rev = px.scatter(
            cp_df.head(1000),
            x="actual_campaign_revenue",
            y="predicted_campaign_revenue",
            color="high_performing_prob",
            title="Actual vs. Predicted Campaign Revenue (Colored by High-Perf Probability)",
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        if not camp_metrics.empty:
            fig_ch = px.box(
                camp_metrics,
                x="advertising_channel",
                y="roas",
                color="campaign_objective",
                title="Campaign ROAS Distribution by Channel & Objective",
            )
            st.plotly_chart(fig_ch, use_container_width=True)

    elif section == "5. Audience Segmentation":
        st.header("👥 Audience Segmentation & Engagement Intelligence")
        if seg_engagement.empty:
            st.warning("Run `make etl` first to generate audience segmentation metrics.")
            return
        fig_seg = px.sunburst(
            seg_engagement,
            path=["loyalty_segment", "travel_interest", "device_type"],
            values="impressions",
            color="avg_engagement_score",
            title="Traveler Impressions & Engagement Score by Loyalty, Travel Interest, and Device",
        )
        st.plotly_chart(fig_seg, use_container_width=True)
        st.dataframe(seg_engagement, use_container_width=True)

    elif section == "6. Causal Experiment Results":
        st.header("🔬 Causal Inference & ML Targeting Experimentation")
        causal = load_json_safe(REPORTS_DIR / "causal_impact_report.json")
        uplift_seg = load_csv_safe(REPORTS_DIR / "uplift_segments.csv")
        if not causal:
            st.warning("Run `make causal` to generate causal inference results.")
            return

        ab = causal["ab_testing"]
        did = causal["difference_in_differences"]
        psm = causal["propensity_score_matching"]
        up = causal["uplift_modeling"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "A/B Relative Lift",
            f"+{ab['relative_lift_pct']:.2f}%",
            f"Abs: +{ab['absolute_lift']:.3%}",
        )
        c2.metric("DiD ATT Estimate", f"+{did['did_att_estimate']:.3%}", f"p={did['p_value']:.2e}")
        c3.metric(
            "PSM Unbiased ATT", f"+{psm['psm_att']:.3%}", f"Naive: +{psm['naive_biased_lift']:.3%}"
        )
        c4.metric("Mean Uplift (CATE)", f"+{up['mean_ate']:.3%}", f"p90: +{up['cate_p90']:.3%}")

        col1, col2 = st.columns(2)
        with col1:
            dec_df = pd.DataFrame(up["decile_uplift"])
            fig_dec = px.bar(
                dec_df,
                x="uplift_decile",
                y="observed_lift",
                color="mean_cate",
                title="Uplift by Predicted CATE Decile (Persuadables at Right)",
            )
            st.plotly_chart(fig_dec, use_container_width=True)
        with col2:
            if not uplift_seg.empty:
                fig_u = px.bar(
                    uplift_seg,
                    x="loyalty_segment",
                    y="mean_predicted_uplift",
                    color="device_type",
                    barmode="group",
                    title="Predicted Treatment Uplift (CATE) by Loyalty & Device",
                )
                st.plotly_chart(fig_u, use_container_width=True)

        st.subheader("PSM Covariate Balance (Standardized Mean Difference)")
        bal_df = pd.DataFrame(
            {
                "Covariate": list(psm["smd_before_matching"].keys()),
                "SMD Before Matching": list(psm["smd_before_matching"].values()),
                "SMD After Matching": list(psm["smd_after_matching"].values()),
            }
        )
        st.dataframe(bal_df, use_container_width=True)

    elif section == "7. Model Evaluation":
        st.header("⚖️ Multi-Algorithm Model Evaluation & MLflow Comparison")
        exps = [
            ("Audience Reach Forecasting", cfg.ml.mlflow_experiment_reach),
            (
                "Inventory Forecasting (ARIMA/SARIMA/Prophet/XGB/LGBM)",
                cfg.ml.mlflow_experiment_inventory,
            ),
            ("Campaign Revenue Regression", cfg.ml.mlflow_experiment_performance),
            (
                "Campaign High-Performance Classification",
                f"{cfg.ml.mlflow_experiment_performance}-cls",
            ),
        ]
        for title, exp in exps:
            st.subheader(title)
            comp_df = load_csv_safe(MODELS_DIR / exp / "model_comparison.csv")
            if comp_df.empty:
                st.info(f"No evaluation results found for {exp}.")
            else:
                st.dataframe(comp_df, use_container_width=True)

    elif section == "8. Model Monitoring":
        st.header("📡 Data Quality, Feature Drift & Model Monitoring")
        mon = load_json_safe(REPORTS_DIR / "monitoring_report.json")
        if not mon:
            st.warning("Run `make monitor` to generate drift and observability metrics.")
            return

        for model_key, dd in mon.get("data_drift", {}).items():
            st.subheader(f"Feature Drift & Data Quality — {model_key}")
            fd_rows = [{"feature": k, **v} for k, v in dd.get("feature_drift", {}).items()]
            if fd_rows:
                fd_df = pd.DataFrame(fd_rows)
                st.plotly_chart(
                    px.bar(
                        fd_df,
                        x="feature",
                        y="psi",
                        color="status",
                        title=f"PSI Feature Drift ({model_key})",
                    ),
                    use_container_width=True,
                )

        st.subheader("Model Performance & Prediction Drift")
        md_rows = [
            {
                "model": k,
                "prediction_psi": v["prediction_psi"],
                "baseline_rmse": v["baseline_rmse"],
                **v["current_metrics"],
                "alerts": len(v["alerts_triggered"]),
            }
            for k, v in mon.get("model_drift", {}).items()
        ]
        if md_rows:
            st.dataframe(pd.DataFrame(md_rows), use_container_width=True)


if __name__ == "__main__":
    main()
