"""Heterogeneous Treatment Effect (Uplift Modeling) using EconML Double Machine Learning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from configs.settings import get_logger

logger = get_logger(__name__)


def run_uplift_modeling(
    df: pd.DataFrame,
    treatment_col: str = "ab_treated",
    outcome_col: str = "ab_converted",
    max_sample: int = 40_000,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate segment-level Conditional Average Treatment Effects (CATE) via CausalForestDML."""
    work = df.sample(n=min(len(df), max_sample), random_state=seed).reset_index(drop=True)

    num_cols = ["travel_intent", "booking_probability", "booking_frequency", "campaign_budget"]
    cat_cols = ["loyalty_segment", "device_type"]

    enc = OneHotEncoder(drop="first", sparse_output=False)
    X_cat = enc.fit_transform(work[cat_cols])
    X = np.hstack([work[num_cols].to_numpy(dtype=float), X_cat])
    T = work[treatment_col].to_numpy(dtype=int)
    Y = work[outcome_col].to_numpy(dtype=float)

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=80, max_depth=7, random_state=seed, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=80, max_depth=6, random_state=seed, n_jobs=-1),
        discrete_treatment=True,
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=25,
        random_state=seed,
    )
    est.fit(Y, T, X=X)
    cate = est.effect(X).ravel()
    work["predicted_uplift_cate"] = cate

    # Classify into strategic uplift personas
    work["uplift_persona"] = np.where(
        cate >= 0.022,
        "Persuadables (High Uplift)",
        np.where(
            cate >= 0.005,
            "Moderate Responders",
            np.where(cate >= -0.002, "Low Uplift / Lost Causes", "Sleeping Dogs (Negative Uplift)"),
        ),
    )

    segment_summary = (
        work.groupby(["loyalty_segment", "device_type"], as_index=False)
        .agg(
            users=("traveler_id", "nunique"),
            avg_travel_intent=("travel_intent", "mean"),
            mean_predicted_uplift=("predicted_uplift_cate", "mean"),
            observed_treat_cvr=(
                outcome_col,
                lambda s: (
                    float(s[work.loc[s.index, treatment_col] == 1].mean())
                    if (work.loc[s.index, treatment_col] == 1).any()
                    else 0.0
                ),
            ),
            observed_ctrl_cvr=(
                outcome_col,
                lambda s: (
                    float(s[work.loc[s.index, treatment_col] == 0].mean())
                    if (work.loc[s.index, treatment_col] == 0).any()
                    else 0.0
                ),
            ),
        )
        .sort_values("mean_predicted_uplift", ascending=False)
        .reset_index(drop=True)
    )
    segment_summary["observed_segment_lift"] = (
        segment_summary["observed_treat_cvr"] - segment_summary["observed_ctrl_cvr"]
    )

    # Decile uplift table for Qini / targeting ROI visualization
    work["uplift_decile"] = pd.qcut(
        work["predicted_uplift_cate"], q=10, labels=False, duplicates="drop"
    )
    decile_stats = (
        work.groupby("uplift_decile", as_index=False)
        .agg(
            n=("predicted_uplift_cate", "count"),
            mean_cate=("predicted_uplift_cate", "mean"),
            treat_cvr=(
                outcome_col,
                lambda s: float(s[work.loc[s.index, treatment_col] == 1].mean()),
            ),
            ctrl_cvr=(
                outcome_col,
                lambda s: float(s[work.loc[s.index, treatment_col] == 0].mean()),
            ),
        )
        .sort_values("uplift_decile", ascending=False)
    )
    decile_stats["observed_lift"] = decile_stats["treat_cvr"] - decile_stats["ctrl_cvr"]

    persona_counts = work["uplift_persona"].value_counts().to_dict()
    summary = {
        "mean_ate": round(float(np.mean(cate)), 6),
        "cate_p10": round(float(np.percentile(cate, 10)), 6),
        "cate_p50": round(float(np.percentile(cate, 50)), 6),
        "cate_p90": round(float(np.percentile(cate, 90)), 6),
        "top_segment": segment_summary.iloc[0].to_dict(),
        "persona_distribution": {str(k): int(v) for k, v in persona_counts.items()},
        "decile_uplift": decile_stats.to_dict(orient="records"),
    }
    logger.info(
        "Uplift Modeling: mean_CATE=%.4f p90=%.4f", summary["mean_ate"], summary["cate_p90"]
    )
    return summary, segment_summary
