"""Propensity Score Matching (PSM) & DoWhy observational causal inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from dowhy import CausalModel
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.settings import get_logger

logger = get_logger(__name__)


def _standardized_mean_diff(x_t: np.ndarray, x_c: np.ndarray) -> float:
    denom = np.sqrt((np.var(x_t) + np.var(x_c)) / 2.0)
    return float(abs(np.mean(x_t) - np.mean(x_c)) / denom) if denom > 1e-8 else 0.0


def run_propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str = "obs_treated",
    outcome_col: str = "obs_converted",
    max_sample: int = 30_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Match treated and untreated observations on propensity scores and estimate ATT (with DoWhy cross-check)."""
    sample_df = df.sample(n=min(len(df), max_sample), random_state=seed).reset_index(drop=True)

    num_cols = ["campaign_budget", "travel_intent", "booking_probability", "booking_frequency"]
    cat_cols = ["loyalty_segment", "device_type"]

    X_num = StandardScaler().fit_transform(sample_df[num_cols])
    X_cat = OneHotEncoder(drop="first", sparse_output=False).fit_transform(sample_df[cat_cols])
    X = np.hstack([X_num, X_cat])
    t = sample_df[treatment_col].to_numpy(dtype=int)
    y = sample_df[outcome_col].to_numpy(dtype=float)

    # Naive (confounded) difference in means
    naive_diff = float(y[t == 1].mean() - y[t == 0].mean())

    # Estimate propensity scores e(X) = P(T=1 | X)
    ps_model = LogisticRegression(max_iter=1000, random_state=seed)
    ps_model.fit(X, t)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    logit_ps = np.log(ps / (1.0 - ps))

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]

    # 1:1 Nearest-Neighbor Matching on logit propensity score with 0.2*SD caliper
    caliper = 0.2 * float(np.std(logit_ps))
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(logit_ps[control_idx].reshape(-1, 1))
    dists, indices = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))

    valid_mask = dists[:, 0] <= caliper
    matched_t_idx = treated_idx[valid_mask]
    matched_c_idx = control_idx[indices[valid_mask, 0]]

    att_diffs = y[matched_t_idx] - y[matched_c_idx]
    att = float(np.mean(att_diffs))
    se_att = float(np.std(att_diffs, ddof=1) / np.sqrt(len(att_diffs)))
    t_stat, p_val = stats.ttest_rel(y[matched_t_idx], y[matched_c_idx])

    # Covariate balance diagnostics (SMD before vs after matching)
    balance_before = {
        col: round(
            _standardized_mean_diff(
                sample_df.loc[t == 1, col].to_numpy(), sample_df.loc[t == 0, col].to_numpy()
            ),
            4,
        )
        for col in num_cols
    }
    balance_after = {
        col: round(
            _standardized_mean_diff(
                sample_df.loc[matched_t_idx, col].to_numpy(),
                sample_df.loc[matched_c_idx, col].to_numpy(),
            ),
            4,
        )
        for col in num_cols
    }

    # DoWhy formal causal graph backdoor adjustment check
    dowhy_ate = None
    try:
        dowhy_df = sample_df[[treatment_col, outcome_col] + num_cols].copy()
        causal_model = CausalModel(
            data=dowhy_df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=num_cols,
        )
        estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
        estimate = causal_model.estimate_effect(
            estimand,
            method_name="backdoor.propensity_score_stratification",
        )
        dowhy_ate = round(float(estimate.value), 6)
    except Exception as exc:
        logger.warning("DoWhy stratification check fallback (%s)", exc)

    result = {
        "sample_size": int(len(sample_df)),
        "matched_pairs": int(len(matched_t_idx)),
        "naive_biased_lift": round(naive_diff, 6),
        "psm_att": round(att, 6),
        "psm_se": round(se_att, 6),
        "p_value": float(p_val),
        "ci_95_lower": round(att - 1.96 * se_att, 6),
        "ci_95_upper": round(att + 1.96 * se_att, 6),
        "dowhy_backdoor_ate": dowhy_ate,
        "smd_before_matching": balance_before,
        "smd_after_matching": balance_after,
        "confounding_bias_removed": round(naive_diff - att, 6),
    }
    logger.info("PSM: naive=%.4f matched_ATT=%.4f dowhy_ATE=%s", naive_diff, att, dowhy_ate)
    return result
