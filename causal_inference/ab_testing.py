"""Cluster-aware A/B Testing module for ML-based vs Traditional targeting."""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from configs.settings import get_logger

logger = get_logger(__name__)


def run_ab_test(
    df: pd.DataFrame,
    treatment_col: str = "ab_treated",
    outcome_col: str = "ab_converted",
    cluster_col: str = "traveler_id",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a randomized A/B test with cluster-robust inference for repeated users.

    Accounts for repeated impressions/events per traveler by:
    1. Estimating a linear probability model with cluster-robust standard errors (CR1).
    2. Cross-checking at the user (randomization unit) level.
    """
    control = df[df[treatment_col] == 0]
    treated = df[df[treatment_col] == 1]

    control_cvr = float(control[outcome_col].mean())
    treatment_cvr = float(treated[outcome_col].mean())
    absolute_lift = treatment_cvr - control_cvr
    relative_lift_pct = (absolute_lift / control_cvr * 100.0) if control_cvr > 0 else 0.0

    # Cluster-robust SE at traveler_id randomization unit
    model = smf.ols(f"{outcome_col} ~ {treatment_col}", data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )
    est_lift = float(model.params[treatment_col])
    cluster_se = float(model.bse[treatment_col])
    p_value = float(model.pvalues[treatment_col])
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lower = est_lift - z_crit * cluster_se
    ci_upper = est_lift + z_crit * cluster_se

    # Randomization-unit (user-level) check
    user_agg = df.groupby([cluster_col, treatment_col], as_index=False)[outcome_col].mean()
    n_clusters_ctrl = int((user_agg[treatment_col] == 0).sum())
    n_clusters_treat = int((user_agg[treatment_col] == 1).sum())

    # Sample Ratio Mismatch (SRM) chi-square sanity check on unique randomization units
    total_users = n_clusters_ctrl + n_clusters_treat
    _, srm_p_value = stats.chisquare(
        [n_clusters_ctrl, n_clusters_treat], f_exp=[total_users / 2, total_users / 2]
    )

    result = {
        "n_observations": int(len(df)),
        "n_control_obs": int(len(control)),
        "n_treatment_obs": int(len(treated)),
        "n_unique_control_users": n_clusters_ctrl,
        "n_unique_treatment_users": n_clusters_treat,
        "srm_p_value": round(float(srm_p_value), 6),
        "srm_passed": bool(srm_p_value >= 0.001),
        "control_conversion_rate": round(control_cvr, 6),
        "treatment_conversion_rate": round(treatment_cvr, 6),
        "absolute_lift": round(absolute_lift, 6),
        "relative_lift_pct": round(relative_lift_pct, 2),
        "cluster_robust_se": round(cluster_se, 6),
        "p_value": float(p_value),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "is_statistically_significant": bool(p_value < alpha),
    }
    logger.info(
        "A/B Test: ctrl_cvr=%.4f treat_cvr=%.4f abs_lift=%.4f (rel=+%.2f%%, p=%.2e)",
        control_cvr,
        treatment_cvr,
        absolute_lift,
        relative_lift_pct,
        p_value,
    )
    return result
