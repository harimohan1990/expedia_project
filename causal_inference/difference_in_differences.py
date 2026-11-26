"""Difference-in-Differences (DiD) quasi-experimental causal estimator."""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from configs.settings import get_logger

logger = get_logger(__name__)


def run_difference_in_differences(
    df: pd.DataFrame,
    treat_group_col: str = "did_treat_market",
    post_col: str = "did_post",
    outcome_col: str = "did_converted",
    cluster_col: str = "traveler_id",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Estimate ATT via two-period Difference-in-Differences with cluster-robust SEs."""
    work = df[[treat_group_col, post_col, outcome_col, cluster_col]].copy()
    work["did_interaction"] = work[treat_group_col] * work[post_col]

    # Cell means
    means = work.groupby([treat_group_col, post_col])[outcome_col].mean().to_dict()
    ctrl_pre = float(means.get((0, 0), 0.0))
    ctrl_post = float(means.get((0, 1), 0.0))
    treat_pre = float(means.get((1, 0), 0.0))
    treat_post = float(means.get((1, 1), 0.0))

    model = smf.ols(
        f"{outcome_col} ~ {treat_group_col} + {post_col} + did_interaction",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work[cluster_col]})

    att = float(model.params["did_interaction"])
    se = float(model.bse["did_interaction"])
    p_val = float(model.pvalues["did_interaction"])
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    result = {
        "control_pre_cvr": round(ctrl_pre, 6),
        "control_post_cvr": round(ctrl_post, 6),
        "treatment_pre_cvr": round(treat_pre, 6),
        "treatment_post_cvr": round(treat_post, 6),
        "control_diff": round(ctrl_post - ctrl_pre, 6),
        "treatment_diff": round(treat_post - treat_pre, 6),
        "did_att_estimate": round(att, 6),
        "cluster_robust_se": round(se, 6),
        "p_value": float(p_val),
        "ci_95_lower": round(att - z_crit * se, 6),
        "ci_95_upper": round(att + z_crit * se, 6),
        "is_statistically_significant": bool(p_val < alpha),
        "assumption_note": "Assumes parallel pre-treatment trends between treatment and comparison markets and no contemporaneous market shocks.",
    }
    logger.info("DiD ATT=%.4f (SE=%.4f, p=%.2e)", att, se, p_val)
    return result
