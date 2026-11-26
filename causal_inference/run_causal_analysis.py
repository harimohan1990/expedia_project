"""Orchestrator for Causal Inference & Experimentation and Business Impact Report generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from causal_inference.ab_testing import run_ab_test
from causal_inference.difference_in_differences import run_difference_in_differences
from causal_inference.propensity_matching import run_propensity_score_matching
from causal_inference.uplift_modeling import run_uplift_modeling
from configs.settings import PlatformConfig, get_config, get_logger

logger = get_logger(__name__)


def run_all_causal_analyses(config: PlatformConfig | None = None) -> dict[str, Any]:
    """Run A/B testing, Difference-in-Differences, PSM, and Uplift Modeling and emit reports."""
    cfg = config or get_config()
    raw_dir = cfg.resolve_path(cfg.paths.raw_data_dir)
    exp_path = raw_dir / "experiment_events.csv"
    if not exp_path.exists():
        exp_path = cfg.resolve_path(cfg.paths.synthetic_data_dir) / "experiment_events.csv"

    df = pd.read_csv(exp_path)

    ab_res = run_ab_test(df)
    did_res = run_difference_in_differences(df)
    psm_res = run_propensity_score_matching(df, seed=cfg.ml.random_state)
    uplift_res, segment_df = run_uplift_modeling(df, seed=cfg.ml.random_state)

    reports_dir = cfg.resolve_path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    segment_df.to_csv(reports_dir / "uplift_segments.csv", index=False)

    full_report = {
        "business_question": "Does ML-based audience targeting improve advertising conversion rates compared with traditional targeting?",
        "ab_testing": ab_res,
        "difference_in_differences": did_res,
        "propensity_score_matching": psm_res,
        "uplift_modeling": uplift_res,
        "causal_assumptions_and_limitations": {
            "randomization_unit": "traveler_id (cluster randomization prevents intra-user treatment spillover and accounts for repeated impressions via CR1 cluster-robust SEs).",
            "sutva": "Stable Unit Treatment Value Assumption: assumes no cross-traveler network interference or ad-auction budget cannibalization across control and treatment.",
            "ignorability_overlap": "PSM & DML assume no unobserved confounders beyond campaign budget, intent, booking probability, frequency, loyalty, and device; SMD < 0.1 confirms post-matching balance.",
            "parallel_trends": "DiD assumes treatment and comparison markets would follow parallel conversion trajectories absent ML targeting rollout.",
            "correlation_vs_causation": f"Observational naive lift ({psm_res['naive_biased_lift']:.4f}) overstates the true causal effect ({psm_res['psm_att']:.4f}) by {psm_res['confounding_bias_removed']:.4f} due to high-intent advertiser self-selection.",
        },
    }
    (reports_dir / "causal_impact_report.json").write_text(
        json.dumps(full_report, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# Causal Inference & Experimentation Business Impact Report",
        "",
        "## Business Question",
        "> Does ML-based audience targeting improve advertising conversion rates compared with traditional targeting?",
        "",
        "## 1. Randomized A/B Experiment (Cluster-Robust at Traveler Level)",
        f"- **Control CVR (Traditional):** {ab_res['control_conversion_rate']:.4%}",
        f"- **Treatment CVR (ML Targeting):** {ab_res['treatment_conversion_rate']:.4%}",
        f"- **Absolute Lift:** +{ab_res['absolute_lift']:.4%} (95% CI: [{ab_res['ci_95_lower']:.4%}, {ab_res['ci_95_upper']:.4%}])",
        f"- **Relative Lift:** +{ab_res['relative_lift_pct']:.2f}% (p-value = {ab_res['p_value']:.2e})",
        f"- **Sample Ratio Mismatch (SRM) Check:** Passed={ab_res['srm_passed']} (p = {ab_res['srm_p_value']:.4f})",
        "",
        "## 2. Difference-in-Differences (Rollout Evaluation)",
        f"- **DiD ATT Estimate:** +{did_res['did_att_estimate']:.4%} (SE = {did_res['cluster_robust_se']:.4%}, p = {did_res['p_value']:.2e})",
        "",
        "## 3. Propensity Score Matching & Confounding Bias",
        f"- **Naive Observational Lift (Biased):** +{psm_res['naive_biased_lift']:.4%}",
        f"- **Matched ATT (Unbiased):** +{psm_res['psm_att']:.4%} (Matched Pairs = {psm_res['matched_pairs']:,})",
        (
            f"- **DoWhy Backdoor ATE:** +{psm_res['dowhy_backdoor_ate']:.4%}"
            if psm_res["dowhy_backdoor_ate"] is not None
            else ""
        ),
        f"- **Confounding Selection Bias Removed:** {psm_res['confounding_bias_removed']:.4%}",
        "",
        "## 4. Heterogeneous Uplift Modeling (EconML CausalForestDML)",
        f"- **Mean Predicted CATE:** +{uplift_res['mean_ate']:.4%}",
        f"- **Top Responsive Segment:** {uplift_res['top_segment']['loyalty_segment']} / {uplift_res['top_segment']['device_type']} (Mean Uplift = +{uplift_res['top_segment']['mean_predicted_uplift']:.4%})",
        "",
        "## 5. Methodological Assumptions & Limitations",
        f"- **Randomization Unit & Clustering:** {full_report['causal_assumptions_and_limitations']['randomization_unit']}",
        f"- **SUTVA:** {full_report['causal_assumptions_and_limitations']['sutva']}",
        f"- **Unconfoundedness & Overlap:** {full_report['causal_assumptions_and_limitations']['ignorability_overlap']}",
        f"- **Parallel Trends:** {full_report['causal_assumptions_and_limitations']['parallel_trends']}",
    ]
    (reports_dir / "causal_impact_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved causal impact reports to %s", reports_dir)
    return full_report


if __name__ == "__main__":
    print(json.dumps(run_all_causal_analyses(), indent=2, default=str))
