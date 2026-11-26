"""Tests for Causal Inference & Experimentation (A/B, DiD, PSM, Uplift Modeling)."""

from __future__ import annotations

import pandas as pd

from causal_inference.ab_testing import run_ab_test
from causal_inference.difference_in_differences import run_difference_in_differences
from causal_inference.propensity_matching import run_propensity_score_matching
from causal_inference.uplift_modeling import run_uplift_modeling


def test_ab_testing_detects_positive_lift(experiment_df: pd.DataFrame) -> None:
    res = run_ab_test(experiment_df)
    assert res["n_observations"] == len(experiment_df)
    assert res["absolute_lift"] > 0
    assert res["ci_95_lower"] < res["absolute_lift"] < res["ci_95_upper"]
    assert res["srm_passed"] is True


def test_difference_in_differences_returns_att(experiment_df: pd.DataFrame) -> None:
    res = run_difference_in_differences(experiment_df)
    assert "did_att_estimate" in res
    assert res["ci_95_lower"] < res["ci_95_upper"]


def test_propensity_score_matching_improves_covariate_balance(experiment_df: pd.DataFrame) -> None:
    res = run_propensity_score_matching(experiment_df, max_sample=2500)
    assert res["matched_pairs"] > 100
    for cov, smd_after in res["smd_after_matching"].items():
        assert smd_after < 0.25, f"Covariate {cov} still imbalanced after matching: {smd_after}"


def test_uplift_modeling_produces_heterogeneous_segments(experiment_df: pd.DataFrame) -> None:
    summary, seg_df = run_uplift_modeling(experiment_df, max_sample=2500)
    assert "mean_ate" in summary
    assert summary["cate_p90"] >= summary["cate_p10"]
    assert not seg_df.empty
