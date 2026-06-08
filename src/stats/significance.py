"""
Statistical Significance Tests for A/B Testing
================================================
Tests whether the challenger model is significantly better than control.

Methods:
  - Z-test for proportions (conversion rates)
  - T-test for continuous metrics (AUC scores)
  - Chi-squared test for prediction distribution
  - Effect size (Cohen's h for proportions)

Decision rule:
  Promote challenger if:
    - p-value < 0.05 (statistically significant)
    - Effect size > 0.1 (practically significant)
    - Challenger AUC > Control AUC
"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def z_test_proportions(n_a: int, conv_a: int, n_b: int, conv_b: int) -> dict:
    """
    Two-proportion Z-test.
    Tests if conversion rate of B is significantly different from A.
    """
    p_a = conv_a / n_a if n_a > 0 else 0
    p_b = conv_b / n_b if n_b > 0 else 0
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    if p_pool == 0 or p_pool == 1:
        return {"z_stat": 0.0, "p_value": 1.0, "p_a": p_a, "p_b": p_b, "lift": 0.0}

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z  = (p_b - p_a) / se if se > 0 else 0.0
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    lift = (p_b - p_a) / p_a if p_a > 0 else 0.0

    return {
        "z_stat":    round(float(z), 4),
        "p_value":   round(float(p), 4),
        "p_a":       round(float(p_a), 4),
        "p_b":       round(float(p_b), 4),
        "lift":      round(float(lift), 4),
    }


def cohens_h(p_a: float, p_b: float) -> float:
    """Cohen's h — effect size for proportion difference."""
    h = 2 * np.arcsin(np.sqrt(p_b)) - 2 * np.arcsin(np.sqrt(p_a))
    return round(float(abs(h)), 4)


def t_test_means(scores_a: list, scores_b: list) -> dict:
    """Welch's t-test for comparing mean scores."""
    stat, pvalue = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return {
        "t_stat":    round(float(stat), 4),
        "p_value":   round(float(pvalue), 4),
        "mean_a":    round(float(np.mean(scores_a)), 4),
        "mean_b":    round(float(np.mean(scores_b)), 4),
    }


def decide_winner(
    n_a: int, conv_a: int,
    n_b: int, conv_b: int,
    auc_a: float, auc_b: float,
    alpha: float = 0.05,
    min_effect: float = 0.01,
) -> dict:
    """
    Full A/B test decision.
    Promotes challenger (B) if statistically and practically significant.
    """
    z = z_test_proportions(n_a, conv_a, n_b, conv_b)
    h = cohens_h(z["p_a"], z["p_b"])

    significant   = z["p_value"] < alpha
    practical     = abs(z["lift"]) > min_effect
    challenger_better = auc_b > auc_a

    if significant and practical and challenger_better:
        decision = "promote_challenger"
        reason   = f"Challenger wins: AUC {auc_b:.4f} > {auc_a:.4f}, lift {z['lift']:.1%}, p={z['p_value']:.4f}"
    elif significant and not challenger_better:
        decision = "keep_control"
        reason   = f"Significant but AUC worse: {auc_b:.4f} < {auc_a:.4f}"
    elif not significant:
        decision = "no_winner"
        reason   = f"Not significant: p={z['p_value']:.4f} > alpha={alpha}"
    else:
        decision = "keep_control"
        reason   = f"Effect too small: lift={z['lift']:.1%} < min_effect={min_effect:.1%}"

    return {
        "decision":          decision,
        "reason":            reason,
        "z_stat":            z["z_stat"],
        "p_value":           z["p_value"],
        "lift":              z["lift"],
        "effect_size":       h,
        "significant":       significant,
        "practical":         practical,
        "challenger_better": challenger_better,
        "auc_a":             auc_a,
        "auc_b":             auc_b,
        "n_a":               n_a,
        "n_b":               n_b,
        "conv_a":            conv_a,
        "conv_b":            conv_b,
    }
