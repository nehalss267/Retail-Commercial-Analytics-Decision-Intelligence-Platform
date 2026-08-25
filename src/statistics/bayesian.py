"""Bayesian Analysis — Bayesian alternatives to classical tests."""
import numpy as np
from scipy import stats


def bayesian_t_test(group1: np.ndarray, group2: np.ndarray,
                    n_samples: int = 10000) -> dict:
    """Bayesian t-test using Monte Carlo simulation.

    Compares posterior distributions of two group means.
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Prior: non-informative
    # Posterior for each group mean (conjugate normal)
    prior_var = 1e6

    mean1, var1, n1 = group1.mean(), group1.var(), len(group1)
    mean2, var2, n2 = group2.mean(), group2.var(), len(group2)

    # Posterior variance
    post_var1 = 1.0 / (1.0 / prior_var + n1 / var1) if var1 > 0 else prior_var
    post_var2 = 1.0 / (1.0 / prior_var + n2 / var2) if var2 > 0 else prior_var

    post_mean1 = post_var1 * (n1 * mean1 / var1) if var1 > 0 else mean1
    post_mean2 = post_var2 * (n2 * mean2 / var2) if var2 > 0 else mean2

    # Sample from posteriors
    samples1 = np.random.normal(post_mean1, np.sqrt(post_var1), n_samples)
    samples2 = np.random.normal(post_mean2, np.sqrt(post_var2), n_samples)

    diff_samples = samples1 - samples2

    # Probability that group1 > group2
    prob_greater = float(np.mean(samples1 > samples2))

    return {
        "mean_difference": round(float(diff_samples.mean()), 4),
        "ci_95_lower": round(float(np.percentile(diff_samples, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(diff_samples, 97.5)), 4),
        "prob_group1_greater": round(prob_greater, 4),
        "prob_group2_greater": round(1 - prob_greater, 4),
        "n_samples": n_samples,
    }


def bayesian_ab_test(control: np.ndarray, treatment: np.ndarray,
                     n_samples: int = 10000) -> dict:
    """Bayesian A/B test — probability of improvement."""
    control = np.array(control)
    treatment = np.array(treatment)

    # Posterior parameters (normal-inverse-gamma conjugate)
    prior_mean = 0
    prior_n = 1

    c_mean, c_var, c_n = control.mean(), control.var(), len(control)
    t_mean, t_var, t_n = treatment.mean(), treatment.var(), len(treatment)

    c_post_var = 1.0 / (prior_n + c_n / c_var) if c_var > 0 else 1.0
    t_post_var = 1.0 / (prior_n + t_n / t_var) if t_var > 0 else 1.0

    c_post_mean = c_post_var * (prior_n * prior_mean + c_n * c_mean / c_var) if c_var > 0 else c_mean
    t_post_mean = t_post_var * (prior_n * prior_mean + t_n * t_mean / t_var) if t_var > 0 else t_mean

    c_samples = np.random.normal(c_post_mean, np.sqrt(c_post_var), n_samples)
    t_samples = np.random.normal(t_post_mean, np.sqrt(t_post_var), n_samples)

    uplift = t_samples - c_samples
    prob_better = float(np.mean(uplift > 0))

    return {
        "prob_treatment_better": round(prob_better, 4),
        "expected_uplift": round(float(uplift.mean()), 4),
        "uplift_ci_95_lower": round(float(np.percentile(uplift, 2.5)), 4),
        "uplift_ci_95_upper": round(float(np.percentile(uplift, 97.5)), 4),
        "control_posterior_mean": round(float(c_post_mean), 4),
        "treatment_posterior_mean": round(float(t_post_mean), 4),
    }
