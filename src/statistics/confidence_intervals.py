"""Confidence Intervals — Standalone CI calculations."""
import numpy as np
from scipy import stats


def mean_ci(data: np.ndarray, confidence: float = 0.95) -> dict:
    """Confidence interval for a mean."""
    data = np.array(data)
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)
    return {
        "mean": round(float(mean), 4),
        "ci_lower": round(float(ci[0]), 4),
        "ci_upper": round(float(ci[1]), 4),
        "std": round(float(data.std()), 4),
        "se": round(float(se), 4),
        "n": n,
        "confidence": confidence,
    }


def proportion_ci(successes: int, n: int, confidence: float = 0.95) -> dict:
    """Confidence interval for a proportion (Wilson score)."""
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return {
        "proportion": round(float(p), 4),
        "ci_lower": round(float(center - margin), 4),
        "ci_upper": round(float(center + margin), 4),
        "n": n,
        "successes": successes,
        "confidence": confidence,
    }


def difference_ci(mean1: float, mean2: float, var1: float, var2: float,
                  n1: int, n2: int, confidence: float = 0.95) -> dict:
    """Confidence interval for difference of two means."""
    diff = mean1 - mean2
    se = np.sqrt(var1 / n1 + var2 / n2)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    return {
        "difference": round(float(diff), 4),
        "ci_lower": round(float(diff - z * se), 4),
        "ci_upper": round(float(diff + z * se), 4),
        "se": round(float(se), 4),
        "confidence": confidence,
    }
