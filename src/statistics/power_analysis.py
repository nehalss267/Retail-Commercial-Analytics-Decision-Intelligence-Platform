"""Power Analysis — Statistical power and sample size calculations."""
import numpy as np
from scipy import stats


def power_two_sample(mean1: float, mean2: float, std1: float, std2: float,
                     n1: int, n2: int, alpha: float = 0.05) -> dict:
    """Compute statistical power for two-sample t-test."""
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    diff = abs(mean1 - mean2)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = (diff / se) - z_alpha
    power = stats.norm.cdf(z_beta)

    return {
        "power": round(float(power), 4),
        "alpha": alpha,
        "effect_size": round(float(diff / np.sqrt((std1**2 + std2**2) / 2)), 4),
        "n1": n1,
        "n2": n2,
    }


def required_sample_size(effect_size: float, power: float = 0.8,
                         alpha: float = 0.05) -> dict:
    """Required sample size per group for two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if effect_size <= 0:
        return {"n_per_group": 0, "effect_size": effect_size}

    n = ((z_alpha + z_beta) / effect_size) ** 2

    return {
        "n_per_group": int(np.ceil(n)),
        "effect_size": round(float(effect_size), 4),
        "power": power,
        "alpha": alpha,
    }


def power_curve(effect_sizes: list[float], n: int, alpha: float = 0.05) -> list[dict]:
    """Compute power across a range of effect sizes."""
    results = []
    for d in effect_sizes:
        p = power_two_sample(0, d, 1, 1, n, n, alpha)
        results.append({"effect_size": round(d, 3), "power": p["power"]})
    return results
