"""Treatment Effect Estimation — Heterogeneous treatment effects."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def estimate_ate(outcome: np.ndarray, treatment: np.ndarray) -> dict:
    """Simple ATE estimation via outcome regression."""
    treated = outcome[treatment == 1]
    control = outcome[treatment == 0]

    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))

    from scipy import stats
    t_stat = ate / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(outcome) - 2))

    return {
        "method": "Direct Outcome Comparison",
        "ate": round(float(ate), 4),
        "ate_pct": round(float(ate / control.mean() * 100), 2) if control.mean() != 0 else 0,
        "ci_95_lower": round(float(ate - 1.96 * se), 4),
        "ci_95_upper": round(float(ate + 1.96 * se), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(f"{p_value:.4e}"),
        "significant": bool(p_value < 0.05),
        "n_treated": int(treatment.sum()),
        "n_control": int((treatment == 0).sum()),
    }


def heterogeneous_effects(data: pd.DataFrame, outcome_col: str,
                          treatment_col: str = "treatment",
                          feature_cols: list[str] | None = None) -> dict:
    """Estimate CATE (Conditional Average Treatment Effect) using T-learner."""
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c not in [outcome_col, treatment_col, "CustomerID"]]

    X = data[feature_cols].fillna(0)
    y = data[outcome_col]
    T = data[treatment_col]

    # T-learner
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    model_t.fit(X[T == 1], y[T == 1])
    model_c.fit(X[T == 0], y[T == 0])

    # Individual treatment effects
    te = model_t.predict(X) - model_c.predict(X)

    return {
        "method": "T-Learner (Gradient Boosting)",
        "mean_ate": round(float(te.mean()), 4),
        "median_te": round(float(np.median(te)), 4),
        "te_std": round(float(te.std()), 4),
        "te_ci_lower": round(float(np.percentile(te, 2.5)), 4),
        "te_ci_upper": round(float(np.percentile(te, 97.5)), 4),
        "n_positive_effect": int((te > 0).sum()),
        "n_negative_effect": int((te <= 0).sum()),
        "feature_cols": feature_cols,
    }
