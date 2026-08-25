"""Churn Model Explainability — SHAP-based explanations."""
import pandas as pd
import numpy as np
import json

from src.config import settings
from src.models.churn.train import load_data, define_churn
from src.models.churn.predict import get_feature_columns


def explain_with_shap(data: pd.DataFrame, n_samples: int = 100) -> dict:
    """Generate SHAP explanations for churn model."""
    import shap
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    feature_cols = get_feature_columns()
    X = data[feature_cols].fillna(0)
    y = data["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Global feature importance (mean |SHAP|)
    global_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Top features per prediction
    top_factors_per_customer = []
    for i in range(min(5, len(X_sample))):
        customer_shap = pd.Series(shap_values[i], index=feature_cols)
        top = customer_shap.abs().nlargest(3)
        top_factors_per_customer.append({
            "customer_index": int(X_sample.index[i]),
            "top_factors": [
                {"feature": feat, "shap_value": round(float(customer_shap[feat]), 4)}
                for feat in top.index
            ],
        })

    return {
        "global_feature_importance": global_importance.to_dict(orient="records"),
        "top_factors_examples": top_factors_per_customer,
        "model_version": "xgboost-shap",
        "n_samples_explained": len(X_sample),
    }


def run_explainability() -> dict:
    """Run SHAP explainability pipeline."""
    print("Loading data...")
    df, rfm = load_data()
    data = define_churn(df, rfm)

    print("Running SHAP explanations...")
    results = explain_with_shap(data)

    with open(settings.PROCESSED_DATA_DIR / "churn_explainability.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== TOP GLOBAL FEATURES ===")
    for row in results["global_feature_importance"][:5]:
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    return results


if __name__ == "__main__":
    run_explainability()
