"""CLV Model Training — Separate training module."""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import settings


def load_rfm() -> pd.DataFrame:
    """Load RFM features."""
    return pd.read_parquet(settings.FEATURES_DIR / "rfm_features.parquet")


def build_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build CLV model features and return feature columns."""
    features = rfm[[
        "CustomerID", "recency", "frequency", "monetary",
        "avg_order_value", "purchase_interval",
    ]].copy()

    features["monetary_log"] = np.log1p(features["monetary"])
    features["frequency_log"] = np.log1p(features["frequency"])
    features["reciprocal_recency"] = 1 / (features["recency"] + 1)

    feature_cols = [
        "recency", "frequency", "monetary", "avg_order_value",
        "purchase_interval", "monetary_log", "frequency_log", "reciprocal_recency",
    ]

    return features, feature_cols


def train_clv_model(features: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train CLV model and return metrics."""
    X = features[feature_cols].fillna(0)
    y = features["monetary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    importances = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))

    return {
        "model_type": "GradientBoostingRegressor",
        "train_r2": round(float(train_r2), 4),
        "test_r2": round(float(test_r2), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "feature_importance": importances,
    }
