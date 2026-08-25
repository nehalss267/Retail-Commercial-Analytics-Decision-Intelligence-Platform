"""Churn Model Evaluation — Comprehensive metrics and calibration."""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.calibration import calibration_curve

from src.models.churn.train import load_data, define_churn, train_churn_models
from src.models.churn.predict import get_feature_columns


def evaluate_models(data: pd.DataFrame) -> dict:
    """Full evaluation of churn models with calibration metrics."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_cols = get_feature_columns()
    X = data[feature_cols].fillna(0)
    y = data["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), False),
        "XGBoost": (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  random_state=42, eval_metric="logloss"), False),
    }

    results = {}
    for name, (model, needs_scaling) in models.items():
        X_tr = scaler.transform(X_train) if needs_scaling else X_train
        X_te = scaler.transform(X_test) if needs_scaling else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calibration
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
            "brier_score": round(float(brier_score_loss(y_test, y_proba)), 4),
            "log_loss": round(float(log_loss(y_test, y_proba)), 4),
            "precision": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]), 4),
            "recall": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]), 4),
            "f1": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"]), 4),
            "accuracy": round(float(classification_report(y_test, y_pred, output_dict=True)["accuracy"]), 4),
            "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            "calibration": {
                "prob_true": [round(float(x), 4) for x in prob_true],
                "prob_pred": [round(float(x), 4) for x in prob_pred],
            },
        }

    return results
