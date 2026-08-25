"""Metrics Utilities — Common metric calculations."""
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score,
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute regression metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mape": round(float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100), 4),
        "n": len(actual),
    }


def classification_metrics(actual: np.ndarray, predicted: np.ndarray,
                           predicted_proba: np.ndarray | None = None) -> dict:
    """Compute classification metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    metrics = {
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "n": len(actual),
    }

    if predicted_proba is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(actual, predicted_proba)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(actual, predicted_proba)), 4)
        except ValueError:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics


def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return round(float((group1.mean() - group2.mean()) / pooled_std), 4)
