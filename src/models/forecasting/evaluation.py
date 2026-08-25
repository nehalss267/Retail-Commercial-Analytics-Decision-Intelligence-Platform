"""Forecasting Evaluation — Model comparison metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute forecast accuracy metrics."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    wape = np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual)) + 1e-8) * 100
    r2 = 1 - np.sum((actual - predicted) ** 2) / (np.sum((actual - np.mean(actual)) ** 2) + 1e-8)

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "wape": round(float(wape), 2),
        "r2": round(float(r2), 4),
        "n_points": len(actual),
    }


def compare_models(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare multiple forecast models."""
    results = []
    for name, pred in predictions.items():
        metrics = evaluate_forecast(actual, pred)
        metrics["model"] = name
        results.append(metrics)
    return pd.DataFrame(results).set_index("model")


def backtest(series: pd.Series, model_fn, n_splits: int = 3, test_size: int = 30) -> dict:
    """Time-series cross-validation backtest."""
    results = []
    n = len(series)

    for i in range(n_splits):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start

        if test_start <= 0:
            break

        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]

        predictions = model_fn(train, len(test))
        metrics = evaluate_forecast(test.values, predictions)
        metrics["split"] = i + 1
        results.append(metrics)

    return {
        "n_splits": len(results),
        "avg_mae": round(np.mean([r["mae"] for r in results]), 2),
        "avg_rmse": round(np.mean([r["rmse"] for r in results]), 2),
        "avg_mape": round(np.mean([r["mape"] for r in results]), 2),
        "splits": results,
    }
