"""Forecasting Baselines — Naive and Moving Average models."""
import pandas as pd


def naive_forecast(series: pd.Series, horizon: int) -> list[float]:
    """Last value carried forward."""
    return [float(series.iloc[-1])] * horizon


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> list[float]:
    """Moving average forecast."""
    ma = float(series.rolling(window).mean().iloc[-1])
    return [ma] * horizon


def seasonal_naive(series: pd.Series, horizon: int, period: int = 7) -> list[float]:
    """Repeat the last period's values."""
    last_period = series.iloc[-period:].values
    predictions = []
    for i in range(horizon):
        predictions.append(float(last_period[i % period]))
    return predictions
