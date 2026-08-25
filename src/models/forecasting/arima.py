"""ARIMA/SARIMA Forecasting — Statistical time series models."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def arima_forecast(series: pd.Series, horizon: int,
                   order: tuple = (1, 1, 1)) -> dict:
    """ARIMA forecast using statsmodels."""
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=horizon)
    ci = fitted.get_forecast(steps=horizon).conf_int(alpha=0.05)

    return {
        "predictions": [round(float(x), 2) for x in forecast],
        "ci_lower": [round(float(x), 2) for x in ci.iloc[:, 0]],
        "ci_upper": [round(float(x), 2) for x in ci.iloc[:, 1]],
        "aic": round(float(fitted.aic), 2),
        "bic": round(float(fitted.bic), 2),
        "order": list(order),
    }


def sarima_forecast(series: pd.Series, horizon: int,
                    order: tuple = (1, 1, 1),
                    seasonal_order: tuple = (1, 1, 1, 7)) -> dict:
    """SARIMA forecast for weekly seasonality."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    forecast = fitted.forecast(steps=horizon)
    ci = fitted.get_forecast(steps=horizon).conf_int(alpha=0.05)

    return {
        "predictions": [round(float(x), 2) for x in forecast],
        "ci_lower": [round(float(x), 2) for x in ci.iloc[:, 0]],
        "ci_upper": [round(float(x), 2) for x in ci.iloc[:, 1]],
        "aic": round(float(fitted.aic), 2),
        "bic": round(float(fitted.bic), 2),
    }
