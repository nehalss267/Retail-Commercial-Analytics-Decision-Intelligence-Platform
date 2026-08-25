"""Prophet Forecasting — Facebook's time series model."""
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def prophet_forecast(series: pd.Series, horizon: int,
                     yearly: bool = True, weekly: bool = True) -> dict:
    """Prophet forecast with uncertainty intervals."""
    from prophet import Prophet

    df = pd.DataFrame({"ds": series.index, "y": series.values})

    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=False,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    result = forecast.tail(horizon)

    return {
        "predictions": [round(float(x), 2) for x in result["yhat"]],
        "ci_lower": [round(float(x), 2) for x in result["yhat_lower"]],
        "ci_upper": [round(float(x), 2) for x in result["yhat_upper"]],
        "trend": [round(float(x), 2) for x in result["trend"]],
    }
