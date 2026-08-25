"""Demand and Revenue Forecasting."""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import json
import warnings
warnings.filterwarnings("ignore")

from src.config import settings


def load_daily() -> pd.DataFrame:
    """Build daily revenue time series from cleaned data."""
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    valid = df[(df["Revenue"] > 0) & (df["HasCustomerID"])]

    daily = valid.groupby(valid["InvoiceDate"].dt.date).agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        units=("Quantity", "sum"),
    ).reset_index()
    daily.columns = ["date", "revenue", "orders", "customers", "units"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").set_index("date")

    # Fill missing dates
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).fillna(0)
    daily.index.name = "date"
    return daily


def naive_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """Last value carried forward."""
    return pd.Series([series.iloc[-1]] * horizon)


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> pd.Series:
    """Moving average forecast."""
    ma = series.rolling(window).mean().iloc[-1]
    return pd.Series([ma] * horizon)


def xgboost_forecast(series: pd.Series, horizon: int) -> dict:
    """XGBoost with lag/rolling features."""
    df = pd.DataFrame({"y": series})

    # Features
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df["y"].rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].rolling(window).std()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols]
    y = df["y"]

    # Train/test split (temporal)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-8))) * 100

    # Recursive forecast
    last_values = series.values.copy()
    predictions = []
    for _ in range(horizon):
        feats = {}
        for lag in [1, 2, 3, 7, 14, 28]:
            feats[f"lag_{lag}"] = last_values[-lag] if len(last_values) >= lag else last_values[-1]
        for window in [7, 14, 28]:
            feats[f"rolling_mean_{window}"] = np.mean(last_values[-window:])
            feats[f"rolling_std_{window}"] = np.std(last_values[-window:])
        feats["dayofweek"] = (len(series) + len(predictions)) % 7
        feats["month"] = 12  # approximate
        pred = model.predict(pd.DataFrame([feats]))[0]
        predictions.append(max(pred, 0))
        last_values = np.append(last_values, pred)

    return {
        "predictions": predictions,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
    }


def run_forecasting() -> dict:
    """Run forecasting pipeline."""
    print("Loading daily data...")
    daily = load_daily()
    print(f"  {len(daily)} days, revenue range: {daily['revenue'].min():.0f} - {daily['revenue'].max():.0f}")

    horizon = 30  # Forecast next 30 days

    print("\n1. Naive baseline...")
    naive_pred = naive_forecast(daily["revenue"], horizon)

    print("2. Moving average...")
    ma_pred = moving_average_forecast(daily["revenue"], horizon)

    print("3. XGBoost forecast...")
    xgb_result = xgboost_forecast(daily["revenue"], horizon)

    # Build forecast DataFrame
    future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecasts = pd.DataFrame({
        "date": future_dates,
        "naive": naive_pred.values,
        "moving_avg_7d": ma_pred.values,
        "xgboost": xgb_result["predictions"],
    })

    forecasts.to_csv(settings.PROCESSED_DATA_DIR / "forecast_30d.csv", index=False)

    results = {
        "horizon_days": horizon,
        "training_days": len(daily),
        "naive": {"description": "Last value carried forward"},
        "moving_average": {"window": 7, "description": "7-day rolling mean"},
        "xgboost": {
            "mae": xgb_result["mae"],
            "rmse": xgb_result["rmse"],
            "mape": xgb_result["mape"],
        },
        "forecast_summary": {
            "naive_last_value": round(float(daily["revenue"].iloc[-1]), 2),
            "xgboost_avg_forecast": round(float(np.mean(xgb_result["predictions"])), 2),
            "xgboost_total_forecast": round(float(np.sum(xgb_result["predictions"])), 2),
        },
    }

    with open(settings.PROCESSED_DATA_DIR / "forecast_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== FORECAST RESULTS ===")
    print(f"XGBoost MAE: {xgb_result['mae']}")
    print(f"XGBoost RMSE: {xgb_result['rmse']}")
    print(f"XGBoost MAPE: {xgb_result['mape']}%")
    print(f"30-day forecast total: £{results['forecast_summary']['xgboost_total_forecast']:,.2f}")

    return results


if __name__ == "__main__":
    run_forecasting()
