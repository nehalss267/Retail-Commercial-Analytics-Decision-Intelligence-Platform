"""XGBoost Forecasting — ML-based time series with lag features."""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def create_features(series: pd.Series) -> pd.DataFrame:
    """Create lag and rolling features for time series."""
    df = pd.DataFrame({"y": series})

    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = df["y"].rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].rolling(window).std()
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear

    return df.dropna()


def xgboost_forecast(series: pd.Series, horizon: int,
                     n_estimators: int = 200, max_depth: int = 4) -> dict:
    """XGBoost forecast with lag/rolling features."""
    df = create_features(series)
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols]
    y = df["y"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
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
        feats["month"] = 12
        feats["dayofyear"] = min(365, len(series) + len(predictions))
        pred = model.predict(pd.DataFrame([feats]))[0]
        predictions.append(max(float(pred), 0))
        last_values = np.append(last_values, pred)

    importances = dict(zip(feature_cols, [round(float(x), 4) for x in model.feature_importances_]))

    return {
        "predictions": predictions,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "feature_importance": importances,
    }
