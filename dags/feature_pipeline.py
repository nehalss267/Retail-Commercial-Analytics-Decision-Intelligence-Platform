"""Airflow DAG — Feature Engineering Pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "retailai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_customer_features():
    from src.features.customer_features import build_customer_features
    import pandas as pd
    from src.config import settings
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    features = build_customer_features(df)
    features.to_parquet(settings.FEATURES_DIR / "customer_features.parquet", index=False)

def run_product_features():
    from src.features.product_features import build_product_features
    import pandas as pd
    from src.config import settings
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    features = build_product_features(df)
    features.to_parquet(settings.FEATURES_DIR / "product_features.parquet", index=False)

def run_temporal_features():
    from src.features.temporal_features import build_temporal_features
    import pandas as pd
    from src.config import settings
    df = pd.read_parquet(settings.PROCESSED_DATA_DIR / "cleaned_retail.parquet")
    features = build_temporal_features(df)
    features.to_parquet(settings.FEATURES_DIR / "temporal_features.parquet", index=False)

def run_rfm_features():
    from src.features.rfm_features import run_segmentation
    run_segmentation()

def run_quality_checks():
    import pandas as pd
    from src.config import settings
    for f in ["customer_features", "product_features", "temporal_features", "rfm_features"]:
        path = settings.FEATURES_DIR / f"{f}.parquet"
        assert path.exists(), f"Missing: {f}"
        df = pd.read_parquet(path)
        assert len(df) > 0, f"Empty: {f}"

with DAG(
    "feature_pipeline",
    default_args=default_args,
    description="Feature engineering pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    t_customer = PythonOperator(task_id="customer_features", python_callable=run_customer_features)
    t_product = PythonOperator(task_id="product_features", python_callable=run_product_features)
    t_temporal = PythonOperator(task_id="temporal_features", python_callable=run_temporal_features)
    t_rfm = PythonOperator(task_id="rfm_features", python_callable=run_rfm_features)
    t_quality = PythonOperator(task_id="quality_checks", python_callable=run_quality_checks)

    [t_customer, t_product, t_temporal, t_rfm] >> t_quality
