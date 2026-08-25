"""Airflow DAG — RetailAI ETL Pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "retailai",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "retail_etl",
    default_args=default_args,
    description="RetailAI ETL Pipeline",
    schedule_interval="@daily",
    catchup=False,
    tags=["retailai", "etl"],
)


def extract_data(**kwargs):
    from src.ingestion.uci_loader import load_raw_data
    df = load_raw_data()
    kwargs["ti"].xcom_push(key="row_count", value=len(df))
    return len(df)


def validate_data(**kwargs):
    from src.preprocessing.quality import load_raw
    df = load_raw()
    assert len(df) > 0, "No data loaded"
    assert df.shape[1] == 8, f"Expected 8 columns, got {df.shape[1]}"
    return True


def clean_data(**kwargs):
    from src.preprocessing.cleaning import clean_pipeline
    df = clean_pipeline()
    kwargs["ti"].xcom_push(key="cleaned_rows", value=len(df))
    return len(df)


def build_features(**kwargs):
    from src.features.rfm_features import run_segmentation
    run_segmentation()
    return True


def run_quality_checks(**kwargs):
    from src.preprocessing.quality import run_full_analysis
    from src.config import settings
    report = run_full_analysis(settings.PROCESSED_DATA_DIR)
    return True


with dag:
    extract = PythonOperator(task_id="extract_uci", python_callable=extract_data)
    validate = PythonOperator(task_id="validate_data", python_callable=validate_data)
    clean = PythonOperator(task_id="clean_data", python_callable=clean_data)
    features = PythonOperator(task_id="build_features", python_callable=build_features)
    quality = PythonOperator(task_id="run_quality_checks", python_callable=run_quality_checks)

    extract >> validate >> clean >> features >> quality
