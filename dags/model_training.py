"""Airflow DAG — Model Training Pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "retailai",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "model_training",
    default_args=default_args,
    description="RetailAI Model Training Pipeline",
    schedule_interval="@weekly",
    catchup=False,
    tags=["retailai", "ml"],
)


def train_churn(**kwargs):
    from src.models.churn.train import run_churn
    run_churn()
    return True


def train_clv(**kwargs):
    from src.models.clv.predict import run_clv
    run_clv()
    return True


def run_forecast(**kwargs):
    from src.models.forecasting.revenue_forecast import run_forecasting
    run_forecasting()
    return True


def run_recs(**kwargs):
    from src.models.recommendation.recommend import run_recommendations
    run_recommendations()
    return True


with dag:
    churn = PythonOperator(task_id="train_churn_model", python_callable=train_churn)
    clv = PythonOperator(task_id="train_clv_model", python_callable=train_clv)
    forecast = PythonOperator(task_id="run_forecasting", python_callable=run_forecast)
    recs = PythonOperator(task_id="run_recommendations", python_callable=run_recs)

    [churn, clv, forecast, recs]
