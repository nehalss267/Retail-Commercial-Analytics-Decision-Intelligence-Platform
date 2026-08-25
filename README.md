# RetailAI — Decision Intelligence Platform

AI-powered commercial decision intelligence platform built on the [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) dataset (~£8.9M revenue, 4,338 customers, 3,665 products, 37 countries). It turns raw transaction data into answers: *who will churn, what revenue to expect, which customers to target, and why.*

## What It Does

| Capability | Approach | Result |
|-----------|----------|--------|
| **Churn prediction** | XGBoost vs Logistic Regression vs Random Forest, SHAP explainability | ~0.97 AUC-ROC |
| **Customer segmentation** | RFM + K-Means (silhouette-selected K) | Meaningful segments (Champion, At Risk, Loyal, ...) |
| **Revenue forecasting** | XGBoost w/ lag+rolling features vs naive & moving-average baselines | 30-day forecast ≈ £1.26M |
| **Recommendations** | Popularity, content-based, collaborative filtering | Precision@5 = 1.00, NDCG@5 = 0.93 |
| **CLV estimation** | XGBoost regressor on customer features | Feeds campaign targeting |
| **Campaign optimization** | SciPy LP under budget/discount constraints | Expected ROI ≈ 68% |
| **Statistical rigor** | Welch's t-test, ANOVA, Kruskal-Wallis, CIs, power analysis | All key differences significant (p < 0.05) |
| **Causal inference** | Difference-in-differences, propensity matching | Treatment effect isolation |
| **AI Copilot** | LangGraph agent + 22 tools + RAG (ChromaDB) | Natural-language analytics |

## Architecture

```
Raw Data (UCI xlsx)
      │
      ▼
Data Cleaning ──► Feature Store (RFM, CLV, product, temporal)
      │                    │
      ▼                    ▼
SQL Analytics      Model Training (churn, CLV, segments,
(parquet views)    forecast, recommenders, optimization)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        FastAPI API   Streamlit     LangGraph Agent
        (15 routes)   Dashboard      ├─ 22 tools
                      (9 pages)      ├─ RAG knowledge base
                                     └─ keyword router
```

## Quick Start

**Requirements:** Python 3.11+

```bash
# Install
pip install -e ".[dev]"

# Place the dataset at data/raw/Online Retail.xlsx

# Launch the dashboard
python -m streamlit run dashboard/app.py          # http://localhost:8501

# Or launch the API
python -m uvicorn src.api.main:app --reload --port 8000   # http://localhost:8000/docs
```

> On Windows/Git Bash, use the `python -m` prefix if bare commands aren't on PATH.

### Optional Infrastructure

```bash
make db-up      # PostgreSQL + pgAdmin via Docker Compose
make mlflow     # MLflow tracking server on :5000
```

## Regenerating Outputs

```bash
python src/preprocessing/cleaning.py        # clean raw transactions
python src/features/feature_pipeline.py    # build feature store
python src/models/churn/train.py           # churn models + report
python src/models/churn/explain.py         # SHAP explanations
python src/models/clv/train.py             # CLV model
python src/models/segmentation/clustering.py
python src/models/forecasting/xgboost.py   # 30-day forecast
python src/models/recommendation/*.py      # recommenders
python src/optimization/scenarios.py       # campaign optimization
python src/ai/rag/ingestion.py             # rebuild vector KB
```

## AI Copilot

A LangGraph `StateGraph` routes business questions to tool categories, executes tools against trained models/data, and synthesizes grounded answers — numbers come from tools, never fabricated.

```python
from src.ai.agent import process_query

result = process_query("Which customers should we target?")
print(result["route"])   # "optimization"
print(result["answer"])
```

22 tools across 7 domains: SQL analytics, churn, forecasting, experiments, causal, recommendations, optimization. RAG adds methodology context from an 84-chunk ChromaDB index.

## Dashboard Pages

Executive Overview · Customer Intelligence · Product Analytics · Churn & CLV · Forecasting · Experimentation · Recommendations · Optimization · AI Copilot

## Testing

```bash
make test        # or: python -m pytest tests/ -v
```

73 passing tests across unit (models, features, statistics), integration (end-to-end pipeline, agent routing), and API suites.

## Notebooks

`notebooks/01`–`07`: data quality, EDA, customer analytics, statistics, churn, forecasting, recommendations.

## Tech Stack

Python 3.11 · pandas · scikit-learn · XGBoost · SHAP · Prophet · SciPy/statsmodels · FastAPI · Streamlit · Plotly · LangChain/LangGraph · ChromaDB · sentence-transformers · MLflow · Airflow · PostgreSQL · Docker Compose

## Notes

- Synthetic scenario data is clearly labeled and used only where no trained-model output exists.
- The Copilot runs without an LLM key (keyword-based routing); connect Ollama on `:11434` for LLM synthesis.
