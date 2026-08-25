# RetailAI — Architecture

## 1. Project Overview

**RetailAI** is an AI-powered commercial decision intelligence platform built using the UCI Online Retail dataset.

The system transforms transactional retail data into actionable business decisions.

### Core question

> **What happened → Why did it happen → What will happen → What should we do?**

The architecture is intentionally designed to demonstrate the complete lifecycle expected from a modern Data Scientist:

- Data ingestion
- Data quality
- SQL analytics
- Statistical analysis
- Machine learning
- Forecasting
- Causal inference
- Experimentation
- Recommendation systems
- Prescriptive analytics
- Generative AI
- Data engineering
- MLOps
- Business visualization

---

# 2. Architecture Principles

## 2.1 Business-first design

Every analytical component must answer a business question.

Models are not built merely to demonstrate algorithms.

## 2.2 Reproducibility

The same raw input should produce the same analytical dataset and model artifacts when the pipeline configuration is unchanged.

## 2.3 Separation of concerns

Separate:

- Data ingestion
- Transformation
- Feature engineering
- Analytics
- Modeling
- Decision logic
- API
- UI
- AI agent

## 2.4 Explainability

Business-facing predictions should include explanations where practical.

Use:

- SHAP
- feature importance
- confidence intervals
- treatment effects
- forecast intervals
- model comparison

## 2.5 Real data vs simulated scenarios

The UCI Online Retail dataset is the historical source.

Variables that do not exist in UCI must be clearly identified as derived or simulated.

Never imply that synthetic promotion/experiment variables were observed historically.

---

# 3. High-Level Architecture

```text
                         ┌───────────────────────────┐
                         │      Business User        │
                         │                           │
                         │  Streamlit Dashboard      │
                         │  AI Business Copilot      │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │         FastAPI           │
                         │      Application API      │
                         └─────────────┬─────────────┘
                                       │
             ┌─────────────────────────┼──────────────────────────┐
             │                         │                          │
             ▼                         ▼                          ▼
   ┌──────────────────┐      ┌──────────────────┐       ┌──────────────────┐
   │ Analytics Layer  │      │   ML Platform    │       │  AI Agent Layer  │
   │                  │      │                  │       │                  │
   │ SQL Analytics    │      │ Churn            │       │ LangGraph        │
   │ KPI Calculation  │      │ CLV              │       │ Tool Calling     │
   │ Cohorts          │      │ Segmentation     │       │ RAG              │
   │ RFM              │      │ Forecasting      │       │ LLM              │
   └────────┬─────────┘      │ Recommendation   │       └────────┬─────────┘
            │                │ Causal           │                │
            │                │ Optimization     │                │
            │                └────────┬─────────┘                │
            │                         │                          │
            └─────────────────────────┼──────────────────────────┘
                                      ▼
                         ┌───────────────────────────┐
                         │       Data Layer          │
                         │                           │
                          │ PostgreSQL (Docker)       │
                         │ Feature Tables            │
                         │ Analytical Views           │
                         └─────────────┬─────────────┘
                                       ▲
                                       │
                         ┌─────────────┴─────────────┐
                         │       Airflow DAGs         │
                         │                           │
                         │ Ingestion → Validation    │
                         │ → Transform → Features    │
                         │ → Training                │
                         └─────────────┬─────────────┘
                                       ▲
                                       │
                         ┌─────────────┴─────────────┐
                         │       Data Sources         │
                         │                           │
                         │ UCI Online Retail         │
                         │ Derived Features           │
                         │ Scenario / Experiment Data│
                         └───────────────────────────┘
```

---

# 4. Complete Folder Structure

```text
retailai/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── docker-compose.yml
├── Makefile
├── pyproject.toml
│
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── business_questions.md
│   └── diagrams/
│
├── data/
│   ├── raw/
│   │   └── online_retail.xlsx
│   │
│   ├── processed/
│   │
│   ├── features/
│   │
│   └── samples/
│
├── sql/
│   ├── schema/
│   │   ├── 001_extensions.sql
│   │   ├── 002_tables.sql
│   │   └── 003_indexes.sql
│   │
│   ├── transformations/
│   │   ├── customers.sql
│   │   ├── orders.sql
│   │   └── products.sql
│   │
│   ├── analytics/
│   │   ├── revenue.sql
│   │   ├── customer_metrics.sql
│   │   ├── cohort_analysis.sql
│   │   ├── retention.sql
│   │   └── product_analysis.sql
│   │
│   └── views/
│
├── src/
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ingestion/
│   │   ├── uci_loader.py
│   │   └── validators.py
│   │
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   ├── transformations.py
│   │   └── quality.py
│   │
│   ├── features/
│   │   ├── customer_features.py
│   │   ├── product_features.py
│   │   ├── temporal_features.py
│   │   ├── rfm_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── analytics/
│   │   ├── kpis.py
│   │   ├── cohort.py
│   │   ├── retention.py
│   │   ├── rfm.py
│   │   └── business_metrics.py
│   │
│   ├── statistics/
│   │   ├── hypothesis_tests.py
│   │   ├── confidence_intervals.py
│   │   ├── anova.py
│   │   ├── power_analysis.py
│   │   └── bayesian.py
│   │
│   ├── experimentation/
│   │   ├── experiment_design.py
│   │   ├── sample_size.py
│   │   ├── ab_test.py
│   │   └── uplift.py
│   │
│   ├── causal/
│   │   ├── propensity.py
│   │   ├── matching.py
│   │   ├── treatment_effect.py
│   │   └── difference_in_differences.py
│   │
│   ├── models/
│   │   │
│   │   ├── churn/
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   ├── evaluate.py
│   │   │   └── explain.py
│   │   │
│   │   ├── clv/
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   │
│   │   ├── segmentation/
│   │   │   ├── clustering.py
│   │   │   └── evaluation.py
│   │   │
│   │   ├── forecasting/
│   │   │   ├── baselines.py
│   │   │   ├── arima.py
│   │   │   ├── prophet.py
│   │   │   ├── xgboost.py
│   │   │   └── evaluation.py
│   │   │
│   │   └── recommendation/
│   │       ├── popularity.py
│   │       ├── collaborative.py
│   │       ├── content_based.py
│   │       └── evaluation.py
│   │
│   ├── optimization/
│   │   ├── objectives.py
│   │   ├── constraints.py
│   │   ├── targeting.py
│   │   └── scenarios.py
│   │
│   ├── ai/
│   │   ├── agent.py
│   │   ├── graph.py
│   │   ├── tools/
│   │   │   ├── sql_tool.py
│   │   │   ├── churn_tool.py
│   │   │   ├── forecast_tool.py
│   │   │   ├── experiment_tool.py
│   │   │   ├── causal_tool.py
│   │   │   ├── recommendation_tool.py
│   │   │   └── optimization_tool.py
│   │   │
│   │   ├── rag/
│   │   │   ├── ingestion.py
│   │   │   ├── embeddings.py
│   │   │   ├── retrieval.py
│   │   │   └── prompts.py
│   │   │
│   │   └── prompts/
│   │       └── business_copilot.txt
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── customers.py
│   │       ├── predictions.py
│   │       ├── forecasting.py
│   │       ├── experimentation.py
│   │       ├── recommendations.py
│   │       ├── optimization.py
│   │       └── agent.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── metrics.py
│       └── io.py
│
├── dags/
│   ├── retail_etl.py
│   ├── feature_pipeline.py
│   └── model_training.py
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   │   ├── executive.py
│   │   ├── customers.py
│   │   ├── products.py
│   │   ├── churn.py
│   │   ├── forecasting.py
│   │   ├── experimentation.py
│   │   ├── recommendations.py
│   │   ├── optimization.py
│   │   └── copilot.py
│   │
│   └── components/
│       ├── charts.py
│       ├── cards.py
│       └── tables.py
│
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_customer_analytics.ipynb
│   ├── 04_statistics.ipynb
│   ├── 05_churn.ipynb
│   ├── 06_forecasting.ipynb
│   └── 07_recommendations.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
│
└── mlruns/
```

---

# 5. Data Architecture

## 5.1 Source

Primary source:

**UCI Online Retail**

The raw source is stored unchanged in:

```text
data/raw/
```

Never modify raw files.

---

# 6. Data Processing Pipeline

```text
UCI Excel
    │
    ▼
Raw Data
    │
    ▼
Schema Validation
    │
    ├── Invalid rows
    ├── Missing values
    ├── Duplicates
    ├── Returns
    └── Invalid prices/quantities
    │
    ▼
Cleaning
    │
    ▼
Canonical Transactions
    │
    ▼
Warehouse
    │
    ├── Customer features
    ├── Product features
    ├── Temporal features
    └── Analytical aggregates
```

---

# 7. Warehouse Model

## 7.1 Customers

```text
customers
----------
customer_id PK
country
first_purchase_date
last_purchase_date
customer_lifetime_days
```

## 7.2 Products

```text
products
--------
product_id PK
stock_code
description
```

## 7.3 Orders

```text
orders
------
order_id PK
customer_id FK
invoice_date
country
is_cancelled
```

## 7.4 Order Items

```text
order_items
-----------
order_item_id PK
order_id FK
product_id FK
quantity
unit_price
revenue
```

Revenue:

```text
revenue = quantity * unit_price
```

with return/cancellation handling explicitly documented.

---

# 8. Analytical Layer

Create derived tables/views.

### Customer metrics

```text
customer_metrics
----------------
customer_id
recency_days
frequency
monetary_value
average_order_value
purchase_frequency
total_orders
total_items
return_rate
active_months
```

### Product metrics

```text
product_metrics
---------------
product_id
orders
units_sold
revenue
average_price
unique_customers
repeat_purchase_rate
```

### Temporal metrics

```text
daily_metrics
-------------
date
orders
customers
units
revenue
average_order_value
```

---

# 9. Feature Engineering

Feature engineering must be implemented as reusable Python modules rather than only notebook code.

## Customer features

Examples:

- Recency
- Frequency
- Monetary value
- Average order value
- Purchase interval
- Number of active months
- Product diversity
- Country
- Return behavior

## Temporal features

- Day
- Week
- Month
- Quarter
- Year
- Day of week
- Month of year
- Lagged revenue
- Rolling revenue
- Rolling order count

---

# 10. Machine Learning Architecture

## 10.1 Churn

```text
Customer History
       ↓
Feature Engineering
       ↓
Train / Validation / Test
       ↓
Baseline
       ↓
Logistic Regression
       ↓
Tree Models
       ↓
XGBoost
       ↓
Evaluation
       ↓
SHAP
       ↓
Prediction API
```

Primary metrics:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Calibration

Avoid using accuracy as the only metric.

---

# 11. Customer Segmentation

```text
RFM Features
     ↓
Scaling
     ↓
K-Means
     ↓
Cluster Evaluation
     ↓
Business Profiling
     ↓
Segment Labels
```

Possible outputs:

- High Value
- Loyal
- At Risk
- New
- Low Value

Cluster labels should be assigned based on actual cluster behavior, not arbitrarily.

---

# 12. CLV Architecture

```text
Historical Customer Behavior
          ↓
Customer Features
          ↓
CLV Model
          ↓
Predicted Future Value
          ↓
CLV Segmentation
          ↓
Business Prioritization
```

---

# 13. Statistical Architecture

Statistical analysis sits before and alongside ML.

```text
Business Question
       ↓
Metric Definition
       ↓
Data Assumptions
       ↓
Statistical Test
       ↓
Confidence Interval
       ↓
Effect Size
       ↓
Business Interpretation
```

The system should report both:

- Statistical significance
- Practical/business significance

A statistically significant result is not automatically a useful business result.

---

# 14. Experimentation Architecture

Because UCI does not contain randomized experiments, experimentation is implemented as a separate scenario layer.

```text
Historical Customer Data
          ↓
Eligibility Rules
          ↓
Scenario Assignment
          ↓
Control / Treatment
          ↓
Outcome Simulation
          ↓
A/B Test Analysis
          ↓
Uplift + CI + p-value
          ↓
Decision
```

The database must clearly identify scenario-generated data.

Example:

```text
data_source = "synthetic_scenario"
```

This prevents accidental presentation of simulated data as historical observations.

---

# 15. Causal Inference Architecture

```text
Treatment
    │
    ▼
Confounder Identification
    │
    ▼
Propensity Model
    │
    ▼
Matching / Weighting
    │
    ▼
Outcome Comparison
    │
    ▼
Treatment Effect
    │
    ▼
Heterogeneous Effects
```

For suitable longitudinal scenarios:

```text
Pre-period
    +
Post-period
    +
Treatment group
    +
Control group
       ↓
Difference-in-Differences
```

---

# 16. Forecasting Architecture

```text
Daily/Weekly Revenue
        ↓
Train/Test Split by Time
        ↓
Naive Baseline
        ↓
Statistical Model
        ↓
ML Model
        ↓
Backtesting
        ↓
MAE / RMSE / WAPE
        ↓
Forecast
```

Never randomly shuffle time-series data.

Use temporal validation.

---

# 17. Recommendation Architecture

```text
Customer
   ↓
Purchase History
   ↓
Candidate Generation
   ↓
Ranking
   ↓
Top-N Recommendations
   ↓
Explanation
```

Models can include:

- Popularity baseline
- Content-based
- Collaborative filtering

Evaluate with ranking metrics where appropriate.

---

# 18. Prescriptive Analytics Architecture

Prediction:

> Customer has 70% churn probability.

Prescriptive analytics:

> Contact this customer using intervention X because the expected incremental value is highest.

Architecture:

```text
Predictions
    +
Treatment Effects
    +
Customer Value
    +
Recommendations
    +
Business Constraints
          ↓
Optimization
          ↓
Recommended Actions
```

Possible objective:

```text
maximize expected incremental revenue
```

subject to:

```text
campaign budget
customer eligibility
inventory
intervention limits
business rules
```

---

# 19. GenAI Architecture

## 19.1 Business Copilot

```text
User
 │
 ▼
LLM Agent
 │
 ├── SQL Tool
 ├── Churn Tool
 ├── CLV Tool
 ├── Forecast Tool
 ├── Experiment Tool
 ├── Causal Tool
 ├── Recommendation Tool
 └── Optimization Tool
 │
 ▼
Evidence / Results
 │
 ▼
LLM
 │
 ▼
Business Explanation
 │
 ▼
Recommendation
```

The agent should not directly generate business metrics from memory.

All numerical claims should originate from analytical tools.

---

# 20. RAG Architecture

Business knowledge:

```text
docs/
   ├── data_dictionary.md
   ├── methodology.md
   ├── business_rules.md
   └── metric_definitions.md
```

Pipeline:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Retriever
   ↓
Agent Context
   ↓
LLM
```

RAG is used for methodological/business context.

Structured numerical questions should preferably use SQL/ML tools.

---

# 21. API Architecture

FastAPI acts as the application boundary.

```text
Streamlit
    │
    ▼
FastAPI
    │
    ├── Analytics Services
    ├── ML Services
    ├── Forecasting Services
    ├── Experimentation Services
    ├── Recommendation Services
    ├── Optimization Services
    └── AI Agent
```

Example:

```text
POST /predict/churn
POST /predict/clv
POST /forecast/revenue
POST /recommend
POST /experiment/analyze
POST /causal/estimate
POST /optimize
POST /agent/query
```

---

# 22. Airflow Architecture

Airflow orchestrates batch workflows.

```text
extract_uci
     ↓
validate_data
     ↓
clean_data
     ↓
load_warehouse
     ↓
build_features
     ↓
run_quality_checks
     ↓
train_models
     ↓
evaluate_models
     ↓
register_models
```

Separate DAGs can be created when the project becomes larger.

---

# 23. MLOps Architecture

MLflow tracks:

```text
Experiment
    ↓
Parameters
    ↓
Metrics
    ↓
Artifacts
    ↓
Model
    ↓
Model Version
    ↓
Registry
```

Each production model should have:

- training dataset version
- feature definition
- model version
- evaluation metrics
- training configuration

---

# 24. Model Serving

```text
MLflow Model
      ↓
Model Service
      ↓
FastAPI
      ↓
Streamlit / Agent
```

Prediction responses should include useful metadata.

Example:

```json
{
  "customer_id": "12345",
  "churn_probability": 0.82,
  "risk_level": "HIGH",
  "model_version": "xgboost-04",
  "top_factors": [
    "low_purchase_frequency",
    "long_recency",
    "declining_revenue"
  ]
}
```

---

# 25. Dashboard Architecture

## Executive Overview

KPIs:

- Revenue
- Orders
- Customers
- AOV
- Growth
- Forecast

## Customer Intelligence

- Segments
- RFM
- CLV
- Churn

## Forecasting

- Historical trends
- Forecast
- Uncertainty
- Product-level demand

## Experimentation

- Control
- Treatment
- Uplift
- Confidence interval
- p-value

## Recommendations

- Target customer
- Recommended action
- Expected impact

## AI Copilot

Natural language business interface.

---

# 26. Deployment Architecture

All services run locally via Docker Compose:

```text
Docker Compose
│
├── PostgreSQL        (port 5432)
├── pgAdmin           (port 5050)
├── FastAPI           (port 8000)
├── Streamlit         (port 8501)
├── MLflow            (port 5000)
└── Airflow           (port 8080)
```

```text
User Browser
      │
      ▼
Streamlit Dashboard
      │
      ▼
FastAPI
      │
      ├── Analytics Services
      ├── ML Models (local)
      ├── AI Agent (Ollama / local LLM)
      │
      ▼
PostgreSQL (Docker)
      │
      ▼
MLflow (local tracking)
```

No cloud services required. Everything runs on a single machine via Docker.

---

# 27. Security and Configuration

Never commit:

- API keys
- Database passwords
- LLM credentials
- Docker environment variables

Use:

```text
.env
.env.example
```

Configuration should be loaded centrally.

---

# 28. Testing Strategy

## Unit tests

Test:

- Feature functions
- Statistical calculations
- Data transformations
- Recommendation logic
- Optimization constraints

## Integration tests

Test:

```text
Database
→ Service
→ API
```

## Model tests

Validate:

- input schema
- output schema
- prediction ranges
- missing-value behavior
- model version

## Data tests

Validate:

- schema
- null rates
- duplicate rates
- numerical ranges
- referential integrity

---

# 29. Observability

Log:

- pipeline execution
- model predictions
- API latency
- failures
- model versions
- agent tool calls

For AI responses, log tool usage and evidence references rather than relying only on final generated text.

---

# 30. Business Decision Flow

The complete product flow is:

```text
                 TRANSACTION DATA
                        │
                        ▼
                    SQL / EDA
                        │
                        ▼
                  CUSTOMER ANALYTICS
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        Churn          CLV        Segmentation
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                   STATISTICS
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        Experimentation       Causal Analysis
             │                     │
             └──────────┬──────────┘
                        ▼
                    FORECASTING
                        │
                        ▼
                 RECOMMENDATIONS
                        │
                        ▼
                   OPTIMIZATION
                        │
                        ▼
                  BUSINESS ACTION
                        │
                        ▼
                    AI COPILOT
```

---

# 31. Technology Stack

## Core

- Python
- SQL
- PostgreSQL
- Pandas
- NumPy

## Statistics

- SciPy
- Statsmodels

## ML

- Scikit-learn
- XGBoost
- LightGBM
- SHAP

## Forecasting

- Statsmodels
- Prophet
- XGBoost

## GenAI

- LangGraph
- LangChain
- LLM APIs
- Embeddings
- Vector database

## Engineering

- FastAPI
- Airflow
- Docker
- MLflow
- Git/GitHub

## Visualization

- Streamlit
- Plotly

## Deployment

- Docker
- Docker Compose

---

# 32. Architecture Boundaries

The following boundaries should be preserved:

```text
data/
    Raw and processed data

sql/
    Database logic

src/features/
    Feature engineering

src/models/
    ML models

src/statistics/
    Statistical methods

src/causal/
    Causal methods

src/optimization/
    Decision logic

src/ai/
    LLM/agent functionality

src/api/
    External service interface

dashboard/
    Presentation layer

dags/
    Workflow orchestration
```

This keeps the codebase maintainable and makes each capability explainable during interviews.

---

# 33. Key Design Decision

RetailAI should not be described as:

> "A dashboard containing multiple machine learning models."

It should be described as:

> **"A decision intelligence platform that combines transactional analytics, statistical inference, predictive modeling, causal analysis, forecasting, recommendation, optimization, and agentic AI to turn retail data into actionable business decisions."**

The architecture exists to support that single product objective.
