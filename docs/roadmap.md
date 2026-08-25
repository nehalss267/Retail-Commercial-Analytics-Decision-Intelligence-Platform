# RetailAI — Roadmap

## 1. Project Vision

RetailAI is an AI-powered commercial decision intelligence platform built around the UCI Online Retail dataset.

The platform follows the decision-making chain:

**What happened? → Why did it happen? → What will happen? → What should we do?**

The goal is to demonstrate practical Data Science skills across SQL, statistics, machine learning, forecasting, customer analytics, experimentation, causal analysis, recommendation systems, prescriptive analytics, Generative AI, data engineering, and MLOps.

---

## 2. Dataset Strategy

### Primary dataset

Use the **UCI Online Retail** dataset as the core transactional source.

The raw dataset contains fields such as:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

### Important constraint

The original dataset does **not** contain every variable required for the complete platform.

Therefore:

- Use the UCI data as the source of truth for historical transactions.
- Derive customer, product, revenue, RFM, cohort, and time-series features from it.
- Do **not** falsely claim that the original dataset contains promotions, campaigns, inventory, or experimental treatment assignments.
- Where additional business scenarios are required, create clearly labeled derived/synthetic scenario tables.
- Keep real observations and simulated variables distinguishable in the warehouse.

---

# 3. Development Phases

## Phase 0 — Scope and Repository Setup

### Goals

Create the project foundation.

### Tasks

- Define business questions.
- Create repository structure.
- Add README.
- Add `.gitignore`.
- Configure Python environment.
- Configure dependency management.
- Add Docker baseline.
- Define configuration management.
- Define data lineage conventions.
- Define coding standards.

### Deliverables

- Working repository
- Initial README
- Environment configuration
- Project architecture document
- Roadmap

---

# Phase 1 — Data Acquisition and Data Quality

### Goals

Load and understand the UCI dataset.

### Tasks

- Download/store the raw dataset.
- Preserve raw data unchanged.
- Inspect schema and data types.
- Analyze missing values.
- Detect duplicate records.
- Detect cancellations/returns.
- Detect invalid quantities.
- Detect invalid prices.
- Examine date ranges.
- Examine customer coverage.
- Examine country distribution.
- Produce a data-quality report.

### Deliverables

- `data/raw/`
- Data ingestion script
- Data-quality report
- Data dictionary
- Reproducible preprocessing pipeline

---

# Phase 2 — SQL Data Warehouse

### Goals

Build a normalized analytical database.

### Suggested tables

- customers
- products
- orders
- order_items
- countries
- calendar
- customer_daily_metrics
- customer_monthly_metrics
- product_monthly_metrics

Additional scenario tables will be introduced later:

- promotions
- experiments
- campaign_exposure
- inventory_scenarios

### Tasks

- Design relational schema.
- Load cleaned data into PostgreSQL.
- Create indexes.
- Create analytical views.
- Write advanced SQL queries.
- Use CTEs.
- Use window functions.
- Build cohort analysis.
- Calculate retention.
- Calculate customer revenue.
- Calculate average order value.
- Calculate purchase frequency.

### Deliverables

- SQL schema
- ETL/load scripts
- Analytical SQL queries
- KPI views
- Data dictionary

---

# Phase 3 — Exploratory Data Analysis

### Goals

Understand customer, product, geographic, and temporal behavior.

### Analysis

- Revenue trends
- Order volume
- Average order value
- Customer distribution
- Product popularity
- Country contribution
- Seasonality
- Repeat purchase behavior
- Customer concentration
- Pareto analysis
- Cancellation/return behavior

### Deliverables

- EDA notebook
- Reusable analysis scripts
- Business findings
- Visualizations
- Executive-level KPI summary

---

# Phase 4 — Customer Analytics

## 4.1 RFM Segmentation

Calculate:

- Recency
- Frequency
- Monetary value

Apply:

- K-Means
- Hierarchical clustering

Compare clustering quality using appropriate metrics.

### Output

Segments such as:

- High Value
- Loyal
- New
- At Risk
- Low Value

## 4.2 Customer Lifetime Value

Build a customer value model using historical purchasing behavior.

### Outputs

- Expected customer value
- Customer value segments
- High-value customer identification

## 4.3 Churn / Inactivity Prediction

Define a defensible inactivity/churn label based on future purchase behavior.

Compare:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Evaluate:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Calibration

Use SHAP for interpretation.

---

# Phase 5 — Statistical Analytics

### Goals

Demonstrate statistical reasoning before machine learning.

### Topics

- Descriptive statistics
- Confidence intervals
- Hypothesis testing
- Effect size
- Statistical significance
- Practical significance
- ANOVA
- Correlation analysis

### Business questions

- Do high-value customers behave differently from other customers?
- Do customer segments have significantly different order values?
- Do countries differ significantly in average order value?
- Are observed revenue differences statistically meaningful?

### Deliverables

- Statistical analysis module
- Reproducible statistical reports
- Business interpretation for each test

---

# Phase 6 — Experimentation and A/B Testing

### Important distinction

The UCI dataset is observational. It does not provide a randomized promotion experiment.

Therefore create a clearly labeled **scenario/experimentation layer**.

### Example

Simulate a promotion experiment using customer/product history as the baseline.

Control:

- No simulated intervention

Treatment:

- Simulated promotion exposure

### Analyze

- Conversion
- Revenue
- Average order value
- Customer response

### Calculate

- Absolute uplift
- Relative uplift
- Confidence interval
- p-value
- Effect size
- Statistical power
- Sample size requirements

### Deliverable

An experimentation module that can answer:

> Did the proposed intervention create a meaningful improvement?

---

# Phase 7 — Causal Inference

### Goal

Distinguish prediction from causation.

Implement at least one robust causal methodology.

Recommended progression:

1. Confounding explanation
2. Propensity score modeling
3. Propensity score matching
4. Average Treatment Effect
5. Conditional/heterogeneous treatment effects
6. Difference-in-differences for an appropriate simulated scenario

### Important

Clearly label simulated treatment data.

Do not present simulated causal effects as historical facts from UCI.

---

# Phase 8 — Demand and Revenue Forecasting

### Goals

Predict future business performance.

### Forecasts

- Daily revenue
- Weekly revenue
- Monthly revenue
- Product demand

### Models

Start with:

- Naive baseline
- Moving average

Then compare:

- ARIMA/SARIMA
- Prophet
- XGBoost with lag/rolling features

### Evaluation

- MAE
- RMSE
- MAPE/WAPE where appropriate

### Deliverables

- Forecasting pipeline
- Forecast API
- Forecast dashboard
- Prediction intervals where supported

---

# Phase 9 — Recommendation System

### Goal

Recommend products to customers.

### Approaches

Start with:

- Popularity baseline
- Content-based recommendation

Then implement:

- Collaborative filtering

Evaluate with ranking-oriented metrics where applicable.

### Output

For a selected customer:

- Top recommended products
- Recommendation score
- Explanation/features where possible

---

# Phase 10 — Prescriptive Analytics

### Goal

Move from prediction to action.

Combine:

- Customer value
- Churn risk
- Recommendation score
- Predicted demand
- Estimated treatment effect
- Business constraints

Create an optimization/scenario engine.

### Example decision

> Which customers should receive an intervention to maximize expected incremental revenue under a fixed campaign budget?

### Constraints

- Budget
- Customer eligibility
- Maximum discount
- Inventory
- Minimum expected ROI

### Output

Recommended:

- Customer targets
- Product targets
- Intervention
- Expected impact

---

# Phase 11 — Generative AI Business Copilot

### Goal

Create an AI interface for business users.

Use an agent framework such as LangGraph/LangChain.

### Tools

The agent should be able to call:

- SQL analytics tool
- Customer segmentation tool
- Churn prediction tool
- CLV tool
- Forecasting tool
- Experimentation tool
- Causal analysis tool
- Recommendation tool
- Optimization tool

### RAG

Create a small business knowledge base containing:

- Data dictionary
- KPI definitions
- Business rules
- Experimentation guidelines
- Analytical methodology notes

### Example questions

- "Why did revenue fall last month?"
- "Which customer segment should we target?"
- "What products are driving revenue?"
- "What does the forecast look like?"
- "What would happen if we increased the intervention budget?"
- "Explain the churn model."

### Important

The agent should retrieve numerical results from tools rather than inventing them.

---

# Phase 12 — Dashboard and Application

Build a Streamlit business dashboard.

### Pages

1. Executive Overview
2. Customer Intelligence
3. Product Analytics
4. Churn & CLV
5. Forecasting
6. Experimentation
7. Causal Analysis
8. Recommendations
9. Scenario/Optimization
10. AI Copilot

Use Plotly for interactive visualizations.

---

# Phase 13 — API Layer

Create a FastAPI service.

### Endpoints

Example:

- `GET /health`
- `GET /customers/{id}`
- `POST /predict/churn`
- `POST /predict/clv`
- `POST /forecast/revenue`
- `POST /recommend`
- `POST /experiment/analyze`
- `POST /causal/estimate`
- `POST /optimize`
- `POST /agent/query`

---

# Phase 14 — Data Engineering

Use Airflow to orchestrate reproducible pipelines.

### Pipeline

Raw UCI data
→ validation
→ cleaning
→ transformation
→ warehouse
→ feature engineering
→ model datasets
→ model training
→ evaluation

Add:

- retries
- logging
- task dependencies
- data validation
- idempotent processing

---

# Phase 15 — MLOps

Use MLflow for:

- Experiment tracking
- Parameters
- Metrics
- Artifacts
- Model versions
- Model registry

Add:

- reproducible training
- model versioning
- evaluation gates
- inference logging

Dockerize:

- FastAPI
- Streamlit
- ML service
- supporting services

---

# Phase 16 — Testing and Reliability

Add tests for:

- Data transformations
- Feature engineering
- SQL outputs
- Model prediction schemas
- API endpoints
- Statistical functions
- Recommendation outputs

Add validation for:

- Missing values
- Schema drift
- Feature ranges
- Prediction distributions

---

# Phase 17 — Deployment

Run everything locally via Docker Compose.

### Services

- PostgreSQL
- pgAdmin
- FastAPI
- Streamlit
- MLflow
- Airflow (when needed)

### Benefits

- Zero cloud cost
- No credit card required
- Fully reproducible
- Works offline
- Easy to demonstrate in interviews

A reliable containerized local deployment is more valuable than a fragile cloud architecture.

---

# Phase 18 — Final Product Polish

Add:

- Executive dashboard
- AI Copilot
- Model explanations
- Business recommendation cards
- Data lineage
- Methodology documentation
- Architecture diagrams
- Demo dataset
- Reproducible setup instructions

---

# 4. Final Skill Coverage

The finished project should demonstrate:

### Programming/Data

- Python
- SQL
- Pandas
- NumPy
- PostgreSQL

### Statistics

- Descriptive statistics
- Hypothesis testing
- Confidence intervals
- A/B testing
- ANOVA
- Effect size
- Sample-size estimation
- Statistical power
- Bayesian analysis

### Machine Learning

- Regression
- Classification
- Decision Trees
- Random Forest
- XGBoost
- KNN
- SVM
- Clustering
- PCA
- Feature engineering
- Cross-validation
- Hyperparameter tuning

### Advanced Analytics

- Customer segmentation
- Churn
- CLV
- Forecasting
- Causal inference
- Recommendation systems
- Prescriptive analytics
- Optimization

### AI

- LLMs
- RAG
- Embeddings
- Vector search
- Tool calling
- Agentic AI

### Engineering

- Airflow
- FastAPI
- Docker
- MLflow
- Git/GitHub

### Communication

- Dashboards
- Data storytelling
- Business recommendations
- Executive summaries

---

# 5. Suggested Build Order

Do not implement everything simultaneously.

### MVP

1. UCI ingestion
2. Cleaning
3. PostgreSQL
4. SQL analytics
5. EDA
6. RFM
7. Clustering
8. Churn
9. Dashboard

### Data Science expansion

10. CLV
11. Statistics
12. A/B testing
13. Causal inference
14. Forecasting
15. Recommendations

### Decision Intelligence

16. Optimization
17. Business recommendations
18. Scenario analysis

### AI

19. RAG
20. Agent
21. Tool calling
22. AI Copilot

### Production

23. FastAPI
24. Airflow
25. MLflow
26. Docker
27. Tests

This ordering ensures that the project is always usable even if later phases are unfinished.

---

# 6. Final Resume Positioning

The finished project should be described as a **commercial decision intelligence platform**, not as a collection of ML models.

The central narrative should be:

**Transactional data → Analytics → Statistics → ML → Causal insights → Forecasts → Recommendations → Optimization → AI-assisted business decisions.**
