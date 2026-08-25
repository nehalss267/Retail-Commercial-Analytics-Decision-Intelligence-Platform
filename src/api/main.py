"""FastAPI Application — RetailAI API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    customers, predictions, forecasting, experimentation,
    recommendations, optimization, agent,
)

app = FastAPI(
    title="RetailAI API",
    description="AI-powered commercial decision intelligence platform",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(customers.router)
app.include_router(predictions.router)
app.include_router(forecasting.router)
app.include_router(experimentation.router)
app.include_router(recommendations.router)
app.include_router(optimization.router)
app.include_router(agent.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/executive/summary")
def executive_summary():
    from src.api.dependencies import load_report
    eda = load_report("eda_report")
    summary = eda.get("revenue_trends", {}).get("summary", {})
    return {
        "total_revenue": summary.get("total_revenue"),
        "total_orders": summary.get("total_orders"),
        "total_customers": summary.get("total_customers"),
        "best_month": summary.get("best_month"),
    }


@app.get("/statistics")
def statistics():
    from src.api.dependencies import load_report
    return load_report("statistics_report")


@app.get("/causal/results")
def causal_results():
    from src.api.dependencies import load_report
    return load_report("causal_report")
