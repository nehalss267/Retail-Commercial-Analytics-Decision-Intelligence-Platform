"""LangGraph Agent Graph — StateGraph for the Business Copilot.

This module defines the agent architecture using LangGraph's StateGraph.
It routes user queries to the appropriate analytical tools and synthesizes
responses using retrieved knowledge base context.

The agent can operate in two modes:
1. Tool-calling mode: when an LLM is available, the LLM decides which tools to call
2. Keyword-routing mode: fallback when no LLM is configured — routes by keywords
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

from src.ai.tools import (
    sql_tool, churn_tool, forecast_tool,
    experiment_tool, causal_tool, recommendation_tool, optimization_tool,
)
from src.ai.rag.retrieval import retrieve_context, format_context_for_prompt


# === State Definition ===

class AgentState(TypedDict):
    question: str
    route: str
    tool_results: list[dict]
    rag_context: str
    answer: str


# === Tool Registry ===

TOOLS = {
    "sql": [
        sql_tool.query_revenue_metrics,
        sql_tool.query_customer_segments,
        sql_tool.query_top_products,
        sql_tool.query_country_revenue,
        sql_tool.query_customer_detail,
        sql_tool.query_daily_metrics,
    ],
    "churn": [
        churn_tool.predict_churn_risk,
        churn_tool.get_churn_model_summary,
        churn_tool.get_churn_explanations,
    ],
    "forecast": [
        forecast_tool.get_forecast_summary,
        forecast_tool.get_forecast_values,
        forecast_tool.compare_forecast_models,
    ],
    "experiment": [
        experiment_tool.get_experiment_results,
        experiment_tool.explain_experiment_methodology,
    ],
    "causal": [
        causal_tool.get_causal_results,
        causal_tool.explain_causal_methodology,
    ],
    "recommendation": [
        recommendation_tool.get_popular_products,
        recommendation_tool.get_product_recommendations,
        recommendation_tool.get_customer_recommendations,
    ],
    "optimization": [
        optimization_tool.get_optimization_results,
        optimization_tool.explain_optimization_methodology,
        optimization_tool.get_sensitivity_analysis,
    ],
}

ALL_TOOLS = [t for tools in TOOLS.values() for t in tools]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# === Routing Logic ===

ROUTE_KEYWORDS = {
    "optimization": [
        "optim", "target", "budget", "roi", "prescriptive",
        "which customer should", "maximize", "campaign", "intervention",
    ],
    "churn": [
        "churn", "inactiv", "at risk", "lost customer",
    ],
    "forecast": [
        "forecast", "future", "next month", "next 30",
        "revenue forecast", "demand",
    ],
    "experiment": [
        "experiment", "a/b test", "ab test", "uplift",
        "campaign result", "treatment effect",
    ],
    "causal": [
        "causal", "causation", "propensity",
        "difference in differences", "did",
    ],
    "recommendation": [
        "recommend", "suggest", "similar product",
        "collaborative", "content-based",
    ],
    "sql": [
        "revenue", "total", "how many", "how much", "orders", "customers",
        "products", "country", "countries", "uk", "daily", "monthly",
        "summary", "kpi", "metric", "segments",
    ],
}


def route_query(state: AgentState) -> AgentState:
    """Route the user query to the appropriate tool category."""
    question = state["question"].lower()

    scores = {}
    for route, keywords in ROUTE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in question)
        if score > 0:
            scores[route] = score

    route = max(scores, key=scores.get) if scores else "sql"

    # Retrieve RAG context for methodology questions
    rag_context = ""
    if any(w in question for w in ["how", "method", "approach", "why", "explain", "what is"]):
        contexts = retrieve_context(question, n_results=2)
        rag_context = format_context_for_prompt(contexts)

    return {**state, "route": route, "rag_context": rag_context}


def call_tools(state: AgentState) -> AgentState:
    """Execute the relevant tools based on the route."""
    import re

    route = state["route"]
    question = state["question"]
    q_lower = question.lower()
    tools = TOOLS.get(route, TOOLS["sql"])
    results = []
    nums = re.findall(r'\d+', question)
    codes = re.findall(r'[A-Z0-9]{5,}', question)

    for tool_fn in tools:
        try:
            name = tool_fn.name

            # Tools needing a customer_id
            if name in ("predict_churn_risk", "get_customer_detail", "get_customer_recommendations"):
                if nums:
                    result = tool_fn.invoke({"customer_id": int(nums[0])})
                    results.append({"tool": name, "result": result})

            # Tools needing product code
            elif name == "get_product_recommendations":
                if codes:
                    result = tool_fn.invoke({"product_code": codes[0], "n": 5})
                    results.append({"tool": name, "result": result})

            # Tools needing n (top N)
            elif name in ("query_top_products", "get_popular_products"):
                result = tool_fn.invoke({"n": 10})
                results.append({"tool": name, "result": result})

            # No-arg tools
            elif len(tool_fn.args) == 0:
                result = tool_fn.invoke({})
                results.append({"tool": name, "result": result})

        except Exception as e:
            results.append({"tool": tool_fn.name, "error": str(e)})

    return {**state, "tool_results": results}


def synthesize_answer(state: AgentState) -> AgentState:
    """Synthesize a final answer from tool results and RAG context."""
    results = state.get("tool_results", [])
    rag = state.get("rag_context", "")
    question = state["question"]

    if not results:
        answer = "I couldn't find relevant data for your question. Try asking about revenue, customers, churn, forecasts, or recommendations."
        return {**state, "answer": answer}

    # Build answer from tool results
    parts = []
    for r in results:
        tool_name = r.get("tool", "unknown")
        result = r.get("result", {})
        error = r.get("error")

        if error:
            parts.append(f"**{tool_name}**: {error}")
            continue

        if isinstance(result, dict):
            # Format key metrics
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if "revenue" in key.lower() or "clv" in key.lower():
                        parts.append(f"- **{key}**: £{value:,.2f}")
                    elif "pct" in key.lower() or "rate" in key.lower() or "power" in key.lower():
                        parts.append(f"- **{key}**: {value}")
                    else:
                        parts.append(f"- **{key}**: {value}")
                elif isinstance(value, str) and len(value) < 200:
                    parts.append(f"- **{key}**: {value}")
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # Table-like data
                for item in result[:5]:
                    name = item.get("Description", item.get("stock_code", item.get("stock_code", "")))[:50]
                    rev = item.get("revenue", item.get("score", ""))
                    if name:
                        parts.append(f"  - {name}: {rev}")
            else:
                parts.append(f"  - {result[:5]}")

    answer = "\n".join(parts)

    # Append RAG context if available
    if rag:
        answer += f"\n\n---\n*Methodology context:* {rag[:500]}"

    return {**state, "answer": answer}


# === Graph Definition ===

def build_graph() -> StateGraph:
    """Build the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_query)
    graph.add_node("call_tools", call_tools)
    graph.add_node("synthesize", synthesize_answer)

    # Add edges
    graph.set_entry_point("route")
    graph.add_edge("route", "call_tools")
    graph.add_edge("call_tools", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
