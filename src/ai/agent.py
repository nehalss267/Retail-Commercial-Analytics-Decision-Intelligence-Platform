"""GenAI Business Copilot — LangGraph agent with tool calling and RAG.

Architecture:
    User Question
         ↓
    LangGraph StateGraph
         ├── Route Node (keyword-based routing to tool category)
         ├── Tool Node (execute analytical tools)
         ├── RAG Context (retrieve business knowledge)
         └── Synthesize Node (build answer from tool results)

The agent routes queries to 7 tool categories:
    - SQL Analytics (revenue, customers, products, countries)
    - Churn Prediction (risk, model summary, SHAP explanations)
    - Forecasting (30-day forecast, model comparison)
    - Experimentation (A/B test results, methodology)
    - Causal Inference (propensity matching, treatment effects)
    - Recommendations (popular, content-based, collaborative)
    - Optimization (targeting, ROI, sensitivity analysis)

All numerical answers come from tool results — never fabricated.
"""
import json

from src.ai.graph import build_graph, AgentState
from src.ai.rag.retrieval import retrieve_context, format_context_for_prompt


# Build the graph once at module load
_agent_graph = None


def _get_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    return _agent_graph


def process_query(question: str) -> dict:
    """Process a user query through the LangGraph agent.

    Returns:
        dict with keys: question, answer, source, tool_calls, rag_context
    """
    graph = _get_graph()

    # Run the graph
    initial_state: AgentState = {
        "question": question,
        "route": "",
        "tool_results": [],
        "rag_context": "",
        "answer": "",
    }

    result = graph.invoke(initial_state)

    # Determine source
    route = result.get("route", "sql")
    has_tool_results = len(result.get("tool_results", [])) > 0
    has_rag = bool(result.get("rag_context"))

    if has_tool_results and has_rag:
        source = "analytics_tools + knowledge_base"
    elif has_tool_results:
        source = "analytics_tools"
    elif has_rag:
        source = "knowledge_base"
    else:
        source = "fallback"

    return {
        "question": question,
        "answer": result.get("answer", "I couldn't process your question."),
        "source": source,
        "route": route,
        "tool_calls": [r.get("tool") for r in result.get("tool_results", [])],
    }


def get_available_tools() -> list[dict]:
    """List all available tools and their descriptions."""
    from src.ai.graph import ALL_TOOLS
    return [
        {"name": t.name, "description": t.description}
        for t in ALL_TOOLS
    ]


if __name__ == "__main__":
    test_queries = [
        "What is the total revenue?",
        "How many customers do we have?",
        "What is the churn rate?",
        "Which customer segments do we have?",
        "What does the forecast look like?",
        "How do you calculate CLV?",
        "What is the RFM methodology?",
        "Which country generates the most revenue?",
        "What products should I recommend for customer 12345?",
        "Which customers should we target for a campaign?",
        "What was the experiment result?",
        "Explain the causal inference methodology",
    ]

    print("=== RetailAI Business Copilot ===\n")
    for q in test_queries:
        result = process_query(q)
        print(f"Q: {q}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Source: {result['source']} | Route: {result['route']} | Tools: {result['tool_calls']}")
        print()
