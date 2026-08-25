"""API Routes — AI Agent endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def agent_query(req: QueryRequest):
    from src.ai.agent import process_query
    return process_query(req.question)
