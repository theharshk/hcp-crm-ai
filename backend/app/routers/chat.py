from fastapi import APIRouter

from app import schemas
from app.agent.graph import run_agent_turn

router = APIRouter()


@router.post("/chat", response_model=schemas.ChatResponse)
def chat_turn(payload: schemas.ChatRequest):
    """Single conversational turn against the LangGraph agent. The frontend
    keeps the running `history` client-side (in Redux) and resends it each
    call, since this is a stateless REST endpoint."""
    history = [{"role": m.role, "content": m.content} for m in (payload.history or [])]
    result = run_agent_turn(history, payload.message, active_hcp_id=payload.hcp_id)

    return schemas.ChatResponse(
        session_id=payload.session_id,
        reply=result["reply"],
        tool_calls=result["tool_calls"],
        state=result.get("state"),
    )
