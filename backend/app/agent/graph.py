"""
LangGraph agent for the HCP Log Interaction chat interface.

Role of this agent:
  The agent sits behind the chat side of the Log Interaction screen. A field
  rep types naturally ("Met Dr. Mehta today, discussed the new cardio study,
  left 10 samples, she wants the phase III data by Friday"), and the agent:
    1. Decides which tool(s) the request needs (log vs edit vs lookup vs
       follow-up vs compliance) via the Groq-hosted LLM's tool-calling.
    2. Calls the relevant tool(s), which read/write Postgres and, for
       logging/editing, invoke the LLM again internally for extraction.
    3. Runs a compliance check automatically after any log/edit so issues
       surface before the rep moves on.
    4. Turns the tool result(s) back into a short, natural confirmation
       message for the rep, rather than raw JSON.

  The same tools power the structured-form path too — the form submits
  directly to the REST endpoints in routers/interactions.py, which call the
  same underlying tool functions, so business logic lives in one place.
"""
import json
from typing import TypedDict, Annotated, Sequence, Optional
import operator

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

from app.config import settings
from app.agent.tools import ALL_TOOLS
from app.database import SessionLocal
from app.models import HCP, Interaction
from app.agent.llm import chat_completion

SYSTEM_PROMPT = """You are the HCP Interaction Agent inside a pharmaceutical CRM,
helping a field representative log and manage interactions with Healthcare
Professionals (HCPs) via natural conversation.

Mandatory Interaction Details:
- Every logged or edited interaction MUST have:
  1. Interaction Type (Meeting, Video Call, or Email)
  2. Sentiment (positive, neutral, or negative)
  3. Topics Discussed (one or more specific topics)
  4. Products Discussed (if none or no products were discussed, this MUST be recorded as "No Product Discussed")

Guidelines:
- If the rep mentions an HCP by name and you do not have their 36-character database UUID (hcp_id), you MUST call resolve_hcp_by_name first to find their ID.
- If the rep describes an interaction, check if the description specifies all 4 Mandatory Interaction Details (Type, Sentiment, Topics, Products).
- If any of these details are missing, do NOT call log_interaction or edit_interaction yet. Proactively ask the user to provide the missing details.
- Specifically, if products are not mentioned, you MUST explicitly ask: "Were any products discussed? If yes, please name the products. If no, let me know so I can record 'No Product Discussed'."
- If the user says there was no product discussion, or "n/a", or "no products", proceed with "No Product Discussed" as the product value in your call.
- Once you have all 4 mandatory details, call log_interaction (or edit_interaction if they are correcting an existing record).
- After logging or editing an interaction, check if the HCP's institution is missing or empty. If institution is missing, ask the user ONLY for the institution (e.g. "Great! I have logged that. Could you also tell me which institution or hospital Dr. X is based at?"). Do NOT ask for email or phone — those are optional and not required.
- If the HCP already has an institution set, do NOT ask for any profile details. Just confirm the interaction was logged and stop.
- If the user provides the missing profile details, call enrich_hcp_profile to update their database record. Pass the 36-character UUID as the hcp_id parameter.
- If the rep wants to correct/change something already logged, call edit_interaction. If they want to correct the associated HCP/doctor (e.g. "Actually it was Dr. Y, not Dr. X"), resolve the new doctor's UUID via resolve_hcp_by_name first, and then pass that new UUID to edit_interaction.
- If the rep asks about past interactions with an HCP, call get_hcp_history.
- If the rep mentions a task/reminder tied to an interaction, call schedule_follow_up.
- After logging or editing an interaction that involves samples, call check_compliance and mention any warnings plainly to the rep.
- NEVER display or write database UUIDs (e.g., 36-character strings like '57471ad8-...') in your chat responses. Always refer to doctor profiles and interactions only by name or date. Keep UUIDs entirely internal to your tool calling arguments.
- NEVER output raw XML/tag-like function blocks in your text responses. Always trigger tool calls natively via the platform's API.
- Always confirm what was recorded in one or two natural sentences - never dump raw JSON.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _build_llm():
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_CHAT_MODEL,
        temperature=0.2,
    ).bind_tools(ALL_TOOLS, parallel_tool_calls=False)


def _agent_node(state: AgentState):
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")  # loop back so the LLM can summarize tool output

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent_turn(history: list[dict], user_message: str, active_hcp_id: Optional[str] = None) -> dict:
    """Convert chat history + new message into LangChain messages, run the
    graph to completion, and return the final assistant reply plus any tool
    calls that were made (useful for the frontend to show Logged chips)."""
    messages: list[BaseMessage] = []
    
    # Inject active HCP context at the top of history
    if active_hcp_id:
        db = SessionLocal()
        try:
            hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
            if hcp:
                messages.append(SystemMessage(
                    content=f"Context: The representative currently has HCP '{hcp.name}' (UUID: '{hcp.id}') selected on their screen. "
                            f"Use this UUID '{hcp.id}' as the hcp_id in tool calls when referring to this doctor."
                ))
        finally:
            db.close()

    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_message))

    graph = get_graph()
    result = graph.invoke({"messages": messages})

    final_messages = result["messages"]
    reply = ""
    tool_calls_made = []
    for msg in final_messages:
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content  # last non-empty AI content wins
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls_made.extend([{"tool": tc["name"], "args": tc["args"]} for tc in msg.tool_calls])

    # Scan for interaction and HCP info
    interaction_id = None
    hcp_id = None
    compliance_warnings = []
    
    for msg in reversed(final_messages):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    if "interaction_id" in data:
                        interaction_id = data["interaction_id"]
                    if "hcp_id" in data:
                        hcp_id = data["hcp_id"]
                    if "hcp" in data and isinstance(data["hcp"], dict) and "id" in data["hcp"]:
                        hcp_id = data["hcp"]["id"]
                    if "matches" in data and isinstance(data["matches"], list) and len(data["matches"]) > 0:
                        hcp_id = data["matches"][0]["id"]
                    if msg.name == "check_compliance":
                        compliance_warnings = data.get("warnings", [])
            except Exception:
                pass

    active_interaction = None
    active_hcp = None

    if interaction_id or hcp_id:
        db = SessionLocal()
        try:
            if interaction_id:
                i_model = db.query(Interaction).filter(Interaction.id == interaction_id).first()
                if i_model:
                    active_interaction = {
                        "id": i_model.id,
                        "hcp_id": i_model.hcp_id,
                        "interaction_type": i_model.interaction_type,
                        "channel": i_model.channel,
                        "interaction_date": i_model.interaction_date.isoformat() if i_model.interaction_date else None,
                        "summary": i_model.summary,
                        "raw_notes": i_model.raw_notes,
                        "topics_discussed": i_model.topics_discussed,
                        "products_discussed": i_model.products_discussed,
                        "sentiment": i_model.sentiment,
                        "samples_provided": i_model.samples_provided,
                        "follow_up_actions": i_model.follow_up_actions,
                    }
                    if not hcp_id:
                        hcp_id = i_model.hcp_id
            
            if hcp_id:
                hcp_model = db.query(HCP).filter(HCP.id == hcp_id).first()
                if hcp_model:
                    active_hcp = {
                        "id": hcp_model.id,
                        "name": hcp_model.name,
                        "specialty": hcp_model.specialty,
                        "institution": hcp_model.institution,
                        "email": hcp_model.email,
                        "phone": hcp_model.phone,
                    }
        finally:
            db.close()

    draft_interaction = None
    if not active_interaction:
        draft_interaction = _extract_draft_interaction(messages)

    return {
        "reply": reply,
        "tool_calls": tool_calls_made,
        "state": {
            "interaction": active_interaction,
            "draft_interaction": draft_interaction,
            "hcp": active_hcp,
            "compliance_warnings": compliance_warnings,
        }
    }


DRAFT_EXTRACTION_PROMPT = """You are a CRM assistant. Scan the conversation history messages between the representative and the assistant.
Extract the CURRENT DRAFT interaction details that have been mentioned or agreed upon so far.
Do NOT guess, extrapolate, or invent fields that have not been explicitly mentioned yet.
If a field has not been discussed or is missing, set it to null (or empty array/list).

Return ONLY valid JSON with these keys:
{
  "interaction_type": "Meeting|Video Call|Email or null",
  "sentiment": "positive|neutral|negative or null",
  "summary": "AI draft summary of the visit or null",
  "topics_discussed": ["..."],
  "products_discussed": ["..."],
  "raw_notes": "raw notes/transcription discussed so far or null"
}
No prose outside the JSON."""


def _extract_draft_interaction(messages: list) -> Optional[dict]:
    """Filter messages, format for LLM, and extract a live draft of the interaction."""
    convo = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            convo.append({"role": role, "content": msg.content})

    if not convo:
        return None

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": DRAFT_EXTRACTION_PROMPT},
                {"role": "user", "content": json.dumps(convo)},
            ],
            temperature=0.1,
        )
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception:
        return None
