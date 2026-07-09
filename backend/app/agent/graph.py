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
from typing import TypedDict, Annotated, Sequence, Optional, Tuple
import operator

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

from app.config import settings
from app.agent.tools import ALL_TOOLS, resolve_hcp_by_name, add_hcp
from app.database import SessionLocal
from app.models import HCP, Interaction
from app.agent.llm import chat_completion

SYSTEM_PROMPT = """You are the HCP Interaction Agent inside a pharmaceutical CRM, helping a field representative log and manage interactions with Healthcare Professionals (HCPs) via natural conversation.

INTENT CLASSIFICATION:
Before executing any workflow, classify the representative's message into exactly one of three categories:
1. Interaction Logging: Met a doctor, had a meeting, video called, etc.
2. HCP Profile Editing: Updates to an existing HCP's name, institution, specialty, etc. (e.g., "Edit Dr Sharma's name", "Actually his name is Dr Pawan Bohra", "Change institution", "Update specialty").
3. General Conversation: Everything else (e.g., greetings, saying Yes/No, chit-chat).

CONVERSATION WORKFLOWS (Only ONE workflow may be active at any time):

WORKFLOW 1: INTERACTION LOGGING
Follow these sequential states:
1. Idle / Identify Doctor: The user mentions meeting a doctor.
   - If the name is incomplete (e.g., "Dr House" or "Mehta" or "Anjali"), you MUST ask: "Could you please provide the doctor's full name?"
   - If the name contains at least a first name and a last name (e.g., "Gregory House" or "Anjali Sharma" or "Robert Baratheon"), you MUST immediately call resolve_hcp_by_name to find their database UUID. Do NOT ask for confirmation, middle names, or other details before calling resolve_hcp_by_name.
   - If found, select that profile and proceed to "Drafting".
   - If NOT found, transition to "HCP Creation Draft".
   - YOU ARE STRICTLY FORBIDDEN from calling log_interaction, get_hcp_history, check_compliance, or schedule_follow_up for this doctor until they are successfully resolved (exists with database UUID) or created via add_hcp.
2. HCP Creation Draft (Must complete before saving):
   - Ask: "I couldn't find [Doctor Name]. Would you like to create a new HCP profile?" with buttons: [Yes, create profile] [No, check name again].
   - If yes, gather Name, Specialty, and Institution in memory as an HCP draft.
   - Support corrections: If the user corrects a draft field (e.g. "Actually his name is Pawan Bohra"), update the draft in memory. Do NOT create duplicate profiles, another draft, or a new interaction workflow.
   - Once the HCP draft is complete (contains name, specialty, institution), call add_hcp silently. This will return a real database UUID. Note: There is NO set_active_hcp tool; active state tracking is handled automatically in the backend. Proceed directly to the "Drafting" state once add_hcp completes.
   - Never call log_interaction using placeholders like "new HCP ID". Saving MUST block until a real database UUID is resolved.
3. Drafting Interaction:
   - Gather Type, Sentiment, Topics, Products, Detailed Notes, and Samples.
   - Ask only ONE concise follow-up question at a time.
   - Interpret user intent naturally (e.g., "yes and no products discussed" means Products Discussed is "No Product Discussed", not "Yes").
4. Awaiting Save Confirmation:
   - Once details are complete, ask: "Would you like me to save this interaction?" with buttons: [Save Interaction] [Discard].
5. Submitting Interaction:
   - If user confirms, verify hcp_id matches active_hcp.id and call log_interaction.
   - Once logged, return ONLY: "Thank you! I've successfully recorded your interaction with [HCP Name]. Would you like to log another HCP interaction? [Yes] [No]".
   - Immediately end this workflow. Do NOT ask for post-save profile enrichment or institution.
   - Transition to save feedback. If user says Yes, reset state and start a fresh logging workflow. If No, transition to Idle.

WORKFLOW 2: HCP PROFILE EDITING
If user requests updates to an existing doctor's profile (after saving or separately):
- Identify the HCP, update only the requested fields, and call enrich_hcp_profile.
- Respond: "Done! I've updated [HCP Name]'s profile. Is there anything else you'd like to update? [Yes] [No]".
- If user replies No, respond: "You're welcome! Would you like to log another HCP interaction? [Yes] [No]" and return to Idle.

EXTRACTION & PLANNING GUIDELINES:
- Always perform entity extraction on the entire user input first. Extract names, specialty, institution, interaction type, topics, products, notes, and samples.
- If the user provides a detailed, information-rich message (e.g., "Had an email with Dr Lisa Cuddyy from Princeton Plainbro..."), extract ALL available entities into the draft first.
- Tolerate minor spelling mistakes or typos. Do NOT repeatedly ask the user to confirm minor spelling mistakes unless the meaning is genuinely ambiguous. If the user misspells "Prinston" or "Lisa Cuddyy", extract the most likely intended value (e.g., "Princeton", "Lisa Cuddy") into the draft.
- Determine only the missing required fields after extraction.
- Execute at most one tool call per conversation turn. If a tool call has already run or failed, do NOT attempt to run it again or call another tool in the same turn. Instead, generate a natural response asking for the missing details or confirmation.

CRITICAL RULES:
- NEVER display, write, or leak database UUIDs, IDs, or 36-character hex strings in your replies to the user. Keep them entirely internal to your tool calls.
- NEVER mention tool or function names (such as log_interaction, resolve_hcp_by_name, add_hcp, enrich_hcp_profile, etc.) in your replies.
- NEVER output raw XML-like or custom function tags (e.g. '<function=...>') in your text replies. Let the API handle tool call formatting natively.
- Use brackets to output option buttons at the end of your replies when options are available (e.g. "[Yes, create profile] [No, check name again]"). Put each option inside square brackets.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _has_resolved_hcp(messages: list) -> bool:
    for msg in messages:
        if isinstance(msg, SystemMessage) and "Context: The representative currently has HCP" in msg.content:
            return True
        if isinstance(msg, ToolMessage):
            # Check if resolve_hcp_by_name or add_hcp successfully resolved/created an HCP
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    if "matches" in data and len(data["matches"]) > 0:
                        return True
                    if "hcp" in data and "id" in data["hcp"]:
                        return True
            except Exception:
                pass
    return False


def _build_llm(messages: list):
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_CHAT_MODEL,
        temperature=0.0,
    )
    if _has_resolved_hcp(messages):
        return llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    else:
        # Only bind lookup and creation tools to prevent premature calls of compliance/logging tools
        lookup_tools = [resolve_hcp_by_name, add_hcp]
        return llm.bind_tools(lookup_tools, parallel_tool_calls=False)


def _agent_node(state: AgentState):
    llm = _build_llm(state["messages"])
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState):
    # Find the index of the last HumanMessage to check current turn boundaries
    messages = state["messages"]
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    # If a tool has already been executed since the user's message, do not loop tools again
    if last_human_idx != -1:
        for m in messages[last_human_idx:]:
            if isinstance(m, ToolMessage):
                return END

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def _should_continue_after_tools(state: AgentState):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.name == "log_interaction":
        # Check if the tool execution was successful (not containing an error key)
        try:
            data = json.loads(last.content)
            if isinstance(data, dict) and "error" not in data:
                return END
        except Exception:
            pass
    return "agent"


def build_graph():
    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    workflow.add_conditional_edges("tools", _should_continue_after_tools, {"agent": "agent", END: END})

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent_turn(history: list[dict], user_message: str, active_hcp_id: Optional[str] = None) -> dict:
    """Convert chat history + new message and run the agent turn."""
    import time
    start_time = time.perf_counter()
    print(f"\n[TIMING] === run_agent_turn started for message: '{user_message}' ===", flush=True)

    # Check if the user is replying "No" to the post-save "log another interaction" question
    if history:
        last_assistant_msg = next((m for m in reversed(history) if m["role"] == "assistant"), None)
        if last_assistant_msg and "Would you like to log another HCP interaction?" in last_assistant_msg["content"]:
            clean_msg = user_message.strip().lower().replace(",", "").replace(".", "")
            if clean_msg in ("no", "no thanks", "no thank you", "nah", "nope", "n", "discard"):
                print(f"[TIMING] Intercepted negative save confirmation reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "You're welcome! Have a great day.",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": None,
                        "compliance_warnings": []
                    }
                }
            else:
                # User wants to log another interaction or has provided new details/doctor name.
                # Clear history completely to avoid workflow contamination and timeouts.
                history = []
        
        # Intercept post-profile-edit negative reply
        if last_assistant_msg and "Is there anything else you'd like to update?" in last_assistant_msg["content"]:
            clean_msg = user_message.strip().lower().replace(",", "").replace(".", "")
            if clean_msg in ("no", "no thanks", "no thank you", "nah", "nope", "n"):
                print(f"[TIMING] Intercepted negative profile edit reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "You're welcome! Would you like to log another HCP interaction? [Yes] [No]",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": None,
                        "compliance_warnings": []
                    }
                }

    print(f"[TIMING] Intent intercepts completed in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)

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

    # Limit context history to the last 6 messages (3 user-turns) to prevent token rate limits (TPM limits)
    for m in history[-6:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_message))

    final_messages = []
    reply = ""
    tool_calls_made = []

    try:
        graph = get_graph()
        graph_start = time.perf_counter()
        print(f"[TIMING] Invoking LangGraph...", flush=True)
        result = graph.invoke({"messages": messages})
        print(f"[TIMING] LangGraph invoke finished in {(time.perf_counter() - graph_start)*1000:.2f} ms", flush=True)
        final_messages = result["messages"]
    except Exception as e:
        print(f"[ERROR] LangGraph execution failed: {e}", flush=True)
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            reply = "I'm experiencing temporary rate limits. Please wait a few seconds and try sending your message again."
        else:
            reply = "I encountered a temporary issue processing your request. Please try again."
        return {
            "reply": reply,
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": None,
                "hcp": None,
                "draft_hcp": None,
                "compliance_warnings": []
            }
        }

    
    # Check if log_interaction was called and executed successfully
    log_tool_message = None
    for msg in reversed(final_messages):
        if isinstance(msg, ToolMessage) and msg.name == "log_interaction":
            log_tool_message = msg
            break
            
    for msg in final_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls_made.extend([{"tool": tc["name"], "args": tc["args"]} for tc in msg.tool_calls])

    if log_tool_message:
        try:
            tool_data = json.loads(log_tool_message.content)
            if "error" not in tool_data:
                hcp_name = tool_data.get("hcp_name", "the doctor")
                reply = f"Thank you! I've successfully recorded your interaction with {hcp_name}. Would you like to log another HCP interaction? [Yes] [No]"
            else:
                reply = f"Sorry, I encountered an error saving the interaction: {tool_data['error']}"
        except Exception:
            reply = "Your interaction has been saved successfully. Thank you! Would you like to record another interaction? [Yes] [No]"
    else:
        for msg in final_messages:
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content  # last non-empty AI content wins

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

    draft_interaction, draft_hcp = None, None
    if not active_interaction or not active_hcp:
        extract_start = time.perf_counter()
        draft_interaction, draft_hcp = _extract_draft_state(messages)
        if active_interaction:
            draft_interaction = None
        if active_hcp:
            draft_hcp = None
        print(f"[TIMING] Draft extraction completed in {(time.perf_counter() - extract_start)*1000:.2f} ms", flush=True)
    else:
        print(f"[TIMING] Draft extraction bypassed (details resolved) in 0.00 ms", flush=True)
    print(f"[TIMING] === Total turn processed in {(time.perf_counter() - start_time)*1000:.2f} ms ===\n", flush=True)

    return {
        "reply": reply,
        "tool_calls": tool_calls_made,
        "state": {
            "interaction": active_interaction,
            "draft_interaction": draft_interaction,
            "hcp": active_hcp,
            "draft_hcp": draft_hcp,
            "compliance_warnings": compliance_warnings,
        }
    }


DRAFT_STATE_EXTRACTION_PROMPT = """You are a CRM assistant. Scan the conversation history messages between the representative and the assistant.
Extract the CURRENT DRAFT state. This includes:
1. The draft HCP (Healthcare Professional) details being discussed or gathered so far. If a field (name, specialty, institution) has not been discussed or is missing, set it to null. If a field was corrected by the user, extract the corrected value.
2. The draft interaction details ONLY for the doctor currently under discussion in the most recent messages. Set missing fields to null.

Return ONLY valid JSON with these keys:
{
  "draft_hcp": {
    "name": "Full Name or null",
    "specialty": "Specialty or null",
    "institution": "Institution/Hospital or null"
  },
  "draft_interaction": {
    "interaction_type": "Meeting|Video Call|Email or null",
    "sentiment": "positive|neutral|negative or null",
    "topics_discussed": ["..."],
    "products_discussed": ["..."],
    "raw_notes": "raw notes/transcription or null",
    "samples_provided": [{"product": "...", "qty": 0}]
  }
}
No prose outside the JSON."""


def _extract_draft_state(messages: list) -> Tuple[Optional[dict], Optional[dict]]:
    """Extract both draft interaction and draft HCP details in a single LLM call to optimize latency."""
    convo = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            convo.append({"role": role, "content": msg.content})

    if not convo:
        return None, None

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": DRAFT_STATE_EXTRACTION_PROMPT},
                {"role": "user", "content": json.dumps(convo)},
            ],
            temperature=0.1,
        )
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        
        draft_hcp = data.get("draft_hcp")
        if draft_hcp and not draft_hcp.get("name") and not draft_hcp.get("specialty") and not draft_hcp.get("institution"):
            draft_hcp = None
            
        draft_interaction = data.get("draft_interaction")
        if draft_interaction:
            is_empty = (
                not draft_interaction.get("interaction_type") and
                not draft_interaction.get("sentiment") and
                not draft_interaction.get("raw_notes") and
                not draft_interaction.get("topics_discussed") and
                not draft_interaction.get("products_discussed") and
                not draft_interaction.get("samples_provided")
            )
            if is_empty:
                draft_interaction = None
                
        return draft_interaction, draft_hcp
    except Exception:
        return None, None
