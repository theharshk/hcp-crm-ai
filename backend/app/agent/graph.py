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
   - A full name has at least two components (e.g., "Ravi Joshi", "Sarah Malik", "Karan Malhotra"). If the name provided contains two or more words, it is complete and you MUST immediately call resolve_hcp_by_name. Do NOT ask for the full name if they already gave you a first and last name.
   - If the name is incomplete (e.g., "Dr House" or "Mehta" or "Anjali"), you MUST ask: "Could you please provide the doctor's full name?"
   - If found, select that profile and proceed to "Drafting".
   - If NOT found: Ask the user for the doctor's specialty and institution/hospital so you can create a profile. Say: "I couldn't find [Name] in our database. Could you tell me their specialty and hospital/institution so I can add them?"
   - Once the user provides specialty and/or institution (or confirms there is none), you MUST immediately call add_hcp with the name, specialty, and institution. If you just asked for specialty/institution, DO NOT mistake their answer for a new doctor's name and DO NOT call resolve_hcp_by_name again. Call add_hcp immediately with whatever you have. Do NOT output a conversational text response pretending to have added them without actually calling the add_hcp tool first.
   - After add_hcp succeeds, confirm the profile was created and immediately continue to the Drafting step.
   - YOU ARE STRICTLY FORBIDDEN from calling log_interaction, get_hcp_history, check_compliance, or schedule_follow_up for this doctor until they are successfully resolved (exists with database UUID).

2. Drafting Interaction:
   - Gather fields in this strict order: Interaction Type -> Sentiment -> Topics Discussed -> Products Discussed -> Detailed Notes -> Samples Provided.
   - When asking for Interaction Type, you MUST strictly only offer these options: meeting, video call, or email. You MUST append buttons: `[Meeting] [Video Call] [Email]`.
   - If the user mentions any type of "phone call", "call", or "anything call", you MUST treat it as a "Video Call".
   - You MUST ask for Sentiment BEFORE asking for Topics Discussed.
   - Ask only ONE concise follow-up question at a time.
   - Support draft corrections: If the user corrects any drafted interaction field (e.g. "Actually it was positive", "No, it was a video call", "Topics discussed was heart disease", "Medication discussed was Drug B"), interpret the change, update the draft state immediately, and confirm the update to the user. Do NOT ask duplicate questions for fields the user already specified.
   - When asking for Sentiment, you MUST append buttons: `[Positive] [Neutral] [Negative]`.
   - When asking for Topics Discussed, you MUST NOT append any bracketed suggestions or buttons (e.g., do NOT output `[Pancreatic Cancer] [Other]` etc.). Topics discussed is a free-text field that the user must type manually.
   - If medications (products discussed) are not mentioned, ask: "Were any medications discussed during the interaction? If yes, please name the medications. If no, let know so I can record 'No Medication Discussed'. [No Medication Discussed]"
   - STRICT SKIP RULE: You are ONLY allowed to offer a skip option for Detailed Notes and Samples Provided. You MUST NOT allow skipping for Interaction Type, Sentiment, Topics Discussed, or Products Discussed.
   - When asking for Detailed Notes or Samples Provided, you MUST append a skip option (e.g., "[Skip Notes]" or "[Skip Samples]") so the user can easily omit them.
   - If the user clicks skip for Notes or Samples, treat that field as complete (e.g. null or empty array) and move to the next field.

4. Awaiting Save Confirmation:
   - Once details are complete, ask: "Would you like me to save this interaction?" with buttons: [Save Interaction] [Discard].
5. Submitting Interaction:
   - YOU ARE STRICTLY FORBIDDEN from calling log_interaction unless the user explicitly clicked the "[Save Interaction]" button or explicitly said to save it. If they are just answering questions about the interaction (like "video call" or "positive"), you must NEVER call log_interaction.
   - If user confirms saving, verify hcp_id matches active_hcp.id and call log_interaction.
   - Once logged, return ONLY: "Thank you! I've successfully recorded your interaction with [HCP Name]. Would you like to log another HCP interaction? [Yes] [No]".
   - Immediately end this workflow. Do NOT ask for post-save profile enrichment or institution.
   - Transition to save feedback. If user says Yes, reset state and start a fresh logging workflow. If No, transition to Idle.

WORKFLOW 2: HCP PROFILE EDITING
If user requests updates to an existing doctor's profile (after saving or separately):
- Identify the HCP, update only the requested fields, and call enrich_hcp_profile.
- Respond: "Done! I've updated [HCP Name]'s profile. Is there anything else you'd like to update? [Yes] [No]".
- If user replies No, respond: "You're welcome! Would you like to log another HCP interaction? [Yes] [No]" and return to Idle.

EXTRACTION & PLANNING GUIDELINES:
- Always perform entity extraction on the entire user input first. Extract names, specialty, institution, interaction type, topics, medications (products), notes, and samples.
- If the user provides a detailed, information-rich message (e.g., "Had an email with Dr Lisa Cuddyy from Princeton Plainbro..."), extract ALL available entities into the draft first.
- Tolerate minor spelling mistakes or typos. Do NOT repeatedly ask the user to confirm minor spelling mistakes unless the meaning is genuinely ambiguous. If the user misspells "Prinston" or "Lisa Cuddyy", extract the most likely intended value (e.g., "Princeton", "Lisa Cuddy") into the draft.
- Determine only the missing required fields after extraction.
- Execute at most one tool call per conversation turn. If a tool call has already run or failed, do NOT attempt to run it again or call another tool in the same turn. Instead, generate a natural response asking for the missing details or confirmation.

CRITICAL RULES:
- ALWAYS provide a natural, conversational text response. Do not output raw JSON, code blocks, or custom tags in your replies.
- NEVER display, write, or leak database UUIDs, IDs, or 36-character hex strings in your replies to the user. Keep them entirely internal to your tool calls.
- NEVER mention tool or function names (such as log_interaction, resolve_hcp_by_name, add_hcp, enrich_hcp_profile, etc.) in your replies.
- Use brackets to output option buttons at the end of your replies when options are available (e.g. "[Yes, create profile] [No, check name again]"). Put each option inside square brackets.
- Do NOT tell the user to click any button in the UI. You handle everything through conversation.
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
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_CHAT_MODEL,
            temperature=0.0,
        )
        if _has_resolved_hcp(messages):
            return llm.bind_tools(ALL_TOOLS)
        else:
            # Only bind lookup and creation tools to prevent premature calls of compliance/logging tools
            lookup_tools = [resolve_hcp_by_name, add_hcp]
            return llm.bind_tools(lookup_tools)
    else:
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
    system_contents = [SYSTEM_PROMPT]
    other_messages = []
    for m in state["messages"]:
        if isinstance(m, SystemMessage):
            system_contents.append(m.content)
        else:
            other_messages.append(m)
    combined_system = SystemMessage(content="\n\n".join(system_contents))
    messages = [combined_system] + other_messages

    # Retry up to 4 times if the model returns an empty response.
    # LangChain raises ValueError BOTH inside llm.invoke() and in its own
    # post-invoke message validation, so we catch all exceptions here.
    last_exc = None
    for attempt in range(4):
        try:
            # On later retries, add a nudge to force the model to produce output
            invoke_messages = messages
            if attempt > 0:
                nudge = HumanMessage(content="Please respond now with a short conversational reply.")
                invoke_messages = messages + [nudge]
                print(f"[WARN] Retry {attempt}: adding nudge message", flush=True)

            response = llm.invoke(invoke_messages)
            has_content = bool(getattr(response, "content", None) and str(response.content).strip())
            has_tools = bool(getattr(response, "tool_calls", None))
            if has_content or has_tools:
                return {"messages": [response]}
            # Response object exists but has no content and no tool calls — retry
            print(f"[WARN] Empty LLM response object on attempt {attempt + 1}, retrying...", flush=True)
        except (ValueError, Exception) as e:
            last_exc = e
            err_str = str(e).lower()
            is_empty_err = (
                "model output" in err_str
                or "empty" in err_str
                or "tool calls" in err_str
                or "output text" in err_str
                or "cannot both be empty" in err_str
            )
            if is_empty_err:
                print(f"[WARN] Empty LLM response on attempt {attempt + 1}: {e}, retrying...", flush=True)
            else:
                # Not an empty-response error — re-raise immediately
                raise

    # All retries exhausted — return a graceful fallback message
    print(f"[ERROR] All LLM retries exhausted. Last error: {last_exc}", flush=True)
    from langchain_core.messages import AIMessage as _AIMessage
    fallback = _AIMessage(content="I'm having a bit of trouble right now. Could you please rephrase or repeat your message?")
    return {"messages": [fallback]}


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


def run_agent_turn(
    history: list[dict], 
    user_message: str, 
    active_hcp_id: Optional[str] = None,
    draft_interaction: dict = None
) -> dict:
    """Convert chat history + new message and run the agent turn."""
    import time
    start_time = time.perf_counter()
    print(f"\n[TIMING] === run_agent_turn started for message: '{user_message}' ===", flush=True)

    clean_msg = user_message.strip().lower().replace(",", "").replace(".", "").replace("]", "").replace("[", "")

    # Resolve initial_active_hcp early — used by all intercepts below
    initial_active_hcp = None
    if active_hcp_id:
        db = SessionLocal()
        try:
            hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
            if hcp:
                initial_active_hcp = {
                    "id": str(hcp.id),
                    "name": hcp.name,
                    "specialty": hcp.specialty,
                    "institution": hcp.institution,
                    "email": hcp.email,
                    "phone": hcp.phone,
                }
        finally:
            db.close()

    # --- Early Doctor Switch Detection ---
    # Detect if user says "actually it was Dr ..." or "wait I think it was Dr ..."
    # we clear the selected doctor context so that proactive resolution can pick up the new name.
    force_cutoff = False
    if active_hcp_id and initial_active_hcp:
        try:
            from app.agent.tools import normalize_name
            norm_active = normalize_name(initial_active_hcp["name"])
            user_msg_lower = user_message.lower()
            user_words = user_msg_lower.replace(".", "").replace(",", "").split()

            has_switch_phrase = any(p in user_msg_lower for p in ["it was", "no it", "no no", "actually", "i think it was"])
            has_met_phrase = any(p in user_msg_lower for p in ["met dr", "met doctor"])

            if has_switch_phrase or has_met_phrase:
                words_after_dr = []
                found_dr = False
                for w in user_words:
                    if w in ("dr", "dr.", "doctor"):
                        found_dr = True
                        continue
                    if found_dr:
                        if w not in ("meeting", "happend", "with", "him", "about", "today", "yesterday", "cancer", "treatment", "some", "saved", "doc"):
                            words_after_dr.append(w)
                        else:
                            break

                if words_after_dr:
                    name_mentioned = " ".join(words_after_dr)
                    if name_mentioned not in norm_active and norm_active not in name_mentioned:
                        print(f"[TIMING] Early switch check cleared doctor '{initial_active_hcp['name']}' (mentioned: {name_mentioned})", flush=True)
                        print(f"[DEBUG] Mid-conversation doctor switch detected: '{name_mentioned}' vs '{norm_active}'. Clearing active_hcp_id.", flush=True)
                        active_hcp_id = None
                        force_cutoff = True
        except Exception as e:
            print(f"[ERROR] Early switch detection failed: {e}", flush=True)

    # Truncate history to only include messages after the last interaction save/reset
    cutoff_index = len(history) if force_cutoff else 0
    if not force_cutoff:
        for i, m in enumerate(history):
            content = m["content"].lower()
            if m["role"] == "assistant":
                if "your interaction has been saved successfully" in content or \
                   "i've successfully recorded your interaction" in content or \
                   "what else can i do for you" in content:
                    cutoff_index = i + 1
                elif "what type of interaction did you have?" in content or \
                     "could you please provide the doctor's full name" in content or \
                     "let's start fresh with a new doctor" in content:
                    cutoff_index = i
            elif m["role"] == "user":
                clean_user = content.strip().replace(",", "").replace(".", "")
                if clean_user == "log interaction" or clean_user.startswith("wait actually change doctor") or clean_user.startswith("actually no another"):
                    cutoff_index = i

    # --- Mid-draft Switch / Reset Doctor Intercept ---
    clean_msg_no_punc = clean_msg.replace(",", "").replace(".", "").strip()
    if clean_msg_no_punc in (
        "no another doctor", "actually another doctor", "switch doctor", "change doctor",
        "different doctor", "actually different doctor", "no different doctor",
        "no log with another doctor", "log with another doctor", "use another doctor",
        "choose another doctor", "switch to another doctor", "another doctor please", "another doctor"
    ):
        print(f"[TIMING] Intercepted mid-draft doctor reset/switch in Python", flush=True)
        return {
            "reply": "Ok, I've cleared the selected doctor. Please tell me the name of the new doctor you would like to log this interaction for.",
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": None,
                "hcp": None,
                "draft_hcp": None,
                "compliance_warnings": []
            }
        }

    # --- Intercept: user sends a message that looks purely like a doctor name ---
    # ALWAYS resolve in Python if the user's entire message is just a doctor name.
    # This eliminates:
    #   1. LLM hallucinating "I found X" for non-existent doctors
    #   2. 60+ second LangGraph timeouts from confused LLM context
    #   3. LLM asking "could you provide the full name?" for complete names
    last_assistant_msg_content = ""
    if history:
        for m in reversed(history):
            if m["role"] == "assistant":
                last_assistant_msg_content = m["content"].lower()
                break

    user_words_clean = user_message.strip().replace(",", "").replace(".", "").lower().split()
    user_looks_like_pure_doctor_name = (
        len(user_words_clean) >= 2
        and len(user_words_clean) <= 4
        and user_words_clean[0] in ("dr", "doctor")
        and not any(w in user_words_clean for w in (
            "meeting", "call", "email", "video", "said", "has", "is", "was",
            "please", "could", "would", "like", "want"
        ))
    )
    new_doctor_name_from_msg = user_message.strip() if user_looks_like_pure_doctor_name else None

    # Fire Python-side resolution for pure doctor name messages when:
    # - No active HCP (or active HCP is different from the name being typed)
    # - The message is purely a doctor name (no other content)
    should_python_resolve = (
        user_looks_like_pure_doctor_name
        and new_doctor_name_from_msg
        and (
            not active_hcp_id  # No doctor selected yet
            or (  # Or the user is typing a DIFFERENT name than the active one
                initial_active_hcp
                and new_doctor_name_from_msg
            )
        )
    )

    if should_python_resolve:
        print(f"[TIMING] Python-side resolution for pure doctor name '{new_doctor_name_from_msg}'", flush=True)
        db = SessionLocal()
        try:
            from app.agent.tools import normalize_name
            import difflib
            all_hcps = db.query(HCP).all()
            normalized_search = normalize_name(new_doctor_name_from_msg)

            # Check if same as active HCP — no need to do anything
            if active_hcp_id and initial_active_hcp:
                norm_active = normalize_name(initial_active_hcp.get("name", ""))
                if normalized_search == norm_active or normalized_search in norm_active or norm_active in normalized_search:
                    print(f"[TIMING] Python-side: same as active HCP, skipping intercept", flush=True)
                    db.close()
                    should_python_resolve = False

            if should_python_resolve:
                resolved_hcp = None
                for h in all_hcps:
                    norm_db_name = normalize_name(h.name)
                    if normalized_search and norm_db_name:
                        if normalized_search in norm_db_name or norm_db_name in normalized_search:
                            resolved_hcp = h
                            break
                        else:
                            ratio = difflib.SequenceMatcher(None, normalized_search, norm_db_name).ratio()
                            if ratio > 0.82:
                                resolved_hcp = h
                                break
                if resolved_hcp:
                    print(f"[TIMING] Python-side resolved to '{resolved_hcp.name}' (ID: {resolved_hcp.id})", flush=True)
                    active_hcp_id = str(resolved_hcp.id)
                    initial_active_hcp = {
                        "id": str(resolved_hcp.id),
                        "name": resolved_hcp.name,
                        "specialty": resolved_hcp.specialty,
                        "institution": resolved_hcp.institution,
                        "email": resolved_hcp.email,
                        "phone": resolved_hcp.phone,
                    }
                    hcp_display_name = resolved_hcp.name
                    if hcp_display_name.lower().startswith("dr. "): hcp_display_name = hcp_display_name[4:]
                    elif hcp_display_name.lower().startswith("dr "): hcp_display_name = hcp_display_name[3:]
                    return {
                        "reply": f"I found Dr. {hcp_display_name}'s profile in our database. Would you like to log an interaction for Dr. {hcp_display_name}? [Log Interaction]",
                        "tool_calls": [],
                        "state": {
                            "interaction": None,
                            "draft_interaction": None,
                            "hcp": initial_active_hcp,
                            "draft_hcp": None,
                            "compliance_warnings": []
                        }
                    }
                else:
                    # Not found — return immediately, no LLM
                    print(f"[TIMING] Python-side: '{new_doctor_name_from_msg}' not found in DB", flush=True)
                    return {
                        "reply": f"I couldn't find {new_doctor_name_from_msg} in our database. Could you tell me their specialty and hospital/institution so I can add them?",
                        "tool_calls": [],
                        "state": {
                            "interaction": None,
                            "draft_interaction": None,
                            "hcp": None,
                            "draft_hcp": {"name": new_doctor_name_from_msg},
                            "compliance_warnings": []
                        }
                    }
        finally:
            db.close()

    # --- Early Draft Extraction & Proactive HCP Resolution/Creation ---

    early_messages = []
    for m in history[cutoff_index:]:
        if m["role"] == "user":
            early_messages.append(HumanMessage(content=m["content"]))
        else:
            early_messages.append(AIMessage(content=m["content"]))
    early_messages.append(HumanMessage(content=user_message))
    
    early_draft_interaction, early_draft_hcp = _extract_draft_state(early_messages, draft_interaction)

    if not active_hcp_id and early_draft_hcp and early_draft_hcp.get("name"):
        db = SessionLocal()
        try:
            from app.agent.tools import normalize_name
            import difflib
            all_hcps = db.query(HCP).all()
            normalized_search = normalize_name(early_draft_hcp["name"])
            resolved_hcp = None
            for h in all_hcps:
                norm_db_name = normalize_name(h.name)
                if normalized_search and norm_db_name:
                    if normalized_search in norm_db_name or norm_db_name in normalized_search:
                        resolved_hcp = h
                        break
                    else:
                        ratio = difflib.SequenceMatcher(None, normalized_search, norm_db_name).ratio()
                        if ratio > 0.82:
                            resolved_hcp = h
                            break
            if resolved_hcp:
                print(f"[TIMING] Proactively resolved active HCP to '{resolved_hcp.name}' (ID: {resolved_hcp.id})", flush=True)
                active_hcp_id = str(resolved_hcp.id)
                initial_active_hcp = {
                    "id": str(resolved_hcp.id),
                    "name": resolved_hcp.name,
                    "specialty": resolved_hcp.specialty,
                    "institution": resolved_hcp.institution,
                    "email": resolved_hcp.email,
                    "phone": resolved_hcp.phone,
                }
        finally:
            db.close()

    # Proactive creation check if details are provided and hcp not yet resolved
    if not active_hcp_id and early_draft_hcp and early_draft_hcp.get("name") and (early_draft_hcp.get("specialty") or early_draft_hcp.get("institution")):
        from app.agent.tools import add_hcp as _add_hcp_proactive
        res_str = _add_hcp_proactive.func(
            name=early_draft_hcp["name"],
            specialty=early_draft_hcp.get("specialty"),
            institution=early_draft_hcp.get("institution"),
            email=early_draft_hcp.get("email"),
            phone=early_draft_hcp.get("phone"),
        )
        try:
            res_data = json.loads(res_str)
            if "error" not in res_data:
                hcp_data = res_data.get("hcp", {})
                dname = hcp_data.get("name", "")
                if dname.lower().startswith("dr. "): dname = dname[4:]
                elif dname.lower().startswith("dr "): dname = dname[3:]
                print(f"[TIMING] Proactively created profile for Dr. {dname} (ID: {hcp_data.get('id')})", flush=True)
                return {
                    "reply": f"I've successfully created the profile for Dr. {dname}. Would you like to log an interaction for Dr. {dname} now? [Log Interaction] [Just Add Profile]",
                    "tool_calls": [{"tool": "add_hcp", "args": early_draft_hcp}],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": hcp_data,
                        "draft_hcp": None,
                        "compliance_warnings": []
                    }
                }
        except Exception as e:
            print(f"[ERROR] Proactive creator failed: {e}", flush=True)

    # 1. Intercept "Save Interaction" (Save confirmation)
    is_save_intent = False
    if clean_msg in ("save interaction", "save", "yes save", "save it", "yes save it", "confirm save", "yes, save", "yes, save it"):
        is_save_intent = True
    elif history:
        last_assistant_msg = next((m for m in reversed(history) if m["role"] == "assistant"), None)
        if last_assistant_msg and "save this interaction?" in last_assistant_msg["content"].lower():
            if clean_msg in ("yes", "y", "yep", "yeah", "sure", "ok", "okay", "save"):
                is_save_intent = True

    if is_save_intent:
        print(f"[TIMING] Intercepted Save Interaction command in Python", flush=True)
        temp_messages = []
        if active_hcp_id:
            db = SessionLocal()
            try:
                hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
                if hcp:
                    temp_messages.append(SystemMessage(
                        content=f"Context: The representative currently has HCP '{hcp.name}' (UUID: '{hcp.id}') selected on their screen. "
                                f"Use this UUID '{hcp.id}' as the hcp_id in tool calls when referring to this doctor."
                    ))
            finally:
                db.close()
        # Pass the full conversation history to ensure _extract_draft_state doesn't miss any populated fields
        for m in history:
            if m["role"] == "user":
                temp_messages.append(HumanMessage(content=m["content"]))
            else:
                temp_messages.append(AIMessage(content=m["content"]))
                
        draft_interaction, draft_hcp = _extract_draft_state(temp_messages, draft_interaction)
        
        if not active_hcp_id:
            # No doctor is selected — can't save without one
            return {
                "reply": "I couldn't save the interaction because no doctor profile is selected. Please tell me the doctor's name first so I can look them up.",
                "tool_calls": [],
                "state": {
                    "interaction": None,
                    "draft_interaction": draft_interaction,
                    "hcp": initial_active_hcp,
                    "draft_hcp": None,
                    "compliance_warnings": []
                }
            }

        if active_hcp_id:

            topics = (draft_interaction.get("topics_discussed") or []) if draft_interaction else []
            products = (draft_interaction.get("products_discussed") or []) if draft_interaction else []
            if not products:
                products = ["No Medication Discussed"]
            notes = (draft_interaction.get("raw_notes") or "Meeting notes logged.") if draft_interaction else "Meeting notes logged."
            samples = (draft_interaction.get("samples_provided") or []) if draft_interaction else []
            itype = (draft_interaction.get("interaction_type") or "Meeting") if draft_interaction else "Meeting"
            sent = (draft_interaction.get("sentiment") or "neutral") if draft_interaction else "neutral"
            
            from app.agent.tools import log_interaction
            res_str = log_interaction.func(
                hcp_id=active_hcp_id,
                interaction_type=itype,
                sentiment=sent,
                topics_discussed=topics,
                products_discussed=products,
                raw_text=notes,
                samples_provided=samples
            )
            
            try:
                res_data = json.loads(res_str)
                if "error" not in res_data:
                    db = SessionLocal()
                    hcp_name = "the doctor"
                    hcp_profile = None
                    try:
                        hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
                        if hcp:
                            hcp_name = hcp.name
                            hcp_profile = {
                                "id": str(hcp.id),
                                "name": hcp.name,
                                "specialty": hcp.specialty,
                                "institution": hcp.institution,
                                "email": hcp.email,
                                "phone": hcp.phone
                            }
                    finally:
                        db.close()
                        
                    from app.agent.tools import check_compliance
                    
                    db = SessionLocal()
                    try:
                        saved_interaction = db.query(Interaction).filter(Interaction.id == res_data["interaction_id"]).first()
                        
                        # Dynamically check compliance warnings
                        warn_list = []
                        try:
                            warn_str = check_compliance.func(hcp_id=active_hcp_id, samples_provided=samples)
                            warn_data = json.loads(warn_str)
                            if isinstance(warn_data, dict):
                                warn_list = warn_data.get("warnings", [])
                        except Exception:
                            pass
                        
                        # Strip existing "Dr." prefix so we don't double it in the reply
                        display_name = hcp_name
                        if display_name.lower().startswith("dr. "):
                            display_name = display_name[4:]
                        elif display_name.lower().startswith("dr "):
                            display_name = display_name[3:]
                        return {
                            "reply": f"Thank you! I've successfully recorded your interaction with Dr. {display_name}. Would you like to log another interaction? With the same doctor or another doctor? [Same Doctor] [Another Doctor]",
                            "tool_calls": [{"tool": "log_interaction", "args": {
                                "hcp_id": active_hcp_id,
                                "interaction_type": itype,
                                "sentiment": sent,
                                "topics_discussed": topics,
                                "products_discussed": products,
                                "raw_notes": notes,
                                "samples_provided": samples
                            }}],
                            "state": {
                                "interaction": {
                                    "id": str(saved_interaction.id),
                                    "hcp_id": str(saved_interaction.hcp_id),
                                    "interaction_type": saved_interaction.interaction_type,
                                    "sentiment": saved_interaction.sentiment,
                                    "summary": saved_interaction.summary,
                                    "topics_discussed": saved_interaction.topics_discussed,
                                    "products_discussed": saved_interaction.products_discussed,
                                    "raw_notes": saved_interaction.raw_notes,
                                    "samples_provided": saved_interaction.samples_provided,
                                    "interaction_date": saved_interaction.interaction_date.isoformat() if saved_interaction.interaction_date else None
                                },
                                "draft_interaction": None,
                                "hcp": hcp_profile,
                                "compliance_warnings": warn_list
                            }
                        }
                    finally:
                        db.close()
                else:
                    return {
                        "reply": f"Sorry, I encountered an error saving the interaction: {res_data['error']}",
                        "tool_calls": [],
                        "state": {
                            "interaction": None,
                            "draft_interaction": draft_interaction,
                            "hcp": None,
                            "compliance_warnings": []
                        }
                    }
            except Exception as e:
                print(f"[ERROR] Direct save parsing failed: {e}", flush=True)
                return {
                    "reply": "I encountered an error trying to save the interaction. Please try again.",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": draft_interaction,
                        "hcp": None,
                        "compliance_warnings": []
                    }
                }

    # 2. Intercept "Just Add Profile"
    if clean_msg in ("just add profile", "just add"):
        print(f"[TIMING] Intercepted Just Add Profile command in Python", flush=True)
        # Fetch HCP profile so it's preserved in frontend state
        just_add_hcp = initial_active_hcp
        if active_hcp_id and not just_add_hcp:
            db = SessionLocal()
            try:
                hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
                if hcp:
                    just_add_hcp = {"id": str(hcp.id), "name": hcp.name, "specialty": hcp.specialty, "institution": hcp.institution, "email": hcp.email, "phone": hcp.phone}
            finally:
                db.close()
        return {
            "reply": "Profile saved. Is there anything else you'd like to do? [Yes] [No]",
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": None,
                "hcp": just_add_hcp,
                "compliance_warnings": []
            }
        }

    # 3. Intercept "+" — user typed "+" meaning "add the draft doctor"
    if clean_msg == "+":
        print(f"[TIMING] Intercepted '+' command — extracting draft HCP and saving", flush=True)
        temp_msgs = []
        for m in history:
            content = m.get("content", "") or ""
            if content.strip():
                if m["role"] == "user":
                    temp_msgs.append(HumanMessage(content=content))
                else:
                    temp_msgs.append(AIMessage(content=content))

        _, draft_hcp = _extract_draft_state(temp_msgs, draft_interaction)

        if draft_hcp and draft_hcp.get("name"):
            from app.agent.tools import add_hcp as _add_hcp_tool
            res_str = _add_hcp_tool.func(
                name=draft_hcp["name"],
                specialty=draft_hcp.get("specialty"),
                institution=draft_hcp.get("institution"),
                email=draft_hcp.get("email"),
                phone=draft_hcp.get("phone"),
            )
            try:
                res_data = json.loads(res_str)
                if "error" not in res_data:
                    hcp_data = res_data.get("hcp", {})
                    dname = hcp_data.get("name", "")
                    if dname.lower().startswith("dr. "):
                        dname = dname[4:]
                    elif dname.lower().startswith("dr "):
                        dname = dname[3:]
                    return {
                        "reply": f"I've created the profile for Dr. {dname}. Would you like to log an interaction for Dr. {dname} now? [Log Interaction] [Just Add Profile]",
                        "tool_calls": [{"tool": "add_hcp", "args": draft_hcp}],
                        "state": {
                            "interaction": None,
                            "draft_interaction": None,
                            "hcp": hcp_data,
                            "draft_hcp": None,
                            "compliance_warnings": []
                        }
                    }
            except Exception as e:
                print(f"[ERROR] '+' intercept add_hcp failed: {e}", flush=True)

        # No draft found — guide the user
        return {
            "reply": "Please tell me the doctor's full name and I'll look them up or create a profile for them.",
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": None,
                "hcp": initial_active_hcp,
                "draft_hcp": None,
                "compliance_warnings": []
            }
        }

    # 4. Intercept "Log Interaction"
    if clean_msg == "log interaction":
        print(f"[TIMING] Intercepted Log Interaction command in Python", flush=True)
        db = SessionLocal()
        hcp_profile = None
        if active_hcp_id:
            try:
                hcp = db.query(HCP).filter(HCP.id == active_hcp_id).first()
                if hcp:
                    hcp_profile = {
                        "id": str(hcp.id),
                        "name": hcp.name,
                        "specialty": hcp.specialty,
                        "institution": hcp.institution,
                        "email": hcp.email,
                        "phone": hcp.phone
                    }
            finally:
                db.close()
        return {
            "reply": "What type of interaction did you have? Was it a meeting, video call, or email? [Meeting] [Video Call] [Email]",
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": {
                    "interaction_type": None,
                    "sentiment": None,
                    "topics_discussed": [],
                    "products_discussed": [],
                    "raw_notes": None,
                    "samples_provided": []
                },
                "hcp": hcp_profile,
                "compliance_warnings": []
            }
        }

    # Check if the user is replying to the post-save "log another interaction" question
    if history:
        last_assistant_msg = next((m for m in reversed(history) if m["role"] == "assistant"), None)
        if last_assistant_msg and ("Would you like to log another interaction?" in last_assistant_msg["content"] or "Would you like to log another HCP interaction?" in last_assistant_msg["content"]):
            clean_msg_strip = user_message.strip().lower().replace(",", "").replace(".", "")
            if clean_msg_strip in ("no", "no thanks", "no thank you", "nah", "nope", "n", "discard", "none"):
                print(f"[TIMING] Intercepted negative save confirmation reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "You're welcome! Have a great day.",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": initial_active_hcp,
                        "compliance_warnings": []
                    }
                }
            elif clean_msg_strip in ("same", "same doctor", "the same", "with the same doctor", "same doc"):
                print(f"[TIMING] Intercepted same doctor log another confirmation in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "Ok! What type of interaction did you have? Was it a meeting, video call, or email? [Meeting] [Video Call] [Email]",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": {
                            "interaction_type": None,
                            "sentiment": None,
                            "topics_discussed": [],
                            "products_discussed": [],
                            "raw_notes": None,
                            "samples_provided": []
                        },
                        "hcp": initial_active_hcp,
                        "draft_hcp": None,
                        "compliance_warnings": []
                    }
                }
            elif clean_msg_strip in ("another", "another doctor", "different", "different doctor", "another doc"):
                print(f"[TIMING] Intercepted another doctor log another confirmation in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "Ok! Please tell me the new doctor's full name and we can log this interaction.",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": None,
                        "draft_hcp": None,
                        "compliance_warnings": []
                    }
                }
            elif clean_msg_strip in ("yes", "y", "yep", "yeah", "sure", "ok", "okay"):
                print(f"[TIMING] Intercepted generic positive save confirmation reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "Would you like to log it with the same doctor or another doctor? [Same Doctor] [Another Doctor]",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": initial_active_hcp,
                        "draft_hcp": None,
                        "compliance_warnings": []
                    }
                }
            else:
                history = []
        
        # Intercept post-profile-edit negative reply
        if last_assistant_msg and "Is there anything else you'd like to update?" in last_assistant_msg["content"]:
            clean_msg_strip = user_message.strip().lower().replace(",", "").replace(".", "")
            if clean_msg_strip in ("no", "no thanks", "no thank you", "nah", "nope", "n"):
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
                
        # Intercept post-creation "Is there anything else you'd like to do?"
        if last_assistant_msg and "Is there anything else you'd like to do?" in last_assistant_msg["content"]:
            clean_msg_strip = user_message.strip().lower().replace(",", "").replace(".", "")
            if clean_msg_strip in ("no", "no thanks", "no thank you", "nah", "nope", "n"):
                print(f"[TIMING] Intercepted negative post-creation reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "Ok!",
                    "tool_calls": [],
                    "state": {
                        "interaction": None,
                        "draft_interaction": None,
                        "hcp": None,
                        "compliance_warnings": []
                    }
                }
            elif clean_msg_strip in ("yes", "yep", "yeah", "y"):
                print(f"[TIMING] Intercepted positive post-creation reply in {(time.perf_counter() - start_time)*1000:.2f} ms", flush=True)
                return {
                    "reply": "Ok! What else can I do for you?",
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
    
    # Detect if user is switching doctors mid-conversation (handled early)
    pass

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

    # cutoff_index is already calculated at the top of the function


    # Limit context history to the last 6 messages AFTER the cutoff
    effective_history = history[cutoff_index:]
    for m in effective_history[-6:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    # The frontend optimistic update might have already added the user message to history.
    if not (messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == user_message):
        messages.append(HumanMessage(content=user_message))

    final_messages = []
    reply = ""
    tool_calls_made = []

    try:
        graph = get_graph()
        graph_start = time.perf_counter()
        print(f"[TIMING] Invoking LangGraph with cutoff_index {cutoff_index}...", flush=True)
        print(f"[DEBUG] Messages: {messages}", flush=True)
        result = graph.invoke({"messages": messages})
        print(f"[TIMING] LangGraph invoke finished in {(time.perf_counter() - graph_start)*1000:.2f} ms", flush=True)
        final_messages = result["messages"]
    except Exception as e:
        print(f"[ERROR] LangGraph execution failed: {e}", flush=True)
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            reply = "I'm experiencing temporary rate limits. Please wait a few seconds and try sending your message again."
        elif "model output" in error_msg.lower() or "tool calls" in error_msg.lower() or "empty" in error_msg.lower():
            reply = "I had a momentary hiccup — please send your message again and I'll pick up right where we left off."
        else:
            reply = "I encountered a temporary issue processing your request. Please try again."
        return {
            "reply": reply,
            "tool_calls": [],
            "state": {
                "interaction": None,
                "draft_interaction": draft_interaction,
                "hcp": initial_active_hcp,
                "draft_hcp": None,
                "compliance_warnings": []
            }
        }

    
    # Check if log_interaction or add_hcp was called and executed successfully
    log_tool_message = None
    add_hcp_tool_message = None
    for msg in reversed(final_messages):
        if isinstance(msg, ToolMessage):
            if msg.name == "log_interaction":
                log_tool_message = msg
            elif msg.name == "add_hcp":
                add_hcp_tool_message = msg
            
    for msg in final_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls_made.extend([{"tool": tc["name"], "args": tc["args"]} for tc in msg.tool_calls])

    if log_tool_message:
        try:
            tool_data = json.loads(log_tool_message.content)
            if "error" not in tool_data:
                hcp_name = tool_data.get("hcp_name", "the doctor")
                # Strip existing "Dr." prefix so we don't double it
                if hcp_name.lower().startswith("dr. "):
                    hcp_name = hcp_name[4:]
                elif hcp_name.lower().startswith("dr "):
                    hcp_name = hcp_name[3:]
                reply = f"Thank you! I've successfully recorded your interaction with Dr. {hcp_name}. Would you like to log another HCP interaction? [Yes] [No]"
            else:
                reply = f"Sorry, I encountered an error saving the interaction: {tool_data['error']}"
        except Exception:
            reply = "Your interaction has been saved successfully. Thank you! Would you like to record another interaction? [Yes] [No]"
    elif add_hcp_tool_message:
        try:
            tool_data = json.loads(add_hcp_tool_message.content)
            if "error" not in tool_data:
                hcp_data = tool_data.get("hcp", {})
                hcp_name = hcp_data.get("name", "the doctor")
                # Strip existing "Dr." prefix to avoid double-prefix
                if hcp_name.lower().startswith("dr. "):
                    hcp_name = hcp_name[4:]
                elif hcp_name.lower().startswith("dr "):
                    hcp_name = hcp_name[3:]
                tool_msg = tool_data.get("message", "")
                if "already exists" in tool_msg.lower():
                    reply = f"I found Dr. {hcp_name} in our database. Would you like to log an interaction for Dr. {hcp_name} now, or did you just want to view their profile? [Log Interaction] [Just View Profile]"
                else:
                    reply = f"I've successfully created the profile for Dr. {hcp_name}. Would you like to log an interaction for Dr. {hcp_name} now, or did you just want to add their profile? [Log Interaction] [Just Add Profile]"
            else:
                reply = f"Sorry, I encountered an error creating the profile: {tool_data['error']}"
        except Exception:
            reply = "I've successfully created the HCP profile. Would you like to log an interaction for this doctor now, or did you just want to add their profile? [Log Interaction] [Just Add Profile]"
    else:
        for msg in final_messages:
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content  # last non-empty AI content wins

    # ── Leaked tool call recovery ─────────────────────────────────────────────
    # Groq/Llama sometimes outputs tool calls as raw text instead of tool_calls field.
    # Detect and execute them so the interaction isn't lost.
    if reply and not log_tool_message and not add_hcp_tool_message:
        import re
        leaked_match = re.search(
            r'(?:function=|<function[_\s]calls?>\s*(?:<invoke[^>]*>)?\s*)'
            r'(add_hcp|log_interaction|resolve_hcp_by_name)'
            r'[>\s]*(?:<parameter[^>]*>)?\s*'
            r'(\{[^}]+(?:\}[^}]*)*\})',
            reply, re.DOTALL | re.IGNORECASE
        )
        if not leaked_match:
            # Alternative pattern: just a JSON blob with tool-like keys
            if 'add_hcp' in reply.lower() and '"name"' in reply:
                leaked_match = re.search(r'(\{[^{}]*"name"[^{}]*\})', reply)
                if leaked_match:
                    leaked_tool = 'add_hcp'
                    leaked_args_str = leaked_match.group(1)
                else:
                    leaked_tool = None
                    leaked_args_str = None
            else:
                leaked_tool = None
                leaked_args_str = None
        else:
            leaked_tool = leaked_match.group(1)
            leaked_args_str = leaked_match.group(2)

        if leaked_tool and leaked_args_str:
            print(f"[WARN] Detected leaked tool call in reply: {leaked_tool}", flush=True)
            try:
                leaked_args = json.loads(leaked_args_str)
                if leaked_tool == "add_hcp":
                    from app.agent.tools import add_hcp as _add_hcp_leaked
                    res_str = _add_hcp_leaked.func(
                        name=leaked_args.get("name"),
                        specialty=leaked_args.get("specialty"),
                        institution=leaked_args.get("institution"),
                        email=leaked_args.get("email"),
                        phone=leaked_args.get("phone"),
                    )
                    res_data = json.loads(res_str)
                    if "error" not in res_data:
                        add_hcp_tool_message = type('FakeToolMsg', (), {
                            'content': res_str, 'name': 'add_hcp'
                        })()
                        # Re-run the add_hcp reply path
                        hcp_data = res_data.get("hcp", {})
                        dname = hcp_data.get("name", "")
                        if dname.lower().startswith("dr. "): dname = dname[4:]
                        elif dname.lower().startswith("dr "): dname = dname[3:]
                        msg_txt = res_data.get("message", "")
                        if "already exists" in msg_txt.lower():
                            reply = f"I found Dr. {dname} in our database. Would you like to log an interaction for Dr. {dname} now? [Log Interaction] [Just View Profile]"
                        else:
                            reply = f"I've successfully created the profile for Dr. {dname}. Would you like to log an interaction for Dr. {dname} now? [Log Interaction] [Just Add Profile]"
                        # Inject the HCP id so it gets picked up in the scan below
                        final_messages.append(ToolMessage(content=res_str, name="add_hcp", tool_call_id="recovered"))
                elif leaked_tool == "resolve_hcp_by_name":
                    from app.agent.tools import resolve_hcp_by_name as _resolve_hcp_leaked
                    res_str = _resolve_hcp_leaked.func(name=leaked_args.get("name"))
                    res_data = json.loads(res_str)
                    matches = res_data.get("matches", [])
                    if matches:
                        hcp_match = matches[0]
                        dname = hcp_match.get("name", "")
                        if dname.lower().startswith("dr. "): dname = dname[4:]
                        elif dname.lower().startswith("dr "): dname = dname[3:]
                        reply = f"I've successfully found Dr. {dname}'s profile. Would you like to log an interaction for Dr. {dname} now? [Log Interaction] [Just View Profile]"
                    else:
                        dname = leaked_args.get("name", "the doctor")
                        reply = f"I couldn't find a doctor named {dname} in our database. Could you tell me their specialty and hospital/institution so I can add them?"
                    final_messages.append(ToolMessage(content=res_str, name="resolve_hcp_by_name", tool_call_id="recovered"))
                elif leaked_tool == "log_interaction":
                    from app.agent.tools import log_interaction as _log_interaction_leaked
                    res_str = _log_interaction_leaked.func(
                        hcp_id=leaked_args.get("hcp_id"),
                        raw_text=leaked_args.get("raw_text") or leaked_args.get("raw_notes") or "Meeting logged.",
                        channel=leaked_args.get("channel", "chat"),
                        interaction_type=leaked_args.get("interaction_type"),
                        sentiment=leaked_args.get("sentiment"),
                        topics_discussed=leaked_args.get("topics_discussed"),
                        products_discussed=leaked_args.get("products_discussed"),
                        samples_provided=leaked_args.get("samples_provided"),
                    )
                    res_data = json.loads(res_str)
                    if "error" not in res_data:
                        log_tool_message = type('FakeToolMsg', (), {
                            'content': res_str, 'name': 'log_interaction'
                        })()
                        hcp_name = res_data.get("hcp_name", "the doctor")
                        if hcp_name.lower().startswith("dr. "): hcp_name = hcp_name[4:]
                        elif hcp_name.lower().startswith("dr "): hcp_name = hcp_name[3:]
                        reply = f"Thank you! I've successfully recorded your interaction with Dr. {hcp_name}. Would you like to log another HCP interaction? [Yes] [No]"
                    else:
                        reply = f"Sorry, I encountered an error saving the interaction: {res_data['error']}"
                    final_messages.append(ToolMessage(content=res_str, name="log_interaction", tool_call_id="recovered"))
            except Exception as lex:
                print(f"[ERROR] Leaked tool call recovery failed: {lex}", flush=True)

    # ── Reply sanitizer: strip any remaining raw tool call artifacts ──────────

    if reply:
        import re
        _TOOL_NAMES = r'(?:add_hcp|resolve_hcp_by_name|log_interaction|enrich_hcp_profile|check_compliance|get_hcp_history|schedule_follow_up)'
        # With closing tag: function=tool_name>JSON</function>
        reply = re.sub(rf'function={_TOOL_NAMES}>[^<]*</function>', '', reply, flags=re.DOTALL).strip()
        # Without closing tag: function=tool_name>JSON (rest of line)
        reply = re.sub(rf'function={_TOOL_NAMES}>\s*\{{[^}}]*\}}', '', reply, flags=re.DOTALL).strip()
        # Without closing tag, trailing: function=tool_name>...anything to end
        reply = re.sub(rf'function={_TOOL_NAMES}>.*$', '', reply, flags=re.DOTALL | re.MULTILINE).strip()
        # XML-style: <function_calls>...</function_calls>
        reply = re.sub(r'<function_calls>.*?</function_calls>', '', reply, flags=re.DOTALL).strip()
        reply = re.sub(r'<invoke[^>]*>.*?</invoke>', '', reply, flags=re.DOTALL).strip()
        # Raw JSON blobs with tool arg shapes
        reply = re.sub(r'\{"name":\s*"[^"]+",\s*"specialty".*?\}', '', reply, flags=re.DOTALL).strip()
        if not reply:
            reply = "I've processed your request. What would you like to do next?"


    # Scan for interaction and HCP info
    interaction_id = None
    hcp_id = active_hcp_id
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
        draft_interaction, draft_hcp = _extract_draft_state(messages, draft_interaction)
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
2. The draft interaction details.

CRITICAL INSTRUCTION FOR STATE RETENTION:
You MUST output the ENTIRE draft state including all fields established so far in the conversation. If the user only updates one field in their latest message (e.g. they say "wait it was actually negative", or "[Skip Notes]"), DO NOT set the other previously established fields (like interaction_type, topics_discussed, etc.) to null. Keep all previously established details in the JSON so they are preserved.

CRITICAL INSTRUCTION FOR ARRAYS (topics_discussed, products_discussed, samples_provided): 
You must evaluate EACH array field independently.
- If the user explicitly changes or mentions a specific array (e.g., they mention a topic, a medication, or samples), replace ONLY that specific array with the new values. 
- If the user's latest message does NOT explicitly mention or change a specific array, you MUST perfectly carry over that array's existing values from the draft interaction state. 
- For example, if the user says "[Skip Samples]", you should set `samples_provided` to `[]`, but you MUST keep `topics_discussed` and `products_discussed` exactly as they were. Do NOT clear them to `[]`.

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


def _extract_draft_state(messages: list, prev_draft_interaction: Optional[dict] = None) -> Tuple[Optional[dict], Optional[dict]]:
    """Extract both draft interaction and draft HCP details in a single LLM call to optimize latency.
    Falls back to merging with prev_draft_interaction to prevent data loss on LLM errors."""
    convo = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content
            # Skip messages with no meaningful content (e.g., AIMessages that only contain tool calls)
            if content and str(content).strip():
                convo.append({"role": role, "content": content})

    if not convo:
        return None, None

    system_prompt = DRAFT_STATE_EXTRACTION_PROMPT
    if prev_draft_interaction:
        system_prompt += f"\n\nCURRENT DRAFT INTERACTION STATE (CARRY THESE EXACT VALUES OVER UNLESS EXPLICITLY OVERRIDDEN BY THE LATEST MESSAGE):\n{json.dumps(prev_draft_interaction, indent=2)}"

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
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
        
        # Merge with previous state to prevent dropping fields if the LLM output is partial or confused
        if draft_interaction and prev_draft_interaction:
            merged = dict(prev_draft_interaction)
            for k, v in draft_interaction.items():
                if v is not None:
                    merged[k] = v
            draft_interaction = merged
        elif prev_draft_interaction and not draft_interaction:
            draft_interaction = prev_draft_interaction

        if draft_interaction:
            # Normalize interaction_type to match React select inputs exactly (case sensitive)
            itype = draft_interaction.get("interaction_type")
            if itype and isinstance(itype, str):
                itype_lower = itype.lower().strip()
                if "meeting" in itype_lower:
                    draft_interaction["interaction_type"] = "Meeting"
                elif "video" in itype_lower or "call" in itype_lower or "phone" in itype_lower:
                    draft_interaction["interaction_type"] = "Video Call"
                elif "email" in itype_lower:
                    draft_interaction["interaction_type"] = "Email"
                    
            # Normalize sentiment to lowercase to match select inputs exactly
            sent = draft_interaction.get("sentiment")
            if sent and isinstance(sent, str):
                draft_interaction["sentiment"] = sent.lower().strip()

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
    except Exception as e:
        print(f"[WARN] Extract draft state failed: {e}. Falling back to previous state.", flush=True)
        return prev_draft_interaction, None
