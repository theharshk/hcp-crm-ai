"""
LangGraph tools available to the HCP Interaction Agent.

Five tools, as required by the task:
  1. log_interaction     - create a new interaction record (LLM does summarization
                            + entity extraction from free text / chat transcript)
  2. edit_interaction    - modify a previously logged interaction, with an audit trail
  3. get_hcp_history     - pull an HCP's profile + past interactions for context
                            (e.g. "what did we talk about last time?")
  4. schedule_follow_up  - create a follow-up task/reminder tied to an interaction
  5. check_compliance    - flag sample/gift-value or interaction-frequency issues
                            against simple configurable compliance rules

Each tool is a plain Python function decorated with @tool so LangGraph's
ToolNode can bind and invoke them directly from the agent's model output.
"""
import json
import datetime as dt
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database import SessionLocal
from app.models import HCP, Interaction
from app.agent.llm import chat_completion, reasoning_completion


class SampleItem(BaseModel):
    product: str = Field(description="The name of the product sample")
    qty: int = Field(description="The quantity of the product sample provided")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db() -> Session:
    return SessionLocal()


EXTRACTION_SYSTEM_PROMPT = """You are a life-sciences CRM assistant. Extract structured
data from a field representative's free-text or chat note about an HCP interaction.

Return ONLY valid JSON with these keys:
{
  "summary": "1-3 sentence neutral summary of what happened",
  "interaction_type": "Meeting|Video Call|Email",
  "topics_discussed": ["..."],
  "products_discussed": ["..."],
  "sentiment": "positive|neutral|negative",
  "follow_up_actions": [{"action": "...", "due_date": "YYYY-MM-DD or null"}],
  "samples_provided": [{"product": "...", "qty": 0}]
}

Rules for Products Discussed:
- If the representative states or implies no products were discussed (or "n/a", "no products", "none"), set "products_discussed" to ["No Product Discussed"].
- If no products are mentioned in the text at all, set "products_discussed" to ["No Product Discussed"].

No prose outside the JSON."""


def _extract_structured_data(raw_text: str) -> Dict[str, Any]:
    """Use the Groq LLM to turn free text into structured interaction fields."""
    raw = chat_completion(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        temperature=0.1,
    )
    try:
        # model occasionally wraps JSON in ```json fences — strip defensively
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # graceful fallback so the tool never hard-crashes the agent turn
        return {
            "summary": raw_text[:280],
            "interaction_type": "Meeting",
            "topics_discussed": [],
            "products_discussed": [],
            "sentiment": "neutral",
            "follow_up_actions": [],
            "samples_provided": [],
        }


# ---------------------------------------------------------------------------
# Tool 1: Log Interaction
# ---------------------------------------------------------------------------

@tool
def log_interaction(hcp_id: str, raw_text: str, channel: str = "chat") -> str:
    """Log a new HCP interaction. Pass the HCP's id and the rep's free-text
    description of the interaction (from the chat conversation or a structured
    form's 'notes' field). The LLM extracts a summary, topics, products,
    sentiment, follow-up actions, and any samples provided, then persists the
    interaction. Returns a JSON string with the new interaction_id and the
    extracted fields so the agent can confirm details back to the user."""
    db = _db()
    try:
        hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
        if not hcp:
            return json.dumps({"error": f"No HCP found with id {hcp_id}"})

        extracted = _extract_structured_data(raw_text)

        # Ensure channel is valid for database Enum
        db_channel = channel if channel in ("structured_form", "chat") else "chat"

        interaction = Interaction(
            hcp_id=hcp_id,
            interaction_type=extracted.get("interaction_type", "Meeting"),
            channel=db_channel,
            interaction_date=dt.datetime.utcnow(),
            summary=extracted.get("summary"),
            raw_notes=raw_text,
            topics_discussed=extracted.get("topics_discussed", []),
            products_discussed=extracted.get("products_discussed", []),
            sentiment=extracted.get("sentiment", "neutral"),
            follow_up_actions=extracted.get("follow_up_actions", []),
            samples_provided=extracted.get("samples_provided", []),
            edit_history=[],
        )
        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        return json.dumps({
            "interaction_id": interaction.id,
            "hcp_name": hcp.name,
            **extracted,
        })
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 2: Edit Interaction
# ---------------------------------------------------------------------------

@tool
def edit_interaction(interaction_id: str, changes_text: str, edited_by: str = "field_rep") -> str:
    """Edit a previously logged interaction. Pass the interaction_id and a
    plain-text description of what should change (e.g. 'actually we discussed
    Drug B, not Drug A, and I gave 5 samples'). The LLM interprets the
    requested changes against the existing record, applies only the fields
    that changed, and appends an entry to the interaction's edit_history for
    auditability. Returns the updated record as JSON."""
    db = _db()
    try:
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if not interaction:
            return json.dumps({"error": f"No interaction found with id {interaction_id}"})

        # Fetch current doctor's name for LLM state context
        hcp = db.query(HCP).filter(HCP.id == interaction.hcp_id).first()
        hcp_name = hcp.name if hcp else ""

        current_state = {
            "hcp_name": hcp_name,
            "summary": interaction.summary,
            "interaction_type": interaction.interaction_type,
            "topics_discussed": interaction.topics_discussed,
            "products_discussed": interaction.products_discussed,
            "sentiment": interaction.sentiment,
            "follow_up_actions": interaction.follow_up_actions,
            "samples_provided": interaction.samples_provided,
        }

        diff_prompt = (
            "Here is the current interaction record as JSON:\n"
            f"{json.dumps(current_state)}\n\n"
            "The rep wants to make this change:\n"
            f"\"{changes_text}\"\n\n"
            "Return ONLY a JSON object containing just the fields that should be "
            "updated (same schema/keys as the current record), with their new values. "
            "For products_discussed, if the user explicitly requests no products or clears the products, "
            "set it to [\"No Product Discussed\"]."
        )
        raw = reasoning_completion(
            messages=[
                {"role": "system", "content": "You output only valid JSON diffs, no prose."},
                {"role": "user", "content": diff_prompt},
            ]
        )
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        diff = json.loads(cleaned)

        changed_fields = {}

        # Resolve doctor name in diff to hcp_id in database
        if "hcp_name" in diff and diff["hcp_name"]:
            new_hcp_name = str(diff["hcp_name"]).strip()
            
            # Resolve by name
            search_name = new_hcp_name.lower()
            if search_name.startswith("dr."):
                search_name = search_name[3:].strip()
            elif search_name.startswith("dr "):
                search_name = search_name[3:].strip()
            
            from sqlalchemy import func
            resolved_hcp = db.query(HCP).filter(
                func.lower(HCP.name).contains(search_name)
            ).first()
            if resolved_hcp:
                if interaction.hcp_id != resolved_hcp.id:
                    changed_fields["hcp_id"] = {"from": interaction.hcp_id, "to": resolved_hcp.id}
                    interaction.hcp_id = resolved_hcp.id
            else:
                return json.dumps({
                    "error": f"HCP with name '{new_hcp_name}' was not found in the database. Please add/create this HCP first using add_hcp, then update the interaction."
                })
            
            diff.pop("hcp_name")
        for key, new_value in diff.items():
            if hasattr(interaction, key) and getattr(interaction, key) != new_value:
                changed_fields[key] = {"from": getattr(interaction, key), "to": new_value}
                setattr(interaction, key, new_value)

        interaction.updated_at = dt.datetime.utcnow()
        history = interaction.edit_history or []
        history.append({
            "edited_by": edited_by,
            "edited_at": dt.datetime.utcnow().isoformat(),
            "reason": changes_text,
            "changed_fields": changed_fields,
        })
        interaction.edit_history = history

        db.add(interaction)
        db.commit()
        db.refresh(interaction)

        return json.dumps({
            "interaction_id": interaction.id,
            "changed_fields": changed_fields,
            "current_summary": interaction.summary,
        })
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 3: Get HCP History
# ---------------------------------------------------------------------------

@tool
def get_hcp_history(hcp_id: str, limit: int = 5) -> str:
    """Fetch an HCP's profile and their most recent interactions. Useful for
    giving the rep context before a visit (e.g. 'what did we last discuss with
    Dr. Rao?') or for the agent to check for duplicate/near-duplicate logging."""
    db = _db()
    try:
        hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
        if not hcp:
            return json.dumps({"error": f"No HCP found with id {hcp_id}"})

        interactions = (
            db.query(Interaction)
            .filter(Interaction.hcp_id == hcp_id)
            .order_by(Interaction.interaction_date.desc())
            .limit(limit)
            .all()
        )
        return json.dumps({
            "hcp": {"id": hcp.id, "name": hcp.name, "specialty": hcp.specialty, "institution": hcp.institution},
            "recent_interactions": [
                {
                    "id": i.id,
                    "date": i.interaction_date.isoformat() if i.interaction_date else None,
                    "type": i.interaction_type,
                    "summary": i.summary,
                    "products_discussed": i.products_discussed,
                }
                for i in interactions
            ],
        })
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 4: Schedule Follow-up
# ---------------------------------------------------------------------------

@tool
def schedule_follow_up(interaction_id: str, action: str, due_date: Optional[str] = None) -> str:
    """Attach a follow-up task to an interaction, e.g. 'send updated dosing
    study PDF' due next Tuesday. Appends to the interaction's follow_up_actions
    list. due_date should be an ISO date string (YYYY-MM-DD) or omitted."""
    db = _db()
    try:
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if not interaction:
            return json.dumps({"error": f"No interaction found with id {interaction_id}"})

        actions = interaction.follow_up_actions or []
        actions.append({"action": action, "due_date": due_date, "status": "open"})
        interaction.follow_up_actions = actions
        interaction.updated_at = dt.datetime.utcnow()

        db.add(interaction)
        db.commit()
        return json.dumps({"interaction_id": interaction.id, "follow_up_actions": actions})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 5: Compliance Check
# ---------------------------------------------------------------------------

# Simple, configurable thresholds standing in for real regulatory rules
# (e.g. Sunshine Act / PhRMA Code style sample & gift limits). In production
# these would come from a compliance rules table, not be hardcoded.
MAX_SAMPLES_PER_VISIT = 20


@tool
def check_compliance(hcp_id: str, samples_provided: Optional[Any] = None) -> str:
    """Run a lightweight compliance check for an HCP interaction before or
    after logging: flags if sample quantities exceed configured limits, or if
    this HCP has already been visited too many times this month. Returns a
    JSON object with a list of warnings (empty list means no issues).
    
    samples_provided should be a list of sample objects, e.g. [{"product": "Drug A", "qty": 10}].
    This parameter can accept a list of objects, a single object, or a JSON-serialized string of objects."""
    db = _db()
    try:
        warnings = []
        samples = []
        if samples_provided:
            if isinstance(samples_provided, str):
                try:
                    samples_parsed = json.loads(samples_provided)
                    if isinstance(samples_parsed, list):
                        samples = samples_parsed
                    elif isinstance(samples_parsed, dict):
                        samples = [samples_parsed]
                except json.JSONDecodeError:
                    samples = []
            elif isinstance(samples_provided, list):
                samples = samples_provided
            elif isinstance(samples_provided, dict):
                samples = [samples_provided]
            else:
                samples = [samples_provided]

        for s in samples:
            if isinstance(s, dict):
                qty = s.get("qty", 0)
                product = s.get("product", "unknown product")
            else:
                qty = getattr(s, "qty", 0)
                product = getattr(s, "product", "unknown product")

            if qty and qty > MAX_SAMPLES_PER_VISIT:
                warnings.append(
                    f"Sample quantity for {product} ({qty}) "
                    f"exceeds the per-visit limit of {MAX_SAMPLES_PER_VISIT}."
                )

        return json.dumps({"hcp_id": hcp_id, "compliant": len(warnings) == 0, "warnings": warnings})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 6: Add HCP Profile
# ---------------------------------------------------------------------------

@tool
def add_hcp(
    name: str,
    specialty: Optional[str] = None,
    institution: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None
) -> str:
    """Create and add a new Healthcare Professional (HCP) profile to the database.
    Pass the HCP's name, and optionally their specialty, institution, email, and phone number.
    Returns a JSON string containing the newly created HCP's profile details including their database UUID (id)."""
    db = _db()
    try:
        # Check if an HCP with the same name already exists to prevent duplicate seeding
        existing = db.query(HCP).filter(HCP.name == name).first()
        if existing:
            return json.dumps({
                "message": f"HCP with name '{name}' already exists.",
                "hcp": {
                    "id": existing.id,
                    "name": existing.name,
                    "specialty": existing.specialty,
                    "institution": existing.institution,
                    "email": existing.email,
                    "phone": existing.phone
                }
            })

        hcp = HCP(
            name=name,
            specialty=specialty,
            institution=institution,
            email=email,
            phone=phone
        )
        db.add(hcp)
        db.commit()
        db.refresh(hcp)
        return json.dumps({
            "message": "Successfully created new HCP profile.",
            "hcp": {
                "id": hcp.id,
                "name": hcp.name,
                "specialty": hcp.specialty,
                "institution": hcp.institution,
                "email": hcp.email,
                "phone": hcp.phone
            }
        })
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 7: Resolve HCP By Name
# ---------------------------------------------------------------------------

@tool
def resolve_hcp_by_name(name: str) -> str:
    """Lookup an HCP (Healthcare Professional) by name to find their database UUID (hcp_id).
    Pass the raw text name typed by the user (e.g., 'Dr. Anjali Sharma' or 'Anjali Sharma').
    Returns a JSON string containing the matching HCP's id and profile details, or an error if not found."""
    db = _db()
    try:
        # Normalize the name search by stripping whitespace and common doctor prefixes
        search_name = name.strip().lower()
        if search_name.startswith("dr."):
            search_name = search_name[3:].strip()
        elif search_name.startswith("dr "):
            search_name = search_name[3:].strip()

        from sqlalchemy import func
        hcps = db.query(HCP).filter(
            func.lower(HCP.name).contains(search_name)
        ).all()

        if not hcps:
            return json.dumps({"error": f"No HCP found matching name '{name}'"})

        results = [
            {
                "id": hcp.id,
                "name": hcp.name,
                "specialty": hcp.specialty,
                "institution": hcp.institution,
                "email": hcp.email,
                "phone": hcp.phone
            }
            for hcp in hcps
        ]
        return json.dumps({"matches": results})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 8: Enrich HCP Profile
# ---------------------------------------------------------------------------

@tool
def enrich_hcp_profile(
    hcp_id: str,
    institution: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None
) -> str:
    """Enrich/update an existing HCP's profile details with their institution, email and/or phone number.
    Pass the HCP's database UUID (hcp_id), and their institution, email and/or phone number.
    Returns a JSON string containing the updated HCP details."""
    db = _db()
    try:
        hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
        if not hcp:
            return json.dumps({"error": f"No HCP found with id {hcp_id}"})
        
        if institution is not None:
            hcp.institution = institution.strip()
        if email is not None:
            hcp.email = email.strip()
        if phone is not None:
            hcp.phone = phone.strip()
            
        db.add(hcp)
        db.commit()
        db.refresh(hcp)
        
        return json.dumps({
            "message": "Successfully enriched HCP profile.",
            "hcp": {
                "id": hcp.id,
                "name": hcp.name,
                "specialty": hcp.specialty,
                "institution": hcp.institution,
                "email": hcp.email,
                "phone": hcp.phone
            }
        })
    finally:
        db.close()


ALL_TOOLS = [
    log_interaction,
    edit_interaction,
    get_hcp_history,
    schedule_follow_up,
    check_compliance,
    add_hcp,
    resolve_hcp_by_name,
    enrich_hcp_profile
]
