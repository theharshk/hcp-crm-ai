import datetime as dt
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class HCPBase(BaseModel):
    name: str
    specialty: Optional[str] = None
    institution: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class HCPCreate(HCPBase):
    pass


class HCPOut(HCPBase):
    id: str
    created_at: dt.datetime

    class Config:
        from_attributes = True


class InteractionCreate(BaseModel):
    hcp_id: str
    interaction_type: str = "Meeting"
    channel: str = "structured_form"
    interaction_date: Optional[dt.datetime] = None
    summary: Optional[str] = None
    raw_notes: Optional[str] = None
    topics_discussed: Optional[List[str]] = None
    products_discussed: Optional[List[str]] = None
    sentiment: Optional[str] = None
    follow_up_actions: Optional[List[Dict[str, Any]]] = None
    samples_provided: Optional[List[Dict[str, Any]]] = None


class InteractionUpdate(BaseModel):
    """All fields optional — only supplied fields are changed."""
    interaction_type: Optional[str] = None
    interaction_date: Optional[dt.datetime] = None
    summary: Optional[str] = None
    raw_notes: Optional[str] = None
    topics_discussed: Optional[List[str]] = None
    products_discussed: Optional[List[str]] = None
    sentiment: Optional[str] = None
    follow_up_actions: Optional[List[Dict[str, Any]]] = None
    samples_provided: Optional[List[Dict[str, Any]]] = None
    edited_by: Optional[str] = "field_rep"
    edit_reason: Optional[str] = None


class InteractionOut(BaseModel):
    id: str
    hcp_id: str
    interaction_type: str
    channel: str
    interaction_date: Optional[dt.datetime]
    summary: Optional[str]
    raw_notes: Optional[str]
    topics_discussed: Optional[List[str]]
    products_discussed: Optional[List[str]]
    sentiment: Optional[str]
    follow_up_actions: Optional[List[Dict[str, Any]]]
    samples_provided: Optional[List[Dict[str, Any]]]
    created_at: dt.datetime
    updated_at: dt.datetime
    edit_history: Optional[List[Dict[str, Any]]]

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    hcp_id: Optional[str] = None
    session_id: str
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    interaction_id: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
