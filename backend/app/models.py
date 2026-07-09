import uuid
import datetime as dt

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class HCP(Base):
    """A Healthcare Professional the field rep engages with."""
    __tablename__ = "hcps"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False)
    specialty = Column(String(255))
    institution = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    interactions = relationship("Interaction", back_populates="hcp", cascade="all, delete-orphan")


class Interaction(Base):
    """A single logged interaction (call, visit, email, sample drop, etc.)."""
    __tablename__ = "interactions"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    hcp_id = Column(String(36), ForeignKey("hcps.id"), nullable=False)

    interaction_type = Column(
        Enum("Meeting", "Video Call", "Email", name="interaction_type"),
        default="Meeting",
    )
    channel = Column(
        Enum("structured_form", "chat", name="interaction_channel"), default="structured_form"
    )

    interaction_date = Column(DateTime, default=dt.datetime.utcnow)
    summary = Column(Text)                 # LLM-generated or user-provided summary
    raw_notes = Column(Text)               # original free-text / chat transcript
    topics_discussed = Column(JSON)        # e.g. ["Product X efficacy", "dosage Q"]
    products_discussed = Column(JSON)      # e.g. ["Drug A", "Drug B"]
    sentiment = Column(String(50))         # positive / neutral / negative (from LLM)
    follow_up_actions = Column(JSON)       # e.g. [{"action": "...", "due_date": "..."}]
    samples_provided = Column(JSON)        # e.g. [{"product": "Drug A", "qty": 10}]

    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    edit_history = Column(JSON, default=list)  # audit trail of edits (who/when/what changed)

    hcp = relationship("HCP", back_populates="interactions")
