from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.agent.tools import check_compliance

router = APIRouter()


# --- HCPs -------------------------------------------------------------

@router.post("/hcps", response_model=schemas.HCPOut)
def create_hcp(payload: schemas.HCPCreate, db: Session = Depends(get_db)):
    hcp = models.HCP(**payload.model_dump())
    db.add(hcp)
    db.commit()
    db.refresh(hcp)
    return hcp


@router.get("/hcps", response_model=list[schemas.HCPOut])
def list_hcps(db: Session = Depends(get_db)):
    return db.query(models.HCP).order_by(models.HCP.name).all()


# --- Interactions (structured-form path) -------------------------------

@router.post("/interactions", response_model=schemas.InteractionOut)
def create_interaction(payload: schemas.InteractionCreate, db: Session = Depends(get_db)):
    hcp = db.query(models.HCP).filter(models.HCP.id == payload.hcp_id).first()
    if not hcp:
        raise HTTPException(404, "HCP not found")

    interaction = models.Interaction(**payload.model_dump(), edit_history=[])
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


@router.get("/interactions", response_model=list[schemas.InteractionOut])
def list_interactions(hcp_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(models.Interaction)
    if hcp_id:
        q = q.filter(models.Interaction.hcp_id == hcp_id)
    return q.order_by(models.Interaction.interaction_date.desc()).all()


@router.get("/interactions/{interaction_id}", response_model=schemas.InteractionOut)
def get_interaction(interaction_id: str, db: Session = Depends(get_db)):
    interaction = db.query(models.Interaction).filter(models.Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(404, "Interaction not found")
    return interaction


@router.patch("/interactions/{interaction_id}", response_model=schemas.InteractionOut)
def update_interaction(interaction_id: str, payload: schemas.InteractionUpdate, db: Session = Depends(get_db)):
    interaction = db.query(models.Interaction).filter(models.Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(404, "Interaction not found")

    changes = payload.model_dump(exclude_unset=True, exclude={"edited_by", "edit_reason"})
    changed_fields = {}
    for key, value in changes.items():
        if getattr(interaction, key) != value:
            changed_fields[key] = {"from": getattr(interaction, key), "to": value}
            setattr(interaction, key, value)

    history = interaction.edit_history or []
    history.append({
        "edited_by": payload.edited_by or "field_rep",
        "reason": payload.edit_reason,
        "changed_fields": changed_fields,
    })
    interaction.edit_history = history

    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    return interaction


@router.get("/interactions/{interaction_id}/compliance")
def run_compliance_check(interaction_id: str, db: Session = Depends(get_db)):
    interaction = db.query(models.Interaction).filter(models.Interaction.id == interaction_id).first()
    if not interaction:
        raise HTTPException(404, "Interaction not found")
    result = check_compliance.invoke({
        "hcp_id": interaction.hcp_id,
        "samples_provided": interaction.samples_provided or [],
    })
    return {"result": result}
