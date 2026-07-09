"""Run `python seed_data.py` once after setting up the database to add demo HCPs."""
from app.database import SessionLocal, Base, engine
from app.models import HCP

Base.metadata.create_all(bind=engine)

demo_hcps = [
    {"name": "Dr. Anjali Sharma", "specialty": "Cardiology", "institution": "Fortis Hospital", "email": "a.sharma@example.com"},
    {"name": "Dr. Rohan Mehta", "specialty": "Endocrinology", "institution": "Apollo Clinic", "email": "r.mehta@example.com"},
    {"name": "Dr. Priya Nair", "specialty": "Oncology", "institution": "Tata Memorial", "email": "p.nair@example.com"},
]

db = SessionLocal()
try:
    for h in demo_hcps:
        if not db.query(HCP).filter(HCP.name == h["name"]).first():
            db.add(HCP(**h))
    db.commit()
    print(f"Seeded {len(demo_hcps)} demo HCPs.")
finally:
    db.close()
