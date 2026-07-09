from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.config import settings
from app.routers import interactions, chat

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI-First HCP CRM API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interactions.router, prefix="/api", tags=["interactions"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
