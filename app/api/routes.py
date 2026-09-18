from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.graph import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
def chat(req: ChatRequest):
    response = run_agent(session_id=req.session_id, user_input=req.message)
    return response
