from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Darukaa Biodiversity Intelligence Chatbot",
    description="RAG + structured-evidence reasoning agent for biodiversity recommendations",
    version="0.1.0",
)
app.include_router(router)
