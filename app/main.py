from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Darukaa Biodiversity Intelligence Chatbot",
    description="RAG + structured-evidence reasoning agent for biodiversity recommendations",
    version="0.1.0",
)
app.include_router(router)

# Serve the static frontend at "/" — added after the API router so /chat
# and /health still take priority; html=True makes "/" serve index.html.
app.mount("/", StaticFiles(directory="static", html=True), name="static")