import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    """RAG knowledge layer: chunked passages from FAO/IPCC reports,
    peer-reviewed studies, or other environmental datasets, with a
    dense vector embedding for similarity search."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[str] = mapped_column(String, index=True)  # e.g. "FAO-2021-SOC", "IPCC-AR6-WG2"
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim))
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)  # page, section, url, year
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StructuredEvidence(Base):
    """Verified metric -> intervention lookup table. This is the
    'structured knowledge layer' distinct from free-text RAG, so
    numeric claims can be traced to a specific row rather than an
    LLM's free-form summary of a document."""

    __tablename__ = "structured_evidence"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric: Mapped[str] = mapped_column(String, index=True)  # e.g. "soil_organic_carbon"
    intervention: Mapped[str] = mapped_column(String)  # e.g. "legume cover cropping"
    expected_improvement_range: Mapped[str] = mapped_column(String)  # e.g. "+15-25% over 2-3 years"
    mechanism: Mapped[str] = mapped_column(Text)
    region_context: Mapped[str] = mapped_column(String, default="general")  # e.g. "semi-arid"
    source_id: Mapped[str] = mapped_column(String, index=True)  # links to a DocumentChunk.source_id or citation
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationTurn(Base):
    """Per-session memory so the agent can track which of the 5 core
    variables (soil, water, land use, climate, human impact) are
    already known and avoid re-asking for them."""

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    known_variables: Mapped[dict] = mapped_column(JSON, default=dict)  # snapshot after this turn
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
