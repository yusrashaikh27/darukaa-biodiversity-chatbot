from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.config import settings
from app.db.models import DocumentChunk, StructuredEvidence
from app.db.session import get_session

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def retrieve_chunks(query: str, k: int = 5) -> list[dict]:
    """Cosine-similarity search over document_chunks via pgvector's <-> operator."""
    session = get_session()
    try:
        query_embedding = get_embedder().encode(query).tolist()
        rows = session.scalars(
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        ).all()
        return [
            {
                "id": r.id,
                "source_id": r.source_id,
                "title": r.title,
                "content": r.content,
                "metadata": r.doc_metadata,
            }
            for r in rows
        ]
    finally:
        session.close()


def retrieve_structured_evidence(metrics_of_interest: list[str], region_context: str | None = None) -> list[dict]:
    """SQL join / filter over the verified metric-intervention table.

    metrics_of_interest is free text (known-variable values, or the raw
    user message as a fallback) — not clean metric keys — so matching
    works by checking whether any significant word from that free text
    appears in a row's metric/intervention/mechanism text, not the other
    way around (a full sentence is never a substring of a short metric
    name like "soil_organic_carbon").
    """
    session = get_session()
    try:
        stmt = select(StructuredEvidence)
        rows = session.scalars(stmt).all()

        # Build a set of significant words (>=4 chars) from all the free-text hints.
        words = set()
        for phrase in metrics_of_interest:
            for w in phrase.lower().replace("-", " ").replace("_", " ").split():
                w = "".join(c for c in w if c.isalnum())
                if len(w) >= 4:
                    words.add(w)

        results = []
        for r in rows:
            haystack = f"{r.metric} {r.intervention} {r.mechanism}".lower().replace("_", " ")
            metric_match = not words or any(w in haystack for w in words)
            region_match = (
                region_context is None
                or r.region_context == "general"
                or region_context.lower() in r.region_context.lower()
                or r.region_context.lower() in region_context.lower()
            )
            if metric_match and region_match:
                results.append(
                    {
                        "id": r.id,
                        "metric": r.metric,
                        "intervention": r.intervention,
                        "expected_improvement_range": r.expected_improvement_range,
                        "mechanism": r.mechanism,
                        "region_context": r.region_context,
                        "source_id": r.source_id,
                    }
                )
        return results
    finally:
        session.close()