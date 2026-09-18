"""Load knowledge into the two knowledge layers.

Usage:
    python -m app.ingestion.ingest --structured data/sample_structured_evidence.csv
    python -m app.ingestion.ingest --pdf-dir data/reports/
"""
import argparse
import csv
import uuid
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db.models import DocumentChunk, StructuredEvidence
from app.db.session import get_session

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def load_structured_csv(path: str) -> int:
    """CSV columns: metric,intervention,expected_improvement_range,mechanism,region_context,source_id"""
    session = get_session()
    count = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session.add(
                StructuredEvidence(
                    id=str(uuid.uuid4()),
                    metric=row["metric"].strip(),
                    intervention=row["intervention"].strip(),
                    expected_improvement_range=row["expected_improvement_range"].strip(),
                    mechanism=row["mechanism"].strip(),
                    region_context=row.get("region_context", "general").strip() or "general",
                    source_id=row["source_id"].strip(),
                )
            )
            count += 1
    session.commit()
    session.close()
    return count


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def load_pdf_dir(pdf_dir: str) -> int:
    embedder = get_embedder()
    session = get_session()
    count = 0
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        source_id = pdf_path.stem
        for i, chunk in enumerate(_chunk_text(full_text)):
            if not chunk.strip():
                continue
            embedding = embedder.encode(chunk).tolist()
            session.add(
                DocumentChunk(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    title=f"{pdf_path.name} [chunk {i}]",
                    content=chunk,
                    embedding=embedding,
                    doc_metadata={"file": pdf_path.name, "chunk_index": i},
                )
            )
            count += 1
    session.commit()
    session.close()
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", help="Path to structured evidence CSV")
    parser.add_argument("--pdf-dir", help="Directory of PDFs to chunk + embed")
    args = parser.parse_args()

    if args.structured:
        n = load_structured_csv(args.structured)
        print(f"Loaded {n} structured evidence rows")
    if args.pdf_dir:
        n = load_pdf_dir(args.pdf_dir)
        print(f"Loaded {n} document chunks")
    if not args.structured and not args.pdf_dir:
        parser.print_help()
