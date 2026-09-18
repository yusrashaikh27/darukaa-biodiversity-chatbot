"""Run with: python -m app.db.init_db"""
from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    print("Tables created: document_chunks, structured_evidence, conversation_turns")


if __name__ == "__main__":
    init_db()
