# Darukaa Biodiversity Intelligence Chatbot

An AI environmental-scientist agent that reasons over a structured
metric-intervention database **and** retrieved scientific literature
(FAO / IPCC / peer-reviewed studies) to produce evidence-backed,
multi-metric biodiversity recommendations — built for the
Darukaa.Earth AI Biodiversity Intelligence Chatbot Challenge.

## Architecture

```
User (text / JSON)
      │
      ▼
FastAPI  (app/main.py, app/api/routes.py)
      │
      ▼
LangGraph state machine (app/agent/graph.py)
   ┌─────────────┬───────────────┬─────────────┐
   │  retrieve    │   reason      │   respond    │
   │ (vector +    │ (LLM call,    │ (validate +  │
   │  SQL join)   │  strict JSON  │  persist     │
   │              │  schema)      │  turn state) │
   └─────────────┴───────────────┴─────────────┘
      │                              │
      ▼                              ▼
 PostgreSQL + pgvector        Conversation memory
 - document_chunks (RAG)      (per-session, in Postgres)
 - structured_evidence
   (metric, intervention,
    expected_improvement_range,
    source_id)
```

**Why this design maps to the rubric:**
- **Knowledge System (20%)** — `document_chunks` (pgvector, RAG over
  FAO/IPCC PDFs) + `structured_evidence` (a verified, queryable
  metric→intervention table) are two distinct, retrievable knowledge
  layers, not just prompt text. See `app/agent/retrieval.py`.
- **Depth of Reasoning / Multi-Metric (30%)** — the reasoning prompt
  (`app/agent/prompts.py`) requires chaining ≥3 environmental
  variables with explicit causal links, per the challenge brief's
  constraint.
- **Scientific Grounding (25%)** — every number in a response must
  trace to a `source_id` in `structured_evidence` or a cited chunk in
  `document_chunks`; the agent is instructed to lower confidence
  rather than fabricate when evidence is thin.
- **Conversational Intelligence (15%)** — LangGraph's checkpointer
  persists state per `session_id`, so the agent asks clarifying
  questions when fewer than 3 of {soil, water, land use, climate,
  human impact} are known, and remembers earlier answers across turns.
- **Output Clarity (10%)** — every response is validated against a
  Pydantic schema (`app/agent/schemas.py`) before being returned.

## Database / Schema

Two Postgres tables (see `app/db/models.py`):

| Table | Purpose | Key columns |
|---|---|---|
| `document_chunks` | RAG corpus (FAO/IPCC reports, papers) | `id, source_id, content, embedding (vector(1536)), metadata (jsonb)` |
| `structured_evidence` | Verified metric→intervention lookup | `id, metric, intervention, expected_improvement_range, source_id, region_context` |

`pgvector`'s `<->` cosine-distance operator powers similarity search
in `app/agent/retrieval.py::retrieve_chunks`.

## Local setup

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# fill in OPENAI/ANTHROPIC key + DATABASE_URL

# 4. Create tables
python -m app.db.init_db

# 5. Ingest sample knowledge (seed structured evidence + a few PDFs)
python -m app.ingestion.ingest --structured data/sample_structured_evidence.csv
python -m app.ingestion.ingest --pdf-dir data/reports/

# 6. Run the API
uvicorn app.main:app --reload

# 7. Chat
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Biodiversity is declining on my land"}'
```

## CI/CD

`.github/workflows/ci.yml` runs lint (`ruff`) + `pytest` against a
`postgres+pgvector` service container on every push. Deployment target
is GCP Cloud Run (container built from the root `Dockerfile`; add a
`deploy.yml` job with `google-github-actions/deploy-cloudrun` once a
GCP project/service account is wired up).

## Repo layout

```
app/
  main.py              FastAPI app + startup
  config.py            Settings (env vars)
  db/
    models.py          SQLAlchemy + pgvector models
    session.py         DB session/engine
    init_db.py          create_all()
  ingestion/
    ingest.py           chunk + embed PDFs, load structured CSV
  agent/
    schemas.py           strict output schema (Pydantic)
    prompts.py            the reasoning system prompt
    retrieval.py           vector search + structured SQL join
    graph.py                LangGraph: retrieve -> reason -> respond
  api/
    routes.py           /chat, /health endpoints
data/
  sample_structured_evidence.csv
tests/
  test_agent.py
```
