# Darukaa Hackathon — Biodiversity Intelligence Chatbot

An AI environmental-scientist agent that reasons over a structured
metric-intervention database **and** retrieved scientific literature
(FAO / IPCC / peer-reviewed studies) to produce evidence-backed,
multi-metric biodiversity recommendations — built for the
Darukaa.Earth AI Biodiversity Intelligence Chatbot Challenge.

**Live demo:** https://darukaa-app-201562091189.us-central1.run.app

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
 (hosted on Supabase)         (per-session, in Postgres)
 - document_chunks (RAG)
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
- **Scientific Grounding (25%)** — every response is required to trace
  its numbers to a `source_id` in `structured_evidence` or a cited
  chunk in `document_chunks`. Sources are attached **deterministically**
  in code after the LLM call (`app/agent/graph.py::_reason`) rather than
  left to the model to remember to cite — this closed a gap where the
  reasoning LLM would sometimes omit retrieved-chunk citations even
  when relevant chunks were found.
- **Conversational Intelligence (15%)** — LangGraph's checkpointer
  persists state per `session_id`, so the agent asks clarifying
  questions when fewer than `min_known_variables` (default 3) of
  {soil, water, land use, climate, human impact} are known, and
  remembers earlier answers across turns.
- **Output Clarity (10%)** — every response is validated against a
  Pydantic schema (`app/agent/schemas.py`) before being returned.

## Database / Schema

Two Postgres tables (see `app/db/models.py`), hosted on Supabase:

| Table | Purpose | Key columns |
|---|---|---|
| `document_chunks` | RAG corpus (ingested FAO/IPCC reports) | `id, source_id, content, embedding (vector(384)), metadata (jsonb)` |
| `structured_evidence` | Verified metric→intervention lookup | `id, metric, intervention, expected_improvement_range, source_id, region_context` |

Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
CPU-only — the Docker image pre-downloads this model at build time so
no network call to Hugging Face happens at runtime). `pgvector`'s
`<->` cosine-distance operator powers similarity search in
`app/agent/retrieval.py::retrieve_chunks`.

## Local setup

```bash
# 1. Create and activate a virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Configure environment (.env)
cat > .env <<EOF
GROQ_API_KEY=your_groq_key
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres
EOF

# 4. Create tables (if not already provisioned on Supabase)
python -m app.db.init_db

# 5. Ingest knowledge (structured evidence + FAO/IPCC PDFs in data/reports/)
python -m app.ingestion.ingest --structured data/sample_structured_evidence.csv
python -m app.ingestion.ingest --pdf-dir data/reports/

# 6. Run the API
uvicorn app.main:app --reload

# 7. Chat
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Soil organic carbon is 0.3%, rainfall is low, and I grow monoculture wheat in a semi-arid region"}'
```

## Deployment (GCP Cloud Run)

Live at **https://darukaa-app-201562091189.us-central1.run.app**

```bash
# Build (Cloud Build — no local Docker required)
gcloud builds submit --tag us-central1-docker.pkg.dev/darukaa-biodiv-2026/darukaa-repo/darukaa-app:latest .

# Deploy
gcloud run deploy darukaa-app \
  --image us-central1-docker.pkg.dev/darukaa-biodiv-2026/darukaa-repo/darukaa-app:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --set-env-vars GROQ_API_KEY=your_groq_key,DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@db.YOUR_PROJECT_REF.supabase.co:5432/postgres"
```

`--memory 2Gi --cpu 2` is required — the default 512Mi is insufficient
for the `torch` + `sentence-transformers` + LangChain stack and causes
503s under load. The Dockerfile installs a **CPU-only** torch build
(`--index-url https://download.pytorch.org/whl/cpu`) to avoid pulling
multi-GB CUDA packages that Cloud Run can't use anyway.

### Try the live demo

```bash
curl -X POST https://darukaa-app-201562091189.us-central1.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Soil organic carbon is 0.3%, rainfall is low, and I grow monoculture wheat in a semi-arid region"}'
```

Returns a JSON object with `recommendation`, `mechanism`,
`impacted_metrics`, `expected_improvement`, `time_horizon`,
`confidence`, `sources` (mix of `structured` and `retrieved` evidence),
`evidence_conflict`, and `clarifying_question`.

## CI/CD

`.github/workflows/ci.yml` runs lint (`ruff`) + `pytest` against a
`postgres+pgvector` service container on every push. Container build
and Cloud Run deploy are currently done manually via `gcloud`; a
`deploy.yml` job using `google-github-actions/deploy-cloudrun` would
automate the steps in **Deployment** above.

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
  reports/               ingested FAO/IPCC source PDFs
tests/
  test_agent.py
```