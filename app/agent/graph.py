import json
import re
import uuid
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.prompts import CLARIFYING_EXTRACTION_PROMPT, SYSTEM_PROMPT
from app.agent.retrieval import retrieve_chunks, retrieve_structured_evidence
from app.agent.schemas import AgentResponse, KnownVariables
from app.config import settings
from app.db.models import ConversationTurn
from app.db.session import get_session

_groq_client = None
_anthropic_client = None

# The reasoning node needs a stronger model than the tiny extraction model
# (allam-2-7b, 4096 ctx) can support. gpt-oss-20b is a reasoning model, so
# reasoning_effort caps how many hidden tokens it burns before answering —
# without this it can silently consume the whole max_tokens budget and
# return empty content.
REASON_MODEL = "openai/gpt-oss-20b"
REASON_MAX_TOKENS = 1200
REASON_EFFORT = "low"


def get_client():
    """Groq client — used by extraction (allam-2-7b) and reasoning
    (gpt-oss-20b)."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def get_anthropic_client():
    """Anthropic client — not currently used (no billing set up), kept so
    switching providers later is a one-line change instead of a rewrite."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def call_llm(
    prompt: str,
    max_tokens: int,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """..."""
    provider = provider_override or settings.llm_provider

    if provider == "groq":                                    # <-- replace from here
        client = get_client()
        kwargs = {
            "model": model_override or settings.groq_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content              # <-- to here
    else:
        client = get_anthropic_client()
        resp = client.messages.create(
            model=model_override or settings.claude_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class GraphState(TypedDict):
    session_id: str
    user_input: str
    known_variables: dict
    retrieved_chunks: list
    structured_evidence: list
    response: Optional[dict]


def _load_prior_known_variables(session_id: str) -> dict:
    session = get_session()
    try:
        last_turn = (
            session.query(ConversationTurn)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.created_at.desc())
            .first()
        )
        return last_turn.known_variables if last_turn else {}
    finally:
        session.close()


def _parse_json_loosely(raw: str) -> dict:
    """Parse a JSON object out of an LLM response even if it's wrapped in
    markdown fences or has stray preamble/explanation text around it."""
    if not raw:
        return {}
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


_KEYWORD_RULES = {
    "soil_health": [
        "soil organic carbon", "soil carbon", " soc", "organic matter",
        "soil health", "soil quality", "erosion", "compaction", "salinity",
    ],
    "water_availability": [
        "rainfall", "rain", "drought", "irrigation", "water table",
        "groundwater", "arid", "monsoon", "moisture", "water stress",
    ],
    "land_use": [
        "monoculture", "crop", "wheat", "rice", "maize", "grazing",
        "pasture", "plantation", "orchard", "fallow", "tillage",
        "farm", "field", "hectare", "acre", "forest", "agroforestry",
    ],
    "climate": [
        "semi-arid", "semiarid", "arid", "tropical", "temperate",
        "humid", "climate", "temperature", "heat", "frost", "season",
    ],
    "human_impact": [
        "fertili", "pesticide", "herbicide", "overgraz", "deforest",
        "burning", "encroach", "settlement", "logging", "mining",
    ],
}


def _keyword_extract(text: str) -> dict:
    """Deterministic floor under the LLM extraction. Never returns a value
    the LLM would overwrite — it only fills keys the model missed, so a
    truncated/empty LLM response never leaves known_variables empty when
    the text plainly states them."""
    lowered = f" {text.lower()} "
    found = {}
    for variable, terms in _KEYWORD_RULES.items():
        hits = [t.strip() for t in terms if t in lowered]
        if hits:
            found[variable] = f"mentioned: {', '.join(hits[:3])}"
    return found


def _extract_known_variables(state: GraphState) -> GraphState:
    """LLM call (Groq, allam-2-7b): merge newly-mentioned variables from
    user_input into prior known_variables. A deterministic keyword pass
    sits underneath both prior state and the LLM result as a floor, so a
    truncated/empty LLM response never leaves known_variables empty when
    the text plainly states them."""
    prior = state.get("known_variables") or _load_prior_known_variables(state["session_id"])
    prompt = CLARIFYING_EXTRACTION_PROMPT.format(prior_state=json.dumps(prior), user_message=state["user_input"])
    raw = call_llm(prompt, max_tokens=500, provider_override="groq")
    extracted = _parse_json_loosely(raw)
    keyword_floor = _keyword_extract(state["user_input"])
    merged = {**keyword_floor, **prior, **extracted}
    state["known_variables"] = merged
    return state


def _retrieve(state: GraphState) -> GraphState:
    known = KnownVariables(**state["known_variables"])
    metrics_hint = [v for v in known.model_dump().values() if v] + [state["user_input"]]

    chunks = retrieve_chunks(state["user_input"], k=5)
    # Cap chunk length so a single call can't blow the reasoning node's
    # token budget, regardless of which model is behind it.
    # NOTE: retrieve_chunks() returns dicts keyed "content", not "text" —
    # fixed below (previously this check silently never fired).
    for chunk in chunks:
        if isinstance(chunk, dict) and "content" in chunk and len(chunk["content"]) > 500:
            chunk["content"] = chunk["content"][:500] + "..."

    evidence = retrieve_structured_evidence(
        metrics_of_interest=metrics_hint,
        region_context=known.land_use or known.climate,
    )

    state["retrieved_chunks"] = chunks[:3]
    state["structured_evidence"] = evidence[:5]
    return state


_FALLBACK_RESPONSE = {
    "recommendation": "Unable to generate a recommendation for this request.",
    "mechanism": "The reasoning model did not return a usable response. Please retry.",
    "impacted_metrics": [],
    "expected_improvement": "unknown",
    "time_horizon": "short",
    "confidence": "low",
    "sources": [],
    "evidence_conflict": None,
    "clarifying_question": None,
}


def _reason(state: GraphState) -> GraphState:
    """LLM call (Groq, gpt-oss-20b): the scored reasoning step. Uses
    reasoning_effort="low" to cap hidden reasoning-token spend — without
    this, gpt-oss models can unpredictably consume the entire max_tokens
    budget on invisible reasoning and return empty content, which breaks
    strict-schema validation downstream. If the model still returns
    nothing usable, fall back to an explicit low-confidence response
    instead of crashing the request."""
    known = KnownVariables(**state["known_variables"])
    prompt = SYSTEM_PROMPT.format(
        min_known_variables=settings.min_known_variables,
        retrieved_chunks=json.dumps(state["retrieved_chunks"]),
        sql_join_results=json.dumps(state["structured_evidence"]),
        langgraph_state=json.dumps(state["known_variables"]),
        user_input=state["user_input"],
    )
    raw = call_llm(
        prompt,
        max_tokens=REASON_MAX_TOKENS,
        provider_override="groq",
        model_override=REASON_MODEL,
        reasoning_effort=REASON_EFFORT,
    )
    parsed_dict = _parse_json_loosely(raw)

    if not parsed_dict:
        parsed_dict = dict(_FALLBACK_RESPONSE)

    parsed = AgentResponse.model_validate(parsed_dict)

    # Deterministically attach evidence sources rather than trusting a
    # small model to reliably cite them in its JSON output — mirrors the
    # extraction-node fallback pattern used elsewhere in this graph.
    # Only fills in sources the model didn't already provide, so a model
    # that *does* cite correctly isn't overridden.
    if not parsed.sources:
        parsed.sources = [
            {"type": "structured", "id": e["id"], "ref": e["source_id"]}
            for e in state["structured_evidence"][:2]
        ] + [
            {"type": "retrieved", "id": c["id"], "ref": c["source_id"]}
            for c in state["retrieved_chunks"][:2]
        ]

    if known.count_known() < settings.min_known_variables and not parsed.clarifying_question:
        missing = [k for k, v in known.model_dump().items() if not v]
        parsed.clarifying_question = f"Could you share more about: {','.join(missing)}?"
        parsed.confidence = "low"

    state["response"] = parsed.model_dump()
    return state


def _persist(state: GraphState) -> GraphState:
    session = get_session()
    try:
        session.add(
            ConversationTurn(
                id=str(uuid.uuid4()),
                session_id=state["session_id"],
                role="user",
                content=state["user_input"],
                known_variables=state["known_variables"],
            )
        )
        session.add(
            ConversationTurn(
                id=str(uuid.uuid4()),
                session_id=state["session_id"],
                role="assistant",
                content=json.dumps(state["response"]),
                known_variables=state["known_variables"],
            )
        )
        session.commit()
    finally:
        session.close()
    return state


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extract_variables", _extract_known_variables)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("reason", _reason)
    graph.add_node("persist", _persist)

    graph.set_entry_point("extract_variables")
    graph.add_edge("extract_variables", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_edge("reason", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


_compiled_graph = None


def run_agent(session_id: str, user_input: str) -> dict:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    result = _compiled_graph.invoke({"session_id": session_id, "user_input": user_input})
    return result["response"]