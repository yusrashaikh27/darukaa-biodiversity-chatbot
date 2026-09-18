from typing import Literal, Optional

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    type: Literal["structured", "retrieved"]
    id: str
    ref: str  # human-readable citation, e.g. "FAO-2021-SOC" or "IPCC AR6 WG2, p.14"


class AgentResponse(BaseModel):
    """Strict output schema enforced on every reasoning-node response.
    Matches the challenge's 'Output Quality' requirement: recommendation,
    impacted metrics, time horizon, confidence."""

    recommendation: str
    mechanism: str = Field(..., description="Causal chain across >=3 environmental variables")
    impacted_metrics: list[str]
    expected_improvement: str
    time_horizon: Literal["short", "medium", "long"]
    confidence: Literal["low", "medium", "high"]
    sources: list[SourceRef]
    evidence_conflict: Optional[str] = Field(
        default=None,
        description="Set when STRUCTURED_EVIDENCE and RETRIEVED_TEXT disagree; describes the discrepancy",
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Non-null when fewer than settings.min_known_variables are known",
    )


class KnownVariables(BaseModel):
    """Tracks which of the 5 core variables have been supplied so far in the conversation."""

    soil_health: Optional[str] = None
    water_availability: Optional[str] = None
    land_use: Optional[str] = None
    climate: Optional[str] = None
    human_impact: Optional[str] = None

    def count_known(self) -> int:
        return sum(1 for v in self.model_dump().values() if v)
