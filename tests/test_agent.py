from app.agent.schemas import AgentResponse, KnownVariables


def test_known_variables_count():
    kv = KnownVariables(soil_health="low SOC", water_availability="low rainfall")
    assert kv.count_known() == 2
    kv2 = KnownVariables()
    assert kv2.count_known() == 0


def test_agent_response_requires_clarifying_question_field_optional():
    resp = AgentResponse(
        recommendation="Introduce legume-based cover crops",
        mechanism="Low SOC -> poor water retention -> drought stress -> reduced pollinator visitation",
        impacted_metrics=["soil_organic_carbon", "species_richness"],
        expected_improvement="+15-25% SOC over 2-3 years",
        time_horizon="medium",
        confidence="medium",
        sources=[{"type": "structured", "id": "abc123", "ref": "FAO-2021-SOC"}],
    )
    assert resp.clarifying_question is None
    assert resp.evidence_conflict is None


def test_agent_response_rejects_bad_time_horizon():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentResponse(
            recommendation="x",
            mechanism="x",
            impacted_metrics=["x"],
            expected_improvement="x",
            time_horizon="eventually",  # invalid
            confidence="low",
            sources=[],
        )
