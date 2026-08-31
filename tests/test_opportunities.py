from ai_search_audit.opportunities import OpportunityWeights, rank_opportunity


def test_opportunity_weights_are_configurable() -> None:
    result = rank_opportunity(
        "op-1",
        "Improve service page",
        commercial_value=0.8,
        organic_gap=0.7,
        ai_gap=0.5,
        strategic_relevance=1,
        confidence=0.8,
        effort=0.5,
        weights=OpportunityWeights(ai_gap=0),
    )
    assert result.rank_value is not None
    assert result.ai_visibility_gap == 0.5
