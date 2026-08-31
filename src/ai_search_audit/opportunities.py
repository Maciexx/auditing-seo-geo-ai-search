from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Opportunity


class OpportunityWeights(BaseModel):
    commercial_value: float = Field(default=1, ge=0)
    organic_gap: float = Field(default=1, ge=0)
    ai_gap: float = Field(default=1, ge=0)
    strategic_relevance: float = Field(default=1, ge=0)


def rank_opportunity(
    opportunity_id: str,
    title: str,
    *,
    commercial_value: float,
    organic_gap: float,
    ai_gap: float,
    strategic_relevance: float,
    confidence: float,
    effort: float,
    weights: OpportunityWeights | None = None,
) -> Opportunity:
    weights = weights or OpportunityWeights()
    components = (
        (commercial_value, weights.commercial_value),
        (organic_gap, weights.organic_gap),
        (ai_gap, weights.ai_gap),
        (strategic_relevance, weights.strategic_relevance),
    )
    total_weight = sum(weight for _, weight in components)
    weighted_value = (
        sum(value * weight for value, weight in components) / total_weight if total_weight else 0
    )
    rank_value = weighted_value * confidence / max(effort, 0.01)
    return Opportunity(
        opportunity_id=opportunity_id,
        title=title,
        commercial_value=commercial_value,
        organic_visibility_gap=organic_gap,
        ai_visibility_gap=ai_gap,
        strategic_relevance=strategic_relevance,
        confidence=confidence,
        implementation_effort=effort,
        rank_value=round(rank_value, 4),
    )
