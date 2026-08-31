from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .knowledge import KnowledgeRegistry
from .models import (
    AIObservation,
    DataState,
    Evidence,
    Finding,
    RuleState,
    ScoreCheckResult,
    ScoreResult,
)

READINESS_DIMENSIONS = (
    "Technical Search Readiness",
    "Entity & Machine Understanding",
    "Authority & Trust",
    "Content Citability",
    "Measurement Maturity",
)


class ReadinessCheck(BaseModel):
    name: str
    state: DataState
    score: float | None = Field(default=None, ge=0, le=100)
    weight: float = Field(default=1, gt=0)
    confidence: float = Field(default=1, ge=0, le=1)
    dimension: str = "readiness"
    rule_id: str | None = None
    rule_evidence_level: Literal["A", "B", "C", "D", "E"] | None = None
    rule_confidence: float | None = Field(default=None, ge=0, le=1)
    rule_scoring_weight: float | None = Field(default=None, ge=0)
    rule_state: RuleState | None = None
    rule_expires_at: date | None = None
    explanation_finding_ids: list[str] = Field(default_factory=list)
    explanation_observation_ids: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def explain_value_or_unavailability(self) -> ReadinessCheck:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if numeric and self.score is None:
            raise ValueError(f"{self.state} readiness checks require a numeric score")
        if not numeric and self.score is not None:
            raise ValueError(f"{self.state} readiness checks cannot have a numeric score")
        if (
            self.score is not None
            and self.score < 100
            and not (self.explanation_finding_ids or self.explanation_observation_ids)
        ):
            raise ValueError("readiness checks below 100 require an explanation reference")
        if self.score is None and not self.unavailable_reason:
            raise ValueError("non-numeric readiness checks require unavailable_reason")
        return self

    def result(self) -> ScoreCheckResult:
        return ScoreCheckResult.model_validate(self.model_dump(exclude={"dimension"}))


def build_rule_backed_check(
    *,
    name: str,
    registry: KnowledgeRegistry,
    rule_id: str,
    as_of: date,
    state: DataState,
    score: float | None,
    confidence: float = 1,
    explanation_finding_ids: list[str] | None = None,
    unavailable_reason: str | None = None,
) -> ReadinessCheck:
    rule = registry.resolve(rule_id, as_of=as_of)
    if rule.state is RuleState.REQUIRES_VERIFICATION:
        state = DataState.UNAVAILABLE
        score = None
        confidence = 0
        unavailable_reason = (
            f"Knowledge rule {rule.rule_id} expired on {rule.expires_at.isoformat()} and "
            "requires verification."
        )
    return ReadinessCheck(
        name=name,
        state=state,
        score=score,
        weight=rule.scoring_weight,
        confidence=min(confidence, rule.confidence),
        rule_id=rule.rule_id,
        rule_evidence_level=rule.evidence_level,
        rule_confidence=rule.confidence,
        rule_scoring_weight=rule.scoring_weight,
        rule_state=rule.state,
        rule_expires_at=rule.expires_at,
        explanation_finding_ids=explanation_finding_ids or [],
        unavailable_reason=unavailable_reason,
    )


def calculate_readiness(name: str, checks: list[ReadinessCheck]) -> ScoreResult:
    relevant = [check for check in checks if check.dimension == "readiness"]
    total_weight = sum(check.weight for check in relevant)
    catalog_gap = 0
    if name in READINESS_DIMENSIONS and len(relevant) == 1:
        catalog_gap = 2
        total_weight = relevant[0].weight * 3
    available = [
        check
        for check in relevant
        if check.state in {DataState.AVAILABLE, DataState.PARTIAL} and check.score is not None
    ]
    available_weight = sum(check.weight for check in available)
    unavailable = [
        check.name
        for check in relevant
        if check.state in {DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.UNSUPPORTED}
    ]
    if catalog_gap:
        unavailable.append(f"dimension coverage: {catalog_gap} checks not assessed")
    check_results = [check.result() for check in relevant]
    total_checks = len(relevant) + catalog_gap
    if not available:
        if any(check.state is DataState.FAILED for check in relevant):
            state = DataState.FAILED
        elif any(check.state is DataState.UNKNOWN for check in relevant):
            state = DataState.UNKNOWN
        else:
            state = DataState.UNAVAILABLE
        return ScoreResult(
            name=name,
            state=state,
            unavailable_inputs=unavailable,
            unavailable_reason="; ".join(
                dict.fromkeys(
                    check.unavailable_reason
                    for check in relevant
                    if check.unavailable_reason is not None
                )
            )
            or "No numeric checks were available for this dimension.",
            total_checks=total_checks,
            checks=check_results,
        )
    coverage = available_weight / total_weight if total_weight else 0
    value = (
        sum(check.score * check.weight for check in available if check.score is not None)
        / available_weight
    )
    confidence = (
        sum(check.confidence * check.weight for check in available) / available_weight * coverage
    )
    state = DataState.AVAILABLE if coverage == 1 else DataState.PARTIAL
    explanation_finding_ids = sorted(
        {
            finding_id
            for check in available
            if check.score is not None and check.score < 100
            for finding_id in check.explanation_finding_ids
        }
    )
    explanation_observation_ids = sorted(
        {
            observation_id
            for check in available
            if check.score is not None and check.score < 100
            for observation_id in check.explanation_observation_ids
        }
    )
    return ScoreResult(
        name=name,
        state=state,
        value=round(value, 2),
        coverage=round(coverage, 4),
        confidence=round(confidence, 4),
        unavailable_inputs=unavailable,
        observed_checks=len(available),
        total_checks=total_checks,
        checks=check_results,
        explanation_finding_ids=explanation_finding_ids,
        explanation_observation_ids=explanation_observation_ids,
    )


def observed_ai_visibility(observations: list[AIObservation]) -> ScoreResult:
    if not observations:
        return ScoreResult(
            name="Observed AI Visibility",
            state=DataState.UNAVAILABLE,
            unavailable_inputs=["grounded AI observations"],
            unavailable_reason="Grounded AI observations were not collected.",
        )
    measurable = [item for item in observations if item.brand_mentioned is not None]
    if not measurable:
        return ScoreResult(
            name="Observed AI Visibility",
            state=DataState.UNKNOWN,
            unavailable_reason="Imported observations did not record brand mention state.",
        )
    value = sum(1 for item in measurable if item.brand_mentioned) / len(measurable) * 100
    coverage = len(measurable) / len(observations)
    return ScoreResult(
        name="Observed AI Visibility",
        state=DataState.AVAILABLE if coverage == 1 else DataState.PARTIAL,
        value=round(value, 2),
        coverage=round(coverage, 4),
        confidence=round(coverage, 4),
        explanation_observation_ids=(
            [item.observation_id for item in measurable] if value < 100 else []
        ),
    )


def validate_score_explainability(
    scores: list[ScoreResult],
    findings: list[Finding],
    evidence: list[Evidence],
    observations: list[AIObservation] | None = None,
) -> None:
    finding_by_id = {finding.finding_id: finding for finding in findings}
    evidence_ids = {item.evidence_id for item in evidence}
    observation_ids = {item.observation_id for item in observations or []}

    def validate_references(
        explanation_ids: list[str], *, score_name: str, rule_id: str | None = None
    ) -> None:
        for finding_id in explanation_ids:
            finding = finding_by_id.get(finding_id)
            if finding is None:
                raise ValueError(f"score {score_name} references missing finding {finding_id}")
            if not finding.evidence_ids:
                raise ValueError(
                    f"score {score_name} references finding {finding_id} without evidence"
                )
            missing_evidence = set(finding.evidence_ids) - evidence_ids
            if missing_evidence:
                raise ValueError(
                    f"score {score_name} finding {finding_id} references missing evidence "
                    f"{sorted(missing_evidence)}"
                )
            if rule_id and finding.rule_id != rule_id:
                raise ValueError(
                    f"score {score_name} finding {finding_id} is not relevant to rule {rule_id}"
                )

    for score in scores:
        validate_references(score.explanation_finding_ids, score_name=score.name)
        missing_observations = set(score.explanation_observation_ids) - observation_ids
        if missing_observations:
            raise ValueError(
                f"score {score.name} references missing observations {sorted(missing_observations)}"
            )
        for check in score.checks:
            validate_references(
                check.explanation_finding_ids,
                score_name=f"{score.name}/{check.name}",
                rule_id=check.rule_id,
            )
            missing_check_observations = set(check.explanation_observation_ids) - observation_ids
            if missing_check_observations:
                raise ValueError(
                    f"score {score.name}/{check.name} references missing observations "
                    f"{sorted(missing_check_observations)}"
                )
