from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ai_search_audit import scoring
from ai_search_audit.knowledge import load_registry
from ai_search_audit.models import AIObservation, DataState, RuleState, ScoreResult
from ai_search_audit.scoring import (
    ReadinessCheck,
    build_rule_backed_check,
    calculate_readiness,
    observed_ai_visibility,
)


def test_unavailable_checks_are_excluded_from_denominator() -> None:
    score = calculate_readiness(
        "Measurement Maturity",
        [
            ReadinessCheck(
                name="public analytics tag",
                state=DataState.AVAILABLE,
                score=100,
                weight=1,
                confidence=0.8,
            ),
            ReadinessCheck(
                name="GSC",
                state=DataState.UNAVAILABLE,
                weight=4,
                confidence=0,
                unavailable_reason="GSC access was not supplied.",
            ),
        ],
    )
    assert score.value == 100
    assert score.coverage == 0.2
    assert score.state is DataState.PARTIAL
    assert score.unavailable_inputs == ["GSC"]


def test_unknown_and_failed_remain_distinct() -> None:
    unknown = calculate_readiness(
        "Authority & Trust",
        [
            ReadinessCheck(
                name="backlinks",
                state=DataState.UNKNOWN,
                unavailable_reason="Backlink state could not be determined.",
            )
        ],
    )
    failed = calculate_readiness(
        "Authority & Trust",
        [
            ReadinessCheck(
                name="backlinks",
                state=DataState.FAILED,
                unavailable_reason="Backlink collection failed.",
            )
        ],
    )
    assert unknown.state is DataState.UNKNOWN
    assert failed.state is DataState.FAILED
    assert unknown.value is None and failed.value is None


def test_observed_visibility_without_observations_is_unavailable() -> None:
    result = observed_ai_visibility([])
    assert result.state is DataState.UNAVAILABLE
    assert result.value is None


def test_observed_visibility_uses_resolvable_observation_explanations() -> None:
    observation = AIObservation(
        observation_id="observation-not-visible",
        prompt_id="prompt-1",
        provider="grounded-test",
        observed_at=datetime(2026, 8, 11, tzinfo=UTC),
        brand_mentioned=False,
        grounded=True,
    )

    result = observed_ai_visibility([observation])

    assert result.value == 0
    assert result.explanation_finding_ids == []
    assert result.explanation_observation_ids == [observation.observation_id]
    scoring.validate_score_explainability([result], [], [], [observation])


def test_sem_checks_cannot_modify_technical_readiness() -> None:
    checks = [
        ReadinessCheck(
            name="crawlability",
            state=DataState.AVAILABLE,
            score=80,
            explanation_finding_ids=["finding-crawlability"],
        )
    ]
    baseline = calculate_readiness("Technical Search Readiness", checks)
    with_sem = calculate_readiness(
        "Technical Search Readiness",
        checks
        + [
            ReadinessCheck(
                name="paid demand",
                state=DataState.AVAILABLE,
                score=0,
                dimension="commercial",
                explanation_finding_ids=["finding-paid-demand"],
            )
        ],
    )
    assert baseline.value == with_sem.value == 80


def test_single_check_readiness_cannot_claim_full_dimension_coverage() -> None:
    score = calculate_readiness(
        "Content Citability",
        [ReadinessCheck(name="has a title", state=DataState.AVAILABLE, score=100)],
    )
    assert score.value == 100
    assert score.coverage < 1
    assert score.state is DataState.PARTIAL


def test_rule_backed_scoring_exposes_metadata_and_stale_rules_become_unavailable() -> None:
    registry = load_registry(Path(__file__).parents[1] / "knowledge")
    current = build_rule_backed_check(
        name="indexability",
        registry=registry,
        rule_id="google-noindex-001",
        as_of=date(2026, 8, 11),
        state=DataState.AVAILABLE,
        score=100,
        confidence=0.9,
    )
    current_score = calculate_readiness("Technical Search Readiness", [current])
    assert current_score.checks[0].rule_evidence_level == "A"
    assert current_score.checks[0].rule_scoring_weight == 1
    assert current_score.checks[0].rule_state is RuleState.CURRENT

    stale = build_rule_backed_check(
        name="OpenAI search crawler access",
        registry=registry,
        rule_id="openai-oai-searchbot-001",
        as_of=date(2027, 8, 11),
        state=DataState.AVAILABLE,
        score=100,
    )
    stale_score = calculate_readiness("Technical Search Readiness", [stale])
    assert stale_score.state is DataState.UNAVAILABLE
    assert stale_score.value is None
    assert stale_score.unavailable_reason
    assert stale_score.checks[0].rule_state is RuleState.REQUIRES_VERIFICATION


def test_score_explainability_rejects_unknown_finding_reference() -> None:
    score = ScoreResult(
        name="Technical Search Readiness",
        state=DataState.AVAILABLE,
        value=80,
        explanation_finding_ids=["missing-finding"],
    )
    with pytest.raises(ValueError, match="missing-finding"):
        scoring.validate_score_explainability([score], [], [])
