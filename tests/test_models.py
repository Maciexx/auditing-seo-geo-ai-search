from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_search_audit.models import (
    AIObservation,
    AIPrompt,
    AuditRun,
    Backlink,
    Competitor,
    DataState,
    Entity,
    Evidence,
    ExternalMention,
    Finding,
    FindingStatus,
    Opportunity,
    Page,
    Priority,
    Recommendation,
    ScoreResult,
    Severity,
    Site,
)


def test_required_model_types_are_available() -> None:
    assert all(
        model is not None
        for model in (
            Site,
            Page,
            Entity,
            Finding,
            Evidence,
            Recommendation,
            AIPrompt,
            AIObservation,
            Competitor,
            ExternalMention,
            Backlink,
            Opportunity,
            AuditRun,
        )
    )


def test_finding_supports_required_fields_and_evidence() -> None:
    finding = Finding(
        finding_id="finding-1",
        category="indexability",
        severity=Severity.HIGH,
        status=FindingStatus.CONFIRMED,
        rule_id="robots-001",
        technical_title="Page is noindexed",
        technical_description="A robots directive contains noindex.",
        client_title="A page is hidden from search",
        client_explanation="Search systems are instructed not to include this page.",
        business_impact="The page cannot be discovered through search while the directive remains.",
        implementation="Remove noindex if the page is intended for search.",
        priority=Priority.P1,
        affected_urls=["https://example.com/page"],
        evidence_ids=["evidence-1"],
        confidence=0.98,
    )
    assert finding.status is FindingStatus.CONFIRMED
    assert finding.evidence_ids == ["evidence-1"]


def test_confirmed_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Finding(
            finding_id="finding-1",
            category="indexability",
            severity=Severity.HIGH,
            status=FindingStatus.CONFIRMED,
            rule_id="robots-001",
            technical_title="No evidence",
            technical_description="Missing evidence.",
            client_title="Missing evidence",
            client_explanation="Missing evidence.",
            business_impact="Unknown.",
            implementation="Collect evidence.",
            priority=Priority.P2,
            confidence=0.5,
        )


def test_unavailable_score_has_no_numeric_value() -> None:
    score = ScoreResult(
        name="Measurement Maturity",
        state=DataState.UNAVAILABLE,
        unavailable_reason="Client measurement access was not supplied.",
    )
    assert score.value is None
    with pytest.raises(ValidationError):
        ScoreResult(
            name="Measurement Maturity",
            state=DataState.UNAVAILABLE,
            value=0,
            explanation_finding_ids=["finding-measurement"],
        )


def test_available_score_may_be_zero() -> None:
    score = ScoreResult(
        name="Observed AI Visibility",
        state=DataState.AVAILABLE,
        value=0,
        coverage=1,
        confidence=0.9,
        explanation_observation_ids=["observation-not-visible"],
    )
    assert score.value == 0
    assert score.explanation_observation_ids == ["observation-not-visible"]


def test_non_numeric_score_requires_unavailable_reason() -> None:
    with pytest.raises(ValidationError, match="unavailable_reason"):
        ScoreResult(name="Measurement Maturity", state=DataState.UNAVAILABLE)


def test_numeric_score_below_100_requires_typed_explanation_reference() -> None:
    with pytest.raises(ValidationError, match="explanation"):
        ScoreResult(
            name="Technical Search Readiness",
            state=DataState.AVAILABLE,
            value=80,
        )


def test_page_records_structured_json_ld_parse_errors() -> None:
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        json_ld_errors=[
            {
                "message": "Extra data",
                "line": 1,
                "column": 8,
                "excerpt": '{"a":1}{"b":2}',
            }
        ],
    )
    assert page.json_ld_errors[0].message == "Extra data"


def test_audit_run_records_reproducible_metadata() -> None:
    run = AuditRun(
        audit_id="run-1",
        site=Site(domain="example.com", base_url="https://example.com"),
        audit_engine_version="0.1.0",
        ruleset_version="2026.08.11",
        ruleset_verified_date="2026-08-11",
        timestamp=datetime(2026, 8, 11, tzinfo=UTC),
        adapter_versions={"native-crawler": "0.1.0"},
        sitemap_state="AVAILABLE",
        scores=[],
    )
    assert run.ruleset_verified_date.isoformat() == "2026-08-11"
    assert run.sitemap_state.value == "AVAILABLE"


def test_ai_prompt_has_versioned_stable_fields() -> None:
    prompt = AIPrompt(
        prompt_id="brand-discovery-en",
        pack_version="1.0.0",
        locale="en",
        intent="brand_discovery",
        text="What is Example?",
        target_entities=["Example"],
        expected_evidence_needs=["official website", "independent source"],
        suggested_providers=["openai", "gemini"],
    )
    assert prompt.pack_version == "1.0.0"


def test_evidence_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="e-1",
            source_url="https://example.com",
            source_type="page",
            collector="native-crawler",
            observed_at=datetime.now(UTC),
            observed_value="value",
            confidence=1.2,
        )
