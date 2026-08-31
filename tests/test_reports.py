from datetime import UTC, datetime

import pytest

from ai_search_audit.models import (
    AuditRun,
    ClaimModality,
    DataState,
    Evidence,
    FactualClaim,
    Finding,
    FindingStatus,
    Priority,
    Severity,
    Site,
)
from ai_search_audit.reports import (
    ClaimGuardError,
    RewriteResult,
    build_protected_claims_manifest,
    build_report_draft,
    claim_guard,
    evidence_validator,
    hash_draft,
    report_data,
)


def run_fixture() -> AuditRun:
    evidence = Evidence(
        evidence_id="e1",
        source_url="https://example.com",
        source_type="web_page",
        collector="crawler",
        observed_at=datetime.now(UTC),
        observed_value={"rooms": 16},
        confidence=0.9,
    )
    finding = Finding(
        finding_id="f1",
        category="entity",
        severity=Severity.MEDIUM,
        status=FindingStatus.INFERRED,
        rule_id="entity-1",
        technical_title="Room count differs",
        technical_description="A source reports 24 rooms.",
        client_title="Room information needs confirmation",
        client_explanation="One public source reports 24 rooms.",
        business_impact="This may make automated summaries less reliable.",
        implementation="Confirm the room count.",
        priority=Priority.P1,
        evidence_ids=["e1"],
        confidence=0.8,
        factual_claims=[
            FactualClaim(
                claim_id="f1-room-count",
                predicate="public_source.room_count",
                value="24",
                evidence_ids=["e1"],
                numbers=["24"],
                modality=ClaimModality.OBSERVED,
                negated=False,
                meaning="One public source reports 24 rooms.",
            ),
            FactualClaim(
                claim_id="f1-impact",
                predicate="automated_summary.reliability",
                value="may decrease",
                evidence_ids=["e1"],
                modality=ClaimModality.POSSIBLE,
                negated=False,
                meaning="The discrepancy may reduce automated-summary reliability.",
            ),
        ],
    )
    return AuditRun(
        audit_id="r1",
        site=Site(domain="example.com", base_url="https://example.com"),
        audit_engine_version="0.1.0",
        ruleset_version="2026.08.11",
        ruleset_verified_date="2026-08-11",
        timestamp=datetime.now(UTC),
        evidence=[evidence],
        findings=[finding],
        scores=[],
    )


def test_manifest_is_bound_to_draft_before_rewriting() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)
    assert manifest.draft_hash == hash_draft(draft)
    assert manifest.findings["f1"]["confidence"] == 0.8
    assert manifest.findings["f1"]["evidence_ids"] == ["e1"]


def test_claim_guard_rejects_stronger_certainty_and_changed_number() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)
    changed = draft.model_copy(deep=True)
    changed.findings[0].factual_claims[0].value = "60"
    changed.findings[0].factual_claims[0].numbers = ["60"]
    changed.findings[0].factual_claims[1].modality = ClaimModality.ASSERTED
    with pytest.raises(ClaimGuardError):
        claim_guard(changed, manifest)


def test_claim_guard_allows_client_wording_to_improve_without_changing_facts() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)
    improved = draft.model_copy(deep=True)
    improved.findings[0].client_title = "Verify the published room count"
    improved.findings[
        0
    ].client_explanation = "A public listing gives a room total that still needs confirmation."
    improved.findings[
        0
    ].business_impact = "This discrepancy could make automated summaries less dependable."
    improved.findings[0].implementation = "Confirm the total, then correct the listing."
    claim_guard(improved, manifest)
    rewrite = RewriteResult(
        draft=improved,
        route="test",
        agent_skill_state=DataState.AVAILABLE,
    )
    output = report_data(draft, rewrite)
    assert output["findings"][0]["client_title"] == "Verify the published room count"


def test_claim_guard_protects_modality_negation_meaning_status_confidence_and_priority() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)
    changes = [
        lambda finding: setattr(finding.factual_claims[0], "modality", ClaimModality.ASSERTED),
        lambda finding: setattr(finding.factual_claims[0], "negated", True),
        lambda finding: setattr(finding.factual_claims[0], "meaning", "A different claim."),
        lambda finding: setattr(finding, "status", FindingStatus.CONFIRMED),
        lambda finding: setattr(finding, "confidence", 1.0),
        lambda finding: setattr(finding, "priority", Priority.P0),
    ]
    for change in changes:
        modified = draft.model_copy(deep=True)
        change(modified.findings[0])
        with pytest.raises(ClaimGuardError):
            claim_guard(modified, manifest)


def test_evidence_validator_rejects_missing_reference() -> None:
    draft = build_report_draft(run_fixture())
    draft.findings[0].evidence_ids = ["missing"]
    with pytest.raises(ClaimGuardError, match="missing evidence"):
        evidence_validator(draft)
