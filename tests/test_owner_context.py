from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_search_audit.data_intake import (
    FactApprovalState,
    OwnerFactInput,
    ProcessedIntake,
    SourceArtifactProvenance,
    VisibilitySource,
)
from ai_search_audit.owner_context import (
    OwnerContext,
    OwnerFactField,
    build_owner_context,
    derive_report_status,
)
from ai_search_audit.project_models import AuditStage, ReportStatus

NOW = datetime(2026, 9, 1, 10, tzinfo=UTC)


def _source() -> SourceArtifactProvenance:
    return SourceArtifactProvenance(
        source_id="owner-answer",
        filename="owner-answers.json",
        sha256="a" * 64,
        byte_count=10,
        platform=VisibilitySource.MANUAL,
        report_type="owner-context",
        processed_at=NOW,
        deleted_at=NOW,
    )


def _processed(*facts: OwnerFactInput) -> ProcessedIntake:
    return ProcessedIntake(
        project_id="example",
        canonical_domain="example.com",
        processed_at=NOW,
        deleted_at=NOW,
        owner_facts=facts,
        sources=(_source(),),
    )


@pytest.mark.parametrize(
    "field",
    (
        OwnerFactField.CANONICAL_ENTITY,
        OwnerFactField.MARKET,
        OwnerFactField.LANGUAGE,
        OwnerFactField.SEASONALITY,
        OwnerFactField.CONTROLLED_PROFILE,
        OwnerFactField.COMPETITOR,
        OwnerFactField.IMPLEMENTATION_DATE,
        OwnerFactField.CONTRADICTION_RESOLUTION,
    ),
)
def test_owner_context_retains_supported_owner_fact_kinds_and_provenance(
    field: OwnerFactField,
) -> None:
    source_fact = OwnerFactInput(
        fact_id=f"fact-{field.value}",
        field=field.value,
        value="confirmed value",
        source_id="owner-answer",
        approval_state=FactApprovalState.APPROVED,
        conflict_ids=("conflict-open",),
        resolved_conflict_ids=("conflict-resolved",),
    )

    context = build_owner_context(_processed(source_fact))

    fact = context.facts[0]
    assert fact.field is field
    assert fact.value == "confirmed value"
    assert fact.approval_state is FactApprovalState.APPROVED
    assert fact.conflict_ids == ("conflict-open",)
    assert fact.resolved_conflict_ids == ("conflict-resolved",)
    assert fact.provenance == _source()


def test_owner_context_preserves_unknown_null_fact_without_inventing_value() -> None:
    context = build_owner_context(
        _processed(
            OwnerFactInput(
                fact_id="fact-market",
                field=OwnerFactField.MARKET.value,
                value=None,
                source_id="owner-answer",
            )
        )
    )

    assert context.facts[0].value is None
    assert context.facts[0].approval_state is FactApprovalState.UNKNOWN


def test_owner_context_rejects_noncanonical_owner_fact_field() -> None:
    intake = _processed(
        OwnerFactInput(
            fact_id="fact-goal",
            field="commercial_goal",
            value="grow revenue",
            source_id="owner-answer",
        )
    )

    with pytest.raises(ValidationError, match="field"):
        build_owner_context(intake)


@pytest.mark.parametrize(
    (
        "approval",
        "conflicts",
        "resolved",
        "resolver_approval",
        "include_resolver",
        "expected",
    ),
    (
        (
            FactApprovalState.UNKNOWN,
            (),
            (),
            FactApprovalState.APPROVED,
            True,
            ReportStatus.CLIENT_CONTEXT_DRAFT,
        ),
        (
            FactApprovalState.APPROVED,
            ("conflict-room-count",),
            (),
            FactApprovalState.APPROVED,
            True,
            ReportStatus.CLIENT_CONTEXT_DRAFT,
        ),
        (
            FactApprovalState.APPROVED,
            ("conflict-room-count",),
            ("conflict-room-count",),
            FactApprovalState.UNKNOWN,
            True,
            ReportStatus.CLIENT_CONTEXT_DRAFT,
        ),
        (
            FactApprovalState.APPROVED,
            ("conflict-room-count",),
            ("conflict-room-count",),
            FactApprovalState.APPROVED,
            False,
            ReportStatus.CLIENT_CONTEXT_DRAFT,
        ),
        (
            FactApprovalState.APPROVED,
            ("conflict-room-count",),
            ("conflict-room-count",),
            FactApprovalState.APPROVED,
            True,
            ReportStatus.CLIENT_VALIDATED,
        ),
        (
            FactApprovalState.APPROVED,
            (),
            (),
            FactApprovalState.APPROVED,
            False,
            ReportStatus.CLIENT_VALIDATED,
        ),
    ),
)
def test_report_status_is_trusted_pipeline_derivation(
    approval: FactApprovalState,
    conflicts: tuple[str, ...],
    resolved: tuple[str, ...],
    resolver_approval: FactApprovalState,
    include_resolver: bool,
    expected: ReportStatus,
) -> None:
    fact = OwnerFactInput(
        fact_id="fact-entity",
        field=OwnerFactField.CANONICAL_ENTITY.value,
        value="Example Client",
        source_id="owner-answer",
        approval_state=approval,
        conflict_ids=conflicts,
    )
    resolution_fact = OwnerFactInput(
        fact_id="fact-resolution",
        field=OwnerFactField.CONTRADICTION_RESOLUTION.value,
        value="resolved by owner",
        source_id="owner-answer",
        approval_state=resolver_approval,
        resolved_conflict_ids=resolved,
    )
    context = build_owner_context(_processed(fact, resolution_fact))
    used_ids = (fact.fact_id, resolution_fact.fact_id) if include_resolver else (fact.fact_id,)

    assert (
        derive_report_status(
            stage=AuditStage.CONTEXT,
            used_fact_ids=used_ids,
            owner_context=context,
        )
        is expected
    )


def test_non_resolution_fact_cannot_resolve_conflict_even_when_approved_and_used() -> None:
    fact = OwnerFactInput(
        fact_id="fact-entity",
        field=OwnerFactField.CANONICAL_ENTITY.value,
        value="Example Client",
        source_id="owner-answer",
        approval_state=FactApprovalState.APPROVED,
        conflict_ids=("c1",),
    )
    false_resolver = OwnerFactInput(
        fact_id="fact-market",
        field=OwnerFactField.MARKET.value,
        value="PL",
        source_id="owner-answer",
        approval_state=FactApprovalState.APPROVED,
        resolved_conflict_ids=("c1",),
    )
    context = build_owner_context(_processed(fact, false_resolver))

    assert (
        derive_report_status(
            stage=AuditStage.CONTEXT,
            used_fact_ids=(fact.fact_id, false_resolver.fact_id, fact.fact_id),
            owner_context=context,
        )
        is ReportStatus.CLIENT_CONTEXT_DRAFT
    )


def test_public_stage_and_empty_context_cannot_be_upgraded() -> None:
    context = build_owner_context(_processed())

    assert (
        derive_report_status(
            stage=AuditStage.PUBLIC,
            used_fact_ids=(),
            owner_context=context,
        )
        is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    )
    assert (
        derive_report_status(
            stage=AuditStage.CONTEXT,
            used_fact_ids=(),
            owner_context=context,
        )
        is ReportStatus.CLIENT_CONTEXT_DRAFT
    )


def test_status_derivation_rejects_unknown_used_fact_reference() -> None:
    context = build_owner_context(_processed())

    with pytest.raises(ValueError, match="unknown owner fact"):
        derive_report_status(
            stage=AuditStage.CONTEXT,
            used_fact_ids=("fact-missing",),
            owner_context=context,
        )


def test_owner_context_models_have_no_report_status_input() -> None:
    payload = build_owner_context(_processed()).model_dump(mode="json")
    payload["report_status"] = ReportStatus.CLIENT_VALIDATED.value

    with pytest.raises(ValidationError, match="report_status"):
        OwnerContext.model_validate(payload)


def test_owner_context_publishes_explicit_collection_bounds() -> None:
    properties = OwnerContext.model_json_schema()["properties"]

    assert properties["facts"]["maxItems"] == 500
    assert properties["sources"]["maxItems"] == 32
