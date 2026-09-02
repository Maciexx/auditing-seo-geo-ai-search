"""Trusted owner-context projection and report-status derivation."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data_intake import (
    MAX_INTAKE_SOURCES,
    MAX_OWNER_FACTS,
    FactApprovalState,
    ProcessedIntake,
    SourceArtifactProvenance,
)
from .project_models import (
    AuditStage,
    ReportStatus,
    normalize_canonical_domain,
    validate_project_id,
)

ScalarFactValue: TypeAlias = str | int | float | bool | None


class OwnerFactField(StrEnum):
    CANONICAL_ENTITY = "canonical_entity"
    MARKET = "market"
    LANGUAGE = "language"
    SEASONALITY = "seasonality"
    CONTROLLED_PROFILE = "controlled_profile"
    COMPETITOR = "competitor"
    IMPLEMENTATION_DATE = "implementation_date"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"


class FrozenOwnerContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class OwnerFact(FrozenOwnerContextModel):
    fact_id: str
    field: OwnerFactField
    value: ScalarFactValue
    as_of: date | None = None
    approval_state: FactApprovalState
    conflict_ids: tuple[str, ...] = ()
    resolved_conflict_ids: tuple[str, ...] = ()
    provenance: SourceArtifactProvenance

    @model_validator(mode="after")
    def validate_conflict_sets(self) -> OwnerFact:
        if set(self.conflict_ids) & set(self.resolved_conflict_ids):
            raise ValueError("owner fact conflict and resolution references must be disjoint")
        return self


class OwnerContext(FrozenOwnerContextModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    canonical_domain: str
    processed_at: datetime
    deleted_at: datetime
    facts: tuple[OwnerFact, ...] = Field(default=(), max_length=MAX_OWNER_FACTS)
    sources: tuple[SourceArtifactProvenance, ...] = Field(
        min_length=1, max_length=MAX_INTAKE_SOURCES
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("canonical_domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)

    @model_validator(mode="after")
    def validate_fact_identity(self) -> OwnerContext:
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("owner context requires unique fact_id values")
        sources_by_id = {source.source_id: source for source in self.sources}
        if len(sources_by_id) != len(self.sources):
            raise ValueError("owner context requires unique source_id values")
        if any(
            sources_by_id.get(fact.provenance.source_id) != fact.provenance for fact in self.facts
        ):
            raise ValueError("owner facts must retain declared source provenance")
        if any(
            source.processed_at != self.processed_at or source.deleted_at != self.deleted_at
            for source in self.sources
        ):
            raise ValueError("owner context timestamps must match source provenance")
        return self


def build_owner_context(processed: ProcessedIntake) -> OwnerContext:
    """Project normalized owner facts onto immutable provenance-backed records."""
    sources = {source.source_id: source for source in processed.sources}
    facts = tuple(
        OwnerFact(
            fact_id=fact.fact_id,
            field=cast(OwnerFactField, fact.field),
            value=fact.value,
            as_of=fact.as_of,
            approval_state=fact.approval_state,
            conflict_ids=fact.conflict_ids,
            resolved_conflict_ids=fact.resolved_conflict_ids,
            provenance=sources[fact.source_id],
        )
        for fact in processed.owner_facts
    )
    return OwnerContext(
        project_id=processed.project_id,
        canonical_domain=processed.canonical_domain,
        processed_at=processed.processed_at,
        deleted_at=processed.deleted_at,
        facts=facts,
        sources=processed.sources,
    )


def derive_report_status(
    *,
    stage: AuditStage,
    used_fact_ids: Collection[str],
    owner_context: OwnerContext,
) -> ReportStatus:
    """Derive status exclusively from the stage and materially used owner facts."""
    stage = AuditStage(stage)
    if stage is AuditStage.PUBLIC:
        return ReportStatus.PUBLIC_EVIDENCE_DRAFT

    used_ids = tuple(dict.fromkeys(used_fact_ids))
    if not used_ids:
        return ReportStatus.CLIENT_CONTEXT_DRAFT
    by_id = {fact.fact_id: fact for fact in owner_context.facts}
    missing = sorted(set(used_ids).difference(by_id))
    if missing:
        raise ValueError(f"unknown owner fact references: {', '.join(missing)}")

    used_facts = tuple(by_id[fact_id] for fact_id in used_ids)
    resolved_conflicts = {
        conflict_id
        for fact in used_facts
        if fact.field is OwnerFactField.CONTRADICTION_RESOLUTION
        and fact.approval_state is FactApprovalState.APPROVED
        and fact.value is not None
        for conflict_id in fact.resolved_conflict_ids
    }
    all_approved = all(fact.approval_state is FactApprovalState.APPROVED for fact in used_facts)
    unresolved = {
        conflict_id
        for fact in used_facts
        for conflict_id in fact.conflict_ids
        if conflict_id not in resolved_conflicts
    }
    if all_approved and not unresolved:
        return ReportStatus.CLIENT_VALIDATED
    return ReportStatus.CLIENT_CONTEXT_DRAFT
