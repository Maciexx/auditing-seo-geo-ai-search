"""Trusted owner-context projection and report-status derivation."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, TypeAlias, cast
from urllib.parse import urlsplit

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
    SEARCH_AI_PROPERTY = "search_ai_property"
    SEARCH_AI_SELECTED = "search_ai_selected"
    SEARCH_AI_EFFECTIVE = "search_ai_effective"
    SEARCH_AI_PARENT = "search_ai_parent"
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


class SearchAIControl(FrozenOwnerContextModel):
    selected: Literal["include", "exclude", "inherit", "unknown"] = "unknown"
    effective: Literal["include", "exclude", "unknown"] = "unknown"
    property: str | None = None
    parent: str | None = None
    as_of: date | None = None
    source_ids: tuple[str, ...] = ()


def search_ai_control(context: OwnerContext | None) -> SearchAIControl:
    """Project complete, approved owner facts; crawling never establishes this setting."""
    if context is None:
        return SearchAIControl()
    fields = {"search_ai_property", "search_ai_selected", "search_ai_effective", "search_ai_parent"}
    facts = [fact for fact in context.facts if fact.field.value in fields]
    by_field = {fact.field.value: fact for fact in facts}
    if (
        len(by_field) != len(facts)
        or not facts
        or any(
            fact.approval_state is not FactApprovalState.APPROVED
            or fact.as_of is None
            or fact.as_of > context.processed_at.date()
            or fact.conflict_ids
            for fact in facts
        )
        or len({fact.as_of for fact in facts}) != 1
    ):
        return SearchAIControl()
    values = {key: fact.value for key, fact in by_field.items()}
    selected, effective = values.get("search_ai_selected"), values.get("search_ai_effective")
    prop, parent = values.get("search_ai_property"), values.get("search_ai_parent")
    if (
        selected not in {"include", "exclude", "inherit"}
        or effective not in {"include", "exclude"}
        or not isinstance(prop, str)
        or (selected == "inherit" and not isinstance(parent, str))
        or (selected != "inherit" and (selected != effective or parent is not None))
    ):
        return SearchAIControl()

    def property_host(value: str) -> str | None:
        if value.startswith("sc-domain:"):
            host = value.removeprefix("sc-domain:")
            return host if host and "/" not in host and ":" not in host else None
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            return None
        return parsed.hostname

    if property_host(prop) != context.canonical_domain:
        return SearchAIControl()
    if isinstance(parent, str):
        parent_host = property_host(parent)
        if not parent_host or parent.rstrip("/") == prop.rstrip("/"):
            return SearchAIControl()
        if parent.startswith("sc-domain:"):
            if not (
                context.canonical_domain == parent_host
                or context.canonical_domain.endswith("." + parent_host)
            ):
                return SearchAIControl()
        elif prop.startswith("sc-domain:") or not prop.startswith(parent.rstrip("/") + "/"):
            return SearchAIControl()
    return SearchAIControl.model_validate(
        {
            "selected": selected,
            "effective": effective,
            "property": prop,
            "parent": parent,
            "as_of": facts[0].as_of,
            "source_ids": tuple(sorted({fact.provenance.source_id for fact in facts})),
        }
    )
