from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ClaimModality,
    DataState,
    FindingStatus,
    Priority,
    RuleState,
    Severity,
    SitemapState,
)

ReportLocale = Literal["pl", "en"]
REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_TEMPLATE_VERSION = "1.0.0"
_COMPATIBLE_REPORT_TEMPLATES = {REPORT_SCHEMA_VERSION: frozenset({REPORT_TEMPLATE_VERSION})}


class ReportCompatibilityError(ValueError):
    pass


class FrozenReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class RendererMetadata(FrozenReportModel):
    renderer: str
    renderer_version: str
    template_version: str
    template_digest: str = Field(min_length=64, max_length=64)
    font_asset_digest: str = Field(min_length=64, max_length=64)
    environment_fingerprint: str = Field(min_length=64, max_length=64)


def validate_report_compatibility(
    *, schema_version: str, template_version: str, renderer: RendererMetadata
) -> None:
    compatible = _COMPATIBLE_REPORT_TEMPLATES.get(schema_version, frozenset())
    if template_version not in compatible or renderer.template_version != template_version:
        raise ReportCompatibilityError(
            "incompatible report schema, template, or renderer template versions: "
            f"schema={schema_version}, template={template_version}, "
            f"renderer={renderer.template_version}"
        )


class EvidenceAppendixItem(FrozenReportModel):
    evidence_id: str
    source_type: str
    source_scope: str
    source: str
    url: str | None = None
    observation: str
    collected_at: datetime
    confidence: float = Field(ge=0, le=1)


class ReportFactualClaim(FrozenReportModel):
    claim_id: str
    predicate: str
    value: str | int | float | bool | None = None
    evidence_ids: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    modality: ClaimModality
    negated: bool = False
    meaning: str


class ReportFinding(FrozenReportModel):
    finding_id: str
    category: str
    severity: Severity
    status: FindingStatus
    rule_id: str
    technical_title: str
    technical_description: str
    client_title: str
    client_explanation: str
    business_impact: str
    implementation: str
    priority: Priority
    affected_urls: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    rule_evidence_level: Literal["A", "B", "C", "D", "E"] | None = None
    rule_confidence: float | None = Field(default=None, ge=0, le=1)
    rule_scoring_weight: float | None = Field(default=None, ge=0)
    rule_state: RuleState | None = None
    rule_expires_at: date | None = None
    factual_claims: tuple[ReportFactualClaim, ...] = ()


class ReportRecommendation(FrozenReportModel):
    recommendation_id: str
    finding_ids: tuple[str, ...]
    title: str
    implementation: str
    priority: Priority
    confidence: float = Field(ge=0, le=1)


class ReportScoreCheck(FrozenReportModel):
    name: str
    state: DataState
    score: float | None = Field(default=None, ge=0, le=100)
    weight: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    rule_id: str | None = None
    rule_evidence_level: Literal["A", "B", "C", "D", "E"] | None = None
    rule_confidence: float | None = Field(default=None, ge=0, le=1)
    rule_scoring_weight: float | None = Field(default=None, ge=0)
    rule_state: RuleState | None = None
    rule_expires_at: date | None = None
    explanation_finding_ids: tuple[str, ...] = ()
    explanation_observation_ids: tuple[str, ...] = ()
    unavailable_reason: str | None = None


class ReportScore(FrozenReportModel):
    name: str
    state: DataState
    value: float | None = Field(default=None, ge=0, le=100)
    coverage: float = Field(default=0, ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    unavailable_inputs: tuple[str, ...] = ()
    observed_checks: int = Field(default=0, ge=0)
    total_checks: int = Field(default=0, ge=0)
    checks: tuple[ReportScoreCheck, ...] = ()
    explanation_finding_ids: tuple[str, ...] = ()
    explanation_observation_ids: tuple[str, ...] = ()
    unavailable_reason: str | None = None


class AISearchReportSection(FrozenReportModel):
    access_finding_ids: tuple[str, ...] = ()
    prompt_count: int = Field(ge=0)
    prompt_pack_version: str | None = None
    observed_visibility_state: DataState


class EntityFactValue(FrozenReportModel):
    value: str
    evidence_ids: tuple[str, ...] = ()


class EntityFact(FrozenReportModel):
    fact: str
    values: tuple[EntityFactValue, ...] = ()


class EntityConsistencyReportSection(FrozenReportModel):
    facts: tuple[EntityFact, ...] = ()
    finding_ids: tuple[str, ...] = ()


class AntiSlopReportMetadata(FrozenReportModel):
    route: str
    agent_skill_state: DataState


class ClientReportData(FrozenReportModel):
    report_schema_version: str
    report_template_version: str
    report_locale: ReportLocale
    audit_id: str
    target_domain: str
    brand: str
    audit_timestamp: datetime
    audit_engine_version: str
    ruleset_version: str
    sitemap_state: SitemapState
    audience: str
    executive_summary: str
    scores: tuple[ReportScore, ...]
    findings: tuple[ReportFinding, ...]
    recommendations: tuple[ReportRecommendation, ...]
    ai_search: AISearchReportSection
    entity_consistency: EntityConsistencyReportSection
    evidence_appendix: tuple[EvidenceAppendixItem, ...]
    methodology: tuple[str, ...]
    limitations: tuple[str, ...]
    anti_slop: AntiSlopReportMetadata
    renderer: RendererMetadata
