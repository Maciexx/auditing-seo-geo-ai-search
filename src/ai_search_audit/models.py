from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class FindingStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    INFERRED = "INFERRED"
    REQUIRES_ACCESS = "REQUIRES_ACCESS"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuleState(StrEnum):
    CURRENT = "CURRENT"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"


class ClaimModality(StrEnum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    POSSIBLE = "POSSIBLE"
    ASSERTED = "ASSERTED"
    REQUIRED = "REQUIRED"


class DataState(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED = "FAILED"


class SitemapState(StrEnum):
    AVAILABLE = "AVAILABLE"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    NOT_DISCOVERED = "NOT_DISCOVERED"
    FETCH_FAILED = "FETCH_FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Priority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Site(AuditModel):
    domain: str
    base_url: HttpUrl
    brand: str | None = None
    languages: list[str] = Field(default_factory=list)


class JsonLdParseError(AuditModel):
    message: str
    line: int = Field(ge=1)
    column: int = Field(ge=1)
    excerpt: str


class Page(AuditModel):
    url: HttpUrl
    final_url: HttpUrl
    status_code: int
    redirect_chain: list[str] = Field(default_factory=list)
    title: str | None = None
    meta_description: str | None = None
    h1: list[str] = Field(default_factory=list)
    h2: list[str] = Field(default_factory=list)
    canonical: str | None = None
    hreflang: dict[str, str] = Field(default_factory=dict)
    robots_directives: list[str] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)
    sitemap_references: list[str] = Field(default_factory=list)
    language: str | None = None
    json_ld: list[dict[str, Any]] = Field(default_factory=list)
    json_ld_errors: list[JsonLdParseError] = Field(default_factory=list)
    content_text: str = ""
    indexable: bool | None = None
    indexability_reasons: list[str] = Field(default_factory=list)
    depth: int | None = None


class Entity(AuditModel):
    brand: str
    type: str
    domain: str
    location: str | None = None
    languages: list[str] = Field(default_factory=list)
    facts: dict[str, list[str]] = Field(default_factory=dict)


class Evidence(AuditModel):
    evidence_id: str
    source_url: HttpUrl | None = None
    source_type: str
    source_scope: str = "same-origin"
    collector: str
    observed_at: datetime
    observed_value: Any
    content_fingerprint: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactualClaim(AuditModel):
    claim_id: str
    predicate: str
    value: str | int | float | bool | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    modality: ClaimModality
    negated: bool = False
    meaning: str


class Finding(AuditModel):
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
    affected_urls: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    rule_evidence_level: Literal["A", "B", "C", "D", "E"] | None = None
    rule_confidence: float | None = Field(default=None, ge=0, le=1)
    rule_scoring_weight: float | None = Field(default=None, ge=0)
    rule_state: RuleState | None = None
    rule_expires_at: date | None = None
    factual_claims: list[FactualClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def confirmed_findings_require_evidence(self) -> Finding:
        if self.status is FindingStatus.CONFIRMED and not self.evidence_ids:
            raise ValueError("confirmed findings require at least one evidence ID")
        return self


class Recommendation(AuditModel):
    recommendation_id: str
    finding_ids: list[str]
    title: str
    implementation: str
    priority: Priority
    confidence: float = Field(ge=0, le=1)


class AIPrompt(AuditModel):
    prompt_id: str
    pack_version: str
    locale: str
    intent: str
    text: str
    target_entities: list[str] = Field(default_factory=list)
    query_themes: list[str] = Field(default_factory=list)
    expected_evidence_needs: list[str] = Field(default_factory=list)
    suggested_providers: list[str] = Field(default_factory=list)


class AIObservation(AuditModel):
    observation_id: str
    prompt_id: str
    provider: str
    observed_at: datetime
    response_excerpt: str | None = None
    response_fingerprint: str | None = None
    citations: list[str] = Field(default_factory=list)
    brand_mentioned: bool | None = None
    prominence: float | None = Field(default=None, ge=0, le=1)
    grounded: bool | None = None
    reviewer_notes: str | None = None


class Competitor(AuditModel):
    name: str
    domain: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ExternalMention(AuditModel):
    mention_id: str
    url: HttpUrl
    publisher: str
    source_class: str
    claims: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    independent_source_key: str


class Backlink(AuditModel):
    source_url: HttpUrl
    target_url: HttpUrl
    anchor_text: str | None = None
    evidence_id: str


class Opportunity(AuditModel):
    opportunity_id: str
    title: str
    commercial_value: float | None = None
    organic_visibility_gap: float | None = None
    ai_visibility_gap: float | None = None
    strategic_relevance: float | None = None
    confidence: float = Field(ge=0, le=1)
    implementation_effort: float | None = None
    rank_value: float | None = None


class ScoreCheckResult(AuditModel):
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
    explanation_finding_ids: list[str] = Field(default_factory=list)
    explanation_observation_ids: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def explain_value_or_unavailability(self) -> ScoreCheckResult:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if numeric and self.score is None:
            raise ValueError(f"{self.state} score checks require a numeric score")
        if not numeric and self.score is not None:
            raise ValueError(f"{self.state} score checks cannot have a numeric score")
        if (
            self.score is not None
            and self.score < 100
            and not (self.explanation_finding_ids or self.explanation_observation_ids)
        ):
            raise ValueError("score checks below 100 require an explanation reference")
        if self.score is None and not self.unavailable_reason:
            raise ValueError("non-numeric score checks require unavailable_reason")
        return self


class ScoreResult(AuditModel):
    name: str
    state: DataState
    value: float | None = Field(default=None, ge=0, le=100)
    coverage: float = Field(default=0, ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    unavailable_inputs: list[str] = Field(default_factory=list)
    observed_checks: int = Field(default=0, ge=0)
    total_checks: int = Field(default=0, ge=0)
    checks: list[ScoreCheckResult] = Field(default_factory=list)
    explanation_finding_ids: list[str] = Field(default_factory=list)
    explanation_observation_ids: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def state_matches_value(self) -> ScoreResult:
        if (
            self.state
            in {
                DataState.UNAVAILABLE,
                DataState.UNKNOWN,
                DataState.UNSUPPORTED,
                DataState.FAILED,
            }
            and self.value is not None
        ):
            raise ValueError(f"{self.state} scores cannot have a numeric value")
        if self.state in {DataState.AVAILABLE, DataState.PARTIAL} and self.value is None:
            raise ValueError(f"{self.state} scores require a numeric value")
        if (
            self.value is not None
            and self.value < 100
            and not (self.explanation_finding_ids or self.explanation_observation_ids)
        ):
            raise ValueError("scores below 100 require an explanation reference")
        if self.value is None and not self.unavailable_reason:
            raise ValueError("non-numeric scores require unavailable_reason")
        return self


class AuditRun(AuditModel):
    audit_id: str
    site: Site
    audit_engine_version: str
    ruleset_version: str
    ruleset_verified_date: date
    timestamp: datetime
    adapter_versions: dict[str, str] = Field(default_factory=dict)
    adapter_states: dict[str, DataState] = Field(default_factory=dict)
    sitemap_state: SitemapState = SitemapState.UNAVAILABLE
    pages: list[Page] = Field(default_factory=list)
    entity: Entity | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    ai_prompts: list[AIPrompt] = Field(default_factory=list)
    ai_observations: list[AIObservation] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    external_mentions: list[ExternalMention] = Field(default_factory=list)
    entity_consistency_matrix: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    backlinks: list[Backlink] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    scores: list[ScoreResult]
    warnings: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
