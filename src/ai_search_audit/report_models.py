from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from .comparisons import (
    ComparisonBasis,
    ComparisonCausality,
    ComparisonLimitation,
    FindingChange,
    ValidationComparison,
    ValidationTimingState,
)
from .data_intake import (
    FactApprovalState,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilitySource,
)
from .diagnostic_models import (
    BenchmarkComparison,
    DiagnosticBinding,
    DiagnosticCollectionRange,
    DiagnosticRunReference,
)
from .models import (
    ClaimModality,
    DataState,
    FindingStatus,
    Priority,
    RuleState,
    Severity,
    SitemapState,
)
from .project_models import AuditStage, ReportStatus
from .visibility_metrics import (
    MetricWindow,
    VisibilityMetric,
    VisibilitySnapshot,
)

ReportLocale = Literal["pl", "en"]
REPORT_SCHEMA_VERSION = "1.2.0"
REPORT_TEMPLATE_VERSION = "1.2.0"
_COMPATIBLE_REPORT_TEMPLATES = {REPORT_SCHEMA_VERSION: frozenset({REPORT_TEMPLATE_VERSION})}


class ReportCompatibilityError(ValueError):
    pass


class FrozenReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        allow_inf_nan=False,
    )


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("report timestamps must be timezone-aware")
    return value


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

    @field_validator("collected_at")
    @classmethod
    def validate_collected_at(cls, value: datetime) -> datetime:
        return _aware_datetime(value)


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


class ProjectReportMetadata(FrozenReportModel):
    project_id: str
    version_id: str
    version_number: int = Field(ge=1)
    stage: AuditStage
    report_status: ReportStatus
    source_audit_id: str | None = None

    @model_validator(mode="after")
    def validate_status_and_source(self) -> ProjectReportMetadata:
        allowed = {
            AuditStage.PUBLIC: frozenset({ReportStatus.PUBLIC_EVIDENCE_DRAFT}),
            AuditStage.CONTEXT: frozenset(
                {ReportStatus.CLIENT_CONTEXT_DRAFT, ReportStatus.CLIENT_VALIDATED}
            ),
            AuditStage.VALIDATION: frozenset(
                {ReportStatus.CLIENT_CONTEXT_DRAFT, ReportStatus.CLIENT_VALIDATED}
            ),
        }[self.stage]
        if self.report_status not in allowed:
            raise ValueError("project report status does not match the trusted audit stage")
        initial_public = self.stage is AuditStage.PUBLIC and self.version_number == 1
        if initial_public and self.source_audit_id is not None:
            raise ValueError("public-v1 project report must not reference a source audit")
        if not initial_public and self.source_audit_id is None:
            raise ValueError("subsequent project report requires a source audit")
        return self


class ReportDateRange(FrozenReportModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> ReportDateRange:
        if self.end < self.start:
            raise ValueError("report source date range end must not precede start")
        return self


class ReportSourceProvenance(FrozenReportModel):
    source_id: str
    filename: str
    sha256: str
    byte_count: int = Field(ge=0)
    platform: VisibilitySource
    report_type: str
    date_range: ReportDateRange | None = None
    filters: tuple[str, ...] = ()
    exported_at: datetime | None = None
    processed_at: datetime
    deleted_at: datetime

    @field_validator("exported_at", "processed_at", "deleted_at")
    @classmethod
    def validate_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_datetime(value)

    @model_validator(mode="after")
    def validate_timestamps(self) -> ReportSourceProvenance:
        if self.deleted_at < self.processed_at:
            raise ValueError("report source deleted timestamp precedes processing")
        if self.exported_at is not None and self.exported_at > self.processed_at:
            raise ValueError("report source export timestamp follows processing")
        if self.date_range is not None and self.date_range.end > self.processed_at.date():
            raise ValueError("report source date range follows processing")
        return self


class OwnerFactReportItem(FrozenReportModel):
    fact_id: str
    field: str
    value: str | int | float | bool | None
    as_of: date | None = None
    approval_state: FactApprovalState
    conflict_ids: tuple[str, ...] = ()
    resolved_conflict_ids: tuple[str, ...] = ()
    provenance: ReportSourceProvenance


class OwnerContextReportSection(FrozenReportModel):
    schema_version: str
    project_id: str
    canonical_domain: str
    processed_at: datetime
    deleted_at: datetime
    facts: tuple[OwnerFactReportItem, ...] = ()
    sources: tuple[ReportSourceProvenance, ...] = ()

    @field_validator("processed_at", "deleted_at")
    @classmethod
    def validate_aware_timestamp(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def validate_owner_context(self) -> OwnerContextReportSection:
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("owner report facts require unique fact IDs")
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("owner report sources require unique source IDs")
        if any(sources.get(fact.provenance.source_id) != fact.provenance for fact in self.facts):
            raise ValueError("owner fact provenance must resolve to an owner source")
        if any(
            source.processed_at != self.processed_at or source.deleted_at != self.deleted_at
            for source in self.sources
        ):
            raise ValueError("owner report source timestamps must match context timestamps")
        return self


class ReportMetricWindow(FrozenReportModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> ReportMetricWindow:
        if self.end < self.start:
            raise ValueError("report metric window end must not precede start")
        return self


class ReportVisibilityPoint(FrozenReportModel):
    period_start: date
    period_end: date | None = None
    value: float

    @model_validator(mode="after")
    def validate_period(self) -> ReportVisibilityPoint:
        if self.period_end is not None and self.period_end < self.period_start:
            raise ValueError("report metric period_end must not precede period_start")
        return self


class ReportVisibilityMetric(FrozenReportModel):
    metric_id: str
    metric: str
    unit: str
    state: DataState
    value: float | None = None
    coverage: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source_ids: tuple[str, ...]
    window: ReportMetricWindow | None = None
    segments: tuple[str, ...] = ()
    definitions: tuple[str, ...] = ()
    points: tuple[ReportVisibilityPoint, ...] = ()

    @model_validator(mode="after")
    def validate_numeric_state(self) -> ReportVisibilityMetric:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if numeric and (
            self.value is None or not self.points or self.window is None or self.coverage <= 0
        ):
            raise ValueError("numeric report metric requires value, points, window, and coverage")
        if not numeric and (self.value is not None or self.points or self.window is not None):
            raise ValueError("non-numeric report metric must not carry an observed value")
        if len(self.source_ids) != len(set(self.source_ids)) or not self.source_ids:
            raise ValueError("report metric source IDs must be non-empty and unique")
        if numeric:
            if self.state is DataState.AVAILABLE and self.coverage != 1:
                raise ValueError("AVAILABLE report metric requires full coverage")
            if self.state is DataState.PARTIAL and self.coverage >= 1:
                raise ValueError("PARTIAL report metric requires partial coverage")
        VisibilityMetric(
            metric_id=self.metric_id,
            metric=self.metric,
            unit=self.unit,
            state=self.state,
            value=self.value,
            coverage=self.coverage,
            confidence=self.confidence,
            source_ids=self.source_ids,
            window=(
                None
                if self.window is None
                else MetricWindow(start=self.window.start, end=self.window.end)
            ),
            segments=self.segments,
            definitions=self.definitions,
            points=tuple(
                VisibilityMetricPoint(
                    period_start=point.period_start,
                    period_end=point.period_end,
                    value=point.value,
                )
                for point in self.points
            ),
        )
        return self


class MeasurementLimitation(FrozenReportModel):
    metric_id: str
    state: DataState


class MeasurementReportSection(FrozenReportModel):
    schema_version: str
    project_id: str
    canonical_domain: str
    processed_at: datetime
    deleted_at: datetime
    metrics: tuple[ReportVisibilityMetric, ...] = ()
    sources: tuple[ReportSourceProvenance, ...] = ()
    limitations: tuple[MeasurementLimitation, ...] = ()

    @field_validator("processed_at", "deleted_at")
    @classmethod
    def validate_aware_timestamp(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def validate_measurement(self) -> MeasurementReportSection:
        metrics = {metric.metric_id: metric for metric in self.metrics}
        if len(metrics) != len(self.metrics):
            raise ValueError("measurement report metrics require unique metric IDs")
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("measurement report sources require unique source IDs")
        referenced_sources = {
            source_id for metric in self.metrics for source_id in metric.source_ids
        }
        if referenced_sources != set(sources):
            raise ValueError("measurement metric sources must resolve exactly")
        if any(
            source.processed_at != self.processed_at or source.deleted_at != self.deleted_at
            for source in self.sources
        ):
            raise ValueError("measurement source timestamps must match snapshot timestamps")
        limitations = {item.metric_id: item for item in self.limitations}
        if len(limitations) != len(self.limitations):
            raise ValueError("measurement limitations require unique metric IDs")
        expected_limited = {
            metric.metric_id for metric in self.metrics if metric.state is not DataState.AVAILABLE
        }
        if set(limitations) != expected_limited:
            raise ValueError("measurement limitations must resolve to limited metrics")
        if any(metrics[item.metric_id].state is not item.state for item in self.limitations):
            raise ValueError("measurement limitation state must match its metric")
        VisibilitySnapshot(
            project_id=self.project_id,
            canonical_domain=self.canonical_domain,
            processed_at=self.processed_at,
            deleted_at=self.deleted_at,
            metrics=tuple(
                VisibilityMetric(
                    metric_id=metric.metric_id,
                    metric=metric.metric,
                    unit=metric.unit,
                    state=metric.state,
                    value=metric.value,
                    coverage=metric.coverage,
                    confidence=metric.confidence,
                    source_ids=metric.source_ids,
                    window=(
                        None
                        if metric.window is None
                        else MetricWindow(
                            start=metric.window.start,
                            end=metric.window.end,
                        )
                    ),
                    segments=metric.segments,
                    definitions=metric.definitions,
                    points=tuple(
                        VisibilityMetricPoint(
                            period_start=point.period_start,
                            period_end=point.period_end,
                            value=point.value,
                        )
                        for point in metric.points
                    ),
                )
                for metric in self.metrics
            ),
            sources=tuple(
                SourceArtifactProvenance.model_validate(source.model_dump(mode="python"))
                for source in self.sources
            ),
        )
        return self


class ReportComparisonWindow(FrozenReportModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> ReportComparisonWindow:
        if self.end < self.start:
            raise ValueError("report comparison window end must not precede start")
        return self


class ReportMetricComparison(FrozenReportModel):
    metric_id: str
    metric: str
    unit: str
    state: DataState
    basis: ComparisonBasis
    baseline_state: DataState
    follow_up_state: DataState
    baseline_value: float | None = None
    follow_up_value: float | None = None
    absolute_delta: float | None = None
    relative_delta_percent: float | None = None
    baseline_window: ReportComparisonWindow | None = None
    follow_up_window: ReportComparisonWindow | None = None
    baseline_source_ids: tuple[str, ...] = ()
    follow_up_source_ids: tuple[str, ...] = ()
    coverage: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    limitations: tuple[ComparisonLimitation, ...] = ()


class ReportFindingComparison(FrozenReportModel):
    stable_identity: str
    rule_id: str
    fact_identity: str
    change: FindingChange
    baseline_finding_ids: tuple[str, ...] = ()
    follow_up_finding_ids: tuple[str, ...] = ()


class ReportAIVisibilityComparison(FrozenReportModel):
    state: DataState
    baseline_value: float | None = None
    follow_up_value: float | None = None
    absolute_delta: float | None = None
    prompt_pack_version: str | None = None
    setup_fingerprint: str | None = None
    baseline_observation_ids: tuple[str, ...] = ()
    follow_up_observation_ids: tuple[str, ...] = ()
    baseline_prompt_ids: tuple[str, ...] = ()
    follow_up_prompt_ids: tuple[str, ...] = ()
    baseline_measurable_prompt_ids: tuple[str, ...] = ()
    follow_up_measurable_prompt_ids: tuple[str, ...] = ()
    baseline_canonical_prompt_ids: tuple[str, ...] = ()
    follow_up_canonical_prompt_ids: tuple[str, ...] = ()
    limitations: tuple[ComparisonLimitation, ...] = ()


class ValidationComparisonReportSection(FrozenReportModel):
    schema_version: str
    baseline_audit_id: str | None = None
    follow_up_audit_id: str | None = None
    implementation_date: date | None = None
    validation_target_date: date | None = None
    observed_at: date | None = None
    timing_state: ValidationTimingState | None = None
    timing_warning: str | None = None
    causality: ComparisonCausality
    chronology_statement: str
    metrics: tuple[ReportMetricComparison, ...] = ()
    findings: tuple[ReportFindingComparison, ...] = ()
    ai_visibility: ReportAIVisibilityComparison | None = None

    @model_validator(mode="after")
    def validate_canonical_comparison(self) -> ValidationComparisonReportSection:
        ValidationComparison.model_validate(self.model_dump(mode="python"))
        return self


class SupplementaryDiagnosticComparison(FrozenReportModel):
    """Validated run provenance and numeric observations, not fresh-crawl findings."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    baseline_reference: DiagnosticRunReference
    follow_up_reference: DiagnosticRunReference
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    follow_up_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_binding: DiagnosticBinding
    follow_up_binding: DiagnosticBinding
    baseline_collection_range: DiagnosticCollectionRange
    follow_up_collection_range: DiagnosticCollectionRange
    comparison: BenchmarkComparison

    @model_validator(mode="after")
    def validate_bindings(self) -> SupplementaryDiagnosticComparison:
        for reference, binding in (
            (self.baseline_reference, self.baseline_binding),
            (self.follow_up_reference, self.follow_up_binding),
        ):
            if reference.source_version != binding.source_version:
                raise ValueError("supplementary diagnostic source reference mismatch")
        if (
            self.baseline_binding.project_id,
            self.baseline_binding.domain,
            self.baseline_binding.report_locale,
        ) != (
            self.follow_up_binding.project_id,
            self.follow_up_binding.domain,
            self.follow_up_binding.report_locale,
        ):
            raise ValueError("supplementary diagnostic project/domain/locale mismatch")
        return self


def report_context_digest(
    project: ProjectReportMetadata,
    owner_context: OwnerContextReportSection | None,
    measurement: MeasurementReportSection | None,
    validation_comparison: ValidationComparisonReportSection | None = None,
    supplementary_diagnostic_comparison: SupplementaryDiagnosticComparison | None = None,
) -> str:
    payload = {
        "project": project.model_dump(mode="json"),
        "owner_context": (None if owner_context is None else owner_context.model_dump(mode="json")),
        "measurement": (None if measurement is None else measurement.model_dump(mode="json")),
        "validation_comparison": (
            None if validation_comparison is None else validation_comparison.model_dump(mode="json")
        ),
    }
    if supplementary_diagnostic_comparison is not None:
        payload["supplementary_diagnostic_comparison"] = (
            supplementary_diagnostic_comparison.model_dump(mode="json")
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    project: ProjectReportMetadata | None = None
    owner_context: OwnerContextReportSection | None = None
    measurement: MeasurementReportSection | None = None
    validation_comparison: ValidationComparisonReportSection | None = None
    supplementary_diagnostic_comparison: SupplementaryDiagnosticComparison | None = None
    context_digest: str | None = Field(default=None, min_length=64, max_length=64)
    anti_slop: AntiSlopReportMetadata
    renderer: RendererMetadata

    @model_serializer(mode="wrap")
    def serialize_legacy_compatible(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        payload: dict[str, object] = handler(self)
        if self.supplementary_diagnostic_comparison is None:
            payload.pop("supplementary_diagnostic_comparison", None)
        return payload

    @field_validator("audit_timestamp")
    @classmethod
    def validate_audit_timestamp(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def validate_canonical_context_digest(self) -> ClientReportData:
        validate_report_compatibility(
            schema_version=self.report_schema_version,
            template_version=self.report_template_version,
            renderer=self.renderer,
        )
        if self.project is None:
            if (
                self.owner_context is not None
                or self.measurement is not None
                or self.validation_comparison is not None
                or self.supplementary_diagnostic_comparison is not None
            ):
                raise ValueError("report context requires project metadata")
            if self.context_digest is not None:
                raise ValueError("public report must not contain a context digest")
            return self
        if self.owner_context is not None and (
            self.owner_context.project_id != self.project.project_id
            or self.owner_context.canonical_domain != self.target_domain
        ):
            raise ValueError("owner report context identity does not match client report")
        if self.measurement is not None and (
            self.measurement.project_id != self.project.project_id
            or self.measurement.canonical_domain != self.target_domain
        ):
            raise ValueError("measurement report identity does not match client report")
        if (
            self.owner_context is not None
            and self.measurement is not None
            and (
                self.owner_context.project_id != self.measurement.project_id
                or self.owner_context.canonical_domain != self.measurement.canonical_domain
            )
        ):
            raise ValueError("owner and measurement report context identities differ")
        if self.project.stage is AuditStage.VALIDATION:
            if self.validation_comparison is None:
                raise ValueError("validation report requires a canonical comparison")
            if any(
                value is None
                for value in (
                    self.validation_comparison.implementation_date,
                    self.validation_comparison.validation_target_date,
                    self.validation_comparison.observed_at,
                    self.validation_comparison.timing_state,
                )
            ):
                raise ValueError("validation report requires complete timing metadata")
            if self.validation_comparison.follow_up_audit_id != self.audit_id:
                raise ValueError("validation comparison follow-up identity differs")
            if (
                self.project.source_audit_id is None
                or self.validation_comparison.baseline_audit_id != self.project.source_audit_id
            ):
                raise ValueError(
                    "validation comparison baseline identity differs from project source"
                )
        elif self.validation_comparison is not None:
            raise ValueError("non-validation report must not contain a validation comparison")
        if self.supplementary_diagnostic_comparison is not None:
            binding = self.supplementary_diagnostic_comparison.baseline_binding
            if (
                self.project.stage is not AuditStage.VALIDATION
                or binding.project_id != self.project.project_id
                or binding.domain != self.target_domain
                or binding.report_locale != self.report_locale
            ):
                raise ValueError("supplementary diagnostic report identity mismatch")
        expected = report_context_digest(
            self.project,
            self.owner_context,
            self.measurement,
            self.validation_comparison,
            self.supplementary_diagnostic_comparison,
        )
        if self.context_digest != expected:
            raise ValueError("report context digest does not match canonical sections")
        return self
