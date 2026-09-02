from datetime import UTC, date, datetime
from importlib import import_module

import pytest
from pydantic import ValidationError

from ai_search_audit import reports
from ai_search_audit.comparisons import compare_visibility
from ai_search_audit.data_intake import (
    DateRange,
    FactApprovalState,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilitySource,
)
from ai_search_audit.models import DataState, ScoreResult
from ai_search_audit.owner_context import OwnerContext, OwnerFact, OwnerFactField
from ai_search_audit.project_models import AuditStage, ReportStatus
from ai_search_audit.report_models import (
    ClientReportData,
    MeasurementReportSection,
    OwnerContextReportSection,
    ProjectReportMetadata,
    ReportVisibilityMetric,
    ReportVisibilityPoint,
    report_context_digest,
)
from ai_search_audit.reports import (
    NarrativeRewritePayload,
    apply_anti_slop,
    build_protected_claims_manifest,
    build_report_draft,
)
from ai_search_audit.visibility_metrics import MetricWindow, VisibilityMetric, VisibilitySnapshot
from tests.test_reports import run_fixture


def renderer_metadata():
    report_models = import_module("ai_search_audit.report_models")
    return report_models.RendererMetadata(
        renderer="WeasyPrint",
        renderer_version="69.0",
        template_version=reports.REPORT_TEMPLATE_VERSION,
        template_digest="a" * 64,
        font_asset_digest="b" * 64,
        environment_fingerprint="c" * 64,
    )


def _source() -> SourceArtifactProvenance:
    timestamp = datetime(2026, 8, 31, 12, tzinfo=UTC)
    return SourceArtifactProvenance(
        source_id="ga4-export",
        filename="ga4.csv",
        sha256="a" * 64,
        byte_count=123,
        platform=VisibilitySource.GOOGLE_ANALYTICS,
        report_type="visibility-export",
        date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 31)),
        filters=("channel=AI Assistant",),
        exported_at=timestamp,
        processed_at=timestamp,
        deleted_at=timestamp,
    )


def _owner_context() -> OwnerContext:
    source = _source().model_copy(
        update={
            "source_id": "owner-answer",
            "filename": "owner.json",
            "platform": VisibilitySource.MANUAL,
            "report_type": "owner-context",
            "date_range": None,
            "filters": (),
        }
    )
    return OwnerContext(
        project_id="example",
        canonical_domain="example.com",
        processed_at=source.processed_at,
        deleted_at=source.deleted_at,
        sources=(source,),
        facts=(
            OwnerFact(
                fact_id="fact-market",
                field=OwnerFactField.MARKET,
                value="Poland",
                approval_state=FactApprovalState.APPROVED,
                conflict_ids=("conflict-market",),
                provenance=source,
            ),
        ),
    )


def _visibility_snapshot() -> VisibilitySnapshot:
    source = _source()
    return VisibilitySnapshot(
        project_id="example",
        canonical_domain="example.com",
        processed_at=source.processed_at,
        deleted_at=source.deleted_at,
        sources=(source,),
        metrics=(
            VisibilityMetric(
                metric_id="ga4-ai-sessions",
                metric="ga4.ai_assistant.sessions",
                unit="sessions",
                state=DataState.AVAILABLE,
                value=0,
                coverage=1,
                confidence=0.9,
                source_ids=(source.source_id,),
                window=MetricWindow(start=date(2026, 8, 1), end=date(2026, 8, 31)),
                segments=("channel=AI Assistant",),
                definitions=(
                    "report_type=visibility-export",
                    "aggregation=sum",
                    "cadence=monthly",
                    "channel=AI Assistant",
                ),
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=0),),
            ),
        ),
    )


def _project_metadata() -> ProjectReportMetadata:
    return ProjectReportMetadata(
        project_id="example",
        version_id="context-v2",
        version_number=2,
        stage=AuditStage.CONTEXT,
        report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
        source_audit_id="audit-public",
    )


def _context_report():
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    return reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=_project_metadata(),
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
    )


def _validation_comparison():
    baseline = _visibility_snapshot()
    follow_up_source = baseline.sources[0].model_copy(
        update={
            "source_id": "ga4-follow-up",
            "filename": "ga4-follow-up.csv",
            "date_range": DateRange(start=date(2026, 9, 1), end=date(2026, 9, 30)),
        }
    )
    follow_up_metric = baseline.metrics[0].model_copy(
        update={
            "source_ids": (follow_up_source.source_id,),
            "window": MetricWindow(start=date(2026, 9, 1), end=date(2026, 9, 30)),
            "points": (VisibilityMetricPoint(period_start=date(2026, 9, 1), value=0),),
        }
    )
    follow_up = baseline.model_copy(
        update={"sources": (follow_up_source,), "metrics": (follow_up_metric,)}
    )
    return compare_visibility(
        baseline,
        follow_up,
        baseline_audit_id="audit-baseline",
        follow_up_audit_id="r1",
        implementation_date=date(2026, 5, 1),
        observed_at=date(2026, 9, 30),
    )


def test_final_client_report_data_is_versioned_frozen_and_evidence_resolved() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)

    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
    )

    assert report.report_schema_version == reports.REPORT_SCHEMA_VERSION
    assert report.report_template_version == reports.REPORT_TEMPLATE_VERSION
    assert report.findings[0].finding_id == "f1"
    assert report.evidence_appendix[0].evidence_id == "e1"
    assert report.evidence_appendix[0].source == "crawler"
    assert report.audit_timestamp == run.timestamp
    with pytest.raises(ValidationError, match="frozen"):
        report.target_domain = "changed.example"  # type: ignore[misc]


def _report_with_nested_data():
    run = run_fixture()
    run.scores = [
        ScoreResult(
            name="Technical Search Readiness",
            state=DataState.AVAILABLE,
            value=100,
            coverage=1,
            confidence=0.9,
        )
    ]
    run.entity_consistency_matrix = {"room_count": {"24": ["e1"]}}
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    return reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
    )


def test_nested_report_findings_are_frozen() -> None:
    report = _report_with_nested_data()

    with pytest.raises(ValidationError, match="frozen"):
        report.findings[0].client_title = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        report.findings[0].evidence_ids += ("e2",)  # type: ignore[assignment]


def test_nested_report_scores_are_frozen() -> None:
    report = _report_with_nested_data()

    with pytest.raises(ValidationError, match="frozen"):
        report.scores[0].value = 0  # type: ignore[misc]


def test_nested_entity_matrix_is_frozen_report_dtos() -> None:
    report = _report_with_nested_data()
    fact = report.entity_consistency.facts[0]

    assert fact.fact == "room_count"
    assert fact.values[0].value == "24"
    assert fact.values[0].evidence_ids == ("e1",)
    with pytest.raises(ValidationError, match="frozen"):
        fact.fact = "brand"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        fact.values[0].evidence_ids += ("e2",)  # type: ignore[assignment]


def test_schema_template_incompatibility_blocks_final_model() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    incompatible = renderer_metadata().model_copy(update={"template_version": "2.0.0"})

    with pytest.raises(reports.ReportCompatibilityError, match="incompatible"):
        reports.build_client_report_data(
            run,
            rewrite,
            manifest,
            renderer_metadata=incompatible,
        )


def test_report_draft_carries_canonical_metadata_without_render_time() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    assert draft.audit_timestamp == run.timestamp
    assert draft.audit_engine_version == run.audit_engine_version
    assert draft.ruleset_version == run.ruleset_version
    assert draft.report_schema_version == reports.REPORT_SCHEMA_VERSION
    assert draft.report_template_version == reports.REPORT_TEMPLATE_VERSION
    assert draft.audit_timestamp != datetime.now(UTC)


def test_project_context_and_measurement_projection_is_deeply_frozen() -> None:
    report = _context_report()

    assert report.project == _project_metadata()
    assert report.owner_context.facts[0].value == "Poland"
    assert report.owner_context.facts[0].provenance.source_id == "owner-answer"
    assert report.measurement.metrics[0].value == 0
    assert report.measurement.metrics[0].coverage == 1
    assert report.measurement.metrics[0].points[0].value == 0
    with pytest.raises(ValidationError, match="frozen"):
        report.project.report_status = ReportStatus.CLIENT_VALIDATED  # type: ignore[union-attr,misc]
    with pytest.raises(ValidationError, match="frozen"):
        report.owner_context.facts[0].value = "changed"  # type: ignore[union-attr,misc]
    with pytest.raises(ValidationError, match="frozen"):
        report.measurement.metrics[0].coverage = 0  # type: ignore[union-attr,misc]
    with pytest.raises(TypeError):
        report.measurement.metrics[0].points[0] = None  # type: ignore[index,union-attr]


def test_validation_comparison_projection_is_frozen_and_canonical() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    project = _project_metadata().model_copy(
        update={
            "version_id": "validation-v3",
            "version_number": 3,
            "stage": AuditStage.VALIDATION,
            "report_status": ReportStatus.CLIENT_CONTEXT_DRAFT,
            "source_audit_id": "audit-baseline",
        }
    )
    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=project,
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
        validation_comparison=_validation_comparison(),
    )

    assert report.validation_comparison is not None
    assert report.validation_comparison.causality.value == "NOT_ESTABLISHED"
    assert report.validation_comparison.metrics[0].absolute_delta == 0
    with pytest.raises(ValidationError, match="frozen"):
        report.validation_comparison.metrics[0].absolute_delta = 2  # type: ignore[misc]


def test_direct_report_load_binds_validation_baseline_to_project_source() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    project = _project_metadata().model_copy(
        update={
            "version_id": "validation-v3",
            "version_number": 3,
            "stage": AuditStage.VALIDATION,
            "source_audit_id": "audit-baseline",
        }
    )
    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=project,
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
        validation_comparison=_validation_comparison(),
    )
    payload = report.model_dump(mode="json")
    payload["validation_comparison"]["baseline_audit_id"] = "audit-other"
    comparison = reports.ValidationComparisonReportSection.model_validate(
        payload["validation_comparison"]
    )
    payload["context_digest"] = report_context_digest(
        report.project,
        report.owner_context,
        report.measurement,
        comparison,
    )

    with pytest.raises(ValidationError, match="baseline.*source"):
        ClientReportData.model_validate(payload)


def test_direct_validation_report_load_requires_complete_timing_after_digest_recompute() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    project = _project_metadata().model_copy(
        update={
            "version_id": "validation-v3",
            "version_number": 3,
            "stage": AuditStage.VALIDATION,
            "source_audit_id": "audit-baseline",
        }
    )
    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=project,
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
        validation_comparison=_validation_comparison(),
    )
    assert report.validation_comparison is not None
    incomplete = report.validation_comparison.model_copy(
        update={
            "implementation_date": None,
            "validation_target_date": None,
            "observed_at": None,
            "timing_state": None,
            "timing_warning": None,
        }
    )
    payload = report.model_dump(mode="json")
    payload["validation_comparison"] = incomplete.model_dump(mode="json")
    payload["context_digest"] = report_context_digest(
        report.project,
        report.owner_context,
        report.measurement,
        incomplete,
    )

    with pytest.raises(ValidationError, match="complete timing"):
        ClientReportData.model_validate(payload)


def test_direct_validation_report_rejects_causal_chronology_after_digest_recompute() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    project = _project_metadata().model_copy(
        update={
            "version_id": "validation-v3",
            "version_number": 3,
            "stage": AuditStage.VALIDATION,
            "source_audit_id": "audit-baseline",
        }
    )
    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=project,
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
        validation_comparison=_validation_comparison(),
    )
    assert report.validation_comparison is not None
    causal = report.validation_comparison.model_copy(
        update={"chronology_statement": "Implementation definitely caused the change."}
    )
    payload = report.model_dump(mode="json")
    payload["validation_comparison"] = causal.model_dump(mode="json")
    payload["context_digest"] = report_context_digest(
        report.project,
        report.owner_context,
        report.measurement,
        causal,
    )

    with pytest.raises(ValidationError, match="chronology_statement"):
        ClientReportData.model_validate(payload)


@pytest.mark.parametrize(
    ("schema_version", "template_version", "renderer_version"),
    (
        ("9.0.0", reports.REPORT_TEMPLATE_VERSION, reports.REPORT_TEMPLATE_VERSION),
        (reports.REPORT_SCHEMA_VERSION, "9.0.0", "9.0.0"),
        (reports.REPORT_SCHEMA_VERSION, reports.REPORT_TEMPLATE_VERSION, "9.0.0"),
    ),
)
def test_direct_report_load_rejects_unsupported_or_mismatched_compatibility(
    schema_version: str,
    template_version: str,
    renderer_version: str,
) -> None:
    payload = _report_with_nested_data().model_dump(mode="json")
    payload["report_schema_version"] = schema_version
    payload["report_template_version"] = template_version
    payload["renderer"]["template_version"] = renderer_version

    with pytest.raises(ValidationError, match="incompatible"):
        ClientReportData.model_validate(payload)


def test_anti_slop_payload_has_no_validation_comparison_state() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)

    def provider(payload: NarrativeRewritePayload) -> NarrativeRewritePayload:
        assert not hasattr(payload, "validation_comparison")
        return payload

    apply_anti_slop(draft, manifest, provider=provider)


def test_public_report_remains_valid_without_optional_project_context() -> None:
    report = _report_with_nested_data()

    assert report.project is None
    assert report.owner_context is None
    assert report.measurement is None


def test_anti_slop_payload_has_no_project_or_measurement_canonical_state() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)

    def provider(payload: NarrativeRewritePayload) -> NarrativeRewritePayload:
        for forbidden in (
            "project_id",
            "report_status",
            "owner_context",
            "measurement",
            "scores",
            "evidence",
            "manifests",
        ):
            assert not hasattr(payload, forbidden)
        return payload

    rewrite = apply_anti_slop(draft, manifest, provider=provider)
    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=_project_metadata(),
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
    )

    assert report.project.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT
    assert report.owner_context.facts[0].value == "Poland"


def test_context_projection_rejects_mismatched_project_identity() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    owner_context = _owner_context().model_copy(update={"project_id": "other"})

    with pytest.raises(ValueError, match="project identity"):
        reports.build_client_report_data(
            run,
            rewrite,
            manifest,
            renderer_metadata=renderer_metadata(),
            project_metadata=_project_metadata(),
            owner_context=owner_context,
            visibility_snapshot=_visibility_snapshot(),
        )


def test_direct_compile_rejects_client_validated_with_unresolved_owner_conflict() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    elevated = _project_metadata().model_copy(
        update={"report_status": ReportStatus.CLIENT_VALIDATED}
    )

    with pytest.raises(ValueError, match="trusted report status"):
        reports.build_client_report_data(
            run,
            rewrite,
            manifest,
            renderer_metadata=renderer_metadata(),
            project_metadata=elevated,
            owner_context=_owner_context(),
            visibility_snapshot=_visibility_snapshot(),
        )


def test_direct_compile_requires_validated_status_for_fully_approved_validation_context() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    owner = _owner_context()
    approved_fact = owner.facts[0].model_copy(update={"conflict_ids": ()})
    owner = owner.model_copy(update={"facts": (approved_fact,)})
    understated = _project_metadata().model_copy(
        update={
            "stage": AuditStage.VALIDATION,
            "report_status": ReportStatus.CLIENT_CONTEXT_DRAFT,
        }
    )

    with pytest.raises(ValueError, match="trusted report status"):
        reports.build_client_report_data(
            run,
            rewrite,
            manifest,
            renderer_metadata=renderer_metadata(),
            project_metadata=understated,
            owner_context=owner,
            visibility_snapshot=_visibility_snapshot(),
        )


@pytest.mark.parametrize("state", (DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED))
def test_non_numeric_report_metric_state_cannot_carry_zero(state: DataState) -> None:
    with pytest.raises(ValidationError, match="non-numeric"):
        ReportVisibilityMetric(
            metric_id="invalid",
            metric="diagnostic.invalid",
            unit="observations",
            state=state,
            value=0,
            coverage=0,
            confidence=0.4,
            source_ids=("source",),
        )


def test_report_context_boundary_resolves_against_canonical_aggregates_and_version() -> None:
    canonical = _context_report()
    assert canonical.project is not None
    changed_project = canonical.project.model_copy(update={"version_id": "context-v9"})
    tampered = canonical.model_copy(
        update={
            "project": changed_project,
            "context_digest": report_context_digest(
                changed_project,
                canonical.owner_context,
                canonical.measurement,
            ),
        }
    )

    with pytest.raises(ValueError, match="trusted version"):
        reports.validate_client_report_context(
            tampered,
            project_metadata=_project_metadata(),
            owner_context=_owner_context(),
            visibility_snapshot=_visibility_snapshot(),
        )


def test_partial_measurement_retains_coverage_and_explicit_limitation() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    snapshot = _visibility_snapshot()
    partial_metric = snapshot.metrics[0].model_copy(
        update={"state": DataState.PARTIAL, "coverage": 0.5}
    )
    partial_snapshot = snapshot.model_copy(update={"metrics": (partial_metric,)})

    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=_project_metadata(),
        owner_context=_owner_context(),
        visibility_snapshot=partial_snapshot,
    )

    assert report.measurement is not None
    assert report.measurement.metrics[0].coverage == 0.5
    assert report.measurement.limitations[0].metric_id == "ga4-ai-sessions"
    assert report.measurement.limitations[0].state is DataState.PARTIAL


def test_metric_period_end_is_retained_in_frozen_report_projection() -> None:
    run = run_fixture()
    draft = build_report_draft(run)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    snapshot = _visibility_snapshot()
    point = snapshot.metrics[0].points[0].model_copy(update={"period_end": date(2026, 8, 31)})
    metric = snapshot.metrics[0].model_copy(update={"points": (point,)})
    snapshot = snapshot.model_copy(update={"metrics": (metric,)})

    report = reports.build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer_metadata(),
        project_metadata=_project_metadata(),
        owner_context=_owner_context(),
        visibility_snapshot=snapshot,
    )

    assert report.measurement is not None
    assert report.measurement.metrics[0].points[0].period_end == date(2026, 8, 31)


def _recompute_context_digest(payload: dict[str, object]) -> None:
    project = ProjectReportMetadata.model_validate(payload["project"])
    owner = OwnerContextReportSection.model_validate(payload["owner_context"])
    measurement = MeasurementReportSection.model_validate(payload["measurement"])
    payload["context_digest"] = report_context_digest(project, owner, measurement)


@pytest.mark.parametrize(
    "mutation",
    (
        "owner-domain",
        "measurement-project",
        "duplicate-owner-source",
        "duplicate-fact",
        "unknown-fact-source",
        "owner-timestamp",
        "duplicate-metric",
        "unknown-metric-source",
        "dangling-limitation",
        "wrong-limitation-state",
        "measurement-timestamp",
        "point-window",
        "metric-value",
        "metric-nan",
        "point-inf",
        "metric-unit",
        "metric-aggregation",
        "metric-cadence",
        "source-platform",
    ),
)
def test_recomputed_digest_cannot_bypass_context_semantic_validation(
    mutation: str,
) -> None:
    payload = _context_report().model_dump(mode="json")
    owner = payload["owner_context"]
    measurement = payload["measurement"]
    assert isinstance(owner, dict)
    assert isinstance(measurement, dict)
    if mutation == "owner-domain":
        owner["canonical_domain"] = "other.example"
    elif mutation == "measurement-project":
        measurement["project_id"] = "other"
    elif mutation == "duplicate-owner-source":
        owner["sources"].append(dict(owner["sources"][0]))
    elif mutation == "duplicate-fact":
        owner["facts"].append(dict(owner["facts"][0]))
    elif mutation == "unknown-fact-source":
        owner["facts"][0]["provenance"]["source_id"] = "missing"
    elif mutation == "owner-timestamp":
        owner["sources"][0]["processed_at"] = "2026-08-30T12:00:00Z"
    elif mutation == "duplicate-metric":
        measurement["metrics"].append(dict(measurement["metrics"][0]))
    elif mutation == "unknown-metric-source":
        measurement["metrics"][0]["source_ids"] = ["missing"]
    elif mutation == "dangling-limitation":
        measurement["limitations"] = [{"metric_id": "missing", "state": "UNKNOWN"}]
    elif mutation == "wrong-limitation-state":
        measurement["limitations"] = [{"metric_id": "ga4-ai-sessions", "state": "UNKNOWN"}]
    elif mutation == "measurement-timestamp":
        measurement["sources"][0]["deleted_at"] = "2026-09-01T12:00:00Z"
    elif mutation == "point-window":
        measurement["metrics"][0]["window"]["start"] = "2026-08-02"
    elif mutation == "metric-value":
        measurement["metrics"][0]["value"] = 17
    elif mutation == "metric-nan":
        measurement["metrics"][0]["value"] = float("nan")
    elif mutation == "point-inf":
        measurement["metrics"][0]["points"][0]["value"] = float("inf")
    elif mutation == "metric-unit":
        measurement["metrics"][0]["unit"] = "users"
    elif mutation == "metric-aggregation":
        measurement["metrics"][0]["definitions"][1] = "aggregation=latest"
    elif mutation == "metric-cadence":
        measurement["metrics"][0]["definitions"][2] = "cadence=window"
    else:
        measurement["sources"][0]["platform"] = "manual"
    with pytest.raises(
        ValidationError,
        match=(
            "context|owner|measurement|source|fact|metric|point|limitation|"
            "timestamp|window|processing|export"
        ),
    ):
        _recompute_context_digest(payload)
        reports.ClientReportData.model_validate(payload)


def test_persisted_report_source_rejects_naive_datetime_as_validation_error() -> None:
    payload = _context_report().model_dump(mode="json")
    owner = payload["owner_context"]
    assert isinstance(owner, dict)
    owner["sources"][0]["processed_at"] = "2026-08-31T12:00:00"

    with pytest.raises(ValidationError, match="timezone-aware"):
        reports.ClientReportData.model_validate(payload)


def test_persisted_report_rejects_naive_audit_timestamp() -> None:
    payload = _context_report().model_dump(mode="json")
    payload["audit_timestamp"] = "2026-08-31T12:00:00"

    with pytest.raises(ValidationError, match="timezone-aware"):
        reports.ClientReportData.model_validate(payload)


def test_persisted_report_rejects_naive_evidence_collection_timestamp() -> None:
    payload = _context_report().model_dump(mode="json")
    payload["evidence_appendix"][0]["collected_at"] = "2026-08-31T12:00:00"

    with pytest.raises(ValidationError, match="timezone-aware"):
        reports.ClientReportData.model_validate(payload)


def test_report_visibility_point_rejects_reversed_period() -> None:
    with pytest.raises(ValidationError, match="period_end"):
        ReportVisibilityPoint(
            period_start=date(2026, 8, 31),
            period_end=date(2026, 8, 1),
            value=1,
        )
