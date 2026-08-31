from datetime import UTC, datetime
from importlib import import_module

import pytest
from pydantic import ValidationError

from ai_search_audit import reports
from ai_search_audit.models import DataState, ScoreResult
from ai_search_audit.reports import (
    apply_anti_slop,
    build_protected_claims_manifest,
    build_report_draft,
)
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
