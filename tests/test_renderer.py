import json
import subprocess
from importlib import import_module
from pathlib import Path

import pytest

from ai_search_audit import reports
from ai_search_audit.models import DataState, ScoreResult
from ai_search_audit.report_models import ClientReportData
from ai_search_audit.reports import (
    apply_anti_slop,
    build_client_report_data,
    build_protected_claims_manifest,
    build_report_draft,
)
from tests.test_reports import run_fixture


def renderer_module():
    return import_module("ai_search_audit.renderer")


def client_report_data(report_locale: str = "en"):
    renderer = renderer_module()
    run = run_fixture()
    run.scores = [
        ScoreResult(
            name="Technical Search Readiness",
            state=DataState.PARTIAL,
            value=100,
            coverage=0.5,
            confidence=0.45,
            unavailable_inputs=["one check not assessed"],
            observed_checks=1,
            total_checks=2,
        ),
        ScoreResult(
            name="Measurement Maturity",
            state=DataState.UNAVAILABLE,
            unavailable_inputs=["client measurement access"],
            unavailable_reason="Client measurement access was not supplied.",
        ),
    ]
    draft = build_report_draft(run, report_locale=report_locale)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    return build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer.build_renderer_metadata(),
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/logo.png",
        "http://example.com/styles.css",
        "data:text/plain,unexpected",
        "file:///etc/passwd",
    ],
)
def test_renderer_rejects_network_and_uncontrolled_resources(url: str) -> None:
    renderer = renderer_module()
    with pytest.raises(renderer.RenderSecurityError):
        renderer.local_asset_fetcher(url)


def test_renderer_accepts_client_report_data_only(tmp_path: Path) -> None:
    renderer = renderer_module()
    with pytest.raises(TypeError, match="ClientReportData"):
        renderer.render_client_report({}, tmp_path / "invalid.pdf")


def test_direct_renderer_rejects_incompatible_persisted_report_data(tmp_path: Path) -> None:
    renderer = renderer_module()
    payload = json.loads(client_report_data().model_dump_json())
    payload["report_template_version"] = "9.0.0"
    persisted = ClientReportData.model_validate(payload)

    with pytest.raises(reports.ReportCompatibilityError, match="incompatible"):
        renderer.render_client_report(persisted, tmp_path / "incompatible.pdf")

    assert not (tmp_path / "incompatible.pdf").exists()


def test_same_environment_render_is_byte_identical(tmp_path: Path) -> None:
    renderer = renderer_module()
    data = client_report_data()

    first = renderer.render_client_report(data, tmp_path / "first.pdf")
    second = renderer.render_client_report(data, tmp_path / "second.pdf")

    assert first.sha256 == second.sha256
    assert (tmp_path / "first.pdf").read_bytes() == (tmp_path / "second.pdf").read_bytes()
    assert first.pdf_identifier == second.pdf_identifier
    assert first.byte_count < 5_000_000


def test_pdf_contains_required_sections_and_explicit_uncertainty(tmp_path: Path) -> None:
    renderer = renderer_module()
    output = tmp_path / "client-report.pdf"
    renderer.render_client_report(client_report_data(), output)
    extracted = subprocess.run(
        ["pdftotext", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    for section in (
        "Executive Summary",
        "Current State",
        "Readiness Scorecard",
        "Prioritized Findings",
        "AI Search Access",
        "Entity Consistency",
        "Implementation Roadmap",
        "Evidence Appendix",
        "Methodology and Limitations",
    ):
        assert section in extracted
    assert "100" in extracted and "in assessed scope" in extracted
    assert "Coverage 50%" in extracted
    assert "UNAVAILABLE" in extracted
    assert "Measurement Maturity\n0" not in extracted


def test_polish_report_localizes_fixed_pdf_labels_and_html_language(tmp_path: Path) -> None:
    renderer = renderer_module()
    data = client_report_data(report_locale="pl")

    assert data.report_locale == "pl"
    html = renderer.render_report_html(data)
    assert '<html lang="pl">' in html

    output = tmp_path / "client-report-pl.pdf"
    renderer.render_client_report(data, output)
    extracted = subprocess.run(
        ["pdftotext", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    for section in (
        "Podsumowanie zarządcze",
        "Bieżący stan",
        "Ocena gotowości",
        "Najważniejsze ustalenia",
        "Dostęp w wyszukiwaniu AI",
        "Spójność encji",
        "Plan wdrożenia",
        "Aneks dowodowy",
        "Metodyka i ograniczenia",
    ):
        assert section in extracted
    assert "Executive Summary" not in extracted
    assert "Current State" not in extracted
    assert "Readiness Scorecard" not in extracted
    assert "Strona 2 /" in extracted


def test_renderer_has_no_rewrite_or_analysis_imports() -> None:
    source = (Path(__file__).parents[1] / "src" / "ai_search_audit" / "renderer.py").read_text()
    for forbidden in (
        "AntiSlop",
        "RewriteProvider",
        "ai_search_audit.reports",
        "ai_search_audit.orchestrator",
        "ai_search_audit.crawler",
        "ai_search_audit.analyzers",
    ):
        assert forbidden not in source


def test_paged_components_avoid_breaks_where_practical() -> None:
    css = (
        Path(__file__).parents[1]
        / "src"
        / "ai_search_audit"
        / "templates"
        / "client-report"
        / "v1"
        / "styles.css"
    ).read_text()
    assert "break-inside: avoid" in css
    assert ".entity-table col.fact" in css
    assert "word-break: break-all" in css
