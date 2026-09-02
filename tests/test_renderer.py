import json
import subprocess
import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_search_audit import reports
from ai_search_audit.comparisons import (
    ComparisonBasis,
    ComparisonLimitation,
    compare_observed_ai_visibility,
)
from ai_search_audit.data_intake import VisibilitySource
from ai_search_audit.models import DataState, ScoreResult
from ai_search_audit.project_models import AuditStage, ReportStatus
from ai_search_audit.report_models import ClientReportData, ReportVisibilityMetric
from ai_search_audit.reports import (
    apply_anti_slop,
    build_client_report_data,
    build_protected_claims_manifest,
    build_report_draft,
)
from tests.test_report_compilation import (
    _context_report,
    _owner_context,
    _validation_comparison,
    _visibility_snapshot,
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


def validation_report_data(report_locale: str = "en"):
    renderer = renderer_module()
    run = run_fixture()
    draft = build_report_draft(run, report_locale=report_locale)
    manifest = build_protected_claims_manifest(draft)
    rewrite = apply_anti_slop(draft, manifest)
    project = reports.ProjectReportMetadata(
        project_id="example",
        version_id="validation-v3",
        version_number=3,
        stage=AuditStage.VALIDATION,
        report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
        source_audit_id="audit-baseline",
    )
    comparison = _validation_comparison().model_copy(
        update={
            "ai_visibility": compare_observed_ai_visibility(
                (),
                (),
                baseline_prompt_pack_version="1.0.0",
                follow_up_prompt_pack_version="1.0.0",
                baseline_setup_fingerprint=None,
                follow_up_setup_fingerprint=None,
            )
        }
    )
    return build_client_report_data(
        run,
        rewrite,
        manifest,
        renderer_metadata=renderer.build_renderer_metadata(),
        project_metadata=project,
        owner_context=_owner_context(),
        visibility_snapshot=_visibility_snapshot(),
        validation_comparison=comparison,
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
    persisted = client_report_data().model_copy(update={"report_template_version": "9.0.0"})

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


@pytest.mark.parametrize(
    ("locale", "heading", "chronology", "limitation", "ai_heading", "forbidden"),
    (
        (
            "en",
            "Validation Comparison",
            "Observed after implementation",
            "Relative change is unavailable because the baseline is zero.",
            "Observed AI visibility comparison",
            "caused by implementation",
        ),
        (
            "pl",
            "Porównanie walidacyjne",
            "Zaobserwowano po wdrożeniu",
            "Zmiana procentowa jest niedostępna, ponieważ wartość bazowa wynosi zero.",
            "Porównanie zaobserwowanej widoczności AI",
            "spowodowane wdrożeniem",
        ),
    ),
)
def test_validation_comparison_is_localized_and_noncausal(
    locale: str,
    heading: str,
    chronology: str,
    limitation: str,
    ai_heading: str,
    forbidden: str,
) -> None:
    html = renderer_module().render_report_html(validation_report_data(locale))

    assert heading in html
    assert chronology in html
    assert limitation in html
    assert ai_heading in html
    assert forbidden not in html.casefold()
    assert "NOT_ESTABLISHED" not in html
    assert "DIRECT" not in html
    assert "audit-baseline" in html


def test_polish_validation_missing_baseline_value_has_non_overlapping_lines(
    tmp_path: Path,
) -> None:
    renderer = renderer_module()
    data = validation_report_data("pl")
    comparison = data.validation_comparison
    assert comparison is not None
    metric = comparison.metrics[0].model_copy(
        update={
            "state": DataState.UNKNOWN,
            "basis": ComparisonBasis.NON_COMPARABLE,
            "baseline_state": DataState.UNAVAILABLE,
            "follow_up_state": DataState.AVAILABLE,
            "baseline_value": None,
            "follow_up_value": 0,
            "absolute_delta": None,
            "relative_delta_percent": None,
            "baseline_window": None,
            "baseline_source_ids": (),
            "coverage": 0,
            "confidence": 0,
            "limitations": (ComparisonLimitation.MISSING_BASELINE,),
        }
    )
    second_metric = metric.model_copy(
        update={
            "metric_id": "second-metric",
            "metric": "second.metric",
        }
    )
    report = data.model_copy(
        update={
            "validation_comparison": comparison.model_copy(
                update={"metrics": (metric, second_metric)}
            )
        }
    )
    output = tmp_path / "validation-pl-missing-baseline.pdf"
    renderer.render_client_report(report, output)
    bbox = subprocess.run(
        ["pdftotext", "-bbox-layout", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    root = ET.fromstring(bbox)
    metric_ids = ("ga4-ai-sessions", "second-metric")
    target_page = next(
        page
        for page in root.iter()
        if page.tag.endswith("page")
        and all(
            metric_id
            in " ".join((word.text or "") for word in page.iter() if word.tag.endswith("word"))
            for metric_id in metric_ids
        )
    )
    target_lines: list[ET.Element] | None = None
    metric_blocks: dict[str, ET.Element] = {}
    blocks = [element for element in target_page.iter() if element.tag.endswith("block")]
    for block in blocks:
        words = [(word.text or "") for word in block.iter() if word.tag.endswith("word")]
        block_text = " ".join(words)
        if "Brak wartości liczbowej" in block_text and "NIEDOSTĘP" in block_text:
            target_lines = target_lines or [
                element for element in block if element.tag.endswith("line")
            ]
        for metric_id in metric_ids:
            if metric_id in block_text:
                metric_blocks[metric_id] = block

    assert target_lines is not None
    assert len(target_lines) >= 3
    vertical_gaps = [
        float(following.attrib["yMin"]) - float(previous.attrib["yMax"])
        for previous, following in zip(target_lines, target_lines[1:], strict=False)
    ]
    assert min(vertical_gaps) >= 1
    assert set(metric_blocks) == {"ga4-ai-sessions", "second-metric"}
    first = metric_blocks["ga4-ai-sessions"]
    second = metric_blocks["second-metric"]
    assert float(first.attrib["yMax"]) < float(second.attrib["yMin"])
    first_row_blocks = [
        block
        for block in blocks
        if float(first.attrib["yMin"]) <= float(block.attrib["yMin"]) < float(second.attrib["yMin"])
    ]
    assert first_row_blocks
    assert max(float(block.attrib["yMax"]) for block in first_row_blocks) < float(
        second.attrib["yMin"]
    )


@pytest.mark.parametrize(
    ("locale", "causality", "bases"),
    (
        (
            "en",
            "Not established",
            ("Direct comparison", "Year-over-year comparison", "Not comparable"),
        ),
        (
            "pl",
            "Nie ustalono",
            ("Porównanie bezpośrednie", "Porównanie rok do roku", "Nieporównywalne"),
        ),
    ),
)
def test_all_validation_comparison_enums_render_as_localized_labels(
    locale: str,
    causality: str,
    bases: tuple[str, ...],
) -> None:
    data = validation_report_data(locale)
    section = data.validation_comparison
    assert section is not None
    direct = section.metrics[0]
    year_over_year = direct.model_copy(update={"basis": ComparisonBasis.YEAR_OVER_YEAR})
    non_comparable = direct.model_copy(
        update={
            "state": DataState.UNKNOWN,
            "basis": ComparisonBasis.NON_COMPARABLE,
            "absolute_delta": None,
            "relative_delta_percent": None,
            "limitations": (ComparisonLimitation.WINDOW_MISMATCH,),
        }
    )
    expanded = section.model_copy(update={"metrics": (direct, year_over_year, non_comparable)})
    html = renderer_module().render_report_html(
        data.model_copy(update={"validation_comparison": expanded})
    )

    assert causality in html
    for label in bases:
        assert label in html
    for raw in (
        "NOT_ESTABLISHED",
        "DIRECT",
        "YEAR_OVER_YEAR",
        "NON_COMPARABLE",
    ):
        assert raw not in html


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


def test_template_version_has_explicit_packaged_directory_mapping() -> None:
    renderer = renderer_module()

    assert renderer.TEMPLATE_DIRECTORY_BY_VERSION[reports.REPORT_TEMPLATE_VERSION] == "v1"
    assert renderer.TEMPLATE_ROOT.name == "v1"


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


@pytest.mark.parametrize(
    ("locale", "expected", "forbidden"),
    (
        (
            "en",
            (
                "Project Context",
                "Measurement Evidence",
                "Context draft",
                "Approved",
                "Coverage 100%",
                "Confidence 90%",
            ),
            ("Kontekst projektu", "Wersja robocza z kontekstem"),
        ),
        (
            "pl",
            (
                "Kontekst projektu",
                "Dowody pomiarowe",
                "Wersja robocza z kontekstem",
                "Zatwierdzone",
                "Zakres 100%",
                "Pewność 90%",
            ),
            ("Project Context", "Context draft"),
        ),
    ),
)
def test_project_and_measurement_sections_use_localized_fixed_labels(
    locale: str,
    expected: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    renderer = renderer_module()
    data = _context_report().model_copy(update={"report_locale": locale})

    html = renderer.render_report_html(data)

    assert f'<html lang="{locale}">' in html
    for label in expected:
        assert label in html
    for label in forbidden:
        assert label not in html


@pytest.mark.parametrize(
    ("locale", "states"),
    (
        ("en", ("AVAILABLE", "UNAVAILABLE", "UNKNOWN", "FAILED")),
        ("pl", ("DOSTĘPNE", "NIEDOSTĘPNE", "NIEZNANE", "BŁĄD")),
    ),
)
def test_measurement_states_are_distinct_and_only_available_zero_is_numeric(
    locale: str,
    states: tuple[str, ...],
) -> None:
    renderer = renderer_module()
    base = _context_report().model_copy(update={"report_locale": locale})
    assert base.measurement is not None
    available = base.measurement.metrics[0]
    metrics = (available,) + tuple(
        ReportVisibilityMetric.model_validate(
            {
                **available.model_dump(mode="python"),
                "metric_id": f"metric-{state.value.casefold()}",
                "state": state,
                "value": None,
                "coverage": 0,
                "confidence": 0.4,
                "window": None,
                "points": (),
            }
        )
        for state in (DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED)
    )
    data = base.model_copy(
        update={
            "measurement": base.measurement.model_copy(update={"metrics": metrics}),
        }
    )

    html = renderer.render_report_html(data)

    for state in states:
        assert state in html
    assert "ga4.ai_assistant.sessions" in html
    assert "0 sessions" in html
    for state in states[1:]:
        state_position = html.index(state)
        assert "0 sessions" not in html[state_position : state_position + 300]


def test_context_pdf_is_deterministic_with_identical_canonical_inputs(tmp_path: Path) -> None:
    renderer = renderer_module()
    data = _context_report().model_copy(update={"renderer": renderer.build_renderer_metadata()})

    first = renderer.render_client_report(data, tmp_path / "context-first.pdf")
    second = renderer.render_client_report(data, tmp_path / "context-second.pdf")

    assert first.sha256 == second.sha256
    assert (tmp_path / "context-first.pdf").read_bytes() == (
        tmp_path / "context-second.pdf"
    ).read_bytes()


def test_direct_persisted_context_render_rejects_changed_canonical_section() -> None:
    payload = json.loads(_context_report().model_dump_json())
    payload["owner_context"]["facts"][0]["value"] = "tampered"

    with pytest.raises(ValidationError, match="context digest"):
        ClientReportData.model_validate(payload)


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_every_visibility_platform_has_a_localized_label(locale: str) -> None:
    renderer = renderer_module()
    translations = renderer._translations(locale)

    assert set(translations["platforms"]) == {source.value for source in VisibilitySource}
    assert all(translations["platforms"][source.value] for source in VisibilitySource)


@pytest.mark.parametrize(
    ("locale", "expected"),
    (
        (
            "en",
            {
                "UNAVAILABLE": "Unavailable",
                "UNKNOWN": "Unknown",
                "FAILED": "Failed",
                "UNSUPPORTED": "Unsupported",
                "AVAILABLE": "Available",
                "PARTIAL": "Partial",
            },
        ),
        (
            "pl",
            {
                "UNAVAILABLE": "Niedostępne",
                "UNKNOWN": "Nieznane",
                "FAILED": "Błąd przetwarzania",
                "UNSUPPORTED": "Nieobsługiwane",
                "AVAILABLE": "Dostępne",
                "PARTIAL": "Częściowe",
            },
        ),
    ),
)
def test_measurement_limitations_use_neutral_state_labels(
    locale: str,
    expected: dict[str, str],
) -> None:
    renderer = renderer_module()

    assert renderer._translations(locale)["measurement_state_limitations"] == expected


@pytest.mark.parametrize(
    ("locale", "platform_label"),
    (("en", "Google Analytics"), ("pl", "Google Analytics")),
)
def test_measurement_template_uses_localized_platform_and_separates_metadata(
    tmp_path: Path,
    locale: str,
    platform_label: str,
) -> None:
    renderer = renderer_module()
    data = _context_report().model_copy(
        update={
            "report_locale": locale,
            "renderer": renderer.build_renderer_metadata(),
        }
    )

    html = renderer.render_report_html(data)

    assert platform_label in html
    assert ">google_analytics<" not in html
    assert "</strong> channel=AI Assistant<br><strong>" in html
    output = tmp_path / f"context-{locale}.pdf"
    renderer.render_client_report(data, output)
    extracted = subprocess.run(
        ["pdftotext", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "AI AssistantDefinitions" not in extracted
    assert "AI AssistantDefinicje" not in extracted
