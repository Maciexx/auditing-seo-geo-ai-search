"""Reviewed editorial content consumes the source-bound measurement projection."""

import hashlib
import json
import subprocess
from importlib import import_module, util

import pytest
from pypdf import PdfReader

from ai_search_audit.cli import main
from ai_search_audit.diagnostic_store import DiagnosticStore
from tests.test_client_delivery import prepared
from tests.test_diagnostic_versions import performance_run
from tests.test_diagnostic_workflow import _hashes, no_network, project
from tests.test_measurement_workflow import command, harness
from tests.test_project_orchestrator import _create

__all__ = ["no_network", "project", "prepared"]


def report_command(root):
    return [
        "project",
        "measurement-report",
        "project:example",
        "--clients-root",
        str(root.parent),
        "--diagnostic-run",
        "public-v1/run-1",
    ]


def projection(root):
    assert util.find_spec("ai_search_audit.measurement_report"), "measurement report missing"
    module = import_module("ai_search_audit.measurement_report")
    dto = module.load_measurement_report(
        "project:example", clients_root=root.parent, diagnostic_run_ref="public-v1/run-1"
    )
    return dto, module.render_measurement_fragment(dto)


@pytest.mark.parametrize("locale", ["pl", "en"])
def test_measure_to_editorial_finalization_real_flow(tmp_path, monkeypatch, capsys, locale):
    from tests.test_client_delivery import digest, finalize
    from tests.test_client_pdf_script import render_edition

    manifest = _create(
        tmp_path,
        domain="https://studio.example",
        client_name="Example Studio",
        report_locale=locale,
    )
    root = tmp_path / "clients/example"

    def measured_values(payload):
        lab = payload["lighthouseResult"]
        lab["categories"]["performance"]["score"] = 0.73
        lab["audits"]["largest-contentful-paint"]["numericValue"] = 4651

    harness(monkeypatch, capsys, crux_origin=True, psi_mutation=measured_values)
    assert main(command(root, {"max_pages": 1})) == 0
    capsys.readouterr()
    before = _hashes(root)
    assert main(report_command(root)) == 0
    fragment = capsys.readouterr().out
    dto, expected = projection(root)
    assert fragment == expected
    assert before == _hashes(root)
    assert dto.run.binding.report_locale == locale
    assert dto.audited_page_count >= len(dto.run.preflight.selected_pages)
    assert "2026-08-01" in fragment and "2026-08-28" in fragment
    assert "2026-09-03" in fragment
    assert "origin" in fragment.lower() and "p75" in fragment
    assert "TBT" in fragment and "INP" in fragment
    assert "PageSpeed Insights" in fragment
    assert "73/100" in fragment
    assert ("4,65 s" if locale == "pl" else "4.65 s") in fragment
    assert ("Brak danych" if locale == "pl" else "Missing") in fragment
    assert ("Nie jest to pełny skan" if locale == "pl" else "Not a full site scan") in fragment
    assert all(len(line.split("|")) <= 7 for line in fragment.splitlines() if line.startswith("|"))
    result, md, pdf, _, render = render_edition(tmp_path, locale=locale, hero=False)
    assert result.returncode == 0
    md.write_text(
        md.read_text()
        + "\n## "
        + ("Technika" if locale == "pl" else "Technical")
        + "\n\n"
        + fragment
    )
    rendered = subprocess.run(
        render + ["--audit-id", manifest.latest_audit_id],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rendered.returncode == 0, rendered.stderr
    destination = finalize(
        dict(
            project_ref="project:example",
            clients_root=root.parent,
            version_id="public-v1",
            markdown_path=md,
            pdf_path=pdf,
            no_hero_reason="No suitable controlled image",
            reviewed_pdf_sha256=digest(pdf),
            diagnostic_run_ref="public-v1/run-1",
        )
    )
    delivery = json.loads((destination / "delivery.json").read_text())
    assert (
        delivery["diagnostic_run"]["measurement_fragment_sha256"]
        == hashlib.sha256(fragment.encode()).hexdigest()
    )
    assert delivery["source_audit_sha256"] == dto.run.binding.source_sha256
    text = " ".join(
        page.extract_text() for page in PdfReader(destination / "client-report.pdf").pages
    )
    assert "PageSpeed Insights" in text and "CrUX" in text
    assert ("1,23 s" if locale == "pl" else "1.23 s") in text and "INP" in text
    assert "73/100" in text and ("4,65 s" if locale == "pl" else "4.65 s") in text
    assert "audit-measurement:" not in text and "diagnostics/" not in text
    assert all(_hashes(root)[name] == sha for name, sha in before.items())


def test_finalizer_requires_consumed_v2_fragment_before_review(prepared):
    from tests.test_client_delivery import finalize

    root, _, args = prepared
    DiagnosticStore(root).publish(performance_run(root))
    with pytest.raises(ValueError, match="measurement fragment"):
        finalize({**args, "diagnostic_run_ref": "public-v1/run-1"})
    assert not (root / "reports").exists()


def test_fragment_rejects_stale_pdf_and_wrong_or_duplicate_projection(prepared):
    from tests.test_client_delivery import finalize

    root, _, args = prepared
    DiagnosticStore(root).publish(performance_run(root))
    _, fragment = projection(root)
    original = args["markdown_path"].read_text()
    changed = fragment.replace("2026-09-04", "2026-09-05")
    assert changed != fragment
    for addition in (changed, fragment + fragment, fragment):
        args["markdown_path"].write_text(original + "\n" + addition)
        with pytest.raises(ValueError):
            finalize({**args, "diagnostic_run_ref": "public-v1/run-1"})
    assert not (root / "reports").exists()


def test_report_dto_is_immutable_and_no_v1_reinterpretation(project, capsys):
    from tests.test_diagnostic_workflow import _run

    _run(project)
    assert main(report_command(project)) == 2
    assert "measurement" in capsys.readouterr().err.lower()
    DiagnosticStore(project).publish(performance_run(project))
    module = import_module("ai_search_audit.measurement_report")
    dto = module.load_measurement_report(
        "project:example", clients_root=project.parent, diagnostic_run_ref="public-v1/run-2"
    )
    with pytest.raises(ValueError):
        dto.run.preflight.profile.max_pages = 5
    with pytest.raises(ValueError):
        module.render_measurement_fragment(dto.model_copy(update={"raw_response": "x"}))


def test_client_fragment_is_compact_and_keeps_operator_provenance_in_dto(
    project, monkeypatch, capsys
):
    harness(monkeypatch, capsys)
    assert main(command(project, {"max_pages": 1})) == 0
    dto, fragment = projection(project)
    assert "U1" not in fragment
    assert "Homepage" in fragment
    assert "attempts" in dto.run.model_dump_json()
    assert dto.run.collections[0].attempts[0].attempt_id not in fragment
    assert "form_factor=" not in fragment
    assert len(fragment) < 5000
    assert "partial" in fragment.lower()


def test_explicit_technical_cli_preserves_legacy_renderer(project, capsys):
    DiagnosticStore(project).publish(performance_run(project))
    dto, fragment = projection(project)
    module = import_module("ai_search_audit.measurement_report")
    assert hasattr(module, "render_measurement_technical_fragment")
    technical = module.render_measurement_technical_fragment(dto)
    assert "U1" in technical and "public-v1/run-1" in technical
    assert "diagnostics/public-v1/run-1/evidence.jsonl" in technical
    assert "Numeric comparison unavailable" in technical
    assert fragment != technical
    assert main(report_command(project) + ["--view", "technical"]) == 0
    assert capsys.readouterr().out == technical


def test_profile_one_technical_bytes_replay_the_pre_client_projection_golden(project):
    from ai_search_audit.diagnostic_workflow import assemble_performance_run
    from ai_search_audit.measurement_profile import prepare_measurement_preflight
    from ai_search_audit.measurement_report import render_measurement_technical_fragment
    from tests.test_diagnostic_workflow import _contract
    from tests.test_measurement_client import report_for

    report = report_for(project)
    source = _contract(project).source
    legacy_profile = report.run.preflight.profile.model_copy(update={"schema_version": "1.0.0"})
    legacy = assemble_performance_run(
        source, prepare_measurement_preflight(source, legacy_profile), report.run.collections
    )
    report = report.model_copy(update={"run": legacy})
    text = render_measurement_technical_fragment(report)
    # Golden captured from the original renderer at 91b5a67, before client projection.
    # The source audit ID is allocated by project creation; normalize only that identity.
    replay = text.replace(report.run.binding.audit_id, "AUDIT-ID").encode()
    assert hashlib.sha256(replay).hexdigest() == (
        "2728401434f3f7c8be9f049ddef587a0c05d16ee9a68043007efcbb3ffa8cf31"
    )


def test_fragment_before_first_editorial_chapter_is_not_consumed(project):
    DiagnosticStore(project).publish(performance_run(project))
    _, fragment = projection(project)
    module = import_module("ai_search_audit.measurement_report")
    with pytest.raises(ValueError, match="chapter"):
        module.require_measurement_fragment(fragment + "\n## Technical\n", fragment)


def test_fragment_cannot_finalize_without_explicit_run(prepared):
    from tests.test_client_delivery import finalize

    root, _, args = prepared
    DiagnosticStore(root).publish(performance_run(root))
    _, fragment = projection(root)
    args["markdown_path"].write_text(args["markdown_path"].read_text() + "\n" + fragment)
    with pytest.raises(ValueError, match="explicit diagnostic run"):
        finalize(args)


def test_compact_numbers_round_without_turning_small_values_into_zero():
    module = import_module("ai_search_audit.measurement_report")
    assert module._number(1234.567890123, "n/a") == "1234.57"
    assert float(module._number(0.000000123456789, "n/a")) > 0
    assert module._number(None, "n/a") == "n/a"
    assert module._number(0.0, "n/a") == "0"


def test_unchecked_report_does_not_echo_nested_values_in_warnings(project):
    import warnings

    DiagnosticStore(project).publish(performance_run(project))
    dto, _ = projection(project)
    malformed = dto.model_copy(update={"run": {"secret": "MUST-NOT-PRINT"}})
    module = import_module("ai_search_audit.measurement_report")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            module.render_measurement_fragment(malformed)
    assert not captured
