"""Real renderer/finalizer regression, with fabricated source-bound evidence only."""

import hashlib
import inspect
import json
import subprocess

import pytest
from pypdf import PdfReader

from ai_search_audit.client_delivery import finalize_client_report
from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.measurement_report import load_measurement_report, render_measurement_fragment
from ai_search_audit.observation_report import load_observation_report, render_observation_fragment
from tests.test_client_delivery import digest, finalize, prepared
from tests.test_client_pdf_script import render_edition
from tests.test_diagnostic_versions import performance_run
from tests.test_diagnostic_workflow import _contract, _hashes, no_network
from tests.test_observation_report import report_for
from tests.test_openai_observations import message, payload
from tests.test_project_orchestrator import _create

__all__ = ["no_network", "prepared"]


def test_provider_markdown_renders_as_readable_inert_quoted_text(tmp_path):
    quoted = (
        "**Example Studio** may offer software; it is not guaranteed. "
        "([studio.example](https://studio.example/)) "
        "@@TOKEN0@@ | column | <b>not markup</b> `literal` [other](https://other.example/) "
        r'literal \u002a &#42; <link href="https://injected.example/">no</link> x**2 ** `'
    )
    data = payload()
    data["output"][1] = message(
        quoted,
        [
            {
                "type": "url_citation",
                "start_index": quoted.index("([studio.example]"),
                "end_index": quoted.index(" @@TOKEN"),
                "url": "https://studio.example/",
            }
        ],
    )
    report = report_for(data=data)
    canonical = report.model_dump_json()
    fragment = render_observation_fragment(report, projection_version="1.1.0")
    _, markdown, pdf, _, command = render_edition(tmp_path, hero=False)
    command += ["--observation-projection-version", "1.1.0"]
    markdown.write_text("## Observations\n\n" + fragment)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    reader = PdfReader(pdf)
    text = " ".join(" ".join(page.extract_text() for page in reader.pages).split())
    expected = (
        quoted.replace("**Example Studio**", "Example Studio")
        .replace("([studio.example](https://studio.example/))", "studio.example")
        .replace("`literal`", "literal")
    )
    assert expected in text
    assert "**Example Studio**" not in text and "([studio.example]" not in text
    assert text.count(r"\u002a") == 2 and r"\u005b" not in text
    uris = [
        str(annotation.get_object()["/A"]["/URI"])
        for page in reader.pages
        for annotation in page.get("/Annots", [])
        if "/A" in annotation.get_object() and "/URI" in annotation.get_object()["/A"]
    ]
    # Each excerpt has an inline and a source-list link, never provider prose links.
    assert uris == ["https://studio.example/"] * 4
    assert report.model_dump_json() == canonical
    assert render_observation_fragment(report, projection_version="1.1.0") == fragment


def require_plural():
    assert "diagnostic_run_refs" in inspect.signature(finalize_client_report).parameters, (
        "versioned multi-run delivery missing"
    )


def publish(root, *, performance=True, projection_version="1.0.0"):
    refs, fragments = [], []
    if performance:
        path = DiagnosticStore(root).publish(performance_run(root))
        ref = "public-v1/" + path.name
        refs.append(ref)
        fragments.append(
            render_measurement_fragment(
                load_measurement_report(
                    "project:example", clients_root=root.parent, diagnostic_run_ref=ref
                )
            )
        )
    data = payload()
    data["output"][1]["content"][0]["annotations"][0]["url"] = "https://studio.example/a_(safe)"
    run = report_for(source=_contract(root).source, data=data).run
    path = DiagnosticStore(root).publish(run)
    ref = "public-v1/" + path.name
    refs.append(ref)
    fragments.append(
        render_observation_fragment(
            load_observation_report(
                "project:example", clients_root=root.parent, diagnostic_run_ref=ref
            ),
            projection_version=projection_version,
        )
    )
    return tuple(refs), tuple(fragments)


def test_literal_projection_finalization_requires_matching_explicit_version(tmp_path):
    manifest = _create(
        tmp_path, domain="https://studio.example", client_name="Example Studio", report_locale="en"
    )
    root = tmp_path / "clients/example"
    refs, fragments = publish(root, performance=False, projection_version="1.1.0")
    before = _hashes(root)
    _, md, pdf, _, command = render_edition(tmp_path, hero=False)
    md.write_text("## Evidence\n\n" + "\n".join(fragments))
    command += ["--audit-id", manifest.latest_audit_id]
    subprocess.run(command, check=True, capture_output=True)
    args = dict(
        project_ref="project:example",
        clients_root=root.parent,
        version_id="public-v1",
        markdown_path=md,
        pdf_path=pdf,
        reviewed_pdf_sha256=digest(pdf),
        no_hero_reason="No suitable controlled image",
        diagnostic_run_refs=refs,
    )
    with pytest.raises(ValueError, match="exact observation fragment"):
        finalize(args)
    with pytest.raises(ValueError, match="renderer observation projection"):
        finalize({**args, "observation_projection_version": "1.1.0"})
    command += ["--observation-projection-version", "1.1.0"]
    subprocess.run(command, check=True, capture_output=True)
    args["reviewed_pdf_sha256"] = digest(pdf)
    destination = finalize({**args, "observation_projection_version": "1.1.0"})
    record = json.loads((destination / "delivery.json").read_text())
    assert (
        record["diagnostic_runs"][0]["client_fragment_sha256"]
        == hashlib.sha256(fragments[0].encode()).hexdigest()
    )
    assert all(_hashes(root)[name] == digest for name, digest in before.items())


@pytest.mark.parametrize("locale,performance,hero", [("en", True, True), ("pl", False, False)])
def test_multirun_finalization_preserves_template_citations_and_prior_editions(
    tmp_path, locale, performance, hero
):
    require_plural()
    manifest = _create(
        tmp_path,
        domain="https://studio.example",
        client_name="Example Studio",
        report_locale=locale,
    )
    root = tmp_path / "clients/example"
    _, md, pdf, image, render = render_edition(tmp_path, locale=locale, hero=hero)
    render += ["--audit-id", manifest.latest_audit_id]
    subprocess.run(render, check=True, capture_output=True)
    args = dict(
        project_ref="project:example",
        clients_root=root.parent,
        version_id="public-v1",
        markdown_path=md,
        pdf_path=pdf,
        reviewed_pdf_sha256=digest(pdf),
    )
    args.update(
        dict(hero_path=image, hero_source="https://studio.example/hero.png")
        if hero
        else dict(no_hero_reason="No suitable controlled image")
    )
    old = finalize(args)
    old_hashes = _hashes(old)
    refs, fragments = publish(root, performance=performance)
    before = _hashes(root)
    md.write_text(md.read_text() + "\n## Evidence\n\n" + "\n".join(fragments))
    result = subprocess.run(render, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rendered = pdf.read_bytes()
    subprocess.run(render, check=True, capture_output=True)
    assert pdf.read_bytes() == rendered
    args.update(reviewed_pdf_sha256=digest(pdf), diagnostic_run_refs=tuple(reversed(refs)))
    destination = finalize(args)
    record = json.loads((destination / "delivery.json").read_text())
    assert record["schema_version"] == "2.0.0" and "diagnostic_run" not in record
    assert [item["path"] for item in record["diagnostic_runs"]] == [
        "diagnostics/" + r for r in refs
    ]
    for binding, ref, fragment in zip(record["diagnostic_runs"], refs, fragments, strict=True):
        assert binding["client_fragment_sha256"] == hashlib.sha256(fragment.encode()).hexdigest()
        assert binding["manifest_sha256"] == digest(root / "diagnostics" / ref / "manifest.json")
    assert all(_hashes(root)[name] == value for name, value in before.items())
    assert _hashes(old) == old_hashes
    assert json.loads((old / "delivery.json").read_text())["schema_version"] == "1.0.0"
    reader = PdfReader(destination / "client-report.pdf")
    assert reader.metadata["/ClientEditionTemplate"] == "editorial-v1"
    assert reader.metadata["/ClientEditionAuditID"] == manifest.latest_audit_id
    assert bool(reader.pages[0].images) == hero
    text = " ".join(page.extract_text() for page in reader.pages)
    assert "OpenAI API web search" in text and "Visit studio.example" in text
    assert "req_fixture" not in text and "audit-measurement:" not in text
    uris = [
        str(annotation.get_object()["/A"]["/URI"])
        for page in reader.pages
        for annotation in page.get("/Annots", [])
        if "/A" in annotation.get_object() and "/URI" in annotation.get_object()["/A"]
    ]
    assert "https://studio.example/a_%28safe%29" in uris
    if hero:
        assert (destination / "cover.png").read_bytes() == image.read_bytes()


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "mixed",
        "empty",
        "twoperformance",
        "twoobservations",
        "missing-fragment",
        "duplicate-fragment",
        "substitution",
        "stale",
    ],
)
def test_plural_binding_rejects_ambiguous_or_unconsumed_evidence(prepared, failure):
    require_plural()
    root, _, args = prepared
    refs, fragments = publish(root)
    original = args["markdown_path"].read_text()
    kwargs = {**args, "diagnostic_run_refs": refs}
    if failure == "duplicate":
        kwargs["diagnostic_run_refs"] = (refs[0], refs[0])
    elif failure == "mixed":
        kwargs["diagnostic_run_ref"] = refs[0]
    elif failure == "empty":
        kwargs["diagnostic_run_refs"] = ()
    elif failure in {"twoperformance", "twoobservations", "substitution"}:
        run = (
            performance_run(root)
            if failure == "twoperformance"
            else report_for(source=_contract(root).source).run
        )
        path = DiagnosticStore(root).publish(run)
        kwargs["diagnostic_run_refs"] = (
            refs[0] if failure != "twoobservations" else refs[1],
            "public-v1/" + path.name,
        )
    addition = "\n".join(fragments)
    if failure == "missing-fragment":
        addition = fragments[0]
    elif failure == "duplicate-fragment":
        addition += fragments[1]
    elif failure == "stale":
        addition = addition.replace("Visit studio.example", "Different studio.example")
    args["markdown_path"].write_text(original + "\n## Evidence\n\n" + addition)
    with pytest.raises(ValueError):
        finalize(kwargs)
    assert not (root / "reports").exists()


def test_plural_rejects_performance_from_previous_source(prepared, tmp_path):
    require_plural()
    from ai_search_audit.project_orchestrator import update_project_context
    from tests.test_project_orchestrator import NOW, _owner_intake

    root, _, args = prepared
    refs, _ = publish(root)
    owned, intake = _owner_intake(tmp_path, entity="Example Studio")
    manifest = update_project_context(
        "project:example",
        clients_root=root.parent,
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )
    with pytest.raises(ValueError, match="selected client edition"):
        finalize(
            {**args, "version_id": manifest.versions[-1].version_id, "diagnostic_run_refs": refs}
        )
    assert not (root / "reports").exists()


def test_cli_plural_boundary_is_explicit_and_preserves_legacy_arg():
    from ai_search_audit.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "project",
            "finalize",
            "project:example",
            "--clients-root",
            "/tmp",
            "--version-id",
            "public-v1",
            "--markdown",
            "/tmp/source.md",
            "--pdf",
            "/tmp/report.pdf",
            "--reviewed-pdf-sha256",
            "a" * 64,
            "--no-hero-reason",
            "fixture",
            "--diagnostic-runs",
            "public-v1/run-1",
            "public-v1/run-2",
        ]
    )
    assert args.diagnostic_runs == ["public-v1/run-1", "public-v1/run-2"]
    assert args.diagnostic_run is None
