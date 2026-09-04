from __future__ import annotations

import json
import socket
from datetime import date, timedelta

import httpx
import pytest

from ai_search_audit import cli, project_orchestrator
from ai_search_audit.models import AuditRun
from ai_search_audit.project_models import AuditStage, ReportStatus
from ai_search_audit.project_store import ProjectStore
from ai_search_audit.prompt_context import PromptTopic
from ai_search_audit.prompts import validate_automatic_prompt_pack
from ai_search_audit.report_models import ClientReportData, ProjectReportMetadata
from tests.test_crawler import public_resolver, transport
from tests.test_project_orchestrator import NOW, _bundle_hashes, _create, _owner_intake
from tests.test_selected_prompt_topics import _selection, _transport


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("refresh tests must not use live network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _refresh(tmp_path, **overrides):
    options = dict(
        clients_root=tmp_path / "clients",
        source_version="public-v1",
        expected_domain="example.com",
        expected_client_name="Example Client",
        now=NOW + timedelta(days=1),
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
    )
    options.update(overrides)
    return project_orchestrator.refresh_project("project:example", **options)


@pytest.mark.parametrize("number, source", [(1, "older-audit"), (2, None)])
def test_public_report_metadata_enforces_initial_and_later_source_linkage(number, source):
    with pytest.raises(ValueError, match="source audit"):
        ProjectReportMetadata(
            project_id="example",
            version_id=f"public-v{number}",
            version_number=number,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            source_audit_id=source,
        )


@pytest.mark.parametrize("number, source", [(1, "older-audit"), (2, None)])
def test_public_bundle_validator_rejects_missing_or_initial_source_before_read(
    tmp_path, number, source
):
    with pytest.raises(ValueError, match="public.*source audit"):
        project_orchestrator.validate_project_bundle(
            tmp_path,
            expected_project_id="example",
            expected_version_number=number,
            expected_source_audit_id=source,
            expected_stage=AuditStage.PUBLIC,
        )


def test_refresh_creates_linked_fresh_public_v2_and_preserves_all_previous_bytes(tmp_path):
    initial = _create(tmp_path)
    root = tmp_path / "clients" / "example"
    old_root = root / initial.versions[0].relative_path
    # Additional project-owned runs and editions are not refresh outputs.
    for relative in ("diagnostics/run-1/result.json", "reports/edition-1/client-report.md"):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text("preserve exactly\n")
    before = _bundle_hashes(root)
    requests = []

    def fresh_transport(request):
        requests.append(str(request.url))
        response = transport(request)
        if request.url.path == "/":
            response = httpx.Response(
                200,
                text=response.text.replace("</body>", "Fresh observation</body>"),
                headers={"Content-Type": "text/html"},
                request=request,
            )
        return response

    updated = _refresh(tmp_path, crawler_transport=httpx.MockTransport(fresh_transport))
    version = updated.versions[-1]
    assert version.version_id == "public-v2"
    assert version.stage is AuditStage.PUBLIC
    assert version.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    assert version.source_audit_id == initial.latest_audit_id
    assert version.audit_id != initial.latest_audit_id
    assert updated.versions[:-1] == initial.versions
    assert updated.latest_audit_id == version.audit_id
    assert requests
    fresh_root = root / version.relative_path
    run = AuditRun.model_validate_json((fresh_root / "engine/audit.json").read_text())
    old = AuditRun.model_validate_json((old_root / "engine/audit.json").read_text())
    assert run.timestamp > old.timestamp
    assert any("Fresh observation" in page.content_text for page in run.pages)
    assert run.configuration["max_pages"] == old.configuration["max_pages"]
    assert run.configuration["source_audit_id"] == initial.latest_audit_id
    assert "implementation_date" not in run.configuration
    assert "baseline_audit_id" not in run.configuration
    assert list((fresh_root / "aggregates").iterdir()) == []
    report = ClientReportData.model_validate_json(
        (fresh_root / "engine/client-report-data.json").read_text()
    )
    assert report.project.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    assert report.project.source_audit_id == initial.latest_audit_id
    assert report.owner_context is None
    assert report.validation_comparison is None
    inputs = json.loads((fresh_root / "manifests/input-manifest.json").read_text())
    assert inputs["source_audit_id"] == initial.latest_audit_id
    assert inputs["raw_inputs"] == inputs["normalized_inputs"] == []
    after = _bundle_hashes(root)
    assert all(after[path] == digest for path, digest in before.items() if path != "project.json")
    project_orchestrator.validate_project_bundle(
        fresh_root,
        expected_project_id="example",
        expected_version_number=2,
        expected_source_audit_id=initial.latest_audit_id,
        expected_stage=AuditStage.PUBLIC,
    )
    from ai_search_audit.diagnostic_sources import load_diagnostic_source

    source = load_diagnostic_source(
        "project:example", clients_root=tmp_path / "clients", source_version="public-v2"
    )
    assert source.binding.audit_id == version.audit_id
    assert source.page_urls == tuple(str(page.url) for page in run.pages)


@pytest.mark.parametrize("source_version", ["public-v1", "validation-v2"])
def test_refresh_global_numbering_after_validation_and_explicit_source(
    tmp_path, monkeypatch, source_version
):
    _create(tmp_path)
    real_run = project_orchestrator.run_public_audit

    def local_run(*args, **kwargs):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_run)
    validated = project_orchestrator.validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=NOW + timedelta(days=15),
    )
    monkeypatch.setattr(project_orchestrator, "run_public_audit", real_run)
    root = tmp_path / "clients/example"
    before = _bundle_hashes(root)
    updated = _refresh(tmp_path, source_version=source_version, now=NOW + timedelta(days=16))
    latest = updated.versions[-1]
    assert latest.version_id == "public-v3"
    assert latest.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    source = next(v for v in validated.versions if v.version_id == source_version)
    assert latest.source_audit_id == source.audit_id
    after = _bundle_hashes(root)
    assert all(after[p] == digest for p, digest in before.items() if p != "project.json")


def test_refresh_from_client_validated_context_never_carries_owner_claims(tmp_path):
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)
    context = project_orchestrator.update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )
    assert context.versions[-1].report_status is ReportStatus.CLIENT_VALIDATED
    root = tmp_path / "clients/example"
    before = _bundle_hashes(root)
    updated = _refresh(tmp_path, source_version="context-v2")
    latest = updated.versions[-1]
    assert latest.version_id == "public-v3"
    assert latest.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    fresh_root = root / latest.relative_path
    run = AuditRun.model_validate_json((fresh_root / "engine/audit.json").read_text())
    report = ClientReportData.model_validate_json(
        (fresh_root / "engine/client-report-data.json").read_text()
    )
    assert report.owner_context is None
    assert report.measurement is None
    assert report.validation_comparison is None
    assert list((fresh_root / "aggregates").iterdir()) == []
    assert not any(item.source_type == "owner_fact" for item in run.evidence)
    after = _bundle_hashes(root)
    assert all(after[p] == digest for p, digest in before.items() if p != "project.json")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source_version": "public-v99"}, "source version"),
        ({"source_version": ""}, "source version"),
        ({"expected_domain": "wrong.example"}, "identity"),
        ({"expected_client_name": "Wrong Entity"}, "identity"),
        ({"expected_client_name": ""}, "identity"),
        ({"expected_client_name": None}, "identity"),
        ({"max_pages": 101}, "max_pages"),
    ],
)
def test_refresh_rejects_invalid_source_identity_and_bounds_before_network(
    tmp_path, overrides, message
):
    _create(tmp_path)
    root = tmp_path / "clients/example"
    before = _bundle_hashes(root)

    def forbidden(request):
        pytest.fail("invalid refresh must be rejected before network")

    with pytest.raises(ValueError, match=message):
        _refresh(tmp_path, crawler_transport=httpx.MockTransport(forbidden), **overrides)
    assert _bundle_hashes(root) == before
    assert not (root / ".staging").exists()


def test_refresh_revalidates_inherited_selected_topics_and_allows_fresh_selection(tmp_path):
    selection = PromptTopic(
        kind="category",
        locale="en",
        value="ceramics workshops",
        source_url="https://studio.example/",
        locator="content_text[29:47]",
        quote="ceramics workshops",
    )
    initial = _create(
        tmp_path,
        domain="studio.example",
        selected_topics=(selection,),
        crawler_transport=_transport(),
    )
    root = tmp_path / "clients/example"
    old_root = root / initial.versions[0].relative_path
    old = AuditRun.model_validate_json((old_root / "engine/audit.json").read_text())
    before = _bundle_hashes(root)
    with pytest.raises(ValueError, match="selection"):
        _refresh(
            tmp_path, expected_domain="studio.example", crawler_transport=_transport("Updated ")
        )
    assert _bundle_hashes(root) == before
    assert list((root / ".staging").iterdir()) == []
    fresh_page = old.pages[0].model_copy(
        update={"content_text": "Updated " + old.pages[0].content_text}
    )
    new_selection = _selection(fresh_page)
    updated = _refresh(
        tmp_path,
        expected_domain="studio.example",
        crawler_transport=_transport("Updated "),
        selected_topics=(new_selection,),
    )
    fresh_root = root / updated.versions[-1].relative_path
    fresh = AuditRun.model_validate_json((fresh_root / "engine/audit.json").read_text())
    assert fresh.configuration["prompt_pack_version"] == "2.1.0"
    assert fresh.configuration["prompt_topic_selection"] == [new_selection.model_dump(mode="json")]
    validate_automatic_prompt_pack(fresh)
    assert _bundle_hashes(old_root) == {
        p.removeprefix(initial.versions[0].relative_path + "/"): h
        for p, h in before.items()
        if p.startswith(initial.versions[0].relative_path + "/")
    }
    inherited = _refresh(
        tmp_path,
        expected_domain="studio.example",
        source_version="public-v2",
        crawler_transport=_transport("Updated "),
        now=NOW + timedelta(days=2),
    )
    inherited_run = AuditRun.model_validate_json(
        (root / inherited.versions[-1].relative_path / "engine/audit.json").read_text()
    )
    assert (
        inherited_run.configuration["prompt_topic_selection"]
        == fresh.configuration["prompt_topic_selection"]
    )


@pytest.mark.parametrize("failure", ["crawl", "promotion", "manifest"])
def test_refresh_failure_rolls_back_owned_staging_and_history(tmp_path, monkeypatch, failure):
    _create(tmp_path)
    root = tmp_path / "clients/example"
    before = _bundle_hashes(root)

    def fail(*args, **kwargs):
        raise RuntimeError("injected refresh failure")

    if failure == "crawl":
        monkeypatch.setattr(project_orchestrator, "run_public_audit", fail)
    elif failure == "promotion":
        monkeypatch.setattr(ProjectStore, "promote", fail)
    else:
        monkeypatch.setattr(ProjectStore, "_atomic_write_manifest", fail)
    with pytest.raises(RuntimeError, match="injected refresh failure"):
        _refresh(tmp_path)
    assert _bundle_hashes(root) == before
    assert list((root / ".staging").iterdir()) == []


def _cli_options():
    return {
        "clients-root": "/tmp/clients",
        "source-version": "public-v1",
        "expected-domain": "example.com",
        "expected-client-name": "Example Client",
    }


def test_refresh_cli_dispatches_explicit_identity_and_source(tmp_path, monkeypatch, capsys):
    initial = _create(tmp_path)
    real_refresh = project_orchestrator.refresh_project

    def local_refresh(*args, **kwargs):
        return real_refresh(
            *args,
            **kwargs,
            now=NOW + timedelta(days=1),
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(cli, "refresh_project", local_refresh)
    options = _cli_options() | {"clients-root": str(tmp_path / "clients"), "max-pages": "1"}
    argv = ["project", "refresh", "project:example"]
    for name, value in options.items():
        argv.extend([f"--{name}", value])
    assert cli.main(argv) == 0
    updated = ProjectStore(tmp_path / "clients").load("example")
    assert updated.versions[-1].source_audit_id == initial.latest_audit_id
    assert updated.versions[-1].version_id == "public-v2"
    output = capsys.readouterr()
    assert "public-v2" in output.out
    assert "Client delivery is pending" in output.err


@pytest.mark.parametrize("missing", _cli_options())
def test_refresh_cli_requires_source_and_identity(missing):
    argv = ["project", "refresh", "project:example"]
    for name, value in _cli_options().items():
        if name != missing:
            argv.extend([f"--{name}", value])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-pages", "0"],
        ["--max-pages", "101"],
        ["--implementation-date", "2026-09-01"],
        ["--intake-root", "/tmp/intake"],
    ],
)
def test_refresh_cli_rejects_unbounded_crawls_and_validation_owner_options(extra):
    argv = ["project", "refresh", "project:example"]
    for name, value in _cli_options().items():
        argv.extend([f"--{name}", value])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv + extra)
