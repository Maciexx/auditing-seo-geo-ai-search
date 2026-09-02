import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from ai_search_audit import cli
from ai_search_audit.cli import build_parser
from ai_search_audit.data_intake import (
    FactApprovalState,
    NormalizedIntake,
    OwnerFactInput,
    SourceArtifactDeclaration,
    VisibilitySource,
    create_owned_intake_dir,
)
from ai_search_audit.project_models import (
    AuditStage,
    AuditVersionRef,
    ProjectManifest,
    ReportStatus,
)
from ai_search_audit.project_orchestrator import create_project_audit
from tests.test_crawler import public_resolver, transport

ROOT = Path(__file__).parents[1]


def test_cli_supports_audit_command() -> None:
    args = build_parser().parse_args(["audit", "example.com", "--output-dir", "out"])
    assert args.command == "audit"
    assert args.domain == "example.com"
    assert str(args.output_dir) == "out"


def test_cli_supports_documented_output_alias() -> None:
    parser = build_parser()
    audit_parser = parser._subparsers._group_actions[0].choices["audit"]
    output_action = next(action for action in audit_parser._actions if action.dest == "output_dir")
    assert {"--output", "--output-dir"} <= set(output_action.option_strings)
    args = parser.parse_args(
        ["audit", "https://lakeside-hotel.example", "--output", "audit-output/sample"]
    )
    assert str(args.output_dir) == "audit-output/sample"


def test_cli_supports_polish_report_locale() -> None:
    args = build_parser().parse_args(["audit", "example.com", "--report-locale", "pl"])

    assert args.report_locale == "pl"


def test_cli_parses_nested_project_create_with_required_identity() -> None:
    args = build_parser().parse_args(
        [
            "project",
            "create",
            "https://example.com",
            "--clients-root",
            "/tmp/clients",
            "--project-id",
            "example",
            "--client-name",
            "Example",
            "--report-locale",
            "en",
            "--max-pages",
            "12",
        ]
    )

    assert args.command == "project"
    assert args.project_command == "create"
    assert args.clients_root == Path("/tmp/clients")
    assert args.project_id == "example"
    assert args.client_name == "Example"
    assert args.report_locale == "en"
    assert args.max_pages == 12


@pytest.mark.parametrize(
    "missing",
    ("clients-root", "project-id", "client-name", "report-locale", "max-pages"),
)
def test_project_create_requires_explicit_identity_arguments(missing: str) -> None:
    values = {
        "clients-root": ["--clients-root", "/tmp/clients"],
        "project-id": ["--project-id", "example"],
        "client-name": ["--client-name", "Example"],
        "report-locale": ["--report-locale", "en"],
        "max-pages": ["--max-pages", "50"],
    }
    argv = ["project", "create", "https://example.com"]
    for name, option in values.items():
        if name != missing:
            argv.extend(option)

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


@pytest.mark.parametrize("max_pages", ("0", "101"))
def test_project_create_rejects_out_of_bounds_max_pages(max_pages: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "project",
                "create",
                "https://example.com",
                "--clients-root",
                "/tmp/clients",
                "--project-id",
                "example",
                "--client-name",
                "Example",
                "--report-locale",
                "en",
                "--max-pages",
                max_pages,
            ]
        )


def test_project_create_main_forwards_arguments_and_prints_only_stable_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clients_root = tmp_path / "clients"
    now = datetime(2026, 8, 31, tzinfo=UTC)
    version = AuditVersionRef(
        version_id="public-v1",
        version_number=1,
        stage=AuditStage.PUBLIC,
        report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        audit_id="audit-example",
        source_audit_id=None,
        created_at=now,
        relative_path="audits/2026-08-31_public-v1",
    )
    manifest = ProjectManifest(
        project_id="example",
        client_name="Example",
        canonical_domains=("example.com",),
        report_locale="en",
        created_at=now,
        latest_audit_id=version.audit_id,
        source_files_policy="delete-after-processing",
        versions=(version,),
    )
    received: dict[str, object] = {}

    def fake_create(domain: str, **kwargs: object) -> ProjectManifest:
        received.update(domain=domain, **kwargs)
        return manifest

    monkeypatch.setattr(cli, "create_project_audit", fake_create)

    result = cli.main(
        [
            "project",
            "create",
            "https://example.com",
            "--clients-root",
            str(clients_root),
            "--project-id",
            "example",
            "--client-name",
            "Example",
            "--report-locale",
            "en",
            "--max-pages",
            "7",
        ]
    )

    assert result == 0
    assert received == {
        "domain": "https://example.com",
        "clients_root": clients_root,
        "project_id": "example",
        "client_name": "Example",
        "report_locale": "en",
        "max_pages": 7,
    }
    output = capsys.readouterr().out.strip()
    assert output == str(clients_root / "example" / version.relative_path)
    assert ".staging" not in output


def test_cli_parses_project_context_update_with_explicit_project_reference() -> None:
    args = build_parser().parse_args(
        [
            "project",
            "update",
            "project:example",
            "--clients-root",
            "/tmp/clients",
            "--intake-dir",
            "/tmp/intake/run-1",
            "--normalized-intake",
            "/tmp/intake/run-1/normalized-intake.json",
        ]
    )

    assert args.project_command == "update"
    assert args.project_ref == "project:example"
    assert args.clients_root == Path("/tmp/clients")
    assert args.intake_dir == Path("/tmp/intake/run-1")
    assert args.normalized_intake == Path("/tmp/intake/run-1/normalized-intake.json")


def test_cli_parses_project_visibility_enrichment_with_explicit_project_reference() -> None:
    args = build_parser().parse_args(
        [
            "project",
            "enrich",
            "project:example",
            "--clients-root",
            "/tmp/clients",
            "--intake-dir",
            "/tmp/intake/run-1",
            "--normalized-intake",
            "/tmp/intake/run-1/normalized-intake.json",
        ]
    )

    assert args.project_command == "enrich"
    assert args.project_ref == "project:example"
    assert args.clients_root == Path("/tmp/clients")
    assert args.intake_dir == Path("/tmp/intake/run-1")
    assert args.normalized_intake == Path("/tmp/intake/run-1/normalized-intake.json")


def test_project_enrich_main_uses_existing_normalized_visibility_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients_root = tmp_path / "clients"
    intake_dir = create_owned_intake_dir(tmp_path / "intake")
    normalized_path = intake_dir / "normalized-intake.json"
    normalized_path.write_text('{"project_id":"example"}', encoding="utf-8")
    now = datetime(2026, 9, 1, tzinfo=UTC)
    public = AuditVersionRef(
        version_id="public-v1",
        version_number=1,
        stage=AuditStage.PUBLIC,
        report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_public-v1",
    )
    context = AuditVersionRef(
        version_id="context-v2",
        version_number=2,
        stage=AuditStage.CONTEXT,
        report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
        audit_id="audit-context",
        source_audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_context-v2",
    )
    manifest = ProjectManifest(
        project_id="example",
        client_name="Example",
        canonical_domains=("example.com",),
        report_locale="en",
        created_at=now,
        latest_audit_id=context.audit_id,
        source_files_policy="delete-after-processing",
        versions=(public, context),
    )
    received: dict[str, object] = {}

    def fake_enrich(project_ref: str, **kwargs: object) -> ProjectManifest:
        received.update(project_ref=project_ref, **kwargs)
        return manifest

    monkeypatch.setattr(cli, "enrich_project", fake_enrich)

    assert (
        cli.main(
            [
                "project",
                "enrich",
                "project:example",
                "--clients-root",
                str(clients_root),
                "--intake-dir",
                str(intake_dir),
                "--normalized-intake",
                str(normalized_path),
            ]
        )
        == 0
    )

    assert received == {
        "project_ref": "project:example",
        "clients_root": clients_root,
        "intake_dir": intake_dir,
        "normalized_intake": {"project_id": "example"},
    }
    assert capsys.readouterr().out.strip() == str(clients_root / "example" / context.relative_path)


def test_project_update_requires_explicit_project_reference() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "project",
                "update",
                "--clients-root",
                "/tmp/clients",
                "--intake-dir",
                "/tmp/intake/run-1",
                "--normalized-intake",
                "/tmp/intake/run-1/normalized-intake.json",
            ]
        )


def test_project_update_main_passes_unvalidated_mapping_inside_deletion_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    intake_dir = create_owned_intake_dir(tmp_path / "intake")
    normalized_path = intake_dir / "normalized-intake.json"
    payload = {"report_status": "CLIENT_VALIDATED", "not": "yet validated"}
    normalized_path.write_text(json.dumps(payload), encoding="utf-8")
    now = datetime(2026, 9, 1, tzinfo=UTC)
    public = AuditVersionRef(
        version_id="public-v1",
        version_number=1,
        stage=AuditStage.PUBLIC,
        report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_public-v1",
    )
    context = AuditVersionRef(
        version_id="context-v2",
        version_number=2,
        stage=AuditStage.CONTEXT,
        report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
        audit_id="audit-context",
        source_audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_context-v2",
    )
    manifest = ProjectManifest(
        project_id="example",
        client_name="Example",
        canonical_domains=("example.com",),
        report_locale="en",
        created_at=now,
        latest_audit_id=context.audit_id,
        source_files_policy="delete-after-processing",
        versions=(public, context),
    )
    received: dict[str, object] = {}

    def fake_update(project_ref: str, **kwargs: object) -> ProjectManifest:
        received.update(project_ref=project_ref, **kwargs)
        return manifest

    monkeypatch.setattr(cli, "update_project_context", fake_update)

    assert (
        cli.main(
            [
                "project",
                "update",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--intake-dir",
                str(intake_dir),
                "--normalized-intake",
                str(normalized_path),
            ]
        )
        == 0
    )

    assert received == {
        "project_ref": "project:example",
        "clients_root": tmp_path / "clients",
        "intake_dir": intake_dir,
        "normalized_intake": payload,
    }
    assert capsys.readouterr().out.strip() == str(
        tmp_path / "clients" / "example" / context.relative_path
    )


def test_project_update_deletes_owned_intake_when_json_syntax_is_invalid(
    tmp_path: Path,
) -> None:
    intake_root = tmp_path / "intake"
    owned = create_owned_intake_dir(intake_root)
    normalized_path = owned / "normalized-intake.json"
    normalized_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        cli.main(
            [
                "project",
                "update",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--intake-dir",
                str(owned),
                "--normalized-intake",
                str(normalized_path),
            ]
        )

    assert not owned.exists()


def _start_coordinated_project_cli(
    tmp_path: Path,
    project_command: str,
) -> tuple[subprocess.Popen[str], Path]:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    argv = [
        sys.executable,
        "-m",
        "ai_search_audit",
        "project",
        project_command,
        "project:example",
        "--clients-root",
        str(tmp_path / "clients"),
        "--intake-root",
        str(tmp_path / "intake"),
    ]
    if project_command == "validate":
        argv.extend(("--implementation-date", "2026-09-01"))
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    assert line.startswith("OWNED_INTAKE_DIR="), (
        process.communicate(timeout=5)[1] if process.poll() is not None else line
    )
    return process, Path(line.removeprefix("OWNED_INTAKE_DIR="))


@pytest.mark.parametrize("project_command", ("update", "enrich", "validate"))
def test_coordinated_project_cli_deletes_owned_intake_when_normalization_fails(
    tmp_path: Path,
    project_command: str,
) -> None:
    process, owned = _start_coordinated_project_cli(tmp_path, project_command)
    (owned / "normalized-intake.json").write_text("{invalid", encoding="utf-8")
    stdout, stderr = process.communicate("READY\n", timeout=10)

    assert process.returncode != 0
    assert stdout == ""
    assert "JSONDecodeError" in stderr
    assert "IntakeOwnershipError" not in stderr
    assert not owned.exists()


def test_coordinated_project_cli_deletes_owned_intake_when_handshake_output_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake_root = tmp_path / "intake"

    def fail_handshake(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError("consumer closed output")

    monkeypatch.setattr("builtins.print", fail_handshake)

    with pytest.raises(BrokenPipeError, match="consumer closed output"):
        cli.main(
            [
                "project",
                "update",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--intake-root",
                str(intake_root),
            ]
        )

    assert intake_root.is_dir()
    assert list(intake_root.iterdir()) == []


def test_coordinated_project_cli_consumes_valid_intake_before_project_failure(
    tmp_path: Path,
) -> None:
    process, owned = _start_coordinated_project_cli(tmp_path, "update")
    content = b'{"preferred_brand_name":"Example"}'
    (owned / "owner.json").write_bytes(content)
    source = SourceArtifactDeclaration(
        source_id="owner-source",
        filename="owner.json",
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        platform=VisibilitySource.MANUAL,
        report_type="owner-context",
    )
    normalized = NormalizedIntake(
        project_id="example",
        canonical_domain="example.com",
        sources=(source,),
        owner_facts=(
            OwnerFactInput(
                fact_id="canonical-entity",
                field="canonical_entity",
                value="Example",
                source_id=source.source_id,
                approval_state=FactApprovalState.APPROVED,
            ),
        ),
    )
    (owned / "normalized-intake.json").write_text(
        normalized.model_dump_json(),
        encoding="utf-8",
    )
    stdout, stderr = process.communicate("READY\n", timeout=10)

    assert process.returncode != 0
    assert stdout == ""
    assert "FileNotFoundError" in stderr
    assert "IntakeOwnershipError" not in stderr
    assert not owned.exists()


def test_coordinated_project_cli_successfully_promotes_context_version(
    tmp_path: Path,
) -> None:
    clients_root = tmp_path / "clients"
    create_project_audit(
        domain="https://example.com",
        clients_root=clients_root,
        project_id="example",
        client_name="Example",
        report_locale="en",
        max_pages=2,
        now=datetime(2026, 8, 31, 10, tzinfo=UTC),
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
    )
    process, owned = _start_coordinated_project_cli(tmp_path, "update")
    content = b'{"canonical_entity":"Example"}\n'
    (owned / "owner.json").write_bytes(content)
    source = SourceArtifactDeclaration(
        source_id="owner-source",
        filename="owner.json",
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        platform=VisibilitySource.MANUAL,
        report_type="owner-context",
    )
    normalized = NormalizedIntake(
        project_id="example",
        canonical_domain="example.com",
        sources=(source,),
        owner_facts=(
            OwnerFactInput(
                fact_id="canonical-entity",
                field="canonical_entity",
                value="Example",
                source_id=source.source_id,
                approval_state=FactApprovalState.APPROVED,
            ),
        ),
    )
    (owned / "normalized-intake.json").write_text(
        normalized.model_dump_json(),
        encoding="utf-8",
    )

    stdout, stderr = process.communicate("READY\n", timeout=30)

    assert process.returncode == 0, stderr
    assert "Client delivery is pending" in stderr
    assert "project finalize" in stderr
    promoted = Path(stdout.strip())
    assert promoted.name.endswith("_context-v2")
    assert promoted.is_dir()
    manifest = json.loads((clients_root / "example" / "project.json").read_text())
    assert manifest["versions"][-1]["version_id"] == "context-v2"
    assert not owned.exists()


def test_project_update_rejects_oversized_normalized_json_and_deletes_intake(
    tmp_path: Path,
) -> None:
    intake_root = tmp_path / "intake"
    owned = create_owned_intake_dir(intake_root)
    normalized_path = owned / "normalized-intake.json"
    normalized_path.write_bytes(b'{"padding":"' + b"x" * (2 * 1024 * 1024) + b'"}')

    with pytest.raises(ValueError, match="size limit"):
        cli.main(
            [
                "project",
                "update",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--intake-dir",
                str(owned),
                "--normalized-intake",
                str(normalized_path),
            ]
        )

    assert not owned.exists()


def test_cli_parses_project_validation_with_optional_owned_intake() -> None:
    args = build_parser().parse_args(
        [
            "project",
            "validate",
            "project:example",
            "--clients-root",
            "/tmp/clients",
            "--implementation-date",
            "2026-09-01",
            "--intake-dir",
            "/tmp/intake/run-1",
            "--normalized-intake",
            "/tmp/intake/run-1/normalized-intake.json",
        ]
    )

    assert args.project_command == "validate"
    assert args.project_ref == "project:example"
    assert args.implementation_date == date(2026, 9, 1)
    assert args.intake_dir == Path("/tmp/intake/run-1")


def test_project_validation_cli_forwards_data_and_prints_stable_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clients_root = tmp_path / "clients"
    intake_dir = create_owned_intake_dir(tmp_path / "intake")
    normalized_path = intake_dir / "normalized-intake.json"
    normalized_path.write_text('{"project_id":"example"}', encoding="utf-8")
    now = datetime(2026, 9, 1, tzinfo=UTC)
    public = AuditVersionRef(
        version_id="public-v1",
        version_number=1,
        stage=AuditStage.PUBLIC,
        report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_public-v1",
    )
    validation = AuditVersionRef(
        version_id="validation-v2",
        version_number=2,
        stage=AuditStage.VALIDATION,
        report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
        audit_id="audit-validation",
        source_audit_id="audit-public",
        created_at=now,
        relative_path="audits/2026-09-01_validation-v2",
    )
    manifest = ProjectManifest(
        project_id="example",
        client_name="Example",
        canonical_domains=("example.com",),
        report_locale="en",
        created_at=now,
        latest_audit_id=validation.audit_id,
        source_files_policy="delete-after-processing",
        versions=(public, validation),
    )
    received: dict[str, object] = {}

    def fake_validate(project_ref: str, **kwargs: object) -> ProjectManifest:
        received.update(project_ref=project_ref, **kwargs)
        return manifest

    monkeypatch.setattr(cli, "validate_project", fake_validate)

    result = cli.main(
        [
            "project",
            "validate",
            "project:example",
            "--clients-root",
            str(clients_root),
            "--implementation-date",
            "2026-09-01",
            "--intake-dir",
            str(intake_dir),
            "--normalized-intake",
            str(normalized_path),
        ]
    )

    assert result == 0
    assert received == {
        "project_ref": "project:example",
        "clients_root": clients_root,
        "implementation_date": date(2026, 9, 1),
        "intake_dir": intake_dir,
        "normalized_intake": {"project_id": "example"},
    }
    assert capsys.readouterr().out.strip() == str(
        clients_root / "example" / validation.relative_path
    )
