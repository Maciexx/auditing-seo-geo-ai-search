from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import httpx
import pytest

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
from ai_search_audit.project_orchestrator import (
    create_project_audit,
    update_project_context,
    validate_project_bundle,
)
from ai_search_audit.project_store import ProjectStore
from tests.test_crawler import public_resolver, transport

ROOT = Path(__file__).parents[1]


def _load_migration_module() -> ModuleType:
    script = ROOT / "scripts" / "migrate_client_project.py"
    spec = importlib.util.spec_from_file_location("migrate_client_project", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    fixed_time = 1_788_134_400  # 2026-08-31T00:00:00Z
    os.utime(path, (fixed_time, fixed_time))
    return path


def _canonical_project(clients_root: Path) -> Path:
    canonical_root = clients_root.parent / "canonical-clients"
    project = canonical_root / "example"
    if project.exists():
        return project
    create_project_audit(
        domain="https://example.com",
        clients_root=canonical_root,
        project_id="example",
        client_name="Example Client",
        report_locale="en",
        max_pages=2,
        now=datetime(2026, 8, 31, 10, tzinfo=UTC),
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
    )
    return project


def _base_args(
    clients_root: Path,
    report: Path,
    *,
    canonical_project: Path | None = None,
) -> list[str]:
    canonical = canonical_project or _canonical_project(clients_root)
    return [
        "--clients-root",
        str(clients_root),
        "--canonical-project",
        str(canonical),
        "--project-id",
        "example",
        "--client-name",
        "Example Client",
        "--domain",
        "example.com",
        "--locale",
        "en",
        "--report-file",
        str(report),
    ]


def _simplified_import(
    clients_root: Path,
    files: tuple[tuple[str, bytes, str], ...],
) -> tuple[Path, ...]:
    project_root = clients_root / "example"
    version_relative = "audits/2026-08-31_public-v1"
    version = AuditVersionRef(
        version_id="public-v1",
        version_number=1,
        stage=AuditStage.PUBLIC,
        report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        audit_id="migration-fabricated",
        source_audit_id=None,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        relative_path=version_relative,
    )
    manifest = ProjectManifest(
        project_id="example",
        client_name="Example Client",
        canonical_domains=("example.com",),
        report_locale="en",
        created_at=version.created_at,
        latest_audit_id=version.audit_id,
        source_files_policy="delete-after-processing",
        versions=(version,),
    )
    records: list[dict[str, object]] = []
    sources: list[Path] = []
    for relative, content, role in files:
        source = _write_source(project_root / relative, content)
        sources.append(source)
        records.append(
            {
                "byte_count": len(content),
                "relative_path": relative,
                "role": role,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    migration_manifest = {
        "schema_version": "1.0.0",
        "project_id": "example",
        "version_id": "public-v1",
        "files": records,
    }
    manifests = project_root / version_relative / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "input-manifest.json").write_text(json.dumps(migration_manifest))
    (manifests / "output-manifest.json").write_text(json.dumps(migration_manifest))
    (project_root / "project.json").write_text(manifest.model_dump_json())
    return tuple(sources)


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_migration_requires_a_separate_canonical_project(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(SystemExit):
        migration.main(
            [
                "--clients-root",
                str(tmp_path / "clients"),
                "--project-id",
                "example",
                "--client-name",
                "Example Client",
                "--domain",
                "example.com",
                "--locale",
                "en",
                "--report-file",
                str(report),
                "--dry-run",
            ]
        )


def test_script_runs_directly_from_the_repository_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/migrate_client_project.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--move-after-verify" in completed.stdout


def test_dry_run_is_deterministic_and_does_not_create_or_remove_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report-bytes")
    expected_hash = _sha256(report)

    assert migration.main([*_base_args(clients_root, report), "--dry-run"]) == 0
    first = capsys.readouterr().out
    assert migration.main([*_base_args(clients_root, report), "--dry-run"]) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["dry_run"] is True
    assert payload["project_manifest"]["versions"][0]["relative_path"] == (
        "audits/2026-08-31_public-v1"
    )
    assert payload["files"] == [
        {
            "byte_count": len(b"report-bytes"),
            "relative_path": "legacy/imported-public-audit/audit.pdf",
            "role": "report",
            "sha256": expected_hash,
        }
    ]
    assert report.read_bytes() == b"report-bytes"
    assert not clients_root.exists()
    assert not (clients_root / "example").exists()
    assert not (clients_root / ".migration-staging").exists()


def test_migration_promotes_public_v1_with_relative_hash_manifest_and_legacy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.md", b"public-report")
    legacy = _write_source(tmp_path / "incoming" / "older.pdf", b"legacy-report")
    report_hash = _sha256(report)
    legacy_hash = _sha256(legacy)

    assert migration.main([*_base_args(clients_root, report), "--legacy-file", str(legacy)]) == 0

    project_root = clients_root / "example"
    manifest = ProjectStore(clients_root).load("example")
    version = manifest.versions[0]
    assert version.version_id == "public-v1"
    assert version.stage is AuditStage.PUBLIC
    assert version.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    assert version.source_audit_id is None
    assert manifest.latest_audit_id == version.audit_id
    imported = project_root / "legacy" / "imported-public-audit"
    assert (imported / report.name).read_bytes() == b"public-report"
    assert (imported / legacy.name).read_bytes() == b"legacy-report"
    output_manifest = json.loads((imported / "manifest.json").read_text())
    assert output_manifest["project_id"] == manifest.project_id
    assert output_manifest["audit_id"] == version.audit_id
    assert output_manifest["version_id"] == version.version_id
    assert output_manifest["files"] == [
        {
            "byte_count": len(b"public-report"),
            "relative_path": f"legacy/imported-public-audit/{report.name}",
            "role": "report",
            "sha256": report_hash,
        },
        {
            "byte_count": len(b"legacy-report"),
            "relative_path": f"legacy/imported-public-audit/{legacy.name}",
            "role": "legacy",
            "sha256": legacy_hash,
        },
    ]
    assert all(not Path(item["relative_path"]).is_absolute() for item in output_manifest["files"])
    assert report.exists() and legacy.exists()
    assert json.loads(capsys.readouterr().out)["promoted"] is True

    with pytest.raises(FileExistsError):
        migration.main(_base_args(clients_root, report))


def test_transplanted_legacy_manifest_provenance_is_rejected_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")
    original_add = migration._add_legacy_import

    def transplant(project_root: Path, **kwargs: object) -> None:
        original_add(project_root, **kwargs)
        manifest_path = project_root / "legacy/imported-public-audit/manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["audit_id"] = "audit-from-another-project"
        manifest_path.write_text(json.dumps(payload))

    monkeypatch.setattr(migration, "_add_legacy_import", transplant)

    with pytest.raises(RuntimeError, match="legacy hash manifest"):
        migration.main(_base_args(clients_root, report))

    assert not (clients_root / "example").exists()


def test_canonical_bundle_is_validated_before_assembly_and_after_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    canonical = _canonical_project(clients_root)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")
    validated_roots: list[Path] = []
    original = migration.validate_project_bundle

    def record_validation(version_root: Path, **kwargs: object) -> object:
        validated_roots.append(version_root)
        return original(version_root, **kwargs)

    monkeypatch.setattr(migration, "validate_project_bundle", record_validation)

    assert migration.main(_base_args(clients_root, report, canonical_project=canonical)) == 0

    version_relative = ProjectStore(clients_root).load("example").versions[0].relative_path
    assert canonical / version_relative in validated_roots
    assert clients_root / "example" / version_relative in validated_roots


def test_incomplete_canonical_bundle_is_rejected_before_dry_run_output(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    canonical = _canonical_project(clients_root)
    manifest = ProjectStore(canonical.parent).load("example")
    (canonical / manifest.versions[0].relative_path / "engine" / "audit.json").unlink()
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="inventory"):
        migration.main(
            [*_base_args(clients_root, report, canonical_project=canonical), "--dry-run"]
        )

    assert not clients_root.exists()


def test_canonical_outer_audit_identity_must_match_validated_engine_audit(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    canonical = _canonical_project(clients_root)
    project_path = canonical / "project.json"
    payload = json.loads(project_path.read_text())
    payload["versions"][0]["audit_id"] = "audit-outer-mismatch"
    payload["latest_audit_id"] = "audit-outer-mismatch"
    project_path.write_text(json.dumps(payload))
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="manifest audit identity"):
        migration.main(
            [*_base_args(clients_root, report, canonical_project=canonical), "--dry-run"]
        )

    assert not clients_root.exists()


def test_canonical_project_symlink_is_rejected_before_copytree(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    canonical = _canonical_project(clients_root)
    external = _write_source(tmp_path / "external" / "secret.pdf", b"secret")
    (canonical / "unexpected-link.pdf").symlink_to(external)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="symlink"):
        migration.main(
            [*_base_args(clients_root, report, canonical_project=canonical), "--dry-run"]
        )

    assert external.read_bytes() == b"secret"
    assert not clients_root.exists()


def test_promoted_canonical_project_can_immediately_create_context_v2(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")
    assert migration.main(_base_args(clients_root, report)) == 0
    owned = create_owned_intake_dir(tmp_path / "intake")
    content = b'{"canonical_entity":"Example Client"}\n'
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
                value="Example Client",
                source_id=source.source_id,
                approval_state=FactApprovalState.APPROVED,
            ),
        ),
    )

    manifest = update_project_context(
        "project:example",
        clients_root=clients_root,
        intake_dir=owned,
        normalized_intake=normalized,
        now=datetime(2026, 9, 1, 10, tzinfo=UTC),
    )

    assert manifest.versions[-1].version_id == "context-v2"
    assert not owned.exists()


def test_move_after_verify_removes_sources_only_after_final_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    original_validation = migration._validate_assembled_project

    def fail_final_validation(project_root: Path, **kwargs: object) -> None:
        if project_root == clients_root / "example":
            raise RuntimeError("forced final validation failure")
        original_validation(project_root, **kwargs)

    monkeypatch.setattr(migration, "_validate_assembled_project", fail_final_validation)

    with pytest.raises(RuntimeError, match="forced final validation failure"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert report.read_bytes() == b"report"
    assert not (clients_root / "example").exists()


def test_move_after_verify_deletes_external_sources_after_success(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")
    report_hash = _sha256(report)

    assert migration.main([*_base_args(clients_root, report), "--move-after-verify"]) == 0

    destination = clients_root / "example" / "legacy/imported-public-audit/audit.pdf"
    assert not report.exists()
    assert _sha256(destination) == report_hash
    ProjectStore(clients_root).load("example")
    assert not (clients_root / ".migration-staging").exists()


def test_move_after_verify_can_replace_a_validated_simplified_import(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report, notes = _simplified_import(
        clients_root,
        (
            ("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),
            ("audits/2026-08-31_public-v1/report/notes.md", b"notes", "report"),
        ),
    )
    args = [
        *_base_args(clients_root, report),
        "--report-file",
        str(notes),
        "--move-after-verify",
    ]

    assert migration.main(args) == 0

    project_root = clients_root / "example"
    imported = project_root / "legacy/imported-public-audit"
    assert (imported / "audit.pdf").read_bytes() == b"report"
    assert (imported / "notes.md").read_bytes() == b"notes"
    ProjectStore(clients_root).load("example")


def test_existing_unlisted_project_content_is_never_overwritten(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    keep = _write_source(clients_root / "example" / "keep.txt", b"keep")

    with pytest.raises(FileExistsError, match="unlisted"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert report.read_bytes() == b"report"
    assert keep.read_bytes() == b"keep"
    assert (clients_root / "example" / "project.json").exists()


def test_existing_simplified_import_with_hash_mismatch_is_never_replaced(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    report.write_bytes(b"tampered")

    with pytest.raises(FileExistsError, match="hash validation"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert report.read_bytes() == b"tampered"
    assert (clients_root / "example" / "project.json").exists()


@pytest.mark.parametrize("mode", ("--dry-run", "--move-after-verify"))
@pytest.mark.parametrize(
    "mutation",
    (
        "client-name",
        "domain",
        "locale",
        "version-id",
        "version-number",
        "stage-status",
        "source-audit",
        "relative-path",
    ),
)
def test_existing_simplified_import_identity_or_public_v1_mismatch_is_never_replaced(
    tmp_path: Path,
    mutation: str,
    mode: str,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    project_root = clients_root / "example"
    project_path = project_root / "project.json"
    payload = json.loads(project_path.read_text())
    version = payload["versions"][0]
    if mutation == "client-name":
        payload["client_name"] = "Different Client"
    elif mutation == "domain":
        payload["canonical_domains"] = ["different.example"]
    elif mutation == "locale":
        payload["report_locale"] = "pl"
    elif mutation == "version-id":
        version["version_id"] = "imported-v1"
    elif mutation == "version-number":
        version["version_number"] = 2
    elif mutation == "stage-status":
        version["stage"] = "context"
        version["report_status"] = "CLIENT_CONTEXT_DRAFT"
    elif mutation == "source-audit":
        version["source_audit_id"] = "audit-unexpected-source"
    elif mutation == "relative-path":
        original = project_root / version["relative_path"]
        replacement = project_root / "audits" / "imported"
        original.rename(replacement)
        version["relative_path"] = "audits/imported"
        report = replacement / "report" / "audit.pdf"
    project_path.write_text(json.dumps(payload))
    before = _tree_snapshot(project_root)

    with pytest.raises((ValueError, FileExistsError)):
        migration.main([*_base_args(clients_root, report), mode])

    assert _tree_snapshot(project_root) == before
    assert not (clients_root / ".migration-staging").exists()


def test_destination_symlink_is_rejected_before_staging_or_swap(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    clients_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    keep = _write_source(external / "keep.txt", b"keep")
    destination = clients_root / "example"
    destination.symlink_to(external, target_is_directory=True)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="symlink"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert destination.is_symlink()
    assert keep.read_bytes() == b"keep"
    assert not (clients_root / ".migration-staging").exists()


def test_symlinked_clients_root_is_rejected_without_touching_target(tmp_path: Path) -> None:
    migration = _load_migration_module()
    external = tmp_path / "external-clients"
    external.mkdir()
    clients_root = tmp_path / "linked-clients"
    clients_root.symlink_to(external, target_is_directory=True)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="symlink"):
        migration.main([*_base_args(clients_root, report), "--dry-run"])

    assert list(external.iterdir()) == []


def test_symlinked_migration_staging_is_rejected_without_writes(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    clients_root.mkdir()
    external = tmp_path / "external-staging"
    external.mkdir()
    (clients_root / ".migration-staging").symlink_to(external, target_is_directory=True)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")

    with pytest.raises(ValueError, match="staging.*symlink"):
        migration.main([*_base_args(clients_root, report), "--dry-run"])

    assert list(external.iterdir()) == []
    assert report.read_bytes() == b"report"


def test_source_through_symlinked_parent_is_rejected_before_dry_run_writes(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    external = tmp_path / "external"
    report = _write_source(external / "audit.pdf", b"report")
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        migration.main([*_base_args(clients_root, linked / report.name), "--dry-run"])

    assert linked.is_symlink()
    assert report.read_bytes() == b"report"
    assert not clients_root.exists()


def test_case_insensitive_destination_collision_is_rejected(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    upper = _write_source(tmp_path / "one" / "Audit.pdf", b"same")
    lower = _write_source(tmp_path / "two" / "audit.pdf", b"same")

    with pytest.raises(ValueError, match="case-insensitive"):
        migration.main(
            [
                *_base_args(clients_root, upper),
                "--report-file",
                str(lower),
                "--dry-run",
            ]
        )

    assert not clients_root.exists()


def test_unicode_equivalent_destination_collision_is_rejected(tmp_path: Path) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    composed = _write_source(
        tmp_path / "one" / "Caf\N{LATIN SMALL LETTER E WITH ACUTE}.pdf", b"same"
    )
    decomposed = _write_source(
        tmp_path / "two" / "Cafe\N{COMBINING ACUTE ACCENT}.pdf",
        b"same",
    )

    with pytest.raises(ValueError, match="collision"):
        migration.main(
            [
                *_base_args(clients_root, composed),
                "--report-file",
                str(decomposed),
                "--dry-run",
            ]
        )

    assert not clients_root.exists()


def test_existing_empty_directory_is_unlisted_content_and_is_preserved(
    tmp_path: Path,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    empty = clients_root / "example" / "empty"
    empty.mkdir()

    with pytest.raises(FileExistsError, match="unlisted"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert report.read_bytes() == b"report"
    assert empty.is_dir()
    assert list(empty.iterdir()) == []
    assert not (clients_root / ".migration-staging").exists()


def test_destination_race_preserves_validated_original_and_reports_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    destination = clients_root / "example"
    original_rename = migration._rename_directory_no_replace

    def inject_race(source: Path, target: Path) -> None:
        if target == destination and source.name == "example":
            destination.mkdir()
            raise FileExistsError("injected destination race")
        original_rename(source, target)

    monkeypatch.setattr(migration, "_rename_directory_no_replace", inject_race)

    with pytest.raises(RuntimeError, match="recovery path") as raised:
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    recovery = Path(str(raised.value).split("recovery path: ", 1)[1])
    assert recovery.is_dir()
    assert (recovery / report.relative_to(destination)).read_bytes() == b"report"
    assert {entry.name for entry in recovery.parent.iterdir()} == {"original"}
    assert destination.is_dir()


def test_backup_mutation_after_swap_fails_closed_and_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    destination = clients_root / "example"
    original_rename = migration._rename_directory_no_replace

    def mutate_backup(source: Path, target: Path) -> None:
        original_rename(source, target)
        if source == destination and target.name == "original":
            (target / "keep.txt").write_bytes(b"keep")

    monkeypatch.setattr(migration, "_rename_directory_no_replace", mutate_backup)

    with pytest.raises(FileExistsError, match="unlisted"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert (destination / "keep.txt").read_bytes() == b"keep"
    assert report.read_bytes() == b"report"


def test_backup_restore_race_reports_exact_validated_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    destination = clients_root / "example"
    original_validation = migration._validate_assembled_project
    original_rename = migration._rename_no_replace

    def fail_final(project_root: Path, **kwargs: object) -> None:
        if project_root == destination:
            raise RuntimeError("forced final validation failure")
        original_validation(project_root, **kwargs)

    def collide_restore(source: Path, target: Path) -> None:
        if source.name == "original" and target == destination:
            destination.mkdir()
        original_rename(source, target)

    monkeypatch.setattr(migration, "_validate_assembled_project", fail_final)
    monkeypatch.setattr(migration, "_rename_no_replace", collide_restore)

    with pytest.raises(RuntimeError, match="recovery path") as raised:
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    recovery = Path(str(raised.value).split("recovery path: ", 1)[1])
    assert recovery.is_dir()
    assert (recovery / report.relative_to(destination)).read_bytes() == b"report"


def test_backup_removal_failure_reports_and_preserves_exact_recovery_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    (report,) = _simplified_import(
        clients_root,
        (("audits/2026-08-31_public-v1/report/audit.pdf", b"report", "report"),),
    )
    canonical = _canonical_project(clients_root)
    original_rmtree = migration.shutil.rmtree

    def fail_backup_removal(path: str | Path, *args: object, **kwargs: object) -> None:
        if Path(path).name == "original":
            raise OSError("forced backup removal failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(migration.shutil, "rmtree", fail_backup_removal)

    with pytest.raises(RuntimeError, match="recovery paths") as raised:
        migration.main(
            [
                *_base_args(clients_root, report, canonical_project=canonical),
                "--move-after-verify",
            ]
        )

    recovery_text = str(raised.value).split("recovery paths: ", 1)[1]
    backup = Path(recovery_text.split(", ", 1)[-1])
    assert backup.name == "original"
    assert backup.is_dir()
    assert (backup / report.relative_to(clients_root / "example")).read_bytes() == b"report"


def test_canonical_mutation_between_inventory_and_copy_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    canonical = _canonical_project(clients_root)
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"report")
    original_copy = migration._copy_canonical_project

    def mutate_then_copy(source: Path, target: Path) -> None:
        (source / "injected.txt").write_bytes(b"injected")
        original_copy(source, target)

    monkeypatch.setattr(migration, "_copy_canonical_project", mutate_then_copy)

    with pytest.raises(RuntimeError, match="canonical inventory"):
        migration.main(_base_args(clients_root, report, canonical_project=canonical))

    assert not (clients_root / "example").exists()


def test_quarantine_collision_preserves_source_and_collision_with_recovery_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"original")
    original_rename = migration._rename_no_replace
    collision: Path | None = None

    def inject_collision(source: Path, target: Path) -> None:
        nonlocal collision
        if source == report and ".migration-quarantine-" in target.name:
            collision = target
            target.write_bytes(b"collision")
        original_rename(source, target)

    monkeypatch.setattr(migration, "_rename_no_replace", inject_collision)

    with pytest.raises(RuntimeError, match="recovery paths"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert collision is not None
    assert report.read_bytes() == b"original"
    assert collision.read_bytes() == b"collision"


def test_restore_collision_preserves_both_paths_with_exact_recovery_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"original")
    original_quarantine = migration._atomic_quarantine_source
    original_rename = migration._rename_no_replace
    quarantine: Path | None = None

    def replace_before_quarantine(record: object) -> Path:
        report.unlink()
        report.write_bytes(b"replacement")
        return original_quarantine(record)

    def collide_on_restore(source: Path, target: Path) -> None:
        nonlocal quarantine
        if ".migration-quarantine-" in source.name and target == report:
            quarantine = source
            target.write_bytes(b"restore-collision")
        original_rename(source, target)

    monkeypatch.setattr(migration, "_atomic_quarantine_source", replace_before_quarantine)
    monkeypatch.setattr(migration, "_rename_no_replace", collide_on_restore)

    with pytest.raises(RuntimeError) as raised:
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert quarantine is not None
    assert str(report) in str(raised.value)
    assert str(quarantine) in str(raised.value)
    assert report.read_bytes() == b"restore-collision"
    assert quarantine.read_bytes() == b"replacement"


def test_source_identity_change_before_removal_never_deletes_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    report = _write_source(tmp_path / "incoming" / "audit.pdf", b"original")
    original_validation = migration._validate_assembled_project
    replaced = False

    def replace_after_final_validation(project_root: Path, **kwargs: object) -> None:
        nonlocal replaced
        original_validation(project_root, **kwargs)
        if project_root == clients_root / "example" and not replaced:
            report.unlink()
            report.write_bytes(b"replacement")
            replaced = True

    monkeypatch.setattr(migration, "_validate_assembled_project", replace_after_final_validation)

    with pytest.raises(RuntimeError, match="identity changed"):
        migration.main([*_base_args(clients_root, report), "--move-after-verify"])

    assert report.read_bytes() == b"replacement"
    validate_project_bundle(
        clients_root
        / "example"
        / ProjectStore(clients_root).load("example").versions[0].relative_path,
        expected_project_id="example",
    )


def test_atomic_quarantine_race_preserves_replacement_and_prior_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    migration = _load_migration_module()
    clients_root = tmp_path / "clients"
    first = _write_source(tmp_path / "incoming-one" / "first.pdf", b"first-original")
    second = _write_source(tmp_path / "incoming-two" / "second.pdf", b"second-original")
    original_quarantine = migration._atomic_quarantine_source

    def inject_race(record: object) -> Path:
        if record.source == second:
            second.unlink()
            second.write_bytes(b"second-replacement")
        return original_quarantine(record)

    monkeypatch.setattr(migration, "_atomic_quarantine_source", inject_race)

    with pytest.raises(RuntimeError, match="source removal failed"):
        migration.main(
            [
                *_base_args(clients_root, first),
                "--report-file",
                str(second),
                "--move-after-verify",
            ]
        )

    assert first.read_bytes() == b"first-original"
    assert second.read_bytes() == b"second-replacement"
    assert "sources_removed" not in capsys.readouterr().out
    assert not list(tmp_path.glob("incoming-*/*.migration-quarantine-*"))


def test_migrator_is_projection_only_and_does_not_import_crawl_entrypoints() -> None:
    source = (ROOT / "scripts" / "migrate_client_project.py").read_text()

    assert "run_public_audit" not in source
    assert "create_project_audit" not in source
