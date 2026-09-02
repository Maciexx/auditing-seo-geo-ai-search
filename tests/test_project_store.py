from __future__ import annotations

import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import ai_search_audit.project_store as project_store_module
from ai_search_audit.project_models import (
    AuditStage,
    AuditVersionRef,
    ProjectManifest,
    ReportStatus,
)
from ai_search_audit.project_store import (
    PendingProject,
    PendingVersion,
    ProjectIdentityError,
    ProjectStore,
)

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


def version_fixture(
    *,
    version_number: int = 1,
    stage: AuditStage = AuditStage.PUBLIC,
    report_status: ReportStatus = ReportStatus.PUBLIC_EVIDENCE_DRAFT,
    audit_id: str | None = None,
    source_audit_id: str | None = None,
    created_at: datetime = NOW,
) -> AuditVersionRef:
    version_id = f"{stage.value}-v{version_number}"
    return AuditVersionRef(
        version_id=version_id,
        version_number=version_number,
        stage=stage,
        report_status=report_status,
        audit_id=audit_id or f"audit-{version_number}",
        source_audit_id=source_audit_id,
        created_at=created_at,
        relative_path=f"audits/{created_at.date().isoformat()}_{version_id}",
    )


def project_fixture(**overrides: Any) -> ProjectManifest:
    first = version_fixture()
    values: dict[str, Any] = {
        "project_id": "example-studio",
        "client_name": "Example Studio",
        "canonical_domains": ("example-studio.example",),
        "report_locale": "pl",
        "created_at": NOW,
        "latest_audit_id": first.audit_id,
        "source_files_policy": "delete-after-processing",
        "versions": (first,),
    }
    values.update(overrides)
    return ProjectManifest(**values)


def completed_version(
    pending: PendingVersion,
    *,
    audit_id: str | None = None,
    report_status: ReportStatus | None = None,
) -> AuditVersionRef:
    pending.staging_path.mkdir(parents=True, exist_ok=True)
    (pending.staging_path / "report.json").write_text("{}", encoding="utf-8")
    statuses = {
        AuditStage.PUBLIC: ReportStatus.PUBLIC_EVIDENCE_DRAFT,
        AuditStage.CONTEXT: ReportStatus.CLIENT_CONTEXT_DRAFT,
        AuditStage.VALIDATION: ReportStatus.CLIENT_VALIDATED,
    }
    return AuditVersionRef(
        version_id=pending.version_id,
        version_number=pending.version_number,
        stage=pending.stage,
        report_status=report_status or statuses[pending.stage],
        audit_id=audit_id or f"audit-{pending.version_number}",
        source_audit_id=(
            None if pending.stage is AuditStage.PUBLIC else f"audit-{pending.version_number - 1}"
        ),
        created_at=pending.created_at,
        relative_path=pending.relative_path,
    )


def test_project_store_documents_single_writer_sequential_contract() -> None:
    contract = ProjectStore.__doc__ or ""

    assert "single-writer" in contract.lower()
    assert "sequential" in contract.lower()


def test_create_project_atomically_writes_one_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("ai_search_audit.project_store.os.replace", recording_replace)
    store = ProjectStore(tmp_path)

    created = store.create_project(project_fixture())

    project_dir = tmp_path / created.project_id
    assert [path.name for path in tmp_path.iterdir() if path.name != ".staging"] == [
        created.project_id
    ]
    assert store.load(created.project_id) == created
    manifest_replacements = [item for item in replacements if item[1].name == "project.json"]
    assert len(manifest_replacements) == 1
    temporary_manifest, staged_manifest = manifest_replacements[0]
    assert staged_manifest.parent.parent == project_dir.parent / ".staging"
    assert temporary_manifest.parent == staged_manifest.parent
    assert temporary_manifest.name != "project.json"


def test_new_project_is_staged_then_promoted_only_after_public_v1_validates(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path)
    pending = store.begin_new_project("example-studio")

    assert pending.staging_path.parent == tmp_path / ".staging"
    assert pending.staging_path.is_dir()
    assert not (tmp_path / "example-studio").exists()

    invalid = project_fixture(
        versions=(
            version_fixture(
                stage=AuditStage.CONTEXT,
                report_status=ReportStatus.CLIENT_CONTEXT_DRAFT,
            ),
        )
    )
    (pending.staging_path / invalid.versions[0].relative_path).mkdir(parents=True)
    with pytest.raises(ValueError, match="public-v1"):
        store.promote_new_project(pending, invalid)

    assert pending.staging_path.exists()
    assert not (tmp_path / "example-studio").exists()

    manifest = project_fixture()
    (pending.staging_path / manifest.versions[0].relative_path).mkdir(parents=True)
    promoted = store.promote_new_project(pending, manifest)

    assert promoted == manifest
    assert not pending.staging_path.exists()
    assert (tmp_path / "example-studio" / manifest.versions[0].relative_path).is_dir()


def test_new_project_requires_every_referenced_version_directory(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    pending = store.begin_new_project("example-studio")

    with pytest.raises(RuntimeError, match="store integrity.*version root.*missing"):
        store.promote_new_project(pending, project_fixture())

    assert not (tmp_path / "example-studio").exists()


def test_new_project_rejects_referenced_version_symlink(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    manifest = project_fixture()
    pending = store.begin_new_project(manifest.project_id)
    real_version = pending.staging_path / "real-public-version"
    real_version.mkdir()
    version_path = pending.staging_path / manifest.versions[0].relative_path
    version_path.parent.mkdir(parents=True)
    version_path.symlink_to(real_version, target_is_directory=True)

    with pytest.raises(RuntimeError, match="store integrity.*version root.*symlink"):
        store.promote_new_project(pending, manifest)

    assert pending.staging_path.is_dir()
    assert version_path.is_symlink()
    assert not pending.destination_path.exists()


def test_new_project_rejects_symlinked_version_path_ancestor(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    manifest = project_fixture()
    pending = store.begin_new_project(manifest.project_id)
    real_audits = pending.staging_path / "real-audits"
    real_audits.mkdir()
    (pending.staging_path / "audits").symlink_to(real_audits, target_is_directory=True)
    (real_audits / "2026-08-31_public-v1").mkdir()

    with pytest.raises(RuntimeError, match="store integrity.*version root.*ancestor.*symlink"):
        store.promote_new_project(pending, manifest)

    assert pending.staging_path.is_dir()
    assert not (pending.staging_path / "project.json").exists()
    assert not pending.destination_path.exists()


def test_begin_new_project_rejects_symlinked_staging_root(tmp_path: Path) -> None:
    clients_root = tmp_path / "clients"
    clients_root.mkdir()
    real_staging = clients_root / "real-staging"
    real_staging.mkdir()
    (clients_root / ".staging").symlink_to(real_staging, target_is_directory=True)

    with pytest.raises(RuntimeError, match="store integrity.*staging root.*symlink"):
        ProjectStore(clients_root).begin_new_project("example-studio")

    assert list(real_staging.iterdir()) == []


def test_new_project_promotion_rejects_symlinked_pending_root(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    manifest = project_fixture()
    pending = store.begin_new_project(manifest.project_id)
    pending.staging_path.rmdir()
    real_staging = pending.staging_path.parent / "another-project-run"
    real_staging.mkdir()
    pending.staging_path.symlink_to(real_staging, target_is_directory=True)

    with pytest.raises(RuntimeError, match="store integrity.*pending project staging.*symlink"):
        store.promote_new_project(pending, manifest)

    assert pending.staging_path.is_symlink()
    assert not pending.destination_path.exists()


def test_discard_pending_project_removes_only_store_owned_staging_root(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    pending = store.begin_new_project("example-studio")
    unrelated = tmp_path / ".staging" / "unrelated-run"
    unrelated.mkdir()
    marker = unrelated / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    store.discard_pending_project(pending)

    assert not pending.staging_path.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_discard_pending_project_rejects_foreign_pending_root(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "clients")
    foreign_store = ProjectStore(tmp_path / "foreign")
    pending = foreign_store.begin_new_project("example-studio")
    forged = PendingProject(
        project_id=pending.project_id,
        run_id=pending.run_id,
        staging_path=pending.staging_path,
        destination_path=pending.destination_path,
    )

    with pytest.raises(ValueError, match="does not belong"):
        store.discard_pending_project(forged)

    assert pending.staging_path.is_dir()


def test_discard_pending_project_unlinks_replaced_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path / "clients")
    pending = store.begin_new_project("example-studio")
    pending.staging_path.rmdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    marker = unrelated / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    pending.staging_path.symlink_to(unrelated, target_is_directory=True)

    store.discard_pending_project(pending)

    assert not pending.staging_path.exists()
    assert not pending.staging_path.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_discard_pending_project_refuses_replaced_staging_parent_symlink(
    tmp_path: Path,
) -> None:
    clients_root = tmp_path / "clients"
    store = ProjectStore(clients_root)
    pending = store.begin_new_project("example-studio")
    original_staging = clients_root / ".staging-original"
    (clients_root / ".staging").rename(original_staging)
    unrelated = tmp_path / "unrelated"
    unrelated_pending = unrelated / pending.staging_path.name
    unrelated_pending.mkdir(parents=True)
    marker = unrelated_pending / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    (clients_root / ".staging").symlink_to(unrelated, target_is_directory=True)

    with pytest.raises(RuntimeError, match="staging root.*symlink"):
        store.discard_pending_project(pending)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert (original_staging / pending.staging_path.name).is_dir()


def test_directory_rename_no_replace_preserves_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    (destination / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        project_store_module._rename_directory_no_replace(source, destination)

    assert (source / "new.txt").read_text(encoding="utf-8") == "new"
    assert (destination / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_directory_rename_no_replace_fails_closed_on_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(sys, "platform", "unsupported-os")

    with pytest.raises(RuntimeError, match="unsupported"):
        project_store_module._rename_directory_no_replace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


@pytest.mark.parametrize("nonempty_destination", (False, True))
def test_new_project_promotion_does_not_replace_boundary_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonempty_destination: bool,
) -> None:
    store = ProjectStore(tmp_path)
    manifest = project_fixture()
    pending = store.begin_new_project(manifest.project_id)
    (pending.staging_path / manifest.versions[0].relative_path).mkdir(parents=True)
    original_rename = getattr(project_store_module, "_rename_directory_no_replace", None)

    def collide_at_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        if nonempty_destination:
            (destination / "existing.txt").write_text("keep", encoding="utf-8")
        assert original_rename is not None
        original_rename(source, destination)

    monkeypatch.setattr(
        project_store_module,
        "_rename_directory_no_replace",
        collide_at_rename,
        raising=False,
    )

    with pytest.raises(FileExistsError):
        store.promote_new_project(pending, manifest)

    assert pending.staging_path.is_dir()
    expected_contents = ["existing.txt"] if nonempty_destination else []
    assert sorted(path.name for path in pending.destination_path.iterdir()) == expected_contents


def test_resolve_accepts_only_project_references_beneath_explicit_clients_root(
    tmp_path: Path,
) -> None:
    clients_root = tmp_path / "clients"
    store = ProjectStore(clients_root)
    store.create_project(project_fixture())

    assert store.resolve("project:example-studio") == clients_root / "example-studio"
    with pytest.raises(ValueError, match="project:<id>"):
        store.resolve("example-studio")
    with pytest.raises(ValueError, match="safe identifier"):
        store.resolve("project:../escape")
    with pytest.raises(TypeError):
        ProjectStore()  # type: ignore[call-arg]


@pytest.mark.parametrize("target_outside_root", (False, True))
def test_load_refuses_symlinked_project_root(tmp_path: Path, target_outside_root: bool) -> None:
    clients_root = tmp_path / "clients"
    clients_root.mkdir()
    target_parent = tmp_path if target_outside_root else clients_root
    target = target_parent / "symlink-target"
    target.mkdir()
    (clients_root / "example-studio").symlink_to(target, target_is_directory=True)

    store = ProjectStore(clients_root)

    with pytest.raises(RuntimeError, match="store integrity.*project root.*symlink"):
        store.load("example-studio")


@pytest.mark.parametrize("entry_kind", ("missing", "file", "symlink"))
def test_load_rejects_invalid_manifest_version_root(tmp_path: Path, entry_kind: str) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    version_path = project_dir / project.versions[0].relative_path
    shutil.rmtree(version_path)
    if entry_kind == "file":
        version_path.write_text("not a directory", encoding="utf-8")
    elif entry_kind == "symlink":
        real_version = project_dir / "real-version"
        real_version.mkdir()
        version_path.symlink_to(real_version, target_is_directory=True)

    with pytest.raises(
        RuntimeError,
        match=rf"store integrity.*version root.*{entry_kind}",
    ):
        store.load(project.project_id)


@pytest.mark.parametrize("project_id", ("../escape", "CON", "nul\x00byte"))
def test_begin_new_project_reuses_manifest_project_id_validation(
    tmp_path: Path, project_id: str
) -> None:
    with pytest.raises(ValueError, match="safe identifier|portable filesystem component"):
        ProjectStore(tmp_path).begin_new_project(project_id)


@pytest.mark.parametrize(
    ("domain", "client_name"),
    (("different.example", "Example Studio"), ("example-studio.example", "Other Entity")),
)
def test_existing_project_id_refuses_different_domain_or_entity(
    tmp_path: Path, domain: str, client_name: str
) -> None:
    store = ProjectStore(tmp_path)
    store.create_project(project_fixture())

    with pytest.raises(ProjectIdentityError, match="different project identity"):
        store.assert_identity("example-studio", domain=domain, client_name=client_name)


def test_assert_identity_reuses_canonical_domain_normalization(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.create_project(project_fixture())

    store.assert_identity(
        "example-studio",
        domain="HTTPS://Example-Studio.Example/path?q=1",
        client_name=None,
    )


def test_allocate_versions_monotonically_as_public_context_validation(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create_project(project_fixture())
    assert created.versions[0].version_id == "public-v1"

    context_pending = store.allocate_version(created.project_id, AuditStage.CONTEXT, NOW)
    context = store.promote(context_pending, completed_version(context_pending))
    validation_pending = store.allocate_version(created.project_id, AuditStage.VALIDATION, NOW)
    validation = store.promote(validation_pending, completed_version(validation_pending))

    assert [item.version_id for item in validation.versions] == [
        "public-v1",
        "context-v2",
        "validation-v3",
    ]
    assert [item.version_number for item in validation.versions] == [1, 2, 3]
    assert context.versions[-1].relative_path == "audits/2026-08-31_context-v2"
    assert validation.versions[-1].relative_path == "audits/2026-08-31_validation-v3"
    assert [item.source_audit_id for item in validation.versions] == [
        None,
        "audit-1",
        "audit-2",
    ]


def test_allocate_version_rejects_later_public_before_creating_staging(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_staging = tmp_path / project.project_id / ".staging"

    with pytest.raises(ValueError, match="public.*initial public-v1"):
        store.allocate_version(project.project_id, AuditStage.PUBLIC, NOW)

    assert not project_staging.exists()


def test_create_project_rejects_manifest_with_later_public_version_before_staging(
    tmp_path: Path,
) -> None:
    first = version_fixture()
    second = version_fixture(version_number=2, stage=AuditStage.PUBLIC, audit_id="audit-2")
    manifest = project_fixture(
        versions=(first, second),
        latest_audit_id=second.audit_id,
    )

    with pytest.raises(RuntimeError, match="subsequent.*non-public"):
        ProjectStore(tmp_path).create_project(manifest)

    assert not (tmp_path / ".staging").exists()
    assert not (tmp_path / manifest.project_id).exists()


@pytest.mark.parametrize("entrypoint", ("create", "promote"))
def test_new_project_rejects_public_v1_with_source_audit_id(
    tmp_path: Path, entrypoint: str
) -> None:
    public = version_fixture(source_audit_id="older-audit")
    manifest = project_fixture(versions=(public,), latest_audit_id=public.audit_id)
    store = ProjectStore(tmp_path)

    with pytest.raises(RuntimeError, match="source_audit_id.*public-v1"):
        if entrypoint == "create":
            store.create_project(manifest)
        else:
            pending = store.begin_new_project(manifest.project_id)
            (pending.staging_path / public.relative_path).mkdir(parents=True)
            store.promote_new_project(pending, manifest)

    assert not (tmp_path / manifest.project_id).exists()


@pytest.mark.parametrize("source_audit_id", (None, "missing-audit"))
def test_later_version_rejects_missing_or_dangling_source_audit_id(
    tmp_path: Path, source_audit_id: str | None
) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    manifest_path = project_dir / "project.json"
    original_manifest = manifest_path.read_bytes()
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    invalid = completed_version(pending).model_copy(update={"source_audit_id": source_audit_id})

    with pytest.raises(RuntimeError, match="source_audit_id.*earlier audit_id"):
        store.promote(pending, invalid)

    assert pending.staging_path.is_dir()
    assert not pending.destination_path.exists()
    assert manifest_path.read_bytes() == original_manifest
    assert store.load(project.project_id) == project


def test_later_version_accepts_source_audit_id_from_earlier_version(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    version = completed_version(pending)

    promoted = store.promote(pending, version)

    assert promoted.versions[-1].source_audit_id == project.versions[0].audit_id


def test_promote_rejects_crafted_later_public_version(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    manifest_path = project_dir / "project.json"
    original_manifest = manifest_path.read_bytes()
    allocated = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    relative_path = "audits/2026-08-31_public-v2"
    pending = PendingVersion(
        project_id=allocated.project_id,
        run_id=allocated.run_id,
        version_number=allocated.version_number,
        stage=AuditStage.PUBLIC,
        created_at=allocated.created_at,
        version_id="public-v2",
        relative_path=relative_path,
        staging_path=allocated.staging_path,
        destination_path=project_dir / relative_path,
    )
    version = completed_version(pending)

    with pytest.raises(RuntimeError, match="subsequent.*non-public"):
        store.promote(pending, version)

    assert pending.staging_path.is_dir()
    assert not pending.destination_path.exists()
    assert manifest_path.read_bytes() == original_manifest
    assert store.load(project.project_id) == project


def test_promote_refuses_to_overwrite_existing_version_directory(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    pending.destination_path.mkdir(parents=True)
    marker = pending.destination_path / "existing.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        store.promote(pending, completed_version(pending))

    assert marker.read_text(encoding="utf-8") == "keep"
    assert [item.version_number for item in store.load(project.project_id).versions] == [1]


def test_later_version_stages_beneath_project_and_replaces_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    assert pending.staging_path.parent == tmp_path / project.project_id / ".staging"

    invalid = completed_version(pending).model_copy(update={"version_id": "context-v99"})
    replacements: list[tuple[Path, Path]] = []
    original_rename = getattr(project_store_module, "_rename_directory_no_replace", None)

    def recording_rename(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        assert original_rename is not None
        original_rename(source, destination)

    monkeypatch.setattr(
        project_store_module,
        "_rename_directory_no_replace",
        recording_rename,
        raising=False,
    )
    with pytest.raises(ValueError, match="does not match pending allocation"):
        store.promote(pending, invalid)

    assert pending.staging_path.exists()
    assert not pending.destination_path.exists()
    assert replacements == []

    promoted = store.promote(pending, completed_version(pending))

    assert promoted.versions[-1].version_id == "context-v2"
    assert not pending.staging_path.exists()
    assert pending.destination_path.is_dir()
    assert replacements[0] == (pending.staging_path, pending.destination_path)


def test_later_version_rejects_symlinked_pending_staging_root(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    manifest_path = project_dir / "project.json"
    original_manifest = manifest_path.read_bytes()
    public_root = project_dir / project.versions[0].relative_path
    public_marker = public_root / "keep.txt"
    public_marker.write_text("promoted", encoding="utf-8")
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    version = completed_version(pending)
    shutil.rmtree(pending.staging_path)
    pending.staging_path.symlink_to(public_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="store integrity.*pending version staging.*symlink"):
        store.promote(pending, version)

    assert pending.staging_path.is_symlink()
    assert not pending.destination_path.exists()
    assert manifest_path.read_bytes() == original_manifest
    assert store.load(project.project_id) == project
    assert public_marker.read_text(encoding="utf-8") == "promoted"


@pytest.mark.parametrize("nonempty_destination", (False, True))
def test_later_version_promotion_does_not_replace_boundary_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonempty_destination: bool,
) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    original_manifest = (project_dir / "project.json").read_bytes()
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    version = completed_version(pending)
    original_rename = getattr(project_store_module, "_rename_directory_no_replace", None)

    def collide_at_rename(source: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        if nonempty_destination:
            (destination / "existing.txt").write_text("keep", encoding="utf-8")
        assert original_rename is not None
        original_rename(source, destination)

    monkeypatch.setattr(
        project_store_module,
        "_rename_directory_no_replace",
        collide_at_rename,
        raising=False,
    )

    with pytest.raises(FileExistsError):
        store.promote(pending, version)

    assert (project_dir / "project.json").read_bytes() == original_manifest
    assert store.load(project.project_id) == project
    assert pending.staging_path.is_dir()
    expected_contents = ["existing.txt"] if nonempty_destination else []
    assert sorted(path.name for path in pending.destination_path.iterdir()) == expected_contents


def test_failed_version_build_does_not_advance_manifest(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())

    def fail_builder(pending: PendingVersion) -> AuditVersionRef:
        (pending.staging_path / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        store.build_version(project.project_id, stage=AuditStage.CONTEXT, builder=fail_builder)

    reloaded = store.load(project.project_id)
    assert [item.version_number for item in reloaded.versions] == [1]
    assert not (tmp_path / project.project_id / "audits" / "2026-08-31_context-v2").exists()


def test_failed_build_cleans_only_its_staging_directory(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    promoted_marker = project_dir / project.versions[0].relative_path / "keep.txt"
    promoted_marker.write_text("promoted", encoding="utf-8")
    unrelated_staging = project_dir / ".staging" / "another-run"
    unrelated_staging.mkdir(parents=True)
    (unrelated_staging / "keep.txt").write_text("unrelated", encoding="utf-8")
    owned_staging: Path | None = None

    def fail_builder(pending: PendingVersion) -> AuditVersionRef:
        nonlocal owned_staging
        owned_staging = pending.staging_path
        nested = pending.staging_path / "nested"
        nested.mkdir()
        (nested / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match="compile failed"):
        store.build_version(project.project_id, stage=AuditStage.CONTEXT, builder=fail_builder)

    assert owned_staging is not None and not owned_staging.exists()
    assert (unrelated_staging / "keep.txt").read_text(encoding="utf-8") == "unrelated"
    assert promoted_marker.read_text(encoding="utf-8") == "promoted"


def test_mismatched_version_build_cleans_staging_without_mutating_project(
    tmp_path: Path,
) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    manifest_path = project_dir / "project.json"
    original_manifest = manifest_path.read_bytes()
    promoted_marker = project_dir / project.versions[0].relative_path / "keep.txt"
    promoted_marker.write_text("promoted", encoding="utf-8")
    owned_staging: Path | None = None

    def mismatched_builder(pending: PendingVersion) -> AuditVersionRef:
        nonlocal owned_staging
        owned_staging = pending.staging_path
        version = completed_version(pending)
        return version.model_copy(update={"version_id": "context-v99"})

    with pytest.raises(ValueError, match="does not match pending allocation"):
        store.build_version(
            project.project_id,
            stage=AuditStage.CONTEXT,
            builder=mismatched_builder,
            now=NOW,
        )

    assert owned_staging is not None and not owned_staging.exists()
    assert manifest_path.read_bytes() == original_manifest
    assert store.load(project.project_id) == project
    assert promoted_marker.read_text(encoding="utf-8") == "promoted"
    assert not (project_dir / "audits" / "2026-08-31_context-v2").exists()


def test_manifest_write_failure_rolls_back_only_new_version_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ProjectStore(tmp_path)
    project = store.create_project(project_fixture())
    project_dir = tmp_path / project.project_id
    original_manifest = (project_dir / "project.json").read_bytes()
    pending = store.allocate_version(project.project_id, AuditStage.CONTEXT, NOW)
    version = completed_version(pending)
    real_replace = os.replace

    def fail_manifest_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == project_dir / "project.json":
            raise OSError("manifest replace failed")
        real_replace(source, destination)

    monkeypatch.setattr("ai_search_audit.project_store.os.replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replace failed"):
        store.promote(pending, version)

    assert (project_dir / "project.json").read_bytes() == original_manifest
    assert not pending.destination_path.exists()
    assert (project_dir / project.versions[0].relative_path).is_dir()


def test_load_validates_manifest_json_after_reading(tmp_path: Path) -> None:
    project_dir = tmp_path / "example-studio"
    project_dir.mkdir()
    (project_dir / "project.json").write_text('{"project_id": "example-studio"}', encoding="utf-8")

    with pytest.raises(ValidationError):
        ProjectStore(tmp_path).load("example-studio")
