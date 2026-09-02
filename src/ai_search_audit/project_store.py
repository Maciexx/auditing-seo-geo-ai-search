from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ai_search_audit.project_models import (
    AuditStage,
    AuditVersionRef,
    ProjectManifest,
    normalize_canonical_domain,
    validate_project_id,
)


class ProjectIdentityError(ValueError):
    """Raised when a project ID is already bound to another client identity."""


class ProjectStoreIntegrityError(RuntimeError):
    """Raised when persisted or staged project structure is not trustworthy."""


@dataclass(frozen=True, slots=True)
class PendingProject:
    project_id: str
    run_id: str
    staging_path: Path
    destination_path: Path


@dataclass(frozen=True, slots=True)
class PendingVersion:
    project_id: str
    run_id: str
    version_number: int
    stage: AuditStage
    created_at: datetime
    version_id: str
    relative_path: str
    staging_path: Path
    destination_path: Path


VersionBuilder = Callable[[PendingVersion], AuditVersionRef]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_COLLISION_ERRNOS = frozenset({errno.EEXIST, errno.ENOTEMPTY})
_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }
)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory, failing if the destination exists."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renamex_np = libc.renamex_np
        except AttributeError as exc:
            raise RuntimeError("atomic no-replace rename is unsupported on this macOS") from exc
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
        primitive = "renamex_np(RENAME_EXCL)"
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise RuntimeError("atomic no-replace rename is unsupported on this Linux") from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        primitive = "renameat2(RENAME_NOREPLACE)"
    elif sys.platform == "win32":
        try:
            os.rename(source, destination)
        except OSError as exc:
            _raise_directory_rename_error(exc.errno, destination, "os.rename")
        return
    else:
        raise RuntimeError(
            f"atomic no-replace directory rename is unsupported on platform {sys.platform!r}"
        )

    if result != 0:
        _raise_directory_rename_error(ctypes.get_errno(), destination, primitive)


def _raise_directory_rename_error(
    error_code: int | None,
    destination: Path,
    primitive: str,
) -> None:
    if error_code in _COLLISION_ERRNOS:
        raise FileExistsError(
            error_code,
            f"destination already exists: {destination}",
            destination,
        )
    if error_code in _UNSUPPORTED_ERRNOS:
        raise RuntimeError(f"atomic no-replace directory rename via {primitive} is unsupported")
    if error_code is None:
        raise OSError(f"atomic directory rename via {primitive} failed")
    raise OSError(error_code, os.strerror(error_code), destination)


class ProjectStore:
    """Single-writer, sequential filesystem persistence for local audit projects.

    Callers must coordinate writes so project mutations never overlap.
    """

    def __init__(self, clients_root: str | Path) -> None:
        self.clients_root = Path(clients_root).resolve()

    def create_project(self, manifest: ProjectManifest) -> ProjectManifest:
        validated = self._validate_manifest(manifest)
        destination = self._project_path(validated.project_id)
        if os.path.lexists(destination):
            existing = self.load(validated.project_id)
            if (
                existing.client_name != validated.client_name
                or existing.canonical_domains != validated.canonical_domains
            ):
                raise ProjectIdentityError(
                    f"project ID {validated.project_id!r} belongs to a different project identity"
                )
            raise FileExistsError(f"project {validated.project_id!r} already exists")

        pending = self.begin_new_project(validated.project_id)
        try:
            for version in validated.versions:
                version_path = pending.staging_path / version.relative_path
                self._require_within(version_path, pending.staging_path)
                version_path.mkdir(parents=True, exist_ok=False)
            return self.promote_new_project(pending, validated)
        except Exception:
            self._cleanup_owned_path(pending.staging_path, pending.staging_path.parent)
            raise

    def load(self, project_id: str) -> ProjectManifest:
        project_id = self._validate_project_id(project_id)
        project_path = self._project_path(project_id)
        if not os.path.lexists(project_path):
            raise FileNotFoundError(f"project {project_id!r} does not exist")
        self._require_real_directory(project_path, role="project root")

        manifest_path = project_path / "project.json"
        self._require_within(manifest_path, project_path)
        manifest = ProjectManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self._validate_source_linkage(manifest)
        if manifest.project_id != project_id:
            raise ValueError("project manifest ID does not match its directory")
        self._require_version_directories(manifest, project_path)
        return manifest

    def resolve(self, reference: str) -> Path:
        prefix = "project:"
        if not reference.startswith(prefix):
            raise ValueError("project reference must use project:<id>")
        project_id = self._validate_project_id(reference.removeprefix(prefix))
        return self._project_path(project_id)

    def assert_identity(
        self,
        project_id: str,
        *,
        domain: str,
        client_name: str | None,
    ) -> None:
        manifest = self.load(project_id)
        canonical_domain = normalize_canonical_domain(domain)
        domain_matches = canonical_domain in manifest.canonical_domains
        client_matches = client_name is None or client_name == manifest.client_name
        if not domain_matches or not client_matches:
            raise ProjectIdentityError(
                f"project ID {project_id!r} belongs to a different project identity"
            )

    def begin_new_project(self, project_id: str) -> PendingProject:
        project_id = self._validate_project_id(project_id)
        destination = self._project_path(project_id)
        if os.path.lexists(destination):
            raise FileExistsError(f"project {project_id!r} already exists")

        staging_root = self.clients_root / ".staging"
        if staging_root.is_symlink():
            self._raise_symlink_integrity_error(staging_root, role="new-project staging root")
        self._require_within(staging_root, self.clients_root)
        staging_root.mkdir(parents=True, exist_ok=True)
        self._require_real_directory(staging_root, role="new-project staging root")
        run_id = uuid4().hex
        staging_path = staging_root / f"{project_id}-{run_id}"
        self._require_within(staging_path, staging_root)
        staging_path.mkdir()
        self._require_real_directory(staging_path, role="pending project staging root")
        return PendingProject(
            project_id=project_id,
            run_id=run_id,
            staging_path=staging_path,
            destination_path=destination,
        )

    def promote_new_project(
        self,
        pending: PendingProject,
        manifest: ProjectManifest,
    ) -> ProjectManifest:
        self._validate_pending_project(pending)
        validated = ProjectManifest.model_validate(manifest.model_dump(mode="python"))
        if validated.project_id != pending.project_id:
            raise ValueError("manifest project ID does not match pending project")
        if len(validated.versions) != 1 or not self._is_public_v1(validated.versions[0]):
            raise ValueError("a new project must contain exactly public-v1")
        self._validate_source_linkage(validated)
        self._require_version_directories(validated, pending.staging_path)
        self._atomic_write_manifest(pending.staging_path, validated)
        _rename_directory_no_replace(pending.staging_path, pending.destination_path)
        return validated

    def discard_pending_project(self, pending: PendingProject) -> None:
        """Remove one pending new-project root after validating store ownership."""
        expected_destination = self._project_path(pending.project_id)
        expected_parent = self.clients_root / ".staging"
        if (
            pending.destination_path != expected_destination
            or pending.staging_path.parent != expected_parent
            or pending.staging_path.name != f"{pending.project_id}-{pending.run_id}"
        ):
            raise ValueError("pending project does not belong to this store")
        self._require_real_directory(expected_parent, role="new-project staging root")
        self._cleanup_owned_path(pending.staging_path, expected_parent)

    def allocate_version(
        self,
        project_id: str,
        stage: AuditStage,
        now: datetime,
    ) -> PendingVersion:
        project_id = self._validate_project_id(project_id)
        stage = AuditStage(stage)
        manifest = self.load(project_id)
        if stage is AuditStage.PUBLIC:
            raise ValueError("public stage is only valid for the initial public-v1")
        project_path = self._project_path(project_id)
        version_number = manifest.versions[-1].version_number + 1
        version_id = f"{stage.value}-v{version_number}"
        relative_path = f"audits/{now.date().isoformat()}_{version_id}"
        destination_path = project_path / relative_path
        self._require_within(destination_path, project_path)
        if os.path.lexists(destination_path):
            raise FileExistsError(f"version destination {relative_path!r} already exists")

        staging_root = project_path / ".staging"
        if staging_root.is_symlink():
            self._raise_symlink_integrity_error(staging_root, role="pending version staging root")
        self._require_within(staging_root, project_path)
        staging_root.mkdir(parents=True, exist_ok=True)
        self._require_real_directory(staging_root, role="pending version staging root")
        run_id = uuid4().hex
        staging_path = staging_root / run_id
        self._require_within(staging_path, staging_root)
        staging_path.mkdir()
        self._require_real_directory(staging_path, role="pending version staging root")
        return PendingVersion(
            project_id=project_id,
            run_id=run_id,
            version_number=version_number,
            stage=stage,
            created_at=now,
            version_id=version_id,
            relative_path=relative_path,
            staging_path=staging_path,
            destination_path=destination_path,
        )

    def promote(
        self,
        pending: PendingVersion,
        version: AuditVersionRef,
    ) -> ProjectManifest:
        self._validate_pending_version(pending)
        validated_version = AuditVersionRef.model_validate(version.model_dump(mode="python"))
        expected_fields = (
            "version_id",
            "version_number",
            "stage",
            "created_at",
            "relative_path",
        )
        if any(
            getattr(validated_version, field_name) != getattr(pending, field_name)
            for field_name in expected_fields
        ):
            raise ValueError("version does not match pending allocation")
        if not pending.staging_path.is_dir():
            raise FileNotFoundError("pending version staging directory does not exist")
        current = self.load(pending.project_id)
        if pending.version_number != current.versions[-1].version_number + 1:
            raise ValueError("pending version is no longer the next manifest version")
        candidate = ProjectManifest.model_validate(
            {
                **current.model_dump(mode="python"),
                "latest_audit_id": validated_version.audit_id,
                "versions": (*current.versions, validated_version),
            }
        )
        self._validate_source_linkage(candidate)

        pending.destination_path.parent.mkdir(parents=True, exist_ok=True)
        _rename_directory_no_replace(pending.staging_path, pending.destination_path)
        try:
            self._atomic_write_manifest(self._project_path(pending.project_id), candidate)
        except Exception:
            self._cleanup_owned_path(pending.destination_path, pending.destination_path.parent)
            raise
        return candidate

    def build_version(
        self,
        project_id: str,
        *,
        stage: AuditStage,
        builder: VersionBuilder,
        now: datetime | None = None,
    ) -> ProjectManifest:
        pending = self.allocate_version(project_id, stage, now or datetime.now(UTC))
        try:
            version = builder(pending)
            return self.promote(pending, version)
        except Exception:
            self._cleanup_owned_path(pending.staging_path, pending.staging_path.parent)
            raise

    def _project_path(self, project_id: str) -> Path:
        candidate = self.clients_root / project_id
        if candidate.is_symlink():
            self._raise_symlink_integrity_error(candidate, role="project root")
        self._require_within(candidate, self.clients_root)
        return candidate

    @staticmethod
    def _validate_project_id(project_id: str) -> str:
        return validate_project_id(project_id)

    def _validate_pending_project(self, pending: PendingProject) -> None:
        expected_destination = self._project_path(pending.project_id)
        expected_parent = self.clients_root / ".staging"
        if (
            pending.destination_path != expected_destination
            or pending.staging_path.parent != expected_parent
        ):
            raise ValueError("pending project does not belong to this store")
        if pending.staging_path.name != f"{pending.project_id}-{pending.run_id}":
            raise ValueError("pending project path does not match its run ID")
        self._require_real_directory(pending.staging_path, role="pending project staging root")
        self._require_within(pending.staging_path, expected_parent)

    def _validate_pending_version(self, pending: PendingVersion) -> None:
        project_path = self._project_path(pending.project_id)
        if pending.staging_path.parent != project_path / ".staging":
            raise ValueError("pending version does not belong to this project")
        if pending.staging_path.name != pending.run_id:
            raise ValueError("pending version path does not match its run ID")
        expected_destination = project_path / pending.relative_path
        if pending.destination_path != expected_destination:
            raise ValueError("pending version destination does not match its relative path")
        self._require_real_directory(pending.staging_path, role="pending version staging root")
        self._require_within(pending.staging_path, project_path)
        self._require_within(pending.destination_path, project_path)

    def _require_version_directories(self, manifest: ProjectManifest, project_path: Path) -> None:
        for version in manifest.versions:
            version_path = project_path / version.relative_path
            self._require_real_directory_tree(
                version_path,
                trusted_root=project_path,
                role="version root",
            )
            self._require_within(version_path, project_path)

    @staticmethod
    def _is_public_v1(version: AuditVersionRef) -> bool:
        expected_relative_path = f"audits/{version.created_at.date().isoformat()}_public-v1"
        return (
            version.version_number == 1
            and version.version_id == "public-v1"
            and version.stage is AuditStage.PUBLIC
            and version.relative_path == expected_relative_path
        )

    @staticmethod
    def _validate_manifest(manifest: ProjectManifest) -> ProjectManifest:
        validated = ProjectManifest.model_validate(manifest.model_dump(mode="python"))
        ProjectStore._validate_source_linkage(validated)
        return validated

    @staticmethod
    def _validate_source_linkage(manifest: ProjectManifest) -> None:
        first = manifest.versions[0]
        if not ProjectStore._is_public_v1(first):
            raise ProjectStoreIntegrityError(
                "project store integrity error: first history entry must be exactly "
                "public-v1 in the public stage"
            )
        if first.source_audit_id is not None:
            raise ProjectStoreIntegrityError(
                "project store integrity error: source_audit_id must be None for public-v1"
            )

        earlier_audit_ids = {first.audit_id}
        for version in manifest.versions[1:]:
            if version.stage is AuditStage.PUBLIC:
                raise ProjectStoreIntegrityError(
                    "project store integrity error: subsequent history entries must be non-public"
                )
            if version.source_audit_id not in earlier_audit_ids:
                raise ProjectStoreIntegrityError(
                    "project store integrity error: source_audit_id for "
                    f"{version.version_id} must reference an earlier audit_id"
                )
            earlier_audit_ids.add(version.audit_id)

    @staticmethod
    def _require_real_directory_tree(
        path: Path,
        *,
        trusted_root: Path,
        role: str,
    ) -> None:
        try:
            relative_path = path.relative_to(trusted_root)
        except ValueError as exc:
            raise ValueError(f"{path} is not beneath trusted root {trusted_root}") from exc

        current = trusted_root
        final_index = len(relative_path.parts) - 1
        for index, component in enumerate(relative_path.parts):
            current /= component
            component_role = role if index == final_index else f"{role} ancestor {component!r}"
            ProjectStore._require_real_directory(current, role=component_role)

    @staticmethod
    def _require_real_directory(path: Path, *, role: str) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ProjectStoreIntegrityError(
                f"project store integrity error: {role} is missing: {path}"
            ) from exc
        if stat.S_ISLNK(mode):
            ProjectStore._raise_symlink_integrity_error(path, role=role)
        if not stat.S_ISDIR(mode):
            raise ProjectStoreIntegrityError(
                f"project store integrity error: {role} is a file, not a real directory: {path}"
            )

    @staticmethod
    def _raise_symlink_integrity_error(path: Path, *, role: str) -> None:
        raise ProjectStoreIntegrityError(
            f"project store integrity error: {role} must not be a symlink: {path}"
        )

    @staticmethod
    def _atomic_write_manifest(directory: Path, manifest: ProjectManifest) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=directory,
            prefix=".project.json.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(manifest.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, directory / "project.json")
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _cleanup_owned_path(path: Path, expected_parent: Path) -> None:
        if path.parent != expected_parent:
            raise ValueError("refusing to clean a path outside the operation staging area")
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    @staticmethod
    def _require_within(path: Path, parent: Path) -> None:
        resolved_parent = parent.resolve()
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(resolved_parent):
            raise ValueError(f"path {path} resolves outside clients root/project")
