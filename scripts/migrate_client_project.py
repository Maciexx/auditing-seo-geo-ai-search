#!/usr/bin/env python3
"""Assemble historical reports around one validated operational public-v1 project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, NamedTuple
from uuid import uuid4

_REPOSITORY_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_SRC))

from ai_search_audit.project_models import (  # noqa: E402
    AuditStage,
    ProjectManifest,
    ReportStatus,
    normalize_canonical_domain,
    validate_project_id,
)
from ai_search_audit.project_orchestrator import validate_project_bundle  # noqa: E402
from ai_search_audit.project_store import ProjectStore, _rename_directory_no_replace  # noqa: E402

_MANIFEST_SCHEMA_VERSION = "1.0.0"
_IMPORTED_ROOT = PurePosixPath("legacy/imported-public-audit")
_IMPORTED_MANIFEST = _IMPORTED_ROOT / "manifest.json"


class SourceRecord(NamedTuple):
    source: Path
    role: Literal["report", "legacy"]
    relative_path: str
    content: bytes
    sha256: str
    byte_count: int
    device: int
    inode: int
    mtime_ns: int
    existing_relative_path: str | None = None


class TreeInventory(NamedTuple):
    files: tuple[tuple[str, str, int], ...]
    directories: tuple[str, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients-root", type=Path, required=True)
    parser.add_argument("--canonical-project", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--locale", choices=("pl", "en"), required=True)
    parser.add_argument("--report-file", action="append", type=Path, required=True)
    parser.add_argument("--legacy-file", action="append", type=Path, default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--move-after-verify", action="store_true")
    return parser


def _reject_symlink_or_alias_components(path: Path, *, role: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{role} must not contain parent-directory aliases")
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            details = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"{role} must not contain a symlink: {current}")


def _read_verified_source(
    path: Path,
    *,
    role: Literal["report", "legacy"],
    destination: Path,
) -> SourceRecord:
    _reject_symlink_or_alias_components(path, role="migration source")
    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."} or PurePosixPath(absolute.name).name != absolute.name:
        raise ValueError(f"migration source has an invalid filename: {path}")
    try:
        path_state = absolute.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"migration source does not exist: {path}") from exc
    if not stat.S_ISREG(path_state.st_mode):
        raise ValueError(f"migration source must be a regular file, not a symlink: {path}")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            path_state.st_dev,
            path_state.st_ino,
        ):
            raise RuntimeError(f"migration source changed before verified read: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"migration source changed during verified read: {path}")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise RuntimeError(f"migration source size changed during verified read: {path}")
    existing_relative = (
        absolute.relative_to(destination).as_posix()
        if absolute.is_relative_to(destination)
        else None
    )
    return SourceRecord(
        source=absolute,
        role=role,
        relative_path=(_IMPORTED_ROOT / absolute.name).as_posix(),
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
        existing_relative_path=existing_relative,
    )


def _source_records(
    *, report_files: Sequence[Path], legacy_files: Sequence[Path], destination: Path
) -> tuple[SourceRecord, ...]:
    records: list[SourceRecord] = []
    seen_sources: set[Path] = set()
    seen_destinations = {unicodedata.normalize("NFC", _IMPORTED_MANIFEST.as_posix()).casefold()}
    for role, paths in (("report", report_files), ("legacy", legacy_files)):
        typed_role: Literal["report", "legacy"] = "report" if role == "report" else "legacy"
        for path in paths:
            record = _read_verified_source(path, role=typed_role, destination=destination)
            if record.source in seen_sources:
                raise ValueError(
                    f"migration source was supplied more than once: {record.source.name}"
                )
            destination_key = unicodedata.normalize("NFC", record.relative_path).casefold()
            if destination_key in seen_destinations:
                raise ValueError(
                    "migration destination has a case-insensitive Unicode-normalized "
                    f"collision: {record.source.name}"
                )
            records.append(record)
            seen_sources.add(record.source)
            seen_destinations.add(destination_key)
    return tuple(records)


def _public_record(record: SourceRecord) -> dict[str, object]:
    return {
        "byte_count": record.byte_count,
        "relative_path": record.relative_path,
        "role": record.role,
        "sha256": record.sha256,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _walk_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            entry = current_path / name
            if stat.S_ISLNK(entry.lstat().st_mode):
                raise ValueError(f"project tree must not contain a symlink: {entry}")
        relative = current_path.relative_to(root)
        directories.update((relative / name).as_posix() for name in directory_names)
        files.update((relative / name).as_posix() for name in file_names)
    return files, directories


def _capture_tree_inventory(root: Path) -> TreeInventory:
    file_paths, directories = _walk_tree(root)
    files: list[tuple[str, str, int]] = []
    for relative in sorted(file_paths):
        content = (root / relative).read_bytes()
        files.append((relative, hashlib.sha256(content).hexdigest(), len(content)))
    return TreeInventory(files=tuple(files), directories=tuple(sorted(directories)))


def _validate_operational_project(
    project_root: Path,
    *,
    project_id: str,
    client_name: str,
    domain: str,
    locale: Literal["pl", "en"],
    expected_manifest: ProjectManifest | None = None,
) -> ProjectManifest:
    _reject_symlink_or_alias_components(project_root, role="canonical project")
    if project_root.name != project_id:
        raise ValueError("canonical project directory must match project ID")
    manifest = ProjectStore(project_root.parent).load(project_id)
    if (
        manifest.client_name != client_name
        or manifest.canonical_domains != (domain,)
        or manifest.report_locale != locale
    ):
        raise ValueError("canonical project identity does not match migration arguments")
    if expected_manifest is not None and manifest != expected_manifest:
        raise RuntimeError("assembled project manifest differs from canonical public-v1")
    if len(manifest.versions) != 1:
        raise ValueError("canonical project must contain exactly one public-v1 version")
    version = manifest.versions[0]
    if (
        version.version_id != "public-v1"
        or version.version_number != 1
        or version.stage is not AuditStage.PUBLIC
        or version.report_status is not ReportStatus.PUBLIC_EVIDENCE_DRAFT
        or version.source_audit_id is not None
    ):
        raise ValueError("canonical project must contain exactly one operational public-v1")
    snapshot = validate_project_bundle(
        project_root / version.relative_path,
        expected_project_id=project_id,
        expected_version_number=1,
        expected_source_audit_id=None,
        expected_stage=AuditStage.PUBLIC,
    )
    engine_audit = json.loads(snapshot.files["engine/audit.json"].content)
    if engine_audit.get("audit_id") != version.audit_id:
        raise ValueError("canonical project manifest audit identity does not match engine audit")
    return manifest


def _validate_canonical_inventory(project_root: Path, manifest: ProjectManifest) -> TreeInventory:
    inventory = _capture_tree_inventory(project_root)
    files = {path for path, _sha256, _size in inventory.files}
    directories = set(inventory.directories)
    version_path = manifest.versions[0].relative_path
    if "project.json" not in files:
        raise ValueError("canonical project is missing project.json")
    if any(path != "project.json" and not path.startswith(f"{version_path}/") for path in files):
        raise ValueError("canonical project contains content outside public-v1")
    allowed = {"audits", version_path}
    if any(path not in allowed and not path.startswith(f"{version_path}/") for path in directories):
        raise ValueError("canonical project contains directories outside public-v1")
    return inventory


def _copy_canonical_project(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)


def _legacy_manifest(
    manifest: ProjectManifest, records: Sequence[SourceRecord]
) -> dict[str, object]:
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "project_id": manifest.project_id,
        "audit_id": manifest.versions[0].audit_id,
        "version_id": manifest.versions[0].version_id,
        "files": [_public_record(record) for record in records],
    }


def _add_legacy_import(
    project_root: Path, *, manifest: ProjectManifest, records: Sequence[SourceRecord]
) -> None:
    imported_root = project_root / _IMPORTED_ROOT
    imported_root.mkdir(parents=True, exist_ok=False)
    for record in records:
        destination = project_root / record.relative_path
        destination.write_bytes(record.content)
        os.utime(destination, ns=(record.mtime_ns, record.mtime_ns))
    _write_json(project_root / _IMPORTED_MANIFEST, _legacy_manifest(manifest, records))


def _validate_legacy_import(
    project_root: Path, *, expected_manifest: ProjectManifest, records: Sequence[SourceRecord]
) -> None:
    imported_root = project_root / _IMPORTED_ROOT
    expected_names = {Path(record.relative_path).name for record in records} | {"manifest.json"}
    if (
        not imported_root.is_dir()
        or {entry.name for entry in imported_root.iterdir()} != expected_names
    ):
        raise RuntimeError("legacy imported-public-audit inventory does not match migration plan")
    manifest_payload = json.loads((project_root / _IMPORTED_MANIFEST).read_text(encoding="utf-8"))
    if manifest_payload != _legacy_manifest(expected_manifest, records):
        raise RuntimeError("legacy hash manifest does not match migration plan")
    for record in records:
        path = project_root / record.relative_path
        details = path.lstat()
        content = path.read_bytes()
        if (
            not stat.S_ISREG(details.st_mode)
            or len(content) != record.byte_count
            or hashlib.sha256(content).hexdigest() != record.sha256
        ):
            raise RuntimeError(f"legacy artifact hash mismatch: {record.relative_path}")


def _validate_assembled_project(
    project_root: Path,
    *,
    expected_manifest: ProjectManifest,
    canonical_inventory: TreeInventory,
    records: Sequence[SourceRecord],
) -> None:
    _validate_operational_project(
        project_root,
        project_id=expected_manifest.project_id,
        client_name=expected_manifest.client_name,
        domain=expected_manifest.canonical_domains[0],
        locale=expected_manifest.report_locale,
        expected_manifest=expected_manifest,
    )
    assembled = _capture_tree_inventory(project_root)
    expected_files = (
        {path for path, _sha256, _size in canonical_inventory.files}
        | {record.relative_path for record in records}
        | {_IMPORTED_MANIFEST.as_posix()}
    )
    expected_directories = set(canonical_inventory.directories) | {
        "legacy",
        _IMPORTED_ROOT.as_posix(),
    }
    if {path for path, _sha256, _size in assembled.files} != expected_files or set(
        assembled.directories
    ) != expected_directories:
        raise RuntimeError("assembled project differs from exact canonical inventory")
    canonical_files = {path: (sha256, size) for path, sha256, size in canonical_inventory.files}
    assembled_files = {path: (sha256, size) for path, sha256, size in assembled.files}
    if any(assembled_files[path] != expected for path, expected in canonical_files.items()):
        raise RuntimeError("assembled project canonical inventory content changed")
    _validate_legacy_import(project_root, expected_manifest=expected_manifest, records=records)


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("simplified migration manifest contains an invalid path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise ValueError("simplified migration manifest path must be relative POSIX")
    return value


def _expected_parent_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _validate_existing_simplified_import(
    destination: Path,
    *,
    project_id: str,
    client_name: str,
    domain: str,
    locale: Literal["pl", "en"],
    records: Sequence[SourceRecord],
) -> None:
    manifest = ProjectManifest.model_validate_json(
        (destination / "project.json").read_text(encoding="utf-8")
    )
    if (
        manifest.project_id != project_id
        or manifest.client_name != client_name
        or manifest.canonical_domains != (domain,)
        or manifest.report_locale != locale
        or len(manifest.versions) != 1
    ):
        raise FileExistsError("existing destination is not a simplified imported public audit")
    version = manifest.versions[0]
    version_path = PurePosixPath(version.relative_path)
    if (
        version.version_id != "public-v1"
        or version.version_number != 1
        or version.stage is not AuditStage.PUBLIC
        or version.report_status is not ReportStatus.PUBLIC_EVIDENCE_DRAFT
        or version.source_audit_id is not None
        or manifest.latest_audit_id != version.audit_id
        or len(version_path.parts) != 2
        or version_path.parts[0] != "audits"
        or not version_path.parts[1].endswith("_public-v1")
    ):
        raise FileExistsError("existing destination is not a valid simplified public-v1")
    manifest_root = destination / version.relative_path / "manifests"
    input_payload = json.loads((manifest_root / "input-manifest.json").read_text())
    output_payload = json.loads((manifest_root / "output-manifest.json").read_text())
    if (
        input_payload != output_payload
        or output_payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or output_payload.get("project_id") != project_id
        or output_payload.get("version_id") != version.version_id
        or not isinstance(output_payload.get("files"), list)
    ):
        raise FileExistsError("existing simplified migration manifests do not match")
    recorded_paths: set[str] = set()
    for item in output_payload["files"]:
        if not isinstance(item, dict):
            raise FileExistsError("existing simplified migration record is invalid")
        relative = _safe_relative_path(item.get("relative_path"))
        path = destination / relative
        details = path.lstat()
        content = path.read_bytes()
        if (
            not stat.S_ISREG(details.st_mode)
            or item.get("byte_count") != len(content)
            or item.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise FileExistsError("existing simplified migration hash validation failed")
        recorded_paths.add(relative)
    supplied_existing = {
        record.existing_relative_path
        for record in records
        if record.existing_relative_path is not None
    }
    if supplied_existing != recorded_paths:
        raise FileExistsError("existing simplified migration sources do not match its manifest")
    expected_files = recorded_paths | {
        "project.json",
        f"{version.relative_path}/manifests/input-manifest.json",
        f"{version.relative_path}/manifests/output-manifest.json",
    }
    files, directories = _walk_tree(destination)
    if files != expected_files or directories != _expected_parent_directories(expected_files):
        raise FileExistsError("existing simplified migration contains unlisted content")


def _plan_payload(
    manifest: ProjectManifest,
    records: Sequence[SourceRecord],
    *,
    dry_run: bool,
    promoted: bool,
    sources_removed: bool,
) -> dict[str, object]:
    return {
        "dry_run": dry_run,
        "files": [_public_record(record) for record in records],
        "project_manifest": manifest.model_dump(mode="json"),
        "promoted": promoted,
        "sources_removed": sources_removed,
    }


def _expected_identity(record: SourceRecord) -> tuple[int, int, int, int]:
    return (record.device, record.inode, record.byte_count, record.mtime_ns)


def _revalidate_sources_before_removal(
    records: Sequence[SourceRecord], *, backup: Path | None
) -> None:
    for record in records:
        path = record.source
        if record.existing_relative_path is not None:
            if backup is None:
                raise RuntimeError("existing source backup is unavailable for identity validation")
            path = backup / record.existing_relative_path
        try:
            details = path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"migration source identity changed before removal: {path}") from exc
        observed = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)
        if not stat.S_ISREG(details.st_mode) or observed != _expected_identity(record):
            raise RuntimeError(f"migration source identity changed before removal: {path}")


def _verify_quarantined_source(record: SourceRecord, quarantine: Path) -> None:
    details = quarantine.lstat()
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeError("quarantined migration source is not a regular file")
    descriptor = os.open(quarantine, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    content = b"".join(chunks)
    if (
        identity != _expected_identity(record)
        or after_identity != identity
        or (details.st_dev, details.st_ino) != (opened.st_dev, opened.st_ino)
        or len(content) != record.byte_count
        or hashlib.sha256(content).hexdigest() != record.sha256
    ):
        raise RuntimeError("quarantined migration source identity or content changed")


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename one filesystem entry without replacing the destination."""
    _rename_directory_no_replace(source, destination)


def _atomic_quarantine_source(record: SourceRecord) -> Path:
    quarantine = record.source.parent / (
        f".{record.source.name}.migration-quarantine-{uuid4().hex}"
    )
    try:
        _rename_no_replace(record.source, quarantine)
    except BaseException as exc:
        raise RuntimeError(
            f"migration source quarantine collision; recovery paths: {record.source}, {quarantine}"
        ) from exc
    try:
        _verify_quarantined_source(record, quarantine)
    except BaseException as exc:
        try:
            _rename_no_replace(quarantine, record.source)
        except BaseException as restore_exc:
            raise RuntimeError(
                "migration source changed before atomic quarantine and restore collided; "
                f"recovery paths: {record.source}, {quarantine}"
            ) from restore_exc
        raise RuntimeError(
            "migration source changed before atomic quarantine; replacement restored at: "
            f"{record.source}"
        ) from exc
    return quarantine


def _restore_quarantined_sources(
    quarantined: Sequence[tuple[SourceRecord, Path]],
) -> list[Path]:
    recovery_paths: list[Path] = []
    for record, quarantine in reversed(quarantined):
        if not os.path.lexists(quarantine):
            continue
        try:
            _rename_no_replace(quarantine, record.source)
        except BaseException:
            recovery_paths.extend((record.source, quarantine))
        else:
            recovery_paths.append(record.source)
    return recovery_paths


def _remove_verified_sources(
    records: Sequence[SourceRecord],
    *,
    backup: Path | None,
    backup_validator: Callable[[], None] | None,
) -> None:
    _revalidate_sources_before_removal(records, backup=backup)
    external = tuple(record for record in records if record.existing_relative_path is None)
    quarantined: list[tuple[SourceRecord, Path]] = []
    try:
        for record in external:
            quarantined.append((record, _atomic_quarantine_source(record)))
        if backup is not None:
            if backup_validator is None:
                raise RuntimeError(f"backup validation is unavailable; recovery path: {backup}")
            backup_validator()
            shutil.rmtree(backup)
        for _record, quarantine in quarantined:
            quarantine.unlink()
    except BaseException as exc:
        recovery_paths = _restore_quarantined_sources(quarantined)
        if backup is not None and backup.exists():
            recovery_paths.append(backup)
        detail = ", ".join(str(path) for path in recovery_paths) or "none"
        raise RuntimeError(
            f"migration source removal failed: {exc}; recovery paths: {detail}"
        ) from exc


def _reject_staging_symlink(staging_parent: Path) -> None:
    if not os.path.lexists(staging_parent):
        return
    details = staging_parent.lstat()
    if stat.S_ISLNK(details.st_mode):
        raise ValueError("migration staging directory must not be a symlink")
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError("migration staging path must be a directory")


def _restore_backup_or_raise(backup: Path, destination: Path) -> None:
    try:
        _rename_no_replace(backup, destination)
    except BaseException as exc:
        raise RuntimeError(
            f"validated original project could not be restored; exact recovery path: {backup}"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _reject_symlink_or_alias_components(args.clients_root, role="clients root")
    clients_root = Path(os.path.abspath(args.clients_root))
    project_id = validate_project_id(args.project_id)
    domain = normalize_canonical_domain(args.domain)
    locale: Literal["pl", "en"] = "pl" if args.locale == "pl" else "en"
    destination = clients_root / project_id
    _reject_symlink_or_alias_components(destination, role="project destination")
    canonical_project = Path(os.path.abspath(args.canonical_project))
    _reject_symlink_or_alias_components(canonical_project, role="canonical project")
    if (
        canonical_project == destination
        or canonical_project.is_relative_to(destination)
        or destination.is_relative_to(canonical_project)
    ):
        raise ValueError("canonical project must be separate from the migration destination")
    manifest = _validate_operational_project(
        canonical_project,
        project_id=project_id,
        client_name=args.client_name,
        domain=domain,
        locale=locale,
    )
    canonical_inventory = _validate_canonical_inventory(canonical_project, manifest)
    records = _source_records(
        report_files=args.report_file,
        legacy_files=args.legacy_file,
        destination=destination,
    )
    staging_parent = clients_root / ".migration-staging"
    _reject_staging_symlink(staging_parent)

    destination_exists = os.path.lexists(destination)
    if destination_exists:
        details = destination.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("project destination must not be a symlink")
        if not stat.S_ISDIR(details.st_mode):
            raise FileExistsError("project destination must be a directory")
        _validate_existing_simplified_import(
            destination,
            project_id=project_id,
            client_name=args.client_name,
            domain=domain,
            locale=locale,
            records=records,
        )
        if not args.dry_run and not args.move_after_verify:
            raise FileExistsError(
                "existing simplified import requires --move-after-verify for replacement"
            )

    if args.dry_run:
        print(
            json.dumps(
                _plan_payload(
                    manifest,
                    records,
                    dry_run=True,
                    promoted=False,
                    sources_removed=False,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    clients_root.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(exist_ok=True)
    _reject_staging_symlink(staging_parent)
    operation_root = Path(tempfile.mkdtemp(prefix=f"{project_id}-", dir=staging_parent))
    staged_project = operation_root / project_id
    backup = operation_root / "original"
    original_swapped = False
    preserve_operation = False

    def validate_backup() -> None:
        _validate_existing_simplified_import(
            backup,
            project_id=project_id,
            client_name=args.client_name,
            domain=domain,
            locale=locale,
            records=records,
        )

    try:
        _copy_canonical_project(canonical_project, staged_project)
        _add_legacy_import(staged_project, manifest=manifest, records=records)
        _validate_assembled_project(
            staged_project,
            expected_manifest=manifest,
            canonical_inventory=canonical_inventory,
            records=records,
        )
        if destination_exists:
            _rename_directory_no_replace(destination, backup)
            original_swapped = True
            validate_backup()
        try:
            _rename_directory_no_replace(staged_project, destination)
        except BaseException as exc:
            if original_swapped:
                if not os.path.lexists(destination):
                    try:
                        _restore_backup_or_raise(backup, destination)
                    except BaseException:
                        preserve_operation = True
                        raise
                    original_swapped = False
                else:
                    shutil.rmtree(staged_project)
                    preserve_operation = True
                    raise RuntimeError(
                        "promotion race blocked rollback; validated original project recovery "
                        f"path: {backup}"
                    ) from exc
            raise
        try:
            _validate_assembled_project(
                destination,
                expected_manifest=manifest,
                canonical_inventory=canonical_inventory,
                records=records,
            )
        except BaseException as exc:
            failed = operation_root / "failed-promotion"
            try:
                _rename_directory_no_replace(destination, failed)
                if original_swapped:
                    _restore_backup_or_raise(backup, destination)
                    original_swapped = False
            except BaseException as rollback_exc:
                preserve_operation = True
                raise RuntimeError(
                    "promoted project validation failed and rollback was incomplete; validated "
                    f"original project recovery path: {backup}"
                ) from rollback_exc
            raise exc
        if args.move_after_verify:
            _remove_verified_sources(
                records,
                backup=backup if original_swapped else None,
                backup_validator=validate_backup if original_swapped else None,
            )
            original_swapped = False
        print(
            json.dumps(
                _plan_payload(
                    manifest,
                    records,
                    dry_run=False,
                    promoted=True,
                    sources_removed=args.move_after_verify,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if original_swapped and not os.path.lexists(destination) and backup.exists():
            try:
                _restore_backup_or_raise(backup, destination)
                original_swapped = False
            except BaseException:
                preserve_operation = True
                raise
        if not preserve_operation and not original_swapped:
            shutil.rmtree(operation_root, ignore_errors=True)
        try:
            staging_parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
