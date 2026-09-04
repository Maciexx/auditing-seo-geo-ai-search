"""Create and validate immutable public-v1 project bundles.

The staging tree is owned and exclusively mutated by the single-writer project
orchestrator. Concurrent same-user mutation of that directory is outside the
approved local threat model; this module intentionally adds no locks or CAS.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .adapters import PublicResearchItem
from .artifact_policy import classification_version, saved_prompt_version
from .comparisons import (
    ValidationComparison,
    compare_findings,
    compare_observed_ai_visibility,
    compare_visibility,
    validate_follow_up_canonical_binding,
)
from .config import AuditConfig
from .crawler import Resolver
from .data_intake import (
    FactApprovalState,
    IntakeValidationError,
    NormalizedIntake,
    consume_intake,
    discard_owned_intake_dir,
)
from .data_requests import (
    DataRequestContext,
    DataRequestPack,
    EntityKind,
    build_data_request,
    render_data_request_markdown,
    write_data_request,
)
from .diagnostic_sources import _diagnostic_validation_operation
from .models import AuditRun, Evidence
from .offer_context import product_sales_urls
from .orchestrator import compile_audit_run, run_public_audit
from .owner_context import (
    OwnerContext,
    OwnerFact,
    OwnerFactField,
    build_owner_context,
    derive_report_status,
)
from .project_models import (
    AuditStage,
    AuditVersionRef,
    ProjectManifest,
    ReportStatus,
    normalize_canonical_domain,
    validate_project_id,
)
from .project_store import PendingVersion, ProjectIdentityError, ProjectStore
from .prompt_context import PromptTopic
from .prompts import SELECTED_PROMPT_PACK_VERSION, validate_automatic_prompt_pack
from .renderer import render_client_report
from .report_models import (
    ClientReportData,
    ProjectReportMetadata,
    SupplementaryDiagnosticComparison,
)
from .reports import ReportDraft, RewriteProvider, validate_client_report_context
from .scoring import observed_ai_visibility
from .visibility_metrics import (
    VisibilitySnapshot,
    build_visibility_snapshot,
    validate_series_definition,
)

_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
_MANIFEST_VERSION: Literal["1.0.0"] = "1.0.0"
_ENGINE_ARTIFACTS = (
    "audit.json",
    "evidence.jsonl",
    "implementation-backlog.csv",
    "ai-prompts.json",
    "report-draft.json",
    "client-report-data.json",
)
_BACKLOG_HEADER = (
    "priority",
    "finding_id",
    "client_title",
    "implementation",
    "affected_urls",
    "evidence_ids",
)
_BUNDLE_DIRECTORIES = frozenset({"engine", "report", "manifests", "aggregates"})
_STATIC_BUNDLE_FILES = frozenset(
    {
        *{f"engine/{name}" for name in _ENGINE_ARTIFACTS},
        "manifests/input-manifest.json",
        "manifests/evidence-manifest.json",
        "next-audit-data-request.json",
    }
)
_OUTPUT_MANIFEST_PATH = "manifests/output-manifest.json"
_REQUEST_MARKDOWN = re.compile(r"next-audit-data-request_(en|pl)\.md")
_REPORT_PDF = re.compile(
    r"report/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"_AI_Search_SEO_Audit_(EN|PL)_v([1-9][0-9]*)\.pdf"
)
_LOCAL_ENTITY_TYPES = frozenset(
    {
        "airport",
        "apartmentcomplex",
        "bedandbreakfast",
        "campground",
        "civicstructure",
        "educationalorganization",
        "emergencyservice",
        "entertainmentbusiness",
        "foodestablishment",
        "governmentoffice",
        "healthandbeautybusiness",
        "hotel",
        "lodgingbusiness",
        "localbusiness",
        "medicalbusiness",
        "motel",
        "professionalservice",
        "resort",
        "sportsactivitylocation",
        "store",
        "touristattraction",
    }
)
_ECOMMERCE_ENTITY_TYPES = frozenset(
    {
        "aggregateoffer",
        "brand",
        "collectionpage",
        "demand",
        "individualproduct",
        "itemlist",
        "offer",
        "offercatalog",
        "onlinebusiness",
        "onlinestore",
        "product",
        "productgroup",
        "productmodel",
        "someproducts",
    }
)


class _FrozenManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("project_id", check_fields=False)
    @classmethod
    def validate_manifest_project_id(cls, value: str) -> str:
        return validate_project_id(value)


class ArtifactMetadata(_FrozenManifestModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if (
            not value
            or value == "."
            or path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or "\\" in value
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return value


class PublicTargetProvenance(_FrozenManifestModel):
    normalized_url: str
    canonical_domain: str
    max_pages: int = Field(ge=1, le=100)
    report_locale: Literal["pl", "en"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)


class InputManifest(_FrozenManifestModel):
    schema_version: Literal["1.0.0"] = _MANIFEST_SCHEMA_VERSION
    version: Literal["1.0.0"] = _MANIFEST_VERSION
    project_id: str
    audit_id: str
    source_audit_id: str | None = None
    public_target: PublicTargetProvenance
    raw_inputs: tuple[ArtifactMetadata, ...] = ()
    normalized_inputs: tuple[ArtifactMetadata, ...] = ()


class EvidenceManifest(_FrozenManifestModel):
    schema_version: Literal["1.0.0"] = _MANIFEST_SCHEMA_VERSION
    version: Literal["1.0.0"] = _MANIFEST_VERSION
    project_id: str
    audit_id: str
    evidence_count: int = Field(ge=0)
    evidence_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact: ArtifactMetadata


class OutputManifest(_FrozenManifestModel):
    schema_version: Literal["1.0.0"] = _MANIFEST_SCHEMA_VERSION
    version: Literal["1.0.0"] = _MANIFEST_VERSION
    project_id: str
    audit_id: str
    artifacts: tuple[ArtifactMetadata, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _SecureFile:
    content: bytes
    sha256: str
    bytes: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _BundleSnapshot:
    directories: frozenset[str]
    files: dict[str, _SecureFile]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _artifact(relative_path: str, file: _SecureFile) -> ArtifactMetadata:
    return ArtifactMetadata(
        relative_path=relative_path,
        sha256=file.sha256,
        bytes=file.bytes,
    )


def _secure_read_regular_file(path: Path) -> _SecureFile:
    try:
        path_state = path.lstat()
    except OSError as exc:
        raise ValueError(f"bundle file cannot be inspected: {path}") from exc
    if not stat.S_ISREG(path_state.st_mode):
        kind = "symlink" if stat.S_ISLNK(path_state.st_mode) else "non-regular file"
        raise ValueError(f"bundle entry must be a regular file, not a {kind}: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"bundle file cannot be opened without following links: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            path_state.st_dev,
            path_state.st_ino,
        ):
            raise ValueError(f"bundle file changed before secure read: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    stable_fields_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    content = b"".join(chunks)
    if stable_fields_before != stable_fields_after or len(content) != before.st_size:
        raise ValueError(f"bundle file changed during secure read: {path}")
    return _SecureFile(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
        device=before.st_dev,
        inode=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )


def _secure_bundle_walk(version_root: Path) -> _BundleSnapshot:
    try:
        root_state = version_root.lstat()
    except OSError as exc:
        raise ValueError("version bundle root cannot be inspected") from exc
    if not stat.S_ISDIR(root_state.st_mode) or stat.S_ISLNK(root_state.st_mode):
        raise ValueError("version bundle root must be a real directory")

    directories: set[str] = set()
    files: dict[str, _SecureFile] = {}

    def walk(directory: Path, relative_directory: PurePosixPath) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"bundle directory cannot be inspected: {directory}") from exc
        for entry in entries:
            relative = relative_directory / entry.name
            relative_path = relative.as_posix()
            try:
                entry_state = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"bundle entry cannot be inspected: {relative_path}") from exc
            if stat.S_ISLNK(entry_state.st_mode):
                raise ValueError(f"bundle entry must not be a symlink: {relative_path}")
            if stat.S_ISDIR(entry_state.st_mode):
                directories.add(relative_path)
                walk(Path(entry.path), relative)
            elif stat.S_ISREG(entry_state.st_mode):
                files[relative_path] = _secure_read_regular_file(Path(entry.path))
            else:
                raise ValueError(
                    f"bundle entry must be a directory or regular file: {relative_path}"
                )

    walk(version_root, PurePosixPath())
    return _BundleSnapshot(directories=frozenset(directories), files=files)


def _validate_inventory(
    snapshot: _BundleSnapshot,
    *,
    require_output_manifest: bool,
    version_number: int = 1,
    include_owner_context: bool = False,
    include_visibility_metrics: bool = False,
    include_validation_comparison: bool = False,
) -> tuple[str, Literal["pl", "en"]]:
    if snapshot.directories != _BUNDLE_DIRECTORIES:
        raise ValueError("public-v1 bundle directory inventory is incomplete or unexpected")
    paths = set(snapshot.files)
    markdown_paths = sorted(path for path in paths if _REQUEST_MARKDOWN.fullmatch(path))
    report_paths = sorted(path for path in paths if _REPORT_PDF.fullmatch(path))
    if len(markdown_paths) != 1 or len(report_paths) != 1:
        raise ValueError("public-v1 bundle requires exactly one request Markdown and report PDF")
    expected = set(_STATIC_BUNDLE_FILES)
    if include_owner_context:
        expected.add("aggregates/owner-context.json")
    if include_visibility_metrics:
        expected.add("aggregates/visibility-metrics.json")
    if include_validation_comparison:
        expected.add("aggregates/validation-comparison.json")
        if "aggregates/supplementary-diagnostic-comparison.json" in paths:
            expected.add("aggregates/supplementary-diagnostic-comparison.json")
    expected.update(markdown_paths)
    expected.update(report_paths)
    if require_output_manifest:
        expected.add(_OUTPUT_MANIFEST_PATH)
    if paths != expected:
        raise ValueError("public-v1 bundle file inventory is incomplete or unexpected")
    locale_match = _REQUEST_MARKDOWN.fullmatch(markdown_paths[0])
    report_match = _REPORT_PDF.fullmatch(report_paths[0])
    if locale_match is None or report_match is None:
        raise ValueError("public-v1 locale artifact names are invalid")
    request_locale = locale_match.group(1)
    report_locale = report_match.group(1).lower()
    report_version = int(report_match.group(2))
    if request_locale != report_locale or request_locale not in {"pl", "en"}:
        raise ValueError("public-v1 locale artifact names are inconsistent")
    if report_version != version_number:
        raise ValueError("bundle report filename does not match the audit version")
    validated_locale: Literal["pl", "en"] = "pl" if request_locale == "pl" else "en"
    return report_paths[0], validated_locale


def _decode(file: _SecureFile, *, role: str) -> str:
    try:
        return file.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role} must be valid UTF-8") from exc


def _implementation_backlog_rows(report: ClientReportData) -> list[list[str]]:
    rows = [list(_BACKLOG_HEADER)]
    for finding in sorted(report.findings, key=lambda item: item.priority.value):
        rows.append(
            [
                finding.priority.value,
                finding.finding_id,
                finding.client_title,
                finding.implementation,
                " | ".join(finding.affected_urls),
                " | ".join(finding.evidence_ids),
            ]
        )
    return rows


def _client_stem(client_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", client_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    stem = re.sub(r"_+", "_", stem)
    if not stem:
        raise ValueError("client_name must contain a portable filename character")
    return stem[:100].rstrip("._-")


def _entity_kind(run: AuditRun) -> EntityKind:
    if classification_version(run) == "1.0.0":
        return _entity_kind_v1(run)
    if product_sales_urls(run.entity, run.pages):
        return EntityKind.ECOMMERCE
    if run.entity is not None and run.entity.type.casefold() in _LOCAL_ENTITY_TYPES:
        return EntityKind.LOCAL
    return EntityKind.GENERIC


def _entity_kind_v1(run: AuditRun) -> EntityKind:
    entity_type = run.entity.type.casefold() if run.entity is not None else ""
    if entity_type in _ECOMMERCE_ENTITY_TYPES:
        return EntityKind.ECOMMERCE
    if entity_type in _LOCAL_ENTITY_TYPES:
        return EntityKind.LOCAL
    structured_nodes = [
        node for page in run.pages for value in page.json_ld for node in _structured_nodes(value)
    ]
    observed_types: set[str] = set()
    for node in structured_nodes:
        raw_type = node.get("@type")
        type_values = raw_type if isinstance(raw_type, list) else [raw_type]
        observed_types.update(item.casefold() for item in type_values if isinstance(item, str))
    ecommerce_features = frozenset(
        {"offers", "price", "priceCurrency", "sku", "gtin", "availability"}
    )
    if observed_types.intersection(_ECOMMERCE_ENTITY_TYPES) or any(
        ecommerce_features.intersection(node) for node in structured_nodes
    ):
        return EntityKind.ECOMMERCE
    return EntityKind.GENERIC


def _data_request_context(
    run: AuditRun,
    *,
    project_id: str,
    locale: Literal["pl", "en"],
) -> DataRequestContext:
    return DataRequestContext(
        project_id=project_id,
        canonical_domain=run.site.domain,
        entity_kind=_entity_kind(run),
        locale=locale,
        observed_features=(),
        detected_languages=tuple(run.site.languages),
        detected_markets=(),
    )


def _structured_nodes(value: dict[str, object]) -> list[dict[str, object]]:
    nodes = [value]
    graph = value.get("@graph")
    if isinstance(graph, list):
        nodes.extend(item for item in graph if isinstance(item, dict))
    return nodes


def _relocate_pdf(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError("audit engine did not produce its deterministic PDF")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(f"report destination already exists: {destination}")
    os.replace(source, destination)


def _write_manifests(
    *,
    version_root: Path,
    project_id: str,
    run: AuditRun,
    config: AuditConfig,
    report_locale: Literal["pl", "en"],
    source_audit_id: str | None = None,
    owner_context_path: str | None = None,
    visibility_metrics_path: str | None = None,
    validation_comparison_path: str | None = None,
    version_number: int = 1,
) -> None:
    manifest_root = version_root / "manifests"
    config_value = {
        "canonical_domain": run.site.domain,
        "max_pages": config.max_pages,
        "normalized_url": config.domain,
        "report_locale": report_locale,
    }
    config_bytes = _json_bytes(config_value)
    normalized_inputs: tuple[ArtifactMetadata, ...] = tuple(
        _artifact(path, _secure_read_regular_file(version_root / path))
        for path in (owner_context_path, visibility_metrics_path)
        if path is not None
    )
    input_manifest = InputManifest(
        project_id=project_id,
        audit_id=run.audit_id,
        source_audit_id=source_audit_id,
        public_target=PublicTargetProvenance(
            normalized_url=config.domain,
            canonical_domain=run.site.domain,
            max_pages=config.max_pages,
            report_locale=report_locale,
            sha256=hashlib.sha256(config_bytes).hexdigest(),
            bytes=len(config_bytes),
        ),
        raw_inputs=(),
        normalized_inputs=normalized_inputs,
    )
    _atomic_json(manifest_root / "input-manifest.json", input_manifest)

    evidence_ids = sorted(item.evidence_id for item in run.evidence)
    evidence_ids_bytes = _json_bytes(evidence_ids)
    evidence_file = _secure_read_regular_file(version_root / "engine" / "evidence.jsonl")
    evidence_manifest = EvidenceManifest(
        project_id=project_id,
        audit_id=run.audit_id,
        evidence_count=len(evidence_ids),
        evidence_ids_sha256=hashlib.sha256(evidence_ids_bytes).hexdigest(),
        artifact=_artifact("engine/evidence.jsonl", evidence_file),
    )
    _atomic_json(manifest_root / "evidence-manifest.json", evidence_manifest)

    snapshot = _secure_bundle_walk(version_root)
    _validate_inventory(
        snapshot,
        require_output_manifest=False,
        version_number=version_number,
        include_owner_context=owner_context_path is not None,
        include_visibility_metrics=visibility_metrics_path is not None,
        include_validation_comparison=validation_comparison_path is not None,
    )
    output_manifest = OutputManifest(
        project_id=project_id,
        audit_id=run.audit_id,
        artifacts=tuple(
            _artifact(relative_path, snapshot.files[relative_path])
            for relative_path in sorted(snapshot.files)
        ),
    )
    _atomic_json(manifest_root / "output-manifest.json", output_manifest)


def _validate_artifact(metadata: ArtifactMetadata, snapshot: _BundleSnapshot) -> None:
    file = snapshot.files.get(metadata.relative_path)
    if file is None:
        raise ValueError(f"manifest references missing artifact: {metadata.relative_path}")
    observed = _artifact(metadata.relative_path, file)
    if observed != metadata:
        raise ValueError(f"artifact metadata mismatch: {metadata.relative_path}")


@_diagnostic_validation_operation
def validate_project_bundle(
    version_root: Path,
    *,
    expected_project_id: str,
    expected_version_number: int = 1,
    expected_source_audit_id: str | None = None,
    expect_owner_context: bool | None = None,
    expect_visibility_metrics: bool = False,
    expect_validation_comparison: bool = False,
    expected_stage: AuditStage | None = None,
) -> _BundleSnapshot:
    """Validate public-v1 before promotion under the local single-writer contract.

    The caller exclusively owns the staging tree. Concurrent same-user mutation
    is outside this validator's approved threat model.
    """
    version_root = Path(version_root)
    expected_project_id = validate_project_id(expected_project_id)
    if expected_stage is AuditStage.PUBLIC or expected_source_audit_id is None:
        if (expected_version_number == 1) != (expected_source_audit_id is None):
            raise ValueError("public bundle source audit must be absent only for public-v1")
    snapshot = _secure_bundle_walk(version_root)
    include_owner_context = (
        expected_source_audit_id is not None and expected_stage is not AuditStage.PUBLIC
        if expect_owner_context is None
        else expect_owner_context
    )
    report_path, artifact_locale = _validate_inventory(
        snapshot,
        require_output_manifest=True,
        version_number=expected_version_number,
        include_owner_context=include_owner_context,
        include_visibility_metrics=expect_visibility_metrics,
        include_validation_comparison=expect_validation_comparison,
    )
    files = snapshot.files

    audit = AuditRun.model_validate_json(files["engine/audit.json"].content)
    draft = ReportDraft.model_validate_json(files["engine/report-draft.json"].content)
    client_report = ClientReportData.model_validate_json(
        files["engine/client-report-data.json"].content
    )
    with io.StringIO(
        _decode(files["engine/implementation-backlog.csv"], role="implementation backlog")
    ) as handle:
        backlog_rows = list(csv.reader(handle))
    if backlog_rows != _implementation_backlog_rows(client_report):
        raise ValueError("implementation backlog does not match client report findings")
    evidence = [
        Evidence.model_validate_json(line)
        for line in _decode(files["engine/evidence.jsonl"], role="evidence JSONL").splitlines()
    ]
    if evidence != audit.evidence:
        raise ValueError("evidence JSONL content or order does not match audit evidence")

    request_pack = DataRequestPack.model_validate_json(
        files["next-audit-data-request.json"].content
    )
    request_markdown_path = f"next-audit-data-request_{artifact_locale}.md"
    request_markdown = _decode(files[request_markdown_path], role="data-request Markdown")

    try:
        pdf = PdfReader(io.BytesIO(files[report_path].content), strict=True)
        if len(pdf.pages) < 1:
            raise ValueError("PDF contains no pages")
    except (PdfReadError, EOFError, OSError, ValueError) as exc:
        raise ValueError("bundle report must be a readable PDF with at least one page") from exc
    try:
        with tempfile.TemporaryDirectory(prefix=".bundle-report-validation.") as temp_name:
            rendered = render_client_report(client_report, Path(temp_name) / "client-report.pdf")
    except Exception as exc:
        raise ValueError("bundle PDF could not be rendered from ClientReportData") from exc
    bundle_pdf = files[report_path]
    if rendered.sha256 != bundle_pdf.sha256 or rendered.byte_count != bundle_pdf.bytes:
        raise ValueError("bundle report PDF does not match the persisted ClientReportData")

    input_manifest = InputManifest.model_validate_json(
        files["manifests/input-manifest.json"].content
    )
    evidence_manifest = EvidenceManifest.model_validate_json(
        files["manifests/evidence-manifest.json"].content
    )
    output_manifest = OutputManifest.model_validate_json(files[_OUTPUT_MANIFEST_PATH].content)
    if {input_manifest.audit_id, evidence_manifest.audit_id, output_manifest.audit_id} != {
        audit.audit_id
    }:
        raise ValueError("manifest audit identity does not match engine audit")
    observed_project_ids = {
        input_manifest.project_id,
        evidence_manifest.project_id,
        output_manifest.project_id,
        request_pack.project_id,
    }
    if observed_project_ids != {expected_project_id}:
        raise ValueError("bundle project identity does not match the expected project")
    if input_manifest.raw_inputs:
        raise ValueError("bundle input manifest must not reference raw inputs")
    if input_manifest.source_audit_id != expected_source_audit_id:
        raise ValueError("input manifest source audit identity mismatch")
    if expected_source_audit_id is None:
        if input_manifest.normalized_inputs:
            raise ValueError("public-v1 must not reference normalized owner context")
    else:
        expected_normalized_paths = set()
        if include_owner_context:
            expected_normalized_paths.add("aggregates/owner-context.json")
        if expect_visibility_metrics:
            expected_normalized_paths.add("aggregates/visibility-metrics.json")
        observed_normalized_paths = {
            item.relative_path for item in input_manifest.normalized_inputs
        }
        if observed_normalized_paths != expected_normalized_paths:
            raise ValueError("context bundle normalized aggregate inventory mismatch")
        for normalized in input_manifest.normalized_inputs:
            _validate_artifact(normalized, snapshot)
    canonical_owner_context: OwnerContext | None = None
    canonical_visibility: VisibilitySnapshot | None = None
    canonical_comparison: ValidationComparison | None = None
    supplementary: SupplementaryDiagnosticComparison | None = None
    supplementary_path = "aggregates/supplementary-diagnostic-comparison.json"
    if supplementary_path in files:
        from .diagnostic_workflow import compare_diagnostic_runs

        supplementary = SupplementaryDiagnosticComparison.model_validate_json(
            files[supplementary_path].content
        )
        if (
            audit.configuration.get("supplementary_diagnostic_comparison_sha256")
            != files[supplementary_path].sha256
        ):
            raise ValueError("supplementary diagnostic comparison aggregate digest mismatch")
        # Both canonical and pending versions are two levels below their project root.
        project_root = version_root.parent.parent
        project_manifest = ProjectStore(project_root.parent).load(expected_project_id)
        for reference in (supplementary.baseline_reference, supplementary.follow_up_reference):
            source_version = next(
                (v for v in project_manifest.versions if v.version_id == reference.source_version),
                None,
            )
            if source_version is None or source_version.version_number >= expected_version_number:
                raise ValueError("supplementary diagnostic source must precede validation version")
        expected_supplementary = compare_diagnostic_runs(
            f"project:{expected_project_id}",
            clients_root=project_root.parent,
            baseline_reference=supplementary.baseline_reference,
            follow_up_reference=supplementary.follow_up_reference,
        )
        if supplementary != expected_supplementary:
            raise ValueError("supplementary diagnostic comparison does not match validated runs")
    elif "supplementary_diagnostic_comparison_sha256" in audit.configuration:
        raise ValueError("missing supplementary diagnostic comparison aggregate")
    if include_owner_context:
        canonical_owner_context = OwnerContext.model_validate_json(
            files["aggregates/owner-context.json"].content
        )
        if (
            canonical_owner_context.project_id != expected_project_id
            or canonical_owner_context.canonical_domain != audit.site.domain
        ):
            raise ValueError("owner-context identity does not match audit project")
        _validate_owner_context_projection(
            canonical_owner_context,
            audit=audit,
            client_report=client_report,
        )
    if expect_visibility_metrics:
        visibility_file = files["aggregates/visibility-metrics.json"]
        try:
            canonical_visibility = VisibilitySnapshot.model_validate_json(visibility_file.content)
        except ValidationError as exc:
            raise ValueError("visibility aggregate is not semantically valid") from exc
        if (
            canonical_visibility.project_id != expected_project_id
            or canonical_visibility.canonical_domain != audit.site.domain
        ):
            raise ValueError("visibility aggregate identity does not match audit project")
        expected_digest = hashlib.sha256(visibility_file.content).hexdigest()
        if audit.configuration.get("visibility_snapshot_sha256") != expected_digest:
            raise ValueError("visibility aggregate does not match canonical audit projection")
        if audit.configuration.get("visibility_metric_ids") != [
            metric.metric_id for metric in canonical_visibility.metrics
        ]:
            raise ValueError("visibility metric identities do not match canonical audit projection")
    if expect_validation_comparison:
        try:
            canonical_comparison = ValidationComparison.model_validate_json(
                files["aggregates/validation-comparison.json"].content
            )
        except ValidationError as exc:
            raise ValueError("validation comparison aggregate is not semantically valid") from exc
        if (
            canonical_comparison.follow_up_audit_id != audit.audit_id
            or canonical_comparison.baseline_audit_id
            != audit.configuration.get("baseline_audit_id")
        ):
            raise ValueError("validation comparison audit identity mismatch")
        if canonical_comparison.baseline_audit_id != expected_source_audit_id:
            raise ValueError(
                "validation comparison baseline identity differs from expected source audit"
            )
        if audit.configuration.get("source_audit_id") != expected_source_audit_id:
            raise ValueError("validation source audit identity mismatch")
        validate_follow_up_canonical_binding(
            canonical_comparison,
            audit=audit,
            visibility_snapshot=canonical_visibility,
            prompt_pack_version=_prompt_pack_version(audit),
            setup_fingerprint=_visibility_setup_fingerprint(audit),
            canonical_prompt_ids=_canonical_prompt_ids(audit),
        )
    expected_project_metadata = None
    if expected_source_audit_id is None or expected_stage is AuditStage.PUBLIC:
        expected_project_metadata = ProjectReportMetadata(
            project_id=expected_project_id,
            version_id=f"public-v{expected_version_number}",
            version_number=expected_version_number,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            source_audit_id=expected_source_audit_id,
        )
    elif expected_stage is AuditStage.VALIDATION:
        validation_status = (
            ReportStatus.CLIENT_CONTEXT_DRAFT
            if canonical_owner_context is None
            else derive_report_status(
                stage=AuditStage.VALIDATION,
                used_fact_ids=tuple(fact.fact_id for fact in canonical_owner_context.facts),
                owner_context=canonical_owner_context,
            )
        )
        expected_project_metadata = ProjectReportMetadata(
            project_id=expected_project_id,
            version_id=f"validation-v{expected_version_number}",
            version_number=expected_version_number,
            stage=AuditStage.VALIDATION,
            report_status=validation_status,
            source_audit_id=expected_source_audit_id,
        )
    elif canonical_owner_context is not None:
        expected_project_metadata = ProjectReportMetadata(
            project_id=expected_project_id,
            version_id=f"context-v{expected_version_number}",
            version_number=expected_version_number,
            stage=AuditStage.CONTEXT,
            report_status=derive_report_status(
                stage=AuditStage.CONTEXT,
                used_fact_ids=tuple(fact.fact_id for fact in canonical_owner_context.facts),
                owner_context=canonical_owner_context,
            ),
            source_audit_id=expected_source_audit_id,
        )
    validate_client_report_context(
        client_report,
        project_metadata=expected_project_metadata,
        owner_context=canonical_owner_context,
        visibility_snapshot=canonical_visibility,
        validation_comparison=canonical_comparison,
        supplementary_diagnostic_comparison=supplementary,
    )
    if (
        input_manifest.public_target.canonical_domain != audit.site.domain
        or input_manifest.public_target.normalized_url != audit.configuration.get("target")
        or input_manifest.public_target.max_pages != audit.configuration.get("max_pages")
        or input_manifest.public_target.report_locale != artifact_locale
    ):
        raise ValueError("input manifest target identity does not match audit")
    public_target_value = {
        "canonical_domain": input_manifest.public_target.canonical_domain,
        "max_pages": input_manifest.public_target.max_pages,
        "normalized_url": input_manifest.public_target.normalized_url,
        "report_locale": input_manifest.public_target.report_locale,
    }
    public_target_bytes = _json_bytes(public_target_value)
    if (
        input_manifest.public_target.bytes != len(public_target_bytes)
        or input_manifest.public_target.sha256 != hashlib.sha256(public_target_bytes).hexdigest()
    ):
        raise ValueError("input manifest public target metadata mismatch")
    evidence_ids = sorted(item.evidence_id for item in evidence)
    evidence_ids_bytes = _json_bytes(evidence_ids)
    if (
        evidence_manifest.evidence_count != len(evidence_ids)
        or evidence_manifest.evidence_ids_sha256 != hashlib.sha256(evidence_ids_bytes).hexdigest()
    ):
        raise ValueError("evidence manifest identity summary mismatch")
    if evidence_manifest.artifact.relative_path != "engine/evidence.jsonl":
        raise ValueError("evidence manifest must reference canonical evidence artifact")
    _validate_artifact(evidence_manifest.artifact, snapshot)
    actual_artifacts = set(files).difference({_OUTPUT_MANIFEST_PATH})
    manifested_artifacts = {artifact.relative_path for artifact in output_manifest.artifacts}
    if manifested_artifacts != actual_artifacts or len(manifested_artifacts) != len(
        output_manifest.artifacts
    ):
        raise ValueError("output manifest artifact set is incomplete or duplicated")
    for artifact in output_manifest.artifacts:
        _validate_artifact(artifact, snapshot)

    expected_brand = audit.entity.brand if audit.entity else audit.site.brand or audit.site.domain
    expected_report_identity = (
        artifact_locale,
        audit.audit_id,
        audit.site.domain,
        expected_brand,
        audit.timestamp,
        audit.audit_engine_version,
        audit.ruleset_version,
        audit.sitemap_state,
    )
    draft_identity = (
        draft.report_locale,
        draft.audit_id,
        draft.target_domain,
        draft.brand,
        draft.audit_timestamp,
        draft.audit_engine_version,
        draft.ruleset_version,
        draft.sitemap_state,
    )
    client_report_identity = (
        client_report.report_locale,
        client_report.audit_id,
        client_report.target_domain,
        client_report.brand,
        client_report.audit_timestamp,
        client_report.audit_engine_version,
        client_report.ruleset_version,
        client_report.sitemap_state,
    )
    if draft_identity != expected_report_identity:
        raise ValueError("report draft identity does not match audit")
    if client_report_identity != expected_report_identity:
        raise ValueError("client report identity does not match audit")
    if (
        draft.findings != audit.findings
        or draft.scores != audit.scores
        or draft.evidence != audit.evidence
        or draft.recommendations != audit.recommendations
        or draft.ai_prompts != audit.ai_prompts
        or draft.entity_consistency_matrix != audit.entity_consistency_matrix
    ):
        raise ValueError("report draft protected content does not match audit")

    if (
        request_pack.project_id != expected_project_id
        or request_pack.canonical_domain != audit.site.domain
        or request_pack.locale != artifact_locale
        or request_pack.entity_kind is not _entity_kind(audit)
    ):
        raise ValueError("data-request identity does not match audit project")
    expected_request_pack = build_data_request(
        _data_request_context(
            audit,
            project_id=expected_project_id,
            locale=artifact_locale,
        )
    )
    if request_pack != expected_request_pack:
        raise ValueError("data-request pack does not match canonical audit-derived request")
    if request_markdown != render_data_request_markdown(request_pack):
        raise ValueError("data-request Markdown does not match its canonical JSON")

    try:
        prompt_payload = json.loads(_decode(files["engine/ai-prompts.json"], role="AI prompt pack"))
    except json.JSONDecodeError as exc:
        raise ValueError("AI prompt pack must be valid JSON") from exc
    expected_prompt_payload = {
        "version": saved_prompt_version(audit),
        "observed_ai_visibility_state": observed_ai_visibility(audit.ai_observations).state.value,
        "prompts": [prompt.model_dump(mode="json") for prompt in audit.ai_prompts],
    }
    if prompt_payload != expected_prompt_payload:
        raise ValueError("AI prompt pack does not match audit prompts or observed state")

    expected_output_paths = {
        name: (report_path if name == "client-report.pdf" else f"engine/{name}")
        for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
    }
    if audit.output_paths != expected_output_paths:
        raise ValueError("audit output paths do not match the stable bundle artifacts")
    return snapshot


def create_project_audit(
    domain: str,
    *,
    clients_root: Path,
    project_id: str,
    client_name: str,
    report_locale: Literal["pl", "en"],
    max_pages: int = 50,
    now: datetime | None = None,
    crawler_transport: httpx.BaseTransport | None = None,
    crawler_resolver: Resolver | None = None,
    research_items: list[PublicResearchItem] | None = None,
    rewrite_provider: RewriteProvider | None = None,
    selected_topics: tuple[PromptTopic, ...] | None = None,
) -> ProjectManifest:
    """Create and atomically promote the immutable initial public audit version."""
    project_id = validate_project_id(project_id)
    if not client_name.strip():
        raise ValueError("client_name must not be blank")
    safe_client_stem = _client_stem(client_name)
    if report_locale not in {"pl", "en"}:
        raise ValueError("report_locale must be 'pl' or 'en'")
    config = AuditConfig(domain=domain, max_pages=max_pages)
    canonical_domain = normalize_canonical_domain(config.domain)
    timestamp = now or datetime.now(UTC)
    store = ProjectStore(clients_root)

    if os.path.lexists(store.clients_root / project_id):
        store.assert_identity(project_id, domain=canonical_domain, client_name=client_name)
        raise FileExistsError(f"project {project_id!r} already exists")

    pending = store.begin_new_project(project_id)
    try:
        version_relative_path = f"audits/{timestamp.date().isoformat()}_public-v1"
        version_root = pending.staging_path / version_relative_path
        engine_root = version_root / "engine"
        report_root = version_root / "report"
        for directory in (
            engine_root,
            report_root,
            version_root / "manifests",
            version_root / "aggregates",
        ):
            directory.mkdir(parents=True, exist_ok=False)

        public_project_metadata = ProjectReportMetadata(
            project_id=project_id,
            version_id="public-v1",
            version_number=1,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            source_audit_id=None,
        )
        run = run_public_audit(
            config.domain,
            output_dir=engine_root,
            max_pages=config.max_pages,
            crawler_transport=crawler_transport,
            crawler_resolver=crawler_resolver,
            research_items=research_items,
            rewrite_provider=rewrite_provider,
            report_locale=report_locale,
            now=timestamp,
            project_metadata=public_project_metadata,
            selected_topics=selected_topics,
        )
        report_name = f"{safe_client_stem}_AI_Search_SEO_Audit_{report_locale.upper()}_v1.pdf"
        _relocate_pdf(engine_root / "client-report.pdf", report_root / report_name)
        stable_paths = {
            name: (f"report/{report_name}" if name == "client-report.pdf" else f"engine/{name}")
            for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
        }
        run.output_paths = stable_paths
        _atomic_json(engine_root / "audit.json", run)

        request_context = _data_request_context(
            run,
            project_id=project_id,
            locale=report_locale,
        )
        write_data_request(build_data_request(request_context), version_root)
        _write_manifests(
            version_root=version_root,
            project_id=project_id,
            run=run,
            config=config,
            report_locale=report_locale,
        )
        validated_snapshot = validate_project_bundle(version_root, expected_project_id=project_id)

        version = AuditVersionRef(
            version_id="public-v1",
            version_number=1,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            audit_id=run.audit_id,
            source_audit_id=None,
            created_at=timestamp,
            relative_path=version_relative_path,
        )
        manifest = ProjectManifest(
            project_id=project_id,
            client_name=client_name,
            canonical_domains=(canonical_domain,),
            report_locale=report_locale,
            created_at=timestamp,
            latest_audit_id=run.audit_id,
            source_files_policy="delete-after-processing",
            versions=(version,),
        )
        final_snapshot = _secure_bundle_walk(version_root)
        _validate_inventory(final_snapshot, require_output_manifest=True)
        if final_snapshot != validated_snapshot:
            raise ValueError("version bundle changed after validation")
        return store.promote_new_project(pending, manifest)
    except BaseException:
        store.discard_pending_project(pending)
        raise


def refresh_project(
    project_ref: str,
    *,
    clients_root: Path,
    source_version: str,
    expected_domain: str,
    expected_client_name: str,
    max_pages: int | None = None,
    now: datetime | None = None,
    crawler_transport: httpx.BaseTransport | None = None,
    crawler_resolver: Resolver | None = None,
    research_items: list[PublicResearchItem] | None = None,
    rewrite_provider: RewriteProvider | None = None,
    selected_topics: tuple[PromptTopic, ...] | None = None,
) -> ProjectManifest:
    """Append a linked public draft from fresh evidence, without owner or causal claims."""
    if not isinstance(expected_client_name, str) or not expected_client_name.strip():
        raise ProjectIdentityError("refresh requires an explicit expected client identity")
    store = ProjectStore(clients_root)
    project_path = store.resolve(project_ref)
    project_id = project_path.name
    manifest = store.load(project_id)
    store.assert_identity(project_id, domain=expected_domain, client_name=expected_client_name)
    source = next((v for v in manifest.versions if v.version_id == source_version), None)
    if source is None:
        raise ValueError(f"unknown source version {source_version!r} for {project_id!r}")
    source_root = project_path / source.relative_path
    aggregates = source_root / "aggregates"
    source_snapshot = validate_project_bundle(
        source_root,
        expected_project_id=project_id,
        expected_version_number=source.version_number,
        expected_source_audit_id=source.source_audit_id,
        expect_owner_context=(aggregates / "owner-context.json").is_file(),
        expect_visibility_metrics=(aggregates / "visibility-metrics.json").is_file(),
        expect_validation_comparison=(aggregates / "validation-comparison.json").is_file(),
        expected_stage=source.stage,
    )
    source_run = AuditRun.model_validate_json(source_snapshot.files["engine/audit.json"].content)
    if source_run.audit_id != source.audit_id:
        raise ValueError("persisted source audit identity does not match project history")
    source_report = ClientReportData.model_validate_json(
        source_snapshot.files["engine/client-report-data.json"].content
    )
    if source_report.report_locale != manifest.report_locale:
        raise ProjectIdentityError("source report locale does not match project identity")
    config = AuditConfig(
        domain=str(source_run.configuration["target"]),
        max_pages=int(source_run.configuration["max_pages"]) if max_pages is None else max_pages,
    )
    canonical_domain = normalize_canonical_domain(expected_domain)
    if (
        normalize_canonical_domain(source_run.site.domain) != canonical_domain
        or normalize_canonical_domain(config.domain) != canonical_domain
    ):
        raise ProjectIdentityError("source audit domain does not match expected project identity")
    if saved_prompt_version(source_run) == SELECTED_PROMPT_PACK_VERSION:
        validate_automatic_prompt_pack(source_run)
        if selected_topics is None:
            selected_topics = tuple(
                PromptTopic.model_validate(item)
                for item in source_run.configuration["prompt_topic_selection"]
            )
    timestamp = now or datetime.now(UTC)
    safe_client_stem = _client_stem(manifest.client_name)

    def build(pending: PendingVersion) -> AuditVersionRef:
        version_root = pending.staging_path
        engine_root = version_root / "engine"
        report_root = version_root / "report"
        for directory in (
            engine_root,
            report_root,
            version_root / "manifests",
            version_root / "aggregates",
        ):
            directory.mkdir(parents=True, exist_ok=False)
        metadata = ProjectReportMetadata(
            project_id=project_id,
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            source_audit_id=source.audit_id,
        )
        run = run_public_audit(
            config.domain,
            output_dir=engine_root,
            max_pages=config.max_pages,
            crawler_transport=crawler_transport,
            crawler_resolver=crawler_resolver,
            research_items=research_items,
            rewrite_provider=rewrite_provider,
            report_locale=manifest.report_locale,
            now=timestamp,
            project_metadata=metadata,
            selected_topics=selected_topics,
        )
        run.configuration["source_audit_id"] = source.audit_id
        report_name = (
            f"{safe_client_stem}_AI_Search_SEO_Audit_"
            f"{manifest.report_locale.upper()}_v{pending.version_number}.pdf"
        )
        _relocate_pdf(engine_root / "client-report.pdf", report_root / report_name)
        run.output_paths = {
            name: (f"report/{report_name}" if name == "client-report.pdf" else f"engine/{name}")
            for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
        }
        _atomic_json(engine_root / "audit.json", run)
        write_data_request(
            build_data_request(
                _data_request_context(run, project_id=project_id, locale=manifest.report_locale)
            ),
            version_root,
        )
        _write_manifests(
            version_root=version_root,
            project_id=project_id,
            run=run,
            config=config,
            report_locale=manifest.report_locale,
            source_audit_id=source.audit_id,
            version_number=pending.version_number,
        )
        validated_snapshot = validate_project_bundle(
            version_root,
            expected_project_id=project_id,
            expected_version_number=pending.version_number,
            expected_source_audit_id=source.audit_id,
            expected_stage=AuditStage.PUBLIC,
        )
        if _secure_bundle_walk(version_root) != validated_snapshot:
            raise ValueError("version bundle changed after validation")
        return AuditVersionRef(
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.PUBLIC,
            report_status=ReportStatus.PUBLIC_EVIDENCE_DRAFT,
            audit_id=run.audit_id,
            source_audit_id=source.audit_id,
            created_at=pending.created_at,
            relative_path=pending.relative_path,
        )

    return store.build_version(project_id, stage=AuditStage.PUBLIC, builder=build, now=timestamp)


def _owner_context_evidence_id(fact_id: str, source_sha256: str) -> str:
    digest = hashlib.sha256(f"{fact_id}:{source_sha256}".encode()).hexdigest()[:20]
    return f"evidence-owner-{digest}"


def _owner_context_value(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _owner_fact_evidence(fact: OwnerFact, *, observed_at: datetime) -> Evidence:
    evidence_id = _owner_context_evidence_id(fact.fact_id, fact.provenance.sha256)
    return Evidence(
        evidence_id=evidence_id,
        source_url=None,
        source_type="owner_fact",
        source_scope="client-supplied",
        collector="owner-context-v1",
        observed_at=observed_at,
        observed_value={
            "fact_id": fact.fact_id,
            "field": fact.field.value,
            "value": fact.value,
            "approval_state": fact.approval_state.value,
            "conflict_ids": list(fact.conflict_ids),
            "resolved_conflict_ids": list(fact.resolved_conflict_ids),
        },
        confidence=1.0 if fact.approval_state is FactApprovalState.APPROVED else 0.5,
        metadata={"source": fact.provenance.model_dump(mode="json")},
    )


def _validate_owner_context_projection(
    owner_context: OwnerContext,
    *,
    audit: AuditRun,
    client_report: ClientReportData,
) -> None:
    expected_evidence = tuple(
        _owner_fact_evidence(fact, observed_at=owner_context.processed_at)
        for fact in owner_context.facts
    )
    observed_evidence = tuple(
        item
        for item in audit.evidence
        if item.collector == "owner-context-v1"
        or item.source_type == "owner_fact"
        or item.evidence_id.startswith("evidence-owner-")
    )
    if observed_evidence != expected_evidence:
        raise ValueError("owner-context does not match canonical audit evidence projection")

    expected_bindings = Counter(
        (
            fact.field.value,
            _owner_context_value(fact.value),
            evidence.evidence_id,
        )
        for fact, evidence in zip(owner_context.facts, expected_evidence, strict=True)
    )
    reserved_evidence_ids = {evidence.evidence_id for evidence in expected_evidence}
    observed_bindings = Counter(
        (field, value, evidence_id)
        for field, values in audit.entity_consistency_matrix.items()
        for value, evidence_ids in values.items()
        for evidence_id in evidence_ids
        if evidence_id in reserved_evidence_ids or evidence_id.startswith("evidence-owner-")
    )
    if observed_bindings != expected_bindings:
        raise ValueError("owner-context does not match canonical audit entity projection")

    report_matrix = {
        fact.fact: {value.value: list(value.evidence_ids) for value in fact.values}
        for fact in client_report.entity_consistency.facts
    }
    if report_matrix != audit.entity_consistency_matrix:
        raise ValueError("owner-context canonical audit matrix does not match report projection")

    expected_appendix = {
        evidence.evidence_id: (
            evidence.source_type,
            evidence.source_scope,
            evidence.collector,
            None,
            json.dumps(
                evidence.observed_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )[:800],
            evidence.observed_at,
            evidence.confidence,
        )
        for evidence in expected_evidence
    }
    observed_appendix = {
        item.evidence_id: (
            item.source_type,
            item.source_scope,
            item.source,
            item.url,
            item.observation,
            item.collected_at,
            item.confidence,
        )
        for item in client_report.evidence_appendix
        if item.evidence_id in reserved_evidence_ids
        or item.evidence_id.startswith("evidence-owner-")
    }
    if observed_appendix != expected_appendix:
        raise ValueError("owner-context canonical audit evidence does not match report projection")


def _canonical_entity_values(owner_context: OwnerContext) -> set[str]:
    return {
        str(fact.value).strip().casefold()
        for fact in owner_context.facts
        if fact.field is OwnerFactField.CANONICAL_ENTITY and fact.value is not None
    }


def _validate_owner_context_identity(
    owner_context: OwnerContext,
    *,
    manifest: ProjectManifest,
    source_run: AuditRun,
) -> None:
    if (
        owner_context.project_id != manifest.project_id
        or owner_context.canonical_domain not in manifest.canonical_domains
        or owner_context.canonical_domain != source_run.site.domain
    ):
        raise ProjectIdentityError("owner context project or domain identity mismatch")

    observed_entity_values = _canonical_entity_values(owner_context)
    accepted_entity_values = {
        manifest.client_name.strip().casefold(),
        source_run.site.brand.strip().casefold() if source_run.site.brand else "",
        source_run.entity.brand.strip().casefold() if source_run.entity else "",
    }
    accepted_entity_values.discard("")
    if observed_entity_values and not observed_entity_values.issubset(accepted_entity_values):
        raise ProjectIdentityError("owner context canonical entity identity mismatch")


def _validate_owner_context_intake(normalized: NormalizedIntake) -> None:
    if not normalized.owner_facts:
        raise IntakeValidationError("owner-context intake requires at least one owner fact")
    if normalized.metric_series or normalized.cited_examples:
        raise IntakeValidationError(
            "owner-context intake must not contain metrics or cited examples"
        )
    referenced_source_ids = {fact.source_id for fact in normalized.owner_facts}
    declared_source_ids = {source.source_id for source in normalized.sources}
    if referenced_source_ids != declared_source_ids:
        raise IntakeValidationError(
            "every owner-context source must be referenced by an owner fact"
        )


def _apply_owner_context(
    source_run: AuditRun,
    owner_context: OwnerContext,
    *,
    audit_id: str,
    timestamp: datetime,
) -> tuple[AuditRun, tuple[str, ...]]:
    run = source_run.model_copy(deep=True)
    run.audit_id = audit_id
    run.timestamp = timestamp
    run.configuration = {
        **source_run.configuration,
        "source_audit_id": source_run.audit_id,
        "context_schema_version": owner_context.schema_version,
    }
    used_fact_ids: list[str] = []
    for fact in owner_context.facts:
        evidence = _owner_fact_evidence(fact, observed_at=owner_context.processed_at)
        run.evidence.append(evidence)
        value = _owner_context_value(fact.value)
        values = run.entity_consistency_matrix.setdefault(fact.field.value, {})
        evidence_ids = values.setdefault(value, [])
        if evidence.evidence_id not in evidence_ids:
            evidence_ids.append(evidence.evidence_id)
        used_fact_ids.append(fact.fact_id)
    return run, tuple(used_fact_ids)


def update_project_context(
    project_ref: str,
    *,
    clients_root: Path,
    intake_dir: Path,
    normalized_intake: NormalizedIntake | Mapping[str, object],
    now: datetime | None = None,
) -> ProjectManifest:
    """Create context-v2 from normalized owner facts without a new public crawl."""
    processed = consume_intake(
        intake_dir,
        normalized_intake,
        intake_root=intake_dir.parent,
        now=now,
        processor=_validate_owner_context_intake,
    )
    owner_context = build_owner_context(processed)
    store = ProjectStore(clients_root)
    project_path = store.resolve(project_ref)
    project_id = project_path.name
    manifest = store.load(project_id)
    if len(manifest.versions) != 1:
        raise ValueError("owner-context update requires public-v1 as the latest version")
    source_version = manifest.versions[0]
    if source_version.stage is not AuditStage.PUBLIC or source_version.version_id != "public-v1":
        raise ValueError("owner-context update requires persisted public-v1")
    source_root = project_path / source_version.relative_path
    source_snapshot = validate_project_bundle(
        source_root,
        expected_project_id=project_id,
        expected_version_number=1,
        expected_source_audit_id=None,
    )
    source_run = AuditRun.model_validate_json(source_snapshot.files["engine/audit.json"].content)
    if source_run.audit_id != source_version.audit_id:
        raise ValueError("persisted public audit identity does not match project history")
    _validate_owner_context_identity(
        owner_context,
        manifest=manifest,
        source_run=source_run,
    )

    timestamp = now or datetime.now(UTC)
    audit_digest = hashlib.sha256(
        (
            source_run.audit_id
            + ":"
            + timestamp.isoformat()
            + ":"
            + owner_context.model_dump_json()
        ).encode()
    ).hexdigest()[:16]
    audit_id = f"audit-{audit_digest}"
    safe_client_stem = _client_stem(manifest.client_name)
    config = AuditConfig(
        domain=str(source_run.configuration["target"]),
        max_pages=int(source_run.configuration["max_pages"]),
    )

    def build(pending: PendingVersion) -> AuditVersionRef:
        if pending.version_number != 2 or pending.version_id != "context-v2":
            raise ValueError("first owner-context update must allocate context-v2")
        version_root = pending.staging_path
        engine_root = version_root / "engine"
        report_root = version_root / "report"
        for directory in (
            engine_root,
            report_root,
            version_root / "manifests",
            version_root / "aggregates",
        ):
            directory.mkdir(parents=True, exist_ok=False)

        run, used_fact_ids = _apply_owner_context(
            source_run,
            owner_context,
            audit_id=audit_id,
            timestamp=timestamp,
        )
        _atomic_json(version_root / "aggregates" / "owner-context.json", owner_context)
        report_status = derive_report_status(
            stage=AuditStage.CONTEXT,
            used_fact_ids=used_fact_ids,
            owner_context=owner_context,
        )
        project_metadata = ProjectReportMetadata(
            project_id=project_id,
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.CONTEXT,
            report_status=report_status,
            source_audit_id=source_run.audit_id,
        )
        compile_audit_run(
            run,
            output_dir=engine_root,
            report_locale=manifest.report_locale,
            project_metadata=project_metadata,
            owner_context=owner_context,
        )
        report_name = (
            f"{safe_client_stem}_AI_Search_SEO_Audit_"
            f"{manifest.report_locale.upper()}_v{pending.version_number}.pdf"
        )
        _relocate_pdf(engine_root / "client-report.pdf", report_root / report_name)
        run.output_paths = {
            name: (f"report/{report_name}" if name == "client-report.pdf" else f"engine/{name}")
            for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
        }
        _atomic_json(engine_root / "audit.json", run)
        write_data_request(
            build_data_request(
                _data_request_context(
                    run,
                    project_id=project_id,
                    locale=manifest.report_locale,
                )
            ),
            version_root,
        )
        _write_manifests(
            version_root=version_root,
            project_id=project_id,
            run=run,
            config=config,
            report_locale=manifest.report_locale,
            source_audit_id=source_run.audit_id,
            owner_context_path="aggregates/owner-context.json",
            version_number=pending.version_number,
        )
        validate_project_bundle(
            version_root,
            expected_project_id=project_id,
            expected_version_number=pending.version_number,
            expected_source_audit_id=source_run.audit_id,
        )
        return AuditVersionRef(
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.CONTEXT,
            report_status=report_status,
            audit_id=run.audit_id,
            source_audit_id=source_run.audit_id,
            created_at=pending.created_at,
            relative_path=pending.relative_path,
        )

    return store.build_version(
        project_id,
        stage=AuditStage.CONTEXT,
        builder=build,
        now=timestamp,
    )


def _validate_visibility_intake(normalized: NormalizedIntake) -> None:
    if not normalized.metric_series or normalized.owner_facts or normalized.cited_examples:
        raise IntakeValidationError(
            "visibility intake requires non-empty metrics and no owner facts or examples"
        )
    sources = {source.source_id: source for source in normalized.sources}
    referenced = {series.source_id for series in normalized.metric_series}
    if referenced != set(sources):
        raise IntakeValidationError("every visibility intake source must be referenced by a metric")
    try:
        for series in normalized.metric_series:
            source = sources[series.source_id]
            validate_series_definition(
                series,
                platform=source.platform,
                source_filters=source.filters,
            )
    except ValueError as exc:
        raise IntakeValidationError("visibility intake contains an invalid metric") from exc


def enrich_project(
    project_ref: str,
    *,
    clients_root: Path,
    intake_dir: Path,
    normalized_intake: NormalizedIntake | Mapping[str, object],
    now: datetime | None = None,
) -> ProjectManifest:
    """Add visibility-only aggregates to an existing context project without crawling."""
    processed = consume_intake(
        intake_dir,
        normalized_intake,
        intake_root=intake_dir.parent,
        now=now,
        processor=_validate_visibility_intake,
    )
    visibility = build_visibility_snapshot(processed)
    store = ProjectStore(clients_root)
    project_path = store.resolve(project_ref)
    project_id = project_path.name
    manifest = store.load(project_id)
    if visibility.project_id != project_id:
        raise ProjectIdentityError("visibility intake project identity mismatch")
    if visibility.canonical_domain not in manifest.canonical_domains:
        raise ProjectIdentityError("visibility intake domain identity mismatch")
    latest = manifest.versions[-1]
    if latest.stage is not AuditStage.CONTEXT or latest.version_number < 2:
        raise ValueError("visibility enrichment requires an existing context-v2")
    source_root = project_path / latest.relative_path
    has_owner = (source_root / "aggregates" / "owner-context.json").is_file()
    has_visibility = (source_root / "aggregates" / "visibility-metrics.json").is_file()
    if not has_owner:
        raise ValueError("visibility enrichment requires persisted owner context")
    source_snapshot = validate_project_bundle(
        source_root,
        expected_project_id=project_id,
        expected_version_number=latest.version_number,
        expected_source_audit_id=latest.source_audit_id,
        expect_owner_context=True,
        expect_visibility_metrics=has_visibility,
    )
    source_run = AuditRun.model_validate_json(source_snapshot.files["engine/audit.json"].content)
    if source_run.audit_id != latest.audit_id:
        raise ValueError("persisted context audit identity does not match project history")
    persisted_owner_context = OwnerContext.model_validate_json(
        source_snapshot.files["aggregates/owner-context.json"].content
    )
    derived_report_status = derive_report_status(
        stage=AuditStage.CONTEXT,
        used_fact_ids=tuple(fact.fact_id for fact in persisted_owner_context.facts),
        owner_context=persisted_owner_context,
    )
    if latest.report_status is not derived_report_status:
        raise ValueError("persisted report status conflicts with trusted owner context")
    timestamp = now or datetime.now(UTC)
    visibility_bytes = _json_bytes(visibility.model_dump(mode="json"))
    visibility_sha256 = hashlib.sha256(visibility_bytes).hexdigest()
    audit_digest = hashlib.sha256(
        f"{source_run.audit_id}:{timestamp.isoformat()}:{visibility_sha256}".encode()
    ).hexdigest()[:16]
    audit_id = f"audit-{audit_digest}"
    config = AuditConfig(
        domain=str(source_run.configuration["target"]),
        max_pages=int(source_run.configuration["max_pages"]),
    )
    safe_client_stem = _client_stem(manifest.client_name)
    owner_bytes = source_snapshot.files["aggregates/owner-context.json"].content

    def build(pending: PendingVersion) -> AuditVersionRef:
        version_root = pending.staging_path
        engine_root = version_root / "engine"
        report_root = version_root / "report"
        for directory in (
            engine_root,
            report_root,
            version_root / "manifests",
            version_root / "aggregates",
        ):
            directory.mkdir(parents=True, exist_ok=False)
        run = source_run.model_copy(deep=True)
        run.audit_id = audit_id
        run.timestamp = timestamp
        run.configuration = {
            **source_run.configuration,
            "source_audit_id": source_run.audit_id,
            "visibility_snapshot_sha256": visibility_sha256,
            "visibility_metric_ids": [metric.metric_id for metric in visibility.metrics],
        }
        owner_path = version_root / "aggregates" / "owner-context.json"
        owner_path.write_bytes(owner_bytes)
        visibility_path = version_root / "aggregates" / "visibility-metrics.json"
        visibility_path.write_bytes(visibility_bytes)
        project_metadata = ProjectReportMetadata(
            project_id=project_id,
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.CONTEXT,
            report_status=derived_report_status,
            source_audit_id=source_run.audit_id,
        )
        compile_audit_run(
            run,
            output_dir=engine_root,
            report_locale=manifest.report_locale,
            project_metadata=project_metadata,
            owner_context=persisted_owner_context,
            visibility_snapshot=visibility,
        )
        report_name = (
            f"{safe_client_stem}_AI_Search_SEO_Audit_"
            f"{manifest.report_locale.upper()}_v{pending.version_number}.pdf"
        )
        _relocate_pdf(engine_root / "client-report.pdf", report_root / report_name)
        run.output_paths = {
            name: (f"report/{report_name}" if name == "client-report.pdf" else f"engine/{name}")
            for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
        }
        _atomic_json(engine_root / "audit.json", run)
        write_data_request(
            build_data_request(
                _data_request_context(run, project_id=project_id, locale=manifest.report_locale)
            ),
            version_root,
        )
        _write_manifests(
            version_root=version_root,
            project_id=project_id,
            run=run,
            config=config,
            report_locale=manifest.report_locale,
            source_audit_id=source_run.audit_id,
            owner_context_path="aggregates/owner-context.json",
            visibility_metrics_path="aggregates/visibility-metrics.json",
            version_number=pending.version_number,
        )
        validate_project_bundle(
            version_root,
            expected_project_id=project_id,
            expected_version_number=pending.version_number,
            expected_source_audit_id=source_run.audit_id,
            expect_owner_context=True,
            expect_visibility_metrics=True,
        )
        return AuditVersionRef(
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.CONTEXT,
            report_status=derived_report_status,
            audit_id=run.audit_id,
            source_audit_id=source_run.audit_id,
            created_at=pending.created_at,
            relative_path=pending.relative_path,
        )

    return store.build_version(
        project_id,
        stage=AuditStage.CONTEXT,
        builder=build,
        now=timestamp,
    )


def _prompt_pack_version(run: AuditRun) -> str | None:
    versions = {prompt.pack_version for prompt in run.ai_prompts}
    return next(iter(versions)) if len(versions) == 1 else None


def _canonical_prompt_ids(run: AuditRun) -> tuple[str, ...]:
    return tuple(sorted(prompt.prompt_id for prompt in run.ai_prompts))


def _seasonal_business(owner_context: OwnerContext | None) -> bool:
    if owner_context is None:
        return False
    values = (
        fact.value for fact in owner_context.facts if fact.field is OwnerFactField.SEASONALITY
    )
    return any(
        value is True
        or (
            isinstance(value, str)
            and value.strip().casefold() not in {"", "false", "no", "none", "unknown"}
        )
        for value in values
    )


def _visibility_setup_fingerprint(run: AuditRun) -> str | None:
    value = run.configuration.get("ai_observation_setup_fingerprint")
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        else None
    )


def validate_project(
    project_ref: str,
    *,
    clients_root: Path,
    intake_dir: Path | None,
    normalized_intake: NormalizedIntake | Mapping[str, object] | None,
    implementation_date: date,
    baseline_diagnostic_run: str | None = None,
    follow_up_diagnostic_run: str | None = None,
    now: datetime | None = None,
    selected_topics: tuple[PromptTopic, ...] | None = None,
) -> ProjectManifest:
    """Run a fresh public audit and create an immutable validation version."""
    if (intake_dir is None) != (normalized_intake is None):
        if intake_dir is not None:
            discard_owned_intake_dir(intake_dir, intake_root=intake_dir.parent)
        raise ValueError(
            "validation intake directory and normalized input must be supplied together"
        )
    follow_up_visibility: VisibilitySnapshot | None = None
    if intake_dir is not None and normalized_intake is not None:
        processed = consume_intake(
            intake_dir,
            normalized_intake,
            intake_root=intake_dir.parent,
            now=now,
            processor=_validate_visibility_intake,
        )
        follow_up_visibility = build_visibility_snapshot(processed)

    supplementary = None
    if (baseline_diagnostic_run is None) != (follow_up_diagnostic_run is None):
        raise ValueError("baseline and follow-up diagnostic runs must be supplied together")
    if baseline_diagnostic_run is not None and follow_up_diagnostic_run is not None:
        from .diagnostic_workflow import compare_diagnostic_runs, parse_diagnostic_run_reference

        supplementary = compare_diagnostic_runs(
            project_ref,
            clients_root=clients_root,
            baseline_reference=parse_diagnostic_run_reference(baseline_diagnostic_run),
            follow_up_reference=parse_diagnostic_run_reference(follow_up_diagnostic_run),
        )
    store = ProjectStore(clients_root)
    project_path = store.resolve(project_ref)
    project_id = project_path.name
    manifest = store.load(project_id)
    timestamp = now or datetime.now(UTC)
    latest = manifest.versions[-1]
    source_root = project_path / latest.relative_path
    has_owner = (source_root / "aggregates" / "owner-context.json").is_file()
    has_visibility = (source_root / "aggregates" / "visibility-metrics.json").is_file()
    has_comparison = (source_root / "aggregates" / "validation-comparison.json").is_file()
    source_snapshot = validate_project_bundle(
        source_root,
        expected_project_id=project_id,
        expected_version_number=latest.version_number,
        expected_source_audit_id=latest.source_audit_id,
        expect_owner_context=has_owner,
        expect_visibility_metrics=has_visibility,
        expect_validation_comparison=has_comparison,
        expected_stage=latest.stage,
    )
    source_run = AuditRun.model_validate_json(source_snapshot.files["engine/audit.json"].content)
    if source_run.audit_id != latest.audit_id:
        raise ValueError("persisted source audit identity does not match project history")
    owner_context = (
        OwnerContext.model_validate_json(
            source_snapshot.files["aggregates/owner-context.json"].content
        )
        if has_owner
        else None
    )
    baseline_visibility = (
        VisibilitySnapshot.model_validate_json(
            source_snapshot.files["aggregates/visibility-metrics.json"].content
        )
        if has_visibility
        else None
    )
    if follow_up_visibility is not None and (
        follow_up_visibility.project_id != project_id
        or follow_up_visibility.canonical_domain not in manifest.canonical_domains
    ):
        raise ProjectIdentityError("validation visibility intake identity mismatch")

    if saved_prompt_version(source_run) == SELECTED_PROMPT_PACK_VERSION:
        validate_automatic_prompt_pack(source_run)
        if selected_topics is None:
            selected_topics = tuple(
                PromptTopic.model_validate(item)
                for item in source_run.configuration["prompt_topic_selection"]
            )

    config = AuditConfig(
        domain=str(source_run.configuration["target"]),
        max_pages=int(source_run.configuration["max_pages"]),
    )
    safe_client_stem = _client_stem(manifest.client_name)

    def build(pending: PendingVersion) -> AuditVersionRef:
        version_root = pending.staging_path
        engine_root = version_root / "engine"
        report_root = version_root / "report"
        for directory in (
            engine_root,
            report_root,
            version_root / "manifests",
            version_root / "aggregates",
        ):
            directory.mkdir(parents=True, exist_ok=False)

        fresh_public = run_public_audit(
            config.domain,
            output_dir=engine_root,
            max_pages=config.max_pages,
            report_locale=manifest.report_locale,
            now=timestamp,
            selected_topics=selected_topics,
        )
        comparison = compare_visibility(
            baseline_visibility,
            follow_up_visibility,
            baseline_audit_id=source_run.audit_id,
            follow_up_audit_id=fresh_public.audit_id,
            implementation_date=implementation_date,
            observed_at=timestamp.date(),
            seasonal=_seasonal_business(owner_context),
        ).model_copy(
            update={
                "findings": compare_findings(source_run.findings, fresh_public.findings),
                "ai_visibility": compare_observed_ai_visibility(
                    source_run.ai_observations,
                    fresh_public.ai_observations,
                    baseline_prompt_pack_version=_prompt_pack_version(source_run),
                    follow_up_prompt_pack_version=_prompt_pack_version(fresh_public),
                    baseline_setup_fingerprint=_visibility_setup_fingerprint(source_run),
                    follow_up_setup_fingerprint=_visibility_setup_fingerprint(fresh_public),
                    baseline_canonical_prompt_ids=_canonical_prompt_ids(source_run),
                    follow_up_canonical_prompt_ids=_canonical_prompt_ids(fresh_public),
                ),
            }
        )
        run = fresh_public
        used_fact_ids: tuple[str, ...] = ()
        if owner_context is not None:
            run, used_fact_ids = _apply_owner_context(
                fresh_public,
                owner_context,
                audit_id=fresh_public.audit_id,
                timestamp=timestamp,
            )
            _atomic_json(version_root / "aggregates" / "owner-context.json", owner_context)
        visibility_path: str | None = None
        if follow_up_visibility is not None:
            visibility_file = version_root / "aggregates" / "visibility-metrics.json"
            _atomic_json(visibility_file, follow_up_visibility)
            visibility_bytes = _secure_read_regular_file(visibility_file).content
            run.configuration = {
                **run.configuration,
                "visibility_snapshot_sha256": hashlib.sha256(visibility_bytes).hexdigest(),
                "visibility_metric_ids": [
                    metric.metric_id for metric in follow_up_visibility.metrics
                ],
            }
            visibility_path = "aggregates/visibility-metrics.json"
        _atomic_json(
            version_root / "aggregates" / "validation-comparison.json",
            comparison,
        )
        run.configuration = {
            **run.configuration,
            "source_audit_id": source_run.audit_id,
            "baseline_audit_id": source_run.audit_id,
            "implementation_date": implementation_date.isoformat(),
            "validation_comparison_schema_version": comparison.schema_version,
        }
        if supplementary is not None:
            supplementary_file = (
                version_root / "aggregates/supplementary-diagnostic-comparison.json"
            )
            _atomic_json(supplementary_file, supplementary)
            run.configuration["supplementary_diagnostic_comparison_sha256"] = (
                _secure_read_regular_file(supplementary_file).sha256
            )
        report_status = (
            ReportStatus.CLIENT_CONTEXT_DRAFT
            if owner_context is None
            else derive_report_status(
                stage=AuditStage.VALIDATION,
                used_fact_ids=used_fact_ids,
                owner_context=owner_context,
            )
        )
        project_metadata = ProjectReportMetadata(
            project_id=project_id,
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.VALIDATION,
            report_status=report_status,
            source_audit_id=source_run.audit_id,
        )
        compile_audit_run(
            run,
            output_dir=engine_root,
            report_locale=manifest.report_locale,
            project_metadata=project_metadata,
            owner_context=owner_context,
            visibility_snapshot=follow_up_visibility,
            validation_comparison=comparison,
            supplementary_diagnostic_comparison=supplementary,
        )
        report_name = (
            f"{safe_client_stem}_AI_Search_SEO_Audit_"
            f"{manifest.report_locale.upper()}_v{pending.version_number}.pdf"
        )
        _relocate_pdf(engine_root / "client-report.pdf", report_root / report_name)
        run.output_paths = {
            name: (f"report/{report_name}" if name == "client-report.pdf" else f"engine/{name}")
            for name in sorted((*_ENGINE_ARTIFACTS, "client-report.pdf"))
        }
        _atomic_json(engine_root / "audit.json", run)
        write_data_request(
            build_data_request(
                _data_request_context(
                    run,
                    project_id=project_id,
                    locale=manifest.report_locale,
                )
            ),
            version_root,
        )
        _write_manifests(
            version_root=version_root,
            project_id=project_id,
            run=run,
            config=config,
            report_locale=manifest.report_locale,
            source_audit_id=source_run.audit_id,
            owner_context_path=("aggregates/owner-context.json" if owner_context else None),
            visibility_metrics_path=visibility_path,
            validation_comparison_path="aggregates/validation-comparison.json",
            version_number=pending.version_number,
        )
        validate_project_bundle(
            version_root,
            expected_project_id=project_id,
            expected_version_number=pending.version_number,
            expected_source_audit_id=source_run.audit_id,
            expect_owner_context=owner_context is not None,
            expect_visibility_metrics=follow_up_visibility is not None,
            expect_validation_comparison=True,
            expected_stage=AuditStage.VALIDATION,
        )
        return AuditVersionRef(
            version_id=pending.version_id,
            version_number=pending.version_number,
            stage=AuditStage.VALIDATION,
            report_status=report_status,
            audit_id=run.audit_id,
            source_audit_id=source_run.audit_id,
            created_at=pending.created_at,
            relative_path=pending.relative_path,
        )

    return store.build_version(
        project_id,
        stage=AuditStage.VALIDATION,
        builder=build,
        now=timestamp,
    )
