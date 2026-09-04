from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from ai_search_audit.diagnostic_models import DiagnosticBinding, DiagnosticSource, FrozenPrompt
from ai_search_audit.models import AuditRun
from ai_search_audit.project_models import (
    AuditVersionRef,
    ProjectManifest,
    normalize_canonical_domain,
    validate_project_id,
)
from ai_search_audit.project_store import ProjectStore
from ai_search_audit.report_models import ClientReportData, ProjectReportMetadata

_SourceKey = tuple[Path, str, str]
_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(slots=True)
class _ValidationScope:
    sources: dict[_SourceKey, DiagnosticSource] = field(default_factory=dict)
    active_sources: set[_SourceKey] = field(default_factory=set)


_VALIDATION_SCOPE: ContextVar[_ValidationScope | None] = ContextVar(
    "diagnostic_source_validation_scope", default=None
)


def _diagnostic_validation_operation(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Reuse only complete source validations within one synchronous read operation.

    The ContextVar transports a transient scope, not a persistent cache. Never wrap
    READY coordination or collection workflows: each later phase must read afresh.
    """

    @wraps(function)
    def operation(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if _VALIDATION_SCOPE.get() is not None:
            return function(*args, **kwargs)
        token = _VALIDATION_SCOPE.set(_ValidationScope())
        try:
            return function(*args, **kwargs)
        finally:
            _VALIDATION_SCOPE.reset(token)

    return operation


@_diagnostic_validation_operation
def load_diagnostic_source(
    project_ref: str,
    *,
    clients_root: str | Path,
    source_version: str,
) -> DiagnosticSource:
    """Read an exact, validated audit version without modifying canonical artifacts."""
    from ai_search_audit.diagnostic_workflow import _project_directory, _real_directory_path

    source_version = validate_project_id(source_version)
    store = ProjectStore(_real_directory_path(clients_root))
    project_root = _project_directory(store.resolve(project_ref))
    manifest = store.load(project_root.name)
    version = next(
        (version for version in manifest.versions if version.version_id == source_version),
        None,
    )
    if version is None:
        raise ValueError(f"unknown source version {source_version!r} for {manifest.project_id!r}")

    scope = _VALIDATION_SCOPE.get()
    assert scope is not None
    key = (project_root.resolve(strict=True), source_version, version.audit_id)
    if key in scope.active_sources:
        raise ValueError("cyclic diagnostic source validation")
    if key in scope.sources:
        return scope.sources[key]
    scope.active_sources.add(key)
    try:
        source = _load_validated_source(project_root, manifest, version)
        scope.sources[key] = source
        return source
    finally:
        scope.active_sources.remove(key)


def _load_validated_source(
    project_root: Path, manifest: ProjectManifest, version: AuditVersionRef
) -> DiagnosticSource:
    from ai_search_audit.project_orchestrator import validate_project_bundle

    version_root = project_root / version.relative_path
    aggregates = version_root / "aggregates"
    snapshot = validate_project_bundle(
        version_root,
        expected_project_id=manifest.project_id,
        expected_version_number=version.version_number,
        expected_source_audit_id=version.source_audit_id,
        expect_owner_context=(aggregates / "owner-context.json").is_file(),
        expect_visibility_metrics=(aggregates / "visibility-metrics.json").is_file(),
        expect_validation_comparison=(aggregates / "validation-comparison.json").is_file(),
        expected_stage=version.stage,
    )
    audit_file = snapshot.files["engine/audit.json"]
    audit = AuditRun.model_validate_json(audit_file.content)
    report = ClientReportData.model_validate_json(
        snapshot.files["engine/client-report-data.json"].content
    )
    if audit.audit_id != version.audit_id:
        raise ValueError("diagnostic source audit identity does not match project history")
    domain = normalize_canonical_domain(audit.site.domain)
    if domain not in manifest.canonical_domains:
        raise ValueError("diagnostic source domain does not match project identity")
    if report.report_locale != manifest.report_locale:
        raise ValueError("diagnostic source report locale does not match project identity")
    expected_metadata = ProjectReportMetadata(
        project_id=manifest.project_id,
        version_id=version.version_id,
        version_number=version.version_number,
        stage=version.stage,
        report_status=version.report_status,
        source_audit_id=version.source_audit_id,
    )
    if report.project != expected_metadata:
        raise ValueError("diagnostic source report metadata does not match project history")

    return DiagnosticSource(
        binding=DiagnosticBinding(
            project_id=manifest.project_id,
            source_version=version.version_id,
            audit_id=audit.audit_id,
            report_locale=manifest.report_locale,
            domain=domain,
            source_sha256=audit_file.sha256,
        ),
        prompts=tuple(
            FrozenPrompt.model_validate(prompt.model_dump()) for prompt in audit.ai_prompts
        ),
        page_urls=tuple(str(page.url) for page in audit.pages),
        canonical_domains=manifest.canonical_domains,
    )
