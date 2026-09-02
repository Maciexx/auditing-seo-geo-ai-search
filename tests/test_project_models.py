from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from ai_search_audit.models import DataState
from ai_search_audit.project_models import (
    AuditStage,
    AuditVersionRef,
    ProjectManifest,
    ReportStatus,
    normalize_canonical_domain,
    validate_project_id,
)

_NON_PORTABLE_COMPONENTS = (
    "nul\x00byte",
    "line\nbreak",
    "delete\x7fchar",
    "bad<name",
    "bad>name",
    'bad"name',
    "bad:name",
    r"bad\name",
    "bad|name",
    "bad?name",
    "bad*name",
    "trailing.",
    "trailing ",
    "CON",
    "prn.txt",
    "AuX",
    "nul.json",
    "COM1",
    "com9.log",
    "LPT1",
    "lpt9.txt",
    "x" * 256,
)


def version_fixture(
    *,
    version_number: int = 1,
    stage: AuditStage = AuditStage.PUBLIC,
    report_status: ReportStatus = ReportStatus.PUBLIC_EVIDENCE_DRAFT,
    audit_id: str | None = None,
    version_id: str | None = None,
    relative_path: str | None = None,
    source_audit_id: str | None = None,
) -> AuditVersionRef:
    return AuditVersionRef(
        version_id=version_id if version_id is not None else f"version-{version_number}",
        version_number=version_number,
        stage=stage,
        report_status=report_status,
        audit_id=audit_id if audit_id is not None else f"audit-{version_number}",
        source_audit_id=source_audit_id,
        created_at=datetime(2026, 8, 31, 10, tzinfo=UTC) + timedelta(minutes=version_number),
        relative_path=(
            relative_path if relative_path is not None else f"versions/{version_number}"
        ),
    )


def project_manifest_fixture(
    *,
    version_numbers: tuple[int, ...] = (1,),
    **overrides: Any,
) -> ProjectManifest:
    versions = tuple(version_fixture(version_number=number) for number in version_numbers)
    values: dict[str, Any] = {
        "project_id": "example-studio",
        "client_name": "Example Studio",
        "canonical_domains": ("example-studio.example",),
        "report_locale": "pl",
        "created_at": datetime(2026, 8, 31, 10, tzinfo=UTC),
        "latest_audit_id": versions[-1].audit_id,
        "source_files_policy": "delete-after-processing",
        "versions": versions,
    }
    values.update(overrides)
    return ProjectManifest(**values)


def test_public_domain_normalizer_matches_manifest_canonicalization() -> None:
    value = "HTTPS://Example-Studio.Example/services?ref=a#overview"

    assert (
        normalize_canonical_domain(value)
        == project_manifest_fixture(canonical_domains=(value,)).canonical_domains[0]
    )


def test_public_project_id_validator_matches_manifest_portability_rules() -> None:
    with pytest.raises(ValueError, match="portable filesystem component"):
        validate_project_id("CON")


def test_project_manifest_accepts_safe_project_id() -> None:
    manifest = project_manifest_fixture(project_id="example-studio")

    assert manifest.project_id == "example-studio"


@pytest.mark.parametrize("client_name", ("", "   "))
def test_project_manifest_rejects_blank_client_names(client_name: str) -> None:
    with pytest.raises(ValidationError, match="client_name must not be blank"):
        project_manifest_fixture(client_name=client_name)


@pytest.mark.parametrize(
    "project_id",
    (
        "../example-studio",
        "example/studio",
        r"example\studio",
        "   ",
        "/tmp/example-studio",
    ),
)
def test_project_manifest_rejects_unsafe_project_ids(project_id: str) -> None:
    with pytest.raises(ValidationError, match="safe identifier"):
        project_manifest_fixture(project_id=project_id)


def test_project_manifest_normalizes_canonical_domains_to_lowercase_hostnames() -> None:
    manifest = project_manifest_fixture(
        canonical_domains=(
            "HTTPS://Example-Studio.Example/services?ref=a#overview",
            "WWW.Secondary.Example/about/",
        )
    )

    assert manifest.canonical_domains == (
        "example-studio.example",
        "www.secondary.example",
    )


def test_project_manifest_normalizes_canonical_domains_from_a_set() -> None:
    manifest = project_manifest_fixture(
        canonical_domains={"HTTPS://Example-Studio.Example/about", "SECONDARY.Example"}
    )

    assert set(manifest.canonical_domains) == {
        "example-studio.example",
        "secondary.example",
    }


def test_project_manifest_normalizes_canonical_domains_from_a_generator() -> None:
    domains = (domain for domain in ("FIRST.Example/path", "https://SECOND.Example"))

    manifest = project_manifest_fixture(canonical_domains=domains)

    assert manifest.canonical_domains == ("first.example", "second.example")


def test_project_manifest_rejects_non_string_canonical_domain_with_validation_error() -> None:
    with pytest.raises(ValidationError):
        project_manifest_fixture(canonical_domains=("example-studio.example", 7))


@pytest.mark.parametrize(
    "canonical_domain",
    ("", "https:///missing.example", "https://user:pass@example-studio.example"),
)
def test_project_manifest_rejects_invalid_canonical_domains(canonical_domain: str) -> None:
    with pytest.raises(ValidationError, match="canonical domain"):
        project_manifest_fixture(canonical_domains=(canonical_domain,))


def test_project_manifest_is_deeply_immutable() -> None:
    manifest = project_manifest_fixture()
    with pytest.raises(ValidationError, match="frozen"):
        manifest.versions[0].report_status = ReportStatus.CLIENT_VALIDATED


def test_project_manifest_rejects_non_monotonic_versions() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        project_manifest_fixture(version_numbers=(1, 3, 2))


@pytest.mark.parametrize("duplicate_field", ("audit_id", "version_id", "relative_path"))
def test_project_manifest_rejects_duplicate_version_identity(duplicate_field: str) -> None:
    first = version_fixture(version_number=1)
    duplicate_value = getattr(first, duplicate_field)
    second_values = {
        "audit_id": "audit-2",
        "version_id": "version-2",
        "relative_path": "versions/2",
    }
    second_values[duplicate_field] = duplicate_value
    second = version_fixture(version_number=2, **second_values)

    with pytest.raises(ValidationError, match=f"unique {duplicate_field}"):
        project_manifest_fixture(
            versions=(first, second),
            latest_audit_id=second.audit_id,
        )


@pytest.mark.parametrize(
    ("stage", "report_status"),
    (
        (AuditStage.PUBLIC, ReportStatus.CLIENT_CONTEXT_DRAFT),
        (AuditStage.PUBLIC, ReportStatus.CLIENT_VALIDATED),
        (AuditStage.CONTEXT, ReportStatus.PUBLIC_EVIDENCE_DRAFT),
        (AuditStage.VALIDATION, ReportStatus.PUBLIC_EVIDENCE_DRAFT),
    ),
)
def test_audit_version_rejects_invalid_stage_status_combinations(
    stage: AuditStage, report_status: ReportStatus
) -> None:
    with pytest.raises(ValidationError, match="stage requires report status"):
        version_fixture(stage=stage, report_status=report_status)


@pytest.mark.parametrize(
    ("stage", "report_status"),
    (
        (AuditStage.PUBLIC, ReportStatus.PUBLIC_EVIDENCE_DRAFT),
        (AuditStage.CONTEXT, ReportStatus.CLIENT_CONTEXT_DRAFT),
        (AuditStage.CONTEXT, ReportStatus.CLIENT_VALIDATED),
        (AuditStage.VALIDATION, ReportStatus.CLIENT_CONTEXT_DRAFT),
        (AuditStage.VALIDATION, ReportStatus.CLIENT_VALIDATED),
    ),
)
def test_audit_version_accepts_matching_stage_status_combinations(
    stage: AuditStage, report_status: ReportStatus
) -> None:
    version = version_fixture(stage=stage, report_status=report_status)

    assert (version.stage, version.report_status) == (stage, report_status)


@pytest.mark.parametrize(
    "field_name",
    ("version_id", "audit_id", "source_audit_id"),
)
def test_audit_version_rejects_unsafe_identifiers(field_name: str) -> None:
    with pytest.raises(ValidationError, match="safe identifier"):
        version_fixture(**{field_name: "../unsafe"})


@pytest.mark.parametrize("field_name", ("version_id", "audit_id", "relative_path"))
def test_audit_version_rejects_empty_identity_and_path_values(field_name: str) -> None:
    with pytest.raises(ValidationError):
        version_fixture(**{field_name: ""})


@pytest.mark.parametrize("unsafe_component", _NON_PORTABLE_COMPONENTS)
def test_project_id_rejects_non_portable_filesystem_components(
    unsafe_component: str,
) -> None:
    with pytest.raises(ValidationError, match="portable filesystem component"):
        project_manifest_fixture(project_id=unsafe_component)


@pytest.mark.parametrize("unsafe_component", _NON_PORTABLE_COMPONENTS)
def test_relative_path_rejects_non_portable_filesystem_components(
    unsafe_component: str,
) -> None:
    with pytest.raises(ValidationError, match="portable filesystem component"):
        version_fixture(relative_path=f"versions/{unsafe_component}")


def test_relative_path_accepts_component_at_255_utf8_bytes() -> None:
    component = f"{'x' * 251}🙂"

    version = version_fixture(relative_path=f"versions/{component}")

    assert version.relative_path == f"versions/{component}"


def test_relative_path_rejects_component_over_255_utf8_bytes() -> None:
    component = f"{'x' * 252}🙂"

    with pytest.raises(ValidationError, match="portable filesystem component"):
        version_fixture(relative_path=f"versions/{component}")


def test_project_id_rejects_lone_surrogate_as_validation_error() -> None:
    with pytest.raises(ValidationError, match="portable filesystem component"):
        project_manifest_fixture(project_id="example-\ud800studio")


def test_relative_path_rejects_lone_surrogate_as_validation_error() -> None:
    with pytest.raises(ValidationError, match="portable filesystem component"):
        version_fixture(relative_path="versions/example-\ud800studio")


@pytest.mark.parametrize(
    "relative_path",
    ("/absolute/version", "../outside", "versions/../../outside", r"versions\1", ".", "   "),
)
def test_audit_version_rejects_unsafe_relative_paths(relative_path: str) -> None:
    with pytest.raises(ValidationError, match="relative path"):
        version_fixture(relative_path=relative_path)


def test_project_manifest_latest_audit_points_to_last_version() -> None:
    with pytest.raises(ValidationError, match="latest_audit_id"):
        project_manifest_fixture(version_numbers=(1, 2), latest_audit_id="audit-1")


def test_project_manifest_requires_at_least_one_version() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        project_manifest_fixture(versions=(), latest_audit_id="audit-1")


def test_project_manifest_does_not_accept_client_root_paths() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        project_manifest_fixture(client_root="/srv/client/projects/example-studio")


def test_unavailable_unknown_and_failed_data_states_remain_distinct() -> None:
    states = {DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED}

    assert len(states) == 3
    assert {state.value for state in states} == {"UNAVAILABLE", "UNKNOWN", "FAILED"}
