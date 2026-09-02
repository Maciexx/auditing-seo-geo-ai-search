from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


class FrozenProjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class AuditStage(StrEnum):
    PUBLIC = "public"
    CONTEXT = "context"
    VALIDATION = "validation"


class ReportStatus(StrEnum):
    PUBLIC_EVIDENCE_DRAFT = "PUBLIC_EVIDENCE_DRAFT"
    CLIENT_CONTEXT_DRAFT = "CLIENT_CONTEXT_DRAFT"
    CLIENT_VALIDATED = "CLIENT_VALIDATED"


def _validate_portable_filesystem_component(value: str) -> str:
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("value must be a UTF-8 portable filesystem component") from exc

    reserved_stem = value.partition(".")[0].upper()
    if (
        not value
        or encoded_length > 255
        or value.endswith((".", " "))
        or reserved_stem in _WINDOWS_RESERVED_NAMES
        or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("value must be a portable filesystem component")
    return value


def _validate_safe_identifier(value: str, *, field_name: str) -> str:
    try:
        _validate_portable_filesystem_component(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a safe identifier and portable filesystem component"
        ) from exc
    if not _SAFE_IDENTIFIER.fullmatch(value) or ".." in value:
        raise ValueError(f"{field_name} must be a safe identifier")
    return value


def validate_project_id(value: str) -> str:
    """Validate a project ID using the manifest's portable path rules."""
    return _validate_safe_identifier(value, field_name="manifest identifier")


def normalize_canonical_domain(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate != value:
        raise ValueError("canonical domain must be a non-empty hostname or URL")

    parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
    if parsed.hostname is None or parsed.username is not None or parsed.password is not None:
        raise ValueError(
            "canonical domain must not contain credentials and must include a hostname"
        )

    try:
        hostname = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        _ = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("canonical domain must contain a valid hostname") from exc

    labels = hostname.split(".")
    if (
        not hostname
        or len(hostname) > 253
        or any(len(label) > 63 or not _HOST_LABEL.fullmatch(label) for label in labels)
    ):
        raise ValueError("canonical domain must contain a valid hostname")
    return hostname


def _validate_relative_path(value: str) -> str:
    if not value:
        raise ValueError("relative path must be a normalized, non-empty POSIX path")

    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        value == "."
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("relative path must remain within the project directory")

    try:
        for component in path.parts:
            _validate_portable_filesystem_component(component)
    except ValueError as exc:
        raise ValueError("relative path contains an invalid portable filesystem component") from exc
    return value


class AuditVersionRef(FrozenProjectModel):
    version_id: str
    version_number: int = Field(ge=1)
    stage: AuditStage
    report_status: ReportStatus
    audit_id: str
    source_audit_id: str | None = None
    created_at: datetime
    relative_path: str

    @field_validator("version_id", "audit_id")
    @classmethod
    def validate_required_identifiers(cls, value: str) -> str:
        return _validate_safe_identifier(value, field_name="version identifier")

    @field_validator("source_audit_id")
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_safe_identifier(value, field_name="source audit identifier")

    @field_validator("relative_path")
    @classmethod
    def validate_version_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_stage_status(self) -> AuditVersionRef:
        allowed_statuses = {
            AuditStage.PUBLIC: frozenset({ReportStatus.PUBLIC_EVIDENCE_DRAFT}),
            AuditStage.CONTEXT: frozenset(
                {ReportStatus.CLIENT_CONTEXT_DRAFT, ReportStatus.CLIENT_VALIDATED}
            ),
            AuditStage.VALIDATION: frozenset(
                {ReportStatus.CLIENT_CONTEXT_DRAFT, ReportStatus.CLIENT_VALIDATED}
            ),
        }[self.stage]
        if self.report_status not in allowed_statuses:
            allowed_values = ", ".join(sorted(status.value for status in allowed_statuses))
            raise ValueError(
                f"{self.stage.value} stage requires report status in: {allowed_values}"
            )
        return self


class ProjectManifest(FrozenProjectModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    client_name: str
    canonical_domains: tuple[str, ...] = Field(min_length=1)
    report_locale: Literal["pl", "en"]
    created_at: datetime
    latest_audit_id: str
    source_files_policy: Literal["delete-after-processing"]
    versions: tuple[AuditVersionRef, ...] = Field(min_length=1)

    @field_validator("project_id")
    @classmethod
    def validate_project_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("latest_audit_id")
    @classmethod
    def validate_latest_audit_identifier(cls, value: str) -> str:
        return _validate_safe_identifier(value, field_name="manifest identifier")

    @field_validator("client_name")
    @classmethod
    def validate_client_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("client_name must not be blank")
        return value

    @field_validator("canonical_domains")
    @classmethod
    def normalize_canonical_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_canonical_domain(domain) for domain in value)

    @model_validator(mode="after")
    def validate_version_history(self) -> ProjectManifest:
        version_numbers = [version.version_number for version in self.versions]
        version_pairs = zip(version_numbers, version_numbers[1:], strict=False)
        if any(current >= following for current, following in version_pairs):
            raise ValueError("version numbers must be strictly increasing")

        for field_name in ("audit_id", "version_id", "relative_path"):
            values = [getattr(version, field_name) for version in self.versions]
            if len(values) != len(set(values)):
                raise ValueError(f"version history requires unique {field_name} values")

        if self.latest_audit_id != self.versions[-1].audit_id:
            raise ValueError("latest_audit_id must point to the last version")
        return self
