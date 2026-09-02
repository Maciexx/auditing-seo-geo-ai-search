"""Ephemeral intake for supported local POSIX, single-writer workflows.

The caller must exclusively control the intake root from capability creation
through consume or discard. Concurrent same-user mutation is outside this
boundary; detected identity races fail closed without reporting deletion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .models import DataState
from .project_models import normalize_canonical_domain, validate_project_id

_OWNERSHIP_MARKER = ".ai-search-audit-owned-intake.json"
_MARKER_SCHEMA_VERSION = "1.0.0"
_SUPPORTED_SUFFIXES = frozenset({".csv", ".xlsx", ".json", ".pdf", ".png", ".jpg", ".jpeg"})
MAX_INTAKE_SOURCES = 32
MAX_OWNER_FACTS = 500
MAX_VISIBILITY_METRIC_SERIES = 200
MAX_CITED_EXAMPLES = 100
MAX_METRIC_POINTS = 512
MAX_SOURCE_FILTERS = 32
MAX_METRIC_DIMENSIONS = 32
MAX_METRIC_FILTERS = 32
_SAFE_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._ -]*[A-Za-z0-9])?$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_RUN_ID = re.compile(r"^run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
_HASH_CHUNK_SIZE = 1024 * 1024
_QUARANTINE_PAYLOAD = "owned"
_METADATA_TOKEN = re.compile(r"[a-z0-9]+")
_ACRONYM_WORD_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_LOWER_UPPER_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_FORBIDDEN_VISIBILITY_METADATA_TOKENS = frozenset(
    {
        "conversion",
        "conversions",
        "purchase",
        "purchases",
        "revenue",
        "revenues",
        "crm",
        "lead",
        "leads",
    }
)
_FORBIDDEN_VISIBILITY_METADATA_COMPOUNDS = frozenset(
    {
        "conversion",
        "conversions",
        "customerid",
        "customerids",
        "customeridentifier",
        "customeridentifiers",
        "formsubmission",
        "formsubmissions",
        "formsubmit",
        "formsubmits",
        "formsubmitted",
        "leadformsubmission",
        "generatelead",
        "keyevent",
        "keyevents",
        "purchase",
        "purchases",
        "revenue",
        "revenues",
        "userid",
        "userids",
        "useridentifier",
        "useridentifiers",
        "userlevelevent",
        "userlevelevents",
    }
)

ScalarFactValue: TypeAlias = str | int | float | bool | None
IntakeProcessor: TypeAlias = Callable[["NormalizedIntake"], None]


class IntakeError(RuntimeError):
    """Base error for the ephemeral normalized intake boundary."""


class IntakeValidationError(IntakeError):
    """The normalized declaration or a declared artifact failed validation."""


class IntakeOwnershipError(IntakeError):
    """The directory is not proven to be owned by this intake boundary."""


class IntakeCleanupError(IntakeError):
    """A verified owned input directory could not be completely deleted."""


def _require_supported_posix_primitives() -> None:
    required_flags = {
        "O_NOFOLLOW": getattr(os, "O_NOFOLLOW", 0),
        "O_DIRECTORY": getattr(os, "O_DIRECTORY", 0),
    }
    missing = [name for name, value in required_flags.items() if not value]
    supported_dir_fd_names = {function.__name__ for function in getattr(os, "supports_dir_fd", ())}
    required_dir_fd = ("mkdir", "open", "stat", "unlink", "rmdir", "rename")
    missing.extend(
        operation for operation in required_dir_fd if operation not in supported_dir_fd_names
    )
    supported_nofollow_names = {
        function.__name__ for function in getattr(os, "supports_follow_symlinks", ())
    }
    if "stat" not in supported_nofollow_names:
        missing.append("stat(follow_symlinks=False)")
    supported_fd_names = {function.__name__ for function in getattr(os, "supports_fd", ())}
    if "listdir" not in supported_fd_names:
        missing.append("listdir(fd)")
    if not callable(getattr(os, "fstat", None)):
        missing.append("fstat(fd)")
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise IntakeOwnershipError(
            f"supported local POSIX intake platform primitives are unavailable: {names}"
        )


class FrozenIntakeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        allow_inf_nan=False,
    )


class VisibilitySource(StrEnum):
    GOOGLE_SEARCH_CONSOLE = "google_search_console"
    GOOGLE_ANALYTICS = "google_analytics"
    BING_WEBMASTER_TOOLS = "bing_webmaster_tools"
    GOOGLE_BUSINESS_PROFILE = "google_business_profile"
    MERCHANT_CENTER = "merchant_center"
    SANITIZED_LOGS = "sanitized_logs"
    CRAWL = "crawl"
    AI_MONITORING = "ai_monitoring"
    FIRST_PARTY = "first_party"
    MANUAL = "manual"
    OTHER = "other"


class FactApprovalState(StrEnum):
    UNKNOWN = "UNKNOWN"
    APPROVED = "APPROVED"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"


class DateRange(FrozenIntakeModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("date range end must not precede start")
        return self


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not _SAFE_ID.fullmatch(value) or ".." in value:
        raise ValueError(f"{field_name} must be a safe identifier")
    return value


def _validate_label(value: str, *, field_name: str) -> str:
    if not _SAFE_LABEL.fullmatch(value) or ".." in value:
        raise ValueError(f"{field_name} must be a bounded plain-text label")
    return value


def _validate_bounded_text(value: str, *, field_name: str, max_length: int = 500) -> str:
    if (
        not value
        or value != value.strip()
        or len(value) > max_length
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field_name} must be non-blank bounded text")
    return value


def canonical_metadata_tokens(value: str) -> tuple[str, ...]:
    """Tokenize separators, camelCase, and acronym-to-word boundaries consistently."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _ACRONYM_WORD_BOUNDARY.sub(r"\1_\2", normalized)
    normalized = _LOWER_UPPER_BOUNDARY.sub(r"\1_\2", normalized)
    return tuple(_METADATA_TOKEN.findall(normalized.casefold()))


def _canonical_token_ngrams(tokens: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        "".join(tokens[start:end])
        for start in range(len(tokens))
        for end in range(start + 1, len(tokens) + 1)
    )


def validate_visibility_only_label(value: str, *, field_name: str) -> str:
    """Reject commercial outcomes and user-level identifiers under separator variants."""
    validated = _validate_bounded_text(value, field_name=field_name, max_length=200)
    tokens = canonical_metadata_tokens(validated)
    token_set = set(tokens)
    semantic_terms = _canonical_token_ngrams(tokens)
    forbidden_terms = (
        _FORBIDDEN_VISIBILITY_METADATA_TOKENS | _FORBIDDEN_VISIBILITY_METADATA_COMPOUNDS
    )
    forbidden_sequence = (
        ("customer" in token_set and bool(token_set & {"id", "ids", "identifier", "identifiers"}))
        or ("user" in token_set and bool(token_set & {"id", "ids", "identifier", "identifiers"}))
        or ({"form", "submission"}.issubset(token_set))
        or ({"form", "submissions"}.issubset(token_set))
        or ({"form", "submit"}.issubset(token_set))
        or ({"form", "submits"}.issubset(token_set))
        or ({"form", "submitted"}.issubset(token_set))
        or ({"key", "event"}.issubset(token_set))
        or ({"key", "events"}.issubset(token_set))
        or ({"user", "level", "event"}.issubset(token_set))
        or ({"user", "level", "events"}.issubset(token_set))
    )
    if (
        token_set & _FORBIDDEN_VISIBILITY_METADATA_TOKENS
        or semantic_terms & forbidden_terms
        or forbidden_sequence
    ):
        raise ValueError(f"{field_name} violates the visibility-only boundary")
    return validated


def _validate_safe_relative_filename(value: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    try:
        component_too_long = any(len(part.encode("utf-8")) > 255 for part in posix_path.parts)
    except UnicodeEncodeError as exc:
        raise ValueError("filename must be a safe relative path") from exc
    if (
        not value
        or value == "."
        or value.endswith("/")
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != value
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or any("\\" in part or "\x00" in part for part in posix_path.parts)
        or component_too_long
    ):
        raise ValueError("filename must be a safe relative path")
    if posix_path.name == _OWNERSHIP_MARKER:
        raise ValueError("filename must not use the reserved intake marker name")
    if posix_path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
        raise ValueError("filename must use a supported source suffix")
    return value


class SourceArtifactDeclaration(FrozenIntakeModel):
    source_id: str
    filename: str
    sha256: str
    byte_count: int = Field(ge=0, strict=True)
    platform: VisibilitySource
    report_type: str
    date_range: DateRange | None = None
    filters: tuple[str, ...] = Field(default=(), max_length=MAX_SOURCE_FILTERS)
    exported_at: datetime | None = None

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="source_id")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _validate_safe_relative_filename(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return value.lower()

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, value: str) -> str:
        validated = validate_visibility_only_label(value, field_name="report_type")
        return _validate_identifier(validated, field_name="report_type")

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(
            validate_visibility_only_label(item, field_name="filter") for item in value
        )
        if len(validated) != len(set(validated)):
            raise ValueError("source filters must contain unique labels")
        return validated

    @field_validator("exported_at")
    @classmethod
    def validate_exported_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _validate_aware_timestamp(value)

    @model_validator(mode="after")
    def validate_export_range(self) -> SourceArtifactDeclaration:
        if (
            self.exported_at is not None
            and self.date_range is not None
            and self.date_range.end > self.exported_at.date()
        ):
            raise ValueError("source date range must not end after exported_at")
        return self


def _validate_aware_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("trusted timestamps must include a timezone")
    return value


def _validate_timestamp_order(processed_at: datetime, deleted_at: datetime) -> None:
    if deleted_at < processed_at:
        raise ValueError("deleted_at must not precede processed_at")


class SourceArtifactProvenance(SourceArtifactDeclaration):
    processed_at: datetime
    deleted_at: datetime

    @field_validator("processed_at", "deleted_at")
    @classmethod
    def validate_trusted_timestamps(cls, value: datetime) -> datetime:
        return _validate_aware_timestamp(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> SourceArtifactProvenance:
        _validate_timestamp_order(self.processed_at, self.deleted_at)
        if self.exported_at is not None and self.exported_at > self.processed_at:
            raise ValueError("source exported_at must not follow processed_at")
        if self.date_range is not None and self.date_range.end > self.processed_at.date():
            raise ValueError("source date range must not end after processed_at")
        return self


class OwnerFactInput(FrozenIntakeModel):
    fact_id: str
    field: str
    value: ScalarFactValue
    source_id: str
    as_of: date | None = None
    approval_state: FactApprovalState = FactApprovalState.UNKNOWN
    conflict_ids: tuple[str, ...] = ()
    resolved_conflict_ids: tuple[str, ...] = ()

    @field_validator("fact_id", "source_id")
    @classmethod
    def validate_ids(cls, value: str, info: ValidationInfo) -> str:
        return _validate_identifier(value, field_name=info.field_name or "identifier")

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        return _validate_identifier(value, field_name="field")

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: ScalarFactValue) -> ScalarFactValue:
        if isinstance(value, str) and (not value or value != value.strip() or len(value) > 2_000):
            raise ValueError("owner fact string values must be non-blank and bounded")
        return value

    @field_validator("conflict_ids", "resolved_conflict_ids")
    @classmethod
    def validate_conflict_ids(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        field_name = info.field_name or "conflict identifiers"
        validated = tuple(
            _validate_identifier(identifier, field_name=field_name) for identifier in value
        )
        if len(validated) != len(set(validated)):
            raise ValueError(f"owner facts require unique {field_name} values")
        return validated

    @model_validator(mode="after")
    def validate_disjoint_conflicts(self) -> OwnerFactInput:
        if set(self.conflict_ids) & set(self.resolved_conflict_ids):
            raise ValueError("conflict_ids and resolved_conflict_ids must be disjoint")
        return self


class VisibilityMetricPoint(FrozenIntakeModel):
    period_start: date
    period_end: date | None = None
    value: float = Field(strict=True)

    @model_validator(mode="after")
    def validate_period(self) -> VisibilityMetricPoint:
        if self.period_end is not None and self.period_end < self.period_start:
            raise ValueError("metric period_end must not precede period_start")
        return self


class VisibilityMetricSeriesInput(FrozenIntakeModel):
    metric_id: str
    source_id: str
    metric: str
    unit: str
    state: DataState
    coverage: float = Field(ge=0, le=1, strict=True)
    confidence: float = Field(ge=0, le=1, strict=True)
    dimensions: tuple[str, ...] = Field(default=(), max_length=MAX_METRIC_DIMENSIONS)
    filters: tuple[str, ...] = Field(default=(), max_length=MAX_METRIC_FILTERS)
    segment_label: str | None = None
    points: tuple[VisibilityMetricPoint, ...] = Field(default=(), max_length=MAX_METRIC_POINTS)

    @field_validator("source_id")
    @classmethod
    def validate_ids(cls, value: str, info: ValidationInfo) -> str:
        return _validate_identifier(value, field_name=info.field_name or "identifier")

    @field_validator("metric_id")
    @classmethod
    def validate_metric_id(cls, value: str) -> str:
        validated = validate_visibility_only_label(value, field_name="metric_id")
        return _validate_identifier(validated, field_name="metric_id")

    @field_validator("metric", "unit")
    @classmethod
    def validate_metric_labels(cls, value: str, info: ValidationInfo) -> str:
        return _validate_label(value, field_name=info.field_name or "label")

    @field_validator("dimensions", "filters")
    @classmethod
    def validate_dimension_and_filter_labels(
        cls, value: tuple[str, ...], info: ValidationInfo
    ) -> tuple[str, ...]:
        field_name = info.field_name or "metric labels"
        labels = tuple(
            validate_visibility_only_label(item, field_name=field_name) for item in value
        )
        if len(labels) != len(set(labels)):
            raise ValueError(f"{field_name} must contain unique labels")
        return labels

    @field_validator("segment_label")
    @classmethod
    def validate_segment_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_visibility_only_label(value, field_name="segment_label")

    @model_validator(mode="after")
    def validate_state_points_relation(self) -> VisibilityMetricSeriesInput:
        numeric_states = {DataState.AVAILABLE, DataState.PARTIAL}
        if self.state in numeric_states and not self.points:
            raise ValueError(f"{self.state.value} metric series requires at least one point")
        if self.state in numeric_states and self.coverage <= 0:
            raise ValueError(f"{self.state.value} metric series requires positive coverage")
        if self.state in numeric_states and self.confidence <= 0:
            raise ValueError(f"{self.state.value} metric series requires positive confidence")
        if self.state is DataState.AVAILABLE and self.coverage != 1:
            raise ValueError("AVAILABLE metric series requires full coverage")
        if self.state is DataState.PARTIAL and self.coverage >= 1:
            raise ValueError("PARTIAL metric series requires coverage below one")
        if self.state not in numeric_states and self.points:
            raise ValueError(f"{self.state.value} metric series must not contain points")
        return self


class NonSensitiveExample(FrozenIntakeModel):
    example_id: str
    source_id: str
    description: str = Field(min_length=1, max_length=2_000)
    citation: str = Field(min_length=1, max_length=2_000)
    non_sensitive: Literal[True]

    @field_validator("example_id", "source_id")
    @classmethod
    def validate_ids(cls, value: str, info: ValidationInfo) -> str:
        return _validate_identifier(value, field_name=info.field_name or "identifier")

    @field_validator("description", "citation")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("cited example text must not have surrounding whitespace")
        return value


def _validate_cross_record_invariants(
    sources: tuple[SourceArtifactDeclaration, ...],
    owner_facts: tuple[OwnerFactInput, ...],
    metric_series: tuple[VisibilityMetricSeriesInput, ...],
    cited_examples: tuple[NonSensitiveExample, ...],
) -> None:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("sources require unique source_id values")

    filenames = [source.filename.casefold() for source in sources]
    if len(filenames) != len(set(filenames)):
        raise ValueError("sources require unique filename values")

    declared_source_ids = set(source_ids)
    referenced_source_ids = {
        *(fact.source_id for fact in owner_facts),
        *(series.source_id for series in metric_series),
        *(example.source_id for example in cited_examples),
    }
    if not referenced_source_ids.issubset(declared_source_ids):
        raise ValueError("normalized records must reference a declared source_id")

    for field_name, identifiers in (
        ("fact_id", [fact.fact_id for fact in owner_facts]),
        ("metric_id", [series.metric_id for series in metric_series]),
        ("example_id", [example.example_id for example in cited_examples]),
    ):
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"normalized records require unique {field_name} values")


class NormalizedIntake(FrozenIntakeModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    canonical_domain: str
    owner_facts: tuple[OwnerFactInput, ...] = Field(default=(), max_length=MAX_OWNER_FACTS)
    metric_series: tuple[VisibilityMetricSeriesInput, ...] = Field(
        default=(), max_length=MAX_VISIBILITY_METRIC_SERIES
    )
    cited_examples: tuple[NonSensitiveExample, ...] = Field(
        default=(), max_length=MAX_CITED_EXAMPLES
    )
    sources: tuple[SourceArtifactDeclaration, ...] = Field(
        min_length=1, max_length=MAX_INTAKE_SOURCES
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("canonical_domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> NormalizedIntake:
        _validate_cross_record_invariants(
            self.sources,
            self.owner_facts,
            self.metric_series,
            self.cited_examples,
        )
        return self


class ProcessedIntake(FrozenIntakeModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    canonical_domain: str
    processed_at: datetime
    deleted_at: datetime
    owner_facts: tuple[OwnerFactInput, ...] = Field(default=(), max_length=MAX_OWNER_FACTS)
    metric_series: tuple[VisibilityMetricSeriesInput, ...] = Field(
        default=(), max_length=MAX_VISIBILITY_METRIC_SERIES
    )
    cited_examples: tuple[NonSensitiveExample, ...] = Field(
        default=(), max_length=MAX_CITED_EXAMPLES
    )
    sources: tuple[SourceArtifactProvenance, ...] = Field(
        min_length=1, max_length=MAX_INTAKE_SOURCES
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("canonical_domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)

    @field_validator("processed_at", "deleted_at")
    @classmethod
    def validate_trusted_timestamps(cls, value: datetime) -> datetime:
        return _validate_aware_timestamp(value)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> ProcessedIntake:
        _validate_timestamp_order(self.processed_at, self.deleted_at)
        _validate_cross_record_invariants(
            self.sources,
            self.owner_facts,
            self.metric_series,
            self.cited_examples,
        )
        if any(
            source.processed_at != self.processed_at or source.deleted_at != self.deleted_at
            for source in self.sources
        ):
            raise ValueError(
                "source provenance timestamps must equal parent processed_at and deleted_at"
            )
        return self


@dataclass(frozen=True)
class _OwnedDirectory:
    path: Path
    root: Path
    run_id: str
    nonce: str
    directory_device: int
    directory_inode: int
    root_device: int
    root_inode: int
    marker_device: int
    marker_inode: int
    canonical_root: str
    canonical_path: str
    identity_anchors: tuple[int, ...] = field(default=(), compare=False, repr=False)


_CapabilityKey: TypeAlias = tuple[str, str, str, int, int, int, int]
_CAPABILITY_LOCK = threading.Lock()
_OWNED_CAPABILITIES: dict[_CapabilityKey, _OwnedDirectory] = {}


def _capability_key(owned: _OwnedDirectory) -> _CapabilityKey:
    return (
        owned.canonical_root,
        owned.canonical_path,
        owned.nonce,
        owned.root_device,
        owned.root_inode,
        owned.directory_device,
        owned.directory_inode,
    )


def _close_identity_anchors(owned: _OwnedDirectory) -> None:
    for descriptor in owned.identity_anchors:
        os.close(descriptor)


def _identity_anchors_are_live(owned: _OwnedDirectory) -> bool:
    if len(owned.identity_anchors) != 2:
        return False
    try:
        directory_fd, marker_fd = owned.identity_anchors
        directory = os.fstat(directory_fd)
        marker = os.fstat(marker_fd)
        return (
            (directory.st_dev, directory.st_ino) == (owned.directory_device, owned.directory_inode)
            and (marker.st_dev, marker.st_ino) == (owned.marker_device, owned.marker_inode)
            and marker.st_nlink == 1
            and not _held_directory_reports_unlinked(directory_fd, directory)
        )
    except OSError:
        return False


def _register_capability(owned: _OwnedDirectory, child_fd: int) -> None:
    # Keep the actual objects alive: device/inode numbers alone can be recycled
    # after unlink/rmdir on local filesystems, including Linux ext4.
    directory_anchor = os.dup(child_fd)
    try:
        marker_anchor = os.open(
            _OWNERSHIP_MARKER,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_anchor,
        )
    except BaseException:
        os.close(directory_anchor)
        raise
    held = replace(owned, identity_anchors=(directory_anchor, marker_anchor))
    try:
        if not _identity_anchors_are_live(held):
            raise IntakeOwnershipError(
                "owned intake capability identity changed before registration"
            )
        key = _capability_key(held)
        with _CAPABILITY_LOCK:
            if key in _OWNED_CAPABILITIES:
                raise IntakeOwnershipError("owned intake capability identity collided")
            _OWNED_CAPABILITIES[key] = held
    except BaseException:
        _close_identity_anchors(held)
        raise


def _claim_issued_capability(
    owned_input_dir: Path,
    intake_root: Path,
) -> _OwnedDirectory:
    try:
        return _claim_live_capability(owned_input_dir, intake_root)
    except BaseException:
        _retire_rejected_capability(owned_input_dir, intake_root)
        raise


def _retire_rejected_capability(owned_input_dir: Path, intake_root: Path) -> None:
    """Release our handles only; never delete a path whose ownership was rejected."""
    root, owned = _absolute_path(intake_root), _absolute_path(owned_input_dir)
    if owned.parent != root or owned == root:
        return
    try:
        canonical_root: str | None = str(root.resolve(strict=False))
    except (OSError, RuntimeError):
        canonical_root = None
    with _CAPABILITY_LOCK:
        stale = [
            key
            for key, capability in _OWNED_CAPABILITIES.items()
            if (capability.root == root and capability.path == owned)
            or (capability.canonical_root == canonical_root and capability.path.name == owned.name)
        ]
        for key in stale:
            _close_identity_anchors(_OWNED_CAPABILITIES.pop(key))


def _claim_live_capability(
    owned_input_dir: Path,
    intake_root: Path,
) -> _OwnedDirectory:
    root, root_details = _validate_intake_root(intake_root, create=False)
    owned = _absolute_path(owned_input_dir)
    if owned == root or owned.parent != root:
        raise IntakeOwnershipError("owned input directory must be a direct child of intake_root")
    owned_details = _directory_lstat(owned, role="owned input directory")
    if owned.resolve(strict=True).parent != root.resolve(strict=True):
        raise IntakeOwnershipError("owned input directory must resolve inside intake_root")
    try:
        marker_details = (owned / _OWNERSHIP_MARKER).lstat()
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker is missing") from exc
    if (
        stat.S_ISLNK(marker_details.st_mode)
        or not stat.S_ISREG(marker_details.st_mode)
        or marker_details.st_nlink != 1
    ):
        raise IntakeOwnershipError("owned intake marker must retain unique regular-file identity")

    canonical_root = str(root.resolve(strict=True))
    canonical_path = str(owned.resolve(strict=True))
    with _CAPABILITY_LOCK:
        matches = [
            (key, capability)
            for key, capability in _OWNED_CAPABILITIES.items()
            if (
                capability.canonical_root,
                capability.canonical_path,
                capability.root_device,
                capability.root_inode,
                capability.directory_device,
                capability.directory_inode,
                capability.marker_device,
                capability.marker_inode,
            )
            == (
                canonical_root,
                canonical_path,
                root_details.st_dev,
                root_details.st_ino,
                owned_details.st_dev,
                owned_details.st_ino,
                marker_details.st_dev,
                marker_details.st_ino,
            )
            and _identity_anchors_are_live(capability)
        ]
        if len(matches) != 1:
            raise IntakeOwnershipError(
                "owned input directory lacks a matching creator-issued capability"
            )
        key, issued = matches[0]
        del _OWNED_CAPABILITIES[key]
    return issued


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _directory_lstat(path: Path, *, role: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise IntakeOwnershipError(f"{role} is not an accessible directory") from exc
    if stat.S_ISLNK(details.st_mode):
        raise IntakeOwnershipError(f"{role} must not be a symlink")
    if not stat.S_ISDIR(details.st_mode):
        raise IntakeOwnershipError(f"{role} must be a directory")
    return details


def _validate_intake_root(intake_root: Path, *, create: bool) -> tuple[Path, os.stat_result]:
    root = _absolute_path(intake_root)
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise IntakeOwnershipError("intake_root could not be created") from exc
    details = _directory_lstat(root, role="intake_root")
    return root, details


def _write_marker_at(directory_fd: int, payload: bytes) -> os.stat_result:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        _OWNERSHIP_MARKER,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise IntakeOwnershipError("owned intake marker lacks unique regular-file ownership")
        return details
    finally:
        os.close(descriptor)


def _trusted_timestamp(value: datetime | None) -> datetime:
    timestamp = value if value is not None else datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise IntakeValidationError("trusted intake timestamps must include a timezone")
    return timestamp


def _mkdir_owned_relative(root_fd: int, run_name: str) -> None:
    os.mkdir(run_name, mode=0o700, dir_fd=root_fd)


def _path_still_names_directory(
    path: Path,
    expected: os.stat_result,
    *,
    role: str,
) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise IntakeOwnershipError(f"{role} identity changed during creation") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise IntakeOwnershipError(f"{role} identity changed during creation")


def _canonical_verified_path(path: Path, expected: os.stat_result, *, role: str) -> str:
    try:
        canonical = path.resolve(strict=True)
        canonical_details = canonical.stat()
    except OSError as exc:
        raise IntakeOwnershipError(f"{role} identity changed during creation") from exc
    if (canonical_details.st_dev, canonical_details.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise IntakeOwnershipError(f"{role} identity changed during creation")
    return str(canonical)


def _cleanup_created_owned_child(
    root_fd: int,
    run_name: str,
    child_fd: int,
    child_details: os.stat_result,
) -> None:
    try:
        current = os.stat(
            run_name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(child_fd)
        if not _same_entry(current, opened) or not _same_entry(child_details, opened):
            return
        _remove_opened_child_directory(
            root_fd,
            run_name,
            child_fd,
            current,
        )
    except BaseException:
        return


def create_owned_intake_dir(intake_root: Path, *, now: datetime | None = None) -> Path:
    """Create a marker-owned, single-run input directory directly below intake_root."""
    _require_supported_posix_primitives()
    root, inspected_root = _validate_intake_root(intake_root, create=True)
    timestamp = _trusted_timestamp(now).astimezone(UTC)
    try:
        root_fd = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise IntakeOwnershipError("intake_root could not be opened safely") from exc
    try:
        root_details = os.fstat(root_fd)
        if not _same_entry(inspected_root, root_details):
            raise IntakeOwnershipError("intake_root identity changed during creation")
        for _attempt in range(10):
            nonce = uuid.uuid4().hex
            run_id = f"run-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{nonce}"
            owned = root / run_id
            try:
                _mkdir_owned_relative(root_fd, run_id)
            except FileExistsError:
                continue

            child_fd = -1
            child_details: os.stat_result | None = None
            try:
                child_fd = os.open(
                    run_id,
                    _directory_open_flags(),
                    dir_fd=root_fd,
                )
                child_details = os.fstat(child_fd)
                marker_payload = json.dumps(
                    {
                        "schema_version": _MARKER_SCHEMA_VERSION,
                        "run_id": run_id,
                        "nonce": nonce,
                        "directory_device": child_details.st_dev,
                        "directory_inode": child_details.st_ino,
                        "root_device": root_details.st_dev,
                        "root_inode": root_details.st_ino,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                marker_details = _write_marker_at(child_fd, marker_payload)
                os.fsync(child_fd)

                _path_still_names_directory(
                    root,
                    root_details,
                    role="intake_root",
                )
                _path_still_names_directory(
                    owned,
                    child_details,
                    role="owned input directory",
                )
                canonical_root = _canonical_verified_path(
                    root,
                    root_details,
                    role="intake_root",
                )
                canonical_path = _canonical_verified_path(
                    owned,
                    child_details,
                    role="owned input directory",
                )
                _path_still_names_directory(
                    root,
                    root_details,
                    role="intake_root",
                )
                _path_still_names_directory(
                    owned,
                    child_details,
                    role="owned input directory",
                )

                capability = _OwnedDirectory(
                    path=owned,
                    root=root,
                    run_id=run_id,
                    nonce=nonce,
                    directory_device=child_details.st_dev,
                    directory_inode=child_details.st_ino,
                    root_device=root_details.st_dev,
                    root_inode=root_details.st_ino,
                    marker_device=marker_details.st_dev,
                    marker_inode=marker_details.st_ino,
                    canonical_root=canonical_root,
                    canonical_path=canonical_path,
                )
                _register_capability(capability, child_fd)
                return owned
            except BaseException:
                if child_fd >= 0 and child_details is not None:
                    _cleanup_created_owned_child(
                        root_fd,
                        run_id,
                        child_fd,
                        child_details,
                    )
                raise
            finally:
                if child_fd >= 0:
                    os.close(child_fd)
        raise IntakeOwnershipError("could not allocate a unique owned intake directory")
    finally:
        os.close(root_fd)


def _read_marker(marker_path: Path) -> tuple[dict[str, object], os.stat_result]:
    try:
        before = marker_path.lstat()
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IntakeOwnershipError("owned intake marker must be a regular non-symlink file")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker_path, flags)
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IntakeOwnershipError("owned intake marker changed during validation")
        raw_marker = os.read(descriptor, 4097)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if len(raw_marker) > 4096 or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise IntakeOwnershipError("owned intake marker changed during validation")
    try:
        decoded = json.loads(raw_marker)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeOwnershipError("owned intake marker is invalid") from exc
    if not isinstance(decoded, dict):
        raise IntakeOwnershipError("owned intake marker is invalid")
    return decoded, before


def _verify_owned_directory(owned_input_dir: Path, intake_root: Path) -> _OwnedDirectory:
    root, root_details = _validate_intake_root(intake_root, create=False)
    owned = _absolute_path(owned_input_dir)
    if owned == root or owned.parent != root:
        raise IntakeOwnershipError("owned input directory must be a direct child of intake_root")
    owned_details = _directory_lstat(owned, role="owned input directory")
    if owned.resolve(strict=True).parent != root.resolve(strict=True):
        raise IntakeOwnershipError("owned input directory must resolve inside intake_root")

    marker, marker_details = _read_marker(owned / _OWNERSHIP_MARKER)
    if set(marker) != {
        "schema_version",
        "run_id",
        "nonce",
        "directory_device",
        "directory_inode",
        "root_device",
        "root_inode",
    }:
        raise IntakeOwnershipError("owned intake marker has unexpected fields")
    schema_version = marker.get("schema_version")
    run_id = marker.get("run_id")
    nonce = marker.get("nonce")
    directory_device = marker.get("directory_device")
    directory_inode = marker.get("directory_inode")
    root_device = marker.get("root_device")
    root_inode = marker.get("root_inode")
    if (
        schema_version != _MARKER_SCHEMA_VERSION
        or not isinstance(run_id, str)
        or not isinstance(nonce, str)
        or not _RUN_ID.fullmatch(run_id)
        or not _NONCE.fullmatch(nonce)
        or run_id != owned.name
        or not run_id.endswith(f"-{nonce}")
        or type(directory_device) is not int
        or type(directory_inode) is not int
        or type(root_device) is not int
        or type(root_inode) is not int
        or (directory_device, directory_inode) != (owned_details.st_dev, owned_details.st_ino)
        or (root_device, root_inode) != (root_details.st_dev, root_details.st_ino)
    ):
        raise IntakeOwnershipError("owned intake marker does not match the directory identity")

    return _OwnedDirectory(
        path=owned,
        root=root,
        run_id=run_id,
        nonce=nonce,
        directory_device=owned_details.st_dev,
        directory_inode=owned_details.st_ino,
        root_device=root_details.st_dev,
        root_inode=root_details.st_ino,
        marker_device=marker_details.st_dev,
        marker_inode=marker_details.st_ino,
        canonical_root=str(root.resolve(strict=True)),
        canonical_path=str(owned.resolve(strict=True)),
    )


def _read_marker_at(directory_fd: int) -> tuple[dict[str, object], os.stat_result]:
    try:
        before = os.stat(
            _OWNERSHIP_MARKER,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IntakeOwnershipError("owned intake marker must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        marker_fd = os.open(_OWNERSHIP_MARKER, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker could not be opened safely") from exc
    try:
        opened = os.fstat(marker_fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IntakeOwnershipError("owned intake marker changed during validation")
        raw_marker = os.read(marker_fd, 4097)
        after = os.fstat(marker_fd)
    finally:
        os.close(marker_fd)
    if len(raw_marker) > 4096 or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise IntakeOwnershipError("owned intake marker changed during validation")
    try:
        decoded = json.loads(raw_marker)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeOwnershipError("owned intake marker is invalid") from exc
    if not isinstance(decoded, dict):
        raise IntakeOwnershipError("owned intake marker is invalid")
    return decoded, before


def _stat_marker_at(directory_fd: int) -> os.stat_result:
    try:
        details = os.stat(
            _OWNERSHIP_MARKER,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntakeOwnershipError("owned intake marker is missing") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise IntakeOwnershipError("owned intake marker must retain unique regular-file identity")
    return details


def _rename_to_quarantine(root_fd: int, source_name: str, quarantine_fd: int) -> None:
    os.rename(
        source_name,
        _QUARANTINE_PAYLOAD,
        src_dir_fd=root_fd,
        dst_dir_fd=quarantine_fd,
    )


def _restore_quarantined_replacement(root_fd: int, source_name: str, quarantine_fd: int) -> None:
    os.rename(
        _QUARANTINE_PAYLOAD,
        source_name,
        src_dir_fd=quarantine_fd,
        dst_dir_fd=root_fd,
    )


def _same_entry(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        stat.S_IFMT(first.st_mode),
    ) == (
        second.st_dev,
        second.st_ino,
        stat.S_IFMT(second.st_mode),
    )


def _require_name_absent(parent_fd: int, entry_name: str) -> None:
    try:
        os.stat(entry_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise IntakeCleanupError(f"quarantined entry still exists after deletion: {entry_name}")


def _unlink_opened_regular_entry(
    parent_fd: int,
    entry_name: str,
    entry_fd: int,
    inspected: os.stat_result,
) -> None:
    opened = os.fstat(entry_fd)
    if (
        not _same_entry(inspected, opened)
        or not stat.S_ISREG(opened.st_mode)
        or inspected.st_nlink != 1
        or opened.st_nlink != 1
    ):
        raise IntakeCleanupError(
            f"quarantined regular entry lacks unique link ownership: {entry_name}"
        )
    try:
        current = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntakeCleanupError(
            f"quarantined regular entry changed before unlink: {entry_name}"
        ) from exc
    if not _same_entry(opened, current) or current.st_nlink != 1:
        raise IntakeCleanupError(f"quarantined regular entry changed before unlink: {entry_name}")
    os.unlink(entry_name, dir_fd=parent_fd)
    unlinked = os.fstat(entry_fd)
    if unlinked.st_nlink != 0:
        raise IntakeCleanupError(
            f"quarantined regular entry still has links after unlink: {entry_name}"
        )
    _require_name_absent(parent_fd, entry_name)


def _unlink_non_directory_entry(
    parent_fd: int,
    entry_name: str,
    inspected: os.stat_result,
) -> None:
    try:
        current = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntakeCleanupError(f"quarantined entry changed before unlink: {entry_name}") from exc
    if not _same_entry(inspected, current):
        raise IntakeCleanupError(f"quarantined entry changed before unlink: {entry_name}")
    os.unlink(entry_name, dir_fd=parent_fd)
    _require_name_absent(parent_fd, entry_name)


def _remove_opened_child_directory(
    parent_fd: int,
    entry_name: str,
    child_fd: int,
    inspected: os.stat_result,
) -> None:
    opened = os.fstat(child_fd)
    if not _same_entry(inspected, opened) or not stat.S_ISDIR(opened.st_mode):
        raise IntakeCleanupError(
            f"quarantined directory entry changed before traversal: {entry_name}"
        )
    _remove_directory_contents(child_fd)
    try:
        current = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntakeCleanupError(
            f"quarantined directory entry changed before rmdir: {entry_name}"
        ) from exc
    if not _same_entry(opened, current):
        raise IntakeCleanupError(f"quarantined directory entry changed before rmdir: {entry_name}")
    _rmdir_relative_directory(parent_fd, entry_name)
    _require_name_absent(parent_fd, entry_name)
    removed = os.fstat(child_fd)
    if (
        not _same_entry(opened, removed)
        or os.listdir(child_fd)
        or not _held_directory_reports_unlinked(child_fd, removed)
    ):
        raise IntakeCleanupError(
            f"held quarantined directory does not report removal: {entry_name}"
        )


def _rmdir_relative_directory(parent_fd: int, entry_name: str) -> None:
    """Remove the checked name; post-rmdir descriptor checks detect replacement races."""
    os.rmdir(entry_name, dir_fd=parent_fd)


def _held_directory_reports_unlinked(directory_fd: int, removed: os.stat_result) -> bool:
    if removed.st_nlink == 0:
        return True
    if sys.platform != "darwin":
        return False

    # macOS retains the pre-removal link count on an open directory descriptor.
    # F_GETPATH distinguishes a stale removed path from a live renamed survivor.
    try:
        import fcntl

        raw_path = fcntl.fcntl(
            directory_fd,
            fcntl.F_GETPATH,
            b"\0" * 1024,
        )
        reported_path = raw_path.split(b"\0", 1)[0]
        if not reported_path:
            return False
        current = os.stat(reported_path, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not _same_entry(removed, current)


def _remove_directory_contents(directory_fd: int) -> None:
    for entry_name in os.listdir(directory_fd):
        details = os.stat(
            entry_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(details.st_mode):
            child_fd = _open_directory_component(directory_fd, entry_name)
            try:
                _remove_opened_child_directory(
                    directory_fd,
                    entry_name,
                    child_fd,
                    details,
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(details.st_mode):
            if details.st_nlink != 1:
                raise IntakeCleanupError(
                    f"quarantined regular entry lacks unique link ownership: {entry_name}"
                )
            entry_fd = os.open(
                entry_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                _unlink_opened_regular_entry(
                    directory_fd,
                    entry_name,
                    entry_fd,
                    details,
                )
            finally:
                os.close(entry_fd)
        else:
            _unlink_non_directory_entry(directory_fd, entry_name, details)


def _delete_quarantined_directory(
    root_fd: int,
    quarantine_name: str,
    quarantine_identity: tuple[int, int],
) -> None:
    quarantine_fd = os.open(
        quarantine_name,
        _directory_open_flags(),
        dir_fd=root_fd,
    )
    try:
        quarantine_details = os.fstat(quarantine_fd)
        if (quarantine_details.st_dev, quarantine_details.st_ino) != quarantine_identity:
            raise IntakeCleanupError("quarantine directory identity changed before deletion")
        _remove_opened_child_directory(
            root_fd,
            quarantine_name,
            quarantine_fd,
            quarantine_details,
        )
    finally:
        os.close(quarantine_fd)


def _create_quarantine(root_fd: int) -> tuple[str, int, tuple[int, int]]:
    for _attempt in range(10):
        quarantine_name = f".quarantine-{uuid.uuid4().hex}"
        quarantine_fd = -1
        created = False
        try:
            os.mkdir(quarantine_name, mode=0o700, dir_fd=root_fd)
            created = True
        except FileExistsError:
            continue
        try:
            quarantine_fd = os.open(
                quarantine_name,
                _directory_open_flags(),
                dir_fd=root_fd,
            )
            details = os.fstat(quarantine_fd)
            return (
                quarantine_name,
                quarantine_fd,
                (details.st_dev, details.st_ino),
            )
        except BaseException:
            if quarantine_fd >= 0:
                os.close(quarantine_fd)
            if created:
                try:
                    os.rmdir(quarantine_name, dir_fd=root_fd)
                except OSError:
                    pass
            raise
    raise IntakeCleanupError("could not allocate a unique quarantine directory")


def _remove_empty_quarantine(root_fd: int, quarantine_name: str, quarantine_fd: int) -> None:
    try:
        os.rmdir(quarantine_name, dir_fd=root_fd)
    except OSError:
        pass


def _delete_verified_owned_directory(
    owned: _OwnedDirectory,
    *,
    require_marker_content: bool,
) -> None:
    root_fd = -1
    quarantine_fd = -1
    quarantine_name: str | None = None
    try:
        root_fd = os.open(owned.root, _directory_open_flags())
        root_details = os.fstat(root_fd)
        if (root_details.st_dev, root_details.st_ino) != (
            owned.root_device,
            owned.root_inode,
        ):
            raise IntakeCleanupError("intake_root identity changed before quarantine")
        quarantine_name, quarantine_fd, quarantine_identity = _create_quarantine(root_fd)
        try:
            _rename_to_quarantine(root_fd, owned.path.name, quarantine_fd)
        except BaseException:
            _remove_empty_quarantine(root_fd, quarantine_name, quarantine_fd)
            raise

        quarantined_fd = os.open(
            _QUARANTINE_PAYLOAD,
            _directory_open_flags(),
            dir_fd=quarantine_fd,
        )
        try:
            quarantined_details = os.fstat(quarantined_fd)
            if (quarantined_details.st_dev, quarantined_details.st_ino) != (
                owned.directory_device,
                owned.directory_inode,
            ):
                _restore_quarantined_replacement(root_fd, owned.path.name, quarantine_fd)
                _remove_empty_quarantine(root_fd, quarantine_name, quarantine_fd)
                raise IntakeCleanupError("owned input directory identity changed at quarantine")
            if require_marker_content:
                marker, marker_details = _read_marker_at(quarantined_fd)
            else:
                marker = None
                marker_details = _stat_marker_at(quarantined_fd)
            if (marker_details.st_dev, marker_details.st_ino) != (
                owned.marker_device,
                owned.marker_inode,
            ) or (
                marker is not None
                and (marker.get("run_id"), marker.get("nonce")) != (owned.run_id, owned.nonce)
            ):
                _restore_quarantined_replacement(root_fd, owned.path.name, quarantine_fd)
                _remove_empty_quarantine(root_fd, quarantine_name, quarantine_fd)
                raise IntakeCleanupError("owned intake marker identity changed at quarantine")
        finally:
            os.close(quarantined_fd)

        os.close(quarantine_fd)
        quarantine_fd = -1
        _delete_quarantined_directory(
            root_fd,
            quarantine_name,
            quarantine_identity,
        )
        try:
            os.stat(
                quarantine_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise IntakeCleanupError("quarantine still exists after deletion")
    except IntakeCleanupError:
        raise
    except BaseException as exc:
        raise IntakeCleanupError("could not delete the verified owned input directory") from exc
    finally:
        if quarantine_fd >= 0:
            os.close(quarantine_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)


def _open_verified_owned_descriptors(owned: _OwnedDirectory) -> tuple[int, int]:
    owned_fd = -1
    try:
        root_fd = os.open(owned.root, _directory_open_flags())
    except OSError as exc:
        raise IntakeOwnershipError("intake_root could not be opened safely") from exc
    try:
        root_details = os.fstat(root_fd)
        if (root_details.st_dev, root_details.st_ino) != (
            owned.root_device,
            owned.root_inode,
        ):
            raise IntakeOwnershipError("intake_root identity changed before file access")
        owned_fd = os.open(
            owned.path.name,
            _directory_open_flags(),
            dir_fd=root_fd,
        )
        owned_details = os.fstat(owned_fd)
        if (owned_details.st_dev, owned_details.st_ino) != (
            owned.directory_device,
            owned.directory_inode,
        ):
            raise IntakeOwnershipError("owned input directory identity changed before file access")
    except BaseException:
        if owned_fd >= 0:
            os.close(owned_fd)
        os.close(root_fd)
        raise
    return root_fd, owned_fd


def _open_directory_component(parent_fd: int, component: str) -> int:
    return os.open(component, _directory_open_flags(), dir_fd=parent_fd)


def _raise_parent_open_error(parent_fd: int, component: str, filename: str, error: OSError) -> None:
    try:
        details = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        message = f"declared source parent is missing: {filename}"
    except OSError:
        message = f"declared source parent is inaccessible: {filename}"
    else:
        if stat.S_ISLNK(details.st_mode):
            message = f"declared source path contains a symlink: {filename}"
        elif not stat.S_ISDIR(details.st_mode):
            message = f"declared source parent is not a directory: {filename}"
        else:
            message = f"declared source parent changed during traversal: {filename}"
    raise IntakeValidationError(message) from error


def _open_artifact_parent(owned_fd: int, filename: str) -> tuple[int, str]:
    relative = PurePosixPath(filename)
    current_fd = os.dup(owned_fd)
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = _open_directory_component(current_fd, component)
            except OSError as exc:
                _raise_parent_open_error(current_fd, component, filename, exc)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, relative.name
    except BaseException:
        os.close(current_fd)
        raise


def _stat_artifact(
    parent_fd: int, filename_component: str, declared_filename: str
) -> os.stat_result:
    try:
        details = os.stat(
            filename_component,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise IntakeValidationError(
            f"declared source file is missing: {declared_filename}"
        ) from exc
    except OSError as exc:
        raise IntakeValidationError(
            f"declared source file is inaccessible: {declared_filename}"
        ) from exc
    if stat.S_ISLNK(details.st_mode):
        raise IntakeValidationError(
            f"declared source file must not be a symlink: {declared_filename}"
        )
    if not stat.S_ISREG(details.st_mode):
        raise IntakeValidationError(f"declared source must be a regular file: {declared_filename}")
    return details


def _hash_declared_artifact(owned_fd: int, source: SourceArtifactDeclaration) -> None:
    parent_fd, filename_component = _open_artifact_parent(owned_fd, source.filename)
    try:
        before = _stat_artifact(parent_fd, filename_component, source.filename)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename_component, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise IntakeValidationError(
                f"declared source could not be opened: {source.filename}"
            ) from exc
    finally:
        os.close(parent_fd)

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise IntakeValidationError(
                f"declared source must be a regular file: {source.filename}"
            )
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise IntakeValidationError(
                f"declared source changed during validation: {source.filename}"
            )
        if before.st_nlink != 1 or opened.st_nlink != 1:
            raise IntakeValidationError(
                f"declared source must have unique link ownership: {source.filename}"
            )
        while chunk := os.read(descriptor, _HASH_CHUNK_SIZE):
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields_before = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
        opened.st_nlink,
    )
    stable_fields_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if (
        stable_fields_before != stable_fields_after
        or bytes_read != after.st_size
        or after.st_nlink != 1
    ):
        raise IntakeValidationError(f"declared source changed during hashing: {source.filename}")
    if bytes_read != source.byte_count:
        raise IntakeValidationError(f"declared source byte_count does not match: {source.filename}")
    if digest.hexdigest() != source.sha256:
        raise IntakeValidationError(f"declared source checksum does not match: {source.filename}")


def _coerce_normalized_intake(
    normalized: NormalizedIntake | Mapping[str, object],
) -> NormalizedIntake:
    candidate: object
    if isinstance(normalized, NormalizedIntake):
        candidate = normalized.model_dump(round_trip=True)
    else:
        candidate = normalized
    try:
        return NormalizedIntake.model_validate(candidate)
    except ValidationError as exc:
        raise IntakeValidationError("normalized intake failed schema validation") from exc


def _build_processed_intake(
    normalized: NormalizedIntake,
    *,
    processed_at: datetime,
    deleted_at: datetime,
) -> ProcessedIntake:
    provenance = tuple(
        SourceArtifactProvenance(
            **source.model_dump(),
            processed_at=processed_at,
            deleted_at=deleted_at,
        )
        for source in normalized.sources
    )
    return ProcessedIntake(
        schema_version=normalized.schema_version,
        project_id=normalized.project_id,
        canonical_domain=normalized.canonical_domain,
        processed_at=processed_at,
        deleted_at=deleted_at,
        owner_facts=normalized.owner_facts,
        metric_series=normalized.metric_series,
        cited_examples=normalized.cited_examples,
        sources=provenance,
    )


def consume_intake(
    owned_input_dir: Path,
    normalized: NormalizedIntake | Mapping[str, object],
    *,
    intake_root: Path,
    now: datetime | None = None,
    processor: IntakeProcessor | None = None,
) -> ProcessedIntake:
    """Consume under exclusive mutation on a supported local POSIX filesystem."""
    _require_supported_posix_primitives()
    owned = _claim_issued_capability(owned_input_dir, intake_root)
    try:
        return _consume_claimed_intake(owned, normalized, now=now, processor=processor)
    finally:
        _close_identity_anchors(owned)


def _consume_claimed_intake(
    owned: _OwnedDirectory,
    normalized: NormalizedIntake | Mapping[str, object],
    *,
    now: datetime | None,
    processor: IntakeProcessor | None,
) -> ProcessedIntake:
    try:
        if _verify_owned_directory(owned.path, owned.root) != owned:
            raise IntakeOwnershipError(
                "owned input directory marker does not match its issued capability"
            )
        validated = _coerce_normalized_intake(normalized)
        processed_at = _trusted_timestamp(now)
        root_fd, owned_fd = _open_verified_owned_descriptors(owned)
        try:
            for source in validated.sources:
                _hash_declared_artifact(owned_fd, source)
        finally:
            os.close(owned_fd)
            os.close(root_fd)
        if processor is not None:
            processor(validated)
    except BaseException:
        _delete_verified_owned_directory(
            owned,
            require_marker_content=False,
        )
        raise

    _delete_verified_owned_directory(
        owned,
        require_marker_content=True,
    )
    deleted_at = _trusted_timestamp(now)
    return _build_processed_intake(
        validated,
        processed_at=processed_at,
        deleted_at=deleted_at,
    )


def discard_owned_intake_dir(
    owned_input_dir: Path,
    *,
    intake_root: Path,
) -> None:
    """Retire and delete one unused capability under exclusive local mutation."""
    _require_supported_posix_primitives()
    owned = _claim_issued_capability(owned_input_dir, intake_root)
    try:
        _delete_verified_owned_directory(
            owned,
            require_marker_content=False,
        )
    finally:
        _close_identity_anchors(owned)
