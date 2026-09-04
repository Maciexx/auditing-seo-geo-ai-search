"""Explicit provider-only diagnostic schema; legacy 1.0.0 is never migrated.

The existing store owns publication. These models validate normalized observations,
not raw provider responses, and make no claim of independent provider authenticity.
"""

from __future__ import annotations

import re
from datetime import UTC
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

from ai_search_audit.benchmark import canonical_hash
from ai_search_audit.diagnostic_models import (
    BenchmarkHash,
    DiagnosticBinding,
    DiagnosticCollectionRange,
    DiagnosticFile,
    DiagnosticSource,
    FrozenDiagnosticModel,
)
from ai_search_audit.measurement_policy import (
    PerformancePipeline,
    allows_lighthouse_fallback,
    performance_pipeline_version,
)
from ai_search_audit.measurement_profile import (
    MeasurementPreflight,
    prepare_measurement_preflight,
)
from ai_search_audit.performance_normalizers import measurement_url_key
from ai_search_audit.performance_providers import PerformanceCollection


class ProviderRunCleanup(FrozenDiagnosticModel):
    """No owned intake exists; runtime resource failures remain in attempt history."""

    status: Literal["not_required"] = "not_required"
    method: Literal["no_owned_intake"] = "no_owned_intake"
    completed_at: None = None


def reject_binary_values(value: object) -> None:
    """Do not let unchecked Python bytes be silently decoded by string validators."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("binary values are not normalized performance evidence")
    if isinstance(value, BaseModel):
        reject_binary_values(value.__dict__)
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_binary_values(key)
            reject_binary_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_binary_values(item)


def ordered_collections(
    preflight: MeasurementPreflight, collections: tuple[PerformanceCollection, ...]
) -> tuple[PerformanceCollection, ...]:
    pages = {page.url: i for i, page in enumerate(preflight.selected_pages)}
    providers = {"pagespeed_insights": 0, "crux": 1, "lighthouse_local": 2}
    return tuple(
        sorted(
            collections,
            key=lambda c: (
                pages.get(c.attempts[0].requested_url, -1),
                c.attempts[0].device != "mobile",
                providers[c.attempts[0].provider],
            ),
        )
    )


def performance_algorithm_hash(*, policy_version: str = "1.0.0") -> str:
    """Keep the historical no-argument fingerprint; new runs pass their frozen policy."""
    return canonical_hash(
        {"pipeline": performance_pipeline_version(policy_version), "normalization": "1.0.0"}
    )


def performance_collection_range(
    collections: tuple[PerformanceCollection, ...],
) -> DiagnosticCollectionRange:
    stamps = [
        stamp.astimezone(UTC)
        for collection in collections
        for attempt in collection.attempts
        for stamp in (attempt.started_at, attempt.ended_at)
    ]
    stamps.extend(
        measurement.observed_at.astimezone(UTC)
        for collection in collections
        for measurement in (collection.lab, collection.field)
        if measurement is not None
    )
    return DiagnosticCollectionRange(
        start=min(stamps) if stamps else None,
        end=max(stamps) if stamps else None,
        reason=None if stamps else "No collection observations are available.",
    )


_REASONS = frozenset(
    {
        "missing_key",
        "invalid_key",
        "unsafe_target",
        "timeout",
        "transport_error",
        "redirect",
        "response_too_large",
        "unsupported_encoding",
        "sensitive_response",
        "malformed_response",
        "target_mismatch",
        "device_mismatch",
        "runtime_error",
        "no_data",
        "invalid_metrics",
        "missing_metrics",
        "no_record",
        "runtime_missing",
        "runtime_unverified",
        "sandbox_unverified",
        "network_unverified",
        "cleanup_failed",
    }
)


class DiagnosticRunV2(FrozenDiagnosticModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    binding: DiagnosticBinding
    pipeline_version: PerformancePipeline = "performance-1.0.0"
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    collection_range: DiagnosticCollectionRange
    preflight: MeasurementPreflight
    collections: tuple[PerformanceCollection, ...] = Field(max_length=30)
    cleanup: ProviderRunCleanup

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        if self.binding != self.preflight.binding:
            raise ValueError("performance preflight binding mismatch")
        if self.preflight.profile.gemini:
            raise ValueError("this performance schema does not accept Gemini execution")
        if self.input_sha256 != canonical_hash(self.preflight.model_dump(mode="json")):
            raise ValueError("performance input hash mismatch")
        policy_version = self.preflight.profile.schema_version
        if self.pipeline_version != performance_pipeline_version(policy_version):
            raise ValueError("performance profile and pipeline version mismatch")
        if self.algorithm_sha256 != performance_algorithm_hash(policy_version=policy_version):
            raise ValueError("performance algorithm fingerprint mismatch")
        if self.collection_range != performance_collection_range(self.collections):
            raise ValueError("performance collection range does not match observations")
        if self.collections != ordered_collections(self.preflight, self.collections):
            raise ValueError("performance collections must follow canonical page/device order")
        _validate_collections(self)
        return self


def _validate_collections(run: DiagnosticRunV2) -> None:
    preflight = run.preflight
    urls = tuple(page.url for page in preflight.selected_pages)
    if len(set(urls)) != len(urls):
        raise ValueError("duplicate performance pages")
    expected = {
        (url, device, provider)
        for url in urls
        for device in preflight.devices
        for provider, enabled in (
            ("pagespeed_insights", preflight.profile.pagespeed_insights),
            ("crux", preflight.profile.crux),
        )
        if enabled
    }
    actual: set[tuple[str, str, str]] = set()
    ids: set[str] = set()
    for collection in run.collections:
        _validate_history(collection, retry_transient=preflight.profile.retry_transient)
        first = collection.attempts[0]
        key = (first.requested_url, first.device, first.provider)
        if key in actual or first.requested_url not in urls:
            raise ValueError("duplicate or unselected performance collection")
        actual.add(key)
        for attempt in collection.attempts:
            if attempt.attempt_id in ids:
                raise ValueError("duplicate performance attempt/evidence ID")
            ids.add(attempt.attempt_id)
            if attempt.device != first.device:
                raise ValueError("performance history device mismatch")
            if (
                attempt.reason is not None
                and attempt.reason not in _REASONS
                and not re.fullmatch(r"http_[1-5][0-9]{2}", attempt.reason)
            ):
                raise ValueError("unsafe provider reason")
        for measurement in (collection.lab, collection.field):
            if measurement is None:
                continue
            if measurement.evidence_id in ids:
                raise ValueError("duplicate performance attempt/evidence ID")
            ids.add(measurement.evidence_id)
            if (
                measurement.binding != run.binding
                or measurement.requested_url != first.requested_url
            ):
                raise ValueError("performance measurement binding or page mismatch")
    primary = {key for key in actual if key[2] != "lighthouse_local"}
    if primary != expected:
        raise ValueError("performance page/device/provider inventory mismatch")
    expected_local = {
        (c.attempts[0].requested_url, c.attempts[0].device, "lighthouse_local")
        for c in run.collections
        if preflight.profile.lighthouse_local
        and c.attempts[0].provider == "pagespeed_insights"
        and allows_lighthouse_fallback(
            c.attempts[-1], policy_version=preflight.profile.schema_version
        )
    }
    if actual - primary != expected_local:
        raise ValueError("local Lighthouse fallback inventory mismatch")
    for url, device, provider in actual - primary:
        if provider != "lighthouse_local" or not preflight.profile.lighthouse_local:
            raise ValueError("unconfigured performance provider")
        psi = next(
            c
            for c in run.collections
            if (c.attempts[0].requested_url, c.attempts[0].device, c.attempts[0].provider)
            == (url, device, "pagespeed_insights")
        )
        if not allows_lighthouse_fallback(
            psi.attempts[-1], policy_version=preflight.profile.schema_version
        ):
            raise ValueError("local Lighthouse requires eligible PSI")
        local = next(
            c
            for c in run.collections
            if (c.attempts[0].requested_url, c.attempts[0].device, c.attempts[0].provider)
            == (url, device, provider)
        )
        if local.attempts[0].started_at < psi.attempts[-1].ended_at:
            raise ValueError("local Lighthouse fallback chronology mismatch")


def validate_performance_source(run: DiagnosticRunV2, source: DiagnosticSource) -> None:
    if run.binding != source.binding:
        raise ValueError("performance run binding does not match canonical source")
    if run.preflight != prepare_measurement_preflight(source, run.preflight.profile):
        raise ValueError("performance preflight does not match canonical source")
    audited = {measurement_url_key(url) for url in source.page_urls}
    for collection in run.collections:
        if (
            collection.lab is not None
            and measurement_url_key(collection.lab.final_url) not in audited
        ):
            raise ValueError("performance final URL is not source attested")
        field = collection.field
        if (
            field is not None
            and field.scope == "url"
            and field.url_normalization is None
            and measurement_url_key(field.requested_url) != measurement_url_key(field.record_key)
        ):
            raise ValueError("CrUX record key requires retained URL normalization attestation")


def _validate_history(collection: PerformanceCollection, *, retry_transient: bool) -> None:
    """Enforce the single-trial retry and URL-to-origin fallback protocol."""
    first = collection.attempts[0]
    tries = 2 if retry_transient else 1
    if first.provider == "lighthouse_local":
        if len(collection.attempts) != 1:
            raise ValueError("local Lighthouse permits one attempt")
        return
    scope = "url"
    scope_count = 1
    page_key = measurement_url_key(first.requested_url)
    for previous, current in zip(collection.attempts, collection.attempts[1:], strict=False):
        if current.started_at < previous.ended_at:
            raise ValueError("performance attempt chronology mismatch")
        current_key = measurement_url_key(current.requested_url)
        if (
            first.provider == "crux"
            and scope == "url"
            and previous.reason in {"no_record", "no_data"}
        ):
            scope, scope_count = "origin", 1
            if current_key[:3] != page_key[:3] or current_key[3:] != ("/", ""):
                raise ValueError("CrUX origin fallback target mismatch")
        else:
            if previous.reason not in {
                "timeout",
                "transport_error",
            } and previous.http_status not in {429, 500, 502, 503, 504}:
                raise ValueError("performance retry is not transient")
            if current_key != measurement_url_key(previous.requested_url):
                raise ValueError("performance retry target mismatch")
            scope_count += 1
        if scope_count > tries:
            raise ValueError("performance history exceeds frozen attempt budget")
    if first.provider == "crux":
        if scope == "url" and collection.attempts[-1].reason in {"no_record", "no_data"}:
            raise ValueError("CrUX URL no-data outcome requires the origin fallback attempt")
        if collection.field is not None and collection.field.scope != scope:
            raise ValueError("CrUX measurement scope must match the actual attempt history")


class DiagnosticManifestV2(FrozenDiagnosticModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    binding: DiagnosticBinding
    run_number: Annotated[int, Field(strict=True, ge=1)]
    files: tuple[DiagnosticFile, DiagnosticFile]
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    cleanup: ProviderRunCleanup

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if tuple(item.filename for item in self.files) != ("diagnostics.json", "evidence.jsonl"):
            raise ValueError("diagnostic manifest inventory mismatch")
        return self


class LoadedDiagnosticRunV2(FrozenDiagnosticModel):
    run: DiagnosticRunV2
    manifest: DiagnosticManifestV2
    manifest_sha256: BenchmarkHash
