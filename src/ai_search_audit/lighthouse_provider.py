"""One source-bound local Lighthouse trial. No retries, persistence or PSI wrapping."""

from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from ai_search_audit.diagnostic_models import DiagnosticSource
from ai_search_audit.lighthouse_runtime import RuntimeResult
from ai_search_audit.models import DataState
from ai_search_audit.performance_models import (
    Device,
    LighthouseRequest,
    MeasurementState,
    ProviderAttempt,
    _hostname,
)
from ai_search_audit.performance_normalizers import (
    NormalizationError,
    decode_provider_json,
    measurement_url_key,
    normalize_lighthouse,
)
from ai_search_audit.performance_providers import PerformanceCollection, _complete, utc_now


class Runtime(Protocol):
    def run(self, request: LighthouseRequest) -> RuntimeResult: ...


class LighthouseProvider:
    def __init__(self, *, runtime: Runtime, now: Callable[[], datetime] = utc_now) -> None:
        self._runtime = runtime
        self._now = now

    def collect(
        self, source: DiagnosticSource, *, url: str, device: Device
    ) -> PerformanceCollection:
        try:
            source = DiagnosticSource.model_validate_json(
                source.model_dump_json(serialize_as_any=True, warnings=False)
            )
            request = LighthouseRequest(
                requested_url=url, device=device, locale=source.binding.report_locale
            )
            target = measurement_url_key(url)
            domains = {_hostname(domain) for domain in source.canonical_domains}
            if (
                target not in {measurement_url_key(page) for page in source.page_urls}
                or target[1] not in domains
                or _hostname(source.binding.domain) not in domains
            ):
                raise ValueError
        except (ValueError, TypeError, AttributeError, RecursionError):
            raise ValueError("invalid performance configuration") from None
        attempt_id = str(uuid4())
        started = self._now()
        response = self._runtime.run(request)
        ended = self._now()
        state: MeasurementState = DataState.FAILED
        reason: str | None = response.failure
        lab = None
        if reason in {
            "runtime_missing",
            "runtime_unverified",
            "sandbox_unverified",
            "network_unverified",
        }:
            state = DataState.UNAVAILABLE
        elif reason is None:
            try:
                if response.fingerprint is None:
                    raise NormalizationError("malformed_response")
                lab = normalize_lighthouse(
                    decode_provider_json(response.body),
                    request=request,
                    binding=source.binding,
                    attempt_id=attempt_id,
                    evidence_id=str(uuid4()),
                    allowed_final_urls=source.page_urls,
                    runtime=response.fingerprint,
                )
                complete = _complete(lab)
                state = DataState.AVAILABLE if complete else DataState.PARTIAL
                reason = None if complete else "missing_metrics"
            except NormalizationError as error:
                reason = error.code
                if reason in {"target_mismatch", "device_mismatch"}:
                    state = DataState.UNKNOWN
                elif reason == "no_data":
                    state = DataState.UNAVAILABLE
        return PerformanceCollection(
            attempts=(
                ProviderAttempt(
                    attempt_id=attempt_id,
                    provider="lighthouse_local",
                    requested_url=url,
                    device=device,
                    started_at=started,
                    ended_at=ended,
                    state=state,
                    reason=reason,
                    http_status=None,
                ),
            ),
            lab=lab,
        )
