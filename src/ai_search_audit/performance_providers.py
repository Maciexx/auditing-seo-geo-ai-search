"""Source-bound, single-trial performance collection without persistence."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Self
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from ai_search_audit.diagnostic_models import DiagnosticSource
from ai_search_audit.models import DataState
from ai_search_audit.performance_http import (
    PerformanceHTTPClient,
    PerformanceRequest,
    RemoteProvider,
)
from ai_search_audit.performance_models import (
    Device,
    FieldMeasurement,
    FrozenPerformanceModel,
    LabMeasurement,
    MeasurementState,
    ProviderAttempt,
    _hostname,
)
from ai_search_audit.performance_normalizers import (
    NormalizationError,
    decode_provider_json,
    measurement_url_key,
    normalize_crux,
    normalize_psi,
)


class PerformanceCollection(FrozenPerformanceModel):
    attempts: tuple[ProviderAttempt, ...] = Field(min_length=1, max_length=4)
    lab: LabMeasurement | None = None
    field: FieldMeasurement | None = None

    @field_validator("attempts", "lab", "field", mode="before")
    @classmethod
    def detach_nested_models(cls, value: object) -> object:
        if isinstance(value, BaseModel):
            return json.loads(value.model_dump_json(serialize_as_any=True, warnings=False))
        if isinstance(value, (tuple, list)):
            return tuple(cls.detach_nested_models(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if len({attempt.attempt_id for attempt in self.attempts}) != len(self.attempts):
            raise ValueError("attempt IDs must be unique")
        if len({attempt.provider for attempt in self.attempts}) != 1:
            raise ValueError("attempts must share a provider")
        if self.lab is not None and self.field is not None:
            raise ValueError("collection has at most one measurement")
        measurement = self.lab if self.lab is not None else self.field
        successes = [
            a for a in self.attempts if a.state in {DataState.AVAILABLE, DataState.PARTIAL}
        ]
        if measurement is None:
            if successes:
                raise ValueError("successful attempts require a measurement")
            return self
        if len(successes) != 1 or successes[0] is not self.attempts[-1]:
            raise ValueError("one completed measurement must end the attempt history")
        attempt = successes[0]
        if (attempt.attempt_id, attempt.provider, attempt.device) != (
            measurement.attempt_id,
            measurement.provider,
            measurement.device,
        ):
            raise ValueError("measurement must match its successful attempt")
        target = measurement.requested_url
        if isinstance(measurement, FieldMeasurement) and measurement.scope == "origin":
            target = measurement.record_key
        if measurement_url_key(attempt.requested_url) != measurement_url_key(target):
            raise ValueError("measurement target must match its attempt scope")
        complete = _complete(measurement)
        if (attempt.state is DataState.AVAILABLE) != complete:
            raise ValueError("measurement coverage must match attempt state")
        if not complete and attempt.reason != "missing_metrics":
            raise ValueError("partial measurement requires missing_metrics reason")
        return self


def _complete(measurement: LabMeasurement | FieldMeasurement) -> bool:
    expected = (
        {"lcp", "cls", "fcp", "tbt", "speed_index"}
        if isinstance(measurement, LabMeasurement)
        else {"lcp", "inp", "cls"}
    )
    return {
        metric.name for metric in measurement.metrics if metric.value is not None
    } == expected and (
        not isinstance(measurement, LabMeasurement) or measurement.performance_score is not None
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def _contains_secret(value: object, key: str) -> bool:
    if not key.strip():
        return False
    if isinstance(value, str):
        return key in value or key in unquote(value)
    if isinstance(value, dict):
        return any(_contains_secret(part, key) for pair in value.items() for part in pair)
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(part, key) for part in value)
    if value is None or isinstance(value, (int, float, bool)):
        return key in json.dumps(value)
    return False


def _model_contains_secret(model: BaseModel, key: str) -> bool:
    if not key.strip():
        return False
    serialized = model.model_dump_json(serialize_as_any=True)
    return key in serialized or _contains_secret(json.loads(serialized), key)


def _source_request(
    source: DiagnosticSource,
    provider: RemoteProvider,
    url: str,
    device: Device,
    retry_transient: bool,
) -> tuple[DiagnosticSource, PerformanceRequest]:
    try:
        if type(retry_transient) is not bool:
            raise ValueError
        source = DiagnosticSource.model_validate_json(
            source.model_dump_json(serialize_as_any=True, warnings=False)
        )
        request = PerformanceRequest(
            provider=provider, requested_url=url, device=device, locale=source.binding.report_locale
        )
        target = measurement_url_key(url)
        if target not in {measurement_url_key(page) for page in source.page_urls}:
            raise ValueError
        domains = {_hostname(domain) for domain in source.canonical_domains}
        if target[1] not in domains or _hostname(source.binding.domain) not in domains:
            raise ValueError
        return source, request
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise ValueError("invalid performance configuration") from None


class PerformanceProviders:
    def __init__(
        self,
        *,
        http: PerformanceHTTPClient,
        now: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._now = now
        self._sleep = sleep

    def pagespeed(
        self,
        source: DiagnosticSource,
        *,
        url: str,
        device: Device,
        api_key: SecretStr | None,
        retry_transient: bool = True,
    ) -> PerformanceCollection:
        return self._collect("pagespeed_insights", source, url, device, api_key, retry_transient)

    def crux(
        self,
        source: DiagnosticSource,
        *,
        url: str,
        device: Device,
        api_key: SecretStr | None,
        retry_transient: bool = True,
    ) -> PerformanceCollection:
        return self._collect("crux", source, url, device, api_key, retry_transient)

    def _collect(
        self,
        provider: RemoteProvider,
        source: DiagnosticSource,
        url: str,
        device: Device,
        api_key: SecretStr | None,
        retry_transient: bool,
    ) -> PerformanceCollection:
        source, request = _source_request(source, provider, url, device, retry_transient)
        key = api_key.get_secret_value() if api_key is not None else ""
        if _contains_secret(
            [json.loads(source.model_dump_json()), request.model_dump(mode="json")], key
        ):
            raise ValueError("invalid performance configuration")
        attempts, lab, field = self._scope(request, source, url, api_key, retry_transient)
        if provider == "crux" and attempts[-1].reason in {"no_record", "no_data"}:
            parts = urlsplit(url)
            origin = f"{parts.scheme}://{parts.netloc}"
            origin_request = PerformanceRequest(
                provider=provider,
                requested_url=origin,
                scope="origin",
                device=device,
                locale=source.binding.report_locale,
            )
            origin_attempts, lab, field = self._scope(
                origin_request, source, url, api_key, retry_transient
            )
            attempts.extend(origin_attempts)
        collection = PerformanceCollection.model_validate_json(
            PerformanceCollection(
                attempts=tuple(attempts),
                lab=lab,
                field=field,
            ).model_dump_json(serialize_as_any=True)
        )
        if _model_contains_secret(collection, key):
            # Canonical/generated metadata cannot be redacted without inventing
            # evidence. Fail closed rather than returning a credential-bearing DTO.
            raise ValueError("invalid performance configuration")
        return collection

    def _scope(
        self,
        request: PerformanceRequest,
        source: DiagnosticSource,
        page_url: str,
        api_key: SecretStr | None,
        retry_transient: bool,
    ) -> tuple[list[ProviderAttempt], LabMeasurement | None, FieldMeasurement | None]:
        provider = request.provider
        attempts = []
        lab = None
        field = None
        for index in range(2 if retry_transient else 1):
            attempt_id = str(uuid4())
            started_at = self._now()
            response = self._http.send(request, api_key=api_key)
            ended_at = self._now()
            state: MeasurementState = DataState.FAILED
            reason: str | None = None
            if response.failure is not None:
                reason = response.failure
                if reason in {"missing_key", "invalid_key", "unsafe_target"}:
                    state = DataState.UNAVAILABLE
            elif response.status_code is None or not 200 <= response.status_code <= 299:
                reason = f"http_{response.status_code}"
                if response.status_code in {401, 403, 429}:
                    state = DataState.UNAVAILABLE
                elif provider == "crux" and response.status_code == 404:
                    state = DataState.UNAVAILABLE
                    reason = "no_record"
            else:
                try:
                    payload = decode_provider_json(response.body)
                    if provider == "pagespeed_insights":
                        lab = normalize_psi(
                            payload,
                            request=request,
                            binding=source.binding,
                            attempt_id=attempt_id,
                            evidence_id=str(uuid4()),
                            allowed_final_urls=source.page_urls,
                        )
                        complete = _complete(lab)
                    else:
                        field = normalize_crux(
                            payload,
                            request=request,
                            binding=source.binding,
                            attempt_id=attempt_id,
                            evidence_id=str(uuid4()),
                            observed_at=ended_at,
                            page_url=page_url,
                        )
                        complete = _complete(field)
                    measurement = lab if lab is not None else field
                    key = api_key.get_secret_value() if api_key is not None else ""
                    if measurement is not None and _model_contains_secret(measurement, key):
                        lab = None
                        field = None
                        reason = "sensitive_response"
                    else:
                        state = DataState.AVAILABLE if complete else DataState.PARTIAL
                        reason = None if complete else "missing_metrics"
                except NormalizationError as error:
                    reason = error.code
                    if reason in {"target_mismatch", "device_mismatch"}:
                        state = DataState.UNKNOWN
                    elif reason == "no_data":
                        state = DataState.UNAVAILABLE
            attempts.append(
                ProviderAttempt(
                    attempt_id=attempt_id,
                    provider=provider,
                    requested_url=request.requested_url,
                    device=request.device,
                    started_at=started_at,
                    ended_at=ended_at,
                    state=state,
                    reason=reason,
                    http_status=response.status_code,
                )
            )
            transient = response.failure in {"timeout", "transport_error"} or (
                response.failure is None and response.status_code in {429, 500, 502, 503, 504}
            )
            if index == 0 and retry_transient and transient:
                self._sleep(1.0)
            else:
                break
        return attempts, lab, field
