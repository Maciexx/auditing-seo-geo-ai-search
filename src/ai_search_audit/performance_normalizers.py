"""Pure, allowlisted projections of transient provider responses."""

import json
import math
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from functools import wraps
from typing import Literal, ParamSpec, TypeVar, cast
from urllib.parse import urlsplit

from pydantic import AwareDatetime, TypeAdapter

from ai_search_audit.diagnostic_models import DiagnosticBinding
from ai_search_audit.performance_http import PerformanceRequest
from ai_search_audit.performance_models import (
    FieldMeasurement,
    FieldPeriod,
    KnownSetting,
    LabConfiguration,
    LabMeasurement,
    LighthouseRequest,
    LocalRuntimeFingerprint,
    PublicURL,
    _origin,
)

ErrorCode = Literal[
    "malformed_response",
    "target_mismatch",
    "device_mismatch",
    "runtime_error",
    "no_data",
    "invalid_metrics",
]
P = ParamSpec("P")
T = TypeVar("T")


class NormalizationError(ValueError):
    """A fixed reason code without provider-controlled exception details."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        super().__init__(code)


def _safe_errors(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except NormalizationError as error:
            raise NormalizationError(error.code) from None
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            raise NormalizationError("malformed_response") from None

    return wrapped


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise NormalizationError("malformed_response")
    return cast(dict[str, object], value)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NormalizationError("malformed_response")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise NormalizationError("malformed_response")
    return number


@_safe_errors
def decode_provider_json(body: bytes) -> dict[str, object]:
    """Decode strict JSON without duplicate keys or nonfinite numeric values."""
    return _object(
        json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_finite_json_float,
            parse_float=_finite_json_float,
        )
    )


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError("invalid_metrics")
    try:
        number = float(value)
    except OverflowError:
        raise NormalizationError("invalid_metrics") from None
    if not math.isfinite(number) or number < 0:
        raise NormalizationError("invalid_metrics")
    return number


@_safe_errors
def measurement_url_key(url: str) -> tuple[str, str, int, str, str]:
    """Validate raw syntax, then compare URL components without rewriting paths."""
    try:
        TypeAdapter(PublicURL).validate_python(url)
        parts = urlsplit(url)
        return (*_origin(url), parts.path or "/", parts.query)
    except (ValueError, TypeError):
        raise NormalizationError("target_mismatch") from None


def _url(value: object) -> str:
    if not isinstance(value, str):
        raise NormalizationError("target_mismatch")
    measurement_url_key(value)
    return value


def _configuration(
    settings: dict[str, object], request: PerformanceRequest | LighthouseRequest
) -> LabConfiguration:
    form_factors = [
        settings[key] for key in ("formFactor", "emulatedFormFactor") if key in settings
    ]
    if not form_factors or any(value != request.device for value in form_factors):
        raise NormalizationError("device_mismatch")
    screen = _object(settings.get("screenEmulation", {}))
    throttling = _object(settings.get("throttling", {}))
    disabled = screen.get("disabled", False)
    if type(disabled) is not bool:
        raise NormalizationError("malformed_response")
    dimensions: dict[str, int | float | None] = {}
    for source, target in (
        ("width", "screen_width"),
        ("height", "screen_height"),
        ("deviceScaleFactor", "device_scale_factor"),
    ):
        value = screen.get(source)
        try:
            _number(value)
        except NormalizationError:
            raise NormalizationError("malformed_response") from None
        if source != "deviceScaleFactor" and value is not None and type(value) is not int:
            raise NormalizationError("malformed_response")
        dimensions[target] = None if disabled or value == 0 else cast(int | float | None, value)
    return LabConfiguration.model_validate(
        dict(
            form_factor=request.device,
            locale=settings.get("locale"),
            throttling_method=settings.get("throttlingMethod"),
            **dimensions,
            cpu_slowdown_multiplier=throttling.get("cpuSlowdownMultiplier"),
            rtt_ms=throttling.get("rttMs"),
            throughput_kbps=throttling.get("throughputKbps"),
        )
    )


def _chrome_version(result: dict[str, object]) -> str | None:
    environment = _object(result.get("environment", {}))
    user_agent = environment.get("hostUserAgent")
    if user_agent is None:
        return None
    if not isinstance(user_agent, str):
        raise NormalizationError("malformed_response")
    versions = re.findall(
        r"(?<![A-Za-z0-9])(?:HeadlessChrome|Chrome)/([0-9]+(?:\.[0-9]+){3})(?![\w.])", user_agent
    )
    return versions[0] if len(versions) == 1 else None


@_safe_errors
def normalize_psi(
    payload: dict[str, object],
    *,
    request: PerformanceRequest,
    binding: DiagnosticBinding,
    attempt_id: str,
    evidence_id: str,
    allowed_final_urls: tuple[str, ...],
) -> LabMeasurement:
    if request.provider != "pagespeed_insights" or request.scope != "url":
        raise NormalizationError("malformed_response")
    result = _object(payload["lighthouseResult"])
    return _normalize_lhr(
        result,
        request=request,
        binding=binding,
        attempt_id=attempt_id,
        evidence_id=evidence_id,
        allowed_final_urls=allowed_final_urls,
    )


LOCAL_CAPABILITY_WARNINGS = (
    "Local runtime does not support chunked request bodies.",
    "Local runtime does not support plaintext WebSocket CONNECT to port 80; WSS 443 is supported.",
)


@_safe_errors
def normalize_lighthouse(
    result: dict[str, object],
    *,
    request: LighthouseRequest,
    binding: DiagnosticBinding,
    attempt_id: str,
    evidence_id: str,
    allowed_final_urls: tuple[str, ...],
    runtime: LocalRuntimeFingerprint,
) -> LabMeasurement:
    request = LighthouseRequest.model_validate(vars(request) | (request.model_extra or {}))
    runtime = LocalRuntimeFingerprint.model_validate(vars(runtime) | (runtime.model_extra or {}))
    # Chromium reduces UA minor/build/patch to zero. The local runtime attests
    # the full browser product version separately; never invent it from the UA.
    reduced_version = runtime.chrome_version.split(".", 1)[0] + ".0.0.0"
    reported_chrome = _chrome_version(result)
    host_ua = _object(result.get("environment", {})).get("hostUserAgent")
    if (
        result.get("lighthouseVersion") != runtime.lighthouse_version
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", runtime.chrome_version) is None
        or not isinstance(host_ua, str)
        or len(re.findall(r"(?<![A-Za-z0-9])(?:HeadlessChrome|Chrome)/", host_ua)) != 1
        or reported_chrome not in {runtime.chrome_version, reduced_version}
    ):
        raise NormalizationError("runtime_error")
    return _normalize_lhr(
        result,
        request=request,
        binding=binding,
        attempt_id=attempt_id,
        evidence_id=evidence_id,
        allowed_final_urls=allowed_final_urls,
        runtime=runtime,
    )


def _normalize_lhr(
    result: dict[str, object],
    *,
    request: PerformanceRequest | LighthouseRequest,
    binding: DiagnosticBinding,
    attempt_id: str,
    evidence_id: str,
    allowed_final_urls: tuple[str, ...],
    runtime: LocalRuntimeFingerprint | None = None,
) -> LabMeasurement:
    requested_url, final_url = _url(result.get("requestedUrl")), _url(result.get("finalUrl"))
    request_key = measurement_url_key(request.requested_url)
    if measurement_url_key(requested_url) != request_key:
        raise NormalizationError("target_mismatch")
    allowed_keys = {request_key, *(measurement_url_key(url) for url in allowed_final_urls)}
    if measurement_url_key(final_url) not in allowed_keys:
        raise NormalizationError("target_mismatch")
    runtime_error = result.get("runtimeError")
    if runtime_error is not None and (
        not isinstance(runtime_error, dict) or runtime_error.get("code") != "NO_ERROR"
    ):
        raise NormalizationError("runtime_error")
    fetch_time = result["fetchTime"]
    if not isinstance(fetch_time, str):
        raise NormalizationError("malformed_response")
    observed_at = datetime.fromisoformat(fetch_time.replace("Z", "+00:00"))
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise NormalizationError("malformed_response")
    settings = _object(result["configSettings"])
    configuration = _configuration(settings, request)
    if runtime is not None:
        configuration = LabConfiguration.model_validate(
            configuration.model_dump() | {"local_runtime": runtime.model_dump()}
        )
    lighthouse_version = TypeAdapter(KnownSetting).validate_python(result["lighthouseVersion"])
    chrome_version: str | None = TypeAdapter(KnownSetting | None).validate_python(
        runtime.chrome_version if runtime is not None else _chrome_version(result)
    )
    warnings = TypeAdapter(
        LabMeasurement.model_fields["warnings"].rebuild_annotation()
    ).validate_python(result.get("runWarnings", ()))
    if runtime is not None:
        warnings = (*warnings, *LOCAL_CAPABILITY_WARNINGS)
    audits = _object(result.get("audits", {}))
    metrics = []
    for provider_name, name in (
        ("largest-contentful-paint", "lcp"),
        ("cumulative-layout-shift", "cls"),
        ("first-contentful-paint", "fcp"),
        ("total-blocking-time", "tbt"),
        ("speed-index", "speed_index"),
    ):
        audit = _object(audits.get(provider_name, {}))
        value = _number(audit.get("numericValue"))
        if value is not None and audit.get("numericUnit") != (
            "unitless" if name == "cls" else "millisecond"
        ):
            raise NormalizationError("invalid_metrics")
        metrics.append(dict(name=name, value=value, unit="unitless" if name == "cls" else "ms"))
    score = _number(
        _object(_object(result.get("categories", {})).get("performance", {})).get("score")
    )
    if score is not None and score > 1:
        raise NormalizationError("invalid_metrics")
    if score is None and all(metric["value"] is None for metric in metrics):
        raise NormalizationError("no_data")
    return LabMeasurement.model_validate(
        dict(
            binding=binding,
            attempt_id=attempt_id,
            evidence_id=evidence_id,
            provider="lighthouse_local" if runtime is not None else "pagespeed_insights",
            requested_url=requested_url,
            final_url=final_url,
            device=request.device,
            observed_at=observed_at,
            lighthouse_version=lighthouse_version,
            chrome_version=chrome_version,
            configuration=configuration,
            warnings=warnings,
            performance_score=score * 100 if score is not None else None,
            metrics=metrics,
        )
    )


def _date(value: object) -> date:
    parts = _object(value)
    if any(type(parts[field]) is not int for field in ("year", "month", "day")):
        raise NormalizationError("malformed_response")
    return date(cast(int, parts["year"]), cast(int, parts["month"]), cast(int, parts["day"]))


def _crux_target(
    payload: dict[str, object],
    key: dict[str, object],
    request: PerformanceRequest,
    page_url: str,
) -> str:
    if ("url" in key) + ("origin" in key) != 1 or request.scope not in key:
        raise NormalizationError("target_mismatch")
    record_url = _url(key[request.scope])
    record_key = measurement_url_key(record_url)
    request_key = measurement_url_key(request.requested_url)
    page_key = measurement_url_key(page_url)
    if record_key[:3] != request_key[:3] or page_key[:3] != request_key[:3]:
        raise NormalizationError("target_mismatch")
    if "urlNormalizationDetails" in payload:
        details = payload["urlNormalizationDetails"]
        if not isinstance(details, dict) or request.scope != "url":
            raise NormalizationError("target_mismatch")
        if (
            measurement_url_key(_url(details.get("originalUrl"))) != request_key
            or _url(details.get("normalizedUrl")) != record_url
        ):
            raise NormalizationError("target_mismatch")
    elif request.scope == "url" and record_key != request_key:
        raise NormalizationError("target_mismatch")
    if request.scope == "url":
        if page_key != request_key:
            raise NormalizationError("target_mismatch")
    elif record_key[3] != "/" or "?" in record_url:
        raise NormalizationError("target_mismatch")
    return record_url


@_safe_errors
def normalize_crux(
    payload: dict[str, object],
    *,
    request: PerformanceRequest,
    binding: DiagnosticBinding,
    attempt_id: str,
    evidence_id: str,
    observed_at: datetime,
    page_url: str,
) -> FieldMeasurement:
    if request.provider != "crux":
        raise NormalizationError("malformed_response")
    record = _object(payload["record"])
    key = _object(record["key"])
    record_url = _crux_target(payload, key, request, page_url)
    # _crux_target has validated this mapping. Retain only the two URL fields,
    # so persisted validation does not lose the reason a different key was accepted.
    normalization = None
    if "urlNormalizationDetails" in payload:
        details = _object(payload["urlNormalizationDetails"])
        normalization = {
            "original_url": _url(details["originalUrl"]),
            "normalized_url": _url(details["normalizedUrl"]),
        }
    if key.get("formFactor") != ("PHONE" if request.device == "mobile" else "DESKTOP"):
        raise NormalizationError("device_mismatch")
    period_data = _object(record["collectionPeriod"])
    period = FieldPeriod(
        first_date=_date(period_data["firstDate"]), last_date=_date(period_data["lastDate"])
    )
    observed_at = TypeAdapter(AwareDatetime).validate_python(observed_at)
    if period.last_date > observed_at.astimezone(UTC).date():
        raise NormalizationError("malformed_response")
    source_metrics = _object(record.get("metrics", {}))
    metrics = []
    for provider_name, name in (
        ("largest_contentful_paint", "lcp"),
        ("interaction_to_next_paint", "inp"),
        ("cumulative_layout_shift", "cls"),
    ):
        value = _object(_object(source_metrics.get(provider_name, {})).get("percentiles", {})).get(
            "p75"
        )
        if value is not None:
            if name == "cls":
                if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                    raise NormalizationError("invalid_metrics")
                value = float(value)
            elif type(value) is not int:
                raise NormalizationError("invalid_metrics")
        metrics.append(
            dict(name=name, value=_number(value), unit="unitless" if name == "cls" else "ms")
        )
    if all(metric["value"] is None for metric in metrics):
        raise NormalizationError("no_data")
    return FieldMeasurement.model_validate(
        dict(
            binding=binding,
            attempt_id=attempt_id,
            evidence_id=evidence_id,
            requested_url=page_url,
            record_key=record_url,
            url_normalization=normalization,
            scope=request.scope,
            device=request.device,
            observed_at=observed_at,
            period=period,
            metrics=metrics,
        )
    )
