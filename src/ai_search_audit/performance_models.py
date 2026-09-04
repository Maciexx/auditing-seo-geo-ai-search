"""Immutable performance observations, separate from provider attempt outcomes.

These DTOs do not fetch URLs or establish DNS/source binding. Publication must
revalidate serialized JSON: Pydantic's model_copy(update=...) is unchecked.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import UTC, date
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

from ai_search_audit.diagnostic_models import DiagnosticBinding
from ai_search_audit.models import DataState


class FrozenPerformanceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Percent = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
Device = Literal["mobile", "desktop"]
Provider = Literal["pagespeed_insights", "crux", "lighthouse_local"]
MetricName = Literal["lcp", "inp", "cls", "fcp", "tbt", "speed_index"]
MeasurementState = Literal[
    DataState.AVAILABLE,
    DataState.PARTIAL,
    DataState.UNAVAILABLE,
    DataState.UNKNOWN,
    DataState.FAILED,
]
SafeID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")]


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must contain nonwhitespace characters")
    return value


def _known_setting(value: str) -> str:
    if value.strip().casefold() in {"unknown", "unavailable", "n/a"}:
        raise ValueError("unknown settings must be null, not placeholders")
    return value


ShortText = Annotated[str, Field(min_length=1, max_length=200), AfterValidator(_nonblank)]
KnownSetting = Annotated[ShortText, AfterValidator(_known_setting)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


def _hostname(host: str) -> str:
    """Validate and produce a comparison key, without rewriting the stored URL."""
    if "%" in host:
        raise ValueError("escaped URL hostnames are forbidden")
    if not host.isascii():
        # Use the URL parser's modern IDNA mapping, not Python's IDNA2003
        # codec, which incorrectly conflates straße.example and strasse.example.
        # ASCII hosts still pass directly through our strict syntax/IP checks;
        # the parser must not repair escaped hosts or alternate IPv4 notation.
        normalized = HttpUrl(f"https://{host}/").host
        if normalized is None:
            raise ValueError("invalid URL hostname")
        try:
            ipaddress.ip_address(normalized)
        except ValueError:
            pass
        else:
            # Unicode conversion is for DNS names only, never for repairing
            # alternate numeric forms into IP literals. IP input must be ASCII.
            raise ValueError("IP literals must use ASCII input")
        host = normalized
    host = host.lower().removesuffix(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("localhost URLs are forbidden")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.split(".")
        if len(host) > 253 or any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels
        ):
            raise ValueError("invalid URL hostname") from None
        # Do not treat alternate IPv4 notation as DNS names (127.1, octal, hex).
        if labels[-1].isdigit() or re.fullmatch(r"0x[0-9a-f]+", labels[-1]):
            raise ValueError("ambiguous numeric URL hostname") from None
    else:
        if not address.is_global:
            raise ValueError("non-global IP literals are forbidden")
        return address.compressed
    return host


def _public_url(value: str) -> str:
    """Pure raw syntax/IP checks, not DNS verification or a network SSRF boundary."""
    if (
        "\\" in value
        or "#" in value
        or any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
    ):
        raise ValueError("URL cannot contain fragments, backslashes, whitespace or controls")
    parts = urlsplit(value)
    host = parts.hostname
    if parts.scheme not in {"http", "https"} or not host:
        raise ValueError("URL requires HTTP(S) and a hostname")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URL credentials are forbidden")
    # Accessing port rejects invalid integers and out-of-range values. Check the
    # authority separately because urlsplit accepts empty ports and bracket junk.
    _ = parts.port
    if parts.netloc.startswith("["):
        if not re.fullmatch(r"\[[0-9A-Fa-f:.]+\](?::[0-9]+)?", parts.netloc):
            raise ValueError("malformed bracketed URL authority")
    elif not re.fullmatch(r"[^:\[\]]+(?::[0-9]+)?", parts.netloc):
        raise ValueError("malformed URL authority")
    _hostname(host)
    return value


PublicURL = Annotated[str, Field(min_length=1, max_length=4096), AfterValidator(_public_url)]
WarningText = Annotated[str, Field(min_length=1, max_length=1000), AfterValidator(_nonblank)]


class PerformanceMetric(FrozenPerformanceModel):
    name: MetricName
    value: Nonnegative | None
    unit: Literal["ms", "unitless"]

    @model_validator(mode="after")
    def validate_unit(self) -> Self:
        if self.unit != ("unitless" if self.name == "cls" else "ms"):
            raise ValueError("CLS is unitless; all other metrics use ms")
        return self


class FieldPeriod(FrozenPerformanceModel):
    first_date: date
    last_date: date

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.first_date > self.last_date:
            raise ValueError("field period must end on or after its first date")
        return self


SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class LocalRuntimeFingerprint(FrozenPerformanceModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )
    image_id: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    architecture: Literal["arm64", "amd64"]
    node_version: KnownSetting
    lighthouse_version: KnownSetting
    chrome_version: KnownSetting
    puppeteer_version: KnownSetting
    runner_sha256: SHA256
    seccomp_sha256: SHA256
    dependency_lock_sha256: SHA256
    policy_sha256: SHA256


class LighthouseRequest(FrozenPerformanceModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )
    requested_url: PublicURL
    device: Device
    locale: Literal["pl", "en"]


class LabConfiguration(FrozenPerformanceModel):
    form_factor: Device
    throttling_method: Literal["simulate", "devtools", "provided"] | None
    locale: KnownSetting | None
    screen_width: PositiveInt | None
    screen_height: PositiveInt | None
    device_scale_factor: PositiveFloat | None
    cpu_slowdown_multiplier: PositiveFloat | None
    rtt_ms: Nonnegative | None
    throughput_kbps: Nonnegative | None
    local_runtime: LocalRuntimeFingerprint | None = None


class ProviderAttempt(FrozenPerformanceModel):
    attempt_id: SafeID
    provider: Provider
    requested_url: PublicURL
    device: Device
    started_at: AwareDatetime
    ended_at: AwareDatetime
    state: MeasurementState
    reason: ShortText | None
    http_status: Annotated[int, Field(strict=True, ge=100, le=599)] | None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("provider attempt cannot end before it starts")
        if self.provider == "lighthouse_local" and self.http_status is not None:
            raise ValueError("local Lighthouse attempts do not have HTTP status")
        if self.state is DataState.AVAILABLE:
            if self.reason is not None:
                raise ValueError("available attempts cannot contain a reason")
        elif self.reason is None:
            raise ValueError("nonavailable attempts require an explicit reason")
        if (
            self.http_status is not None
            and not 200 <= self.http_status <= 299
            and self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        ):
            raise ValueError("non-2xx HTTP status cannot be available or partial")
        return self


class LabMeasurement(FrozenPerformanceModel):
    binding: DiagnosticBinding
    evidence_id: SafeID
    attempt_id: SafeID
    provider: Literal["pagespeed_insights", "lighthouse_local"]
    kind: Literal["lab"] = "lab"
    requested_url: PublicURL
    final_url: PublicURL
    device: Device
    observed_at: AwareDatetime
    lighthouse_version: KnownSetting
    chrome_version: KnownSetting | None
    configuration: LabConfiguration
    warnings: Annotated[tuple[WarningText, ...], Field(max_length=30)] = ()
    performance_score: Percent | None
    metrics: Annotated[tuple[PerformanceMetric, ...], Field(max_length=5)]

    @model_validator(mode="after")
    def validate_lab(self) -> Self:
        runtime = self.configuration.local_runtime
        if self.provider == "pagespeed_insights" and runtime is not None:
            raise ValueError("PSI cannot carry local runtime provenance")
        if self.provider == "lighthouse_local" and (
            runtime is None
            or runtime.chrome_version != self.chrome_version
            or runtime.lighthouse_version != self.lighthouse_version
        ):
            raise ValueError("local Lighthouse requires matching runtime provenance")
        names = [metric.name for metric in self.metrics]
        if "inp" in names or len(names) != len(set(names)):
            raise ValueError("lab metrics must be unique and cannot contain INP")
        if self.device != self.configuration.form_factor:
            raise ValueError("lab device must match configuration form factor")
        if self.performance_score is None and not any(
            metric.value is not None for metric in self.metrics
        ):
            raise ValueError("empty lab data belongs in a provider attempt")
        return self


def _origin(value: str) -> tuple[str, str, int]:
    parts = urlsplit(value)
    # URL validation has established a hostname. Normalize only this comparison,
    # never the saved requested URL or provider record key.
    host = _hostname(parts.hostname or "")
    port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
    return parts.scheme, host, port


def _url_identity(value: str) -> tuple[str, str, int, str, str]:
    parts = urlsplit(value)
    return (*_origin(value), parts.path or "/", parts.query)


class URLNormalization(FrozenPerformanceModel):
    """Allowlisted CrUX provider attestation, not independently authenticated proof."""

    original_url: PublicURL
    normalized_url: PublicURL

    @model_validator(mode="after")
    def validate_origin(self) -> Self:
        if _origin(self.original_url) != _origin(self.normalized_url):
            raise ValueError("URL normalization cannot cross origins")
        return self


class FieldMeasurement(FrozenPerformanceModel):
    binding: DiagnosticBinding
    evidence_id: SafeID
    attempt_id: SafeID
    provider: Literal["crux"] = "crux"
    kind: Literal["field"] = "field"
    requested_url: PublicURL
    record_key: PublicURL
    url_normalization: URLNormalization | None = None
    scope: Literal["url", "origin"]
    device: Device
    observed_at: AwareDatetime
    percentile: Literal[75] = 75
    period: FieldPeriod
    metrics: Annotated[tuple[PerformanceMetric, ...], Field(max_length=3)]

    @model_validator(mode="after")
    def validate_field(self) -> Self:
        names = [metric.name for metric in self.metrics]
        if not set(names) <= {"lcp", "inp", "cls"} or len(names) != len(set(names)):
            raise ValueError("field metrics must be unique LCP, INP or CLS")
        if not any(metric.value is not None for metric in self.metrics):
            raise ValueError("empty field data belongs in a provider attempt")
        record = urlsplit(self.record_key)
        if self.scope == "origin" and (record.path not in {"", "/"} or "?" in self.record_key):
            raise ValueError("origin record keys cannot contain a page path or query")
        if _origin(self.requested_url) != _origin(self.record_key):
            raise ValueError("record key must share the requested URL origin")
        if self.url_normalization is not None and (
            self.scope != "url"
            or _url_identity(self.url_normalization.original_url)
            != _url_identity(self.requested_url)
            or _url_identity(self.url_normalization.normalized_url)
            != _url_identity(self.record_key)
        ):
            raise ValueError("URL normalization must bind the requested URL to its record key")
        if self.period.last_date > self.observed_at.astimezone(UTC).date():
            raise ValueError("field period cannot extend past the UTC observation date")
        return self
