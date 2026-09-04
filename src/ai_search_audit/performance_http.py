"""One-attempt HTTP boundary for remote performance providers.

Bodies remain transient memory and are excluded from response repr. This module
does not promise a physical memory wipe or interruption of arbitrary synchronous
resolver/transport code; its deadline is checked between blocking operations.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Self
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import Field, HttpUrl, SecretStr, model_validator

from ai_search_audit.crawler import Resolver, validate_public_url
from ai_search_audit.performance_models import Device, FrozenPerformanceModel, PublicURL

RemoteProvider = Literal["pagespeed_insights", "crux"]
TransportFailure = Literal[
    "missing_key",
    "invalid_key",
    "unsafe_target",
    "timeout",
    "transport_error",
    "redirect",
    "response_too_large",
    "unsupported_encoding",
    "sensitive_response",
]


class HTTPRequestLimits(FrozenPerformanceModel):
    timeout_seconds: float = Field(default=60.0, strict=True, ge=1, le=120)
    max_response_bytes: int = Field(default=4194304, strict=True, ge=1, le=8388608)


_DEFAULT_LIMITS = HTTPRequestLimits()


class PerformanceRequest(FrozenPerformanceModel):
    provider: RemoteProvider
    requested_url: PublicURL
    device: Device
    locale: Literal["pl", "en"]
    scope: Literal["url", "origin"] = "url"

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        if self.scope == "origin":
            parts = urlsplit(self.requested_url)
            if self.provider != "crux" or parts.path not in {"", "/"} or "?" in self.requested_url:
                raise ValueError("origin scope requires CrUX and a query-free origin URL")
        return self


@dataclass(frozen=True)
class ProviderHTTPResponse:
    status_code: int | None
    body: bytes = field(default=b"", repr=False)
    failure: TransportFailure | None = None


def _redact_trace(event: str, info: dict[str, object]) -> None:
    # httpcore invokes this request-local hook before formatting DEBUG output.
    # Completed/failed details contain raw headers and exception text. Started
    # info is also used as live operation kwargs and must remain untouched.
    if event.endswith((".complete", ".failed")):
        info.clear()


class _PrivateMetadataTransport(httpx.BaseTransport):
    """Prevent library logs from exposing provider metadata, without global edits."""

    def __init__(self, transport: httpx.BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.extensions["trace"] = _redact_trace
        response = self._transport.handle_request(request)
        # httpx logs these immediately after the transport returns. Fall back to
        # the status-derived phrase and default protocol rather than remote text.
        response.extensions.pop("reason_phrase", None)
        response.extensions.pop("http_version", None)
        # httpx builds a next_request even with redirects disabled. Remove the
        # unused target before malformed/non-HTTP locations can be parsed.
        if 300 <= response.status_code < 400:
            response.headers.pop("Location", None)
        return response

    def close(self) -> None:
        self._transport.close()


class PerformanceHTTPClient:
    """Fresh client/transport per send unless an owned transport is injected.

    Ownership of an injected transport starts when it is acquired for a network
    attempt; it is then closed afterward. Preflight failures leave an unused
    injected transport with its caller. Callers supplying single-use transports
    must construct a new wrapper per network attempt; reusable test transports
    may support repeated sends. The default transport is always constructed
    afresh, so retries by a coordinator use a new pool.
    """

    def __init__(
        self,
        *,
        limits: HTTPRequestLimits = _DEFAULT_LIMITS,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._resolver = resolver
        self._limits = limits
        self._clock = clock

    def send(
        self, request: PerformanceRequest, *, api_key: SecretStr | None
    ) -> ProviderHTTPResponse:
        request = PerformanceRequest.model_validate_json(
            request.model_dump_json(serialize_as_any=True)
        )
        limits = HTTPRequestLimits.model_validate_json(
            self._limits.model_dump_json(serialize_as_any=True)
        )
        key = api_key.get_secret_value() if api_key is not None else ""
        if not key.strip():
            return ProviderHTTPResponse(None, failure="missing_key")
        if len(key) > 512 or any(not 33 <= ord(char) <= 126 for char in key):
            return ProviderHTTPResponse(None, failure="invalid_key")
        if key in request.requested_url or key in unquote(request.requested_url):
            return ProviderHTTPResponse(None, failure="unsafe_target")
        try:
            # PublicURL has already rejected ambiguous raw syntax. Pydantic now
            # supplies modern IDNA normalization for the DNS-only safety check.
            parts = urlsplit(request.requested_url)
            dns_url = str(HttpUrl(f"{parts.scheme}://{parts.netloc}/"))
            validate_public_url(dns_url, resolver=self._resolver)
        except (ValueError, OSError):
            return ProviderHTTPResponse(None, failure="unsafe_target")
        params: dict[str, str] | None = None
        payload: dict[str, str | list[str]] | None = None
        if request.provider == "pagespeed_insights":
            method = "GET"
            endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
            params = {
                "url": request.requested_url,
                "strategy": request.device,
                "category": "performance",
                "locale": request.locale,
            }
        else:
            method = "POST"
            endpoint = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
            payload = {
                request.scope: request.requested_url,
                "formFactor": "PHONE" if request.device == "mobile" else "DESKTOP",
                "metrics": [
                    "largest_contentful_paint",
                    "interaction_to_next_paint",
                    "cumulative_layout_shift",
                ],
            }
        status: int | None = None
        deadline = self._clock() + limits.timeout_seconds
        try:
            with httpx.Client(
                transport=_PrivateMetadataTransport(
                    self._transport
                    if self._transport is not None
                    else httpx.HTTPTransport(retries=0, trust_env=False)
                ),
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(limits.timeout_seconds),
                headers={
                    "X-Goog-Api-Key": key,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            ) as client:
                with client.stream(method, endpoint, params=params, json=payload) as response:
                    status = response.status_code
                    if 300 <= status < 400:
                        return ProviderHTTPResponse(status, failure="redirect")
                    if not 200 <= status < 300:
                        return ProviderHTTPResponse(status)
                    if (
                        response.headers.get("Content-Encoding", "identity").strip().lower()
                        != "identity"
                    ):
                        return ProviderHTTPResponse(status, failure="unsupported_encoding")
                    declared = response.headers.get("Content-Length", "").strip()
                    # Compare decimal strings so even huge declarations do not
                    # require unbounded integer conversion. Actual bytes remain
                    # authoritative for absent, invalid, or dishonest lengths.
                    if declared.isascii() and declared.isdecimal():
                        digits = declared.lstrip("0") or "0"
                        maximum = str(limits.max_response_bytes)
                        if len(digits) > len(maximum) or (
                            len(digits) == len(maximum) and digits > maximum
                        ):
                            return ProviderHTTPResponse(status, failure="response_too_large")
                    body_buffer = bytearray()
                    if self._clock() >= deadline:
                        return ProviderHTTPResponse(status, failure="timeout")
                    # Do not request buffered 64KiB chunks: a tiny-byte drip must
                    # return control for a deadline check on each network chunk.
                    for chunk in response.iter_raw():
                        if self._clock() >= deadline:
                            return ProviderHTTPResponse(status, failure="timeout")
                        if len(body_buffer) + len(chunk) > limits.max_response_bytes:
                            return ProviderHTTPResponse(status, failure="response_too_large")
                        for offset in range(0, len(chunk), 65536):
                            body_buffer.extend(chunk[offset : offset + 65536])
                    if self._clock() >= deadline:
                        return ProviderHTTPResponse(status, failure="timeout")
                    body = bytes(body_buffer)
                    if key.encode("ascii") in body:
                        return ProviderHTTPResponse(status, failure="sensitive_response")
                    return ProviderHTTPResponse(status, body)
        except httpx.TimeoutException:
            return ProviderHTTPResponse(status, failure="timeout")
        except httpx.HTTPError:
            return ProviderHTTPResponse(status, failure="transport_error")
