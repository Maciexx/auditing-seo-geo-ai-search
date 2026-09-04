"""One fresh, bounded Responses request per selected canonical prompt.

Raw provider JSON stays transient; citations never trigger local fetches. Request
deadlines are checked between blocking operations and rely on HTTP timeouts for
network reads, not on interrupting arbitrary injected synchronous transports.
No retry follows an uncertain paid outcome. Collection performs no persistence.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal, cast
from uuid import uuid4

import httpx
from pydantic import SecretStr, TypeAdapter

from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_responses
from ai_search_audit.diagnostic_models import (
    BenchmarkCitation,
    BenchmarkResponseInput,
    DiagnosticSource,
    FrozenDiagnosticModel,
    FrozenPrompt,
    _lossless_diagnostic_values,
    _normalize_citation_host,
)
from ai_search_audit.diagnostic_observations import (
    DiagnosticObservationRun,
    ObservationAttempt,
    ObservationCitationAnnotation,
    ObservationSearchAction,
    ObservationUsage,
    assemble_observation_run,
)
from ai_search_audit.observation_profile import (
    MAX_REQUEST_BYTES,
    RESPONSES_ENDPOINT,
    ObservationProfile,
    observation_setup,
    request_payload,
)
from ai_search_audit.observation_usage import (
    ObservationCostEstimate,
    estimate_attempt,
    next_request_reservation,
    normalize_usage,
    price_is_current,
    trusted_price_snapshot,
)
from ai_search_audit.performance_http import _PrivateMetadataTransport
from ai_search_audit.performance_providers import _contains_secret

ErrorCategory = Literal[
    "timeout",
    "transport_error",
    "http_429",
    "http_5xx",
    "http_error",
    "invalid_key",
    "malformed_response",
    "response_too_large",
    "sensitive_response",
    "model_mismatch",
    "tier_mismatch",
    "invalid_usage",
    "invalid_citations",
    "unsupported_response",
]
StopReason = Literal[
    "paid_not_authorized",
    "missing_key",
    "invalid_key",
    "pricing_unavailable",
    "unknown_cost",
    "allowance_exhausted",
]
_CITATION = TypeAdapter(BenchmarkCitation)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")


class ObservationCollection(FrozenDiagnosticModel):
    run: DiagnosticObservationRun
    estimates: tuple[ObservationCostEstimate, ...]
    stop_reason: StopReason | None
    limitations: tuple[str, ...] = (
        "API observations are not consumer-interface measurements.",
        "Ambiguous phrase-only entity matches remain unknown.",
        "Cost estimates are not invoices; operational limits are not provider monetary caps.",
        "Hosted search input may exceed the next-request operational reservation.",
        "store=false does not establish zero provider retention.",
    )


@dataclass(frozen=True)
class _HTTPResult:
    body: bytes = field(default=b"", repr=False)
    request_id: str | None = None
    error: ErrorCategory | None = None


class _InvalidResponse(Exception):
    def __init__(self, category: ErrorCategory) -> None:
        super().__init__(category)
        self.category = category


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _safe_secret(value: object, key: str) -> bool:
    if _contains_secret(value, key):
        return True
    if isinstance(value, str):
        decoded = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), value)
        return _contains_secret(decoded, key)
    if isinstance(value, dict):
        return any(_safe_secret(part, key) for pair in value.items() for part in pair)
    if isinstance(value, (list, tuple)):
        return any(_safe_secret(part, key) for part in value)
    return False


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON keys")
    return result


def _decode(body: bytes, key: str) -> dict[str, object]:
    try:
        payload: object = json.loads(body, object_pairs_hook=_unique_object)
        _lossless_diagnostic_values(payload)
    except (ValueError, RecursionError):
        raise _InvalidResponse("malformed_response") from None
    if _safe_secret(payload, key):
        raise _InvalidResponse("sensitive_response")
    if not isinstance(payload, dict):
        raise _InvalidResponse("malformed_response")
    return cast(dict[str, object], payload)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _InvalidResponse("malformed_response")
    return cast(dict[str, object], value)


def _array(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise _InvalidResponse("unsupported_response")
    return value


def _search_action(item: dict[str, object]) -> ObservationSearchAction:
    action = _object(item.get("action"))
    action_type = action.get("type")
    if action_type not in {"search", "open_page", "find_in_page"}:
        raise _InvalidResponse("unsupported_response")
    status = item.get("status")
    if status in {"in_progress", "searching"}:
        status = "incomplete"
    if status not in {"completed", "incomplete", "failed"}:
        raise _InvalidResponse("unsupported_response")
    sources = None
    if action.get("sources") is not None:
        raw_sources = _array(action["sources"], 100)
        sources = tuple(
            _CITATION.validate_python(_object(entry).get("url"))
            for entry in raw_sources
            if _object(entry).get("type") == "url"
        )
        if len(sources) != len(raw_sources):
            raise _InvalidResponse("unsupported_response")
    return ObservationSearchAction.model_validate(
        {
            "action": action_type,
            "status": status,
            "call_id": _identifier(item.get("id")),
            "consulted_sources": sources,
        }
    )


def _mentions(text: str, source: DiagnosticSource, *, grounded: bool) -> bool | None:
    """No semantic classifier: exact public host in visible text is decisive.

    Entity-name candidates, including lookalike/embedded domains, remain unknown.
    Citation/consulted metadata never manufactures a visible-text mention.
    """
    domains = {_normalize_citation_host(domain) for domain in source.canonical_domains}
    # Parse original host spelling with the same modern IDNA policy as citations.
    # Case folding the whole answer first would conflate distinct straße/strasse.
    # Remove whole URLs before scanning bare hosts; a query/fragment/path cannot
    # masquerade as a visible canonical host. URLs are inert, never requested.
    urls = re.findall(r"https?://[^\s<>\[\]()\"']+", text, flags=re.IGNORECASE)
    for url in urls:
        try:
            parsed = _CITATION.validate_python(url.rstrip(".,;!"))
            if parsed.host in domains:
                return True
        except ValueError:
            pass
    bare_text = text
    for url in urls:
        bare_text = bare_text.replace(url, " ")
    # Keep combining marks (including leading/uncomposable marks) inside the
    # candidate. A word-character match can extract a different host suffix from
    # a Unicode label. Shared IDNA validation accepts or rejects the whole token.
    for host in re.findall(r"[^\s<>\[\]()\"',;!]+", bare_text):
        try:
            if _normalize_citation_host(host) in domains:
                return True
        except ValueError:
            pass
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    aliases = {domain for domain in domains}
    aliases.update(
        " ".join(unicodedata.normalize("NFKC", alias).casefold().split())
        for prompt in source.prompts
        for alias in prompt.target_entities
        if alias.strip()
    )
    if any(alias in normalized for alias in aliases):
        return None
    return False if grounded else None


@dataclass
class _ParsedOutput:
    text: str = field(default="", repr=False)
    actions: tuple[ObservationSearchAction, ...] = ()
    annotations: tuple[ObservationCitationAnnotation, ...] = ()
    metadata_complete: bool = True
    refusal: bool = False
    incomplete: bool = False


def _parse_output(value: object, profile: ObservationProfile, result: _ParsedOutput) -> None:
    """Retain the validated action prefix even when later output is unusable.

    An error still makes the attempt ineligible and its complete call inventory
    unknown. Never append duplicate IDs or actions beyond the configured bound.
    """
    texts: list[str] = []
    annotations: list[ObservationCitationAnnotation] = []
    offset = 0
    for raw_item in _array(value, 128):
        item = _object(raw_item)
        kind = item.get("type")
        if kind == "web_search_call":
            if len(result.actions) >= profile.max_tool_calls:
                raise _InvalidResponse("unsupported_response")
            action = _search_action(item)
            if action.call_id is not None and any(
                previous.call_id == action.call_id for previous in result.actions
            ):
                raise _InvalidResponse("malformed_response")
            result.actions = (*result.actions, action)
        elif kind == "reasoning":
            # Reasoning is not visible answer text or mention evidence.
            for summary in _array(item.get("summary", []), 128):
                if _object(summary).get("type") != "summary_text":
                    raise _InvalidResponse("unsupported_response")
        elif kind == "message":
            if item.get("role") != "assistant" or item.get("status") not in {
                "completed",
                "incomplete",
                "in_progress",
            }:
                raise _InvalidResponse("unsupported_response")
            result.incomplete |= item.get("status") != "completed"
            for raw_part in _array(item.get("content"), 128):
                part = _object(raw_part)
                if part.get("type") == "refusal":
                    if not isinstance(part.get("refusal"), str):
                        raise _InvalidResponse("malformed_response")
                    result.refusal = True
                    continue
                if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise _InvalidResponse("unsupported_response")
                text = cast(str, part["text"])
                if texts:
                    offset += 1  # one explicit separator between visible content blocks
                metadata = part.get("annotations")
                if metadata is None:
                    result.metadata_complete = False
                else:
                    try:
                        for raw_annotation in _array(metadata, 200):
                            entry = _object(raw_annotation)
                            if entry.get("type") != "url_citation":
                                raise ValueError
                            annotation = ObservationCitationAnnotation.model_validate(
                                {
                                    "url": entry.get("url"),
                                    "start_index": entry.get("start_index"),
                                    "end_index": entry.get("end_index"),
                                }
                            )
                            if annotation.end_index > len(text):
                                raise ValueError
                            annotations.append(
                                ObservationCitationAnnotation(
                                    url=annotation.url,
                                    start_index=annotation.start_index + offset,
                                    end_index=annotation.end_index + offset,
                                )
                            )
                            if len(annotations) > 200:
                                raise ValueError
                    except (ValueError, _InvalidResponse):
                        raise _InvalidResponse("invalid_citations") from None
                texts.append(text)
                offset += len(text)
        else:
            raise _InvalidResponse("unsupported_response")
    result.text = "\n".join(texts)
    result.annotations = tuple(annotations)


def _normalize(
    http: _HTTPResult,
    source: DiagnosticSource,
    profile: ObservationProfile,
    prompt: FrozenPrompt,
    started_at: datetime,
    ended_at: datetime,
    key: str,
) -> tuple[ObservationAttempt, BenchmarkResponseInput | None]:
    error = http.error
    returned_model = None
    returned_tier = None
    usage: ObservationUsage | None = None
    parsed = _ParsedOutput()
    response = None
    status: Literal["completed", "incomplete", "refused", "no_text", "failed"] = "failed"
    if error is None:
        try:
            payload = _decode(http.body, key)
            returned_model = _identifier(payload.get("model"))
            returned_tier = _identifier(payload.get("service_tier"))
            try:
                usage = normalize_usage(payload.get("usage"))
                if (
                    usage is not None
                    and usage.output_tokens is not None
                    and usage.output_tokens > profile.max_output_tokens
                ):
                    raise ValueError
            except ValueError:
                raise _InvalidResponse("invalid_usage") from None
            _parse_output(payload.get("output"), profile, parsed)
            if payload.get("object") != "response" or (
                payload.get("status") == "completed"
                and payload.get("incomplete_details") is not None
            ):
                raise _InvalidResponse("unsupported_response")
            if returned_model != profile.model_id:
                raise _InvalidResponse("model_mismatch")
            if returned_tier != "default":
                raise _InvalidResponse("tier_mismatch")
            if payload.get("status") not in {"completed", "incomplete"}:
                raise _InvalidResponse("unsupported_response")
            if payload.get("error") is not None:
                raise _InvalidResponse("unsupported_response")
            if payload.get("status") == "incomplete" or parsed.incomplete:
                status = "incomplete"
            elif parsed.refusal:
                status = "refused"
            elif not parsed.text.strip():
                status = "no_text"
            else:
                status = "completed"
                grounded = any(
                    a.action == "search" and a.status == "completed" for a in parsed.actions
                )
                response = BenchmarkResponseInput(
                    prompt_id=prompt.prompt_id,
                    prompt_text=prompt.text,
                    observed_at=ended_at,
                    response_text=parsed.text,
                    grounded=grounded,
                    brand_mentioned=_mentions(parsed.text, source, grounded=grounded),
                    citations=tuple(dict.fromkeys(a.url for a in parsed.annotations)),
                    citations_complete=parsed.metadata_complete,
                    complete=True,
                    response_truncated=False,
                    inspection_scope="full_response",
                )
        except _InvalidResponse as exc:
            error = exc.category
        except (ValueError, TypeError, RecursionError):
            error = "malformed_response"
    if error is not None:
        status = "failed"
        response = None
    attempt = ObservationAttempt(
        attempt_id=str(uuid4()),
        prompt_id=prompt.prompt_id,
        locale=prompt.locale,
        started_at=started_at,
        ended_at=ended_at,
        requested_model=profile.model_id,
        returned_model=returned_model,
        requested_service_tier="default",
        returned_service_tier=returned_tier,
        status=status,
        complete=status == "completed",
        error_category=error,
        request_id=http.request_id,
        search_actions=parsed.actions,
        usage=usage,
        response_hash=hashlib.sha256(parsed.text.encode()).hexdigest()
        if response is not None
        else None,
        citation_annotations=parsed.annotations if response is not None else None,
        citation_metadata_complete=parsed.metadata_complete if response is not None else False,
    )
    return attempt, response


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OpenAIObservations:
    """Owned transport per attempted request; no live calls without explicit authority."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], datetime] = _utc_now,
        clock: Callable[[], float] = time.monotonic,
        pricing_date: date | None = None,
    ) -> None:
        self._transport, self._now, self._clock = transport, now, clock
        self._pricing_date = pricing_date

    def collect(
        self,
        source: DiagnosticSource,
        *,
        profile: ObservationProfile,
        api_key: SecretStr | None,
        paid_authorized: bool = False,
    ) -> ObservationCollection:
        key = api_key.get_secret_value() if api_key is not None else ""
        try:
            source = DiagnosticSource.model_validate(_lossless_diagnostic_values(source))
            profile = ObservationProfile.model_validate(_lossless_diagnostic_values(profile))
            if type(paid_authorized) is not bool:
                raise ValueError
            prompts = tuple(p for p in source.prompts if p.prompt_id in profile.selected_prompt_ids)
            if tuple(p.prompt_id for p in prompts) != profile.selected_prompt_ids:
                raise ValueError
            setup = observation_setup(profile, source.binding.report_locale)
            worksheet = prepare_benchmark_worksheet(source, setup.benchmark_setup)
            validate_benchmark_responses(source, worksheet, [])
            if _safe_secret([source.model_dump(mode="json"), profile.model_dump(mode="json")], key):
                raise ValueError
            for prompt in prompts:
                if _identifier(prompt.prompt_id) is None or _identifier(prompt.locale) is None:
                    raise ValueError
                encoded = json.dumps(request_payload(profile, prompt.text), ensure_ascii=False)
                if len(encoded.encode()) > MAX_REQUEST_BYTES:
                    raise ValueError
        except (ValueError, TypeError, AttributeError, RecursionError):
            raise ValueError("invalid observation configuration") from None
        today = self._pricing_date or self._now().date()
        snapshot = trusted_price_snapshot()
        stop: StopReason | None = None
        if not paid_authorized:
            stop = "paid_not_authorized"
        elif not key.strip():
            stop = "missing_key"
        elif len(key) > 512 or any(not 33 <= ord(char) <= 126 for char in key):
            stop = "invalid_key"
        elif not price_is_current(snapshot, today):
            stop = "pricing_unavailable"
        attempts: list[ObservationAttempt] = []
        responses: list[BenchmarkResponseInput] = []
        estimates: list[ObservationCostEstimate] = []
        spent = Decimal(0)
        seen_call_ids: set[str] = set()
        for prompt in prompts:
            if stop is not None:
                break
            if spent + next_request_reservation(profile, prompt.text) > (
                profile.operational_allowance_usd
            ):
                stop = "allowance_exhausted"
                break
            started = self._now()
            http = self._send(profile, prompt.text, key)
            attempt, response = _normalize(http, source, profile, prompt, started, self._now(), key)
            if any(a.call_id in seen_call_ids for a in attempt.search_actions if a.call_id):
                # Duplicate provider IDs are not unique evidence identifiers.
                # Do not invent replacements or lose the earlier paid response.
                values = attempt.model_dump(mode="python")
                values.update(
                    status="failed",
                    complete=False,
                    error_category="malformed_response",
                    response_hash=None,
                    citation_annotations=None,
                    citation_metadata_complete=False,
                    search_actions=tuple(
                        a.model_copy(update={"call_id": None}) if a.call_id in seen_call_ids else a
                        for a in attempt.search_actions
                    ),
                )
                attempt = ObservationAttempt.model_validate(values)
                response = None
            seen_call_ids.update(a.call_id for a in attempt.search_actions if a.call_id is not None)
            attempts.append(attempt)
            if response is not None:
                responses.append(response)
            estimate = estimate_attempt(attempt, snapshot, on=today)
            estimates.append(estimate)
            if estimate.amount_usd is None:
                stop = "unknown_cost"
            else:
                spent += estimate.amount_usd
                if spent >= profile.operational_allowance_usd:
                    stop = "allowance_exhausted"
        sample = validate_benchmark_responses(source, worksheet, responses)
        run = assemble_observation_run(
            source,
            worksheet,
            setup,
            profile.selected_prompt_ids,
            sample,
            tuple(attempts),
            price_provenance=snapshot if price_is_current(snapshot, today) else None,
        )
        result = ObservationCollection(run=run, estimates=tuple(estimates), stop_reason=stop)
        if _safe_secret(result.model_dump(mode="json"), key):
            raise ValueError("invalid observation configuration")
        return result

    def _send(self, profile: ObservationProfile, prompt: str, key: str) -> _HTTPResult:
        deadline = self._clock() + profile.timeout_seconds
        request_id = None
        try:
            with httpx.Client(
                transport=_PrivateMetadataTransport(
                    self._transport
                    if self._transport is not None
                    else httpx.HTTPTransport(retries=0, trust_env=False)
                ),
                trust_env=False,
                follow_redirects=False,
                timeout=httpx.Timeout(profile.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            ) as client:
                with client.stream(
                    "POST", RESPONSES_ENDPOINT, json=request_payload(profile, prompt)
                ) as response:
                    request_id = _identifier(response.headers.get("x-request-id"))
                    if _safe_secret(request_id, key):
                        return _HTTPResult(error="sensitive_response")
                    code = response.status_code
                    if not 200 <= code < 300:
                        category: ErrorCategory = "http_error"
                        if code in {401, 403}:
                            category = "invalid_key"
                        elif code == 429:
                            category = "http_429"
                        elif 500 <= code < 600:
                            category = "http_5xx"
                        return _HTTPResult(request_id=request_id, error=category)
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        return _HTTPResult(request_id=request_id, error="unsupported_response")
                    declared = response.headers.get("content-length", "").strip()
                    if declared.isascii() and declared.isdecimal():
                        digits = declared.lstrip("0") or "0"
                        limit = str(profile.max_response_bytes)
                        if len(digits) > len(limit) or (
                            len(digits) == len(limit) and digits > limit
                        ):
                            return _HTTPResult(request_id=request_id, error="response_too_large")
                    body = bytearray()
                    if self._clock() >= deadline:
                        return _HTTPResult(request_id=request_id, error="timeout")
                    for chunk in response.iter_raw():
                        if self._clock() >= deadline:
                            return _HTTPResult(request_id=request_id, error="timeout")
                        if len(body) + len(chunk) > profile.max_response_bytes:
                            return _HTTPResult(request_id=request_id, error="response_too_large")
                        body.extend(chunk)
                    if self._clock() >= deadline:
                        return _HTTPResult(request_id=request_id, error="timeout")
                    return _HTTPResult(bytes(body), request_id)
        except httpx.TimeoutException:
            return _HTTPResult(request_id=request_id, error="timeout")
        except httpx.HTTPError:
            return _HTTPResult(request_id=request_id, error="transport_error")
