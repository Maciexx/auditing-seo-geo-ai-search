from __future__ import annotations

import hashlib
import ipaddress
import math
from datetime import date, datetime
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    StrictBool,
    field_validator,
    model_validator,
)

from ai_search_audit.comparisons import AIVisibilityComparison
from ai_search_audit.models import ClaimModality, DataState, FindingStatus, RuleState
from ai_search_audit.project_models import normalize_canonical_domain, validate_project_id


class FrozenDiagnosticModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _lossless_diagnostic_values(value: object) -> object:
    """Revalidate unchecked provider DTOs before serializers can omit or coerce data."""
    if isinstance(value, BaseModel):
        if value.__pydantic_extra__ or set(value.__dict__) - set(type(value).model_fields):
            raise ValueError("unchecked model contains unknown fields")
        return _lossless_diagnostic_values(value.__dict__)
    if isinstance(value, dict):
        return {key: _lossless_diagnostic_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_lossless_diagnostic_values(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("binary values are not normalized diagnostic evidence")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite values are not diagnostic evidence")
    return value


class DiagnosticBinding(FrozenDiagnosticModel):
    project_id: str
    source_version: str
    audit_id: str
    report_locale: Literal["pl", "en"]
    domain: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("project_id", "source_version", "audit_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)


class FrozenPrompt(FrozenDiagnosticModel):
    prompt_id: str
    pack_version: str
    locale: str
    intent: str
    text: str
    target_entities: tuple[str, ...] = ()
    query_themes: tuple[str, ...] = ()
    expected_evidence_needs: tuple[str, ...] = ()
    suggested_providers: tuple[str, ...] = ()


class DiagnosticSource(FrozenDiagnosticModel):
    binding: DiagnosticBinding
    prompts: tuple[FrozenPrompt, ...]
    page_urls: tuple[str, ...]
    canonical_domains: tuple[str, ...]


BenchmarkSetting = Annotated[str, Field(min_length=1, max_length=200)]
BenchmarkAccountState = Literal["anonymous", "signed_in_free", "signed_in_paid", "enterprise"]


class BenchmarkSetup(FrozenDiagnosticModel):
    """Observed setup categories only; unknown dimensions remain null."""

    provider: BenchmarkSetting | None = None
    product: BenchmarkSetting | None = None
    model_id: BenchmarkSetting | None = None
    interface: Literal["api", "consumer_ui"] | None = None
    search_mode: Literal["enabled", "disabled"] | None = None
    locale: BenchmarkSetting | None = None
    market: BenchmarkSetting | None = None
    account_state: BenchmarkAccountState | None = None
    reset_method: BenchmarkSetting | None = None

    @field_validator("provider", "product", "model_id", "locale", "market", "reset_method")
    @classmethod
    def reject_unknown_placeholders(cls, value: str | None) -> str | None:
        if value is not None and value.strip().casefold() in {"", "unknown", "unavailable", "n/a"}:
            raise ValueError("unknown benchmark settings must be null, not a placeholder")
        return value


class BenchmarkWorksheet(FrozenDiagnosticModel):
    """Locally consistent inputs; use source-bound validation before accepting imports."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    binding: DiagnosticBinding
    prompts: tuple[FrozenPrompt, ...]
    pack_version: str
    pack_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup: BenchmarkSetup | None = None
    setup_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    instructions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        from ai_search_audit.benchmark import _validate_worksheet_integrity

        _validate_worksheet_integrity(self)
        return self


class BenchmarkMetrics(FrozenDiagnosticModel):
    state: Literal[
        DataState.AVAILABLE,
        DataState.PARTIAL,
        DataState.UNKNOWN,
        DataState.UNAVAILABLE,
        DataState.FAILED,
    ]
    expected: Annotated[int, Field(strict=True, ge=1)]
    measured: Annotated[int, Field(strict=True, ge=0)]
    mention_rate: float | None = Field(default=None, ge=0, le=100)
    citation_rate: float | None = Field(default=None, ge=0, le=100)
    citation_measured: Annotated[int, Field(strict=True, ge=0)]
    coverage: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.state not in {DataState.AVAILABLE, DataState.PARTIAL} and (
            self.mention_rate is not None or self.citation_rate is not None
        ):
            raise ValueError("nonnumeric benchmark states cannot contain rates")
        if not 0 <= self.citation_measured <= self.measured <= self.expected:
            raise ValueError("invalid benchmark denominator")
        if (
            self.coverage != self.measured / self.expected
            or self.citation_coverage != self.citation_measured / self.expected
            or self.confidence != self.coverage
        ):
            raise ValueError("benchmark coverage must match observed denominators")
        if self.state in {DataState.AVAILABLE, DataState.PARTIAL}:
            if not self.measured or self.mention_rate is None:
                raise ValueError("numeric benchmark requires measured mentions")
            if (self.citation_rate is None) != (self.citation_measured == 0):
                raise ValueError("citation rate requires a known citation denominator")
            complete = self.measured == self.citation_measured == self.expected
            if (self.state is DataState.AVAILABLE) != complete:
                raise ValueError("benchmark state does not match coverage")
        return self


MAX_BENCHMARK_EXCERPT_CHARS = 2000
MAX_BENCHMARK_RESPONSE_BYTES = 2 * 1024 * 1024
BenchmarkHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _normalize_citation_host(value: str) -> str:
    """Use the citation URL's IDNA policy without changing canonical project identities."""
    if not value or any(
        character in "/\\?#@:%[]" or character.isspace() or ord(character) < 32
        for character in value
    ):
        raise ValueError("approved citation domain must be a bare hostname")
    # Convert Unicode with the same parser as citations before the legacy ASCII
    # syntax check: IDNA2003 would otherwise conflate straße with distinct strasse.
    hostname = HttpUrl(f"https://{value}/").host
    if hostname is None:
        raise ValueError("citation domain must contain a hostname")
    return normalize_canonical_domain(hostname)


def _validate_citation_url(value: object) -> object:
    """Validate inert URLs before HttpUrl can normalize away malformed input."""
    if isinstance(value, HttpUrl):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("citation URL must be a string")
    if "\\" in value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("malformed citation URL")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("citation URL must use HTTP or HTTPS")
    if parts.username is not None or parts.password is not None:
        raise ValueError("credentials are not allowed in citation URLs")
    _ = parts.port
    _normalize_citation_host(parts.hostname)
    return value


BenchmarkCitation = Annotated[HttpUrl, BeforeValidator(_validate_citation_url)]


class _BenchmarkResponseMetadata(FrozenDiagnosticModel):
    prompt_id: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    observed_at: AwareDatetime
    grounded: StrictBool | None
    brand_mentioned: StrictBool | None
    citations: tuple[BenchmarkCitation, ...]
    citations_complete: StrictBool
    complete: StrictBool
    response_truncated: StrictBool
    inspection_scope: Literal["full_response", "excerpt", "unknown"]

    @field_validator("observed_at", mode="before")
    @classmethod
    def explicit_timestamp(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError("response timestamp requires an explicit aware datetime")
        return value

    @model_validator(mode="after")
    def validate_inspection(self) -> Self:
        if self.response_truncated and self.inspection_scope == "full_response":
            raise ValueError("truncated response cannot establish full-response inspection")
        return self


class BenchmarkResponseInput(_BenchmarkResponseMetadata):
    """Ephemeral captured text; import this, never provider-computed sample results."""

    response_text: str = Field(min_length=1, max_length=MAX_BENCHMARK_RESPONSE_BYTES, repr=False)
    response_hash: BenchmarkHash | None = None

    @field_validator("response_text")
    @classmethod
    def bounded_response(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("response text cannot be blank")
        if len(value.encode("utf-8")) > MAX_BENCHMARK_RESPONSE_BYTES:
            raise ValueError("response text exceeds 2 MiB limit")
        return value


class BenchmarkResponse(_BenchmarkResponseMetadata):
    """Bounded evidence. A source digest is provenance, not proof without its source text."""

    response_excerpt: str = Field(min_length=1, max_length=MAX_BENCHMARK_EXCERPT_CHARS)
    # Digest of the actual captured text, including when that capture is partial.
    response_hash: BenchmarkHash
    # A full-response digest exists only when the actual capture was not truncated.
    source_response_hash: BenchmarkHash | None
    excerpt_hash: BenchmarkHash
    captured_text_length: Annotated[int, Field(strict=True, ge=1, le=MAX_BENCHMARK_RESPONSE_BYTES)]
    excerpt_truncated: StrictBool

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if hashlib.sha256(self.response_excerpt.encode("utf-8")).hexdigest() != self.excerpt_hash:
            raise ValueError("response excerpt hash mismatch")
        if len(self.response_excerpt) != min(
            self.captured_text_length, MAX_BENCHMARK_EXCERPT_CHARS
        ):
            raise ValueError("response excerpt length does not match captured text")
        if (
            self.captured_text_length == len(self.response_excerpt)
            and self.response_hash != self.excerpt_hash
        ):
            raise ValueError("captured response hash mismatch")
        if self.source_response_hash != (None if self.response_truncated else self.response_hash):
            raise ValueError("source response hash does not match capture scope")
        if self.excerpt_truncated != (
            self.response_truncated or self.captured_text_length > MAX_BENCHMARK_EXCERPT_CHARS
        ):
            raise ValueError("response excerpt truncation does not match capture scope")
        return self


class BenchmarkSample(FrozenDiagnosticModel):
    """Trusted processing output, not an alternate intake format for provider results."""

    worksheet: BenchmarkWorksheet
    approved_domains: tuple[str, ...]
    responses: tuple[BenchmarkResponse, ...]
    metrics: BenchmarkMetrics

    @property
    def measurable_prompt_ids(self) -> tuple[str, ...]:
        from ai_search_audit.benchmark import _mention_eligible

        return tuple(sorted(item.prompt_id for item in self.responses if _mention_eligible(item)))

    @property
    def citation_measurable_prompt_ids(self) -> tuple[str, ...]:
        from ai_search_audit.benchmark import _citation_state, _mention_eligible

        return tuple(
            sorted(
                item.prompt_id
                for item in self.responses
                if _mention_eligible(item)
                and _citation_state(item, self.approved_domains) is not None
            )
        )

    @model_validator(mode="after")
    def validate_sample(self) -> Self:
        from ai_search_audit.benchmark import _validate_sample_integrity

        _validate_sample_integrity(self)
        return self


class BenchmarkComparison(FrozenDiagnosticModel):
    state: Literal[DataState.AVAILABLE, DataState.PARTIAL, DataState.UNKNOWN]
    baseline_metrics: BenchmarkMetrics
    follow_up_metrics: BenchmarkMetrics
    baseline_citation_prompt_ids: tuple[str, ...]
    follow_up_citation_prompt_ids: tuple[str, ...]
    baseline_approved_domains: tuple[str, ...]
    follow_up_approved_domains: tuple[str, ...]
    mention_comparison: AIVisibilityComparison | None = None
    mention_delta: float | None = Field(default=None, ge=-100, le=100)
    citation_delta: float | None = Field(default=None, ge=-100, le=100)
    limitations: tuple[str, ...] = ()
    causality: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    chronology_statement: Literal["A difference between samples does not establish causality."] = (
        "A difference between samples does not establish causality."
    )

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.state is DataState.UNKNOWN:
            if self.mention_delta is not None or self.citation_delta is not None:
                raise ValueError("nonnumeric benchmark comparison cannot contain deltas")
            if not self.limitations:
                raise ValueError("noncomparable benchmark requires a limitation")
            if (
                self.mention_comparison is not None
                and self.mention_comparison.state is not DataState.UNKNOWN
            ):
                raise ValueError("nonnumeric benchmark cannot contain a numeric legacy comparison")
        else:
            if (
                self.mention_comparison is None
                or self.mention_comparison.state not in {DataState.AVAILABLE, DataState.PARTIAL}
                or self.mention_delta is None
                or self.mention_delta != self.mention_comparison.absolute_delta
            ):
                raise ValueError("benchmark mention delta requires the legacy comparison decision")
            for actual, legacy_rate in (
                (self.baseline_metrics.mention_rate, self.mention_comparison.baseline_value),
                (self.follow_up_metrics.mention_rate, self.mention_comparison.follow_up_value),
            ):
                if actual is None or legacy_rate is None or abs(actual - legacy_rate) > 1e-9:
                    raise ValueError("benchmark rates do not match the legacy comparison")
            for metrics, canonical_ids, measured_ids in (
                (
                    self.baseline_metrics,
                    self.mention_comparison.baseline_canonical_prompt_ids,
                    self.mention_comparison.baseline_measurable_prompt_ids,
                ),
                (
                    self.follow_up_metrics,
                    self.mention_comparison.follow_up_canonical_prompt_ids,
                    self.mention_comparison.follow_up_measurable_prompt_ids,
                ),
            ):
                if metrics.expected != len(canonical_ids) or metrics.measured != len(measured_ids):
                    raise ValueError("benchmark denominators must match legacy prompt references")
            if self.citation_delta is not None:
                if (
                    not self.baseline_citation_prompt_ids
                    or self.baseline_citation_prompt_ids != self.follow_up_citation_prompt_ids
                    or self.baseline_approved_domains != self.follow_up_approved_domains
                ):
                    raise ValueError(
                        "citation delta requires identical measured prompts and domains"
                    )
                before, after = (
                    self.baseline_metrics.citation_rate,
                    self.follow_up_metrics.citation_rate,
                )
                if (
                    before is None
                    or after is None
                    or abs(self.citation_delta - (after - before)) > 1e-9
                ):
                    raise ValueError("citation delta does not match measured rates")
            if self.state is DataState.AVAILABLE and (
                self.baseline_metrics.state is not DataState.AVAILABLE
                or self.follow_up_metrics.state is not DataState.AVAILABLE
                or self.mention_comparison.state is not DataState.AVAILABLE
                or self.citation_delta is None
            ):
                raise ValueError("available benchmark comparison requires full coverage")
            for citation_ids, mention_ids in (
                (
                    self.baseline_citation_prompt_ids,
                    self.mention_comparison.baseline_measurable_prompt_ids,
                ),
                (
                    self.follow_up_citation_prompt_ids,
                    self.mention_comparison.follow_up_measurable_prompt_ids,
                ),
            ):
                if not set(citation_ids).issubset(mention_ids):
                    raise ValueError(
                        "citation references must resolve to measurable mention prompts"
                    )
        for prompt_ids, metrics in (
            (self.baseline_citation_prompt_ids, self.baseline_metrics),
            (self.follow_up_citation_prompt_ids, self.follow_up_metrics),
        ):
            if (
                len(prompt_ids) != len(set(prompt_ids))
                or len(prompt_ids) != metrics.citation_measured
            ):
                raise ValueError("citation prompt references do not match measured counts")
        for domains in (self.baseline_approved_domains, self.follow_up_approved_domains):
            if domains != tuple(sorted({_normalize_citation_host(domain) for domain in domains})):
                raise ValueError("approved citation domains must be normalized and unique")
        return self


class Section(FrozenDiagnosticModel):
    section_id: str
    capture_id: str
    locator: str
    level: int
    heading: str
    heading_path: tuple[str, ...]
    text: str
    block_kinds: tuple[str, ...]


class ExtractedContent(FrozenDiagnosticModel):
    capture_id: str
    text: str
    sections: tuple[Section, ...]
    extraction_version: str = "1.0.0"
    fallback_used: bool
    limitations: tuple[str, ...]


MAX_CAPTURE_BYTES = 2 * 1024 * 1024


def is_html_content_type(content_type: str | None) -> bool:
    media_type = (content_type or "").split(";", 1)[0].strip().casefold()
    return media_type in {"text/html", "application/xhtml+xml"}


CaptureKind = Literal["raw", "rendered"]
ConsentState = Literal["none", "accepted", "rejected", "unknown"]
Viewport = tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]]
DiagnosticState = Literal[
    DataState.AVAILABLE,
    DataState.PARTIAL,
    DataState.UNKNOWN,
    DataState.UNAVAILABLE,
    DataState.FAILED,
]


def _validate_capture_url(value: str) -> str:
    """Validate inert capture metadata; live collection also validates resolved addresses."""
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("capture URL must be HTTP or HTTPS")
    if parts.username is not None or parts.password is not None:
        raise ValueError("credentials are not allowed in capture URLs")
    _ = parts.port
    hostname = parts.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("private capture URLs are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("private or reserved capture URLs are not allowed")
    return value


class CaptureMetadata(FrozenDiagnosticModel):
    kind: CaptureKind
    url: str
    final_url: str
    observed_at: AwareDatetime
    locale: Annotated[str, Field(min_length=1, max_length=64)] | None
    # An anonymous collection label, never a browser cookie or account identifier.
    session_key: Annotated[str, Field(pattern=r"^anonymous-[A-Za-z0-9_-]{1,96}$")] | None
    consent_state: ConsentState
    account_state: Literal["anonymous"] = "anonymous"
    viewport: Viewport | None = None
    collector: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    status_code: Annotated[int, Field(strict=True, ge=100, le=599)] | None
    complete: StrictBool
    truncated: StrictBool

    @field_validator("url", "final_url")
    @classmethod
    def validate_urls(cls, value: str) -> str:
        return _validate_capture_url(value)

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        from ai_search_audit.crawler import _canonical_scope

        if not _canonical_scope(self.url, self.final_url):
            raise ValueError("final capture URL is outside requested canonical scope")
        if self.complete and self.truncated:
            raise ValueError("a truncated capture cannot be complete")
        if self.locale is not None and not self.locale.strip():
            raise ValueError("locale cannot be blank")
        return self


class CaptureInput(CaptureMetadata):
    """Ephemeral input only: never persist raw HTML in a diagnostic output."""

    html: str = Field(repr=False)

    @field_validator("html")
    @classmethod
    def bounded_html(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_CAPTURE_BYTES:
            raise ValueError("capture HTML exceeds 2 MiB limit")
        return value


class PageCaptureResult(FrozenDiagnosticModel):
    """Ephemeral, bounded native HTTP response or explicit collection failure."""

    url: str | None
    final_url: str | None
    observed_at: AwareDatetime
    collector: str = Field(min_length=1)
    status_code: Annotated[int, Field(strict=True, ge=100, le=599)] | None
    content_type: str | None
    body: Annotated[bytes, Field(strict=True, max_length=MAX_CAPTURE_BYTES)] | None = Field(
        default=None, repr=False
    )
    html: str | None = Field(repr=False)
    complete: StrictBool
    truncated: StrictBool
    state: Literal[DataState.AVAILABLE, DataState.UNKNOWN, DataState.FAILED]
    limitations: tuple[str, ...] = ()

    @field_validator("url", "final_url")
    @classmethod
    def validate_urls(cls, value: str | None) -> str | None:
        return _validate_capture_url(value) if value is not None else None

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        from ai_search_audit.crawler import _canonical_scope

        if (
            self.url is not None
            and self.final_url is not None
            and not _canonical_scope(self.url, self.final_url)
        ):
            raise ValueError("response final URL is outside requested canonical scope")
        if self.state is DataState.FAILED:
            if (
                self.body is not None
                or self.html is not None
                or self.complete
                or not self.limitations
            ):
                raise ValueError("failed collection requires a reason and no successful body")
            return self
        if (
            self.body is None
            or self.url is None
            or self.final_url is None
            or self.status_code is None
        ):
            raise ValueError("collected response requires actual response metadata and body")
        if self.html is not None:
            if not is_html_content_type(self.content_type):
                raise ValueError("usable HTML requires an explicit supported content type")
            CaptureInput.bounded_html(self.html)
        if not self.complete or self.truncated:
            raise ValueError("successful collection cannot be incomplete or truncated")
        if self.state is DataState.AVAILABLE and (
            self.html is None or not 200 <= self.status_code < 300
        ):
            raise ValueError("available response requires successful HTTP status and usable HTML")
        if self.state is DataState.UNKNOWN and not self.limitations:
            raise ValueError("unknown response requires limitations")
        return self


class ContentCapture(CaptureMetadata):
    capture_id: str = Field(pattern=r"^capture-[0-9a-f]{64}$")
    content_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    extracted: ExtractedContent | None
    state: DiagnosticState
    limitations: tuple[str, ...] = ()

    @property
    def text(self) -> str | None:
        return self.extracted.text if self.extracted is not None else None

    @property
    def sections(self) -> tuple[Section, ...]:
        return self.extracted.sections if self.extracted is not None else ()

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.state is DataState.UNAVAILABLE:
            raise ValueError("unavailable collection has no content capture")
        if self.state is DataState.FAILED:
            if (
                self.extracted is not None
                or self.content_sha256 is not None
                or not self.limitations
            ):
                raise ValueError("failed capture requires a reason and no extracted content")
            return self
        if self.extracted is None or self.content_sha256 is None:
            raise ValueError("nonfailed capture requires extracted content and its hash")
        if self.extracted.capture_id != self.capture_id or any(
            section.capture_id != self.capture_id for section in self.sections
        ):
            raise ValueError("extraction references a different capture")
        if len({section.section_id for section in self.sections}) != len(self.sections):
            raise ValueError("duplicate section identifiers")
        if self.content_sha256 != hashlib.sha256(self.extracted.text.encode()).hexdigest():
            raise ValueError("content hash does not match extracted content")
        if self.state is DataState.AVAILABLE and (
            not self.complete
            or self.truncated
            or self.status_code is None
            or not 200 <= self.status_code < 300
        ):
            raise ValueError("available capture requires complete successful HTTP collection")
        if self.state is DataState.PARTIAL and (
            self.complete or self.status_code is None or not 200 <= self.status_code < 300
        ):
            raise ValueError("partial capture requires incomplete successful HTTP collection")
        if self.state is not DataState.AVAILABLE and not self.limitations:
            raise ValueError("nonavailable capture requires limitations")
        return self


class KeyPassage(FrozenDiagnosticModel):
    capture_id: str = Field(min_length=1)
    # Whole-capture quotes may span sections; a supplied section ID is always exact.
    section_id: Annotated[str, Field(min_length=1)] | None = None
    quote: str = Field(min_length=1, max_length=2000)

    @field_validator("quote")
    @classmethod
    def nonempty_quote(cls, value: str) -> str:
        from ai_search_audit.content_sections import normalize_text

        if not normalize_text(value):
            raise ValueError("key passage does not resolve to rendered source")
        return value


class PairDiagnostic(FrozenDiagnosticModel):
    state: DiagnosticState
    raw_capture_id: str | None
    rendered_capture_id: str | None
    limitations: tuple[str, ...] = ()
    rendered_only_passages: tuple[KeyPassage, ...] = ()

    @property
    def rendered_only_quotes(self) -> tuple[str, ...]:
        return tuple(passage.quote for passage in self.rendered_only_passages)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.state is DataState.AVAILABLE:
            if self.raw_capture_id is None or self.rendered_capture_id is None:
                raise ValueError("available pair requires both capture references")
        elif self.rendered_only_passages or not self.limitations:
            raise ValueError(
                "noncomparable pair requires limitations and no rendered-only passages"
            )
        if any(
            passage.capture_id != self.rendered_capture_id
            for passage in self.rendered_only_passages
        ):
            raise ValueError("rendered passage references a different capture")
        return self


ReviewCriterion = Literal["directness", "entity_context", "conditions", "sources", "ambiguity"]
ReviewResult = Literal["adequate", "needs_review", "unknown"]
ReviewText = Annotated[str, Field(strict=True, min_length=1, max_length=2000)]


class SectionReview(FrozenDiagnosticModel):
    """Untrusted narrative input only; trusted diagnostic metadata is pipeline-owned."""

    section_id: str = Field(min_length=1)
    criterion: ReviewCriterion
    result: ReviewResult
    quotes: tuple[ReviewText, ...]
    rationale: ReviewText

    @model_validator(mode="after")
    def validate_narrative(self) -> Self:
        from ai_search_audit.content_sections import normalize_text

        if not self.section_id.strip():
            raise ValueError("section ID cannot be blank")
        if not normalize_text(self.rationale) or any(
            not normalize_text(quote) for quote in self.quotes
        ):
            raise ValueError("rationale and quotes cannot be blank")
        if self.result != "unknown" and not self.quotes:
            raise ValueError("affirmative assessment requires quote evidence")
        return self


class DiagnosticSelectedPage(FrozenDiagnosticModel):
    url: str
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _validate_capture_url(value)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selected page reason cannot be blank")
        return value


class DiagnosticSectionReviewInput(FrozenDiagnosticModel):
    """Group untrusted section assessments against one rendered capture."""

    capture_id: str = Field(min_length=1)
    reviews: tuple[SectionReview, ...] = ()

    @model_validator(mode="after")
    def validate_review_workload(self) -> Self:
        if len({review.section_id for review in self.reviews}) > 20:
            raise ValueError("at most twenty distinct reviewed sections per capture")
        keys = [(review.section_id, review.criterion) for review in self.reviews]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate section criterion review")
        return self


class DiagnosticRunReference(FrozenDiagnosticModel):
    """An explicit same-project reference, never an arbitrary filesystem path."""

    source_version: str
    run_id: str = Field(pattern=r"^run-[1-9][0-9]*$")

    @field_validator("source_version")
    @classmethod
    def validate_source_version(cls, value: str) -> str:
        validated = validate_project_id(value)
        if validated.casefold() == "latest":
            raise ValueError("baseline source version must be explicit, not latest")
        return validated


class DiagnosticIntake(FrozenDiagnosticModel):
    """Ephemeral declarations, not verified diagnostic results or first-party metrics.

    Source-bound URLs, capture/section references and prompt content are checked by
    the coordinating pipeline. When a worksheet has no setup yet, separate setup
    records newly observed settings; two supplied setups must agree exactly.
    """

    schema_version: Literal["1.0.0"] = "1.0.0"
    expected_binding: DiagnosticBinding
    selected_pages: tuple[DiagnosticSelectedPage, ...] = Field(default=(), max_length=10)
    rendered_captures: tuple[CaptureInput, ...] = Field(default=(), max_length=10)
    key_passages: tuple[KeyPassage, ...] = ()
    section_reviews: tuple[DiagnosticSectionReviewInput, ...] = Field(default=(), max_length=10)
    worksheet: BenchmarkWorksheet | None = None
    setup: BenchmarkSetup | None = None
    responses: tuple[BenchmarkResponseInput, ...] = ()
    baseline_run: DiagnosticRunReference | None = None

    @model_validator(mode="after")
    def validate_local_consistency(self) -> Self:
        if any(capture.kind != "rendered" for capture in self.rendered_captures):
            raise ValueError("diagnostic intake accepts rendered captures only")
        selected_urls = [page.url for page in self.selected_pages]
        capture_urls = [capture.url for capture in self.rendered_captures]
        review_ids = [group.capture_id for group in self.section_reviews]
        if any(
            len(values) != len(set(values)) for values in (selected_urls, capture_urls, review_ids)
        ):
            raise ValueError("selected pages, rendered pages and review groups must be unique")
        if self.responses and self.worksheet is None:
            raise ValueError("benchmark responses require a worksheet")
        if self.worksheet is not None:
            if self.worksheet.binding != self.expected_binding:
                raise ValueError("worksheet binding contradicts expected binding")
            if (
                self.setup is not None
                and self.worksheet.setup is not None
                and self.setup != self.worksheet.setup
            ):
                raise ValueError("worksheet setup contradicts diagnostic setup")
        return self


class ValidatedSectionReview(FrozenDiagnosticModel):
    review: SectionReview
    capture_id: str
    locator: str
    evidence: tuple[KeyPassage, ...]
    assessment_kind: Literal["agent_assessment"] = "agent_assessment"
    quotation_verified: Literal[True] = True
    source_instructions: Literal["inert"] = "inert"
    evidence_review_required: Literal[True] = True
    limitations: tuple[str, ...]


class SectionDirectness(FrozenDiagnosticModel):
    section_id: str
    capture_id: str
    state: Literal[DataState.AVAILABLE, DataState.UNKNOWN]
    result: ReviewResult


class PageSectionReviews(FrozenDiagnosticModel):
    capture_id: str
    reviews: tuple[ValidatedSectionReview, ...]
    directness: tuple[SectionDirectness, ...]


class DiagnosticRule(FrozenDiagnosticModel):
    """A detached snapshot, never a nested mutable Knowledge Registry object."""

    rule_id: str
    vendor: str
    statement: str
    source_url: HttpUrl
    source_type: str
    evidence_level: Literal["A", "B", "C", "D", "E"]
    verified_at: date
    review_interval_days: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    scoring_weight: float = Field(ge=0, le=0)
    state: RuleState
    expires_at: date


ObservedContentPredicate = Literal[
    "section_heading", "section_has_body", "section_excerpt", "rendered_only_passage"
]
DiagnosticFindingStatus = Literal[FindingStatus.INFERRED, FindingStatus.REQUIRES_VERIFICATION]


class ObservedContentClaim(FrozenDiagnosticModel):
    predicate: ObservedContentPredicate
    value: Annotated[str, Field(max_length=2000)] | StrictBool
    modality: Literal[ClaimModality.OBSERVED] = ClaimModality.OBSERVED
    capture_id: str
    section_id: str | None
    locator: str | None
    evidence: tuple[KeyPassage, ...] = ()


class InferredContentClaim(FrozenDiagnosticModel):
    meaning: ReviewText
    modality: Literal[ClaimModality.INFERRED] = ClaimModality.INFERRED
    assessment_kind: Literal["agent_assessment", "audit_inference"]
    evidence: tuple[KeyPassage, ...]
    evidence_review_required: Literal[True] = True


class DiagnosticFinding(FrozenDiagnosticModel):
    finding_id: str
    rule: DiagnosticRule
    resolved_as_of: date
    status: DiagnosticFindingStatus
    observed_properties: tuple[ObservedContentClaim, ...]
    possible_impacts: tuple[InferredContentClaim, ...]
    assessments: tuple[ValidatedSectionReview, ...] = ()
    compared_capture_ids: tuple[str, ...] = ()
    evidence_review_required: Literal[True] = True
    limitations: tuple[str, ...]


class DiagnosticContract(FrozenDiagnosticModel):
    source: DiagnosticSource
    worksheet: BenchmarkWorksheet
    selected_pages: tuple[DiagnosticSelectedPage, ...] = Field(max_length=10)
    # A conditions label, not shared cookies or proof of an authenticated session.
    session_key: str
    instructions: tuple[str, ...]

    @property
    def selected_page_urls(self) -> tuple[str, ...]:
        return tuple(page.url for page in self.selected_pages)


class DiagnosticTextEvidence(FrozenDiagnosticModel):
    """A bounded excerpt and a distinct digest of the full normalized source text."""

    excerpt: str = Field(max_length=2000)
    source_sha256: BenchmarkHash
    excerpt_sha256: BenchmarkHash
    source_length: Annotated[int, Field(strict=True, ge=0)]
    truncated: StrictBool

    @model_validator(mode="after")
    def validate_excerpt(self) -> Self:
        digest = hashlib.sha256(self.excerpt.encode()).hexdigest()
        if digest != self.excerpt_sha256:
            raise ValueError("normalized excerpt hash mismatch")
        if len(self.excerpt) != min(self.source_length, 2000):
            raise ValueError("normalized excerpt length mismatch")
        if self.truncated != (self.source_length > len(self.excerpt)):
            raise ValueError("normalized excerpt truncation mismatch")
        if not self.truncated and self.source_sha256 != digest:
            raise ValueError("normalized source hash mismatch")
        return self


class DiagnosticSectionEvidence(FrozenDiagnosticModel):
    section_id: str
    capture_id: str
    locator: str
    level: int
    heading: DiagnosticTextEvidence
    body: DiagnosticTextEvidence
    heading_path: tuple[DiagnosticTextEvidence, ...]
    block_kinds: tuple[str, ...]


class DiagnosticAttemptEvidence(FrozenDiagnosticModel):
    url: str | None
    final_url: str | None
    observed_at: AwareDatetime
    collector: str
    status_code: Annotated[int, Field(strict=True, ge=100, le=599)] | None
    content_type: str | None
    body_sha256: BenchmarkHash | None
    complete: StrictBool
    truncated: StrictBool
    state: Literal[DataState.AVAILABLE, DataState.UNKNOWN, DataState.FAILED]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        from ai_search_audit.crawler import _canonical_scope

        for url in (self.url, self.final_url):
            if url is not None:
                _validate_capture_url(url)
        if (
            self.url is not None
            and self.final_url is not None
            and not _canonical_scope(self.url, self.final_url)
        ):
            raise ValueError("attempt final URL is outside the requested canonical scope")
        if self.state is DataState.FAILED:
            if self.body_sha256 is not None or self.complete or not self.limitations:
                raise ValueError("failed attempt requires limitations and no response body digest")
        elif (
            self.body_sha256 is None
            or self.url is None
            or self.final_url is None
            or self.status_code is None
            or not self.complete
            or self.truncated
        ):
            raise ValueError("collected attempt requires complete response metadata")
        if self.state is DataState.UNKNOWN and not self.limitations:
            raise ValueError("unknown attempt requires limitations")
        if self.state is DataState.AVAILABLE and (
            self.status_code is None
            or not 200 <= self.status_code < 300
            or not is_html_content_type(self.content_type)
        ):
            raise ValueError("available attempt requires successful HTML response")
        return self


class DiagnosticCaptureEvidence(FrozenDiagnosticModel):
    evidence_id: str = Field(pattern=r"^(capture|attempt)-[0-9a-f]{64}$")
    capture_id: str | None
    kind: CaptureKind
    url: str
    observed_at: AwareDatetime
    metadata: CaptureMetadata | None
    attempt: DiagnosticAttemptEvidence | None
    state: DiagnosticState
    content_sha256: BenchmarkHash | None
    text: DiagnosticTextEvidence | None
    sections: tuple[DiagnosticSectionEvidence, ...]
    passages: tuple[KeyPassage, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_capture_evidence(self) -> Self:
        _validate_capture_url(self.url)
        if self.state is DataState.UNAVAILABLE:
            raise ValueError("unavailable collection cannot contain capture evidence")
        if (self.kind == "raw") != (self.attempt is not None):
            raise ValueError("raw evidence requires the original HTTP attempt")
        if self.metadata is not None:
            if (
                self.metadata.kind != self.kind
                or self.metadata.url != self.url
                or self.metadata.observed_at != self.observed_at
                or self.capture_id != self.evidence_id
            ):
                raise ValueError("capture evidence metadata mismatch")
            if self.state is DataState.FAILED:
                if self.text is not None or not self.limitations:
                    raise ValueError("failed extraction requires no text and a limitation")
            elif self.text is None:
                raise ValueError("successful capture extraction requires text evidence")
            if self.state is DataState.AVAILABLE and (
                not self.metadata.complete
                or self.metadata.truncated
                or self.metadata.status_code is None
                or not 200 <= self.metadata.status_code < 300
            ):
                raise ValueError("available capture requires successful complete metadata")
            if self.state is DataState.PARTIAL and (
                self.metadata.complete
                or self.metadata.status_code is None
                or not 200 <= self.metadata.status_code < 300
            ):
                raise ValueError("partial capture contradicts actual collection completeness")
        elif self.capture_id is not None or self.text is not None or self.sections or self.passages:
            raise ValueError("unextracted attempt cannot contain extracted evidence")
        if self.text is None:
            if self.content_sha256 is not None or self.sections:
                raise ValueError("content digest requires extracted evidence")
        elif self.content_sha256 != self.text.source_sha256:
            raise ValueError("capture source digest mismatch")
        if self.attempt is not None:
            if self.attempt.observed_at != self.observed_at:
                raise ValueError("attempt timestamp mismatch")
            if self.attempt.url is not None and self.attempt.url != self.url:
                raise ValueError("attempt requested URL mismatch")
            if self.metadata is not None and any(
                getattr(self.attempt, field) != getattr(self.metadata, field)
                for field in (
                    "url",
                    "final_url",
                    "collector",
                    "status_code",
                    "complete",
                    "truncated",
                )
            ):
                raise ValueError("attempt and extracted metadata mismatch")
            if self.attempt.state is not DataState.AVAILABLE and self.state != self.attempt.state:
                raise ValueError("original HTTP attempt state must be preserved")
        if len({section.section_id for section in self.sections}) != len(self.sections):
            raise ValueError("duplicate evidence sections")
        if any(section.capture_id != self.capture_id for section in self.sections):
            raise ValueError("section references a different evidence capture")
        ids = {section.section_id for section in self.sections}
        if any(
            p.capture_id != self.capture_id
            or (p.section_id is not None and p.section_id not in ids)
            for p in self.passages
        ):
            raise ValueError("passage reference does not resolve to capture evidence")
        from ai_search_audit.content_sections import normalize_text

        for passage in self.passages:
            quote = normalize_text(passage.quote)
            if (
                self.text is not None
                and not self.text.truncated
                and quote not in normalize_text(self.text.excerpt)
            ):
                raise ValueError("passage does not resolve to retained complete capture text")
            section = next((s for s in self.sections if s.section_id == passage.section_id), None)
            if section is not None and not section.heading.truncated and not section.body.truncated:
                if quote not in normalize_text(
                    section.heading.excerpt + "\n" + section.body.excerpt
                ):
                    raise ValueError("passage does not resolve to retained complete section text")
        return self


class DiagnosticCollectionRange(FrozenDiagnosticModel):
    start: AwareDatetime | None
    end: AwareDatetime | None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start is None or self.end is None:
            if self.start is not None or self.end is not None or not self.reason:
                raise ValueError("empty collection range requires no timestamps and a reason")
        elif self.start > self.end or self.reason is not None:
            raise ValueError("invalid actual collection range")
        return self


class DiagnosticModuleStates(FrozenDiagnosticModel):
    raw_capture: DiagnosticState
    render_parity: DiagnosticState
    section_reviews: DiagnosticState
    benchmark: DiagnosticState


class DiagnosticCleanup(FrozenDiagnosticModel):
    status: Literal["deleted"] = "deleted"
    method: Literal["consume_owned_payload"] = "consume_owned_payload"
    completed_at: AwareDatetime


class DiagnosticBaseline(FrozenDiagnosticModel):
    reference: DiagnosticRunReference
    binding: DiagnosticBinding
    manifest_sha256: BenchmarkHash


class DiagnosticRun(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    binding: DiagnosticBinding
    pipeline_version: Literal["1.0.0"] = "1.0.0"
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    collection_range: DiagnosticCollectionRange
    selected_pages: tuple[DiagnosticSelectedPage, ...] = Field(max_length=10)
    session_key: str
    worksheet: BenchmarkWorksheet
    captures: tuple[DiagnosticCaptureEvidence, ...] = Field(max_length=20)
    pairs: tuple[PairDiagnostic, ...] = Field(max_length=10)
    section_reviews: tuple[PageSectionReviews, ...] = Field(max_length=10)
    findings: tuple[DiagnosticFinding, ...]
    benchmark: BenchmarkSample | None
    comparison: BenchmarkComparison | None
    baseline: DiagnosticBaseline | None
    module_states: DiagnosticModuleStates
    cleanup: DiagnosticCleanup

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        from ai_search_audit.diagnostic_workflow import _validate_run_integrity

        _validate_run_integrity(self)
        return self


class DiagnosticFile(FrozenDiagnosticModel):
    filename: Literal["diagnostics.json", "evidence.jsonl"]
    byte_count: Annotated[int, Field(strict=True, ge=0, le=8 * 1024 * 1024)]
    sha256: BenchmarkHash


class DiagnosticManifest(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    binding: DiagnosticBinding
    run_number: Annotated[int, Field(strict=True, ge=1)]
    files: tuple[DiagnosticFile, DiagnosticFile]
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    cleanup: DiagnosticCleanup

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if tuple(item.filename for item in self.files) != ("diagnostics.json", "evidence.jsonl"):
            raise ValueError("diagnostic manifest inventory mismatch")
        return self


class LoadedDiagnosticRun(FrozenDiagnosticModel):
    run: DiagnosticRun
    manifest: DiagnosticManifest
    manifest_sha256: BenchmarkHash
