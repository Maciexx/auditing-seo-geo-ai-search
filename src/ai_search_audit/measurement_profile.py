"""Secret-free operator configuration and pure measurement preparation."""

import hashlib
import re
from typing import Annotated, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

from ai_search_audit.benchmark import _pack_content_hash, canonical_hash
from ai_search_audit.diagnostic_models import DiagnosticBinding, DiagnosticSource, FrozenPrompt
from ai_search_audit.models import AuditRun
from ai_search_audit.performance_http import HTTPRequestLimits
from ai_search_audit.performance_models import (
    FrozenPerformanceModel,
    PublicURL,
    ShortText,
    _hostname,
)
from ai_search_audit.performance_normalizers import decode_provider_json, measurement_url_key
from ai_search_audit.prompts import validate_automatic_prompt_pack


class MeasurementProfile(FrozenPerformanceModel):
    schema_version: Literal["1.0.0", "1.1.0"] = "1.1.0"
    pagespeed_insights: StrictBool = True
    crux: StrictBool = True
    lighthouse_local: StrictBool = False
    gemini: StrictBool = False
    max_pages: int = Field(default=3, strict=True, ge=1, le=5)
    http_limits: HTTPRequestLimits = Field(default_factory=HTTPRequestLimits)
    retry_transient: StrictBool = True
    gemini_model: str | None = Field(default=None, strict=True, min_length=1, max_length=200)
    paid_use_consent: StrictBool = False
    max_prompts: int = Field(default=12, strict=True, ge=1, le=12)

    @field_validator("gemini_model")
    @classmethod
    def explicit_model(cls, value: str | None) -> str | None:
        if value is not None and (
            not re.fullmatch(r"gemini-[0-9][A-Za-z0-9._-]*", value)
            or {"latest", "auto"} & set(re.split(r"[._-]", value.casefold()))
        ):
            raise ValueError("Gemini requires an explicit versioned model, not an alias or path")
        return value

    @model_validator(mode="after")
    def supported_modules(self) -> Self:
        if self.gemini and self.gemini_model is None:
            raise ValueError("enabled Gemini requires an explicit model")
        if self.lighthouse_local and not self.pagespeed_insights:
            raise ValueError("local Lighthouse is supported only as a PSI fallback")
        return self


class MeasurementSelectedPage(FrozenPerformanceModel):
    url: PublicURL
    selection_reason: ShortText
    # DiagnosticSource has no verified page-language observations. Neither a URL
    # path nor the report locale is evidence of the page's actual language.
    locale: None = None


AttemptCount = Annotated[int, Field(strict=True, ge=0)]


class MeasurementAttemptBudget(FrozenPerformanceModel):
    """Potential attempts, not repeated successful trials or a monetary limit."""

    pagespeed_insights: AttemptCount
    crux: AttemptCount
    lighthouse_local: AttemptCount
    gemini: AttemptCount

    @property
    def total(self) -> int:
        return self.pagespeed_insights + self.crux + self.lighthouse_local + self.gemini


class MeasurementPreflight(FrozenPerformanceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    binding: DiagnosticBinding
    profile: MeasurementProfile
    selected_pages: tuple[MeasurementSelectedPage, ...] = Field(max_length=5)
    devices: tuple[Literal["mobile"], Literal["desktop"]] = ("mobile", "desktop")
    prompts: tuple[FrozenPrompt, ...] = Field(max_length=12)
    full_prompt_count: AttemptCount
    full_prompt_pack_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_attempts: MeasurementAttemptBudget
    gemini_blocked_reason: Literal["disabled", "paid_use_not_authorized"] | None


def _reject_unserialized_fields(value: object) -> None:
    """model_copy can inject unknown keys which model_dump would silently omit."""
    if isinstance(value, BaseModel):
        if set(value.__dict__) - set(type(value).model_fields):
            raise ValueError("unchecked model contains unknown fields")
        for item in value.__dict__.values():
            _reject_unserialized_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_unserialized_fields(item)


_OFFER_PATHS = frozenset({"offer", "offers", "oferta", "services", "service", "products", "shop"})
_INFORMATION_PATHS = frozenset({"about", "contact", "faq", "information", "o-nas", "kontakt"})
# Empty performance-only sources have no benchmark pack/version. This explicit
# sentinel is not a benchmark worksheet hash and must not authorize AI execution.
_EMPTY_PROMPT_PACK_HASH = canonical_hash({"pack_version": None, "prompts": []})


def _select_pages(source: DiagnosticSource, maximum: int) -> tuple[MeasurementSelectedPage, ...]:
    domains = {_hostname(domain) for domain in source.canonical_domains}
    if _hostname(source.binding.domain) not in domains:
        raise ValueError("source binding must belong to canonical domains")
    candidates: list[str] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for url in source.page_urls:
        try:
            key = measurement_url_key(url)
        except ValueError:
            continue
        if key[1] in domains and key not in seen:
            candidates.append(url)
            seen.add(key)

    selected: list[MeasurementSelectedPage] = []
    remaining = candidates.copy()
    for role, path_tokens in (
        ("homepage", frozenset()),
        ("offer", _OFFER_PATHS),
        ("information", _INFORMATION_PATHS),
    ):
        for url in remaining:
            parts = urlsplit(url)
            tokens = set(unquote(parts.path).casefold().strip("/").split("/"))
            matches = (
                parts.path in {"", "/"} and not parts.query
                if role == "homepage"
                else bool(tokens & path_tokens)
            )
            if matches:
                selected.append(
                    MeasurementSelectedPage(
                        url=url,
                        selection_reason=(
                            "Existing audited homepage URL (root path without query)."
                            if role == "homepage"
                            else f"Existing audited {role} candidate by URL-path heuristic; "
                            "role unverified."
                        ),
                    )
                )
                remaining.remove(url)
                break
    selected.extend(
        MeasurementSelectedPage(
            url=url, selection_reason="Existing audited URL order; page role unknown."
        )
        for url in remaining
    )
    return tuple(selected[:maximum])


def _validate_gemini_source(source: DiagnosticSource, audit_json: bytes | None) -> None:
    if not isinstance(audit_json, bytes):
        raise ValueError("Gemini preparation requires canonical audit JSON bytes")
    if hashlib.sha256(audit_json).hexdigest() != source.binding.source_sha256:
        raise ValueError("canonical audit byte hash does not match source binding")
    # Reject duplicate keys and nonfinite numbers before Pydantic can normalize
    # ambiguous JSON. This decoder performs no provider requests or other I/O.
    decode_provider_json(audit_json)
    audit = AuditRun.model_validate_json(audit_json)
    if (
        audit.audit_id != source.binding.audit_id
        or _hostname(audit.site.domain) != _hostname(source.binding.domain)
        or tuple(str(page.url) for page in audit.pages) != source.page_urls
        or tuple(FrozenPrompt.model_validate(p.model_dump()) for p in audit.ai_prompts)
        != source.prompts
    ):
        raise ValueError("canonical audit does not match diagnostic source")
    validate_automatic_prompt_pack(audit)


def prepare_measurement_preflight(
    source: DiagnosticSource,
    profile: MeasurementProfile,
    *,
    audit_json: bytes | None = None,
) -> MeasurementPreflight:
    """Freeze a deterministic single-trial plan, without credentials or I/O.

    PublicURL checks raw syntax and IP literals only. Execution must still apply
    the provider network boundary, including fresh DNS validation. Canonical
    audit bytes are required for Gemini's source-bound automatic prompt gate.
    """
    _reject_unserialized_fields(source)
    _reject_unserialized_fields(profile)
    source = DiagnosticSource.model_validate_json(
        source.model_dump_json(serialize_as_any=True, warnings=False)
    )
    # JSON serialization would turn unchecked bytes into text before strict
    # field validation. Preserve Python values, while detaching nested models.
    profile = MeasurementProfile.model_validate(
        profile.model_dump(mode="python", serialize_as_any=True, warnings=False)
    )
    selected = _select_pages(source, profile.max_pages)
    if profile.gemini:
        _validate_gemini_source(source, audit_json)
    prompts = source.prompts[: profile.max_prompts]
    page_devices = len(selected) * 2
    tries = 2 if profile.retry_transient else 1
    blocked: Literal["disabled", "paid_use_not_authorized"] | None = (
        "disabled"
        if not profile.gemini
        else "paid_use_not_authorized"
        if not profile.paid_use_consent
        else None
    )
    return MeasurementPreflight(
        binding=source.binding,
        profile=profile,
        selected_pages=selected,
        prompts=prompts,
        full_prompt_count=len(source.prompts),
        full_prompt_pack_hash=(
            _pack_content_hash(source.prompts) if source.prompts else _EMPTY_PROMPT_PACK_HASH
        ),
        maximum_attempts=MeasurementAttemptBudget(
            pagespeed_insights=page_devices * tries if profile.pagespeed_insights else 0,
            crux=page_devices * tries * 2 if profile.crux else 0,
            lighthouse_local=page_devices if profile.lighthouse_local else 0,
            gemini=len(prompts) if blocked is None else 0,
        ),
        gemini_blocked_reason=blocked,
    )
