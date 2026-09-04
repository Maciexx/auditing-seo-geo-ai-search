"""Explicit, bounded API observation workflow using the existing diagnostic store."""

import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from .benchmark import prepare_benchmark_worksheet, validate_benchmark_responses
from .diagnostic_models import (
    DiagnosticSource,
    FrozenDiagnosticModel,
    FrozenPrompt,
    _lossless_diagnostic_values,
)
from .diagnostic_observations import ObservationPriceProvenance
from .diagnostic_sources import load_diagnostic_source
from .diagnostic_store import DiagnosticStore
from .observation_profile import (
    MAX_REQUEST_BYTES,
    ObservationProfile,
    observation_setup,
    request_payload,
)
from .observation_usage import trusted_price_snapshot
from .openai_observations import ObservationCollection, OpenAIObservations, _safe_secret
from .performance_normalizers import decode_provider_json


def observation_scope(prompt: FrozenPrompt) -> Literal["branded", "discovery"]:
    # Legacy packs used discovery names for questions containing the target brand.
    return (
        "discovery"
        if prompt.pack_version == "2.1.0" and prompt.intent == "category_discovery"
        else "branded"
    )


class ObservationPromptScope(FrozenDiagnosticModel):
    prompt_id: str
    locale: str
    scope: Literal["branded", "discovery"]


class ObservationPreflight(FrozenDiagnosticModel):
    product: Literal["OpenAI API web search"] = "OpenAI API web search"
    selected_prompt_ids: tuple[str, ...]
    prompts: tuple[ObservationPromptScope, ...]
    model_id: str
    max_output_tokens: int
    max_tool_calls: int
    timeout_seconds: float
    max_response_bytes: int
    operational_allowance_usd: Decimal
    price_basis: ObservationPriceProvenance
    limitations: tuple[str, ...] = (
        "Estimates are not invoices; the operational allowance is not a provider spend cap.",
        "No retry after an uncertain paid outcome.",
        "OpenAI API web search is not ChatGPT consumer visibility or Google AI Overviews.",
    )


def load_observation_profile(path: Path) -> ObservationProfile:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError
            content = stream.read(65537)
        if not content or len(content) > 65536:
            raise ValueError
        return ObservationProfile.model_validate(decode_provider_json(content))
    except (OSError, ValueError, RecursionError):
        raise ValueError("invalid secret-free observation profile") from None


def prepare_observation_preflight(
    source: DiagnosticSource, profile: ObservationProfile, *, api_key: SecretStr | None = None
) -> ObservationPreflight:
    """Read-only validation before any source/profile field may be printed."""
    try:
        source = DiagnosticSource.model_validate(_lossless_diagnostic_values(source))
        profile = ObservationProfile.model_validate(_lossless_diagnostic_values(profile))
        key = api_key.get_secret_value() if api_key is not None else ""
        if _safe_secret([source.model_dump(mode="json"), profile.model_dump(mode="json")], key):
            raise ValueError
        selected = tuple(p for p in source.prompts if p.prompt_id in profile.selected_prompt_ids)
        if tuple(p.prompt_id for p in selected) != profile.selected_prompt_ids:
            raise ValueError
        worksheet = prepare_benchmark_worksheet(
            source, observation_setup(profile, source.binding.report_locale).benchmark_setup
        )
        validate_benchmark_responses(source, worksheet, [])
        for prompt in selected:
            if any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", value)
                for value in (prompt.prompt_id, prompt.locale)
            ):
                raise ValueError
            if (
                len(json.dumps(request_payload(profile, prompt.text), ensure_ascii=False).encode())
                > MAX_REQUEST_BYTES
            ):
                raise ValueError
        return ObservationPreflight(
            selected_prompt_ids=profile.selected_prompt_ids,
            prompts=tuple(
                ObservationPromptScope(
                    prompt_id=p.prompt_id, locale=p.locale, scope=observation_scope(p)
                )
                for p in selected
            ),
            model_id=profile.model_id,
            max_output_tokens=profile.max_output_tokens,
            max_tool_calls=profile.max_tool_calls,
            timeout_seconds=profile.timeout_seconds,
            max_response_bytes=profile.max_response_bytes,
            operational_allowance_usd=profile.operational_allowance_usd,
            price_basis=trusted_price_snapshot(),
        )
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise ValueError("invalid observation configuration") from None


@dataclass(frozen=True)
class ObservationWorkflowResult:
    destination: Path | None
    collection: ObservationCollection | None


def run_observations(
    project_ref: str,
    *,
    clients_root: Path,
    source_version: str,
    profile: ObservationProfile,
    on_preflight: Callable[[ObservationPreflight], None],
    paid_authorized: bool = False,
    preflight_only: bool = False,
) -> ObservationWorkflowResult:
    if type(paid_authorized) is not bool or type(preflight_only) is not bool:
        raise ValueError("invalid observation configuration")
    source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=source_version
    )
    profile = ObservationProfile.model_validate(_lossless_diagnostic_values(profile))
    value = os.environ.get("AUDIT_OPENAI_API_KEY")
    key = SecretStr(value) if value else None
    preflight = prepare_observation_preflight(source, profile, api_key=key)
    on_preflight(preflight)
    if preflight_only:
        return ObservationWorkflowResult(None, None)
    collection = OpenAIObservations().collect(
        source, profile=profile, api_key=key, paid_authorized=paid_authorized
    )
    destination = (
        DiagnosticStore(clients_root / source.binding.project_id).publish(collection.run)
        if collection.run.attempts
        else None
    )
    return ObservationWorkflowResult(destination, collection)
