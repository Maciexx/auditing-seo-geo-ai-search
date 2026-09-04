"""Nullable actual usage and dated source-derived estimates, never invoices.

Search content is billed at model input rates. This estimate uses reported input
tokens, not a guarantee of invoice completeness. Cached input and reasoning output
are subdivisions, not extra tokens.
Only actual search actions incur the per-search fee; query strings are not calls.
"""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import HttpUrl

from ai_search_audit.diagnostic_models import FrozenDiagnosticModel, _lossless_diagnostic_values
from ai_search_audit.diagnostic_observations import (
    ObservationAttempt,
    ObservationPriceProvenance,
    ObservationUsage,
)
from ai_search_audit.observation_profile import MODEL_ID, SYSTEM_INSTRUCTIONS, ObservationProfile


def trusted_price_snapshot() -> ObservationPriceProvenance:
    return ObservationPriceProvenance(
        model_id=MODEL_ID,
        service_tier="default",
        currency="USD",
        source_urls=(
            HttpUrl("https://developers.openai.com/api/docs/pricing"),
            HttpUrl("https://developers.openai.com/api/docs/models/gpt-5.4-mini"),
        ),
        as_of=date(2026, 9, 4),
        input_per_million=Decimal("0.75"),
        cached_input_per_million=Decimal("0.075"),
        output_per_million=Decimal("4.50"),
        search_per_thousand=Decimal("10"),
    )


def price_is_current(snapshot: ObservationPriceProvenance | None, on: date) -> bool:
    return snapshot == trusted_price_snapshot() and 0 <= (on - snapshot.as_of).days <= 30


class ObservationCostEstimate(FrozenDiagnosticModel):
    attempt_id: str
    amount_usd: Decimal | None
    search_calls: int | None
    reason: Literal["unavailable_usage", "unsupported_outcome", "untrusted_pricing"] | None
    kind: Literal["source_derived_estimate_not_invoice"] = "source_derived_estimate_not_invoice"


def normalize_usage(value: object) -> ObservationUsage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid_usage")
    allowed = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
    if value.keys() - allowed:
        raise ValueError("invalid_usage")
    details: list[int | None] = []
    for key, supported in (
        ("input_tokens_details", "cached_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ):
        entry = value.get(key)
        if entry is None:
            details.append(None)
            continue
        if not isinstance(entry, dict):
            raise ValueError("invalid_usage")
        # Unsupported nonzero subdivisions must survive as an invalid_usage
        # outcome, not disappear and become free when a saved run is re-priced.
        for name, count in entry.items():
            if name != supported and (type(count) is not int or count != 0):
                raise ValueError("invalid_usage")
        details.append(entry.get(supported))
    return ObservationUsage(
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
        total_tokens=value.get("total_tokens"),
        cached_input_tokens=details[0],
        reasoning_output_tokens=details[1],
    )


def estimate_attempt(
    attempt: ObservationAttempt, snapshot: ObservationPriceProvenance | None, *, on: date
) -> ObservationCostEstimate:
    attempt = ObservationAttempt.model_validate(_lossless_diagnostic_values(attempt))
    amount = None
    calls = None
    reason: Literal["unavailable_usage", "unsupported_outcome", "untrusted_pricing"] | None
    reason = "unsupported_outcome"
    if not price_is_current(snapshot, on):
        reason = "untrusted_pricing"
    elif (
        attempt.error_category is None
        and attempt.returned_model == MODEL_ID
        and attempt.returned_service_tier == "default"
        and all(a.status == "completed" for a in attempt.search_actions)
    ):
        calls = sum(a.action == "search" for a in attempt.search_actions)
        usage = attempt.usage
        if (
            usage is None
            or usage.input_tokens is None
            or usage.output_tokens is None
            or usage.total_tokens is None
            or usage.cached_input_tokens is None
        ):
            reason = "unavailable_usage"
        else:
            # Trusted equality above binds these fixed Decimal rates.
            amount = (
                Decimal(usage.input_tokens - usage.cached_input_tokens) * Decimal("0.75")
                + Decimal(usage.cached_input_tokens) * Decimal("0.075")
                + Decimal(usage.output_tokens) * Decimal("4.50")
            ) / Decimal(1000000) + Decimal(calls) * Decimal("0.01")
            reason = None
    return ObservationCostEstimate(
        attempt_id=attempt.attempt_id, amount_usd=amount, search_calls=calls, reason=reason
    )


def next_request_reservation(profile: ObservationProfile, prompt: str) -> Decimal:
    """Reserve known bounded components, NOT a maximum provider charge.

    UTF-8 bytes plus an envelope margin conservatively allow for submitted text
    tokens. Hosted search input/cumulative tool context is not bounded here and
    may exceed this operational reservation. Use provider billing controls too.
    """
    input_allowance = len((prompt + SYSTEM_INSTRUCTIONS).encode("utf-8")) + 4096
    return (
        Decimal(input_allowance) * Decimal("0.75")
        + Decimal(profile.max_output_tokens) * Decimal("4.50")
    ) / Decimal(1000000) + Decimal(profile.max_tool_calls) * Decimal("0.01")
