"""Source-bound API observation evidence, never a performance or consumer-UI run.

Provider authenticity cannot be proved by local hashes. This contract verifies
inventory, provenance consistency and benchmark derivations without network access.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictBool, field_validator, model_validator

from ai_search_audit.benchmark import (
    _source_bound_sample,
    canonical_hash,
    compare_benchmarks,
    validate_benchmark_worksheet,
)
from ai_search_audit.diagnostic_models import (
    BenchmarkCitation,
    BenchmarkComparison,
    BenchmarkHash,
    BenchmarkSample,
    BenchmarkSetup,
    BenchmarkWorksheet,
    DiagnosticBinding,
    DiagnosticCollectionRange,
    DiagnosticFile,
    DiagnosticSource,
    FrozenDiagnosticModel,
    _lossless_diagnostic_values,
)
from ai_search_audit.diagnostic_performance import ProviderRunCleanup
from ai_search_audit.models import DataState

_Identifier = Annotated[
    str, Field(strict=True, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
]
_Count = Annotated[int, Field(strict=True, ge=0)]
_Rate = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class ObservationSetup(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    benchmark_setup: BenchmarkSetup
    system_instruction_sha256: BenchmarkHash
    effective_request_policy_sha256: BenchmarkHash
    authentication: Literal["api_key"]

    @model_validator(mode="after")
    def separate_api_authentication(self) -> Self:
        if self.benchmark_setup.account_state is not None:
            raise ValueError("API authentication does not establish a consumer account state")
        return self


class ObservationUsage(FrozenDiagnosticModel):
    input_tokens: _Count | None
    output_tokens: _Count | None
    total_tokens: _Count | None
    cached_input_tokens: _Count | None
    reasoning_output_tokens: _Count | None

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("usage total must exclude double-counted subdivisions")
        known_lower_bound = 0
        for child, parent in (
            (self.cached_input_tokens, self.input_tokens),
            (self.reasoning_output_tokens, self.output_tokens),
        ):
            if child is not None and parent is not None and child > parent:
                raise ValueError("usage subdivision exceeds its parent")
            known_lower_bound += parent if parent is not None else (child or 0)
        if self.total_tokens is not None and self.total_tokens < known_lower_bound:
            raise ValueError("usage total contradicts its known token lower bound")
        return self


class ObservationPriceProvenance(FrozenDiagnosticModel):
    """Explicit dated snapshot only. Cost calculation and trust policy live in Task 3."""

    model_id: _Identifier
    service_tier: _Identifier
    currency: Literal["USD"]
    source_urls: tuple[BenchmarkCitation, ...] = Field(min_length=1, max_length=8)
    as_of: date
    input_per_million: _Rate | None
    cached_input_per_million: _Rate | None
    output_per_million: _Rate | None
    search_per_thousand: _Rate | None


class ObservationSearchAction(FrozenDiagnosticModel):
    action: Literal["search", "open_page", "find_in_page"]
    status: Literal["completed", "incomplete", "failed"]
    call_id: _Identifier | None = None
    consulted_sources: tuple[BenchmarkCitation, ...] | None = Field(default=None, max_length=100)


class ObservationCitationAnnotation(FrozenDiagnosticModel):
    """Unicode code-point offsets in captured visible text, not retained-excerpt offsets."""

    url: BenchmarkCitation
    start_index: _Count
    end_index: _Count

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.start_index >= self.end_index:
            raise ValueError("citation annotation needs a nonempty forward span")
        return self


class ObservationAttempt(FrozenDiagnosticModel):
    attempt_id: _Identifier
    prompt_id: _Identifier
    locale: _Identifier
    started_at: AwareDatetime
    ended_at: AwareDatetime
    requested_model: _Identifier
    returned_model: _Identifier | None
    requested_service_tier: _Identifier
    returned_service_tier: _Identifier | None
    status: Literal["completed", "incomplete", "refused", "no_text", "failed"]
    complete: StrictBool
    error_category: (
        Literal[
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
        | None
    )
    request_id: _Identifier | None
    search_actions: tuple[ObservationSearchAction, ...] = Field(max_length=32)
    usage: ObservationUsage | None
    response_hash: BenchmarkHash | None
    citation_annotations: tuple[ObservationCitationAnnotation, ...] | None = Field(max_length=200)
    citation_metadata_complete: StrictBool

    @field_validator("started_at", "ended_at", mode="before")
    @classmethod
    def explicit_timestamp(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError("attempt timestamp requires an explicit aware datetime")
        return value

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.started_at > self.ended_at:
            raise ValueError("attempt chronology mismatch")
        success = self.status == "completed"
        if self.complete != success or (self.response_hash is not None) != success:
            raise ValueError("only completed text-bearing attempts may bind responses")
        if success and (
            self.returned_model != self.requested_model
            or self.returned_service_tier != self.requested_service_tier
        ):
            raise ValueError("completed attempt model or service tier mismatch")
        if (self.status == "failed") != (self.error_category is not None):
            raise ValueError("failed attempts require a safe error category")
        if self.citation_metadata_complete and self.citation_annotations is None:
            raise ValueError("complete citation metadata cannot be absent")
        if not success and (self.citation_annotations or self.citation_metadata_complete):
            raise ValueError("citation annotations require a retained response")
        ids = [item.call_id for item in self.search_actions if item.call_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate received search call IDs")
        return self


def observation_algorithm_hash() -> str:
    return canonical_hash(
        {"pipeline": "api-observation-1.0.0", "algorithm": "1.0.0", "benchmark": "1.0.0"}
    )


def observation_collection_range(
    attempts: tuple[ObservationAttempt, ...],
) -> DiagnosticCollectionRange:
    stamps = [
        stamp.astimezone(UTC) for item in attempts for stamp in (item.started_at, item.ended_at)
    ]
    return DiagnosticCollectionRange(
        start=min(stamps) if stamps else None,
        end=max(stamps) if stamps else None,
        reason=None if stamps else "No API requests were attempted.",
    )


def observation_input_hash(
    worksheet: BenchmarkWorksheet,
    setup: ObservationSetup,
    selected_prompt_ids: tuple[str, ...],
    sample: BenchmarkSample,
    attempts: tuple[ObservationAttempt, ...],
    price_provenance: ObservationPriceProvenance | None,
) -> str:
    return canonical_hash(
        {
            "worksheet": worksheet.model_dump(mode="json"),
            "setup": setup.model_dump(mode="json"),
            "selected_prompt_ids": selected_prompt_ids,
            "sample": sample.model_dump(mode="json"),
            "attempts": [item.model_dump(mode="json") for item in attempts],
            "price_provenance": price_provenance.model_dump(mode="json")
            if price_provenance
            else None,
        }
    )


class DiagnosticObservationRun(FrozenDiagnosticModel):
    schema_version: Literal["3.0.0"] = "3.0.0"
    binding: DiagnosticBinding
    pipeline_version: Literal["api-observation-1.0.0"] = "api-observation-1.0.0"
    algorithm_version: Literal["1.0.0"] = "1.0.0"
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    collection_range: DiagnosticCollectionRange
    worksheet: BenchmarkWorksheet
    setup: ObservationSetup
    selected_prompt_ids: tuple[_Identifier, ...] = Field(min_length=1, max_length=12)
    sample: BenchmarkSample
    attempts: tuple[ObservationAttempt, ...] = Field(max_length=12)
    price_provenance: ObservationPriceProvenance | None
    cleanup: ProviderRunCleanup

    @model_validator(mode="before")
    @classmethod
    def lossless_values(cls, value: object) -> object:
        return _lossless_diagnostic_values(value)

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        if self.binding != self.worksheet.binding or self.sample.worksheet != self.worksheet:
            raise ValueError("observation worksheet binding mismatch")
        if self.setup.benchmark_setup != self.worksheet.setup:
            raise ValueError("observation setup snapshot mismatch")
        if self.setup.benchmark_setup.interface != "api":
            raise ValueError("observation run requires API interface")
        if self.algorithm_sha256 != observation_algorithm_hash():
            raise ValueError("observation algorithm fingerprint mismatch")
        if self.collection_range != observation_collection_range(self.attempts):
            raise ValueError("observation collection range mismatch")
        _validate_observation_inventory(self)
        if self.input_sha256 != observation_input_hash(
            self.worksheet,
            self.setup,
            self.selected_prompt_ids,
            self.sample,
            self.attempts,
            self.price_provenance,
        ):
            raise ValueError("observation input hash mismatch")
        return self


def _validate_observation_inventory(run: DiagnosticObservationRun) -> None:
    selected = run.selected_prompt_ids
    prompts = {p.prompt_id: p for p in run.worksheet.prompts}
    if selected != tuple(p for p in prompts if p in selected):
        raise ValueError("selected prompts must be unique and in canonical order")
    attempt_ids = tuple(a.attempt_id for a in run.attempts)
    prompt_ids = tuple(a.prompt_id for a in run.attempts)
    if len(set(attempt_ids)) != len(attempt_ids) or prompt_ids != tuple(
        p for p in selected if p in prompt_ids
    ):
        raise ValueError("attempt inventory requires unique selected canonical prompts")
    responses = {r.prompt_id: r for r in run.sample.responses}
    if set(responses) != {a.prompt_id for a in run.attempts if a.status == "completed"}:
        raise ValueError("sample response and completed attempt inventory mismatch")
    known_call_ids = [
        action.call_id
        for a in run.attempts
        for action in a.search_actions
        if action.call_id is not None
    ]
    if len(known_call_ids) != len(set(known_call_ids)):
        raise ValueError("duplicate received search call IDs")
    for attempt in run.attempts:
        if (
            attempt.locale != prompts[attempt.prompt_id].locale
            or attempt.requested_model != run.setup.benchmark_setup.model_id
        ):
            raise ValueError("attempt prompt locale or requested model mismatch")
        response = responses.get(attempt.prompt_id)
        if response is None:
            continue
        if (
            not response.complete
            or response.response_hash != attempt.response_hash
            or not attempt.started_at <= response.observed_at <= attempt.ended_at
        ):
            raise ValueError("attempt response hash, completion or timestamp mismatch")
        if response.grounded is True and not any(
            a.action == "search" and a.status == "completed" for a in attempt.search_actions
        ):
            raise ValueError("grounding requires an actual completed search action")
        annotations = attempt.citation_annotations
        if annotations is not None:
            if any(a.end_index > response.captured_text_length for a in annotations):
                raise ValueError("citation annotation is outside captured text")
            urls, citations = {a.url for a in annotations}, set(response.citations)
            if not urls <= citations or (attempt.citation_metadata_complete and urls != citations):
                raise ValueError("citation annotation URL inventory mismatch")
        if response.citations_complete and not attempt.citation_metadata_complete:
            raise ValueError("complete citations require complete annotation metadata")


def validate_observation_source(
    run: DiagnosticObservationRun, source: DiagnosticSource
) -> DiagnosticObservationRun:
    validated = DiagnosticObservationRun.model_validate(_lossless_diagnostic_values(run))
    if validated.binding != source.binding:
        raise ValueError("observation run binding does not match canonical source")
    validate_benchmark_worksheet(source, validated.worksheet)
    _source_bound_sample(source, validated.sample)
    return validated


def assemble_observation_run(
    source: DiagnosticSource,
    worksheet: BenchmarkWorksheet,
    setup: ObservationSetup,
    selected_prompt_ids: tuple[str, ...],
    sample: BenchmarkSample,
    attempts: tuple[ObservationAttempt, ...],
    *,
    price_provenance: ObservationPriceProvenance | None,
) -> DiagnosticObservationRun:
    """Build only from validated bounded evidence, never trust caller cached metrics."""
    worksheet = validate_benchmark_worksheet(source, worksheet)
    setup = ObservationSetup.model_validate(_lossless_diagnostic_values(setup))
    sample = _source_bound_sample(source, sample)
    attempts = tuple(
        ObservationAttempt.model_validate(_lossless_diagnostic_values(a)) for a in attempts
    )
    if price_provenance is not None:
        price_provenance = ObservationPriceProvenance.model_validate(
            _lossless_diagnostic_values(price_provenance)
        )
    run = DiagnosticObservationRun(
        binding=source.binding,
        worksheet=worksheet,
        setup=setup,
        selected_prompt_ids=selected_prompt_ids,
        sample=sample,
        attempts=attempts,
        price_provenance=price_provenance,
        cleanup=ProviderRunCleanup(),
        input_sha256=observation_input_hash(
            worksheet, setup, selected_prompt_ids, sample, attempts, price_provenance
        ),
        algorithm_sha256=observation_algorithm_hash(),
        collection_range=observation_collection_range(attempts),
    )
    return validate_observation_source(run, source)


class ObservationComparison(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    comparison: BenchmarkComparison


class DiagnosticObservationManifest(FrozenDiagnosticModel):
    schema_version: Literal["3.0.0"] = "3.0.0"
    binding: DiagnosticBinding
    run_number: Annotated[int, Field(strict=True, ge=1)]
    files: tuple[DiagnosticFile, DiagnosticFile]
    input_sha256: BenchmarkHash
    algorithm_sha256: BenchmarkHash
    cleanup: ProviderRunCleanup

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if tuple(item.filename for item in self.files) != ("diagnostics.json", "evidence.jsonl"):
            raise ValueError("diagnostic manifest inventory mismatch")
        return self


class LoadedDiagnosticObservationRun(FrozenDiagnosticModel):
    run: DiagnosticObservationRun
    manifest: DiagnosticObservationManifest
    manifest_sha256: BenchmarkHash


def compare_observations(
    baseline: DiagnosticObservationRun,
    follow_up: DiagnosticObservationRun,
    *,
    baseline_source: DiagnosticSource,
    follow_up_source: DiagnosticSource,
) -> ObservationComparison:
    before = validate_observation_source(baseline, baseline_source)
    after = validate_observation_source(follow_up, follow_up_source)
    comparison = compare_benchmarks(
        before.sample,
        after.sample,
        baseline_source=baseline_source,
        follow_up_source=follow_up_source,
    )
    reasons = []
    if before.setup != after.setup:
        reasons.append("Observation setup differs.")
    if before.selected_prompt_ids != after.selected_prompt_ids:
        reasons.append("Selected observation prompt IDs differ.")
    if (
        not before.sample.measurable_prompt_ids
        or before.sample.measurable_prompt_ids != after.sample.measurable_prompt_ids
    ):
        reasons.append("Eligible observation sample subsets are unavailable or differ.")
    actual = [
        tuple(
            (a.prompt_id, a.returned_model, a.returned_service_tier)
            for a in run.attempts
            if a.status == "completed"
        )
        for run in (before, after)
    ]
    if not actual[0] or actual[0] != actual[1]:
        reasons.append("Actual observation models or service tiers are unavailable or differ.")
    if reasons:
        values = comparison.model_dump(mode="python")
        values.update(
            state=DataState.UNKNOWN,
            mention_comparison=None,
            mention_delta=None,
            citation_delta=None,
            limitations=(*comparison.limitations, *reasons),
        )
        comparison = BenchmarkComparison.model_validate(values)
    return ObservationComparison(comparison=comparison)
