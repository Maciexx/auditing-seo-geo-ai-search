from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from ai_search_audit import comparisons
from ai_search_audit.diagnostic_models import (
    MAX_BENCHMARK_EXCERPT_CHARS,
    BenchmarkComparison,
    BenchmarkMetrics,
    BenchmarkResponse,
    BenchmarkResponseInput,
    BenchmarkSample,
    BenchmarkSetup,
    BenchmarkWorksheet,
    DiagnosticSource,
    FrozenPrompt,
    _lossless_diagnostic_values,
    _normalize_citation_host,
)
from ai_search_audit.models import AIObservation, DataState

_INSTRUCTIONS_PL = (
    "Użyj dokładnego tekstu każdego promptu. Nie tłumacz go ani nie zmieniaj.",
    "Zapisz znane ustawienia narzędzia; nieznane wartości pozostaw puste (null). "
    "Nie zapisuj danych konta ani danych logowania.",
    "Dla każdego promptu zapisz datę, odpowiedź i adresy cytowanych źródeł. "
    "Brak odpowiedzi oznacza brak testu, nie wynik negatywny.",
    "Nie zakładaj, że dostęp przez przeglądarkę oznacza użycie wyszukiwania. "
    "Niepełne ustawienia nie pozwalają na kontrolowane porównanie liczbowe.",
)
_INSTRUCTIONS_EN = (
    "Use the exact text of every prompt. Do not translate or change it.",
    "Record known tool settings; leave unknown values empty (null). "
    "Do not record account identifiers or credentials.",
    "For each prompt, record the date, response and cited source URLs. "
    "A missing response means untested, not a negative result.",
    "Browser access does not establish that search was used. "
    "Incomplete settings do not allow a controlled numerical comparison.",
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sample_metrics(
    *, expected: int, mentioned: tuple[bool, ...], cited: tuple[bool | None, ...]
) -> BenchmarkMetrics:
    """Measure eligible observations, never treating an omitted prompt as a negative."""
    if (
        type(expected) is not int
        or expected < 1
        or len(mentioned) != len(cited)
        or len(mentioned) > expected
        or any(type(value) is not bool for value in mentioned)
        or any(value is not None and type(value) is not bool for value in cited)
    ):
        raise ValueError("invalid benchmark denominator or flags")
    count = len(mentioned)
    if not count:
        return BenchmarkMetrics(
            state=DataState.UNAVAILABLE,
            expected=expected,
            measured=0,
            citation_measured=0,
            coverage=0,
            citation_coverage=0,
            confidence=0,
            limitations=("No grounded completed responses.",),
        )
    known_citations = tuple(value for value in cited if value is not None)
    citation_count = len(known_citations)
    coverage = count / expected
    return BenchmarkMetrics(
        state=DataState.AVAILABLE if count == citation_count == expected else DataState.PARTIAL,
        expected=expected,
        measured=count,
        citation_measured=citation_count,
        coverage=coverage,
        citation_coverage=citation_count / expected,
        confidence=coverage,
        mention_rate=100 * sum(mentioned) / count,
        citation_rate=100 * sum(known_citations) / citation_count if citation_count else None,
        limitations=("Confidence describes this observed sample only.",),
    )


def benchmark_setup_fingerprint(setup: BenchmarkSetup | None) -> str | None:
    """Hash validated setup dimensions only when every critical value is known."""
    if setup is None:
        return None
    validated = BenchmarkSetup.model_validate(_lossless_diagnostic_values(setup))
    fields = validated.model_dump(mode="json")
    if any(value is None for value in fields.values()):
        return None
    return canonical_hash(fields)


def _pack_version(prompts: tuple[FrozenPrompt, ...]) -> str:
    if not prompts:
        raise ValueError("benchmark prompt pack cannot be empty")
    if any(
        not value.strip()
        for prompt in prompts
        for value in (
            prompt.prompt_id,
            prompt.pack_version,
            prompt.locale,
            prompt.intent,
            prompt.text,
        )
    ):
        raise ValueError("required prompt fields cannot be blank")
    if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
        raise ValueError("duplicate prompt IDs in benchmark pack")
    versions = {prompt.pack_version for prompt in prompts}
    if len(versions) != 1:
        raise ValueError("mixed prompt pack versions")
    return prompts[0].pack_version


def _pack_content_hash(prompts: tuple[FrozenPrompt, ...]) -> str:
    return canonical_hash(
        {
            "pack_version": _pack_version(prompts),
            "prompts": [prompt.model_dump(mode="json") for prompt in prompts],
        }
    )


def _validate_worksheet_integrity(worksheet: BenchmarkWorksheet) -> None:
    if worksheet.pack_version != _pack_version(worksheet.prompts):
        raise ValueError("worksheet pack version does not match its prompts")
    if worksheet.pack_content_hash != _pack_content_hash(worksheet.prompts):
        raise ValueError("worksheet prompt pack content hash mismatch")
    if worksheet.setup_fingerprint != benchmark_setup_fingerprint(worksheet.setup):
        raise ValueError("worksheet setup fingerprint mismatch")
    instructions = _INSTRUCTIONS_PL if worksheet.binding.report_locale == "pl" else _INSTRUCTIONS_EN
    if worksheet.instructions != instructions:
        raise ValueError("worksheet instructions do not match report locale")


def prepare_benchmark_worksheet(
    source: DiagnosticSource, setup: BenchmarkSetup | None = None
) -> BenchmarkWorksheet:
    """Freeze canonical inputs without generating prompts, responses or network calls."""
    validated = DiagnosticSource.model_validate(_lossless_diagnostic_values(source))
    return BenchmarkWorksheet(
        binding=validated.binding,
        prompts=validated.prompts,
        pack_version=_pack_version(validated.prompts),
        pack_content_hash=_pack_content_hash(validated.prompts),
        setup=setup,
        setup_fingerprint=benchmark_setup_fingerprint(setup),
        instructions=_INSTRUCTIONS_PL
        if validated.binding.report_locale == "pl"
        else _INSTRUCTIONS_EN,
    )


def validate_benchmark_worksheet(source: DiagnosticSource, worksheet: object) -> BenchmarkWorksheet:
    """Accept imported inputs only against the trusted canonical source, not a hash alone."""
    validated = BenchmarkWorksheet.model_validate(_lossless_diagnostic_values(worksheet))
    canonical = prepare_benchmark_worksheet(source, validated.setup)
    if validated.binding != canonical.binding:
        raise ValueError("worksheet binding does not match canonical source")
    if validated.prompts != canonical.prompts:
        raise ValueError("worksheet prompts do not match canonical source")
    return validated


def _mention_eligible(response: BenchmarkResponse) -> bool:
    return (
        response.complete
        and response.grounded is True
        and response.brand_mentioned is not None
        and (response.brand_mentioned or response.inspection_scope == "full_response")
    )


def _citation_state(response: BenchmarkResponse, approved_domains: tuple[str, ...]) -> bool | None:
    if any(
        url.host is not None and _normalize_citation_host(url.host) in approved_domains
        for url in response.citations
    ):
        return True
    return False if response.citations_complete else None


def _approved_domains(source: DiagnosticSource) -> tuple[str, ...]:
    return tuple(sorted({_normalize_citation_host(domain) for domain in source.canonical_domains}))


def _response_metrics(
    worksheet: BenchmarkWorksheet,
    responses: tuple[BenchmarkResponse, ...],
    approved_domains: tuple[str, ...],
) -> BenchmarkMetrics:
    eligible = tuple(item for item in responses if _mention_eligible(item))
    return sample_metrics(
        expected=len(worksheet.prompts),
        mentioned=tuple(item.brand_mentioned is True for item in eligible),
        cited=tuple(_citation_state(item, approved_domains) for item in eligible),
    )


def _validate_sample_integrity(sample: BenchmarkSample) -> None:
    prompts = {prompt.prompt_id: prompt.text for prompt in sample.worksheet.prompts}
    if len({item.prompt_id for item in sample.responses}) != len(sample.responses):
        raise ValueError("duplicate benchmark response prompt IDs")
    for item in sample.responses:
        if prompts.get(item.prompt_id) != item.prompt_text:
            raise ValueError("response prompt does not match exact worksheet prompt")
    if sample.approved_domains != tuple(
        sorted({_normalize_citation_host(domain) for domain in sample.approved_domains})
    ):
        raise ValueError("approved citation domains must be normalized and unique")
    if sample.metrics != _response_metrics(
        sample.worksheet, sample.responses, sample.approved_domains
    ):
        raise ValueError("sample metrics do not match validated responses")


def validate_benchmark_responses(
    source: DiagnosticSource, worksheet: object, responses: Sequence[object]
) -> BenchmarkSample:
    """Bind actual captured responses to canonical prompts and derive bounded evidence.

    No citation is fetched. Explicit inspection metadata records the reviewer's scope;
    the processor never infers full inspection merely because it has a complete capture.
    """
    validated_worksheet = validate_benchmark_worksheet(source, worksheet)
    observations: list[BenchmarkResponse] = []
    for supplied in responses:
        response = BenchmarkResponseInput.model_validate(_lossless_diagnostic_values(supplied))
        digest = hashlib.sha256(response.response_text.encode("utf-8")).hexdigest()
        if response.response_hash is not None and response.response_hash != digest:
            raise ValueError("captured response hash mismatch")
        excerpt = response.response_text[:MAX_BENCHMARK_EXCERPT_CHARS]
        observations.append(
            BenchmarkResponse(
                **response.model_dump(exclude={"response_text", "response_hash"}),
                response_excerpt=excerpt,
                response_hash=digest,
                source_response_hash=None if response.response_truncated else digest,
                excerpt_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                captured_text_length=len(response.response_text),
                excerpt_truncated=response.response_truncated
                or len(excerpt) < len(response.response_text),
            )
        )
    frozen = tuple(sorted(observations, key=lambda item: item.prompt_id))
    domains = _approved_domains(source)
    return BenchmarkSample(
        worksheet=validated_worksheet,
        approved_domains=domains,
        responses=frozen,
        metrics=_response_metrics(validated_worksheet, frozen, domains),
    )


def _source_bound_sample(source: DiagnosticSource, sample: BenchmarkSample) -> BenchmarkSample:
    validated = BenchmarkSample.model_validate(_lossless_diagnostic_values(sample))
    validate_benchmark_worksheet(source, validated.worksheet)
    if validated.approved_domains != _approved_domains(source):
        raise ValueError("sample approved citation domains do not match canonical source")
    return validated


def _legacy_observations(sample: BenchmarkSample) -> tuple[AIObservation, ...]:
    """Transient adapter only; never place mutable legacy observations in an output DTO."""
    setup = sample.worksheet.setup
    # The caller first requires the fully known setup fingerprint.
    if setup is None or setup.provider is None:
        raise ValueError("legacy adapter requires a known benchmark provider")
    return tuple(
        AIObservation(
            observation_id=canonical_hash(
                {
                    "prompt_id": item.prompt_id,
                    "response_hash": item.response_hash,
                    "observed_at": item.observed_at.isoformat(),
                }
            ),
            prompt_id=item.prompt_id,
            provider=setup.provider,
            observed_at=item.observed_at,
            response_excerpt=item.response_excerpt,
            response_fingerprint=item.response_hash,
            citations=[str(url) for url in item.citations],
            brand_mentioned=item.brand_mentioned if _mention_eligible(item) else None,
            grounded=item.grounded,
        )
        for item in sample.responses
    )


def compare_benchmarks(
    baseline: BenchmarkSample,
    follow_up: BenchmarkSample,
    *,
    baseline_source: DiagnosticSource,
    follow_up_source: DiagnosticSource,
) -> BenchmarkComparison:
    """Compare trusted processed samples, rechecking bindings and every cached metric.

    Caller-owned canonical sources may be different versions of the same project. New
    provider data must go through validate_benchmark_responses, not this output format.
    """
    before = _source_bound_sample(baseline_source, baseline)
    after = _source_bound_sample(follow_up_source, follow_up)
    left, right = before.worksheet, after.worksheet
    limitations: list[str] = []
    if left.pack_content_hash != right.pack_content_hash:
        limitations.append("Full prompt pack content differs.")
    before_setup = benchmark_setup_fingerprint(left.setup)
    after_setup = benchmark_setup_fingerprint(right.setup)
    if before_setup is None or before_setup != after_setup:
        limitations.append("Benchmark setup is unknown or differs.")
    if (left.binding.project_id, left.binding.domain, left.binding.report_locale) != (
        right.binding.project_id,
        right.binding.domain,
        right.binding.report_locale,
    ):
        limitations.append("Benchmark project/domain/report-locale binding differs.")
    if (
        before.responses
        and after.responses
        and max(item.observed_at for item in before.responses)
        >= min(item.observed_at for item in after.responses)
    ):
        limitations.append("Benchmark observations are not in chronological run order.")
    if limitations:
        return BenchmarkComparison(
            state=DataState.UNKNOWN,
            baseline_metrics=before.metrics,
            follow_up_metrics=after.metrics,
            limitations=tuple(limitations),
            baseline_citation_prompt_ids=before.citation_measurable_prompt_ids,
            follow_up_citation_prompt_ids=after.citation_measurable_prompt_ids,
            baseline_approved_domains=before.approved_domains,
            follow_up_approved_domains=after.approved_domains,
        )
    legacy = comparisons.compare_observed_ai_visibility(
        _legacy_observations(before),
        _legacy_observations(after),
        baseline_prompt_pack_version=left.pack_version,
        follow_up_prompt_pack_version=right.pack_version,
        baseline_setup_fingerprint=before_setup,
        follow_up_setup_fingerprint=after_setup,
        baseline_canonical_prompt_ids=tuple(prompt.prompt_id for prompt in left.prompts),
        follow_up_canonical_prompt_ids=tuple(prompt.prompt_id for prompt in right.prompts),
    )
    if legacy.state not in {DataState.AVAILABLE, DataState.PARTIAL}:
        return BenchmarkComparison(
            state=DataState.UNKNOWN,
            baseline_metrics=before.metrics,
            follow_up_metrics=after.metrics,
            mention_comparison=legacy,
            limitations=tuple(str(reason) for reason in legacy.limitations),
            baseline_citation_prompt_ids=before.citation_measurable_prompt_ids,
            follow_up_citation_prompt_ids=after.citation_measurable_prompt_ids,
            baseline_approved_domains=before.approved_domains,
            follow_up_approved_domains=after.approved_domains,
        )
    citation_delta = None
    if (
        not before.citation_measurable_prompt_ids
        or before.citation_measurable_prompt_ids != after.citation_measurable_prompt_ids
    ):
        limitations.append("Citation-measurable prompt subsets are unavailable or differ.")
    elif before.approved_domains != after.approved_domains:
        limitations.append("Approved citation domains differ.")
    else:
        before_rate, after_rate = before.metrics.citation_rate, after.metrics.citation_rate
        if before_rate is not None and after_rate is not None:
            citation_delta = after_rate - before_rate
    complete = (
        before.metrics.state is after.metrics.state is DataState.AVAILABLE
        and legacy.state is DataState.AVAILABLE
        and citation_delta is not None
    )
    return BenchmarkComparison(
        state=DataState.AVAILABLE if complete else DataState.PARTIAL,
        baseline_metrics=before.metrics,
        follow_up_metrics=after.metrics,
        mention_comparison=legacy,
        mention_delta=legacy.absolute_delta,
        citation_delta=citation_delta,
        limitations=tuple(limitations),
        baseline_citation_prompt_ids=before.citation_measurable_prompt_ids,
        follow_up_citation_prompt_ids=after.citation_measurable_prompt_ids,
        baseline_approved_domains=before.approved_domains,
        follow_up_approved_domains=after.approved_domains,
    )
