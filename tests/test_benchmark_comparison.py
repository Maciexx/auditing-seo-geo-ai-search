import json
import socket

import pytest
from pydantic import ValidationError

from ai_search_audit.diagnostic_models import DiagnosticSource
from tests.test_benchmark import _response, _sample, _setup, _source


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("benchmark comparison must not perform network requests")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def _pair(
    source=None, follow_source=None, *, before=None, after=None, setup=None, follow_setup=None
):
    source = source or _source()
    follow_source = follow_source or source
    setup = setup or _setup()
    follow_setup = follow_setup or setup
    if before is None:
        before = tuple(_response(source, index, brand_mentioned=False) for index in range(2))
    if after is None:
        after = tuple(
            _response(
                follow_source,
                index,
                observed_at="2026-09-04T12:00:00Z",
                citations=(f"https://{follow_source.binding.domain}/source",),
            )
            for index in range(2)
        )
    return _sample(source, before, setup), _sample(follow_source, after, follow_setup)


def _compare(baseline, follow_up, source=None, follow_source=None):
    from ai_search_audit.benchmark import compare_benchmarks

    source = source or _source()
    return compare_benchmarks(
        baseline,
        follow_up,
        baseline_source=source,
        follow_up_source=follow_source or source,
    )


@pytest.mark.parametrize("locale", ["pl", "en"])
@pytest.mark.parametrize("business", ["studio", "shop"])
def test_comparable_full_samples_report_distinct_rates_and_noncausal_deltas(locale, business):
    source = _source(locale, business)
    baseline, follow_up = _pair(source)
    result = _compare(baseline, follow_up, source)
    assert result.state == "AVAILABLE"
    assert result.mention_delta == 100.0
    assert result.citation_delta == 100.0
    assert result.baseline_metrics == baseline.metrics
    assert result.follow_up_metrics == follow_up.metrics
    assert result.causality == "NOT_ESTABLISHED"
    assert (
        result.chronology_statement == "A difference between samples does not establish causality."
    )


def test_comparison_calls_legacy_comparator_with_real_observations_and_full_canonical_ids(
    monkeypatch,
):
    from ai_search_audit import comparisons
    from ai_search_audit.benchmark import benchmark_setup_fingerprint

    original = comparisons.compare_observed_ai_visibility
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(comparisons, "compare_observed_ai_visibility", record)
    baseline, follow_up = _pair(
        before=(_response(brand_mentioned=False),),
        after=(_response(observed_at="2026-09-04T12:00:00Z", citations_complete=False),),
    )
    result = _compare(baseline, follow_up)
    assert len(calls) == 1
    args, kwargs, legacy_result = calls[0]
    assert tuple(item.prompt_id for item in args[0]) == ("p-pl",)
    assert tuple(item.prompt_id for item in args[1]) == ("p-pl",)
    assert kwargs["baseline_canonical_prompt_ids"] == ("p-pl", "p-en")
    assert kwargs["follow_up_canonical_prompt_ids"] == ("p-pl", "p-en")
    assert kwargs["baseline_setup_fingerprint"] == benchmark_setup_fingerprint(_setup())
    assert result.mention_comparison == legacy_result
    assert result.mention_delta == legacy_result.absolute_delta == 100.0
    assert result.state == "PARTIAL"
    assert result.citation_delta is None
    assert result.baseline_metrics.coverage == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", None),
        ("provider", None),
        ("interface", "api"),
        ("locale", "pl"),
        ("search_mode", "disabled"),
        ("market", "GB"),
    ],
)
def test_unknown_or_changed_critical_setup_never_produces_a_numeric_comparison(field, value):
    baseline, follow_up = _pair(follow_setup=_setup(**{field: value}))
    result = _compare(baseline, follow_up)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None
    assert any("setup" in reason.lower() for reason in result.limitations)
    assert result.baseline_metrics.mention_rate == 0.0
    assert result.follow_up_metrics.mention_rate == 100.0


def test_identical_unknown_setup_does_not_become_comparable():
    baseline, follow_up = _pair(setup=_setup(model_id=None))
    result = _compare(baseline, follow_up)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None


def test_same_version_different_full_pack_content_blocks_comparison():
    changed = _source().model_dump(mode="json")
    changed["prompts"][1]["text"] = "Different canonical wording?"
    changed_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(
        follow_source=changed_source,
        before=(_response(),),
        after=(_response(changed_source, observed_at="2026-09-04T12:00:00Z"),),
    )
    assert baseline.worksheet.pack_version == follow_up.worksheet.pack_version
    result = _compare(baseline, follow_up, follow_source=changed_source)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None
    assert any("pack" in reason.lower() for reason in result.limitations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "other"),
        ("domain", "other.example"),
        ("report_locale", "en"),
    ],
)
def test_cross_binding_comparison_is_noncomparable_even_for_same_pack_and_setup(field, value):
    changed = _source().model_dump(mode="json")
    changed["binding"][field] = value
    changed_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(follow_source=changed_source)
    result = _compare(baseline, follow_up, follow_source=changed_source)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None
    assert any("binding" in reason.lower() for reason in result.limitations)


def test_explicit_different_source_versions_are_allowed_with_same_canonical_pack_and_setup():
    changed = _source().model_dump(mode="json")
    changed["binding"].update(
        source_version="public-v2", audit_id="audit-2", source_sha256="b" * 64
    )
    follow_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(follow_source=follow_source)
    assert _compare(baseline, follow_up, follow_source=follow_source).mention_delta == 100.0


def test_unequal_measured_prompt_subsets_reuse_legacy_rejection():
    baseline, follow_up = _pair(
        after=(
            _response(prompt_index=0, observed_at="2026-09-04T12:00:00Z"),
            _response(prompt_index=1, observed_at="2026-09-04T12:00:00Z", grounded=False),
        )
    )
    result = _compare(baseline, follow_up)
    assert result.mention_comparison.state == "UNKNOWN"
    assert "PROMPT_SET_MISMATCH" in result.mention_comparison.limitations
    assert result.mention_delta is result.citation_delta is None


def test_same_count_but_different_observed_prompt_ids_cannot_compare():
    baseline, follow_up = _pair(
        before=(_response(prompt_index=0),),
        after=(_response(prompt_index=1, observed_at="2026-09-04T12:00:00Z"),),
    )
    result = _compare(baseline, follow_up)
    assert result.mention_comparison.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None


def test_incomplete_response_is_not_upgraded_by_legacy_adapter():
    baseline, follow_up = _pair(
        before=(_response(complete=False),),
        after=(_response(complete=False, observed_at="2026-09-04T12:00:00Z"),),
    )
    result = _compare(baseline, follow_up)
    assert result.mention_comparison.baseline_prompt_ids == ("p-pl",)
    assert result.mention_comparison.baseline_measurable_prompt_ids == ()
    assert result.mention_delta is result.citation_delta is None


def test_both_missing_samples_do_not_compare_as_zero():
    baseline, follow_up = _pair(before=(), after=())
    result = _compare(baseline, follow_up)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None
    assert "Observed change" not in result.chronology_statement


def test_citation_delta_uses_its_own_matching_measured_subset_not_mention_denominator():
    baseline, follow_up = _pair(
        before=(
            _response(prompt_index=0, brand_mentioned=False),
            _response(prompt_index=1, brand_mentioned=False, citations_complete=False),
        ),
        after=(
            _response(
                prompt_index=0,
                observed_at="2026-09-04T12:00:00Z",
                citations=("https://studio.example/",),
            ),
            _response(prompt_index=1, observed_at="2026-09-04T12:00:00Z", citations_complete=False),
        ),
    )
    result = _compare(baseline, follow_up)
    assert result.state == "PARTIAL"
    assert result.mention_delta == result.citation_delta == 100.0
    assert result.baseline_metrics.measured == 2
    assert result.baseline_metrics.citation_measured == 1
    assert result.baseline_metrics.citation_coverage == 0.5


def test_same_citation_count_different_prompt_ids_blocks_only_citation_delta():
    baseline, follow_up = _pair(
        before=(
            _response(prompt_index=0),
            _response(prompt_index=1, citations_complete=False),
        ),
        after=(
            _response(prompt_index=0, observed_at="2026-09-04T12:00:00Z", citations_complete=False),
            _response(
                prompt_index=1,
                observed_at="2026-09-04T12:00:00Z",
                citations=("https://studio.example/",),
            ),
        ),
    )
    result = _compare(baseline, follow_up)
    assert result.mention_delta == 0.0
    assert result.citation_delta is None
    assert result.state == "PARTIAL"
    assert any("citation" in reason.lower() for reason in result.limitations)


def test_changed_approved_citation_domains_cannot_produce_citation_delta():
    changed = _source().model_dump(mode="json")
    changed["canonical_domains"] = ["studio.example", "www.studio.example"]
    follow_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(follow_source=follow_source)
    result = _compare(baseline, follow_up, follow_source=follow_source)
    assert result.mention_delta == 100.0
    assert result.citation_delta is None


@pytest.mark.parametrize("follow_time", ["2026-09-03T12:00:00Z", "2026-09-02T12:00:00Z"])
def test_nonchronological_observations_do_not_claim_change_over_time(follow_time):
    baseline, follow_up = _pair(
        after=tuple(_response(prompt_index=index, observed_at=follow_time) for index in range(2))
    )
    result = _compare(baseline, follow_up)
    assert result.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None
    assert any("chronolog" in reason.lower() for reason in result.limitations)
    assert "Observed change" not in result.chronology_statement


def test_comparison_rejects_tampered_cached_metrics_instead_of_trusting_model_copy():
    baseline, follow_up = _pair()
    forged = follow_up.model_copy(
        update={"metrics": follow_up.metrics.model_copy(update={"mention_rate": 0.0})}
    )
    with pytest.raises(ValueError, match="metrics"):
        _compare(baseline, forged)


def test_comparison_rejects_provider_rebound_sample_against_original_source():
    changed = _source().model_dump(mode="json")
    changed["binding"]["project_id"] = "other"
    changed_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(follow_source=changed_source)
    with pytest.raises(ValueError, match="binding|source"):
        _compare(baseline, follow_up)


def test_comparison_rejects_tampered_approved_domains_at_source_boundary():
    baseline, follow_up = _pair()
    # An unrelated extra domain does not change these metrics, but is not source-approved.
    forged = follow_up.model_copy(update={"approved_domains": ("extra.example", "studio.example")})
    with pytest.raises(ValueError, match="domain"):
        _compare(baseline, forged)


def test_comparison_stored_json_is_deeply_frozen_and_nonnumeric_state_rejects_deltas():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    result = _compare(baseline, follow_up)
    loaded = BenchmarkComparison.model_validate_json(result.model_dump_json())
    assert loaded == result
    with pytest.raises(ValidationError):
        loaded.mention_comparison.absolute_delta = 0
    with pytest.raises(ValidationError):
        loaded.baseline_metrics.mention_rate = 1
    damaged = json.loads(result.model_dump_json())
    damaged["state"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        BenchmarkComparison.model_validate(damaged)


def test_legacy_comparator_rejection_cannot_be_overridden_by_new_rate_arithmetic(monkeypatch):
    from ai_search_audit import comparisons

    original = comparisons.compare_observed_ai_visibility

    def reject(*args, **kwargs):
        # Exercise its real rejection path to prove this adapter consumes that decision.
        return original(*args, **{**kwargs, "follow_up_setup_fingerprint": None})

    monkeypatch.setattr(comparisons, "compare_observed_ai_visibility", reject)
    baseline, follow_up = _pair()
    result = _compare(baseline, follow_up)
    assert result.mention_comparison.state == "UNKNOWN"
    assert result.mention_delta is result.citation_delta is None


def test_fractional_rates_reuse_legacy_delta_without_float_equality_failure():
    changed = _source().model_dump(mode="json")
    changed["prompts"].append(
        {**changed["prompts"][0], "prompt_id": "p-third", "text": "Third prompt?"}
    )
    source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(
        source,
        before=tuple(_response(source, index, brand_mentioned=index == 0) for index in range(3)),
        after=tuple(
            _response(source, index, brand_mentioned=index != 0, observed_at="2026-09-04T12:00:00Z")
            for index in range(3)
        ),
    )
    result = _compare(baseline, follow_up, source)
    assert result.mention_delta == pytest.approx(100 / 3)


def test_stored_comparison_rejects_citation_delta_when_inspection_subsets_differ():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair(
        before=(
            _response(prompt_index=0),
            _response(prompt_index=1, citations_complete=False),
        ),
        after=(
            _response(prompt_index=0, observed_at="2026-09-04T12:00:00Z", citations_complete=False),
            _response(
                prompt_index=1,
                observed_at="2026-09-04T12:00:00Z",
                citations=("https://studio.example/",),
            ),
        ),
    )
    result = _compare(baseline, follow_up)
    damaged = json.loads(result.model_dump_json())
    damaged["citation_delta"] = 100.0
    with pytest.raises(ValidationError, match="citation|Citation"):
        BenchmarkComparison.model_validate(damaged)


def test_unknown_comparison_cannot_retain_numeric_nested_legacy_comparison():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    damaged = json.loads(_compare(baseline, follow_up).model_dump_json())
    damaged.update(
        state="UNKNOWN", mention_delta=None, citation_delta=None, limitations=["Unknown"]
    )
    with pytest.raises(ValidationError, match="numeric|legacy"):
        BenchmarkComparison.model_validate(damaged)


def test_stored_citation_references_must_resolve_to_measured_mentions():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    damaged = json.loads(_compare(baseline, follow_up).model_dump_json())
    damaged["baseline_citation_prompt_ids"] = ["dangling", "p-en"]
    damaged["follow_up_citation_prompt_ids"] = ["dangling", "p-en"]
    with pytest.raises(ValidationError, match="citation|Citation"):
        BenchmarkComparison.model_validate(damaged)


def test_available_comparison_requires_available_legacy_decision():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    damaged = json.loads(_compare(baseline, follow_up).model_dump_json())
    damaged["mention_comparison"].update(
        state="PARTIAL",
        baseline_canonical_prompt_ids=["p-pl", "p-en", "p-third"],
        follow_up_canonical_prompt_ids=["p-pl", "p-en", "p-third"],
    )
    with pytest.raises(ValidationError, match="available|coverage|legacy"):
        BenchmarkComparison.model_validate(damaged)


def test_equivalent_unicode_and_punycode_approved_domains_remain_comparable():
    from ai_search_audit.diagnostic_models import BenchmarkSample

    changed = _source().model_dump(mode="json")
    changed["canonical_domains"] = ["straße.example"]
    source = DiagnosticSource.model_validate(changed)
    changed["canonical_domains"] = ["xn--strae-oqa.example"]
    follow_source = DiagnosticSource.model_validate(changed)
    baseline, follow_up = _pair(
        source,
        follow_source,
        after=tuple(
            _response(
                follow_source,
                index,
                observed_at="2026-09-04T12:00:00Z",
                citations=("https://straße.example/source",),
            )
            for index in range(2)
        ),
    )
    baseline = BenchmarkSample.model_validate_json(baseline.model_dump_json())
    follow_up = BenchmarkSample.model_validate_json(follow_up.model_dump_json())
    result = _compare(baseline, follow_up, source, follow_source)
    assert (
        result.baseline_approved_domains
        == result.follow_up_approved_domains
        == ("xn--strae-oqa.example",)
    )
    assert result.citation_delta == 100.0


def test_stored_comparison_denominators_must_match_legacy_prompt_references():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    damaged = _compare(baseline, follow_up).model_dump(mode="json")
    for key in ("baseline_metrics", "follow_up_metrics"):
        damaged[key].update(
            expected=200,
            measured=200,
            citation_measured=2,
            citation_coverage=0.01,
            state="PARTIAL",
        )
    damaged["state"] = "PARTIAL"
    with pytest.raises(ValidationError, match="denominator|prompt|measured"):
        BenchmarkComparison.model_validate(damaged)


def test_stored_partial_comparison_preserves_actual_prompt_denominators():
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair(
        before=(_response(),),
        after=(_response(observed_at="2026-09-04T12:00:00Z"),),
    )
    result = _compare(baseline, follow_up)
    assert result.state == "PARTIAL"
    assert result.baseline_metrics.expected == result.follow_up_metrics.expected == 2
    assert result.baseline_metrics.measured == result.follow_up_metrics.measured == 1
    assert BenchmarkComparison.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("domain", ["straße.example", "https://studio.example/"])
def test_stored_comparison_domains_must_already_use_local_canonical_host_policy(domain):
    from ai_search_audit.diagnostic_models import BenchmarkComparison

    baseline, follow_up = _pair()
    damaged = _compare(baseline, follow_up).model_dump(mode="json")
    damaged["baseline_approved_domains"] = damaged["follow_up_approved_domains"] = [domain]
    with pytest.raises(ValidationError, match="domain|host"):
        BenchmarkComparison.model_validate(damaged)
