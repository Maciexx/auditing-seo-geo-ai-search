"""Fabricated API evidence only: no provider authenticity or live acceptance claims."""

import hashlib
from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest

from ai_search_audit.benchmark import (
    benchmark_setup_fingerprint,
    prepare_benchmark_worksheet,
    validate_benchmark_responses,
)
from tests.test_benchmark import _setup, _source, no_network

__all__ = ["no_network"]


def api():
    return import_module("ai_search_audit.diagnostic_observations")


def observation(source=None, *, day=4, selected=None, status="completed", **changes):
    module = api()
    source = source or _source()
    setup = module.ObservationSetup(
        benchmark_setup=_setup(interface="api", market=None, account_state=None),
        system_instruction_sha256="a" * 64,
        effective_request_policy_sha256="b" * 64,
        authentication="api_key",
    )
    worksheet = prepare_benchmark_worksheet(source, setup.benchmark_setup)
    selected = selected or tuple(p.prompt_id for p in source.prompts[:2])
    prompt = next(p for p in worksheet.prompts if p.prompt_id == selected[0])
    stamp = datetime(2026, 9, day, tzinfo=UTC)
    response = dict(
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.text,
        observed_at=stamp + timedelta(seconds=1),
        response_text="Example studio offers services.",
        grounded=True,
        brand_mentioned=True,
        citations=(),
        citations_complete=False,
        complete=True,
        response_truncated=False,
        inspection_scope="full_response",
    )
    response.update(changes)
    sample = validate_benchmark_responses(
        source, worksheet, [response] if status == "completed" else []
    )
    attempt = module.ObservationAttempt(
        attempt_id="attempt-1",
        prompt_id=prompt.prompt_id,
        locale=prompt.locale,
        started_at=stamp,
        ended_at=stamp + timedelta(seconds=2),
        requested_model=setup.benchmark_setup.model_id,
        returned_model=setup.benchmark_setup.model_id,
        requested_service_tier="default",
        returned_service_tier="default",
        status=status,
        complete=status == "completed",
        error_category="timeout" if status == "failed" else None,
        request_id="req_fixture_1",
        search_actions=(module.ObservationSearchAction(action="search", status="completed"),),
        usage=module.ObservationUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_input_tokens=2,
            reasoning_output_tokens=1,
        ),
        response_hash=sample.responses[0].response_hash if sample.responses else None,
        citation_annotations=tuple(
            module.ObservationCitationAnnotation(url=url, start_index=0, end_index=7)
            for url in response["citations"]
        )
        if response["citations_complete"]
        else None,
        citation_metadata_complete=response["citations_complete"],
    )
    return module.assemble_observation_run(
        source, worksheet, setup, selected, sample, (attempt,), price_provenance=None
    )


def revalidate(run, **changes):
    values = run.model_dump(mode="python")
    values.update(changes)
    return api().DiagnosticObservationRun.model_validate(values)


def test_versioned_partial_sample_preserves_whole_pack_and_authentication():
    run = observation()
    assert run.schema_version == "3.0.0"
    assert run.setup.schema_version == "1.0.0"
    assert run.setup.benchmark_setup.account_state is None
    assert run.setup.benchmark_setup.market is None
    assert run.sample.metrics.expected == 2
    assert run.sample.metrics.measured == 1
    assert run.sample.metrics.citation_measured == 0
    assert run.sample.metrics.citation_rate is None
    assert run.collection_range.start == run.attempts[0].started_at
    assert run.collection_range.end == run.attempts[0].ended_at
    assert run.attempts[0].locale == run.worksheet.prompts[0].locale
    with pytest.raises(ValueError):
        run.setup.authentication = "consumer_ui"


@pytest.mark.parametrize("status", ["failed", "incomplete", "refused", "no_text"])
def test_non_successful_requests_remain_attempts_not_negative_answers(status):
    run = observation(status=status)
    assert len(run.attempts) == 1
    assert run.sample.responses == ()
    assert run.sample.metrics.mention_rate is None
    assert run.sample.metrics.measured == 0


@pytest.mark.parametrize("selection", [("missing",), ("p-pl", "p-pl"), ("p-en", "p-pl")])
def test_invalid_or_noncanonical_selection_rejects(selection):
    run = observation()
    with pytest.raises(ValueError):
        api().assemble_observation_run(
            _source(),
            run.worksheet,
            run.setup,
            selection,
            run.sample,
            run.attempts,
            price_provenance=None,
        )


@pytest.mark.parametrize("mutation", ["foreign", "duplicate", "locale", "model", "time", "hash"])
def test_attempt_inventory_and_response_linkage_reject(mutation):
    run = observation()
    attempt = run.attempts[0].model_dump(mode="python")
    if mutation == "foreign":
        attempt["prompt_id"] = "foreign"
    elif mutation == "locale":
        attempt["locale"] = "foreign"
    elif mutation == "model":
        attempt["returned_model"] = "different-model"
    elif mutation == "time":
        attempt["ended_at"] = attempt["started_at"]
    elif mutation == "hash":
        attempt["response_hash"] = "c" * 64
    attempts = (attempt, attempt) if mutation == "duplicate" else (attempt,)
    with pytest.raises(ValueError):
        revalidate(run, attempts=attempts)


def test_model_tier_mismatch_preserved_as_ineligible_attempt_with_usage():
    run = observation(status="failed")
    attempt = run.attempts[0].model_copy(
        update={
            "returned_model": "actual-other",
            "returned_service_tier": "priority",
            "error_category": "model_mismatch",
        }
    )
    result = api().assemble_observation_run(
        _source(),
        run.worksheet,
        run.setup,
        run.selected_prompt_ids,
        run.sample,
        (attempt,),
        price_provenance=None,
    )
    assert result.attempts[0].returned_model == "actual-other"
    assert result.attempts[0].usage.total_tokens == 15
    assert result.sample.metrics.measured == 0


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "10"])
def test_usage_rejects_nonfinite_coercible_or_negative_values(field, value):
    usage = dict(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=2,
        reasoning_output_tokens=1,
    )
    usage[field] = value
    with pytest.raises(ValueError):
        api().ObservationUsage.model_validate(usage)


def test_usage_subdivisions_and_totals_are_not_double_counted():
    values = dict(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=2,
        reasoning_output_tokens=1,
    )
    assert api().ObservationUsage(**values).total_tokens == 15
    for mutation in (
        {"total_tokens": 18},
        {"cached_input_tokens": 11},
        {"reasoning_output_tokens": 6},
    ):
        with pytest.raises(ValueError):
            api().ObservationUsage(**dict(values, **mutation))


@pytest.mark.parametrize("entrypoint", ["model", "assembled_run"])
@pytest.mark.parametrize(
    "reported",
    [
        {"input_tokens": 10, "total_tokens": 5},
        {"output_tokens": 10, "total_tokens": 5},
        {"cached_input_tokens": 6, "total_tokens": 5},
        {"cached_input_tokens": 3, "reasoning_output_tokens": 3, "total_tokens": 5},
        {
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "reasoning_output_tokens": 3,
            "total_tokens": 12,
        },
        {
            "output_tokens": 10,
            "cached_input_tokens": 3,
            "reasoning_output_tokens": 2,
            "total_tokens": 12,
        },
    ],
)
def test_partial_usage_total_cannot_contradict_known_lower_bounds(reported, entrypoint):
    values = dict.fromkeys(api().ObservationUsage.model_fields)
    values.update(reported)
    with pytest.raises(ValueError, match="known.*lower bound"):
        if entrypoint == "model":
            api().ObservationUsage(**values)
        else:
            run = observation()
            unchecked = api().ObservationUsage.model_construct(**values)
            attempt = run.attempts[0].model_copy(update={"usage": unchecked})
            api().assemble_observation_run(
                _source(),
                run.worksheet,
                run.setup,
                run.selected_prompt_ids,
                run.sample,
                (attempt,),
                price_provenance=None,
            )


@pytest.mark.parametrize(
    "reported",
    [
        {},
        {"total_tokens": 0},
        {"input_tokens": 10, "cached_input_tokens": 2, "total_tokens": 10},
        {"output_tokens": 10, "reasoning_output_tokens": 2, "total_tokens": 10},
        {"cached_input_tokens": 3, "reasoning_output_tokens": 3, "total_tokens": 6},
        {
            "input_tokens": 10,
            "cached_input_tokens": 2,
            "reasoning_output_tokens": 3,
            "total_tokens": 13,
        },
        {
            "output_tokens": 10,
            "cached_input_tokens": 3,
            "reasoning_output_tokens": 2,
            "total_tokens": 13,
        },
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_input_tokens": 2,
            "reasoning_output_tokens": 1,
        },
    ],
)
def test_valid_partial_usage_preserves_nulls_without_double_counting(reported):
    values = dict.fromkeys(api().ObservationUsage.model_fields)
    values.update(reported)
    assert api().ObservationUsage(**values).model_dump(mode="python") == values


@pytest.mark.parametrize("field", ["usage", "request_id", "returned_model", "error_category"])
def test_required_nullable_attempt_fields_cannot_disappear(field):
    values = observation().attempts[0].model_dump(mode="python")
    del values[field]
    with pytest.raises(ValueError):
        api().ObservationAttempt.model_validate(values)


def test_partial_capture_retains_existing_hash_and_inspection_semantics():
    run = observation(response_truncated=True, inspection_scope="excerpt", brand_mentioned=None)
    response = run.sample.responses[0]
    assert response.source_response_hash is None
    assert response.response_hash == hashlib.sha256(response.response_excerpt.encode()).hexdigest()
    assert run.sample.metrics.measured == 0


@pytest.mark.parametrize(
    "field",
    ["algorithm_sha256", "input_sha256", "collection_range", "worksheet", "setup", "sample"],
)
def test_changed_cached_contract_rejects(field):
    run = observation()
    values = run.model_dump(mode="python")
    if field.endswith("sha256"):
        values[field] = "d" * 64
    elif field == "collection_range":
        values[field]["end"] += timedelta(seconds=1)
    elif field == "worksheet":
        values[field]["prompts"][0]["text"] = "forged"
    elif field == "setup":
        values[field]["benchmark_setup"]["model_id"] = "other"
    else:
        values[field]["metrics"]["mention_rate"] = 0
    with pytest.raises(ValueError):
        api().DiagnosticObservationRun.model_validate(values)


def test_legacy_setup_fingerprint_and_worksheet_bytes_unchanged():
    setup = _setup()
    expected = (
        b'{"account_state":"anonymous","interface":"consumer_ui","locale":"en",'
        b'"market":"PL","model_id":"example-1.0","product":"Example Search",'
        b'"provider":"Example AI","reset_method":"new_conversation","search_mode":"enabled"}'
    )
    assert benchmark_setup_fingerprint(setup) == hashlib.sha256(expected).hexdigest()
    worksheet = prepare_benchmark_worksheet(_source(), setup)
    before = worksheet.model_dump_json()
    api().ObservationSetup(
        benchmark_setup=_setup(interface="api", account_state=None),
        system_instruction_sha256="a" * 64,
        effective_request_policy_sha256="b" * 64,
        authentication="api_key",
    )
    assert worksheet.model_dump_json() == before
    assert "system_instruction_sha256" not in type(worksheet).model_fields


@pytest.mark.parametrize(
    "dimension", ["system_instruction_sha256", "effective_request_policy_sha256"]
)
def test_changed_new_setup_dimension_blocks_controlled_comparison(dimension):
    before, after = observation(), observation(day=5)
    changed = after.setup.model_copy(update={dimension: "f" * 64})
    after = api().assemble_observation_run(
        _source(),
        after.worksheet,
        changed,
        after.selected_prompt_ids,
        after.sample,
        after.attempts,
        price_provenance=None,
    )
    result = api().compare_observations(
        before, after, baseline_source=_source(), follow_up_source=_source()
    )
    assert result.schema_version == "1.0.0"
    assert result.comparison.mention_delta is None
    assert any("Observation setup" in item for item in result.comparison.limitations)


def test_unknown_legacy_settings_still_block_numeric_comparison():
    result = api().compare_observations(
        observation(), observation(day=5), baseline_source=_source(), follow_up_source=_source()
    )
    assert result.comparison.state == "UNKNOWN"
    assert result.comparison.mention_delta is None


def test_unchecked_provider_input_is_losslessly_revalidated():
    from ai_search_audit.diagnostic_models import BenchmarkResponseInput

    run = observation()
    response = run.sample.responses[0]
    values = response.model_dump(
        mode="python",
        exclude={
            "response_excerpt",
            "response_hash",
            "source_response_hash",
            "excerpt_hash",
            "captured_text_length",
            "excerpt_truncated",
        },
    )
    values["response_text"] = response.response_excerpt
    original = BenchmarkResponseInput(**values)
    for changed in (
        original.model_copy(update={"grounded": float("nan")}),
        original.model_copy(update={"unbounded_error": "forbidden"}),
        original.model_copy(update={"response_text": b"binary"}),
    ):
        with pytest.raises(ValueError):
            validate_benchmark_responses(_source(), run.worksheet, [changed])


def test_search_sources_and_citation_annotations_are_distinct_and_bounded():
    module = api()
    action = module.ObservationSearchAction(
        action="search", status="completed", consulted_sources=("https://other.example/",)
    )
    run = observation(citations=("https://studio.example/",), citations_complete=True)
    annotation = module.ObservationCitationAnnotation(
        url="https://studio.example/", start_index=0, end_index=7
    )
    attempt = run.attempts[0].model_copy(
        update={
            "search_actions": (action,),
            "citation_annotations": (annotation,),
            "citation_metadata_complete": True,
        }
    )
    result = module.assemble_observation_run(
        _source(),
        run.worksheet,
        run.setup,
        run.selected_prompt_ids,
        run.sample,
        (attempt,),
        price_provenance=None,
    )
    assert str(result.attempts[0].search_actions[0].consulted_sources[0]) != str(annotation.url)
    for changes in ({"end_index": 1000}, {"url": "https://foreign.example/"}):
        invalid = annotation.model_copy(update=changes)
        with pytest.raises(ValueError):
            module.assemble_observation_run(
                _source(),
                run.worksheet,
                run.setup,
                run.selected_prompt_ids,
                run.sample,
                (attempt.model_copy(update={"citation_annotations": (invalid,)}),),
                price_provenance=None,
            )


def test_completed_response_grounding_needs_completed_search_action():
    run = observation()
    with pytest.raises(ValueError):
        api().assemble_observation_run(
            _source(),
            run.worksheet,
            run.setup,
            run.selected_prompt_ids,
            run.sample,
            (run.attempts[0].model_copy(update={"search_actions": ()}),),
            price_provenance=None,
        )


def test_price_provenance_retains_decimal_snapshot_without_calculating_cost():
    from decimal import Decimal

    values = dict(
        model_id="example-1.0",
        service_tier="default",
        currency="USD",
        source_urls=("https://pricing.example/models",),
        as_of="2026-09-04",
        input_per_million="1.25",
        cached_input_per_million="0.125",
        output_per_million="10",
        search_per_thousand="10",
    )
    snapshot = api().ObservationPriceProvenance(**values)
    assert snapshot.input_per_million == Decimal("1.25")
    for value in ("NaN", "Infinity", "-1"):
        with pytest.raises(ValueError):
            api().ObservationPriceProvenance(**dict(values, input_per_million=value))


def test_duplicate_received_search_ids_are_rejected():
    run = observation()
    action = api().ObservationSearchAction(action="search", status="completed", call_id="ws_1")
    with pytest.raises(ValueError):
        api().assemble_observation_run(
            _source(),
            run.worksheet,
            run.setup,
            run.selected_prompt_ids,
            run.sample,
            (run.attempts[0].model_copy(update={"search_actions": (action, action)}),),
            price_provenance=None,
        )


@pytest.mark.parametrize(
    "account_state", ["anonymous", "signed_in_free", "signed_in_paid", "enterprise"]
)
def test_api_authentication_cannot_manufacture_legacy_consumer_account(account_state):
    with pytest.raises(ValueError, match="consumer account"):
        api().ObservationSetup(
            benchmark_setup=_setup(interface="api", account_state=account_state),
            system_instruction_sha256="a" * 64,
            effective_request_policy_sha256="b" * 64,
            authentication="api_key",
        )


@pytest.mark.parametrize("reason", ["missing_key", "budget_exhausted"])
def test_preflight_nonexecution_cannot_be_stored_as_actual_request(reason):
    values = observation(status="failed").attempts[0].model_dump(mode="python")
    values["error_category"] = reason
    with pytest.raises(ValueError):
        api().ObservationAttempt.model_validate(values)


def test_max_twelve_selection_never_shrinks_whole_pack_denominator():
    source = _source()
    source = source.model_copy(
        update={
            "prompts": tuple(
                source.prompts[0].model_copy(update={"prompt_id": f"p-{i:02d}"}) for i in range(13)
            )
        }
    )
    run = observation(source, selected=tuple(p.prompt_id for p in source.prompts[:12]))
    assert len(run.selected_prompt_ids) == 12
    assert run.sample.metrics.expected == 13
    assert run.sample.metrics.measured == 1
    with pytest.raises(ValueError, match="at most 12"):
        revalidate(
            run,
            attempts=tuple(
                run.attempts[0].model_copy(
                    update={"attempt_id": f"a-{i}", "prompt_id": p.prompt_id}
                )
                for i, p in enumerate(source.prompts)
            ),
        )
    with pytest.raises(ValueError):
        api().assemble_observation_run(
            source,
            run.worksheet,
            run.setup,
            tuple(p.prompt_id for p in source.prompts),
            run.sample,
            run.attempts,
            price_provenance=None,
        )


def test_unavailable_usage_is_null_not_zero_and_empty_sources_are_distinct():
    run = observation()
    attempt = run.attempts[0].model_copy(update={"usage": None, "request_id": None})
    result = api().assemble_observation_run(
        _source(),
        run.worksheet,
        run.setup,
        run.selected_prompt_ids,
        run.sample,
        (attempt,),
        price_provenance=None,
    )
    assert result.attempts[0].usage is None
    assert result.attempts[0].request_id is None
    assert result.attempts[0].search_actions[0].consulted_sources is None


def test_comparison_explicitly_checks_selected_ids_and_actual_model_tier():
    before, after = observation(), observation(day=5, selected=("p-pl",))
    result = api().compare_observations(
        before, after, baseline_source=_source(), follow_up_source=_source()
    )
    assert any("Selected observation" in item for item in result.comparison.limitations)
    failed = observation(day=5, status="failed")
    result = api().compare_observations(
        before, failed, baseline_source=_source(), follow_up_source=_source()
    )
    assert any("Actual observation models" in item for item in result.comparison.limitations)


def test_retained_search_action_uses_actual_find_in_page_name():
    assert api().ObservationSearchAction(action="find_in_page", status="completed").action == (
        "find_in_page"
    )
    with pytest.raises(ValueError):
        api().ObservationSearchAction(action="find", status="completed")


def test_extra_allow_subclass_fields_cannot_disappear_before_revalidation():
    from pydantic import ConfigDict

    class ExtendedAttempt(api().ObservationAttempt):
        model_config = ConfigDict(extra="allow")

    run = observation()
    unchecked = ExtendedAttempt(
        **run.attempts[0].model_dump(mode="python"), unknown_extra="must not disappear"
    )
    assert unchecked.__pydantic_extra__
    with pytest.raises(ValueError, match="unknown fields"):
        api().assemble_observation_run(
            _source(),
            run.worksheet,
            run.setup,
            run.selected_prompt_ids,
            run.sample,
            (unchecked,),
            price_provenance=None,
        )
