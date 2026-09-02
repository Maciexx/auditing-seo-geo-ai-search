from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from ai_search_audit.comparisons import (
    AIVisibilityComparison,
    ComparisonBasis,
    ComparisonCausality,
    ComparisonLimitation,
    FindingChange,
    MetricComparison,
    ValidationComparison,
    ValidationTimingState,
    compare_findings,
    compare_observed_ai_visibility,
    compare_visibility,
    validate_follow_up_canonical_binding,
    validation_target_date,
)
from ai_search_audit.data_intake import (
    DateRange,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilitySource,
)
from ai_search_audit.models import (
    AIObservation,
    AuditRun,
    ClaimModality,
    DataState,
    FactualClaim,
    Finding,
    FindingStatus,
    Priority,
    Severity,
)
from ai_search_audit.visibility_metrics import MetricWindow, VisibilityMetric, VisibilitySnapshot

NOW = datetime(2026, 9, 1, 9, tzinfo=UTC)


def test_canonical_binding_rejects_coherent_but_invented_ai_follow_up_value() -> None:
    baseline = AIObservation(
        observation_id="observation-baseline",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=False,
        grounded=True,
    )
    follow_up = baseline.model_copy(
        update={"observation_id": "observation-follow-up", "brand_mentioned": True}
    )
    ai = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
        baseline_canonical_prompt_ids=("prompt-1",),
        follow_up_canonical_prompt_ids=("prompt-1",),
    )
    tampered_ai = AIVisibilityComparison.model_validate(
        {
            **ai.model_dump(mode="python"),
            "follow_up_value": 50,
            "absolute_delta": 50,
        }
    )
    comparison = compare_visibility(
        None,
        None,
        implementation_date=date(2026, 6, 1),
        observed_at=NOW.date(),
    ).model_copy(update={"ai_visibility": tampered_ai})
    audit = AuditRun.model_construct(
        timestamp=NOW,
        configuration={
            "implementation_date": "2026-06-01",
            "validation_comparison_schema_version": comparison.schema_version,
        },
        findings=[],
        ai_observations=[follow_up],
    )

    with pytest.raises(ValueError, match="AI.*value"):
        validate_follow_up_canonical_binding(
            comparison,
            audit=audit,
            visibility_snapshot=None,
            prompt_pack_version="1.0.0",
            setup_fingerprint="a" * 64,
            canonical_prompt_ids=("prompt-1",),
        )


def _source(source_id: str, *, start: date, end: date) -> SourceArtifactProvenance:
    return SourceArtifactProvenance(
        source_id=source_id,
        filename=f"{source_id}.csv",
        sha256=("a" if source_id == "baseline" else "b") * 64,
        byte_count=10,
        platform=VisibilitySource.GOOGLE_ANALYTICS,
        report_type="visibility-export",
        date_range=DateRange(start=start, end=end),
        filters=("channel=organic_search",),
        exported_at=NOW,
        processed_at=NOW,
        deleted_at=NOW,
    )


def _snapshot(
    *,
    source_id: str,
    start: date,
    end: date,
    value: float | None,
    state: DataState = DataState.AVAILABLE,
    unit: str = "sessions",
    definitions: tuple[str, ...] = (
        "report_type=visibility-export",
        "aggregation=sum",
        "cadence=monthly",
        "channel=organic_search",
    ),
    segments: tuple[str, ...] = ("organic_search",),
    coverage: float = 1,
    confidence: float = 0.9,
) -> VisibilitySnapshot:
    numeric = state in {DataState.AVAILABLE, DataState.PARTIAL}
    points = (VisibilityMetricPoint(period_start=start, value=value or 0),) if numeric else ()
    metric = VisibilityMetric(
        metric_id="organic-sessions",
        metric="ga4.organic_search.sessions",
        unit=unit,
        state=state,
        value=value,
        coverage=coverage,
        confidence=confidence,
        source_ids=(source_id,),
        window=MetricWindow(start=start, end=end) if numeric else None,
        segments=segments,
        definitions=definitions,
        points=points,
    )
    return VisibilitySnapshot(
        project_id="example",
        canonical_domain="example.com",
        processed_at=NOW,
        deleted_at=NOW,
        metrics=(metric,),
        sources=(_source(source_id, start=start, end=end),),
    )


def _finding(
    *,
    title: str,
    value: str = "blocked",
    affected_urls: tuple[str, ...] = ("https://example.com/a",),
) -> Finding:
    return Finding(
        finding_id="finding-robots",
        category="ai_search_access",
        severity=Severity.HIGH,
        status=FindingStatus.CONFIRMED,
        rule_id="crawler-access-001",
        technical_title=title,
        technical_description=title,
        client_title=title,
        client_explanation=title,
        business_impact=title,
        implementation=title,
        priority=Priority.P1,
        affected_urls=list(affected_urls),
        evidence_ids=["evidence-robots"],
        confidence=0.9,
        factual_claims=[
            FactualClaim(
                claim_id="claim-robots-oai",
                predicate="crawler_access",
                value=value,
                evidence_ids=["evidence-robots"],
                modality=ClaimModality.OBSERVED,
                meaning="OAI-SearchBot access state",
            )
        ],
    )


def test_default_validation_target_is_90_days_after_implementation() -> None:
    assert validation_target_date(date(2026, 9, 1)) == date(2026, 11, 30)


def test_early_validation_has_explicit_state_and_warning() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
        implementation_date=date(2026, 7, 1),
        observed_at=date(2026, 8, 1),
    )

    assert result.timing_state is ValidationTimingState.EARLY
    assert result.timing_warning is not None
    assert result.validation_target_date == date(2026, 9, 29)


@pytest.mark.parametrize(
    "mutation",
    (
        {"chronology_statement": "Implementation definitely caused the change."},
        {"timing_state": ValidationTimingState.EARLY},
        {"timing_warning": "False warning."},
    ),
)
def test_persisted_comparison_rejects_causal_or_dateless_timing_claims(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ValidationComparison.model_validate(mutation)


def test_persisted_early_warning_must_equal_system_text() -> None:
    payload = compare_visibility(
        None,
        None,
        implementation_date=date(2026, 9, 1),
        observed_at=date(2026, 9, 15),
    ).model_dump(mode="json")
    payload["timing_warning"] = "Possibly early."

    with pytest.raises(ValidationError, match="system warning"):
        ValidationComparison.model_validate(payload)


def test_target_reached_forbids_timing_warning() -> None:
    payload = compare_visibility(
        None,
        None,
        implementation_date=date(2026, 9, 1),
        observed_at=date(2026, 11, 30),
    ).model_dump(mode="json")
    payload["timing_warning"] = "Early validation."

    with pytest.raises(ValidationError, match="TARGET_REACHED"):
        ValidationComparison.model_validate(payload)


def test_validation_rejects_observation_before_implementation() -> None:
    with pytest.raises(ValueError, match="before implementation"):
        compare_visibility(
            None,
            None,
            implementation_date=date(2026, 9, 1),
            observed_at=date(2026, 8, 31),
        )


@pytest.mark.parametrize(
    ("baseline_window", "follow_up_window", "implementation_date", "limitation"),
    (
        (
            (date(2026, 8, 1), date(2026, 8, 31)),
            (date(2026, 7, 1), date(2026, 7, 31)),
            None,
            ComparisonLimitation.INVALID_WINDOW_ORDER,
        ),
        (
            (date(2026, 7, 1), date(2026, 7, 31)),
            (date(2026, 8, 1), date(2026, 8, 31)),
            date(2026, 9, 1),
            ComparisonLimitation.NON_POST_IMPLEMENTATION_WINDOW,
        ),
    ),
)
def test_delta_requires_ordered_postimplementation_windows(
    baseline_window: tuple[date, date],
    follow_up_window: tuple[date, date],
    implementation_date: date | None,
    limitation: ComparisonLimitation,
) -> None:
    comparison = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=baseline_window[0],
            end=baseline_window[1],
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=follow_up_window[0],
            end=follow_up_window[1],
            value=12,
        ),
        implementation_date=implementation_date,
        observed_at=(date(2026, 10, 1) if implementation_date else None),
    ).metrics[0]

    assert comparison.state is DataState.UNKNOWN
    assert comparison.basis is ComparisonBasis.NON_COMPARABLE
    assert comparison.absolute_delta is None
    assert limitation in comparison.limitations


@pytest.mark.parametrize(
    "mutation",
    (
        "unit",
        "definitions",
        "segments",
        "window",
    ),
)
def test_direct_delta_requires_equivalent_metric_semantics_and_window(
    mutation: str,
) -> None:
    baseline = _snapshot(
        source_id="baseline",
        start=date(2026, 7, 1),
        end=date(2026, 7, 31),
        value=10,
    )
    follow_up = _snapshot(
        source_id="follow-up",
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        value=12,
    )
    metric = follow_up.metrics[0]
    if mutation == "unit":
        metric = metric.model_copy(update={"unit": "users"})
    elif mutation == "definitions":
        metric = metric.model_copy(update={"definitions": ("channel=direct",)})
    elif mutation == "segments":
        metric = metric.model_copy(update={"segments": ("all_traffic",)})
    else:
        metric = metric.model_copy(
            update={"window": MetricWindow(start=date(2026, 8, 1), end=date(2026, 8, 15))}
        )
    follow_up = follow_up.model_copy(update={"metrics": (metric,)})

    comparison = compare_visibility(baseline, follow_up).metrics[0]

    assert comparison.state is DataState.UNKNOWN
    assert comparison.absolute_delta is None
    assert comparison.relative_delta_percent is None
    assert comparison.basis is ComparisonBasis.NON_COMPARABLE
    assert comparison.limitations


def test_matching_yoy_windows_are_preferred_for_seasonal_business() -> None:
    comparison = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2025, 8, 1),
            end=date(2025, 8, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
        seasonal=True,
    ).metrics[0]

    assert comparison.state is DataState.AVAILABLE
    assert comparison.basis is ComparisonBasis.YEAR_OVER_YEAR
    assert comparison.absolute_delta == 2
    assert comparison.relative_delta_percent == 20


def test_non_numeric_states_stay_non_numeric_and_distinct() -> None:
    comparison = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=None,
            state=DataState.UNAVAILABLE,
            coverage=0,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=None,
            state=DataState.FAILED,
            coverage=0,
        ),
    ).metrics[0]

    assert comparison.state is DataState.UNKNOWN
    assert comparison.baseline_state is DataState.UNAVAILABLE
    assert comparison.follow_up_state is DataState.FAILED
    assert comparison.absolute_delta is None


def test_delta_carries_sources_quality_and_noncausal_language() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=0,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=5,
            state=DataState.PARTIAL,
            coverage=0.8,
            confidence=0.7,
        ),
    )

    metric = result.metrics[0]
    assert metric.absolute_delta == 5
    assert metric.relative_delta_percent is None
    assert metric.baseline_source_ids == ("baseline",)
    assert metric.follow_up_source_ids == ("follow-up",)
    assert metric.coverage == 0.8
    assert metric.confidence == 0.7
    assert metric.limitations
    assert result.causality is ComparisonCausality.NOT_ESTABLISHED
    serialized = result.model_dump_json().casefold()
    assert "caused" not in serialized
    assert "spowodował" not in serialized


def test_public_findings_are_compared_by_rule_and_fact_not_prose() -> None:
    follow_up = _finding(title="Completely different wording")
    follow_up.factual_claims[0].meaning = "Reworded explanation of the same fact"
    comparison = compare_findings(
        (_finding(title="Old wording"),),
        (follow_up,),
    )

    assert comparison[0].change is FindingChange.UNCHANGED


def test_public_finding_identity_includes_normalized_affected_urls() -> None:
    wording_only = compare_findings(
        (_finding(title="Old", affected_urls=("HTTPS://EXAMPLE.COM/a#section",)),),
        (_finding(title="New", affected_urls=("https://example.com/a",)),),
    )
    moved = compare_findings(
        (_finding(title="Old", affected_urls=("https://example.com/a",)),),
        (_finding(title="New", affected_urls=("https://example.com/b",)),),
    )

    assert [item.change for item in wording_only] == [FindingChange.UNCHANGED]
    assert {item.change for item in moved} == {FindingChange.RESOLVED, FindingChange.ADDED}


def test_ai_visibility_requires_same_prompt_pack_and_reproducible_setup() -> None:
    observation = AIObservation(
        observation_id="observation-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=True,
        grounded=True,
    )

    mismatch = compare_observed_ai_visibility(
        (observation,),
        (observation.model_copy(update={"observation_id": "observation-2"}),),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.1.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
    )
    no_setup = compare_observed_ai_visibility(
        (observation,),
        (observation.model_copy(update={"observation_id": "observation-2"}),),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint=None,
        follow_up_setup_fingerprint=None,
    )

    assert mismatch.state is DataState.UNKNOWN
    assert no_setup.state is DataState.UNKNOWN
    assert mismatch.absolute_delta is None
    assert no_setup.absolute_delta is None


def test_ungrounded_ai_observations_do_not_create_numeric_visibility_delta() -> None:
    baseline = AIObservation(
        observation_id="observation-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=True,
        grounded=False,
    )
    follow_up = baseline.model_copy(update={"observation_id": "observation-2", "grounded": True})

    result = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
    )

    assert result.state is DataState.UNKNOWN
    assert result.absolute_delta is None


def test_ai_observations_require_the_same_prompt_set() -> None:
    baseline = AIObservation(
        observation_id="observation-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=True,
        grounded=True,
    )
    follow_up = baseline.model_copy(
        update={"observation_id": "observation-2", "prompt_id": "prompt-2"}
    )

    result = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
    )

    assert result.state is DataState.UNKNOWN
    assert result.absolute_delta is None


def test_ai_visibility_requires_same_grounded_measurable_prompt_subset() -> None:
    baseline = (
        AIObservation(
            observation_id="baseline-1",
            prompt_id="prompt-1",
            provider="provider",
            observed_at=NOW,
            brand_mentioned=True,
            grounded=True,
        ),
        AIObservation(
            observation_id="baseline-2",
            prompt_id="prompt-2",
            provider="provider",
            observed_at=NOW,
            brand_mentioned=False,
            grounded=True,
        ),
    )
    follow_up = (
        baseline[0].model_copy(update={"observation_id": "follow-up-1"}),
        baseline[1].model_copy(update={"observation_id": "follow-up-2", "grounded": False}),
    )

    result = compare_observed_ai_visibility(
        baseline,
        follow_up,
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
    )

    assert result.state is DataState.UNKNOWN
    assert result.absolute_delta is None
    assert result.baseline_prompt_ids == ("prompt-1", "prompt-2")
    assert result.follow_up_prompt_ids == ("prompt-1", "prompt-2")
    assert result.baseline_measurable_prompt_ids == ("prompt-1", "prompt-2")
    assert result.follow_up_measurable_prompt_ids == ("prompt-1",)
    assert ComparisonLimitation.PROMPT_SET_MISMATCH in result.limitations


def test_ai_visibility_also_requires_same_full_observation_prompt_set() -> None:
    baseline = (
        AIObservation(
            observation_id="baseline-1",
            prompt_id="prompt-1",
            provider="provider",
            observed_at=NOW,
            brand_mentioned=True,
            grounded=True,
        ),
        AIObservation(
            observation_id="baseline-2",
            prompt_id="prompt-2",
            provider="provider",
            observed_at=NOW,
            brand_mentioned=None,
            grounded=False,
        ),
    )
    follow_up = (baseline[0].model_copy(update={"observation_id": "follow-up-1"}),)

    result = compare_observed_ai_visibility(
        baseline,
        follow_up,
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
    )

    assert result.state is DataState.UNKNOWN
    assert result.absolute_delta is None
    assert result.baseline_prompt_ids == ("prompt-1", "prompt-2")
    assert result.follow_up_prompt_ids == ("prompt-1",)
    assert result.baseline_measurable_prompt_ids == ("prompt-1",)
    assert result.follow_up_measurable_prompt_ids == ("prompt-1",)
    assert ComparisonLimitation.PROMPT_SET_MISMATCH in result.limitations


def test_numeric_ai_comparison_persists_canonical_prompt_ids() -> None:
    baseline = AIObservation(
        observation_id="baseline-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=False,
        grounded=True,
    )
    follow_up = baseline.model_copy(
        update={"observation_id": "follow-up-1", "brand_mentioned": True}
    )

    result = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
        baseline_canonical_prompt_ids=("prompt-1",),
        follow_up_canonical_prompt_ids=("prompt-1",),
    )

    assert result.state is DataState.AVAILABLE
    assert result.baseline_prompt_ids == ("prompt-1",)
    assert result.follow_up_prompt_ids == ("prompt-1",)
    assert result.baseline_measurable_prompt_ids == ("prompt-1",)
    assert result.follow_up_measurable_prompt_ids == ("prompt-1",)
    assert result.baseline_canonical_prompt_ids == ("prompt-1",)
    assert result.follow_up_canonical_prompt_ids == ("prompt-1",)


def test_same_incomplete_ai_prompt_subset_is_partial_not_available() -> None:
    baseline = AIObservation(
        observation_id="baseline-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=False,
        grounded=True,
    )
    follow_up = baseline.model_copy(
        update={"observation_id": "follow-up-1", "brand_mentioned": True}
    )

    result = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
        baseline_canonical_prompt_ids=("prompt-1", "prompt-2"),
        follow_up_canonical_prompt_ids=("prompt-1", "prompt-2"),
    )

    assert result.state is DataState.PARTIAL
    assert result.absolute_delta == 100
    assert result.limitations == ()
    assert result.baseline_canonical_prompt_ids == ("prompt-1", "prompt-2")
    assert result.follow_up_canonical_prompt_ids == ("prompt-1", "prompt-2")


def test_differing_incomplete_ai_prompt_subsets_are_unknown() -> None:
    baseline = AIObservation(
        observation_id="baseline-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=False,
        grounded=True,
    )
    follow_up = baseline.model_copy(
        update={"observation_id": "follow-up-1", "prompt_id": "prompt-2"}
    )

    result = compare_observed_ai_visibility(
        (baseline,),
        (follow_up,),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
        baseline_canonical_prompt_ids=("prompt-1", "prompt-2"),
        follow_up_canonical_prompt_ids=("prompt-1", "prompt-2"),
    )

    assert result.state is DataState.UNKNOWN
    assert result.absolute_delta is None
    assert ComparisonLimitation.PROMPT_SET_MISMATCH in result.limitations


def test_ai_observation_prompt_refs_must_resolve_to_canonical_prompt_pack() -> None:
    observation = AIObservation(
        observation_id="observation-1",
        prompt_id="prompt-outside-pack",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=True,
        grounded=True,
    )

    result = compare_observed_ai_visibility(
        (observation,),
        (observation.model_copy(update={"observation_id": "observation-2"}),),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="a" * 64,
        follow_up_setup_fingerprint="a" * 64,
        baseline_canonical_prompt_ids=("prompt-1",),
        follow_up_canonical_prompt_ids=("prompt-1",),
    )

    assert result.state is DataState.UNKNOWN
    assert ComparisonLimitation.PROMPT_SET_MISMATCH in result.limitations


def test_ai_comparison_treats_invalid_setup_fingerprint_as_unknown() -> None:
    observation = AIObservation(
        observation_id="observation-1",
        prompt_id="prompt-1",
        provider="provider",
        observed_at=NOW,
        brand_mentioned=True,
        grounded=True,
    )

    result = compare_observed_ai_visibility(
        (observation,),
        (observation.model_copy(update={"observation_id": "observation-2"}),),
        baseline_prompt_pack_version="1.0.0",
        follow_up_prompt_pack_version="1.0.0",
        baseline_setup_fingerprint="A" * 64,
        follow_up_setup_fingerprint="A" * 64,
    )

    assert result.state is DataState.UNKNOWN
    assert result.setup_fingerprint is None
    assert ComparisonLimitation.OBSERVATION_SETUP_MISMATCH in result.limitations


def test_persisted_comparison_rejects_recomputed_numeric_delta() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
    )
    payload = result.model_dump(mode="json")
    payload["metrics"][0]["absolute_delta"] = 999

    with pytest.raises(ValidationError, match="delta"):
        type(result).model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"basis": ComparisonBasis.DIRECT, "limitations": ()}, "UNKNOWN"),
        ({"limitations": ()}, "limitation"),
        ({"absolute_delta": 2}, "delta"),
    ),
)
def test_persisted_unknown_metric_requires_noncomparable_explained_shape(
    mutation: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "metric_id": "metric-1",
        "metric": "ga4.organic_search.sessions",
        "unit": "sessions",
        "state": DataState.UNKNOWN,
        "basis": ComparisonBasis.NON_COMPARABLE,
        "baseline_state": DataState.AVAILABLE,
        "follow_up_state": DataState.AVAILABLE,
        "coverage": 1,
        "confidence": 0.9,
        "limitations": (ComparisonLimitation.WINDOW_MISMATCH,),
    }
    values.update(mutation)

    with pytest.raises(ValidationError, match=match):
        MetricComparison.model_validate(values)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"baseline_source_ids": ()}, "source"),
        ({"baseline_window": None}, "window"),
        (
            {
                "follow_up_window": {
                    "start": date(2026, 8, 1),
                    "end": date(2026, 8, 15),
                }
            },
            "equivalent",
        ),
        ({"baseline_state": DataState.PARTIAL}, "state"),
        ({"coverage": 0.5}, "AVAILABLE"),
        ({"confidence": 0}, "confidence"),
    ),
)
def test_persisted_numeric_metric_requires_complete_coherent_provenance(
    mutation: dict[str, object],
    match: str,
) -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
    )
    payload = result.metrics[0].model_dump(mode="python")
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        MetricComparison.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"baseline_observation_ids": ()}, "observation"),
        ({"baseline_observation_ids": ("duplicate", "duplicate")}, "unique"),
        ({"baseline_prompt_ids": ()}, "prompt"),
        ({"follow_up_prompt_ids": ("prompt-2",)}, "prompt"),
        ({"setup_fingerprint": "A" * 64}, "pattern"),
        ({"prompt_pack_version": ""}, "pack"),
    ),
)
def test_persisted_numeric_ai_comparison_rejects_tampered_provenance(
    mutation: dict[str, object],
    match: str,
) -> None:
    payload: dict[str, object] = {
        "state": DataState.AVAILABLE,
        "baseline_value": 0,
        "follow_up_value": 100,
        "absolute_delta": 100,
        "prompt_pack_version": "1.0.0",
        "setup_fingerprint": "a" * 64,
        "baseline_observation_ids": ("baseline-1",),
        "follow_up_observation_ids": ("follow-up-1",),
        "baseline_prompt_ids": ("prompt-1",),
        "follow_up_prompt_ids": ("prompt-1",),
        "baseline_measurable_prompt_ids": ("prompt-1",),
        "follow_up_measurable_prompt_ids": ("prompt-1",),
        "baseline_canonical_prompt_ids": ("prompt-1",),
        "follow_up_canonical_prompt_ids": ("prompt-1",),
    }
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        AIVisibilityComparison.model_validate(payload)


def test_persisted_numeric_ai_comparison_requires_empty_limitations() -> None:
    payload: dict[str, object] = {
        "state": DataState.AVAILABLE,
        "baseline_value": 0,
        "follow_up_value": 100,
        "absolute_delta": 100,
        "prompt_pack_version": "1.0.0",
        "setup_fingerprint": "a" * 64,
        "baseline_observation_ids": ("baseline-1",),
        "follow_up_observation_ids": ("follow-up-1",),
        "baseline_prompt_ids": ("prompt-1",),
        "follow_up_prompt_ids": ("prompt-1",),
        "baseline_measurable_prompt_ids": ("prompt-1",),
        "follow_up_measurable_prompt_ids": ("prompt-1",),
        "baseline_canonical_prompt_ids": ("prompt-1",),
        "follow_up_canonical_prompt_ids": ("prompt-1",),
        "limitations": (ComparisonLimitation.PROMPT_SET_MISMATCH,),
    }

    with pytest.raises(ValidationError, match="numeric AI.*limitation"):
        AIVisibilityComparison.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        {"baseline_canonical_prompt_ids": ()},
        {"baseline_canonical_prompt_ids": ("prompt-1", "prompt-1")},
        {"follow_up_canonical_prompt_ids": ("prompt-2",)},
        {
            "baseline_prompt_ids": ("prompt-outside-pack",),
            "follow_up_prompt_ids": ("prompt-outside-pack",),
            "baseline_measurable_prompt_ids": ("prompt-outside-pack",),
            "follow_up_measurable_prompt_ids": ("prompt-outside-pack",),
        },
    ),
)
def test_persisted_numeric_ai_comparison_validates_canonical_pack_refs(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "state": DataState.AVAILABLE,
        "baseline_value": 0,
        "follow_up_value": 100,
        "absolute_delta": 100,
        "prompt_pack_version": "1.0.0",
        "setup_fingerprint": "a" * 64,
        "baseline_observation_ids": ("baseline-1",),
        "follow_up_observation_ids": ("follow-up-1",),
        "baseline_prompt_ids": ("prompt-1",),
        "follow_up_prompt_ids": ("prompt-1",),
        "baseline_measurable_prompt_ids": ("prompt-1",),
        "follow_up_measurable_prompt_ids": ("prompt-1",),
        "baseline_canonical_prompt_ids": ("prompt-1",),
        "follow_up_canonical_prompt_ids": ("prompt-1",),
    }
    payload.update(mutation)

    with pytest.raises(ValidationError, match="canonical"):
        AIVisibilityComparison.model_validate(payload)


def test_validation_comparison_schema_version_is_fixed() -> None:
    payload = ValidationComparison().model_dump(mode="json")
    payload["schema_version"] = "9.0.0"

    with pytest.raises(ValidationError, match="schema_version"):
        ValidationComparison.model_validate(payload)


@pytest.mark.parametrize(
    "limitation",
    (
        ComparisonLimitation.UNIT_MISMATCH,
        ComparisonLimitation.DEFINITION_FILTER_MISMATCH,
        ComparisonLimitation.SEGMENT_MISMATCH,
        ComparisonLimitation.WINDOW_MISMATCH,
        ComparisonLimitation.MISSING_BASELINE,
    ),
)
def test_persisted_numeric_metric_rejects_mismatch_limitations(
    limitation: ComparisonLimitation,
) -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
    )
    payload = result.metrics[0].model_dump(mode="python")
    payload["limitations"] = (limitation,)

    with pytest.raises(ValidationError, match="numeric.*limitation"):
        MetricComparison.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        {"state": DataState.PARTIAL, "coverage": 0.8, "limitations": ()},
        {"limitations": (ComparisonLimitation.PARTIAL_COVERAGE,)},
        {"limitations": (ComparisonLimitation.ZERO_BASELINE,)},
    ),
)
def test_persisted_numeric_metric_limitations_match_derived_state(
    mutation: dict[str, object],
) -> None:
    baseline = _snapshot(
        source_id="baseline",
        start=date(2026, 7, 1),
        end=date(2026, 7, 31),
        value=10,
    )
    follow_up = _snapshot(
        source_id="follow-up",
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        value=12,
    )
    payload = compare_visibility(baseline, follow_up).metrics[0].model_dump(mode="python")
    if mutation.get("state") is DataState.PARTIAL:
        payload["baseline_state"] = DataState.PARTIAL
    payload.update(mutation)

    with pytest.raises(ValidationError, match="limitation"):
        MetricComparison.model_validate(payload)


def test_persisted_validation_rejects_numeric_preimplementation_window() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
        implementation_date=date(2026, 7, 1),
        observed_at=date(2026, 9, 1),
    )
    payload = result.model_dump(mode="json")
    payload["implementation_date"] = "2026-08-15"
    payload["validation_target_date"] = "2026-11-13"
    payload["timing_state"] = "EARLY"
    payload["timing_warning"] = "Early validation."

    with pytest.raises(ValidationError, match="post-implementation"):
        ValidationComparison.model_validate(payload)


def test_future_ending_follow_up_window_stays_unknown() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
        implementation_date=date(2026, 7, 1),
        observed_at=date(2026, 8, 15),
    )

    metric = result.metrics[0]
    assert metric.state is DataState.UNKNOWN
    assert metric.absolute_delta is None
    assert ComparisonLimitation.WINDOW_AFTER_OBSERVATION in metric.limitations


def test_persisted_validation_rejects_numeric_window_after_observation() -> None:
    result = compare_visibility(
        _snapshot(
            source_id="baseline",
            start=date(2026, 7, 1),
            end=date(2026, 7, 31),
            value=10,
        ),
        _snapshot(
            source_id="follow-up",
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
            value=12,
        ),
        implementation_date=date(2026, 7, 1),
        observed_at=date(2026, 9, 1),
    )
    payload = result.model_dump(mode="json")
    payload["observed_at"] = "2026-08-15"

    with pytest.raises(ValidationError, match="after observation"):
        ValidationComparison.model_validate(payload)


def test_compare_visibility_rejects_project_identity_mismatch() -> None:
    baseline = _snapshot(
        source_id="baseline",
        start=date(2026, 7, 1),
        end=date(2026, 7, 31),
        value=10,
    )
    follow_up = _snapshot(
        source_id="follow-up",
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        value=12,
    ).model_copy(update={"project_id": "other-project"})

    with pytest.raises(ValueError, match="project identity"):
        compare_visibility(baseline, follow_up)


def test_compare_visibility_rejects_domain_identity_mismatch() -> None:
    baseline = _snapshot(
        source_id="baseline",
        start=date(2026, 7, 1),
        end=date(2026, 7, 31),
        value=10,
    )
    follow_up = _snapshot(
        source_id="follow-up",
        start=date(2026, 8, 1),
        end=date(2026, 8, 31),
        value=12,
    ).model_copy(update={"canonical_domain": "other.example"})

    with pytest.raises(ValueError, match="domain identity"):
        compare_visibility(baseline, follow_up)
