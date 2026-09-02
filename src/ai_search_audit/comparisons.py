"""Deterministic, like-for-like validation comparisons.

The comparison layer reports chronology and measurement compatibility. It does
not infer that implementation caused an observed change.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import re
from datetime import date, timedelta
from enum import StrEnum
from typing import Literal, NoReturn
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import AIObservation, AuditRun, DataState, Finding
from .visibility_metrics import MetricWindow, VisibilityMetric, VisibilitySnapshot

VALIDATION_COMPARISON_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
VALIDATION_CHRONOLOGY_STATEMENT: Literal[
    "Observed after implementation; causality is not established."
] = "Observed after implementation; causality is not established."
EARLY_VALIDATION_WARNING = "Validation was run before the default 90-day target."


class _FrozenComparisonModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        allow_inf_nan=False,
    )


class ComparisonCausality(StrEnum):
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class ValidationTimingState(StrEnum):
    EARLY = "EARLY"
    TARGET_REACHED = "TARGET_REACHED"


class ComparisonBasis(StrEnum):
    DIRECT = "DIRECT"
    YEAR_OVER_YEAR = "YEAR_OVER_YEAR"
    NON_COMPARABLE = "NON_COMPARABLE"


class FindingChange(StrEnum):
    ADDED = "ADDED"
    RESOLVED = "RESOLVED"
    UNCHANGED = "UNCHANGED"


class ComparisonLimitation(StrEnum):
    MISSING_BASELINE = "MISSING_BASELINE"
    MISSING_FOLLOW_UP = "MISSING_FOLLOW_UP"
    NON_NUMERIC_STATE = "NON_NUMERIC_STATE"
    METRIC_DEFINITION_MISMATCH = "METRIC_DEFINITION_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    DEFINITION_FILTER_MISMATCH = "DEFINITION_FILTER_MISMATCH"
    SEGMENT_MISMATCH = "SEGMENT_MISMATCH"
    WINDOW_MISMATCH = "WINDOW_MISMATCH"
    INVALID_WINDOW_ORDER = "INVALID_WINDOW_ORDER"
    NON_POST_IMPLEMENTATION_WINDOW = "NON_POST_IMPLEMENTATION_WINDOW"
    WINDOW_AFTER_OBSERVATION = "WINDOW_AFTER_OBSERVATION"
    PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
    ZERO_BASELINE = "ZERO_BASELINE"
    PROMPT_PACK_MISMATCH = "PROMPT_PACK_MISMATCH"
    PROMPT_SET_MISMATCH = "PROMPT_SET_MISMATCH"
    OBSERVATION_SETUP_MISMATCH = "OBSERVATION_SETUP_MISMATCH"
    GROUNDED_OBSERVATIONS_UNAVAILABLE = "GROUNDED_OBSERVATIONS_UNAVAILABLE"


class ComparisonWindow(_FrozenComparisonModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> ComparisonWindow:
        if self.end < self.start:
            raise ValueError("comparison window end must not precede start")
        return self


class MetricComparison(_FrozenComparisonModel):
    metric_id: str
    metric: str
    unit: str
    state: DataState
    basis: ComparisonBasis
    baseline_state: DataState
    follow_up_state: DataState
    baseline_value: float | None = None
    follow_up_value: float | None = None
    absolute_delta: float | None = None
    relative_delta_percent: float | None = None
    baseline_window: ComparisonWindow | None = None
    follow_up_window: ComparisonWindow | None = None
    baseline_source_ids: tuple[str, ...] = ()
    follow_up_source_ids: tuple[str, ...] = ()
    coverage: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    limitations: tuple[ComparisonLimitation, ...] = ()

    @model_validator(mode="after")
    def validate_delta_state(self) -> MetricComparison:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if self.state not in {DataState.AVAILABLE, DataState.PARTIAL, DataState.UNKNOWN}:
            raise ValueError("metric comparison state must be AVAILABLE, PARTIAL, or UNKNOWN")
        if numeric and (self.baseline_value is None or self.follow_up_value is None):
            raise ValueError("numeric comparison requires both observed values")
        if self.state is DataState.UNKNOWN and self.basis is not ComparisonBasis.NON_COMPARABLE:
            raise ValueError("UNKNOWN metric comparison must be NON_COMPARABLE")
        if self.state is DataState.UNKNOWN and not self.limitations:
            raise ValueError("UNKNOWN metric comparison requires a limitation")
        if not numeric and any(
            value is not None
            for value in (
                self.absolute_delta,
                self.relative_delta_percent,
            )
        ):
            raise ValueError("non-numeric comparison must not contain a delta")
        if self.basis is ComparisonBasis.NON_COMPARABLE and not self.limitations:
            raise ValueError("non-comparable metric requires a limitation")
        if numeric:
            if self.basis is ComparisonBasis.NON_COMPARABLE:
                raise ValueError("numeric delta must use a comparable basis")
            numeric_states = {DataState.AVAILABLE, DataState.PARTIAL}
            if (
                self.baseline_state not in numeric_states
                or self.follow_up_state not in numeric_states
            ):
                raise ValueError("numeric delta requires numeric source states")
            expected_state = (
                DataState.AVAILABLE
                if self.baseline_state is self.follow_up_state is DataState.AVAILABLE
                else DataState.PARTIAL
            )
            if self.state is not expected_state:
                raise ValueError("comparison state does not match source states")
            if self.absolute_delta is None:
                raise ValueError("numeric comparison requires an absolute delta")
            if self.baseline_window is None or self.follow_up_window is None:
                raise ValueError("numeric comparison requires both measurement windows")
            if self.baseline_window.end >= self.follow_up_window.start:
                raise ValueError("numeric comparison requires ordered, non-overlapping windows")
            baseline_complete_month = (
                self.baseline_window.start.day == 1
                and self.baseline_window.end.day
                == calendar.monthrange(
                    self.baseline_window.end.year,
                    self.baseline_window.end.month,
                )[1]
            )
            follow_up_complete_month = (
                self.follow_up_window.start.day == 1
                and self.follow_up_window.end.day
                == calendar.monthrange(
                    self.follow_up_window.end.year,
                    self.follow_up_window.end.month,
                )[1]
            )
            if baseline_complete_month and follow_up_complete_month:
                baseline_span = (
                    (self.baseline_window.end.year - self.baseline_window.start.year) * 12
                    + self.baseline_window.end.month
                    - self.baseline_window.start.month
                )
                follow_up_span = (
                    (self.follow_up_window.end.year - self.follow_up_window.start.year) * 12
                    + self.follow_up_window.end.month
                    - self.follow_up_window.start.month
                )
                windows_equivalent = baseline_span == follow_up_span
            else:
                windows_equivalent = (self.baseline_window.end - self.baseline_window.start) == (
                    self.follow_up_window.end - self.follow_up_window.start
                )
            if self.basis is ComparisonBasis.YEAR_OVER_YEAR:
                windows_equivalent = windows_equivalent and (
                    self.baseline_window.start.month,
                    self.baseline_window.start.day,
                    self.baseline_window.end.month,
                    self.baseline_window.end.day,
                ) == (
                    self.follow_up_window.start.month,
                    self.follow_up_window.start.day,
                    self.follow_up_window.end.month,
                    self.follow_up_window.end.day,
                )
            if not windows_equivalent:
                raise ValueError("numeric comparison requires equivalent windows")
            for label, source_ids in (
                ("baseline", self.baseline_source_ids),
                ("follow-up", self.follow_up_source_ids),
            ):
                if not source_ids or len(source_ids) != len(set(source_ids)):
                    raise ValueError(f"numeric comparison requires unique {label} source refs")
            if self.coverage <= 0:
                raise ValueError("numeric comparison requires positive coverage")
            if self.confidence <= 0:
                raise ValueError("numeric comparison requires positive confidence")
            if self.state is DataState.AVAILABLE and self.coverage != 1:
                raise ValueError("AVAILABLE comparison requires full coverage")
            if self.state is DataState.PARTIAL and self.coverage >= 1:
                raise ValueError("PARTIAL comparison requires coverage below one")
            baseline_value = self.baseline_value
            follow_up_value = self.follow_up_value
            if baseline_value is None or follow_up_value is None:
                raise ValueError("numeric comparison requires both observed values")
            expected_delta = follow_up_value - baseline_value
            if abs(self.absolute_delta - expected_delta) > 1e-9:
                raise ValueError("absolute delta does not match observed values")
            expected_relative = (
                None if baseline_value == 0 else expected_delta / abs(baseline_value) * 100
            )
            if expected_relative is None and self.relative_delta_percent is not None:
                raise ValueError("relative delta is undefined for a zero baseline")
            if expected_relative is not None and (
                self.relative_delta_percent is None
                or abs(self.relative_delta_percent - expected_relative) > 1e-9
            ):
                raise ValueError("relative delta does not match observed values")
            expected_limitations: set[ComparisonLimitation] = set()
            if self.state is DataState.PARTIAL:
                expected_limitations.add(ComparisonLimitation.PARTIAL_COVERAGE)
            if baseline_value == 0:
                expected_limitations.add(ComparisonLimitation.ZERO_BASELINE)
            if set(self.limitations) != expected_limitations or len(self.limitations) != len(
                expected_limitations
            ):
                raise ValueError("numeric comparison limitations do not match derived state")
        return self


class FindingComparison(_FrozenComparisonModel):
    stable_identity: str
    rule_id: str
    fact_identity: str
    change: FindingChange
    baseline_finding_ids: tuple[str, ...] = ()
    follow_up_finding_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_change(self) -> FindingComparison:
        if not self.baseline_finding_ids and not self.follow_up_finding_ids:
            raise ValueError("finding comparison requires a finding reference")
        expected = (
            FindingChange.UNCHANGED
            if self.baseline_finding_ids and self.follow_up_finding_ids
            else FindingChange.RESOLVED
            if self.baseline_finding_ids
            else FindingChange.ADDED
        )
        if self.change is not expected:
            raise ValueError("finding change does not match its references")
        return self


class AIVisibilityComparison(_FrozenComparisonModel):
    state: DataState
    baseline_value: float | None = None
    follow_up_value: float | None = None
    absolute_delta: float | None = None
    prompt_pack_version: str | None = Field(default=None, min_length=1)
    setup_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    baseline_observation_ids: tuple[str, ...] = ()
    follow_up_observation_ids: tuple[str, ...] = ()
    baseline_prompt_ids: tuple[str, ...] = ()
    follow_up_prompt_ids: tuple[str, ...] = ()
    baseline_measurable_prompt_ids: tuple[str, ...] = ()
    follow_up_measurable_prompt_ids: tuple[str, ...] = ()
    baseline_canonical_prompt_ids: tuple[str, ...] = ()
    follow_up_canonical_prompt_ids: tuple[str, ...] = ()
    limitations: tuple[ComparisonLimitation, ...] = ()

    @model_validator(mode="after")
    def validate_delta(self) -> AIVisibilityComparison:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if self.state not in {DataState.AVAILABLE, DataState.PARTIAL, DataState.UNKNOWN}:
            raise ValueError("AI comparison state must be AVAILABLE, PARTIAL, or UNKNOWN")
        if numeric:
            if (
                self.baseline_value is None
                or self.follow_up_value is None
                or self.absolute_delta is None
                or self.prompt_pack_version is None
                or self.setup_fingerprint is None
            ):
                raise ValueError("numeric AI comparison requires reproducible observations")
            for label, references in (
                ("baseline observation", self.baseline_observation_ids),
                ("follow-up observation", self.follow_up_observation_ids),
                ("baseline prompt", self.baseline_prompt_ids),
                ("follow-up prompt", self.follow_up_prompt_ids),
                ("baseline measurable prompt", self.baseline_measurable_prompt_ids),
                ("follow-up measurable prompt", self.follow_up_measurable_prompt_ids),
                ("baseline canonical prompt", self.baseline_canonical_prompt_ids),
                ("follow-up canonical prompt", self.follow_up_canonical_prompt_ids),
            ):
                if not references:
                    raise ValueError(f"numeric AI comparison requires nonempty {label} refs")
                if len(references) != len(set(references)):
                    raise ValueError(f"numeric AI comparison requires unique {label} refs")
            if set(self.baseline_prompt_ids) != set(self.follow_up_prompt_ids):
                raise ValueError("numeric AI comparison requires matching full prompt refs")
            if set(self.baseline_canonical_prompt_ids) != set(self.follow_up_canonical_prompt_ids):
                raise ValueError("numeric AI comparison requires matching canonical prompt refs")
            if set(self.baseline_measurable_prompt_ids) != set(
                self.follow_up_measurable_prompt_ids
            ):
                raise ValueError("numeric AI comparison requires matching measurable prompt refs")
            if not set(self.baseline_measurable_prompt_ids).issubset(
                self.baseline_prompt_ids
            ) or not set(self.follow_up_measurable_prompt_ids).issubset(self.follow_up_prompt_ids):
                raise ValueError("numeric AI measurable prompt refs must resolve to full refs")
            if not set(self.baseline_prompt_ids).issubset(
                self.baseline_canonical_prompt_ids
            ) or not set(self.follow_up_prompt_ids).issubset(self.follow_up_canonical_prompt_ids):
                raise ValueError("numeric AI observed prompt refs must resolve to canonical refs")
            if len(self.baseline_observation_ids) != len(
                self.baseline_measurable_prompt_ids
            ) or len(self.follow_up_observation_ids) != len(self.follow_up_measurable_prompt_ids):
                raise ValueError("numeric AI observation refs must resolve one-to-one to prompts")
            complete = set(self.baseline_prompt_ids) == set(
                self.baseline_canonical_prompt_ids
            ) == set(self.baseline_measurable_prompt_ids) and set(self.follow_up_prompt_ids) == set(
                self.follow_up_canonical_prompt_ids
            ) == set(self.follow_up_measurable_prompt_ids)
            expected_state = DataState.AVAILABLE if complete else DataState.PARTIAL
            if self.state is not expected_state:
                raise ValueError("AI comparison state does not match prompt coverage")
            if self.limitations:
                raise ValueError("numeric AI comparison requires empty limitations")
            if abs(self.absolute_delta - (self.follow_up_value - self.baseline_value)) > 1e-9:
                raise ValueError("AI visibility delta does not match observed values")
        elif any(
            value is not None
            for value in (self.baseline_value, self.follow_up_value, self.absolute_delta)
        ):
            raise ValueError("non-numeric AI comparison must not contain a delta")
        if self.state is DataState.UNKNOWN and not self.limitations:
            raise ValueError("UNKNOWN AI comparison requires a limitation")
        return self


class ValidationComparison(_FrozenComparisonModel):
    schema_version: Literal["1.0.0"] = VALIDATION_COMPARISON_SCHEMA_VERSION
    baseline_audit_id: str | None = None
    follow_up_audit_id: str | None = None
    implementation_date: date | None = None
    validation_target_date: date | None = None
    observed_at: date | None = None
    timing_state: ValidationTimingState | None = None
    timing_warning: str | None = None
    causality: ComparisonCausality = ComparisonCausality.NOT_ESTABLISHED
    chronology_statement: Literal[
        "Observed after implementation; causality is not established."
    ] = VALIDATION_CHRONOLOGY_STATEMENT
    metrics: tuple[MetricComparison, ...] = ()
    findings: tuple[FindingComparison, ...] = ()
    ai_visibility: AIVisibilityComparison | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> ValidationComparison:
        dates = (
            self.implementation_date,
            self.validation_target_date,
            self.observed_at,
        )
        if any(value is not None for value in dates) and any(value is None for value in dates):
            raise ValueError("validation timing dates must be supplied together")
        if all(value is None for value in dates):
            if self.timing_state is not None or self.timing_warning is not None:
                raise ValueError("dateless validation must not contain timing state or warning")
        if self.implementation_date is not None:
            if self.observed_at < self.implementation_date:  # type: ignore[operator]
                raise ValueError("validation observation cannot be before implementation")
            if self.validation_target_date != validation_target_date(self.implementation_date):
                raise ValueError("validation target must be 90 days after implementation")
            expected = (
                ValidationTimingState.EARLY
                if self.observed_at < self.validation_target_date  # type: ignore[operator]
                else ValidationTimingState.TARGET_REACHED
            )
            if self.timing_state is not expected:
                raise ValueError("validation timing state does not match trusted dates")
            if any(
                metric.state in {DataState.AVAILABLE, DataState.PARTIAL}
                and (
                    metric.follow_up_window is None
                    or metric.follow_up_window.start < self.implementation_date
                )
                for metric in self.metrics
            ):
                raise ValueError(
                    "numeric validation metrics require a post-implementation follow-up window"
                )
            if any(
                metric.state in {DataState.AVAILABLE, DataState.PARTIAL}
                and (
                    metric.follow_up_window is None
                    or metric.follow_up_window.end > self.observed_at  # type: ignore[operator]
                )
                for metric in self.metrics
            ):
                raise ValueError("numeric validation metric window cannot end after observation")
            if expected is ValidationTimingState.EARLY and (
                self.timing_warning != EARLY_VALIDATION_WARNING
            ):
                raise ValueError("early validation requires the fixed system warning")
            if expected is ValidationTimingState.TARGET_REACHED and self.timing_warning is not None:
                raise ValueError("TARGET_REACHED validation must not contain a timing warning")
        metric_ids = [item.metric_id for item in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("validation comparison requires unique metric IDs")
        finding_ids = [item.stable_identity for item in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("validation comparison requires unique finding identities")
        return self


def validation_target_date(implementation_date: date) -> date:
    """Return the default follow-up target, exactly 90 calendar days later."""
    return implementation_date + timedelta(days=90)


def _comparison_window(window: MetricWindow | None) -> ComparisonWindow | None:
    if window is None:
        return None
    return ComparisonWindow(start=window.start, end=window.end)


def _same_window_shape(
    baseline: MetricWindow | None,
    follow_up: MetricWindow | None,
    *,
    seasonal: bool,
    monthly: bool,
) -> tuple[bool, ComparisonBasis]:
    if baseline is None or follow_up is None:
        return False, ComparisonBasis.NON_COMPARABLE
    if monthly:
        baseline_complete = (
            baseline.start.day == 1
            and baseline.end.day == calendar.monthrange(baseline.end.year, baseline.end.month)[1]
        )
        follow_up_complete = (
            follow_up.start.day == 1
            and follow_up.end.day == calendar.monthrange(follow_up.end.year, follow_up.end.month)[1]
        )
        baseline_span = (baseline.end.year - baseline.start.year) * 12 + (
            baseline.end.month - baseline.start.month
        )
        follow_up_span = (follow_up.end.year - follow_up.start.year) * 12 + (
            follow_up.end.month - follow_up.start.month
        )
        if not baseline_complete or not follow_up_complete or baseline_span != follow_up_span:
            return False, ComparisonBasis.NON_COMPARABLE
    elif (baseline.end - baseline.start).days != (follow_up.end - follow_up.start).days:
        return False, ComparisonBasis.NON_COMPARABLE
    if not seasonal:
        return True, ComparisonBasis.DIRECT
    matching_calendar = (
        baseline.start.month,
        baseline.start.day,
        baseline.end.month,
        baseline.end.day,
    ) == (
        follow_up.start.month,
        follow_up.start.day,
        follow_up.end.month,
        follow_up.end.day,
    )
    if matching_calendar and follow_up.start.year > baseline.start.year:
        return True, ComparisonBasis.YEAR_OVER_YEAR
    return False, ComparisonBasis.NON_COMPARABLE


def _metric_comparison(
    baseline: VisibilityMetric | None,
    follow_up: VisibilityMetric | None,
    *,
    seasonal: bool,
    implementation_date: date | None,
    observed_at: date | None,
) -> MetricComparison:
    present = follow_up or baseline
    if present is None:  # pragma: no cover - caller never passes two missing metrics
        raise ValueError("comparison requires at least one metric")
    limitations: list[ComparisonLimitation] = []
    if baseline is None:
        limitations.append(ComparisonLimitation.MISSING_BASELINE)
    if follow_up is None:
        limitations.append(ComparisonLimitation.MISSING_FOLLOW_UP)
    if baseline is None or follow_up is None:
        return MetricComparison(
            metric_id=present.metric_id,
            metric=present.metric,
            unit=present.unit,
            state=DataState.UNKNOWN,
            basis=ComparisonBasis.NON_COMPARABLE,
            baseline_state=baseline.state if baseline else DataState.UNAVAILABLE,
            follow_up_state=follow_up.state if follow_up else DataState.UNAVAILABLE,
            baseline_value=baseline.value if baseline else None,
            follow_up_value=follow_up.value if follow_up else None,
            baseline_window=_comparison_window(baseline.window) if baseline else None,
            follow_up_window=_comparison_window(follow_up.window) if follow_up else None,
            baseline_source_ids=baseline.source_ids if baseline else (),
            follow_up_source_ids=follow_up.source_ids if follow_up else (),
            coverage=0,
            confidence=0,
            limitations=tuple(limitations),
        )

    numeric_states = {DataState.AVAILABLE, DataState.PARTIAL}
    if baseline.state not in numeric_states or follow_up.state not in numeric_states:
        limitations.append(ComparisonLimitation.NON_NUMERIC_STATE)
    if baseline.metric != follow_up.metric:
        limitations.append(ComparisonLimitation.METRIC_DEFINITION_MISMATCH)
    if baseline.unit != follow_up.unit:
        limitations.append(ComparisonLimitation.UNIT_MISMATCH)
    if baseline.definitions != follow_up.definitions:
        limitations.append(ComparisonLimitation.DEFINITION_FILTER_MISMATCH)
    if baseline.segments != follow_up.segments:
        limitations.append(ComparisonLimitation.SEGMENT_MISMATCH)
    windows_match, basis = _same_window_shape(
        baseline.window,
        follow_up.window,
        seasonal=seasonal,
        monthly="cadence=monthly" in baseline.definitions,
    )
    if not windows_match:
        limitations.append(ComparisonLimitation.WINDOW_MISMATCH)
    if (
        baseline.window is not None
        and follow_up.window is not None
        and baseline.window.end >= follow_up.window.start
    ):
        limitations.append(ComparisonLimitation.INVALID_WINDOW_ORDER)
    if (
        implementation_date is not None
        and follow_up.window is not None
        and follow_up.window.start < implementation_date
    ):
        limitations.append(ComparisonLimitation.NON_POST_IMPLEMENTATION_WINDOW)
    if (
        observed_at is not None
        and follow_up.window is not None
        and follow_up.window.end > observed_at
    ):
        limitations.append(ComparisonLimitation.WINDOW_AFTER_OBSERVATION)
    comparable = not limitations
    if not comparable:
        return MetricComparison(
            metric_id=present.metric_id,
            metric=present.metric,
            unit=present.unit,
            state=DataState.UNKNOWN,
            basis=ComparisonBasis.NON_COMPARABLE,
            baseline_state=baseline.state,
            follow_up_state=follow_up.state,
            baseline_value=baseline.value,
            follow_up_value=follow_up.value,
            baseline_window=_comparison_window(baseline.window),
            follow_up_window=_comparison_window(follow_up.window),
            baseline_source_ids=baseline.source_ids,
            follow_up_source_ids=follow_up.source_ids,
            coverage=min(baseline.coverage, follow_up.coverage),
            confidence=min(baseline.confidence, follow_up.confidence),
            limitations=tuple(limitations),
        )

    baseline_value = baseline.value
    follow_up_value = follow_up.value
    if baseline_value is None or follow_up_value is None:  # guarded by metric state
        raise ValueError("numeric metric is missing its value")
    absolute_delta = follow_up_value - baseline_value
    relative_delta = None if baseline_value == 0 else absolute_delta / abs(baseline_value) * 100
    state = (
        DataState.AVAILABLE
        if baseline.state is follow_up.state is DataState.AVAILABLE
        else DataState.PARTIAL
    )
    if state is DataState.PARTIAL:
        limitations.append(ComparisonLimitation.PARTIAL_COVERAGE)
    if relative_delta is None:
        limitations.append(ComparisonLimitation.ZERO_BASELINE)
    return MetricComparison(
        metric_id=present.metric_id,
        metric=present.metric,
        unit=present.unit,
        state=state,
        basis=basis,
        baseline_state=baseline.state,
        follow_up_state=follow_up.state,
        baseline_value=baseline_value,
        follow_up_value=follow_up_value,
        absolute_delta=absolute_delta,
        relative_delta_percent=relative_delta,
        baseline_window=_comparison_window(baseline.window),
        follow_up_window=_comparison_window(follow_up.window),
        baseline_source_ids=baseline.source_ids,
        follow_up_source_ids=follow_up.source_ids,
        coverage=min(baseline.coverage, follow_up.coverage),
        confidence=min(baseline.confidence, follow_up.confidence),
        limitations=tuple(limitations),
    )


def compare_visibility(
    baseline: VisibilitySnapshot | None,
    follow_up: VisibilitySnapshot | None,
    *,
    baseline_audit_id: str | None = None,
    follow_up_audit_id: str | None = None,
    implementation_date: date | None = None,
    observed_at: date | None = None,
    seasonal: bool = False,
) -> ValidationComparison:
    """Compare visibility aggregates only when their semantics are equivalent."""
    if baseline is not None and follow_up is not None:
        if baseline.project_id != follow_up.project_id:
            raise ValueError("visibility snapshot project identity mismatch")
        if baseline.canonical_domain != follow_up.canonical_domain:
            raise ValueError("visibility snapshot domain identity mismatch")
    baseline_metrics = (
        {} if baseline is None else {item.metric_id: item for item in baseline.metrics}
    )
    follow_up_metrics = (
        {} if follow_up is None else {item.metric_id: item for item in follow_up.metrics}
    )
    metric_ids = sorted(set(baseline_metrics) | set(follow_up_metrics))
    comparisons = tuple(
        _metric_comparison(
            baseline_metrics.get(metric_id),
            follow_up_metrics.get(metric_id),
            seasonal=seasonal,
            implementation_date=implementation_date,
            observed_at=observed_at,
        )
        for metric_id in metric_ids
    )
    target = None
    timing_state = None
    warning = None
    if implementation_date is not None:
        if observed_at is None:
            raise ValueError("observed_at is required with implementation_date")
        if observed_at < implementation_date:
            raise ValueError("validation observation cannot be before implementation")
        target = validation_target_date(implementation_date)
        timing_state = (
            ValidationTimingState.EARLY
            if observed_at < target
            else ValidationTimingState.TARGET_REACHED
        )
        if timing_state is ValidationTimingState.EARLY:
            warning = EARLY_VALIDATION_WARNING
    elif observed_at is not None:
        raise ValueError("implementation_date is required with observed_at")
    return ValidationComparison(
        baseline_audit_id=baseline_audit_id,
        follow_up_audit_id=follow_up_audit_id,
        implementation_date=implementation_date,
        validation_target_date=target,
        observed_at=observed_at,
        timing_state=timing_state,
        timing_warning=warning,
        metrics=comparisons,
    )


def _normalize_affected_url(value: str) -> str:
    value = value.strip()
    parts = urlsplit(value)
    if not parts.scheme or not parts.hostname:
        return value
    scheme = parts.scheme.casefold()
    host = parts.hostname.casefold()
    port = parts.port
    netloc = (
        host
        if port is None or (scheme, port) in {("http", 80), ("https", 443)}
        else f"{host}:{port}"
    )
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _fact_identity(finding: Finding) -> str:
    claims = sorted(
        (
            claim.predicate,
            json.dumps(claim.value, ensure_ascii=False, sort_keys=True),
            claim.modality.value,
            claim.negated,
        )
        for claim in finding.factual_claims
    )
    payload = {
        "claims": claims,
        "affected_urls": sorted({_normalize_affected_url(url) for url in finding.affected_urls}),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_findings(
    baseline: tuple[Finding, ...] | list[Finding],
    follow_up: tuple[Finding, ...] | list[Finding],
) -> tuple[FindingComparison, ...]:
    """Compare crawl findings by stable rule and factual identity, never prose."""
    baseline_by_key: dict[tuple[str, str], list[Finding]] = {}
    follow_up_by_key: dict[tuple[str, str], list[Finding]] = {}
    for collection, destination in (
        (baseline, baseline_by_key),
        (follow_up, follow_up_by_key),
    ):
        for finding in collection:
            key = (finding.rule_id, _fact_identity(finding))
            destination.setdefault(key, []).append(finding)
    results: list[FindingComparison] = []
    for rule_id, fact_identity in sorted(set(baseline_by_key) | set(follow_up_by_key)):
        before = baseline_by_key.get((rule_id, fact_identity), [])
        after = follow_up_by_key.get((rule_id, fact_identity), [])
        change = (
            FindingChange.UNCHANGED
            if before and after
            else FindingChange.RESOLVED
            if before
            else FindingChange.ADDED
        )
        stable_identity = hashlib.sha256(f"{rule_id}:{fact_identity}".encode()).hexdigest()
        results.append(
            FindingComparison(
                stable_identity=stable_identity,
                rule_id=rule_id,
                fact_identity=fact_identity,
                change=change,
                baseline_finding_ids=tuple(sorted(item.finding_id for item in before)),
                follow_up_finding_ids=tuple(sorted(item.finding_id for item in after)),
            )
        )
    return tuple(results)


def validate_follow_up_canonical_binding(
    comparison: ValidationComparison,
    *,
    audit: AuditRun,
    visibility_snapshot: VisibilitySnapshot | None,
    prompt_pack_version: str | None,
    setup_fingerprint: str | None,
    canonical_prompt_ids: tuple[str, ...],
) -> None:
    """Revalidate a persisted comparison against its canonical follow-up audit."""

    def reject(detail: str) -> NoReturn:
        raise ValueError(f"canonical validation comparison {detail}")

    if audit.configuration.get("implementation_date") != (
        comparison.implementation_date.isoformat()
        if comparison.implementation_date is not None
        else None
    ):
        reject("implementation date mismatch")
    if audit.configuration.get("validation_comparison_schema_version") != (
        comparison.schema_version
    ):
        reject("schema version mismatch")
    if comparison.observed_at != audit.timestamp.date():
        reject("observation date mismatch")

    follow_up_metrics = () if visibility_snapshot is None else visibility_snapshot.metrics
    canonical_metrics = {item.metric_id: item for item in follow_up_metrics}
    observed_metrics = {item.metric_id: item for item in comparison.metrics}
    if not set(canonical_metrics).issubset(observed_metrics):
        reject("does not cover every follow-up metric")
    for metric_id, item in observed_metrics.items():
        canonical = canonical_metrics.get(metric_id)
        if canonical is None:
            if not (
                item.follow_up_state is DataState.UNAVAILABLE
                and item.follow_up_value is None
                and item.follow_up_window is None
                and not item.follow_up_source_ids
                and ComparisonLimitation.MISSING_FOLLOW_UP in item.limitations
            ):
                reject(f"metric {metric_id} invents a follow-up observation")
            continue
        expected_window = _comparison_window(canonical.window)
        if (
            item.metric != canonical.metric
            or item.unit != canonical.unit
            or item.follow_up_state is not canonical.state
            or item.follow_up_value != canonical.value
            or item.follow_up_window != expected_window
            or item.follow_up_source_ids != canonical.source_ids
            or item.coverage > canonical.coverage
            or item.confidence > canonical.confidence
        ):
            reject(f"metric {metric_id} differs from the follow-up snapshot")

    expected_findings = {
        item.stable_identity: item for item in compare_findings((), audit.findings)
    }
    observed_follow_findings = {
        item.stable_identity: item for item in comparison.findings if item.follow_up_finding_ids
    }
    if set(observed_follow_findings) != set(expected_findings):
        reject("finding identities do not cover the follow-up audit")
    for identity, expected in expected_findings.items():
        observed = observed_follow_findings[identity]
        if (
            observed.rule_id != expected.rule_id
            or observed.fact_identity != expected.fact_identity
            or observed.follow_up_finding_ids != expected.follow_up_finding_ids
        ):
            reject(f"finding {identity} differs from the follow-up audit")

    ai = comparison.ai_visibility
    if ai is None:
        reject("is missing the AI follow-up projection")
    measurable = tuple(
        item
        for item in audit.ai_observations
        if item.grounded is True and item.brand_mentioned is not None
    )
    expected_observation_ids = tuple(item.observation_id for item in measurable)
    expected_full_prompt_ids = tuple(sorted(item.prompt_id for item in audit.ai_observations))
    expected_measurable_prompt_ids = tuple(sorted(item.prompt_id for item in measurable))
    expected_canonical_prompt_ids = tuple(sorted(canonical_prompt_ids))
    if (
        ai.follow_up_observation_ids != expected_observation_ids
        or ai.follow_up_prompt_ids != expected_full_prompt_ids
        or ai.follow_up_measurable_prompt_ids != expected_measurable_prompt_ids
        or ai.follow_up_canonical_prompt_ids != expected_canonical_prompt_ids
    ):
        reject("AI observation or prompt references differ from the follow-up audit")
    if ai.prompt_pack_version not in {prompt_pack_version, None}:
        reject("AI prompt-pack version differs from the follow-up audit")
    if ai.prompt_pack_version is None and (
        ComparisonLimitation.PROMPT_PACK_MISMATCH not in ai.limitations
    ):
        reject("AI prompt-pack mismatch is not explicit")
    if ai.setup_fingerprint not in {setup_fingerprint, None}:
        reject("AI observation setup differs from the follow-up audit")
    if (
        ai.setup_fingerprint is None
        and setup_fingerprint is not None
        and (ComparisonLimitation.OBSERVATION_SETUP_MISMATCH not in ai.limitations)
    ):
        reject("AI observation setup mismatch is not explicit")
    numeric = ai.state in {DataState.AVAILABLE, DataState.PARTIAL}
    follow_up_can_be_numeric = (
        prompt_pack_version is not None
        and setup_fingerprint is not None
        and bool(measurable)
        and len(expected_full_prompt_ids) == len(set(expected_full_prompt_ids))
        and len(expected_measurable_prompt_ids) == len(set(expected_measurable_prompt_ids))
        and set(expected_full_prompt_ids).issubset(expected_canonical_prompt_ids)
        and set(expected_measurable_prompt_ids).issubset(expected_full_prompt_ids)
    )
    if numeric and not follow_up_can_be_numeric:
        reject("AI numeric state is unavailable from the follow-up audit")
    if numeric:
        expected_value = (
            sum(bool(item.brand_mentioned) for item in measurable) / len(measurable) * 100
        )
        if ai.follow_up_value != expected_value:
            reject("AI follow-up value differs from the follow-up audit")
        complete = (
            set(expected_full_prompt_ids)
            == set(expected_measurable_prompt_ids)
            == set(expected_canonical_prompt_ids)
        )
        expected_state = DataState.AVAILABLE if complete else DataState.PARTIAL
        if ai.state is not expected_state:
            reject("AI state differs from canonical follow-up prompt coverage")


def compare_observed_ai_visibility(
    baseline: tuple[AIObservation, ...] | list[AIObservation],
    follow_up: tuple[AIObservation, ...] | list[AIObservation],
    *,
    baseline_prompt_pack_version: str | None,
    follow_up_prompt_pack_version: str | None,
    baseline_setup_fingerprint: str | None,
    follow_up_setup_fingerprint: str | None,
    baseline_canonical_prompt_ids: tuple[str, ...] | None = None,
    follow_up_canonical_prompt_ids: tuple[str, ...] | None = None,
) -> AIVisibilityComparison:
    """Compare grounded observations only under one reproducible observation setup."""
    limitations: list[ComparisonLimitation] = []
    if (
        baseline_prompt_pack_version is None
        or baseline_prompt_pack_version != follow_up_prompt_pack_version
    ):
        limitations.append(ComparisonLimitation.PROMPT_PACK_MISMATCH)
    matching_setup = (
        baseline_setup_fingerprint is not None
        and baseline_setup_fingerprint == follow_up_setup_fingerprint
        and re.fullmatch(r"[0-9a-f]{64}", baseline_setup_fingerprint) is not None
    )
    if not matching_setup:
        limitations.append(ComparisonLimitation.OBSERVATION_SETUP_MISMATCH)
    baseline_measurable = [
        item for item in baseline if item.brand_mentioned is not None and item.grounded is True
    ]
    follow_up_measurable = [
        item for item in follow_up if item.brand_mentioned is not None and item.grounded is True
    ]
    baseline_prompt_ids = tuple(sorted(item.prompt_id for item in baseline))
    follow_up_prompt_ids = tuple(sorted(item.prompt_id for item in follow_up))
    baseline_measurable_prompt_ids = tuple(sorted(item.prompt_id for item in baseline_measurable))
    follow_up_measurable_prompt_ids = tuple(sorted(item.prompt_id for item in follow_up_measurable))
    if (
        baseline_prompt_ids != follow_up_prompt_ids
        or len(baseline_prompt_ids) != len(set(baseline_prompt_ids))
        or len(follow_up_prompt_ids) != len(set(follow_up_prompt_ids))
        or baseline_measurable_prompt_ids != follow_up_measurable_prompt_ids
        or len(baseline_measurable_prompt_ids) != len(set(baseline_measurable_prompt_ids))
        or len(follow_up_measurable_prompt_ids) != len(set(follow_up_measurable_prompt_ids))
    ):
        limitations.append(ComparisonLimitation.PROMPT_SET_MISMATCH)
    baseline_canonical = tuple(sorted(baseline_canonical_prompt_ids or ()))
    follow_up_canonical = tuple(sorted(follow_up_canonical_prompt_ids or ()))
    if (
        not baseline_canonical
        or baseline_canonical != follow_up_canonical
        or len(baseline_canonical) != len(set(baseline_canonical))
        or len(follow_up_canonical) != len(set(follow_up_canonical))
        or not set(baseline_prompt_ids).issubset(baseline_canonical)
        or not set(follow_up_prompt_ids).issubset(follow_up_canonical)
    ):
        limitations.append(ComparisonLimitation.PROMPT_SET_MISMATCH)
    if not baseline_measurable or not follow_up_measurable:
        limitations.append(ComparisonLimitation.GROUNDED_OBSERVATIONS_UNAVAILABLE)
    if limitations:
        return AIVisibilityComparison(
            state=DataState.UNKNOWN,
            prompt_pack_version=(
                baseline_prompt_pack_version
                if baseline_prompt_pack_version == follow_up_prompt_pack_version
                else None
            ),
            setup_fingerprint=(baseline_setup_fingerprint if matching_setup else None),
            baseline_observation_ids=tuple(item.observation_id for item in baseline_measurable),
            follow_up_observation_ids=tuple(item.observation_id for item in follow_up_measurable),
            baseline_prompt_ids=baseline_prompt_ids,
            follow_up_prompt_ids=follow_up_prompt_ids,
            baseline_measurable_prompt_ids=baseline_measurable_prompt_ids,
            follow_up_measurable_prompt_ids=follow_up_measurable_prompt_ids,
            baseline_canonical_prompt_ids=baseline_canonical,
            follow_up_canonical_prompt_ids=follow_up_canonical,
            limitations=tuple(limitations),
        )
    baseline_value = (
        sum(bool(item.brand_mentioned) for item in baseline_measurable)
        / len(baseline_measurable)
        * 100
    )
    follow_up_value = (
        sum(bool(item.brand_mentioned) for item in follow_up_measurable)
        / len(follow_up_measurable)
        * 100
    )
    complete = set(baseline_prompt_ids) == set(baseline_canonical) == set(
        baseline_measurable_prompt_ids
    ) and set(follow_up_prompt_ids) == set(follow_up_canonical) == set(
        follow_up_measurable_prompt_ids
    )
    state = DataState.AVAILABLE if complete else DataState.PARTIAL
    return AIVisibilityComparison(
        state=state,
        baseline_value=baseline_value,
        follow_up_value=follow_up_value,
        absolute_delta=follow_up_value - baseline_value,
        prompt_pack_version=baseline_prompt_pack_version,
        setup_fingerprint=baseline_setup_fingerprint,
        baseline_observation_ids=tuple(item.observation_id for item in baseline_measurable),
        follow_up_observation_ids=tuple(item.observation_id for item in follow_up_measurable),
        baseline_prompt_ids=baseline_prompt_ids,
        follow_up_prompt_ids=follow_up_prompt_ids,
        baseline_measurable_prompt_ids=baseline_measurable_prompt_ids,
        follow_up_measurable_prompt_ids=follow_up_measurable_prompt_ids,
        baseline_canonical_prompt_ids=baseline_canonical,
        follow_up_canonical_prompt_ids=follow_up_canonical,
    )
