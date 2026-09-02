"""Immutable, visibility-only aggregate projection for client-supplied data."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data_intake import (
    MAX_METRIC_DIMENSIONS,
    MAX_METRIC_FILTERS,
    MAX_METRIC_POINTS,
    MAX_SOURCE_FILTERS,
    ProcessedIntake,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilityMetricSeriesInput,
    VisibilitySource,
    canonical_metadata_tokens,
    validate_visibility_only_label,
)
from .models import DataState
from .project_models import normalize_canonical_domain, validate_project_id


class FrozenVisibilityModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=False,
        allow_inf_nan=False,
    )


class MetricWindow(FrozenVisibilityModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> MetricWindow:
        if self.end < self.start:
            raise ValueError("metric window end must not precede start")
        return self


class VisibilityMetric(FrozenVisibilityModel):
    metric_id: str
    metric: str
    unit: str
    state: DataState
    value: float | None
    coverage: float = Field(ge=0, le=1, strict=True)
    confidence: float = Field(ge=0, le=1, strict=True)
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    window: MetricWindow | None = None
    segments: tuple[str, ...] = Field(default=(), max_length=1)
    definitions: tuple[str, ...] = Field(
        default=(),
        max_length=3 + MAX_SOURCE_FILTERS + MAX_METRIC_DIMENSIONS + MAX_METRIC_FILTERS,
    )
    points: tuple[VisibilityMetricPoint, ...] = Field(default=(), max_length=MAX_METRIC_POINTS)

    @field_validator("metric_id")
    @classmethod
    def validate_metric_identifier(cls, value: str) -> str:
        return validate_visibility_only_label(value, field_name="metric_id")

    @field_validator("segments", "definitions")
    @classmethod
    def validate_retained_metadata(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            validate_visibility_only_label(item, field_name="retained metadata") for item in value
        )

    @model_validator(mode="after")
    def validate_state(self) -> VisibilityMetric:
        numeric = self.state in {DataState.AVAILABLE, DataState.PARTIAL}
        if numeric and (self.value is None or not self.points or self.window is None):
            raise ValueError("available visibility metric requires value, points, and window")
        if numeric and self.coverage <= 0:
            raise ValueError("available visibility metric requires positive coverage")
        if numeric and self.confidence <= 0:
            raise ValueError("available visibility metric requires positive confidence")
        if self.state is DataState.AVAILABLE and self.coverage != 1:
            raise ValueError("AVAILABLE visibility metric requires full coverage")
        if self.state is DataState.PARTIAL and self.coverage >= 1:
            raise ValueError("PARTIAL visibility metric requires coverage below one")
        if not numeric and (self.value is not None or self.points or self.window is not None):
            raise ValueError("non-numeric visibility state must not contain numeric observations")
        _validate_persisted_metric(self)
        return self


class VisibilitySnapshot(FrozenVisibilityModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    canonical_domain: str
    processed_at: datetime
    deleted_at: datetime
    metrics: tuple[VisibilityMetric, ...] = Field(min_length=1)
    sources: tuple[SourceArtifactProvenance, ...] = Field(min_length=1)

    @field_validator("project_id")
    @classmethod
    def validate_project_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("canonical_domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)

    @model_validator(mode="after")
    def validate_identity(self) -> VisibilitySnapshot:
        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("visibility snapshot requires unique metric_id values")
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("visibility snapshot requires unique source_id values")
        referenced = {source_id for metric in self.metrics for source_id in metric.source_ids}
        if referenced != set(sources):
            raise ValueError("every visibility source must be referenced by a metric")
        if any(
            source.processed_at != self.processed_at or source.deleted_at != self.deleted_at
            for source in self.sources
        ):
            raise ValueError("visibility snapshot timestamps must match source provenance")
        for metric in self.metrics:
            source = sources[metric.source_ids[0]]
            _validate_metric_source_binding(metric, source)
        return self


class AggregationMode(StrEnum):
    SUM = "sum"
    LATEST = "latest"


class MetricCadence(StrEnum):
    MONTHLY = "monthly"
    POINT_IN_TIME = "point_in_time"
    WINDOW = "window"


class ValueKind(StrEnum):
    COUNT = "count"
    PERCENT = "percent"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    platform: VisibilitySource
    units: frozenset[str]
    aggregation: AggregationMode
    cadence: MetricCadence
    value_kind: ValueKind


def _definition(
    platform: VisibilitySource,
    units: set[str],
    aggregation: AggregationMode,
    cadence: MetricCadence,
    value_kind: ValueKind = ValueKind.COUNT,
) -> MetricDefinition:
    return MetricDefinition(platform, frozenset(units), aggregation, cadence, value_kind)


SUM_MONTHLY = (AggregationMode.SUM, MetricCadence.MONTHLY)
LATEST_MONTHLY = (AggregationMode.LATEST, MetricCadence.MONTHLY)
LATEST_POINT = (AggregationMode.LATEST, MetricCadence.POINT_IN_TIME)

_CATALOG: dict[str, MetricDefinition] = {
    "gsc.impressions": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"impressions"}, *SUM_MONTHLY
    ),
    "gsc.clicks": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"clicks", "count"}, *SUM_MONTHLY
    ),
    "gsc.query_coverage": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"percent"}, *LATEST_MONTHLY, ValueKind.PERCENT
    ),
    "gsc.page_coverage": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"percent"}, *LATEST_MONTHLY, ValueKind.PERCENT
    ),
    "gsc.country_coverage": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"percent"}, *LATEST_MONTHLY, ValueKind.PERCENT
    ),
    "gsc.device_coverage": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"percent"}, *LATEST_MONTHLY, ValueKind.PERCENT
    ),
    "gsc.search_appearance": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"impressions", "clicks"}, *SUM_MONTHLY
    ),
    "gsc.page_indexing": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"pages"}, *LATEST_POINT
    ),
    "gsc.sitemaps": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"sitemaps", "urls"}, *LATEST_POINT
    ),
    "gsc.core_web_vitals": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"urls"}, *LATEST_POINT
    ),
    "gsc.crawl_stats": _definition(
        VisibilitySource.GOOGLE_SEARCH_CONSOLE, {"requests"}, *LATEST_MONTHLY
    ),
    "ga4.organic_search.sessions": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"sessions"}, *SUM_MONTHLY
    ),
    "ga4.organic_search.users": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"users"}, *LATEST_MONTHLY
    ),
    "ga4.organic_search.source_medium": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"sessions"}, *SUM_MONTHLY
    ),
    "ga4.organic_search.landing_pages": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"sessions"}, *SUM_MONTHLY
    ),
    "ga4.ai_assistant.sessions": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"sessions"}, *SUM_MONTHLY
    ),
    "ga4.ai_assistant.users": _definition(
        VisibilitySource.GOOGLE_ANALYTICS, {"users"}, *LATEST_MONTHLY
    ),
    "merchant_center.impressions": _definition(
        VisibilitySource.MERCHANT_CENTER, {"impressions"}, *SUM_MONTHLY
    ),
    "merchant_center.clicks": _definition(
        VisibilitySource.MERCHANT_CENTER, {"clicks"}, *SUM_MONTHLY
    ),
    "merchant_center.ctr": _definition(
        VisibilitySource.MERCHANT_CENTER, {"percent"}, *LATEST_MONTHLY, ValueKind.PERCENT
    ),
    "merchant_center.traffic_method_coverage": _definition(
        VisibilitySource.MERCHANT_CENTER, {"percent"}, *LATEST_POINT, ValueKind.PERCENT
    ),
    "merchant_center.product_issue_count": _definition(
        VisibilitySource.MERCHANT_CENTER, {"issues"}, *LATEST_POINT
    ),
    "merchant_center.account_issue_count": _definition(
        VisibilitySource.MERCHANT_CENTER, {"issues"}, *LATEST_POINT
    ),
    "gbp.search_views": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"views"}, *SUM_MONTHLY
    ),
    "gbp.profile_views": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"views"}, *SUM_MONTHLY
    ),
    "gbp.searches": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"searches"}, *SUM_MONTHLY
    ),
    "gbp.website_clicks": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"clicks"}, *SUM_MONTHLY
    ),
    "gbp.profile_fact_coverage": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"percent"}, *LATEST_POINT, ValueKind.PERCENT
    ),
    "gbp.category_count": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"categories"}, *LATEST_POINT
    ),
    "gbp.location_count": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"locations"}, *LATEST_POINT
    ),
    "gbp.update_count": _definition(
        VisibilitySource.GOOGLE_BUSINESS_PROFILE, {"updates"}, *LATEST_POINT
    ),
    "bing.impressions": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"impressions"}, *SUM_MONTHLY
    ),
    "bing.clicks": _definition(VisibilitySource.BING_WEBMASTER_TOOLS, {"clicks"}, *SUM_MONTHLY),
    "bing.query_coverage": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"percent"}, *LATEST_POINT, ValueKind.PERCENT
    ),
    "bing.page_coverage": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"percent"}, *LATEST_POINT, ValueKind.PERCENT
    ),
    "bing.indexed_pages": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"pages"}, *LATEST_POINT
    ),
    "bing.sitemaps": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"sitemaps"}, *LATEST_POINT
    ),
    "bing.crawl_issues": _definition(
        VisibilitySource.BING_WEBMASTER_TOOLS, {"issues"}, *LATEST_POINT
    ),
    "logs.search_bot_requests": _definition(
        VisibilitySource.SANITIZED_LOGS, {"requests"}, AggregationMode.SUM, MetricCadence.WINDOW
    ),
    "crawl.indexable_pages": _definition(VisibilitySource.CRAWL, {"pages"}, *LATEST_POINT),
    "ai_monitoring.citations": _definition(
        VisibilitySource.AI_MONITORING, {"citations"}, AggregationMode.SUM, MetricCadence.WINDOW
    ),
    "ai_monitoring.mentions": _definition(
        VisibilitySource.AI_MONITORING, {"mentions"}, AggregationMode.SUM, MetricCadence.WINDOW
    ),
    "ai_monitoring.google_ai_overviews.citations": _definition(
        VisibilitySource.AI_MONITORING, {"citations"}, AggregationMode.SUM, MetricCadence.WINDOW
    ),
    "ai_monitoring.google_ai_mode.citations": _definition(
        VisibilitySource.AI_MONITORING, {"citations"}, AggregationMode.SUM, MetricCadence.WINDOW
    ),
}

_ALIASES: dict[tuple[VisibilitySource, str], str] = {
    (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "clicks"): "gsc.clicks",
    (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "impressions"): "gsc.impressions",
    (VisibilitySource.GOOGLE_ANALYTICS, "organic sessions"): "ga4.organic_search.sessions",
    (VisibilitySource.GOOGLE_ANALYTICS, "organic users"): "ga4.organic_search.users",
}

_GSC_SEARCH_TYPES = frozenset({"web", "image", "video", "news", "discover", "google_news"})
_GSC_SEARCH_RESULT_METRICS = frozenset(
    {
        "gsc.impressions",
        "gsc.clicks",
        "gsc.query_coverage",
        "gsc.page_coverage",
        "gsc.country_coverage",
        "gsc.device_coverage",
        "gsc.search_appearance",
    }
)
_CHANNEL_KEYS = frozenset({"channel", "default_channel_group", "session_default_channel_group"})
_RESERVED_DEFINITION_KEYS = frozenset({"reporttype", "aggregation", "cadence"})


def _canonical_label(value: str) -> str:
    return "_".join(canonical_metadata_tokens(value))


def _metadata_items(
    source_filters: tuple[str, ...], series: VisibilityMetricSeriesInput
) -> tuple[str, ...]:
    return (
        *source_filters,
        *series.dimensions,
        *series.filters,
        *((series.segment_label,) if series.segment_label else ()),
    )


def _validate_visibility_only_metadata(labels: tuple[str, ...]) -> None:
    for label in labels:
        validate_visibility_only_label(label, field_name="metadata")


def _key_values(labels: tuple[str, ...]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for label in labels:
        parts = re.split(r"[:=]", label, maxsplit=1)
        if len(parts) == 2:
            key = _canonical_label(parts[0])
            value = _canonical_label(parts[1])
            if key and value:
                values.setdefault(key, set()).add(value)
    return values


def _contains_reserved_definition_key(labels: tuple[str, ...]) -> bool:
    for label in labels:
        key = re.split(r"[:=]", label, maxsplit=1)[0]
        tokens = canonical_metadata_tokens(key)
        terms = {
            "".join(tokens[start:end])
            for start in range(len(tokens))
            for end in range(start + 1, len(tokens) + 1)
        }
        if terms & _RESERVED_DEFINITION_KEYS:
            return True
    return False


def _validate_platform_provenance(
    metric_name: str,
    labels: tuple[str, ...],
    *,
    segment_label: str | None = None,
) -> None:
    values = _key_values(labels)
    if metric_name in _GSC_SEARCH_RESULT_METRICS:
        search_types = values.get("search_type", set())
        if len(search_types) != 1 or not search_types.issubset(_GSC_SEARCH_TYPES):
            raise ValueError("GSC metric requires one supported explicit search_type")
        if metric_name == "gsc.search_appearance" and not values.get("search_appearance"):
            raise ValueError("search_appearance metric requires explicit dimension provenance")
    if metric_name.startswith("ga4."):
        channels = set().union(*(values.get(key, set()) for key in _CHANNEL_KEYS))
        if segment_label:
            segment = _canonical_label(segment_label)
            if segment in {"organic_search", "ai_assistant"}:
                channels.add(segment)
        expected = (
            "organic_search" if metric_name.startswith("ga4.organic_search.") else "ai_assistant"
        )
        if channels != {expected}:
            raise ValueError(f"GA4 metric requires explicit non-conflicting {expected} channel")
        if expected == "organic_search":
            if any(_contains_ai_search_alias(label) for label in labels):
                raise ValueError("Organic Search metadata cannot assert an AI observation")


def _contains_ai_search_alias(label: str) -> bool:
    tokens = canonical_metadata_tokens(label)
    token_set = set(tokens)
    collapsed = "".join(tokens)
    return (
        bool(token_set & {"aio", "sge"})
        or "aioverview" in collapsed
        or "aimode" in collapsed
        or "aigeneratedoverview" in collapsed
        or ("ai" in token_set and bool(token_set & {"overview", "overviews", "mode"}))
    )


def validate_series_definition(
    series: VisibilityMetricSeriesInput,
    *,
    platform: VisibilitySource,
    source_filters: tuple[str, ...] = (),
) -> str:
    """Resolve one exact catalog key and reject out-of-scope measurements."""
    metric_name = _ALIASES.get((platform, series.metric), series.metric)
    definition = _CATALOG.get(metric_name)
    if definition is None:
        raise ValueError(f"metric is not an allowlisted visibility metric: {series.metric}")
    if platform is not definition.platform:
        raise ValueError("metric source platform does not match the allowlisted definition")
    if series.unit not in definition.units:
        raise ValueError("metric unit does not match the allowlisted definition")
    labels = _metadata_items(source_filters, series)
    _validate_visibility_only_metadata(labels)
    if _contains_reserved_definition_key(labels):
        raise ValueError("metric metadata must not override reserved definitions")
    _validate_platform_provenance(
        metric_name,
        labels,
        segment_label=series.segment_label,
    )
    return metric_name


def _month_end(value: date) -> date:
    return value.replace(day=calendar.monthrange(value.year, value.month)[1])


def _effective_end(point: VisibilityMetricPoint, cadence: MetricCadence) -> date:
    if point.period_end is not None:
        return point.period_end
    if cadence is MetricCadence.MONTHLY:
        return _month_end(point.period_start)
    return point.period_start


def _validate_points(
    points: tuple[VisibilityMetricPoint, ...],
    definition: MetricDefinition,
) -> None:
    periods = [point.period_start for point in points]
    if len(periods) != len(set(periods)):
        raise ValueError("visibility metric points require unique periods")
    if definition.cadence is MetricCadence.MONTHLY:
        for point in points:
            if point.period_start.day != 1:
                raise ValueError("monthly visibility metrics require month-boundary periods")
            if point.period_end is not None and point.period_end != _month_end(point.period_start):
                raise ValueError("monthly visibility metric period_end must be the month end")
    for previous, current in zip(points, points[1:], strict=False):
        if _effective_end(previous, definition.cadence) >= current.period_start:
            raise ValueError("visibility metric periods must not overlap")
    for point in points:
        if point.value < 0:
            raise ValueError("visibility metric values must be non-negative")
        if definition.value_kind is ValueKind.PERCENT and point.value > 100:
            raise ValueError("percent visibility metric values must be between 0 and 100")


def _inclusive_calendar_months(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _validate_source_window(
    points: tuple[VisibilityMetricPoint, ...],
    definition: MetricDefinition,
    source: SourceArtifactProvenance,
) -> None:
    if source.platform in {
        VisibilitySource.GOOGLE_SEARCH_CONSOLE,
        VisibilitySource.GOOGLE_ANALYTICS,
    }:
        if source.date_range is None:
            raise ValueError("numeric GSC/GA4 visibility metrics require source date_range")
        if _inclusive_calendar_months(source.date_range.start, source.date_range.end) > 16:
            raise ValueError("GSC/GA4 source date_range must not exceed 16 calendar months")
    if source.date_range is None:
        return
    for point in points:
        if (
            point.period_start < source.date_range.start
            or _effective_end(point, definition.cadence) > source.date_range.end
        ):
            raise ValueError("visibility metric point falls outside source date_range")


def _expected_window(
    points: tuple[VisibilityMetricPoint, ...],
    definition: MetricDefinition,
) -> MetricWindow:
    return MetricWindow(
        start=points[0].period_start,
        end=max(_effective_end(point, definition.cadence) for point in points),
    )


def _expected_value(
    points: tuple[VisibilityMetricPoint, ...],
    definition: MetricDefinition,
) -> float:
    if definition.aggregation is AggregationMode.LATEST:
        return points[-1].value
    return sum(point.value for point in points)


def _validate_monthly_temporal_coverage(
    points: tuple[VisibilityMetricPoint, ...],
    definition: MetricDefinition,
    *,
    coverage: float,
    source: SourceArtifactProvenance | None,
) -> None:
    if (
        definition.aggregation is not AggregationMode.SUM
        or definition.cadence is not MetricCadence.MONTHLY
        or not points
    ):
        return
    if source is not None and source.date_range is not None:
        expected = _inclusive_calendar_months(source.date_range.start, source.date_range.end)
    else:
        expected = _inclusive_calendar_months(points[0].period_start, points[-1].period_start)
    observed = len({(point.period_start.year, point.period_start.month) for point in points})
    maximum_coverage = observed / expected
    if coverage > maximum_coverage + 1e-9:
        raise ValueError("monthly additive coverage exceeds observed contiguous temporal coverage")


def _reserved_definition_prefix(
    definition: MetricDefinition,
    *,
    report_type: str | None = None,
) -> tuple[str, ...]:
    prefix = () if report_type is None else (f"report_type={report_type}",)
    return (
        *prefix,
        f"aggregation={definition.aggregation.value}",
        f"cadence={definition.cadence.value}",
    )


def _validate_persisted_metric(metric: VisibilityMetric) -> None:
    definition = _CATALOG.get(metric.metric)
    if definition is None:
        raise ValueError("persisted metric is not in the visibility catalog")
    if metric.unit not in definition.units:
        raise ValueError("persisted metric unit conflicts with the visibility catalog")
    if len(metric.definitions) < 3 or not metric.definitions[0].startswith("report_type="):
        raise ValueError("persisted metric lacks reserved report_type definition")
    if metric.definitions[1:3] != _reserved_definition_prefix(definition):
        raise ValueError("persisted metric aggregation or cadence definition is invalid")
    if _contains_reserved_definition_key(metric.definitions[3:]):
        raise ValueError("persisted metric contains a duplicate reserved definition")
    if metric.points:
        if metric.points != tuple(sorted(metric.points, key=lambda point: point.period_start)):
            raise ValueError("persisted metric points must be chronological")
        _validate_points(metric.points, definition)
        if metric.window != _expected_window(metric.points, definition):
            raise ValueError("persisted metric window does not match its points")
        if metric.value != _expected_value(metric.points, definition):
            raise ValueError("persisted metric value does not match canonical aggregation")
        _validate_monthly_temporal_coverage(
            metric.points,
            definition,
            coverage=metric.coverage,
            source=None,
        )


def _validate_metric_source_binding(
    metric: VisibilityMetric,
    source: SourceArtifactProvenance,
) -> None:
    definition = _CATALOG[metric.metric]
    if source.platform is not definition.platform:
        raise ValueError("persisted metric platform conflicts with source provenance")
    expected_prefix = _reserved_definition_prefix(
        definition,
        report_type=source.report_type,
    )
    if metric.definitions[:3] != expected_prefix:
        raise ValueError("persisted metric report_type conflicts with source provenance")
    source_filter_end = 3 + len(source.filters)
    if metric.definitions[3:source_filter_end] != source.filters:
        raise ValueError("persisted metric source filters conflict with source provenance")
    retained_labels = (*metric.definitions[3:], *metric.segments)
    _validate_platform_provenance(
        metric.metric,
        retained_labels,
        segment_label=metric.segments[0] if metric.segments else None,
    )
    if metric.points:
        _validate_source_window(metric.points, definition, source)
        _validate_monthly_temporal_coverage(
            metric.points,
            definition,
            coverage=metric.coverage,
            source=source,
        )


def aggregate_series(
    series: VisibilityMetricSeriesInput,
    *,
    source: SourceArtifactProvenance,
) -> VisibilityMetric:
    """Validate and aggregate one canonical visibility series."""
    metric_name = validate_series_definition(
        series,
        platform=source.platform,
        source_filters=source.filters,
    )
    definition = _CATALOG[metric_name]
    if source.source_id != series.source_id:
        raise ValueError("metric series source does not match source provenance")

    numeric = series.state in {DataState.AVAILABLE, DataState.PARTIAL}
    points = tuple(sorted(series.points, key=lambda point: point.period_start)) if numeric else ()
    window = None
    value = None
    if points:
        _validate_points(points, definition)
        _validate_source_window(points, definition, source)
        _validate_monthly_temporal_coverage(
            points,
            definition,
            coverage=series.coverage,
            source=source,
        )
        window = _expected_window(points, definition)
        value = _expected_value(points, definition)
    definitions = (
        f"report_type={source.report_type}",
        f"aggregation={definition.aggregation.value}",
        f"cadence={definition.cadence.value}",
        *source.filters,
        *series.dimensions,
        *series.filters,
    )
    segments = (series.segment_label,) if series.segment_label else ()
    return VisibilityMetric(
        metric_id=series.metric_id,
        metric=metric_name,
        unit=series.unit,
        state=series.state,
        value=value,
        coverage=series.coverage,
        confidence=series.confidence,
        source_ids=(source.source_id,),
        window=window,
        segments=segments,
        definitions=definitions,
        points=points,
    )


def build_visibility_snapshot(processed: ProcessedIntake) -> VisibilitySnapshot:
    """Build the immutable normalized aggregate retained after raw deletion."""
    if not processed.metric_series or processed.owner_facts or processed.cited_examples:
        raise ValueError("visibility intake must contain metrics only")
    sources = {source.source_id: source for source in processed.sources}
    metrics = tuple(
        aggregate_series(series, source=sources[series.source_id])
        for series in processed.metric_series
    )
    return VisibilitySnapshot(
        project_id=processed.project_id,
        canonical_domain=processed.canonical_domain,
        processed_at=processed.processed_at,
        deleted_at=processed.deleted_at,
        metrics=metrics,
        sources=processed.sources,
    )
