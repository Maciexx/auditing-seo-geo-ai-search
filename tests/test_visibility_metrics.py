from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from ai_search_audit.data_intake import (
    MAX_METRIC_DIMENSIONS,
    MAX_METRIC_FILTERS,
    MAX_METRIC_POINTS,
    MAX_SOURCE_FILTERS,
    DateRange,
    ProcessedIntake,
    SourceArtifactProvenance,
    VisibilityMetricPoint,
    VisibilityMetricSeriesInput,
    VisibilitySource,
)
from ai_search_audit.models import DataState
from ai_search_audit.visibility_metrics import (
    VisibilityMetric,
    VisibilitySnapshot,
    aggregate_series,
    build_visibility_snapshot,
)

NOW = datetime(2026, 9, 1, 9, 30, tzinfo=UTC)


def _source(
    source_id: str = "ga4",
    platform: VisibilitySource = VisibilitySource.GOOGLE_ANALYTICS,
    **overrides: object,
) -> SourceArtifactProvenance:
    values: dict[str, object] = {
        "source_id": source_id,
        "filename": f"{source_id}.csv",
        "sha256": "a" * 64,
        "byte_count": 10,
        "platform": platform,
        "report_type": "visibility-export",
        "processed_at": NOW,
        "deleted_at": NOW,
    }
    if platform in {
        VisibilitySource.GOOGLE_ANALYTICS,
        VisibilitySource.GOOGLE_SEARCH_CONSOLE,
    }:
        values["date_range"] = DateRange(
            start=date(2026, 6, 1),
            end=date(2026, 8, 31),
        )
    values.update(overrides)
    return SourceArtifactProvenance(**values)


def _series(**overrides: object) -> VisibilityMetricSeriesInput:
    values: dict[str, object] = {
        "metric_id": "ga4-ai-sessions",
        "source_id": "ga4",
        "metric": "ga4.ai_assistant.sessions",
        "unit": "sessions",
        "state": DataState.AVAILABLE,
        "coverage": 1.0,
        "confidence": 0.9,
        "filters": ("channel=AI Assistant",),
        "points": (
            VisibilityMetricPoint(period_start=date(2026, 6, 1), value=0),
            VisibilityMetricPoint(period_start=date(2026, 7, 1), value=0),
            VisibilityMetricPoint(period_start=date(2026, 8, 1), value=0),
        ),
    }
    values.update(overrides)
    return VisibilityMetricSeriesInput(**values)


def test_ga4_ai_assistant_zero_is_available_not_missing() -> None:
    metric = aggregate_series(_series(), source=_source())

    assert metric.state is DataState.AVAILABLE
    assert metric.value == 0
    assert metric.coverage == 1
    assert metric.confidence == 0.9
    assert metric.metric == "ga4.ai_assistant.sessions"
    assert "report_type=visibility-export" in metric.definitions


@pytest.mark.parametrize("state", (DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED))
def test_non_numeric_states_remain_distinct_and_never_become_zero(
    state: DataState,
) -> None:
    metric = aggregate_series(
        _series(
            state=state,
            points=(),
            coverage=0,
            confidence=0.4,
        ),
        source=_source(date_range=DateRange(start=date(2026, 7, 1), end=date(2026, 8, 31))),
    )

    assert metric.state is state
    assert metric.value is None
    assert metric.points == ()


def test_snapshot_retains_only_immutable_aggregate_visibility_evidence() -> None:
    source = _source()
    processed = ProcessedIntake(
        project_id="example",
        canonical_domain="example.com",
        processed_at=NOW,
        deleted_at=NOW,
        sources=(source,),
        metric_series=(_series(),),
    )

    snapshot = build_visibility_snapshot(processed)

    assert snapshot.project_id == "example"
    assert snapshot.metrics[0].source_ids == ("ga4",)
    assert snapshot.metrics[0].window.start == date(2026, 6, 1)
    assert snapshot.metrics[0].window.end == date(2026, 8, 31)
    with pytest.raises(ValidationError):
        snapshot.metrics[0].value = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.metrics[0].definitions[0] = "channel=Organic Search"  # type: ignore[index]


@pytest.mark.parametrize(
    ("platform", "metric", "unit"),
    (
        (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "gsc.impressions", "impressions"),
        (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "gsc.page_indexing", "pages"),
        (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "gsc.sitemaps", "sitemaps"),
        (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "gsc.core_web_vitals", "urls"),
        (VisibilitySource.GOOGLE_SEARCH_CONSOLE, "gsc.crawl_stats", "requests"),
        (VisibilitySource.GOOGLE_ANALYTICS, "ga4.organic_search.sessions", "sessions"),
        (VisibilitySource.GOOGLE_ANALYTICS, "ga4.organic_search.users", "users"),
        (VisibilitySource.MERCHANT_CENTER, "merchant_center.impressions", "impressions"),
        (VisibilitySource.MERCHANT_CENTER, "merchant_center.ctr", "percent"),
        (VisibilitySource.MERCHANT_CENTER, "merchant_center.traffic_method_coverage", "percent"),
        (VisibilitySource.MERCHANT_CENTER, "merchant_center.product_issue_count", "issues"),
        (VisibilitySource.MERCHANT_CENTER, "merchant_center.account_issue_count", "issues"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.search_views", "views"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.searches", "searches"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.website_clicks", "clicks"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.profile_fact_coverage", "percent"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.category_count", "categories"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.location_count", "locations"),
        (VisibilitySource.GOOGLE_BUSINESS_PROFILE, "gbp.update_count", "updates"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.impressions", "impressions"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.query_coverage", "percent"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.page_coverage", "percent"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.indexed_pages", "pages"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.sitemaps", "sitemaps"),
        (VisibilitySource.BING_WEBMASTER_TOOLS, "bing.crawl_issues", "issues"),
        (VisibilitySource.SANITIZED_LOGS, "logs.search_bot_requests", "requests"),
        (VisibilitySource.CRAWL, "crawl.indexable_pages", "pages"),
        (VisibilitySource.AI_MONITORING, "ai_monitoring.citations", "citations"),
    ),
)
def test_metric_catalog_accepts_scoped_visibility_measurements(
    platform: VisibilitySource,
    metric: str,
    unit: str,
) -> None:
    source_overrides: dict[str, object] = {}
    if platform in {
        VisibilitySource.GOOGLE_SEARCH_CONSOLE,
        VisibilitySource.GOOGLE_ANALYTICS,
    }:
        source_overrides["date_range"] = DateRange(
            start=date(2026, 8, 1),
            end=date(2026, 8, 31),
        )
    source = _source("source", platform, **source_overrides)
    filters: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    if platform is VisibilitySource.GOOGLE_SEARCH_CONSOLE:
        filters = ("search_type=web",)
        if metric == "gsc.search_appearance":
            dimensions = ("search_appearance=merchant_listings",)
    elif metric.startswith("ga4.organic_search"):
        filters = ("channel=Organic Search",)
    result = aggregate_series(
        _series(
            metric_id="metric",
            source_id="source",
            metric=metric,
            unit=unit,
            filters=filters,
            dimensions=dimensions,
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        ),
        source=source,
    )

    assert result.metric == metric


@pytest.mark.parametrize(
    "metric",
    (
        "ga4.conversions",
        "ga4.revenue",
        "form_submissions",
        "leads",
        "crm.customer_ids",
        "ga4.user_events",
    ),
)
def test_metric_catalog_rejects_commercial_or_user_level_fields(metric: str) -> None:
    with pytest.raises(ValueError, match="allowlisted visibility metric"):
        aggregate_series(_series(metric=metric), source=_source())


def test_organic_search_cannot_be_reclassified_as_google_ai_overviews() -> None:
    with pytest.raises(ValueError, match="source platform|allowlisted"):
        aggregate_series(
            _series(metric="google.ai_overviews.impressions", unit="impressions"),
            source=_source(),
        )


def test_available_metrics_require_explicit_positive_coverage_and_confidence() -> None:
    with pytest.raises(ValidationError, match="coverage"):
        VisibilityMetric.model_validate(
            {
                "metric_id": "bad",
                "metric": "gsc.clicks",
                "unit": "clicks",
                "state": "AVAILABLE",
                "value": 1,
                "coverage": 0,
                "confidence": 0.5,
                "source_ids": ["gsc"],
                "points": [{"period_start": "2026-08-01", "value": 1}],
                "window": {"start": "2026-08-01", "end": "2026-08-01"},
            }
        )


def test_coverage_diagnostic_uses_latest_observation_instead_of_summing_percentages() -> None:
    metric = aggregate_series(
        _series(
            metric_id="query-coverage",
            source_id="gsc",
            metric="gsc.query_coverage",
            unit="percent",
            filters=("search_type=web",),
            points=(
                VisibilityMetricPoint(period_start=date(2026, 7, 1), value=60),
                VisibilityMetricPoint(period_start=date(2026, 8, 1), value=75),
            ),
        ),
        source=_source(
            "gsc",
            VisibilitySource.GOOGLE_SEARCH_CONSOLE,
            date_range=DateRange(start=date(2026, 7, 1), end=date(2026, 8, 31)),
        ),
    )

    assert metric.value == 75
    assert "aggregation=latest" in metric.definitions


@pytest.mark.parametrize(
    ("placement", "label"),
    (
        ("source", "ConVersion = purchase"),
        ("filter", "customer-ID=123"),
        ("filter", "customer/account/id=123"),
        ("dimension", "REVENUE: gross"),
        ("segment", "crm/customer ids"),
        ("filter", "lead_form-submission=true"),
        ("dimension", "USER.ID=abc"),
        ("segment", "user-level events"),
    ),
)
def test_scope_guard_rejects_commercial_and_user_level_metadata(
    placement: str,
    label: str,
) -> None:
    source_filters = (label,) if placement == "source" else ()
    series_overrides: dict[str, object] = {}
    if placement == "filter":
        series_overrides["filters"] = ("channel=AI Assistant", label)
    elif placement == "dimension":
        series_overrides["dimensions"] = (label,)
    elif placement == "segment":
        series_overrides["segment_label"] = label

    with pytest.raises(ValueError, match="visibility-only"):
        aggregate_series(
            _series(**series_overrides),
            source=_source(filters=source_filters),
        )


def test_persisted_metric_rejects_commercial_definition() -> None:
    valid = aggregate_series(_series(), source=_source())
    payload = valid.model_dump(mode="json")
    payload["definitions"] = [*payload["definitions"], "Revenue=100"]

    with pytest.raises(ValidationError, match="visibility-only"):
        VisibilityMetric.model_validate(payload)


@pytest.mark.parametrize(
    "label",
    ("aggregation=latest", "cadence=window", "report_type=other", "Agg-Regation=latest"),
)
def test_persisted_metric_rejects_appended_reserved_definition(label: str) -> None:
    valid = aggregate_series(_series(), source=_source())
    payload = valid.model_dump(mode="json")
    payload["definitions"] = [*payload["definitions"], label]

    with pytest.raises(ValidationError, match="reserved"):
        VisibilityMetric.model_validate(payload)


def test_persisted_metric_allows_non_reserved_retained_metadata() -> None:
    metric = aggregate_series(
        _series(dimensions=("landing_page",), filters=("channel=AI Assistant",)),
        source=_source(filters=("country=PL",)),
    )

    assert "country=PL" in metric.definitions
    assert "landing_page" in metric.definitions


@pytest.mark.parametrize(
    "label",
    ("aggregation=latest", "cad-ence=monthly", "Report-Type=other"),
)
def test_input_metadata_rejects_reserved_key_variants(label: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        aggregate_series(
            _series(filters=("channel=AI Assistant", label)),
            source=_source(),
        )


def test_normalized_series_rejects_commercial_metric_identifier() -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        _series(metric_id="revenue-total")


@pytest.mark.parametrize(
    "event_label",
    (
        "event_name=form_submit",
        "eventName=FormSubmit",
        "event-name=form submitted",
        "event_name=form/submits",
        "event_name=form-submit",
        "event_name=generate_lead",
        "eventName=GenerateLead",
    ),
)
def test_canonical_organic_metric_rejects_form_and_lead_event_metadata(
    event_label: str,
) -> None:
    with pytest.raises(ValidationError, match="visibility-only"):
        _series(
            metric="ga4.organic_search.sessions",
            filters=("channel=Organic Search", event_label),
        )


@pytest.mark.parametrize(
    ("placement", "label"),
    (
        ("source", "metric=totalRevenue"),
        ("filter", "metric=purchaseRevenue"),
        ("dimension", "rate=sessionConversionRate"),
        ("segment", "keyEvent"),
        ("definition", "metric=keyEvents"),
    ),
)
def test_camel_case_ga4_commercial_metadata_is_rejected(
    placement: str,
    label: str,
) -> None:
    if placement == "definition":
        valid = aggregate_series(_series(), source=_source())
        payload = valid.model_dump(mode="json")
        payload["definitions"] = [*payload["definitions"], label]
        with pytest.raises(ValidationError, match="visibility-only"):
            VisibilityMetric.model_validate(payload)
        return
    source_filters = (label,) if placement == "source" else ()
    overrides: dict[str, object] = {}
    if placement == "filter":
        overrides["filters"] = ("channel=AI Assistant", label)
    elif placement == "dimension":
        overrides["dimensions"] = (label,)
    elif placement == "segment":
        overrides["segment_label"] = label
    with pytest.raises(ValidationError, match="visibility-only"):
        series = _series(**overrides)
        _source(filters=source_filters)
        aggregate_series(series, source=_source(filters=source_filters))


def test_key_events_metric_is_not_allowlisted_visibility_measurement() -> None:
    with pytest.raises(ValueError, match="allowlisted visibility metric"):
        aggregate_series(_series(metric="keyEvents"), source=_source())


@pytest.mark.parametrize(
    "label",
    (
        "dimension=sessionSourceMedium",
        "metric=totalUsers",
        "deviceCategory=mobile",
    ),
)
def test_legal_camel_case_visibility_labels_remain_allowed(label: str) -> None:
    series = _series(filters=("channel=AI Assistant", label))
    assert aggregate_series(series, source=_source()).value == 0


@pytest.mark.parametrize("search_type", ("web", "image", "video", "news", "discover"))
def test_gsc_requires_supported_explicit_search_type(search_type: str) -> None:
    metric = aggregate_series(
        _series(
            metric_id="gsc-clicks",
            source_id="gsc",
            metric="gsc.clicks",
            unit="clicks",
            filters=(f"search_type={search_type}",),
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        ),
        source=_source(
            "gsc",
            VisibilitySource.GOOGLE_SEARCH_CONSOLE,
            date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 31)),
        ),
    )
    assert f"search_type={search_type}" in metric.definitions


@pytest.mark.parametrize(
    "filters", ((), ("search_type=shopping",), ("search_type=web", "search_type=image"))
)
def test_gsc_rejects_missing_unsupported_or_conflicting_search_type(
    filters: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="search_type"):
        aggregate_series(
            _series(
                metric_id="gsc-clicks",
                source_id="gsc",
                metric="gsc.clicks",
                unit="clicks",
                filters=filters,
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
            ),
            source=_source("gsc", VisibilitySource.GOOGLE_SEARCH_CONSOLE),
        )


@pytest.mark.parametrize(
    ("metric", "unit"),
    (
        ("gsc.page_indexing", "pages"),
        ("gsc.sitemaps", "sitemaps"),
        ("gsc.core_web_vitals", "urls"),
        ("gsc.crawl_stats", "requests"),
    ),
)
def test_gsc_diagnostics_do_not_require_fake_search_type(metric: str, unit: str) -> None:
    result = aggregate_series(
        _series(
            metric_id="diagnostic",
            source_id="gsc",
            metric=metric,
            unit=unit,
            filters=(),
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        ),
        source=_source("gsc", VisibilitySource.GOOGLE_SEARCH_CONSOLE),
    )
    assert result.metric == metric


def test_search_appearance_metric_requires_explicit_dimension_provenance() -> None:
    with pytest.raises(ValueError, match="search_appearance"):
        aggregate_series(
            _series(
                metric_id="appearance",
                source_id="gsc",
                metric="gsc.search_appearance",
                unit="impressions",
                filters=("search_type=web",),
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
            ),
            source=_source("gsc", VisibilitySource.GOOGLE_SEARCH_CONSOLE),
        )


@pytest.mark.parametrize(
    ("metric", "filters"),
    (
        ("ga4.organic_search.sessions", ()),
        ("ga4.organic_search.sessions", ("channel=AI Assistant",)),
        ("ga4.ai_assistant.sessions", ()),
        ("ga4.ai_assistant.sessions", ("channel=Organic Search",)),
        ("ga4.organic_search.sessions", ("channel=Organic Search", "channel=AI Assistant")),
    ),
)
def test_ga4_channels_require_distinct_explicit_classification(
    metric: str,
    filters: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="channel"):
        aggregate_series(_series(metric=metric, filters=filters), source=_source())


@pytest.mark.parametrize(
    "label",
    (
        "google_ai_overviews",
        "AI-Mode",
        "channel=AI Overviews",
        "product=AIO",
        "feature=SGE",
        "Google AI Generated Overview",
        "google-ai-generated-overviews",
        "AIOverview",
        "AIMode",
        "source=googleAIO",
    ),
)
def test_organic_metadata_cannot_reclassify_ai_observations(label: str) -> None:
    with pytest.raises(ValueError, match="AI observation|channel"):
        aggregate_series(
            _series(
                metric="ga4.organic_search.sessions",
                filters=("channel=Organic Search", label),
            ),
            source=_source(),
        )


def test_explicit_ai_monitoring_source_accepts_google_ai_overview_observation() -> None:
    metric = aggregate_series(
        _series(
            metric_id="aio-citations",
            source_id="monitor",
            metric="ai_monitoring.google_ai_overviews.citations",
            unit="citations",
            filters=("product=Google AI Overviews",),
            points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
        ),
        source=_source("monitor", VisibilitySource.AI_MONITORING),
    )
    assert metric.value == 1


def test_monthly_series_rejects_non_month_boundary_points() -> None:
    with pytest.raises(ValueError, match="monthly"):
        aggregate_series(
            _series(points=(VisibilityMetricPoint(period_start=date(2026, 8, 15), value=1),)),
            source=_source(),
        )


def test_aggregate_points_are_unique_and_chronologically_normalized() -> None:
    metric = aggregate_series(
        _series(
            points=(
                VisibilityMetricPoint(period_start=date(2026, 8, 1), value=2),
                VisibilityMetricPoint(period_start=date(2026, 7, 1), value=1),
            )
        ),
        source=_source(date_range=DateRange(start=date(2026, 7, 1), end=date(2026, 8, 31))),
    )
    assert tuple(point.period_start for point in metric.points) == (
        date(2026, 7, 1),
        date(2026, 8, 1),
    )

    with pytest.raises(ValueError, match="unique periods"):
        aggregate_series(
            _series(
                points=(
                    VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),
                    VisibilityMetricPoint(period_start=date(2026, 8, 1), value=2),
                )
            ),
            source=_source(),
        )


@pytest.mark.parametrize(
    ("metric", "unit", "value"),
    (
        ("logs.search_bot_requests", "requests", -1),
        ("crawl.indexable_pages", "pages", -1),
        ("merchant_center.ctr", "percent", 101),
        ("gsc.query_coverage", "percent", -0.1),
    ),
)
def test_metric_bounds_reject_negative_counts_and_invalid_percentages(
    metric: str,
    unit: str,
    value: float,
) -> None:
    platform = {
        "logs": VisibilitySource.SANITIZED_LOGS,
        "crawl": VisibilitySource.CRAWL,
        "merchant_center": VisibilitySource.MERCHANT_CENTER,
        "gsc": VisibilitySource.GOOGLE_SEARCH_CONSOLE,
    }[metric.split(".", 1)[0]]
    filters = ("search_type=web",) if platform is VisibilitySource.GOOGLE_SEARCH_CONSOLE else ()
    with pytest.raises(ValueError, match="non-negative|percent"):
        aggregate_series(
            _series(
                metric_id="bounded",
                source_id="source",
                metric=metric,
                unit=unit,
                filters=filters,
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=value),),
            ),
            source=_source("source", platform),
        )


@pytest.mark.parametrize(
    ("metric", "platform", "unit"),
    (
        ("logs.search_bot_requests", VisibilitySource.SANITIZED_LOGS, "requests"),
        ("merchant_center.product_issue_count", VisibilitySource.MERCHANT_CENTER, "issues"),
    ),
)
def test_additive_metrics_reject_overlapping_windows(
    metric: str,
    platform: VisibilitySource,
    unit: str,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        aggregate_series(
            _series(
                metric_id="overlap",
                source_id="source",
                metric=metric,
                unit=unit,
                filters=(),
                points=(
                    VisibilityMetricPoint(
                        period_start=date(2026, 8, 1),
                        period_end=date(2026, 8, 20),
                        value=10,
                    ),
                    VisibilityMetricPoint(
                        period_start=date(2026, 8, 15),
                        period_end=date(2026, 8, 31),
                        value=12,
                    ),
                ),
            ),
            source=_source("source", platform),
        )


def test_unique_user_counts_are_not_summed_across_months() -> None:
    metric = aggregate_series(
        _series(
            metric="ga4.organic_search.users",
            unit="users",
            filters=("channel=Organic Search",),
            points=(
                VisibilityMetricPoint(period_start=date(2026, 7, 1), value=100),
                VisibilityMetricPoint(period_start=date(2026, 8, 1), value=120),
            ),
        ),
        source=_source(),
    )
    assert metric.value == 120
    assert "aggregation=latest" in metric.definitions


def test_snapshot_rejects_source_timestamp_mismatch() -> None:
    source = _source(processed_at=NOW.replace(hour=8))
    with pytest.raises(ValidationError, match="timestamps"):
        VisibilitySnapshot(
            project_id="example",
            canonical_domain="example.com",
            processed_at=NOW,
            deleted_at=NOW,
            metrics=(aggregate_series(_series(), source=_source()),),
            sources=(source,),
        )


def test_numeric_gsc_and_ga4_metrics_require_source_date_range() -> None:
    for platform, series in (
        (VisibilitySource.GOOGLE_ANALYTICS, _series()),
        (
            VisibilitySource.GOOGLE_SEARCH_CONSOLE,
            _series(
                metric_id="gsc-clicks",
                source_id="source",
                metric="gsc.clicks",
                unit="clicks",
                filters=("search_type=web",),
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
            ),
        ),
    ):
        with pytest.raises(ValueError, match="date_range"):
            aggregate_series(
                series,
                source=_source(
                    "source" if platform is VisibilitySource.GOOGLE_SEARCH_CONSOLE else "ga4",
                    platform,
                    date_range=None,
                ),
            )


def test_non_numeric_google_state_does_not_fabricate_source_window() -> None:
    metric = aggregate_series(
        _series(
            state=DataState.UNAVAILABLE,
            points=(),
            coverage=0,
            confidence=0.4,
        ),
        source=_source(date_range=None),
    )
    assert metric.state is DataState.UNAVAILABLE
    assert metric.window is None


@pytest.mark.parametrize(
    ("start", "end", "accepted"),
    (
        (date(2025, 5, 1), date(2026, 8, 31), True),
        (date(2025, 4, 1), date(2026, 8, 31), False),
    ),
)
def test_gsc_ga4_source_window_is_bounded_to_16_inclusive_calendar_months(
    start: date,
    end: date,
    accepted: bool,
) -> None:
    series = _series(
        state=DataState.PARTIAL,
        coverage=1 / 16,
        points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=1),),
    )
    source = _source(date_range=DateRange(start=start, end=end))
    if accepted:
        assert aggregate_series(series, source=source).value == 1
    else:
        with pytest.raises(ValueError, match="16 calendar months"):
            aggregate_series(series, source=source)


def test_points_must_fall_inside_declared_source_window() -> None:
    source = _source(date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 31)))
    with pytest.raises(ValueError, match="source date_range"):
        aggregate_series(
            _series(points=(VisibilityMetricPoint(period_start=date(2026, 7, 1), value=1),)),
            source=source,
        )


def test_optional_non_google_source_range_still_bounds_points() -> None:
    source = _source(
        "crawl",
        VisibilitySource.CRAWL,
        date_range=DateRange(start=date(2026, 8, 1), end=date(2026, 8, 1)),
    )
    assert (
        aggregate_series(
            _series(
                metric_id="crawl-pages",
                source_id="crawl",
                metric="crawl.indexable_pages",
                unit="pages",
                filters=(),
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 1), value=5),),
            ),
            source=source,
        ).value
        == 5
    )
    with pytest.raises(ValueError, match="source date_range"):
        aggregate_series(
            _series(
                metric_id="crawl-pages",
                source_id="crawl",
                metric="crawl.indexable_pages",
                unit="pages",
                filters=(),
                points=(VisibilityMetricPoint(period_start=date(2026, 8, 2), value=5),),
            ),
            source=source,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("value", 999),
        ("window", {"start": "2026-06-01", "end": "2026-07-31"}),
        ("unit", "clicks"),
        (
            "definitions",
            [
                "report_type=visibility-export",
                "aggregation=latest",
                "cadence=monthly",
                "channel=AI Assistant",
            ],
        ),
    ),
)
def test_persisted_metric_recomputes_protected_semantics(
    field: str,
    value: object,
) -> None:
    payload = aggregate_series(_series(), source=_source()).model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match="unit|aggregation|window|value"):
        VisibilityMetric.model_validate(payload)


def test_persisted_metric_rejects_non_chronological_points() -> None:
    payload = aggregate_series(_series(), source=_source()).model_dump(mode="json")
    payload["points"] = list(reversed(payload["points"]))

    with pytest.raises(ValidationError, match="chronological"):
        VisibilityMetric.model_validate(payload)


@pytest.mark.parametrize("tamper", ("platform", "source_id", "report_type"))
def test_persisted_snapshot_binds_metric_to_exact_source_provenance(tamper: str) -> None:
    source = _source()
    metric = aggregate_series(_series(), source=source)
    if tamper == "platform":
        source = source.model_copy(update={"platform": VisibilitySource.MERCHANT_CENTER})
    elif tamper == "source_id":
        source = source.model_copy(update={"source_id": "other"})
    else:
        source = source.model_copy(update={"report_type": "other-report"})

    with pytest.raises(ValidationError, match="source|platform|report_type"):
        VisibilitySnapshot(
            project_id="example",
            canonical_domain="example.com",
            processed_at=NOW,
            deleted_at=NOW,
            metrics=(metric,),
            sources=(source,),
        )


def test_persisted_gsc_metric_requires_retained_search_type_provenance() -> None:
    source = _source("gsc", VisibilitySource.GOOGLE_SEARCH_CONSOLE, filters=("search_type=web",))
    metric = aggregate_series(
        _series(
            metric_id="gsc-clicks",
            source_id="gsc",
            metric="gsc.clicks",
            unit="clicks",
            filters=(),
        ),
        source=source,
    )
    payload = VisibilitySnapshot(
        project_id="example",
        canonical_domain="example.com",
        processed_at=NOW,
        deleted_at=NOW,
        metrics=(metric,),
        sources=(source,),
    ).model_dump(mode="json")
    payload["metrics"][0]["definitions"] = [
        item for item in payload["metrics"][0]["definitions"] if item != "search_type=web"
    ]
    payload["sources"][0]["filters"] = []

    with pytest.raises(ValidationError, match="search_type|source filters"):
        VisibilitySnapshot.model_validate(payload)


def test_persisted_ga4_ai_metric_cannot_be_relabelled_organic() -> None:
    source = _source(filters=("channel=AI Assistant",))
    metric = aggregate_series(_series(filters=()), source=source)
    payload = VisibilitySnapshot(
        project_id="example",
        canonical_domain="example.com",
        processed_at=NOW,
        deleted_at=NOW,
        metrics=(metric,),
        sources=(source,),
    ).model_dump(mode="json")
    payload["metrics"][0]["definitions"] = [
        "channel=Organic Search" if item == "channel=AI Assistant" else item
        for item in payload["metrics"][0]["definitions"]
    ]
    payload["sources"][0]["filters"] = ["channel=Organic Search"]

    with pytest.raises(ValidationError, match="AI Assistant|ai_assistant|channel"):
        VisibilitySnapshot.model_validate(payload)


def test_monthly_additive_full_coverage_rejects_missing_month() -> None:
    source = _source(date_range=DateRange(start=date(2026, 1, 1), end=date(2026, 3, 31)))
    points = (
        VisibilityMetricPoint(period_start=date(2026, 1, 1), value=10),
        VisibilityMetricPoint(period_start=date(2026, 3, 1), value=20),
    )
    with pytest.raises(ValueError, match="coverage|contiguous"):
        aggregate_series(_series(points=points), source=source)

    partial = aggregate_series(
        _series(state=DataState.PARTIAL, coverage=0.66, points=points),
        source=source,
    )
    assert partial.coverage == 0.66


def test_monthly_additive_complete_boundary_accepts_full_coverage() -> None:
    source = _source(date_range=DateRange(start=date(2026, 1, 1), end=date(2026, 3, 31)))
    metric = aggregate_series(
        _series(
            points=tuple(
                VisibilityMetricPoint(period_start=date(2026, month, 1), value=month)
                for month in (1, 2, 3)
            )
        ),
        source=source,
    )
    assert metric.coverage == 1


def test_visibility_resource_bounds_are_explicit() -> None:
    schema = VisibilityMetricSeriesInput.model_json_schema()["properties"]
    assert schema["points"]["maxItems"] == MAX_METRIC_POINTS
    assert schema["dimensions"]["maxItems"] == MAX_METRIC_DIMENSIONS
    assert schema["filters"]["maxItems"] == MAX_METRIC_FILTERS
    source_schema = SourceArtifactProvenance.model_json_schema()["properties"]
    assert source_schema["filters"]["maxItems"] == MAX_SOURCE_FILTERS


def test_metric_point_bound_accepts_512_and_rejects_513() -> None:
    common = {
        "metric_id": "bounded",
        "source_id": "logs",
        "metric": "logs.search_bot_requests",
        "unit": "requests",
        "state": DataState.AVAILABLE,
        "coverage": 1,
        "confidence": 1,
    }
    points = tuple(
        VisibilityMetricPoint(period_start=date(2020, 1, 1), value=index)
        for index in range(MAX_METRIC_POINTS)
    )
    assert len(VisibilityMetricSeriesInput(**common, points=points).points) == MAX_METRIC_POINTS
    with pytest.raises(ValidationError, match="512"):
        VisibilityMetricSeriesInput(
            **common,
            points=(*points, VisibilityMetricPoint(period_start=date(2020, 1, 2), value=1)),
        )
