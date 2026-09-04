import json
import socket
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ai_search_audit import performance_models as performance
from ai_search_audit.models import DataState
from ai_search_audit.performance_models import FieldPeriod, PerformanceMetric


def configuration_fields():
    return dict(
        form_factor="mobile",
        throttling_method=None,
        locale=None,
        screen_width=None,
        screen_height=None,
        device_scale_factor=None,
        cpu_slowdown_multiplier=None,
        rtt_ms=None,
        throughput_kbps=None,
    )


def test_observed_zero_is_distinct_from_unknown_metric():
    zero = PerformanceMetric(name="lcp", value=0.0, unit="ms")
    unknown = PerformanceMetric(name="lcp", value=None, unit="ms")
    assert zero.value == 0.0
    assert unknown.value is None
    assert zero != unknown


@pytest.mark.parametrize("value", [True, "12", -1, float("nan"), float("inf")])
def test_metric_rejects_non_strict_or_invalid_numbers(value):
    with pytest.raises(ValidationError):
        PerformanceMetric(name="lcp", value=value, unit="ms")


def test_cls_rejects_milliseconds():
    with pytest.raises(ValidationError):
        PerformanceMetric(name="cls", value=0.1, unit="ms")


def test_field_period_rejects_reversed_dates():
    with pytest.raises(ValidationError):
        FieldPeriod(first_date="2026-08-31", last_date="2026-08-01")


def test_configuration_preserves_explicit_unknowns():
    config = performance.LabConfiguration(**configuration_fields())
    assert config.model_dump() == configuration_fields() | {"local_runtime": None}


@pytest.mark.parametrize("field", list(configuration_fields()))
def test_configuration_requires_every_setting(field):
    fields = configuration_fields()
    del fields[field]
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**fields)


@pytest.mark.parametrize("field", ["screen_width", "screen_height"])
@pytest.mark.parametrize("value", [True, "12", 1.5, 0, -1])
def test_screen_dimensions_are_strict_positive_integers(field, value):
    fields = configuration_fields() | {field: value}
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**fields)


@pytest.mark.parametrize("field", ["device_scale_factor", "cpu_slowdown_multiplier"])
@pytest.mark.parametrize("value", [True, "12", 0, -1, float("nan"), float("inf")])
def test_positive_configuration_numbers(field, value):
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**(configuration_fields() | {field: value}))


@pytest.mark.parametrize("field", ["rtt_ms", "throughput_kbps"])
@pytest.mark.parametrize("value", [True, "12", -1, float("nan"), float("inf")])
def test_nonnegative_configuration_numbers(field, value):
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**(configuration_fields() | {field: value}))


@pytest.mark.parametrize("locale", ["", " ", "\t", "unknown", "UNAVAILABLE", "n/a", "x" * 201])
def test_unknown_or_invalid_locale_is_not_fabricated(locale):
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**(configuration_fields() | {"locale": locale}))


def test_configuration_records_known_settings_and_zero_network_dimensions():
    fields = dict(
        form_factor="desktop",
        throttling_method="provided",
        locale="en-US",
        screen_width=1440,
        screen_height=900,
        device_scale_factor=1.0,
        cpu_slowdown_multiplier=1.0,
        rtt_ms=0.0,
        throughput_kbps=0.0,
    )
    assert performance.LabConfiguration(**fields).model_dump() == fields | {"local_runtime": None}


@pytest.mark.parametrize("method", ["simulate", "devtools", "provided"])
def test_configuration_supports_documented_throttling_methods(method):
    assert (
        performance.LabConfiguration(
            **(configuration_fields() | {"throttling_method": method})
        ).throttling_method
        == method
    )


@pytest.mark.parametrize(
    "field,value", [("form_factor", "tablet"), ("throttling_method", "unknown")]
)
def test_configuration_rejects_unknown_categories(field, value):
    with pytest.raises(ValidationError):
        performance.LabConfiguration(**(configuration_fields() | {field: value}))


def attempt_fields():
    return dict(
        attempt_id="attempt-01",
        provider="pagespeed_insights",
        requested_url="https://perf.example/products?locale=en",
        device="mobile",
        started_at=datetime(2026, 9, 3, 10, tzinfo=UTC),
        ended_at=datetime(2026, 9, 3, 10, 1, tzinfo=UTC),
        state=DataState.AVAILABLE,
        reason=None,
        http_status=200,
    )


@pytest.mark.parametrize("provider", ["pagespeed_insights", "crux", "lighthouse_local"])
@pytest.mark.parametrize(
    "state",
    [
        DataState.AVAILABLE,
        DataState.PARTIAL,
        DataState.UNAVAILABLE,
        DataState.UNKNOWN,
        DataState.FAILED,
    ],
)
def test_each_adapter_preserves_its_explicit_outcome(provider, state):
    fields = attempt_fields() | dict(
        provider=provider,
        state=state,
        http_status=None,
        reason=None if state is DataState.AVAILABLE else "provider did not supply complete data",
    )
    attempt = performance.ProviderAttempt(**fields)
    assert attempt.state is state
    assert attempt.state is not DataState.FAILED or state is DataState.FAILED


@pytest.mark.parametrize("status", [None, 200, 204, 299])
def test_available_http_attempt_status(status):
    assert (
        performance.ProviderAttempt(**(attempt_fields() | {"http_status": status})).state
        is DataState.AVAILABLE
    )


@pytest.mark.parametrize(
    "state", [DataState.PARTIAL, DataState.UNAVAILABLE, DataState.UNKNOWN, DataState.FAILED]
)
@pytest.mark.parametrize("reason", [None, "", "  ", "x" * 201])
def test_nonavailable_attempt_requires_bounded_explicit_reason(state, reason):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | dict(state=state, reason=reason)))


@pytest.mark.parametrize("state", [DataState.AVAILABLE, DataState.PARTIAL])
@pytest.mark.parametrize("status", [100, 199, 300, 404, 429, 500, 599])
def test_non2xx_cannot_describe_available_or_partial(state, status):
    fields = attempt_fields() | dict(
        state=state, http_status=status, reason=None if state is DataState.AVAILABLE else "partial"
    )
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**fields)


@pytest.mark.parametrize("status", [100, 404, 429, 500, 599])
def test_unsuccessful_http_attempt_can_record_error_status(status):
    attempt = performance.ProviderAttempt(
        **(attempt_fields() | dict(state=DataState.FAILED, reason="HTTP error", http_status=status))
    )
    assert attempt.http_status == status


@pytest.mark.parametrize("status", [True, "200", 200.5, 99, 600])
def test_http_status_is_strict_and_bounded(status):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"http_status": status}))


def test_local_attempt_cannot_claim_http_status():
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"provider": "lighthouse_local"}))


def test_available_attempt_has_no_reason():
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"reason": "not complete"}))


@pytest.mark.parametrize("field", list(attempt_fields()))
def test_attempt_requires_explicit_fields(field):
    fields = attempt_fields()
    del fields[field]
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**fields)


@pytest.mark.parametrize(
    "field,value", [("provider", "other"), ("state", DataState.UNSUPPORTED), ("device", "tablet")]
)
def test_attempt_rejects_unsupported_categories(field, value):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {field: value}))


@pytest.mark.parametrize("attempt_id", ["", "../a", "a/b", "a.b", "_a", "a b", "a" * 97, "é"])
def test_attempt_requires_safe_identifier(attempt_id):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"attempt_id": attempt_id}))


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
def test_attempt_rejects_naive_timestamp(field):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {field: datetime(2026, 9, 3)}))


def test_attempt_rejects_reversed_time_order():
    fields = attempt_fields()
    fields["ended_at"] = fields["started_at"] - timedelta(seconds=1)
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**fields)


def test_attempt_allows_same_start_and_end():
    fields = attempt_fields()
    fields["ended_at"] = fields["started_at"]
    assert performance.ProviderAttempt(**fields).ended_at == fields["started_at"]


BAD_URLS = [
    "",
    "ftp://perf.example/",
    "https:///products",
    "https://",
    "//perf.example/",
    "https://user:secret@perf.example/",
    "https://user@perf.example/",
    "https://perf.example/#fragment",
    "https://perf.example/#",
    "https://perf.example\\@other.example/",
    " https://perf.example/",
    "https://perf.example/ ",
    "https://perf.example/\npath",
    "https://perf.example/\x00path",
    "https://perf.example/\x7f",
    "https://perf.example/\x85",
    "https://localhost/",
    "https://api.localhost/",
    "https://LOCALHOST./",
    "https://127.0.0.1/",
    "https://10.0.0.1/",
    "https://169.254.169.254/",
    "https://192.168.1.1/",
    "https://0.0.0.0/",
    "https://100.64.0.1/",
    "https://192.0.2.1/",
    "https://[::1]/",
    "https://[fe80::1]/",
    "https://[fc00::1]/",
    "https://[::ffff:127.0.0.1]/",
    "https://[::]/",
    "https://[2001:db8::1]/",
    "https://[fe80::1%25en0]/",
    "https://perf.example:65536/",
    "https://perf.example:wrong/",
    "https://perf.example:/",
    "https://perf.example:-1/",
    "https://[2606:4700:4700::1111]junk/",
    "https://[2606:4700:4700::1111]:/",
    "https://perf.example]/",
    "https://[perf.example]/",
    "https://per%66.example/",
    "https://-perf.example/",
    "https://perf..example/",
    "https://127.1/",
    "https://2130706433/",
    "https://0x7f000001/",
    "https://0177.0.0.1/",
    "https://perf.example/" + "x" * 4096,
]


@pytest.mark.parametrize("url", BAD_URLS)
def test_attempt_rejects_unsafe_or_malformed_raw_url(url):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"requested_url": url}))


@pytest.mark.parametrize(
    "url",
    [
        "http://perf.example/",
        "https://perf.example:8443/path?a=1&b=2",
        "HTTPS://PERF.example/%20path?q=%2F",
        "https://8.8.8.8/",
        "https://[2606:4700:4700::1111]/",
        "https://perf.example./",
    ],
)
def test_attempt_validates_without_dns_and_preserves_raw_url(url, monkeypatch):
    def forbidden_dns(*args, **kwargs):
        pytest.fail("performance DTO validation must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_dns)
    monkeypatch.setattr(socket, "gethostbyname", forbidden_dns)
    assert (
        performance.ProviderAttempt(**(attempt_fields() | {"requested_url": url})).requested_url
        == url
    )


def binding_fields():
    return dict(
        project_id="performance-fixture",
        source_version="source-v1",
        audit_id="audit-20260903",
        report_locale="en",
        domain="perf.example",
        source_sha256="b" * 64,
    )


def lab_fields():
    return dict(
        binding=binding_fields(),
        evidence_id="lab-01",
        attempt_id="attempt-01",
        provider="pagespeed_insights",
        requested_url="https://perf.example/product?lang=en",
        final_url="https://perf.example/product/?lang=en",
        device="mobile",
        observed_at=datetime(2026, 9, 3, 10, 1, tzinfo=UTC),
        lighthouse_version="13.0.0",
        chrome_version=None,
        configuration=configuration_fields(),
        performance_score=None,
        metrics=[dict(name="lcp", value=1234.0, unit="ms")],
    )


def test_remote_lab_success_uses_typed_metrics_and_explicit_unknowns():
    # Local runtime provenance and actual Chrome version are mandatory; covered
    # by the native Lighthouse projection tests rather than a PSI-shaped fixture.
    lab = performance.LabMeasurement(**lab_fields())
    assert lab.kind == "lab"
    assert lab.provider == "pagespeed_insights"
    assert lab.metrics[0].value == 1234.0
    assert lab.chrome_version is None
    assert lab.performance_score is None
    assert lab.warnings == ()


@pytest.mark.parametrize("field", list(lab_fields()))
def test_lab_required_fields_are_not_silently_filled(field):
    fields = lab_fields()
    del fields[field]
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**fields)


@pytest.mark.parametrize(
    "metrics,score",
    [
        ([], 0.0),
        ([dict(name="lcp", value=None, unit="ms")], 0.0),
        ([dict(name="cls", value=0.0, unit="unitless")], None),
    ],
)
def test_lab_allows_individual_absence_null_and_observed_zero(metrics, score):
    lab = performance.LabMeasurement(
        **(lab_fields() | dict(metrics=metrics, performance_score=score))
    )
    assert lab.performance_score == score
    assert lab.metrics == tuple(PerformanceMetric(**metric) for metric in metrics)


@pytest.mark.parametrize("metrics", [[], [dict(name="lcp", value=None, unit="ms")]])
def test_lab_without_any_numeric_data_belongs_in_attempt(metrics):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {"metrics": metrics}))


@pytest.mark.parametrize("score", [True, "12", -1, 100.1, float("nan"), float("inf")])
def test_lab_score_is_a_strict_finite_percentage(score):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {"performance_score": score}))


def test_lab_records_all_five_lab_metrics_and_maximum_score():
    metrics = [
        dict(name=name, value=0.0, unit="unitless" if name == "cls" else "ms")
        for name in ["lcp", "cls", "fcp", "tbt", "speed_index"]
    ]
    lab = performance.LabMeasurement(
        **(lab_fields() | dict(metrics=metrics, performance_score=100.0))
    )
    assert len(lab.metrics) == 5
    assert lab.performance_score == 100.0


@pytest.mark.parametrize(
    "metrics",
    [
        [dict(name="inp", value=5.0, unit="ms")],
        [dict(name="lcp", value=1.0, unit="ms"), dict(name="lcp", value=None, unit="ms")],
        [dict(name="lcp", value=1.0, unit="ms")] * 6,
    ],
)
def test_lab_rejects_field_only_duplicate_or_excess_metrics(metrics):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {"metrics": metrics}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "crux"),
        ("kind", "field"),
        ("device", "desktop"),
        ("evidence_id", "../lab"),
        ("attempt_id", "../attempt"),
        ("observed_at", datetime(2026, 9, 3)),
    ],
)
def test_lab_rejects_inconsistent_identity_metadata(field, value):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {field: value}))


@pytest.mark.parametrize("field", ["lighthouse_version", "chrome_version"])
@pytest.mark.parametrize("value", ["", " ", "x" * 201, "unknown", "n/a", "UNAVAILABLE"])
def test_lab_rejects_blank_or_fabricated_versions(field, value):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {field: value}))


def test_lab_requires_known_lighthouse_version_but_chrome_can_be_unknown():
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {"lighthouse_version": None}))
    assert (
        performance.LabMeasurement(
            **(lab_fields() | {"chrome_version": "140.0.0.0"})
        ).chrome_version
        == "140.0.0.0"
    )


@pytest.mark.parametrize("warnings", [[""], [" "], ["x" * 1001], ["notice"] * 31])
def test_lab_warnings_are_bounded_nonblank_text(warnings):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {"warnings": warnings}))


def test_lab_warning_limits_include_boundary():
    lab = performance.LabMeasurement(**(lab_fields() | {"warnings": ["x" * 1000] * 30}))
    assert len(lab.warnings) == 30


@pytest.mark.parametrize("field", ["requested_url", "final_url"])
@pytest.mark.parametrize("url", BAD_URLS)
def test_lab_url_fields_share_raw_url_validation(field, url):
    with pytest.raises(ValidationError):
        performance.LabMeasurement(**(lab_fields() | {field: url}))


def field_fields():
    return dict(
        binding=binding_fields(),
        evidence_id="field-01",
        attempt_id="attempt-02",
        requested_url="https://perf.example/product?lang=en",
        record_key="https://perf.example/product?lang=en",
        scope="url",
        device="mobile",
        observed_at=datetime(2026, 9, 3, 10, 1, tzinfo=UTC),
        period=dict(first_date="2026-08-01", last_date="2026-08-28"),
        metrics=[dict(name="lcp", value=1600.0, unit="ms")],
    )


@pytest.mark.parametrize(
    "scope,key",
    [
        ("url", "https://perf.example/product?lang=en"),
        ("origin", "https://perf.example"),
        ("origin", "https://perf.example/"),
    ],
)
def test_crux_success_preserves_url_versus_origin(scope, key):
    field = performance.FieldMeasurement(**(field_fields() | dict(scope=scope, record_key=key)))
    assert field.kind == "field"
    assert field.provider == "crux"
    assert field.percentile == 75
    assert field.scope == scope
    assert field.record_key == key
    assert field.requested_url == "https://perf.example/product?lang=en"


@pytest.mark.parametrize("field", list(field_fields()))
def test_field_requires_explicit_observation_fields(field):
    fields = field_fields()
    del fields[field]
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**fields)


def test_field_records_core_web_vitals_with_null_and_zero():
    metrics = [
        dict(name="lcp", value=None, unit="ms"),
        dict(name="inp", value=0.0, unit="ms"),
        dict(name="cls", value=0.0, unit="unitless"),
    ]
    field = performance.FieldMeasurement(**(field_fields() | {"metrics": metrics}))
    assert tuple(metric.value for metric in field.metrics) == (None, 0.0, 0.0)


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        [dict(name="lcp", value=None, unit="ms")],
        [dict(name=name, value=1.0, unit="ms") for name in ["lcp", "lcp"]],
        [dict(name="lcp", value=1.0, unit="ms")] * 4,
        [dict(name="fcp", value=1.0, unit="ms")],
        [dict(name="tbt", value=1.0, unit="ms")],
        [dict(name="speed_index", value=1.0, unit="ms")],
    ],
)
def test_field_rejects_empty_duplicate_excess_or_lab_only_metrics(metrics):
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**(field_fields() | {"metrics": metrics}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "pagespeed_insights"),
        ("provider", "lighthouse_local"),
        ("kind", "lab"),
        ("scope", "site"),
        ("percentile", 50),
        ("percentile", "75"),
        ("device", "tablet"),
        ("evidence_id", "../field"),
        ("attempt_id", "../attempt"),
        ("observed_at", datetime(2026, 9, 3)),
    ],
)
def test_field_rejects_wrong_identity_metadata(field, value):
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**(field_fields() | {field: value}))


@pytest.mark.parametrize(
    "key",
    [
        "https://perf.example/products",
        "https://perf.example/?a=1",
        "https://perf.example?",
        "https://perf.example/?",
    ],
)
def test_origin_record_cannot_contain_path_or_query(key):
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**(field_fields() | dict(scope="origin", record_key=key)))


@pytest.mark.parametrize("scope", ["url", "origin"])
@pytest.mark.parametrize(
    "key", ["http://perf.example", "https://other.example", "https://perf.example:8443"]
)
def test_field_key_cannot_cross_requested_origin(scope, key):
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**(field_fields() | dict(scope=scope, record_key=key)))


@pytest.mark.parametrize(
    "requested,key",
    [
        ("https://perf.example:443/product", "https://PERF.example/"),
        ("http://perf.example/product", "http://perf.example:80"),
        ("https://perf.example:8443/product", "https://perf.example:8443/"),
    ],
)
def test_field_origin_comparison_uses_scheme_host_and_effective_port(requested, key):
    field = performance.FieldMeasurement(
        **(field_fields() | dict(scope="origin", requested_url=requested, record_key=key))
    )
    assert field.record_key == key


def test_url_record_preserves_vendor_path_and_query_for_later_attestation():
    key = "https://perf.example/product/?lang=en&variant=1"
    field = performance.FieldMeasurement(**(field_fields() | {"record_key": key}))
    assert field.scope == "url"
    assert field.record_key == key


@pytest.mark.parametrize(
    "observed,end,valid",
    [
        ("2026-09-03T00:30:00+02:00", "2026-09-03", False),
        ("2026-09-03T00:30:00+02:00", "2026-09-02", True),
        ("2026-09-03T23:30:00-02:00", "2026-09-04", True),
        ("2026-09-03T23:30:00Z", "2026-09-04", False),
    ],
)
def test_field_period_cannot_extend_past_utc_observation_date(observed, end, valid):
    fields = field_fields() | dict(
        observed_at=observed, period=dict(first_date="2026-08-01", last_date=end)
    )
    if valid:
        assert performance.FieldMeasurement(**fields).period.last_date.isoformat() == end
    else:
        with pytest.raises(ValidationError):
            performance.FieldMeasurement(**fields)


@pytest.mark.parametrize("field", ["requested_url", "record_key"])
@pytest.mark.parametrize("url", BAD_URLS)
def test_field_urls_share_raw_validation(field, url):
    with pytest.raises(ValidationError):
        performance.FieldMeasurement(**(field_fields() | {field: url}))


@pytest.mark.parametrize(
    "model,factory",
    [
        ("ProviderAttempt", attempt_fields),
        ("LabMeasurement", lab_fields),
        ("FieldMeasurement", field_fields),
        ("LabConfiguration", configuration_fields),
        ("PerformanceMetric", lambda: dict(name="cls", value=0.0, unit="unitless")),
        ("FieldPeriod", lambda: dict(first_date="2026-08-01", last_date="2026-08-28")),
    ],
)
@pytest.mark.parametrize("extra", ["raw_provider_payload", "credentials", "narrative", "arbitrary"])
def test_dtos_forbid_extra_payloads(model, factory, extra):
    with pytest.raises(ValidationError):
        getattr(performance, model)(**(factory() | {extra: {"unexpected": "value"}}))


@pytest.mark.parametrize(
    "model,factory,field,value",
    [
        ("ProviderAttempt", attempt_fields, "reason", "changed"),
        ("LabMeasurement", lab_fields, "performance_score", 50.0),
        ("FieldMeasurement", field_fields, "scope", "origin"),
        ("LabConfiguration", configuration_fields, "locale", "pl-PL"),
        ("PerformanceMetric", lambda: dict(name="cls", value=0.0, unit="unitless"), "value", 1.0),
        (
            "FieldPeriod",
            lambda: dict(first_date="2026-08-01", last_date="2026-08-28"),
            "last_date",
            "2026-08-29",
        ),
    ],
)
def test_each_dto_is_frozen(model, factory, field, value):
    dto = getattr(performance, model)(**factory())
    with pytest.raises(ValidationError):
        setattr(dto, field, value)


def test_lab_input_lists_detach_and_nested_binding_configuration_metrics_warnings_are_frozen():
    fields = lab_fields() | {"warnings": ["Observed warning"]}
    lab = performance.LabMeasurement(**fields)
    fields["binding"]["domain"] = "other.example"
    fields["configuration"]["locale"] = "pl-PL"
    fields["metrics"][0]["value"] = 9000.0
    fields["metrics"].append(dict(name="tbt", value=1.0, unit="ms"))
    fields["warnings"].append("Changed")
    assert lab.binding.domain == "perf.example"
    assert lab.configuration.locale is None
    assert len(lab.metrics) == 1
    assert lab.metrics[0].value == 1234.0
    assert lab.warnings == ("Observed warning",)
    for nested, field, value in [
        (lab.binding, "domain", "other.example"),
        (lab.configuration, "locale", "pl-PL"),
        (lab.metrics[0], "value", 9000.0),
    ]:
        with pytest.raises(ValidationError):
            setattr(nested, field, value)
    with pytest.raises(TypeError):
        lab.metrics[0] = lab.metrics[0]
    with pytest.raises(TypeError):
        lab.warnings[0] = "changed"


def test_field_input_detaches_and_nested_period_and_metrics_are_frozen():
    fields = field_fields()
    field = performance.FieldMeasurement(**fields)
    fields["period"]["last_date"] = "2026-09-03"
    fields["metrics"].clear()
    assert field.period.last_date.isoformat() == "2026-08-28"
    assert len(field.metrics) == 1
    for nested, key, value in [
        (field.period, "last_date", "2026-09-03"),
        (field.binding, "domain", "other.example"),
        (field.metrics[0], "value", 0.0),
    ]:
        with pytest.raises(ValidationError):
            setattr(nested, key, value)
    with pytest.raises(TypeError):
        field.metrics[0] = field.metrics[0]


@pytest.mark.parametrize(
    "model,factory",
    [
        ("ProviderAttempt", attempt_fields),
        ("LabMeasurement", lab_fields),
        ("FieldMeasurement", field_fields),
    ],
)
def test_json_round_trip_preserves_values_and_nested_immutability(model, factory):
    cls = getattr(performance, model)
    original = cls(**factory())
    restored = cls.model_validate_json(original.model_dump_json())
    assert restored == original
    with pytest.raises(ValidationError):
        restored.device = "desktop"
    if hasattr(restored, "metrics"):
        assert isinstance(restored.metrics, tuple)
        with pytest.raises(ValidationError):
            restored.metrics[0].value = 0.0


@pytest.mark.parametrize(
    "model,factory,changes",
    [
        ("ProviderAttempt", attempt_fields, {"state": "UNKNOWN", "reason": None}),
        ("ProviderAttempt", attempt_fields, {"http_status": 500}),
        ("ProviderAttempt", attempt_fields, {"requested_url": "https://user:secret@perf.example/"}),
        ("LabMeasurement", lab_fields, {"metrics": [{"name": "inp", "value": 0.0, "unit": "ms"}]}),
        ("LabMeasurement", lab_fields, {"performance_score": 101.0}),
        ("LabMeasurement", lab_fields, {"device": "desktop"}),
        ("LabMeasurement", lab_fields, {"final_url": "https://127.0.0.1/"}),
        ("FieldMeasurement", field_fields, {"scope": "origin"}),
        ("FieldMeasurement", field_fields, {"record_key": "https://other.example/"}),
        (
            "FieldMeasurement",
            field_fields,
            {"period": {"first_date": "2026-08-01", "last_date": "2026-09-04"}},
        ),
        ("FieldMeasurement", field_fields, {"raw_provider_payload": {}}),
    ],
)
def test_json_reloading_runs_invariants_instead_of_trusting_serialized_input(
    model, factory, changes
):
    cls = getattr(performance, model)
    data = json.loads(cls(**factory()).model_dump_json())
    data.update(changes)
    with pytest.raises(ValidationError):
        cls.model_validate_json(json.dumps(data))


def test_unchecked_model_copy_is_not_a_trust_boundary():
    original = performance.LabMeasurement(**lab_fields())
    unchecked = original.model_copy(update={"performance_score": -1.0})
    assert unchecked.performance_score == -1.0
    with pytest.raises(ValidationError):
        performance.LabMeasurement.model_validate_json(unchecked.model_dump_json())


@pytest.mark.parametrize(
    "url",
    [
        "https://perf.example../",
        "https://foo。localhost/",
        "https://ｌｏｃａｌｈｏｓｔ/",
        "https://１２７。０。０。１/",
        "https://perf.example。。/",
        "https://per%66。example/",
    ],
)
def test_raw_authority_and_idna_cannot_hide_local_hosts_or_empty_labels(url):
    with pytest.raises(ValidationError):
        performance.ProviderAttempt(**(attempt_fields() | {"requested_url": url}))


def test_ipv6_origin_comparison_preserves_equivalent_raw_spellings():
    fields = field_fields() | dict(
        requested_url="https://[2606:4700:4700:0:0:0:0:1111]/page",
        record_key="https://[2606:4700:4700::1111]:443",
        scope="origin",
    )
    result = performance.FieldMeasurement(**fields)
    assert result.record_key == fields["record_key"]
    assert result.requested_url == fields["requested_url"]


@pytest.mark.parametrize("name", ["lcp", "inp", "fcp", "tbt", "speed_index"])
def test_all_timing_metrics_require_milliseconds(name):
    with pytest.raises(ValidationError):
        PerformanceMetric(name=name, value=1.0, unit="unitless")


@pytest.mark.parametrize(
    "fields",
    [
        dict(name="lcp", unit="ms"),
        dict(name="other", value=1.0, unit="ms"),
        dict(name="lcp", value=1.0, unit="seconds"),
    ],
)
def test_metric_cannot_omit_unknown_value_or_invent_name_or_unit(fields):
    with pytest.raises(ValidationError):
        PerformanceMetric(**fields)


def test_field_period_allows_single_day():
    period = FieldPeriod(first_date="2026-09-03", last_date="2026-09-03")
    assert period.first_date == period.last_date


@pytest.mark.parametrize(
    "model,factory", [("LabMeasurement", lab_fields), ("FieldMeasurement", field_fields)]
)
def test_measurement_validation_and_json_reload_do_not_resolve_dns(model, factory, monkeypatch):
    def forbidden_dns(*args, **kwargs):
        pytest.fail("measurement validation must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_dns)
    monkeypatch.setattr(socket, "gethostbyname", forbidden_dns)
    cls = getattr(performance, model)
    original = cls(**factory())
    assert cls.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize("scope", ["url", "origin"])
@pytest.mark.parametrize("json_input", [False, True])
def test_field_origin_keeps_distinct_unicode_and_ascii_domains_separate(scope, json_input):
    fields = field_fields() | dict(
        requested_url="https://straße.example/page",
        record_key="https://strasse.example/",
        scope=scope,
    )
    with pytest.raises(ValidationError, match="requested URL origin"):
        if json_input:
            performance.FieldMeasurement.model_validate_json(json.dumps(fields, default=str))
        else:
            performance.FieldMeasurement(**fields)


@pytest.mark.parametrize("scope", ["url", "origin"])
@pytest.mark.parametrize("json_input", [False, True])
def test_field_origin_accepts_correct_unicode_punycode_equivalence(scope, json_input):
    fields = field_fields() | dict(
        requested_url="https://straße.example/page",
        record_key="https://xn--strae-oqa.example/",
        scope=scope,
    )
    if json_input:
        field = performance.FieldMeasurement.model_validate_json(json.dumps(fields, default=str))
    else:
        field = performance.FieldMeasurement(**fields)
    assert field.requested_url == fields["requested_url"]
    assert field.record_key == fields["record_key"]
    assert performance.FieldMeasurement.model_validate_json(field.model_dump_json()) == field


@pytest.mark.parametrize("host", ["０x08080808", "１３４７４４０７２", "８.８", "８.８.８.８"])
@pytest.mark.parametrize("json_input", [False, True])
def test_unicode_host_cannot_be_repaired_into_ip_literal(host, json_input):
    fields = attempt_fields() | {"requested_url": f"https://{host}/"}
    with pytest.raises(ValidationError):
        if json_input:
            performance.ProviderAttempt.model_validate_json(json.dumps(fields, default=str))
        else:
            performance.ProviderAttempt(**fields)


@pytest.mark.parametrize("host", ["8.8.8.8", "[2606:4700:4700::1111]"])
@pytest.mark.parametrize("json_input", [False, True])
def test_public_ascii_ip_literals_remain_valid_without_repair(host, json_input):
    fields = attempt_fields() | {"requested_url": f"https://{host}/"}
    if json_input:
        attempt = performance.ProviderAttempt.model_validate_json(json.dumps(fields, default=str))
    else:
        attempt = performance.ProviderAttempt(**fields)
    assert attempt.requested_url == fields["requested_url"]
