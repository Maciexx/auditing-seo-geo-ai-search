import json
from datetime import UTC, date, datetime

import pytest

from ai_search_audit import performance_normalizers as normalizers
from ai_search_audit.diagnostic_models import DiagnosticBinding
from ai_search_audit.performance_http import PerformanceRequest
from ai_search_audit.performance_normalizers import normalize_crux, normalize_psi

URL = "https://perf.example/article?a=1"
OBSERVED = datetime(2026, 9, 3, tzinfo=UTC)


def binding():
    return DiagnosticBinding(
        project_id="example",
        source_version="v1",
        audit_id="audit1",
        report_locale="en",
        domain="perf.example",
        source_sha256="a" * 64,
    )


def request(provider="pagespeed_insights", **kwargs):
    return PerformanceRequest.model_validate(
        dict(provider=provider, requested_url=URL, device="mobile", locale="en") | kwargs
    )


def psi_payload():
    return {
        "lighthouseResult": {
            "requestedUrl": URL,
            "finalUrl": URL,
            "fetchTime": "2026-09-03T00:00:00Z",
            "lighthouseVersion": "12.8.2",
            "configSettings": {"formFactor": "mobile"},
            "categories": {"performance": {"score": 0.75}},
            "audits": {
                "largest-contentful-paint": {"numericValue": 1234, "numericUnit": "millisecond"}
            },
        }
    }


def crux_payload():
    return {
        "record": {
            "key": {"url": URL, "formFactor": "PHONE"},
            "collectionPeriod": {
                "firstDate": {"year": 2026, "month": 8, "day": 1},
                "lastDate": {"year": 2026, "month": 8, "day": 28},
            },
            "metrics": {
                "largest_contentful_paint": {"percentiles": {"p75": 1234}},
                "interaction_to_next_paint": {"percentiles": {"p75": 80}},
                "cumulative_layout_shift": {"percentiles": {"p75": "0.00"}},
            },
        }
    }


def psi(payload=None, **kwargs):
    return normalize_psi(
        psi_payload() if payload is None else payload,
        **(
            dict(
                request=request(),
                binding=binding(),
                attempt_id="attempt1",
                evidence_id="evidence1",
                allowed_final_urls=(),
            )
            | kwargs
        ),
    )


def crux(payload=None, **kwargs):
    return normalize_crux(
        crux_payload() if payload is None else payload,
        **(
            dict(
                request=request("crux"),
                binding=binding(),
                attempt_id="attempt1",
                evidence_id="evidence1",
                observed_at=OBSERVED,
                page_url=URL,
            )
            | kwargs
        ),
    )


def test_psi_projects_numeric_metric_and_scales_score():
    result = psi()
    assert result.performance_score == 75.0
    assert [(m.name, m.value, m.unit) for m in result.metrics] == [
        ("lcp", 1234.0, "ms"),
        ("cls", None, "unitless"),
        ("fcp", None, "ms"),
        ("tbt", None, "ms"),
        ("speed_index", None, "ms"),
    ]
    assert result.requested_url == result.final_url == URL
    assert result.lighthouse_version == "12.8.2"
    assert result.observed_at == OBSERVED


def test_crux_projects_percentile_units_dates_and_observed_zero():
    result = crux()
    assert [(m.name, m.value, m.unit) for m in result.metrics] == [
        ("lcp", 1234.0, "ms"),
        ("inp", 80.0, "ms"),
        ("cls", 0.0, "unitless"),
    ]
    assert result.period.first_date == date(2026, 8, 1)
    assert result.period.last_date == date(2026, 8, 28)
    assert result.requested_url == result.record_key == URL


def assert_error(code, call):
    with pytest.raises(normalizers.NormalizationError) as caught:
        call()
    assert caught.value.code == code
    assert str(caught.value) == code
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "body",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e999}',
        b'{"a":-1e999}',
        b"[]",
        b"null",
        b'{"a":"\xff"}',
        b"{",
    ],
)
def test_json_rejects_ambiguous_nonfinite_or_malformed_payload(body):
    assert_error("malformed_response", lambda: normalizers.decode_provider_json(body))


def test_json_accepts_finite_nested_values():
    assert normalizers.decode_provider_json(b'{"a":[0,1.5,null,{"b":true}]}') == {
        "a": [0, 1.5, None, {"b": True}]
    }


def test_json_suppresses_recursion_errors(monkeypatch):
    def recursion_error(*args, **kwargs):
        raise RecursionError("sensitive provider body")

    monkeypatch.setattr(json, "loads", recursion_error)
    assert_error("malformed_response", lambda: normalizers.decode_provider_json(b"{}"))


@pytest.mark.parametrize("value", [True, "12", -1, float("nan"), float("inf"), 10**400])
def test_psi_rejects_invalid_metric_numbers(value):
    payload = psi_payload()
    payload["lighthouseResult"]["audits"]["largest-contentful-paint"]["numericValue"] = value
    assert_error("invalid_metrics", lambda: psi(payload))


@pytest.mark.parametrize("value", [True, "0.75", -0.1, 1.01, float("nan"), 10**400])
def test_psi_rejects_invalid_score(value):
    payload = psi_payload()
    payload["lighthouseResult"]["categories"]["performance"]["score"] = value
    assert_error("invalid_metrics", lambda: psi(payload))


@pytest.mark.parametrize("unit", [None, "ms", "unitless", 1])
def test_psi_requires_provider_numeric_unit_for_known_metric(unit):
    payload = psi_payload()
    payload["lighthouseResult"]["audits"]["largest-contentful-paint"]["numericUnit"] = unit
    assert_error("invalid_metrics", lambda: psi(payload))


@pytest.mark.parametrize(
    "name,value",
    [
        ("largest_contentful_paint", True),
        ("largest_contentful_paint", "12"),
        ("largest_contentful_paint", 12.0),
        ("interaction_to_next_paint", -1),
        ("interaction_to_next_paint", 10**400),
        ("cumulative_layout_shift", 0.1),
        ("cumulative_layout_shift", " 0.1"),
        ("cumulative_layout_shift", "1e2"),
        ("cumulative_layout_shift", "NaN"),
        ("cumulative_layout_shift", "-0.1"),
        ("cumulative_layout_shift", "9" * 400),
    ],
)
def test_crux_rejects_invalid_percentiles(name, value):
    payload = crux_payload()
    payload["record"]["metrics"][name]["percentiles"]["p75"] = value
    assert_error("invalid_metrics", lambda: crux(payload))


def test_missing_and_null_metrics_stay_unknown_and_display_values_are_ignored():
    payload = psi_payload()
    payload["lighthouseResult"]["audits"] = {
        "largest-contentful-paint": {"numericValue": None, "displayValue": "1.2 s"}
    }
    assert all(metric.value is None for metric in psi(payload).metrics)
    payload = crux_payload()
    payload["record"]["metrics"] = {"cumulative_layout_shift": {"percentiles": {"p75": "0"}}}
    assert [metric.value for metric in crux(payload).metrics] == [None, None, 0.0]


@pytest.mark.parametrize("provider", ["psi", "crux"])
def test_empty_observations_are_no_data(provider):
    payload = psi_payload() if provider == "psi" else crux_payload()
    if provider == "psi":
        payload["lighthouseResult"].pop("categories")
        payload["lighthouseResult"]["audits"] = {}
    else:
        payload["record"]["metrics"] = {}
    assert_error("no_data", lambda: (psi if provider == "psi" else crux)(payload))


def test_psi_observed_zero_is_not_unknown():
    payload = psi_payload()
    payload["lighthouseResult"]["categories"]["performance"]["score"] = 0
    payload["lighthouseResult"]["audits"]["largest-contentful-paint"]["numericValue"] = 0
    result = psi(payload)
    assert result.performance_score == result.metrics[0].value == 0.0


def test_psi_projects_only_allowlisted_configuration_and_host_browser_version():
    payload = psi_payload()
    result = payload["lighthouseResult"]
    result["configSettings"].update(
        locale="en-US",
        throttlingMethod="simulate",
        screenEmulation=dict(width=390, height=844, deviceScaleFactor=2),
        throttling=dict(cpuSlowdownMultiplier=4, rttMs=0, throughputKbps=1600),
        arbitrarySetting="DO_NOT_COPY",
    )
    result["environment"] = dict(
        hostUserAgent="Mozilla/5.0 HeadlessChrome/130.0.1.2 Safari/537.36",
        networkUserAgent="Chrome/999.0.0.0",
    )
    result["runWarnings"] = ["Synthetic warning"]
    result["title"] = "DO_NOT_COPY"
    result["audits"]["screenshots"] = {"details": "DO_NOT_COPY"}
    payload["loadingExperience"] = {"DO_NOT_COPY": True}
    measurement = psi(payload)
    assert measurement.configuration.model_dump() == dict(
        local_runtime=None,
        form_factor="mobile",
        locale="en-US",
        throttling_method="simulate",
        screen_width=390,
        screen_height=844,
        device_scale_factor=2.0,
        cpu_slowdown_multiplier=4.0,
        rtt_ms=0.0,
        throughput_kbps=1600.0,
    )
    assert measurement.chrome_version == "130.0.1.2"
    assert measurement.warnings == ("Synthetic warning",)
    assert "DO_NOT_COPY" not in measurement.model_dump_json()


@pytest.mark.parametrize(
    "user_agent",
    [
        "Mozilla Safari/537.36",
        "Chrome/not-a-version",
        "Chrome/130.0.0.0 HeadlessChrome/131.0.0.0",
        "Chrome/130.0.0.0 Chrome/130.0.0.0",
    ],
)
def test_ambiguous_or_unknown_host_browser_is_null(user_agent):
    payload = psi_payload()
    payload["lighthouseResult"]["environment"] = {
        "hostUserAgent": user_agent,
        "networkUserAgent": "Chrome/999.0.0.0",
    }
    assert psi(payload).chrome_version is None


@pytest.mark.parametrize(
    "screen",
    [
        dict(width=0, height=0, deviceScaleFactor=0),
        dict(disabled=True, width=390, height=844, deviceScaleFactor=2),
    ],
)
def test_inactive_screen_overrides_are_null(screen):
    payload = psi_payload()
    payload["lighthouseResult"]["configSettings"]["screenEmulation"] = screen
    config = psi(payload).configuration
    assert (config.screen_width, config.screen_height, config.device_scale_factor) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("screenEmulation", None),
        ("screenEmulation", []),
        ("screenEmulation", {"disabled": 1}),
        ("screenEmulation", {"disabled": None}),
        ("screenEmulation", {"disabled": True, "width": -1}),
        ("screenEmulation", {"disabled": True, "height": "844"}),
        ("screenEmulation", {"width": 1.5}),
        ("screenEmulation", {"deviceScaleFactor": True}),
        ("throttling", {"cpuSlowdownMultiplier": 0}),
        ("throttling", {"rttMs": "12"}),
        ("throttling", {"throughputKbps": 10**400}),
        ("throttling", None),
        ("locale", "unknown"),
        ("locale", 123),
        ("throttlingMethod", "invented"),
    ],
)
def test_invalid_supplied_configuration_is_rejected(field, value):
    payload = psi_payload()
    payload["lighthouseResult"]["configSettings"][field] = value
    assert_error("malformed_response", lambda: psi(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("fetchTime", "2026-09-03T00:00:00"),
        ("fetchTime", 1788393600),
        ("fetchTime", "1788393600"),
        ("lighthouseVersion", "unknown"),
        ("lighthouseVersion", "x" * 201),
        ("runWarnings", ["x" * 1001]),
        ("runWarnings", ["warn"] * 31),
        ("runWarnings", "warning"),
        ("runWarnings", [123]),
        ("environment", []),
        ("environment", {"hostUserAgent": 123}),
    ],
)
def test_invalid_psi_metadata_is_rejected_without_payload_details(field, value):
    payload = psi_payload()
    payload["lighthouseResult"][field] = value
    assert_error("malformed_response", lambda: psi(payload))


@pytest.mark.parametrize(
    "settings",
    [
        {"formFactor": "desktop"},
        {"formFactor": "tablet"},
        {"formFactor": "mobile", "emulatedFormFactor": "desktop"},
        {},
    ],
)
def test_psi_rejects_missing_or_conflicting_device(settings):
    payload = psi_payload()
    payload["lighthouseResult"]["configSettings"] = settings
    assert_error("device_mismatch", lambda: psi(payload))


def test_psi_accepts_legacy_device_and_explicit_desktop():
    payload = psi_payload()
    payload["lighthouseResult"]["configSettings"] = {"emulatedFormFactor": "mobile"}
    assert psi(payload).configuration.form_factor == "mobile"
    payload["lighthouseResult"]["configSettings"] = {"formFactor": "desktop"}
    assert psi(payload, request=request(device="desktop")).device == "desktop"


@pytest.mark.parametrize("runtime_error", [None, {"code": "NO_ERROR"}])
def test_psi_accepts_no_runtime_error(runtime_error):
    payload = psi_payload()
    payload["lighthouseResult"]["runtimeError"] = runtime_error
    assert psi(payload).performance_score == 75.0


@pytest.mark.parametrize("runtime_error", [{"code": "FAILED", "message": "secret"}, {}, "secret"])
def test_psi_rejects_runtime_error(runtime_error):
    payload = psi_payload()
    payload["lighthouseResult"]["runtimeError"] = runtime_error
    assert_error("runtime_error", lambda: psi(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("requestedUrl", "https://other.example/article?a=1"),
        ("requestedUrl", "https://perf.example/article?a=2"),
        ("finalUrl", "https://perf.example/redirect"),
        ("finalUrl", "http://127.1/"),
    ],
)
def test_psi_rejects_unbound_requested_or_final_target(field, value):
    payload = psi_payload()
    payload["lighthouseResult"][field] = value
    assert_error("target_mismatch", lambda: psi(payload))


def test_psi_preserves_raw_urls_and_accepts_explicit_final_target_only():
    payload = psi_payload()
    payload["lighthouseResult"]["requestedUrl"] = "https://PERF.example:443/article?a=1"
    payload["lighthouseResult"]["finalUrl"] = "https://perf.example/redirect"
    result = psi(payload, allowed_final_urls=("https://perf.example/redirect",))
    assert result.requested_url == "https://PERF.example:443/article?a=1"
    assert result.final_url == "https://perf.example/redirect"


def test_url_key_compares_modern_hosts_effective_ports_and_exact_path_query():
    key = normalizers.measurement_url_key
    assert key("https://PERF.example:443") == key("https://perf.example/")
    assert key("https://straße.example/") == key("https://xn--strae-oqa.example/")
    assert key("https://straße.example/") != key("https://strasse.example/")
    assert key("https://perf.example/a%2Fb?q=%2F") != key("https://perf.example/a/b?q=/")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.1/",
        "https://user:pass@perf.example/",
        "https://perf.example/#fragment",
        "https://perf.example/\n",
        "https://perf.example:/",
    ],
)
def test_url_key_validates_raw_public_url_before_comparison(url):
    assert_error("target_mismatch", lambda: normalizers.measurement_url_key(url))


@pytest.mark.parametrize(
    "key",
    [
        {"url": URL},
        {"url": URL, "formFactor": "TABLET"},
        {"url": URL, "formFactor": "DESKTOP"},
        {"url": URL, "origin": "https://perf.example", "formFactor": "PHONE"},
        {"origin": "https://perf.example", "formFactor": "PHONE"},
    ],
)
def test_crux_rejects_wrong_or_aggregate_record_key(key):
    payload = crux_payload()
    payload["record"]["key"] = key
    code = "target_mismatch" if "origin" in key else "device_mismatch"
    assert_error(code, lambda: crux(payload))


def test_crux_accepts_desktop_record():
    payload = crux_payload()
    payload["record"]["key"]["formFactor"] = "DESKTOP"
    assert crux(payload, request=request("crux", device="desktop")).device == "desktop"


def test_crux_accepts_same_origin_url_normalization_attested_by_provider():
    payload = crux_payload()
    normalized_url = "https://perf.example/article"
    payload["record"]["key"]["url"] = normalized_url
    payload["urlNormalizationDetails"] = {"originalUrl": URL, "normalizedUrl": normalized_url}
    result = crux(payload)
    assert result.requested_url == URL
    assert result.record_key == normalized_url
    assert result.url_normalization.original_url == URL
    assert result.url_normalization.normalized_url == normalized_url


@pytest.mark.parametrize(
    "returned,attestation",
    [
        ("https://perf.example/article", None),
        (URL, {}),
        (URL, {"originalUrl": "https://perf.example/wrong", "normalizedUrl": URL}),
        (URL, {"originalUrl": URL, "normalizedUrl": "https://perf.example/wrong"}),
        (
            "https://other.example/article",
            {"originalUrl": URL, "normalizedUrl": "https://other.example/article"},
        ),
    ],
)
def test_crux_rejects_absent_forged_or_cross_origin_url_attestation(returned, attestation):
    payload = crux_payload()
    payload["record"]["key"]["url"] = returned
    if attestation is not None:
        payload["urlNormalizationDetails"] = attestation
    assert_error("target_mismatch", lambda: crux(payload))


def test_crux_rejects_explicit_null_attestation():
    payload = crux_payload()
    payload["urlNormalizationDetails"] = None
    assert_error("target_mismatch", lambda: crux(payload))


def test_crux_origin_keeps_original_audited_page():
    payload = crux_payload()
    payload["record"]["key"] = {"origin": "https://PERF.example:443/", "formFactor": "PHONE"}
    result = crux(
        payload, request=request("crux", requested_url="https://perf.example", scope="origin")
    )
    assert result.scope == "origin"
    assert result.requested_url == URL
    assert result.record_key == "https://PERF.example:443/"


@pytest.mark.parametrize(
    "record_key,page_url",
    [
        ("https://perf.example/path", URL),
        ("https://perf.example/?", URL),
        ("https://other.example", URL),
        ("https://perf.example", "https://other.example/page"),
    ],
)
def test_crux_origin_rejects_page_paths_queries_and_other_origins(record_key, page_url):
    payload = crux_payload()
    payload["record"]["key"] = {"origin": record_key, "formFactor": "PHONE"}
    assert_error(
        "target_mismatch",
        lambda: crux(
            payload,
            page_url=page_url,
            request=request("crux", requested_url="https://perf.example", scope="origin"),
        ),
    )


def test_crux_url_scope_cannot_relabel_an_unrelated_audited_page():
    assert_error("target_mismatch", lambda: crux(page_url="https://perf.example/other"))


@pytest.mark.parametrize(
    "parts",
    [
        dict(year=True, month=8, day=1),
        dict(year=2026, month="8", day=1),
        dict(year=2026, month=8, day=1.0),
        dict(year=2026, month=2, day=30),
        dict(year=10**400, month=8, day=1),
    ],
)
def test_crux_rejects_invalid_or_noninteger_dates(parts):
    payload = crux_payload()
    payload["record"]["collectionPeriod"]["firstDate"] = parts
    assert_error("malformed_response", lambda: crux(payload))


@pytest.mark.parametrize(
    "first,last",
    [
        (dict(year=2026, month=8, day=29), dict(year=2026, month=8, day=28)),
        (dict(year=2026, month=8, day=1), dict(year=2026, month=9, day=4)),
    ],
)
def test_crux_rejects_reversed_or_future_period(first, last):
    payload = crux_payload()
    payload["record"]["collectionPeriod"] = {"firstDate": first, "lastDate": last}
    assert_error("malformed_response", lambda: crux(payload))


def test_crux_period_uses_utc_observation_date():
    payload = crux_payload()
    payload["record"]["collectionPeriod"]["lastDate"] = dict(year=2026, month=9, day=3)
    assert_error(
        "malformed_response",
        lambda: crux(payload, observed_at=datetime.fromisoformat("2026-09-03T00:30:00+02:00")),
    )


def test_normalizers_are_detached_immutable_and_do_not_resolve_dns(monkeypatch):
    import socket

    from pydantic import ValidationError

    def no_network(*args, **kwargs):
        raise AssertionError("Normalizer performed DNS")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    lab_payload, field_payload = psi_payload(), crux_payload()
    lab_payload["lighthouseResult"]["runWarnings"] = ["Original warning"]
    lab, field = psi(lab_payload), crux(field_payload)
    lab_payload["lighthouseResult"]["runWarnings"][0] = "Mutated warning"
    lab_payload["lighthouseResult"]["audits"]["largest-contentful-paint"]["numericValue"] = 999
    field_payload["record"]["metrics"]["largest_contentful_paint"]["percentiles"]["p75"] = 999
    assert lab.warnings == ("Original warning",)
    assert lab.metrics[0].value == field.metrics[0].value == 1234.0
    with pytest.raises(ValidationError):
        lab.metrics[0].value = 999
    with pytest.raises(ValidationError):
        field.period.first_date = date(2020, 1, 1)


@pytest.mark.parametrize("provider", ["psi", "crux"])
def test_missing_required_provider_envelope_is_sanitized(provider):
    assert_error("malformed_response", lambda: (psi if provider == "psi" else crux)({}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("lighthouseVersion", "unknown"),
        ("runWarnings", ["x" * 1001]),
        ("environment", {"hostUserAgent": 123}),
        ("environment", {"hostUserAgent": "Chrome/" + "1" * 201 + ".0.0.0"}),
    ],
)
def test_empty_psi_rejects_invalid_retained_metadata_before_no_data(field, value):
    payload = psi_payload()
    result = payload["lighthouseResult"]
    result["audits"] = {}
    result["categories"] = {}
    result[field] = value
    assert_error("malformed_response", lambda: psi(payload))


@pytest.mark.parametrize(
    "last_date",
    ["bad", {"year": 2026, "month": 2, "day": 30}, {"year": 2026, "month": 9, "day": 4}],
)
def test_empty_crux_rejects_invalid_period_before_no_data(last_date):
    payload = crux_payload()
    payload["record"]["metrics"] = {}
    payload["record"]["collectionPeriod"]["lastDate"] = last_date
    assert_error("malformed_response", lambda: crux(payload))


def test_empty_crux_rejects_naive_observation_before_no_data():
    payload = crux_payload()
    payload["record"]["metrics"] = {}
    assert_error("malformed_response", lambda: crux(payload, observed_at=datetime(2026, 9, 3)))


@pytest.mark.parametrize(
    "normalizer,other_request",
    [
        (psi, request("crux")),
        (crux, request()),
        (psi, request().model_copy(update={"scope": "origin"})),
    ],
)
def test_normalizer_rejects_wrong_request_provider_or_scope(normalizer, other_request):
    assert_error("malformed_response", lambda: normalizer(request=other_request))
