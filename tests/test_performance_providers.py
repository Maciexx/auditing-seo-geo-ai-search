import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from ai_search_audit.diagnostic_models import DiagnosticSource, FrozenPrompt
from ai_search_audit.models import DataState
from ai_search_audit.performance_http import PerformanceHTTPClient
from ai_search_audit.performance_providers import PerformanceCollection, PerformanceProviders
from tests.test_performance_normalizers import URL, binding, crux_payload, psi_payload

KEY = SecretStr("synthetic-test-credential")


def source(**updates):
    return DiagnosticSource.model_validate(
        dict(binding=binding(), prompts=(), page_urls=(URL,), canonical_domains=("perf.example",))
        | updates
    )


def full_psi():
    payload = psi_payload()
    for name in (
        "cumulative-layout-shift",
        "first-contentful-paint",
        "total-blocking-time",
        "speed-index",
    ):
        payload["lighthouseResult"]["audits"][name] = {
            "numericValue": 0,
            "numericUnit": "unitless" if name == "cumulative-layout-shift" else "millisecond",
        }
    return payload


class Harness:
    def __init__(self, responses, addresses=None):
        self.responses = iter(responses)
        self.requests = []
        self.dns = []
        self.sleeps = []
        self.times = []
        self.addresses = iter(addresses) if addresses is not None else None
        self.providers = PerformanceProviders(
            http=PerformanceHTTPClient(
                transport=httpx.MockTransport(self.send), resolver=self.resolve
            ),
            now=self.now,
            sleep=self.sleeps.append,
        )

    def resolve(self, host):
        self.dns.append(host)
        return next(self.addresses) if self.addresses else ["93.184.216.34"]

    def now(self):
        stamp = datetime(2026, 9, 3, tzinfo=UTC) + timedelta(seconds=len(self.times))
        self.times.append(stamp)
        return stamp

    def send(self, request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        status, body = response
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, stream=httpx.ByteStream(body))

    def call(self, provider="pagespeed", **kwargs):
        return getattr(self.providers, provider)(
            kwargs.pop("source", source()),
            **(dict(url=URL, device="mobile", api_key=KEY) | kwargs),
        )


@pytest.mark.parametrize("provider,payload", [("pagespeed", full_psi), ("crux", crux_payload)])
def test_complete_measurement_has_pipeline_identity_and_real_request(provider, payload):
    harness = Harness([(200, payload())])
    result = harness.call(provider)
    assert isinstance(result, PerformanceCollection)
    measurement = result.lab if provider == "pagespeed" else result.field
    assert measurement is not None
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.state is DataState.AVAILABLE
    assert attempt.reason is None
    assert attempt.http_status == 200
    assert attempt.started_at == harness.times[0]
    assert attempt.ended_at == harness.times[1]
    assert attempt.requested_url == measurement.requested_url == URL
    assert attempt.provider == measurement.provider
    assert attempt.attempt_id == measurement.attempt_id
    assert UUID(attempt.attempt_id) != UUID(measurement.evidence_id)
    assert measurement.binding == source().binding
    assert measurement.binding is not source().binding
    assert harness.dns == ["perf.example"]
    assert harness.sleeps == []
    request = harness.requests[0]
    assert request.headers["X-Goog-Api-Key"] == KEY.get_secret_value()
    if provider == "pagespeed":
        assert request.method == "GET"
        assert request.url.params["url"] == URL
        assert request.url.params["locale"] == "en"
        assert request.url.params["strategy"] == "mobile"
        assert measurement.performance_score == 75
        assert len(measurement.metrics) == 5
    else:
        assert request.method == "POST"
        assert json.loads(request.content)["url"] == URL
        assert measurement.scope == "url"
        assert measurement.observed_at == attempt.ended_at
    with pytest.raises(ValidationError):
        measurement.binding.project_id = "changed"
    with pytest.raises(ValidationError):
        result.attempts = ()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_retries_once_and_retains_failed_attempt(status):
    harness = Harness([(status, {}), (200, full_psi())])
    result = harness.call()
    assert [a.http_status for a in result.attempts] == [status, 200]
    assert result.attempts[0].state is (
        DataState.UNAVAILABLE if status == 429 else DataState.FAILED
    )
    assert result.attempts[0].reason == f"http_{status}"
    assert result.attempts[1].state is DataState.AVAILABLE
    assert result.lab.attempt_id == result.attempts[1].attempt_id
    assert len({a.attempt_id for a in result.attempts}) == 2
    assert harness.sleeps == [1.0]
    assert len(harness.dns) == 2
    assert [(a.started_at, a.ended_at) for a in result.attempts] == [
        (harness.times[0], harness.times[1]),
        (harness.times[2], harness.times[3]),
    ]


@pytest.mark.parametrize(
    "error,reason",
    [
        (httpx.ReadTimeout("private error"), "timeout"),
        (httpx.ConnectError("private error"), "transport_error"),
    ],
)
def test_transport_failure_twice_stops_with_complete_history(error, reason):
    harness = Harness([error, error])
    result = harness.call()
    assert len(result.attempts) == len(harness.requests) == 2
    assert all(a.state is DataState.FAILED and a.reason == reason for a in result.attempts)
    assert result.lab is result.field is None
    assert harness.sleeps == [1.0]
    assert "private error" not in result.model_dump_json()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retry_disabled_never_sends_again(status):
    harness = Harness([(status, {})])
    result = harness.call(retry_transient=False)
    assert len(result.attempts) == len(harness.requests) == 1
    assert harness.sleeps == []


@pytest.mark.parametrize(
    "status,state",
    [
        (401, DataState.UNAVAILABLE),
        (403, DataState.UNAVAILABLE),
        (400, DataState.FAILED),
        (404, DataState.FAILED),
        (501, DataState.FAILED),
    ],
)
def test_nontransient_status_never_retries(status, state):
    harness = Harness([(status, {"error": KEY.get_secret_value()})])
    result = harness.call()
    assert len(result.attempts) == 1
    assert result.attempts[0].state is state
    assert result.attempts[0].reason == f"http_{status}"
    assert harness.sleeps == []
    assert KEY.get_secret_value() not in result.model_dump_json()


@pytest.mark.parametrize(
    "key,reason",
    [
        (None, "missing_key"),
        (SecretStr(""), "missing_key"),
        (SecretStr("contains space"), "invalid_key"),
    ],
)
def test_missing_or_invalid_key_short_circuit_has_aware_attempt_without_io(key, reason):
    harness = Harness([])
    result = harness.call(api_key=key)
    assert harness.requests == harness.dns == harness.sleeps == []
    assert len(result.attempts) == 1
    assert result.attempts[0].state is DataState.UNAVAILABLE
    assert result.attempts[0].reason == reason
    assert result.attempts[0].http_status is None
    assert result.attempts[0].ended_at > result.attempts[0].started_at


def test_retry_rechecks_dns_and_blocks_rebinding():
    harness = Harness([(429, {})], addresses=[["93.184.216.34"], ["127.0.0.1"]])
    result = harness.call()
    assert len(harness.requests) == 1
    assert len(harness.dns) == 2
    assert result.attempts[-1].state is DataState.UNAVAILABLE
    assert result.attempts[-1].reason == "unsafe_target"
    assert result.attempts[-1].http_status is None


@pytest.mark.parametrize("body", [b"{", b"[]", b'{"a":1,"a":2}'])
def test_malformed_json_is_failed_without_retry(body):
    harness = Harness([(200, body)])
    result = harness.call()
    assert result.attempts[0].state is DataState.FAILED
    assert result.attempts[0].reason == "malformed_response"
    assert harness.sleeps == []


@pytest.mark.parametrize(
    "change,state,reason",
    [
        ({"requestedUrl": "https://other.example/"}, DataState.UNKNOWN, "target_mismatch"),
        ({"finalUrl": "https://perf.example/unapproved"}, DataState.UNKNOWN, "target_mismatch"),
        ({"configSettings": {"formFactor": "desktop"}}, DataState.UNKNOWN, "device_mismatch"),
        (
            {"runtimeError": {"code": "CRASH", "message": "private"}},
            DataState.FAILED,
            "runtime_error",
        ),
        ({"categories": {"performance": {"score": True}}}, DataState.FAILED, "invalid_metrics"),
        ({"categories": {}, "audits": {}}, DataState.UNAVAILABLE, "no_data"),
    ],
)
def test_normalization_errors_map_to_fixed_outcomes(change, state, reason):
    payload = full_psi()
    payload["lighthouseResult"].update(change)
    harness = Harness([(200, payload)])
    result = harness.call()
    assert result.attempts[0].state is state
    assert result.attempts[0].reason == reason
    assert result.lab is None
    assert harness.sleeps == []


@pytest.mark.parametrize("provider,payload", [("pagespeed", psi_payload), ("crux", crux_payload)])
def test_missing_metrics_are_partial_but_zero_is_present(provider, payload):
    body = payload()
    if provider == "crux":
        body["record"]["metrics"].pop("interaction_to_next_paint")
    harness = Harness([(200, body)])
    result = harness.call(provider)
    assert result.attempts[0].state is DataState.PARTIAL
    assert result.attempts[0].reason == "missing_metrics"
    assert len(result.attempts) == 1


def test_zero_score_and_metrics_are_available():
    body = full_psi()
    body["lighthouseResult"]["categories"]["performance"]["score"] = 0
    for metric in body["lighthouseResult"]["audits"].values():
        metric["numericValue"] = 0
    result = Harness([(200, body)]).call()
    assert result.attempts[0].state is DataState.AVAILABLE
    assert result.lab.performance_score == 0


def test_redirect_is_failed_without_retry():
    harness = Harness([(302, {})])
    result = harness.call()
    assert result.attempts[0].state is DataState.FAILED
    assert result.attempts[0].reason == "redirect"
    assert harness.sleeps == []


def origin_payload():
    payload = crux_payload()
    payload["record"]["key"] = {"origin": "https://perf.example", "formFactor": "PHONE"}
    return payload


@pytest.mark.parametrize(
    "first",
    [
        (404, {}),
        (
            200,
            {
                "record": {
                    "key": {"url": URL, "formFactor": "PHONE"},
                    "collectionPeriod": crux_payload()["record"]["collectionPeriod"],
                    "metrics": {},
                }
            },
        ),
    ],
)
def test_crux_url_absence_falls_back_to_origin_with_original_page_binding(first):
    harness = Harness([first, (200, origin_payload())])
    result = harness.call("crux")
    assert len(result.attempts) == 2
    assert result.attempts[0].reason in {"no_record", "no_data"}
    assert result.attempts[0].state is DataState.UNAVAILABLE
    assert result.attempts[1].requested_url == "https://perf.example"
    assert result.attempts[1].state is DataState.AVAILABLE
    assert result.field.requested_url == URL
    assert result.field.record_key == "https://perf.example"
    assert result.field.scope == "origin"
    assert result.field.attempt_id == result.attempts[1].attempt_id
    assert result.lab is None
    assert harness.sleeps == []
    assert [json.loads(req.content) for req in harness.requests] == [
        {
            "url": URL,
            "formFactor": "PHONE",
            "metrics": [
                "largest_contentful_paint",
                "interaction_to_next_paint",
                "cumulative_layout_shift",
            ],
        },
        {
            "origin": "https://perf.example",
            "formFactor": "PHONE",
            "metrics": [
                "largest_contentful_paint",
                "interaction_to_next_paint",
                "cumulative_layout_shift",
            ],
        },
    ]


def test_crux_both_scopes_absent_retains_two_attempts_and_no_measurement():
    harness = Harness([(404, {}), (404, {})])
    result = harness.call("crux")
    assert len(result.attempts) == 2
    assert all(
        a.reason == "no_record" and a.state is DataState.UNAVAILABLE for a in result.attempts
    )
    assert result.field is result.lab is None


def test_crux_maximum_four_attempts_including_each_scope_retry():
    harness = Harness([(429, {}), (404, {}), (429, {}), (200, origin_payload())])
    result = harness.call("crux")
    assert [a.http_status for a in result.attempts] == [429, 404, 429, 200]
    assert [a.requested_url for a in result.attempts] == [
        URL,
        URL,
        "https://perf.example",
        "https://perf.example",
    ]
    assert harness.sleeps == [1.0, 1.0]
    assert len(harness.requests) == len(harness.dns) == 4


@pytest.mark.parametrize(
    "responses,reason",
    [
        ([(403, {})], "http_403"),
        ([(429, {}), (429, {})], "http_429"),
        ([httpx.ReadTimeout("x"), httpx.ReadTimeout("x")], "timeout"),
        ([(200, b"{")], "malformed_response"),
        ([(200, origin_payload())], "target_mismatch"),
    ],
)
def test_crux_never_falls_back_for_failures_other_than_absence(responses, reason):
    harness = Harness(responses)
    result = harness.call("crux")
    assert result.attempts[-1].reason == reason
    assert all("url" in json.loads(req.content) for req in harness.requests)


def test_crux_no_retry_flag_still_allows_absence_fallback():
    result = Harness([(404, {}), (200, origin_payload())]).call("crux", retry_transient=False)
    assert len(result.attempts) == 2


def test_origin_derivation_preserves_scheme_and_nondefault_port():
    url = "http://perf.example:8080/page?query=1"
    body = origin_payload()
    body["record"]["key"]["origin"] = "http://perf.example:8080"
    harness = Harness([(404, {}), (200, body)])
    result = harness.call("crux", url=url, source=source(page_urls=(url,)))
    assert result.field.requested_url == url
    assert result.attempts[-1].requested_url == "http://perf.example:8080"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://perf.example/not-in-source"},
        {"url": "http://127.0.0.1/"},
        {"source": source(canonical_domains=("other.example",))},
        {"source": source().model_copy(update={"page_urls": (True,)})},
        {
            "source": source().model_copy(
                update={"binding": binding().model_copy(update={"report_locale": True})}
            )
        },
        {"source": source().model_copy(update={"prompts": (True,)})},
        {"source": source(canonical_domains=("https://perf.example/",))},
        {"device": "tablet"},
        {"retry_transient": 1},
        {"retry_transient": "false"},
    ],
)
def test_invalid_configuration_is_fixed_error_before_credentials_or_io(kwargs):
    class ForbiddenCredential:
        def get_secret_value(self):
            pytest.fail("credentials accessed before configuration validation")

    harness = Harness([])
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        harness.call(api_key=ForbiddenCredential(), **kwargs)
    assert harness.requests == harness.dns == harness.sleeps == harness.times == []


@pytest.mark.parametrize(
    "host,domain,accepted",
    [
        ("straße.example", "xn--strae-oqa.example", True),
        ("straße.example", "strasse.example", False),
        ("PERF.example.", "perf.example", True),
    ],
)
def test_canonical_membership_uses_modern_hostname_semantics(host, domain, accepted):
    url = f"https://{host}/"
    body = full_psi()
    body["lighthouseResult"].update(requestedUrl=url, finalUrl=url)
    harness = Harness([(200, body)])
    candidate = source(
        binding=binding().model_copy(update={"domain": domain}),
        page_urls=(url,),
        canonical_domains=(domain,),
    )
    if accepted:
        result = harness.call(source=candidate, url=url)
        assert result.lab.requested_url == url
    else:
        with pytest.raises(ValueError, match="^invalid performance configuration$"):
            harness.call(source=candidate, url=url)
        assert harness.requests == harness.dns == []


def test_source_page_membership_accepts_equivalent_root_url():
    url = "https://PERF.example:443"
    body = full_psi()
    body["lighthouseResult"].update(requestedUrl=url, finalUrl=url)
    result = Harness([(200, body)]).call(
        url=url, source=source(page_urls=("https://perf.example/",))
    )
    assert result.lab.requested_url == url


def test_only_explicit_source_pages_allow_psi_final_url():
    body = full_psi()
    body["lighthouseResult"]["finalUrl"] = "https://perf.example/explicit"
    result = Harness([(200, body)]).call(
        source=source(page_urls=(URL, "https://perf.example/explicit"))
    )
    assert result.lab.final_url == "https://perf.example/explicit"


@pytest.mark.parametrize("key", [KEY, SecretStr('quote"and\\slash')])
def test_decoded_warning_secret_is_failed_and_discarded(key):
    body = full_psi()
    body["lighthouseResult"]["runWarnings"] = [f"prefix {key.get_secret_value()} suffix"]
    # Escape every character: raw-body substring checking cannot see this echo.
    raw = (
        json.dumps(body).replace(
            key.get_secret_value(),
            "".join(f"\\u{ord(char):04x}" for char in key.get_secret_value()),
        )
        if key is KEY
        else json.dumps(body)
    )
    harness = Harness([(200, raw.encode())])
    result = harness.call(api_key=key)
    assert result.attempts[0].reason == "sensitive_response"
    assert result.attempts[0].state is DataState.FAILED
    assert result.lab is None
    assert key.get_secret_value() not in repr(result)
    assert key.get_secret_value() not in result.model_dump_json()
    assert harness.sleeps == []


def test_raw_body_secret_is_discarded():
    body = full_psi()
    body["lighthouseResult"]["runWarnings"] = [KEY.get_secret_value()]
    result = Harness([(200, body)]).call()
    assert result.attempts[0].reason == "sensitive_response"
    assert result.lab is None


@pytest.mark.parametrize("where", ["url", "binding"])
def test_secret_in_canonical_inputs_is_rejected_without_emitting_unsafe_attempt(where):
    value = KEY.get_secret_value()
    url = f"https://perf.example/{value}" if where == "url" else URL
    binding_data = (
        binding().model_copy(update={"audit_id": value}) if where == "binding" else binding()
    )
    candidate = source(binding=binding_data, page_urls=(url,))
    harness = Harness([])
    with pytest.raises(ValueError, match="^invalid performance configuration$") as caught:
        harness.call(source=candidate, url=url)
    assert value not in str(caught.value)
    assert harness.requests == harness.dns == harness.times == []


def test_invalid_source_does_not_emit_serialization_warnings(recwarn):
    candidate = source().model_copy(update={"page_urls": (True,)})
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        Harness([]).call(source=candidate)
    assert not recwarn.list


def collection_payload(provider="pagespeed"):
    payload = full_psi() if provider == "pagespeed" else crux_payload()
    return Harness([(200, payload)]).call(provider).model_dump(mode="json")


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "too_many",
        "duplicate",
        "no_measurement",
        "orphan_measurement",
        "provider_mismatch",
        "device_mismatch",
        "target_mismatch",
        "failed_matching_attempt",
        "unmatched_success",
        "success_then_failure",
        "both_measurements",
        "available_missing_metric",
        "available_missing_score",
        "partial_complete",
        "partial_wrong_reason",
        "tampered_bool",
    ],
)
def test_collection_rejects_inconsistent_or_tampered_histories(mutation):
    data = collection_payload()
    attempt = data["attempts"][0]
    if mutation == "empty":
        data["attempts"] = []
    elif mutation == "too_many":
        data["attempts"] *= 5
    elif mutation == "duplicate":
        data["attempts"] *= 2
    elif mutation == "no_measurement":
        data["lab"] = None
    elif mutation == "orphan_measurement":
        data["lab"]["attempt_id"] = "orphan"
    elif mutation == "provider_mismatch":
        attempt["provider"] = "crux"
    elif mutation == "device_mismatch":
        attempt["device"] = "desktop"
    elif mutation == "target_mismatch":
        attempt["requested_url"] = "https://perf.example/wrong"
    elif mutation == "failed_matching_attempt":
        attempt.update(state="FAILED", reason="malformed_response")
    elif mutation == "unmatched_success":
        data["attempts"].insert(0, attempt | {"attempt_id": "unmatched"})
    elif mutation == "success_then_failure":
        data["attempts"].append(
            attempt | {"attempt_id": "later", "state": "FAILED", "reason": "timeout"}
        )
    elif mutation == "both_measurements":
        data["field"] = collection_payload("crux")["field"]
    elif mutation == "available_missing_metric":
        data["lab"]["metrics"] = data["lab"]["metrics"][:-1]
    elif mutation == "available_missing_score":
        data["lab"]["performance_score"] = None
    elif mutation == "partial_complete":
        attempt.update(state="PARTIAL", reason="missing_metrics")
    elif mutation == "partial_wrong_reason":
        attempt.update(state="PARTIAL", reason="some_reason")
        data["lab"]["metrics"][0]["value"] = None
    else:
        data["lab"]["metrics"][0]["value"] = True
    with pytest.raises(ValueError):
        PerformanceCollection.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "mutation",
    [
        "origin_attempt_page",
        "origin_wrong_host",
        "url_attempt_other_page",
        "available_missing_metric",
    ],
)
def test_collection_rejects_field_scope_or_coverage_mismatch(mutation):
    result = Harness([(404, {}), (200, origin_payload())]).call("crux")
    data = (
        result.model_dump(mode="json")
        if mutation.startswith("origin")
        else collection_payload("crux")
    )
    if mutation == "origin_attempt_page":
        data["attempts"][-1]["requested_url"] = URL
    elif mutation == "origin_wrong_host":
        data["attempts"][-1]["requested_url"] = "https://other.example"
    elif mutation == "url_attempt_other_page":
        data["attempts"][-1]["requested_url"] = "https://perf.example/other"
    else:
        data["field"]["metrics"] = data["field"]["metrics"][:-1]
    with pytest.raises(ValueError):
        PerformanceCollection.model_validate_json(json.dumps(data))


def test_failed_attempt_history_cannot_mix_providers():
    data = Harness([(429, {}), (429, {})]).call().model_dump(mode="json")
    data["attempts"][1]["provider"] = "crux"
    with pytest.raises(ValueError):
        PerformanceCollection.model_validate(data)


def test_constructor_revalidates_unchecked_nested_model_copy_without_warnings(recwarn):
    result = Harness([(200, full_psi())]).call()
    metric = result.lab.metrics[0].model_copy(update={"value": True})
    tampered = result.lab.model_copy(update={"metrics": (metric, *result.lab.metrics[1:])})
    with pytest.raises(ValueError):
        PerformanceCollection(attempts=result.attempts, lab=tampered)
    assert not recwarn.list


def test_collection_roundtrip_rejects_unchecked_model_copy():
    result = Harness([(200, full_psi())]).call()
    tampered = result.model_copy(update={"lab": None})
    with pytest.raises(ValueError):
        PerformanceCollection.model_validate_json(tampered.model_dump_json(serialize_as_any=True))


def test_field_attested_normalized_record_keeps_request_reference():
    payload = crux_payload()
    payload["record"]["key"]["url"] = "https://perf.example/normalized"
    payload["urlNormalizationDetails"] = {
        "originalUrl": URL,
        "normalizedUrl": "https://perf.example/normalized",
    }
    result = Harness([(200, payload)]).call("crux")
    assert result.field.requested_url == result.attempts[0].requested_url == URL
    assert result.field.record_key != URL


def test_source_binding_domain_must_be_one_of_the_canonical_domains():
    harness = Harness([])
    candidate = source(binding=binding().model_copy(update={"domain": "wrong.example"}))
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        harness.call(source=candidate)
    assert harness.requests == harness.dns == harness.times == []


def test_source_binding_primary_domain_and_explicit_alias_are_accepted():
    candidate = source(
        binding=binding().model_copy(update={"domain": "primary.example"}),
        canonical_domains=("primary.example", "perf.example"),
    )
    result = Harness([(200, full_psi())]).call(source=candidate)
    assert result.lab.binding.domain == "primary.example"


def test_current_key_cannot_escape_through_fixed_attempt_metadata():
    harness = Harness([httpx.ReadTimeout("irrelevant")])
    fabricated_credential = "timeout"
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        harness.call(api_key=SecretStr(fabricated_credential), retry_transient=False)


def test_current_numeric_key_cannot_escape_through_http_status_metadata():
    harness = Harness([(299, full_psi())])
    fabricated_credential = "299"
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        harness.call(api_key=SecretStr(fabricated_credential))
    assert len(harness.requests) == 1
    assert harness.sleeps == []


@pytest.mark.parametrize("key", ['":"', "["])
def test_current_key_cannot_escape_through_serialized_json_punctuation(key):
    harness = Harness([(200, full_psi())])
    with pytest.raises(ValueError, match="^invalid performance configuration$"):
        harness.call(api_key=SecretStr(key))
    assert len(harness.requests) == 1
    assert harness.sleeps == []


def test_measurement_json_punctuation_collision_discards_only_measurement():
    key = '"screen_width":null'
    harness = Harness([(200, full_psi())])
    result = harness.call(api_key=SecretStr(key))
    assert result.attempts[0].state is DataState.FAILED
    assert result.attempts[0].reason == "sensitive_response"
    assert result.lab is None
    assert key not in result.model_dump_json()


@pytest.mark.parametrize("key", [None, "", " ", "\t", "\n", " \t\n"])
def test_whitespace_only_key_is_missing_with_real_source_prompts(key):
    candidate = source(
        prompts=(
            FrozenPrompt(
                prompt_id="p1",
                pack_version="2.0.0",
                locale="en",
                intent="brand",
                text="What does this company offer?\t\nMore context. \t\n",
            ),
        )
    )
    harness = Harness([])
    result = harness.call(source=candidate, api_key=None if key is None else SecretStr(key))
    assert len(result.attempts) == 1
    assert result.attempts[0].state is DataState.UNAVAILABLE
    assert result.attempts[0].reason == "missing_key"
    assert result.attempts[0].http_status is None
    assert harness.requests == harness.dns == harness.sleeps == []
    assert len(harness.times) == 2


def test_final_collection_is_revalidated_after_normalizer_tampering(monkeypatch):
    from ai_search_audit import performance_providers as providers

    original = providers.normalize_psi

    def tampered(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.model_copy(update={"attempt_id": "forged"})

    monkeypatch.setattr(providers, "normalize_psi", tampered)
    with pytest.raises(ValueError):
        Harness([(200, full_psi())]).call()
