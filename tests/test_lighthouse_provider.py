import copy

import pytest
from pydantic import ValidationError

from ai_search_audit import performance_models as models
from ai_search_audit import performance_normalizers as normalizers
from tests.test_performance_normalizers import URL, binding


def fingerprint_data():
    return dict(
        image_id="sha256:" + "a" * 64,
        architecture="arm64",
        node_version="22.19.0",
        lighthouse_version="13.4.1",
        chrome_version="152.0.7977.75",
        puppeteer_version="25.10.0",
        runner_sha256="b" * 64,
        seccomp_sha256="c" * 64,
        dependency_lock_sha256="d" * 64,
        policy_sha256="e" * 64,
    )


def lhr():
    # Native LHR, not a PSI response envelope.
    return {
        "requestedUrl": URL,
        "finalUrl": URL,
        "fetchTime": "2026-09-04T00:00:00Z",
        "lighthouseVersion": "13.4.1",
        "environment": {"hostUserAgent": "HeadlessChrome/152.0.7977.75"},
        "configSettings": {"formFactor": "mobile", "locale": "en"},
        "categories": {"performance": {"score": 0.7}},
        "audits": {"largest-contentful-paint": {"numericValue": 123, "numericUnit": "millisecond"}},
        "runWarnings": ["Synthetic warning"],
        "rawSecret": "must-not-be-retained",
    }


def project(payload=None, **updates):
    function = getattr(normalizers, "normalize_lighthouse", None)
    assert function is not None, "native Lighthouse projection is missing"
    request = models.LighthouseRequest(requested_url=URL, device="mobile", locale="en")
    return function(
        lhr() if payload is None else payload,
        **(
            dict(
                request=request,
                binding=binding(),
                attempt_id="a1",
                evidence_id="e1",
                allowed_final_urls=(),
                runtime=models.LocalRuntimeFingerprint(**fingerprint_data()),
            )
            | updates
        ),
    )


def test_native_lhr_is_local_evidence_with_actual_versions_and_null_metrics():
    measurement = project()
    assert measurement.provider == "lighthouse_local"
    assert measurement.lighthouse_version == "13.4.1"
    assert measurement.chrome_version == "152.0.7977.75"
    assert measurement.configuration.local_runtime.image_id == "sha256:" + "a" * 64
    assert measurement.metrics[0].value == 123
    assert all(metric.value is None for metric in measurement.metrics[1:])
    assert "must-not-be-retained" not in measurement.model_dump_json()
    assert "plaintext WebSocket" in " ".join(measurement.warnings)


@pytest.mark.parametrize("product", ["HeadlessChrome", "Chrome"])
def test_reduced_host_ua_retains_verified_full_local_browser_version(product):
    payload = lhr()
    payload["environment"]["hostUserAgent"] = (
        f"Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) {product}/152.0.0.0 Safari/537.36"
    )
    original = copy.deepcopy(payload)
    measurement = project(payload)
    assert measurement.chrome_version == "152.0.7977.75"
    assert measurement.configuration.local_runtime.chrome_version == "152.0.7977.75"
    assert payload == original


@pytest.mark.parametrize(
    "version",
    [
        "153.0.0.0",
        "151.0.7977.75",
        "152.0.7977.74",
        "152.0.1.0",
        "152.1.0.0",
        "152.0.0",
        "garbage",
        "152.0.0.0 Chrome/152.0.7977.75",
    ],
)
def test_local_ua_reduction_does_not_allow_other_version_mismatches(version):
    payload = lhr()
    payload["environment"]["hostUserAgent"] = f"HeadlessChrome/{version}"
    with pytest.raises(normalizers.NormalizationError, match="runtime_error"):
        project(payload)


@pytest.mark.parametrize(
    "user_agent",
    [
        "HeadlessChrome/152.0.0.0 Chrome/garbage",
        "Chrome/garbage HeadlessChrome/152.0.0.0",
        "Chrome/152.0.7977.75 HeadlessChrome/",
        "HeadlessChrome/ Chrome/152.0.7977.75",
    ],
)
def test_local_ua_rejects_malformed_second_chrome_token_without_mutation(user_agent):
    payload = lhr()
    payload["environment"]["hostUserAgent"] = user_agent
    original = copy.deepcopy(payload)
    with pytest.raises(normalizers.NormalizationError, match="runtime_error"):
        project(payload)
    assert payload == original


def test_psi_still_retains_reduced_provider_ua_version_without_local_override():
    from tests.test_performance_normalizers import psi, psi_payload

    payload = psi_payload()
    payload["lighthouseResult"]["environment"] = {"hostUserAgent": "Chrome/152.0.0.0"}
    measurement = psi(payload)
    assert measurement.chrome_version == "152.0.0.0"
    assert measurement.configuration.local_runtime is None


@pytest.mark.parametrize(
    "mutation,code",
    [
        ({"requestedUrl": "https://foreign.example/"}, "target_mismatch"),
        ({"finalUrl": "https://foreign.example/"}, "target_mismatch"),
        ({"configSettings": {"formFactor": "desktop"}}, "device_mismatch"),
        ({"runtimeError": {"code": "ERRORED_DOCUMENT_REQUEST"}}, "runtime_error"),
        ({"lighthouseVersion": "13.0.1"}, "runtime_error"),
        ({"environment": {}}, "runtime_error"),
        ({"environment": {"hostUserAgent": "Chrome/150.0.0.0"}}, "runtime_error"),
        ({"audits": {}, "categories": {}}, "no_data"),
        ({"lighthouseResult": lhr(), "requestedUrl": None}, "target_mismatch"),
    ],
)
def test_native_lhr_rejects_inconsistent_evidence(mutation, code):
    with pytest.raises(normalizers.NormalizationError, match=code):
        project(lhr() | copy.deepcopy(mutation))


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_id", "mutable:latest"),
        ("runner_sha256", "x"),
        ("architecture", "emulated"),
        ("node_version", "unknown"),
        ("credentials", "not-allowed"),
    ],
)
def test_fingerprint_is_strict_and_frozen(field, value):
    cls = getattr(models, "LocalRuntimeFingerprint", None)
    assert cls is not None, "local runtime fingerprint is missing"
    with pytest.raises(ValidationError):
        cls(**(fingerprint_data() | {field: value}))
    with pytest.raises(ValidationError):
        cls(**fingerprint_data()).node_version = "changed"


def test_tampered_runtime_instance_revalidated():
    cls = getattr(models, "LocalRuntimeFingerprint", None)
    assert cls is not None
    runtime = cls(**fingerprint_data()).model_copy(update={"image_id": "bad"})
    with pytest.raises(normalizers.NormalizationError, match="malformed_response"):
        project(runtime=runtime)


@pytest.mark.parametrize("updates", [{"image_id": b"sha256:" + b"a" * 64}, {"unknown": "data"}])
def test_fingerprint_python_mode_tampering_is_rejected(updates):
    runtime = models.LocalRuntimeFingerprint(**fingerprint_data()).model_copy(update=updates)
    with pytest.raises(normalizers.NormalizationError, match="malformed_response"):
        project(runtime=runtime)


@pytest.mark.parametrize(
    "mutation",
    [
        {"provider": "pagespeed_insights"},
        {"chrome_version": "150.0.0.0"},
        {"lighthouse_version": "13.0.1"},
    ],
)
def test_measurement_rejects_provenance_mismatch(mutation):
    data = project().model_dump() | mutation
    with pytest.raises(ValidationError):
        models.LabMeasurement.model_validate(data)


def test_local_measurement_requires_fingerprint():
    data = project().model_dump()
    data["configuration"]["local_runtime"] = None
    with pytest.raises(ValidationError):
        models.LabMeasurement.model_validate(data)


def provider_module():
    import importlib.util

    assert importlib.util.find_spec("ai_search_audit.lighthouse_provider"), "provider is missing"
    from ai_search_audit import lighthouse_provider

    return lighthouse_provider


@pytest.mark.parametrize(
    "failure,state",
    [
        ("runtime_missing", "unavailable"),
        ("runtime_unverified", "unavailable"),
        ("sandbox_unverified", "unavailable"),
        ("network_unverified", "unavailable"),
        ("timeout", "failed"),
        ("cleanup_failed", "failed"),
    ],
)
def test_local_provider_failure_is_an_attempt_not_site_evidence(failure, state):
    from ai_search_audit.lighthouse_runtime import RuntimeResult
    from tests.test_performance_providers import source

    module = provider_module()

    class FakeRuntime:
        def run(self, request):
            return RuntimeResult(failure=failure)

    collection = module.LighthouseProvider(runtime=FakeRuntime()).collect(
        source(), url=URL, device="mobile"
    )
    assert len(collection.attempts) == 1
    assert collection.attempts[0].state.value == state.upper()
    assert collection.attempts[0].provider == "lighthouse_local"
    assert collection.attempts[0].reason == failure
    assert collection.attempts[0].http_status is None
    assert collection.lab is None


def test_local_provider_projects_native_result_and_validates_source_before_launch():
    import json

    from ai_search_audit.lighthouse_runtime import RuntimeResult
    from tests.test_performance_providers import source

    module = provider_module()
    calls = []

    class FakeRuntime:
        def run(self, request):
            calls.append(request)
            return RuntimeResult(
                body=json.dumps(lhr()).encode(),
                fingerprint=models.LocalRuntimeFingerprint(**fingerprint_data()),
            )

    provider = module.LighthouseProvider(runtime=FakeRuntime())
    collection = provider.collect(source(), url=URL, device="mobile")
    assert collection.lab.provider == "lighthouse_local"
    assert collection.attempts[0].state.value == "PARTIAL"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="invalid performance configuration"):
        provider.collect(source(), url="https://other.example/", device="mobile")
    assert len(calls) == 1
