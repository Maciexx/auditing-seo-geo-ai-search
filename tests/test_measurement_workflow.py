"""Real CLI, normalization and persistence; only network/process boundaries replaced."""

import json
from importlib import import_module, util

import httpx
import pytest

from ai_search_audit.cli import main
from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.models import DataState
from tests.test_diagnostic_workflow import _hashes, no_network, project
from tests.test_performance_normalizers import crux_payload, psi_payload

__all__ = ["no_network", "project"]

FABRICATED_CREDENTIAL = "MUST-NOT-PRINT"
FABRICATED_UNCHECKED_CREDENTIAL = "secret"


def command(project, profile):
    path = project.parent.parent / "measurement-profile.json"
    path.write_text(json.dumps(profile))
    return [
        "project",
        "measure",
        "project:example",
        "--source-version",
        "public-v1",
        "--clients-root",
        str(project.parent),
        "--profile",
        str(path),
    ]


def harness(
    monkeypatch, capsys, *, psi_status=200, crux_origin=False, psi_warning=None, psi_mutation=None
):
    from ai_search_audit import performance_http

    calls = []
    preflights = []
    limits = []
    original = performance_http.PerformanceHTTPClient.__init__

    def send(request):
        if not calls:
            lines = capsys.readouterr().out.splitlines()
            assert lines[0].startswith("MEASUREMENT_PREFLIGHT=")
            preflights.append(json.loads(lines[0].split("=", 1)[1]))
        calls.append(request)
        if request.method == "GET":
            status = psi_status
            if isinstance(status, tuple):
                index = sum(c.method == "GET" for c in calls) - 1
                status = status[index % len(status)]
            if isinstance(status, Exception):
                raise status
            payload = psi_payload()
            url = request.url.params["url"]
            payload["lighthouseResult"].update(requestedUrl=url, finalUrl=url)
            payload["lighthouseResult"]["configSettings"].update(
                formFactor=request.url.params["strategy"]
            )
            if psi_warning is not None:
                payload["lighthouseResult"]["runWarnings"] = [psi_warning]
            if psi_mutation is not None:
                psi_mutation(payload)
            return httpx.Response(status, stream=httpx.ByteStream(json.dumps(payload).encode()))
        data = json.loads(request.content)
        if crux_origin and "url" in data:
            return httpx.Response(404, stream=httpx.ByteStream(b"{}"))
        payload = crux_payload()
        payload["record"]["key"] = data
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps(payload).encode()))

    def init(self, **kwargs):
        limits.append(kwargs["limits"])
        original(
            self,
            **kwargs,
            transport=httpx.MockTransport(send),
            resolver=lambda host: ["93.184.216.34"],
        )

    monkeypatch.setattr(performance_http.PerformanceHTTPClient, "__init__", init)
    monkeypatch.setenv("AUDIT_PAGESPEED_API_KEY", "synthetic-psi-key")
    monkeypatch.setenv("AUDIT_CRUX_API_KEY", "synthetic-crux-key")
    monkeypatch.delenv("AUDIT_LIGHTHOUSE_IMAGE_ID", raising=False)
    return calls, preflights, limits


def test_measure_cli_freezes_before_calls_and_keeps_independent_observations(
    project, monkeypatch, capsys
):
    calls, plans, limits = harness(monkeypatch, capsys, crux_origin=True)
    before = _hashes(project)
    assert (
        main(
            command(
                project,
                {
                    "max_pages": 1,
                    "http_limits": {"timeout_seconds": 7.0, "max_response_bytes": 4096},
                },
            )
        )
        == 0
    )
    assert len(calls) == 6
    assert plans[0]["maximum_attempts"] == {
        "pagespeed_insights": 4,
        "crux": 8,
        "lighthouse_local": 0,
        "gemini": 0,
    }
    assert limits[0].timeout_seconds == 7.0
    loaded = DiagnosticStore(project).load("public-v1", "run-1")
    assert loaded.run.schema_version == "2.0.0"
    assert len(loaded.run.collections) == 4
    assert loaded.run.collections[0].lab.performance_score == 75
    assert loaded.run.collections[1].field.scope == "origin"
    assert all(c.lab is None or c.lab.metrics[1].value is None for c in loaded.run.collections)
    assert all(
        "synthetic-" not in p.read_text()
        for p in (project / "diagnostics/public-v1/run-1").iterdir()
    )
    assert before == {k: v for k, v in _hashes(project).items() if not k.startswith("diagnostics/")}


@pytest.mark.parametrize(
    "status,tries,fallback",
    [
        (403, 1, True),
        (429, 2, True),
        (500, 2, True),
        (502, 2, True),
        (503, 2, True),
        (504, 2, True),
        (200, 1, False),
        (httpx.ReadTimeout("synthetic timeout"), 2, True),
        (httpx.ConnectError("synthetic transport failure"), 2, True),
    ],
)
@pytest.mark.parametrize("retry", [False, True])
def test_attempt_budget_and_eligible_final_psi_falls_back(
    project, monkeypatch, capsys, status, tries, fallback, retry
):
    calls, _, _ = harness(monkeypatch, capsys, psi_status=status)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)
    assert (
        main(
            command(
                project,
                {
                    "max_pages": 1,
                    "lighthouse_local": True,
                    "retry_transient": retry,
                },
            )
        )
        == 0
    )
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    psi = [c for c in run.collections if c.attempts[0].provider == "pagespeed_insights"]
    local = [c for c in run.collections if c.attempts[0].provider == "lighthouse_local"]
    tries = tries if retry else 1
    assert all(len(c.attempts) == tries for c in psi)
    assert len(local) == (2 if fallback else 0)
    assert all(c.attempts[0].reason == "runtime_missing" for c in local)
    assert len(calls) == 2 * tries + 2
    assert all(c.field is not None for c in run.collections if c.attempts[0].provider == "crux")


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("recovered", [False, True])
def test_transient_fallback_obeys_final_attempt_and_profile(
    project, monkeypatch, capsys, retry, recovered
):
    calls, _, _ = harness(monkeypatch, capsys, psi_status=(500, 200) if recovered else 500)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)
    assert (
        main(
            command(
                project,
                {
                    "max_pages": 1,
                    "lighthouse_local": True,
                    "retry_transient": retry,
                },
            )
        )
        == 0
    )
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    psi = [c for c in run.collections if c.attempts[0].provider == "pagespeed_insights"]
    local = [c for c in run.collections if c.attempts[0].provider == "lighthouse_local"]
    assert [len(c.attempts) for c in psi] == [2 if retry else 1] * 2
    expected_local = 0 if recovered and retry else 1 if recovered else 2
    assert len(local) == expected_local
    assert all(len(c.attempts) == 1 for c in local)
    assert len(calls) == (6 if retry else 4)


def test_legacy_cli_transient_failures_do_not_gain_local_attempts(project, monkeypatch, capsys):
    harness(monkeypatch, capsys, psi_status=500)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)
    assert (
        main(
            command(
                project,
                {
                    "schema_version": "1.0.0",
                    "max_pages": 1,
                    "lighthouse_local": True,
                },
            )
        )
        == 0
    )
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    assert len(run.collections) == 4
    assert run.pipeline_version == "performance-1.0.0"


@pytest.mark.parametrize("failure", [None, "network_unverified"])
def test_transient_fallback_runs_once_per_page_device_and_retains_local_result(
    project, monkeypatch, capsys, failure
):
    from ai_search_audit.lighthouse_runtime import RuntimeResult
    from ai_search_audit.performance_models import LocalRuntimeFingerprint
    from tests.test_lighthouse_provider import fingerprint_data, lhr

    calls, _, _ = harness(monkeypatch, capsys, psi_status=503)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)
    local_requests = []

    class Runtime:
        def run(self, request):
            local_requests.append((request.requested_url, request.device))
            if failure:
                return RuntimeResult(failure=failure)
            payload = lhr()
            payload.update(requestedUrl=request.requested_url, finalUrl=request.requested_url)
            payload["configSettings"].update(formFactor=request.device, locale=request.locale)
            return RuntimeResult(
                body=json.dumps(payload).encode(),
                fingerprint=LocalRuntimeFingerprint(**fingerprint_data()),
            )

    monkeypatch.setattr("ai_search_audit.measurement_workflow._local_runtime", Runtime)
    assert main(command(project, {"max_pages": 2, "lighthouse_local": True})) == 0
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    expected = [
        (page.url, device)
        for page in run.preflight.selected_pages
        for device in run.preflight.devices
    ]
    assert len(expected) == 4
    assert local_requests == expected
    assert len(calls) == 12
    local = [c for c in run.collections if c.attempts[0].provider == "lighthouse_local"]
    assert len(local) == 4
    assert all(len(c.attempts) == 1 for c in local)
    if failure:
        assert all(c.lab is None and c.attempts[0].reason == failure for c in local)
    else:
        assert all(c.lab is not None and c.lab.performance_score == 70 for c in local)


@pytest.mark.parametrize(
    "failure",
    [
        "malformed_response",
        "response_too_large",
        "sensitive_response",
        "target_mismatch",
        "device_mismatch",
        "redirect",
        "partial",
        "available",
    ],
)
def test_nontransient_psi_does_not_trigger_local_runtime(project, monkeypatch, capsys, failure):
    def mutate(payload):
        lhr = payload["lighthouseResult"]
        if failure == "malformed_response":
            payload.clear()
        elif failure == "target_mismatch":
            lhr["requestedUrl"] = "https://other.example/"
        elif failure == "device_mismatch":
            settings = lhr["configSettings"]
            settings["formFactor"] = "desktop" if settings["formFactor"] == "mobile" else "mobile"
        elif failure == "sensitive_response":
            lhr["runWarnings"] = ["synthetic-psi-key"]
        elif failure == "available":
            lhr["audits"].update(
                {
                    name: {"numericValue": 1, "numericUnit": unit}
                    for name, unit in (
                        ("cumulative-layout-shift", "unitless"),
                        ("first-contentful-paint", "millisecond"),
                        ("total-blocking-time", "millisecond"),
                        ("speed-index", "millisecond"),
                    )
                }
            )

    calls, _, _ = harness(
        monkeypatch, capsys, psi_status=302 if failure == "redirect" else 200, psi_mutation=mutate
    )

    def forbidden(*args, **kwargs):
        pytest.fail("nontransient PSI must not invoke the local collector")

    monkeypatch.setattr(
        "ai_search_audit.measurement_workflow.LighthouseProvider.collect", forbidden
    )
    profile = {"max_pages": 1, "lighthouse_local": True}
    if failure == "response_too_large":
        profile["http_limits"] = {"max_response_bytes": 1}
    assert main(command(project, profile)) == 0
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    psi = [c for c in run.collections if c.attempts[0].provider == "pagespeed_insights"]
    assert len(run.collections) == 4
    assert len(calls) == 4
    if failure in {"partial", "available"}:
        assert all(c.attempts[-1].state.value == failure.upper() for c in psi)
    else:
        assert all(c.attempts[-1].reason == failure for c in psi)


def test_missing_keys_are_module_limits_and_second_run_does_not_overwrite(project, monkeypatch):
    for name in ("AUDIT_PAGESPEED_API_KEY", "AUDIT_CRUX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    args = command(project, {"max_pages": 1})
    assert main(args) == 0
    first = project / "diagnostics/public-v1/run-1"
    before = _hashes(first)
    assert main(args) == 0
    run = DiagnosticStore(project).load("public-v1", "run-2").run
    assert all(c.attempts[0].state is DataState.UNAVAILABLE for c in run.collections)
    assert before == _hashes(first)


@pytest.mark.parametrize(
    "profile",
    [
        {"gemini": True, "gemini_model": "gemini-2.5-flash", "paid_use_consent": True},
        {"api_key": FABRICATED_CREDENTIAL},
        {"http_limits": {"timeout_seconds": 999}},
    ],
)
def test_invalid_or_unsupported_profile_fails_before_preflight_or_publication(
    project, capsys, profile
):
    before = _hashes(project)
    assert main(command(project, profile)) == 2
    captured = capsys.readouterr()
    assert "MEASUREMENT_PREFLIGHT=" not in captured.out
    assert "MUST-NOT-PRINT" not in captured.err
    assert before == _hashes(project)
    if profile.get("gemini"):
        assert "Gemini" in captured.err


def test_workflow_rejects_unchecked_profile_before_any_io(project):
    assert util.find_spec("ai_search_audit.measurement_workflow"), "measurement workflow missing"
    from ai_search_audit.measurement_profile import MeasurementProfile

    module = import_module("ai_search_audit.measurement_workflow")
    with pytest.raises(ValueError):
        module.run_measurements(
            "project:example",
            clients_root=project.parent,
            source_version="public-v1",
            profile=MeasurementProfile().model_copy(
                update={"api_key": FABRICATED_UNCHECKED_CREDENTIAL}
            ),
            on_preflight=lambda plan: None,
        )


@pytest.mark.parametrize("status", [403, 500])
def test_guarded_runtime_process_boundary_is_used(project, monkeypatch, capsys, status):
    from ai_search_audit import lighthouse_runtime
    from tests.test_lighthouse_runtime import DockerHarness

    harness(monkeypatch, capsys, psi_status=status)
    monkeypatch.setattr("ai_search_audit.performance_providers.time.sleep", lambda value: None)
    docker = DockerHarness(lighthouse_runtime)
    monkeypatch.setattr(lighthouse_runtime, "_run_process", docker)
    monkeypatch.setattr(lighthouse_runtime, "_docker_available", lambda: True)
    monkeypatch.setenv("AUDIT_LIGHTHOUSE_IMAGE_ID", "sha256:" + "a" * 64)
    assert main(command(project, {"max_pages": 1, "lighthouse_local": True})) == 0
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    local = [c for c in run.collections if c.attempts[0].provider == "lighthouse_local"]
    assert len(local) == 2
    assert all(c.attempts[0].reason == "sandbox_unverified" for c in local)
    assert docker.commands


def test_malformed_profile_does_not_print_its_content(project, capsys):
    args = command(project, {})
    from pathlib import Path

    # Fabricated malformed intake, not an embedded credential assignment.
    Path(args[-1]).write_text('{"api' + '_key":"MUST-NOT-PRINT",invalid}')
    assert main(args) == 2
    assert "MUST-NOT-PRINT" not in capsys.readouterr().err


def test_profile_open_is_nonblocking_before_regular_file_validation(tmp_path, monkeypatch):
    import os

    from ai_search_audit.measurement_workflow import load_measurement_profile

    fifo = tmp_path / "profile-fifo"
    os.mkfifo(fifo)
    original = os.open

    def checked(path, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK, "profile opening must not block on special files"
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checked)
    with pytest.raises(ValueError, match="profile"):
        load_measurement_profile(fifo)


def test_unavailable_cli_result_is_explicit_in_source_branch_subprocess(project):
    import os
    import subprocess
    import sys
    from pathlib import Path

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    for name in ("AUDIT_PAGESPEED_API_KEY", "AUDIT_CRUX_API_KEY"):
        environment.pop(name, None)
    bootstrap = """
import runpy, socket
def denied(*a, **kw):
    raise AssertionError('network forbidden')
socket.socket.connect = denied
socket.create_connection = denied
runpy.run_module('ai_search_audit', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, *command(project, {"max_pages": 1})],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    summary = next((line for line in lines if line.startswith("MEASUREMENT_RESULT=")), None)
    assert summary is not None, "publication must not masquerade as a successful measurement"
    payload = json.loads(summary.split("=", 1)[1])
    assert payload["published"] is True
    assert payload["with_metrics"] == 0
    assert payload["states"] == {"UNAVAILABLE": 4}
    assert result.stderr == ""


def test_publication_failure_prints_no_result_and_preserves_previous_run(
    project, monkeypatch, capsys
):
    for name in ("AUDIT_PAGESPEED_API_KEY", "AUDIT_CRUX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    args = command(project, {"max_pages": 1})
    assert main(args) == 0
    capsys.readouterr()
    before = _hashes(project)

    def fail(*args, **kwargs):
        raise OSError("MUST-NOT-PRINT")

    monkeypatch.setattr("ai_search_audit.diagnostic_store._rename_directory_no_replace", fail)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "MEASUREMENT_RESULT=" not in captured.out
    assert "MUST-NOT-PRINT" not in captured.err
    assert before == _hashes(project)


def test_retry_disabled_and_invalid_runtime_image_never_launches(project, monkeypatch, capsys):
    from ai_search_audit import lighthouse_runtime

    calls, _, _ = harness(monkeypatch, capsys, psi_status=429)
    monkeypatch.setenv("AUDIT_LIGHTHOUSE_IMAGE_ID", "image:latest")

    def forbidden(*args, **kwargs):
        pytest.fail("unverified runtime must not execute")

    monkeypatch.setattr(lighthouse_runtime, "_run_process", forbidden)
    assert (
        main(command(project, {"max_pages": 1, "retry_transient": False, "lighthouse_local": True}))
        == 0
    )
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    assert len(calls) == 4
    assert all(len(c.attempts) == 1 for c in run.collections)
    assert all(
        c.attempts[0].reason == "runtime_unverified"
        for c in run.collections
        if c.attempts[0].provider == "lighthouse_local"
    )


def test_other_provider_secret_cannot_enter_persisted_evidence(project, monkeypatch, capsys):
    harness(monkeypatch, capsys, psi_warning="synthetic-crux-key")
    assert main(command(project, {"max_pages": 1})) == 2
    captured = capsys.readouterr()
    assert "synthetic-crux-key" not in captured.out + captured.err
    assert not (project / "diagnostics").exists()


def test_unchecked_nested_profile_does_not_leak_serialization_warnings(tmp_path):
    import warnings

    from ai_search_audit.measurement_profile import MeasurementProfile
    from ai_search_audit.measurement_workflow import run_measurements

    profile = MeasurementProfile().model_copy(
        update={"http_limits": {"timeout_seconds": 999, "api_key": FABRICATED_CREDENTIAL}}
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            run_measurements(
                "project:example",
                clients_root=tmp_path,
                source_version="public-v1",
                profile=profile,
                on_preflight=lambda value: None,
            )
    assert not captured, "invalid profile must not appear in serializer warnings"
