"""The versioned fallback gate is pure and uses exact normalized failure evidence."""

from datetime import UTC, datetime
from importlib import import_module, util

import pytest

from ai_search_audit.performance_models import ProviderAttempt


def policy():
    assert util.find_spec("ai_search_audit.measurement_policy"), "shared fallback policy missing"
    return import_module("ai_search_audit.measurement_policy").allows_lighthouse_fallback


def attempt(*, state="FAILED", reason="http_500", status=500, provider="pagespeed_insights"):
    return ProviderAttempt(
        attempt_id="attempt-policy",
        provider=provider,
        requested_url="https://studio.example/",
        device="mobile",
        started_at=datetime(2026, 9, 4, tzinfo=UTC),
        ended_at=datetime(2026, 9, 4, tzinfo=UTC),
        state=state,
        reason=reason,
        http_status=status,
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_only_current_policy_allows_matching_transient_http_failure(version, status):
    assert policy()(attempt(reason=f"http_{status}", status=status), policy_version=version) is (
        version == "1.1.0"
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
@pytest.mark.parametrize("reason", ["timeout", "transport_error"])
def test_only_current_policy_allows_transport_failure_without_status(version, reason):
    assert policy()(attempt(reason=reason, status=None), policy_version=version) is (
        version == "1.1.0"
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
def test_unavailable_psi_retains_legacy_eligibility(version):
    assert policy()(
        attempt(state="UNAVAILABLE", reason="unsafe_target", status=None), policy_version=version
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
@pytest.mark.parametrize("provider", ["crux", "lighthouse_local"])
def test_non_psi_is_never_eligible(version, provider):
    assert not policy()(
        attempt(provider=provider, state="UNAVAILABLE", reason="missing_key", status=None),
        policy_version=version,
    )


@pytest.mark.parametrize("version", ["", "1.2.0", "2.0.0", None])
def test_unknown_policy_rejects_before_provider_or_state_checks(version):
    with pytest.raises(ValueError, match="policy"):
        policy()(
            attempt(provider="crux", state="UNAVAILABLE", reason="missing_key", status=None),
            policy_version=version,
        )


@pytest.mark.parametrize(
    "state,reason,status",
    [
        ("FAILED", "http_501", 501),
        ("FAILED", "http_500", 502),
        ("FAILED", "http_500", None),
        ("FAILED", "timeout", 200),
        ("FAILED", "timeout", 500),
        ("FAILED", "transport_error", 200),
        ("FAILED", "transport_error", 503),
        ("FAILED", "redirect", 302),
        ("FAILED", "http_429", 429),
        ("FAILED", "http_403", 403),
        ("PARTIAL", "missing_metrics", 200),
        ("UNKNOWN", "target_mismatch", 500),
        ("AVAILABLE", None, 200),
        *[
            ("FAILED", reason, status)
            for reason in (
                "malformed_response",
                "response_too_large",
                "sensitive_response",
                "target_mismatch",
                "device_mismatch",
                "redirect",
                "unsafe_target",
            )
            for status in (None, 200, 500)
        ],
    ],
)
def test_other_states_and_mismatched_failure_evidence_are_ineligible(state, reason, status):
    assert not policy()(attempt(state=state, reason=reason, status=status), policy_version="1.1.0")
