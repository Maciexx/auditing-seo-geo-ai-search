"""Pure versioned eligibility shared by execution and persisted inventory validation."""

from typing import Literal

from ai_search_audit.models import DataState
from ai_search_audit.performance_models import ProviderAttempt

PerformancePipeline = Literal["performance-1.0.0", "performance-1.1.0"]


def performance_pipeline_version(policy_version: str) -> PerformancePipeline:
    """Reject unknown policies rather than interpreting them as historical defaults."""
    if policy_version == "1.0.0":
        return "performance-1.0.0"
    if policy_version == "1.1.0":
        return "performance-1.1.0"
    raise ValueError("unknown measurement policy version")


def allows_lighthouse_fallback(attempt: ProviderAttempt, *, policy_version: str) -> bool:
    """Use only the final normalized PSI outcome; this does not authorize local I/O."""
    performance_pipeline_version(policy_version)
    if attempt.provider != "pagespeed_insights":
        return False
    if attempt.state is DataState.UNAVAILABLE:
        return True
    if policy_version == "1.0.0" or attempt.state is not DataState.FAILED:
        return False
    if attempt.http_status in {500, 502, 503, 504}:
        return attempt.reason == f"http_{attempt.http_status}"
    return attempt.http_status is None and attempt.reason in {"timeout", "transport_error"}
