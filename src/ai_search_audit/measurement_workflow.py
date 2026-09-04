"""Bounded execution of a frozen performance plan, through the existing store."""

import os
import stat
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr

from .diagnostic_sources import load_diagnostic_source
from .diagnostic_store import DiagnosticStore
from .diagnostic_workflow import assemble_performance_run
from .lighthouse_provider import LighthouseProvider, Runtime
from .lighthouse_runtime import LighthouseRuntime, LighthouseRuntimeConfig, RuntimeResult
from .measurement_policy import allows_lighthouse_fallback
from .measurement_profile import (
    MeasurementPreflight,
    MeasurementProfile,
    _reject_unserialized_fields,
    prepare_measurement_preflight,
)
from .performance_http import PerformanceHTTPClient
from .performance_models import LighthouseRequest
from .performance_normalizers import decode_provider_json
from .performance_providers import (
    PerformanceCollection,
    PerformanceProviders,
    _model_contains_secret,
)


class UnsupportedMeasurementProfile(ValueError):
    """An explicitly enabled module is not implemented by this stage."""


def load_measurement_profile(path: Path) -> MeasurementProfile:
    """No secret fields, duplicate JSON keys, special files or unbounded inputs."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("profile must be a regular file")
            content = stream.read(65537)
        if not content or len(content) > 65536:
            raise ValueError("profile exceeds size bound")
        payload = decode_provider_json(content)
        return MeasurementProfile.model_validate(payload)
    except (OSError, ValueError):
        raise ValueError("invalid secret-free measurement profile") from None


class _UnavailableRuntime:
    def __init__(self, *, missing: bool) -> None:
        self._missing = missing

    def run(self, request: LighthouseRequest) -> RuntimeResult:
        return RuntimeResult(failure="runtime_missing" if self._missing else "runtime_unverified")


def _local_runtime() -> Runtime:
    image = os.environ.get("AUDIT_LIGHTHOUSE_IMAGE_ID")
    if not image:
        return _UnavailableRuntime(missing=True)
    try:
        config = LighthouseRuntimeConfig(image_id=image)
    except ValueError:
        return _UnavailableRuntime(missing=False)
    return LighthouseRuntime(config)


def run_measurements(
    project_ref: str,
    *,
    clients_root: Path,
    source_version: str,
    profile: MeasurementProfile,
    on_preflight: Callable[[MeasurementPreflight], None],
) -> Path:
    """Emit preflight before provider I/O; publish only a validated complete inventory.

    Each provider owns its bounded transient retry policy. A completed result is
    never repeated or selected for a better score. Failures remain module-local
    observations, not changes to canonical readiness or the source audit.
    """
    _reject_unserialized_fields(profile)
    profile = MeasurementProfile.model_validate(
        profile.model_dump(mode="python", serialize_as_any=True, warnings=False)
    )
    if profile.gemini:
        raise UnsupportedMeasurementProfile("Gemini execution is not supported by this stage")
    source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=source_version
    )
    preflight = prepare_measurement_preflight(source, profile)
    keys = {
        provider: SecretStr(value) if (value := os.environ.get(name)) else None
        for provider, name in (
            ("pagespeed_insights", "AUDIT_PAGESPEED_API_KEY"),
            ("crux", "AUDIT_CRUX_API_KEY"),
        )
    }
    if any(
        key is not None and _model_contains_secret(preflight, key.get_secret_value())
        for key in keys.values()
    ):
        raise ValueError("invalid performance configuration")
    on_preflight(preflight)
    providers = PerformanceProviders(http=PerformanceHTTPClient(limits=profile.http_limits))
    local = LighthouseProvider(runtime=_local_runtime()) if profile.lighthouse_local else None
    collections: list[PerformanceCollection] = []
    for page in preflight.selected_pages:
        for device in preflight.devices:
            psi = None
            if profile.pagespeed_insights:
                psi = providers.pagespeed(
                    source,
                    url=page.url,
                    device=device,
                    api_key=keys["pagespeed_insights"],
                    retry_transient=profile.retry_transient,
                )
                collections.append(psi)
            if profile.crux:
                collections.append(
                    providers.crux(
                        source,
                        url=page.url,
                        device=device,
                        api_key=keys["crux"],
                        retry_transient=profile.retry_transient,
                    )
                )
            if (
                local is not None
                and psi is not None
                and allows_lighthouse_fallback(
                    psi.attempts[-1], policy_version=profile.schema_version
                )
            ):
                collections.append(local.collect(source, url=page.url, device=device))
    run = assemble_performance_run(source, preflight, tuple(collections))
    if any(
        key is not None and _model_contains_secret(run, key.get_secret_value())
        for key in keys.values()
    ):
        raise ValueError("sensitive performance evidence")
    return DiagnosticStore(clients_root / source.binding.project_id).publish(run)
