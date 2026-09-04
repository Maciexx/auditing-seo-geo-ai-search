"""Fixed Docker launcher. Never builds/pulls; owns only fresh per-attempt resources.

The local Docker daemon and operator-selected immutable image are trusted. Client
code runs only in the networkless browser container, never in the proxy sidecar.
Raw stdout is transient and excluded from repr; stderr is bounded then discarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import ConfigDict, Field

from ai_search_audit.lighthouse_network import parse_authority
from ai_search_audit.performance_models import (
    FrozenPerformanceModel,
    LighthouseRequest,
    LocalRuntimeFingerprint,
    _hostname,
)
from ai_search_audit.performance_normalizers import decode_provider_json

ASSETS = Path(__file__).with_name("lighthouse_assets")
CONTAINER_ROOT = "/opt/audit-lighthouse"
MAX_OUTPUT = 8388608
FailureCode = Literal[
    "runtime_missing",
    "runtime_unverified",
    "sandbox_unverified",
    "network_unverified",
    "timeout",
    "response_too_large",
    "runtime_error",
    "malformed_response",
    "cleanup_failed",
]


class RuntimeFailure(Exception):
    def __init__(self, code: FailureCode) -> None:
        self.code = code
        super().__init__(code)


class LighthouseRuntimeConfig(FrozenPerformanceModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    timeout_seconds: float = Field(default=120.0, ge=10, le=180)
    max_output_bytes: int = Field(default=MAX_OUTPUT, ge=1024, le=MAX_OUTPUT)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes = field(repr=False)


@dataclass(frozen=True)
class RuntimeResult:
    body: bytes = field(default=b"", repr=False)
    fingerprint: LocalRuntimeFingerprint | None = None
    failure: FailureCode | None = None


def _run_process(
    argv: tuple[str, ...],
    *,
    deadline: float,
    max_bytes: int,
    input_bytes: bytes | None = None,
) -> ProcessResult:
    """Bound both streams while reading; kill this owned group on every exit path."""
    if time.monotonic() >= deadline:
        raise RuntimeFailure("timeout")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    output = bytearray()
    captured = 0
    selector = selectors.DefaultSelector()
    try:
        assert process.stdout is not None and process.stderr is not None
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        pending = memoryview(input_bytes or b"")
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeFailure("timeout")
            for key, event in selector.select(min(remaining, 0.1)):
                selected_stream = key.fileobj
                if event == selectors.EVENT_WRITE:
                    if pending:
                        count = os.write(key.fd, pending[:4096])
                        pending = pending[count:]
                    if not pending:
                        selector.unregister(selected_stream)
                        assert process.stdin is not None
                        process.stdin.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(selected_stream)
                    continue
                captured += len(chunk)
                if captured > max_bytes:
                    raise RuntimeFailure("response_too_large")
                if selected_stream is process.stdout:
                    output.extend(chunk)
        try:
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise RuntimeFailure("timeout") from None
        return ProcessResult(returncode, bytes(output))
    finally:
        selector.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for final_stream in (process.stdin, process.stdout, process.stderr):
            if final_stream is not None:
                final_stream.close()


def _docker_binary() -> str:
    return "/usr/local/bin/docker" if Path("/usr/local/bin/docker").is_file() else "/usr/bin/docker"


def _docker_socket() -> Path:
    desktop = Path.home() / ".docker" / "run" / "docker.sock"
    return desktop if desktop.exists() else Path("/var/run/docker.sock")


def _docker_available() -> bool:
    try:
        return Path(_docker_binary()).is_file() and stat.S_ISSOCK(_docker_socket().stat().st_mode)
    except OSError:
        return False


def _architecture() -> Literal["arm64", "amd64"]:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    raise RuntimeFailure("runtime_unverified")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_labels() -> dict[str, str]:
    return {
        "org.ai-search-audit.runner": _sha(ASSETS / "runner.mjs"),
        "org.ai-search-audit.seccomp": _sha(ASSETS / "seccomp.json"),
        "org.ai-search-audit.lock": _sha(ASSETS / "package-lock.json"),
        "org.ai-search-audit.isolation": _sha(ASSETS / "isolation.py"),
        "org.ai-search-audit.sidecar": _sha(ASSETS / "sidecar.py"),
        "org.ai-search-audit.network": _sha(Path(__file__).with_name("lighthouse_network.py")),
        "org.ai-search-audit.proxy": _sha(Path(__file__).with_name("lighthouse_proxy.py")),
    }


class LighthouseRuntime:
    def __init__(self, config: LighthouseRuntimeConfig) -> None:
        self.config = LighthouseRuntimeConfig.model_validate(
            vars(config) | (config.model_extra or {})
        )

    def run(self, request: LighthouseRequest) -> RuntimeResult:
        request = LighthouseRequest.model_validate(vars(request) | (request.model_extra or {}))
        if not _docker_available():
            return RuntimeResult(failure="runtime_missing")
        try:
            with tempfile.TemporaryDirectory(prefix="audit-lighthouse-") as directory:
                # Isolated CLI configuration: no registry credentials or user plugins.
                Path(directory, "config.json").write_text("{}", encoding="utf-8")
                return self._run(request, Path(directory))
        except (OSError, ValueError):
            return RuntimeResult(failure="runtime_unverified")

    def _run(self, request: LighthouseRequest, directory: Path) -> RuntimeResult:
        from urllib.parse import urlsplit

        parts = urlsplit(request.requested_url)
        # Raw URL syntax/credentials/port were already validated by LighthouseRequest.
        # Normalize only this policy authority; keep the original evidence URL intact.
        host = _hostname(parts.hostname or "")
        authority = f"[{host}]" if ":" in host else host
        if parts.port is not None:
            authority += f":{parts.port}"
        parse_authority(authority, 443 if parts.scheme == "https" else 80)
        prefix = (
            _docker_binary(),
            "--host",
            "unix://" + str(_docker_socket()),
            "--config",
            str(directory),
        )
        deadline = time.monotonic() + self.config.timeout_seconds
        labels = _asset_labels()
        owned_containers: list[str] = []
        volume: str | None = None
        owner = uuid4().hex
        result = RuntimeResult(failure="runtime_error")

        def call(
            args: tuple[str, ...], *, body: bytes | None = None, cleanup: bool = False
        ) -> ProcessResult:
            return _run_process(
                prefix + args,
                deadline=time.monotonic() + 10 if cleanup else deadline,
                max_bytes=self.config.max_output_bytes,
                input_bytes=body,
            )

        def checked(args: tuple[str, ...], *, body: bytes | None = None) -> bytes:
            response = call(args, body=body)
            if response.returncode:
                raise RuntimeFailure("runtime_error")
            return response.stdout

        def attest(kind: str, identifier: str, *, cleanup: bool = False) -> bool:
            inspection = call(
                (kind, "inspect", "--format", "{{json .}}", identifier), cleanup=cleanup
            )
            if inspection.returncode:
                return False
            metadata = decode_provider_json(inspection.stdout)
            config = metadata.get("Config") if kind == "container" else metadata
            resource_labels = config.get("Labels") if isinstance(config, dict) else None
            return (
                isinstance(resource_labels, dict)
                and resource_labels.get("org.ai-search-audit.owner") == owner
            )

        try:
            inspection = call(("image", "inspect", "--format", "{{json .}}", self.config.image_id))
            if inspection.returncode:
                raise RuntimeFailure("runtime_missing")
            image = decode_provider_json(inspection.stdout)
            config = image.get("Config")
            image_labels = config.get("Labels") if isinstance(config, dict) else None
            architecture = _architecture()
            daemon_arch = checked(("info", "--format", "{{.Architecture}}")).decode().strip()
            if daemon_arch in {"aarch64", "x86_64"}:
                daemon_arch = "arm64" if daemon_arch == "aarch64" else "amd64"
            if (
                image.get("Id") != self.config.image_id
                or image.get("Os") != "linux"
                or image.get("Architecture") != architecture
                or daemon_arch != architecture
                or not isinstance(image_labels, dict)
                or any(image_labels.get(key) != value for key, value in labels.items())
            ):
                raise RuntimeFailure("runtime_unverified")
            token = uuid4().hex
            volume = "audit-lh-socket-" + token
            checked(
                (
                    "volume",
                    "create",
                    "--label",
                    "org.ai-search-audit.owner=" + owner,
                    "--name",
                    volume,
                )
            )
            if not attest("volume", volume):
                raise RuntimeFailure("runtime_unverified")
            common = (
                "create",
                "--pull=never",
                "--init",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--user=1000:1000",
                "--pids-limit=256",
                "--memory=1g",
                "--cpus=2",
                "--ipc=private",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=512m",
                "--shm-size=256m",
                "--log-driver=none",
            )
            for role, network, executable, script, readonly in (
                ("proxy", "bridge", "/usr/bin/python3", "sidecar.py", ""),
                ("browser", "none", "/usr/local/bin/node", "runner.mjs", ",readonly"),
            ):
                name = "audit-lh-" + role + "-" + token
                # Reserve fresh names before create so interrupted CLI calls still clean up.
                owned_containers.append(name)
                security = (
                    ("--security-opt=seccomp=" + str(ASSETS / "seccomp.json"),)
                    if role == "browser"
                    else ()
                )
                args = (
                    common
                    + security
                    + (
                        "--name",
                        name,
                        "--label",
                        "org.ai-search-audit.owner=" + owner,
                        "--network=" + network,
                        "--mount=type=volume,src=" + volume + ",dst=/run/audit-proxy" + readonly,
                        "--entrypoint",
                        executable,
                    )
                    + (("--interactive",) if role == "browser" else ())
                    + (
                        self.config.image_id,
                        CONTAINER_ROOT + "/" + script,
                    )
                )
                identifier = checked(args).decode().strip()
                if not re.fullmatch(r"[a-f0-9]{64}", identifier):
                    raise RuntimeFailure("runtime_error")
                # Use immutable IDs after success; an ambiguous create timeout
                # still uses its reserved name, but only after ownership attestation.
                owned_containers[-1] = identifier
                if not attest("container", identifier):
                    raise RuntimeFailure("runtime_unverified")
                if role == "proxy":
                    checked(("start", identifier))
                else:
                    body = request.model_dump_json().encode()
                    raw = checked(("start", "--attach", "--interactive", identifier), body=body)
                    envelope = decode_provider_json(raw)
                    if envelope.get("status") != "ok":
                        reason = envelope.get("reason")
                        if reason == "runtime_unverified":
                            raise RuntimeFailure("runtime_unverified")
                        if reason == "response_too_large":
                            raise RuntimeFailure("response_too_large")
                        if reason in {"sandbox_unverified", "network_unverified"}:
                            raise RuntimeFailure(
                                "sandbox_unverified"
                                if reason == "sandbox_unverified"
                                else "network_unverified"
                            )
                        raise RuntimeFailure("runtime_error")
                    versions = envelope.get("versions")
                    if (
                        not isinstance(versions, dict)
                        or versions
                        != {
                            "node": "22.19.0",
                            "lighthouse": "13.4.1",
                            "chrome": "152.0.7977.75",
                            "puppeteer": "25.10.0",
                        }
                        or envelope.get("isolation") is not True
                    ):
                        raise RuntimeFailure("runtime_unverified")
                    policy = json.dumps(
                        {
                            "labels": labels,
                            "limits": self.config.model_dump(),
                            "common": common,
                            "launcher_sha256": _sha(Path(__file__)),
                        },
                        sort_keys=True,
                    ).encode()
                    fingerprint = LocalRuntimeFingerprint(
                        image_id=self.config.image_id,
                        architecture=architecture,
                        node_version=versions["node"],
                        lighthouse_version=versions["lighthouse"],
                        chrome_version=versions["chrome"],
                        puppeteer_version=versions["puppeteer"],
                        runner_sha256=labels["org.ai-search-audit.runner"],
                        seccomp_sha256=labels["org.ai-search-audit.seccomp"],
                        dependency_lock_sha256=labels["org.ai-search-audit.lock"],
                        policy_sha256=hashlib.sha256(policy).hexdigest(),
                    )
                    result = RuntimeResult(
                        body=json.dumps(envelope.get("lhr")).encode(), fingerprint=fingerprint
                    )
        except RuntimeFailure as error:
            result = RuntimeResult(failure=error.code)
        except (OSError, ValueError, TypeError, KeyError):
            result = RuntimeResult(failure="runtime_error")
        finally:
            clean = True
            for identifier in reversed(owned_containers):
                try:
                    clean = (
                        attest("container", identifier, cleanup=True)
                        and call(("rm", "--force", identifier), cleanup=True).returncode == 0
                        and clean
                    )
                except (RuntimeFailure, OSError, ValueError):
                    clean = False
            if volume is not None:
                try:
                    clean = (
                        attest("volume", volume, cleanup=True)
                        and call(("volume", "rm", volume), cleanup=True).returncode == 0
                        and clean
                    )
                except (RuntimeFailure, OSError, ValueError):
                    clean = False
            if not clean:
                result = RuntimeResult(failure="cleanup_failed")
        return result
