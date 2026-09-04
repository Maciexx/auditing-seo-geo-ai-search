import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


def runtime():
    assert importlib.util.find_spec("ai_search_audit.lighthouse_runtime"), "runtime is missing"
    from ai_search_audit import lighthouse_runtime

    return lighthouse_runtime


def test_process_capture_has_minimal_environment_and_bounded_output():
    module = runtime()
    result = module._run_process(
        (sys.executable, "-c", "import os,json;print(json.dumps(dict(os.environ)))"),
        deadline=time.monotonic() + 5,
        max_bytes=4096,
    )
    assert result.returncode == 0
    # macOS CoreFoundation injects this noncredential value during Python startup.
    assert set(json.loads(result.stdout)) <= {"PATH", "LANG", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
    assert "HOME" not in json.loads(result.stdout)
    with pytest.raises(module.RuntimeFailure, match="response_too_large"):
        module._run_process(
            (sys.executable, "-c", "print('x'*20000)"), deadline=time.monotonic() + 5, max_bytes=100
        )


def test_process_timeout_kills_its_process_group():
    module = runtime()
    started = time.monotonic()
    with pytest.raises(module.RuntimeFailure, match="timeout"):
        module._run_process(
            (sys.executable, "-c", "import os,time;os.fork();time.sleep(30)"),
            deadline=started + 0.15,
            max_bytes=100,
        )
    assert time.monotonic() - started < 3


def test_stderr_is_bounded_and_never_returned():
    module = runtime()
    result = module._run_process(
        (sys.executable, "-c", "import sys;print('SECRET',file=sys.stderr)"),
        deadline=time.monotonic() + 5,
        max_bytes=100,
    )
    assert "SECRET" not in repr(result)
    with pytest.raises(module.RuntimeFailure, match="response_too_large"):
        module._run_process(
            (sys.executable, "-c", "import sys;sys.stderr.write('x'*99999)"),
            deadline=time.monotonic() + 5,
            max_bytes=100,
        )


def test_runtime_configuration_requires_local_pinned_image():
    module = runtime()
    for image in ("image:latest", "https://image.example/", "sha256:bad"):
        with pytest.raises(ValueError):
            module.LighthouseRuntimeConfig(image_id=image)
    config = module.LighthouseRuntimeConfig(image_id="sha256:" + "a" * 64)
    assert not hasattr(config, "command")


class DockerHarness:
    def __init__(self, module, failure=None):
        self.module, self.failure, self.commands = module, failure, []
        self.owner = ""

    def __call__(self, argv, *, deadline, max_bytes, input_bytes=None):
        self.commands.append((argv, input_bytes))
        args = argv[5:]
        if "--label" in args:
            self.owner = args[args.index("--label") + 1].split("=", 1)[1]
        if args[:2] == ("volume", "inspect"):
            return self.module.ProcessResult(
                0, json.dumps({"Labels": {"org.ai-search-audit.owner": self.owner}}).encode()
            )
        if args[:2] == ("container", "inspect"):
            return self.module.ProcessResult(
                0,
                json.dumps(
                    {"Config": {"Labels": {"org.ai-search-audit.owner": self.owner}}}
                ).encode(),
            )
        if self.failure and self.failure in args:
            raise self.module.RuntimeFailure("timeout")
        if args[:2] == ("image", "inspect"):
            body = dict(
                Id="sha256:" + "a" * 64,
                Os="linux",
                Architecture=self.module._architecture(),
                Config={"Labels": self.module._asset_labels()},
            )
            return self.module.ProcessResult(0, json.dumps(body).encode())
        if args[:2] == ("info", "--format"):
            return self.module.ProcessResult(0, self.module._architecture().encode())
        if args[0] == "create":
            return self.module.ProcessResult(
                0, ("b" if "--network=none" in args else "c").encode() * 64
            )
        if args[0] == "start" and "--attach" in args:
            return self.module.ProcessResult(
                0, b'{"status":"unavailable","reason":"sandbox_unverified"}'
            )
        return self.module.ProcessResult(0, b"")


@pytest.mark.parametrize("failure", [None, "create", "start"])
def test_launch_is_fixed_isolated_and_owned_resources_are_cleaned(monkeypatch, failure):
    module = runtime()
    harness = DockerHarness(module, failure)
    monkeypatch.setattr(module, "_run_process", harness)
    monkeypatch.setattr(module, "_docker_available", lambda: True)
    request = module.LighthouseRequest(
        requested_url="https://perf.example/", device="mobile", locale="en"
    )
    result = module.LighthouseRuntime(
        module.LighthouseRuntimeConfig(image_id="sha256:" + "a" * 64)
    ).run(request)
    assert result.failure in {"sandbox_unverified", "timeout"}
    commands = [command for command, _ in harness.commands]
    assert all(command[0] in {"/usr/local/bin/docker", "/usr/bin/docker"} for command in commands)
    assert not any("pull" in command or "build" in command for command in commands)
    creates = [command for command in commands if "create" in command and "volume" not in command]
    for command in creates:
        for flag in (
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--pull=never",
        ):
            assert flag in command
        assert not any("type=bind" in part or "docker.sock" in part for part in command[5:])
        assert "--no-sandbox" not in " ".join(command)
    browser = [command for command in creates if "--network=none" in command]
    if browser:
        assert any("readonly" in part for part in browser[0])
        assert "/usr/local/bin/node" in browser[0]
        assert "/opt/audit-lighthouse/runner.mjs" in browser[0]
    assert any("volume" in command and "rm" in command for command in commands)


def run_harness(monkeypatch, harness, url="https://perf.example/"):
    module = runtime()
    monkeypatch.setattr(module, "_run_process", harness)
    monkeypatch.setattr(module, "_docker_available", lambda: True)
    return module.LighthouseRuntime(
        module.LighthouseRuntimeConfig(image_id="sha256:" + "a" * 64)
    ).run(module.LighthouseRequest(requested_url=url, device="mobile", locale="en"))


@pytest.mark.parametrize(
    "url",
    [
        "https://straße.example/",
        "https://straße.example:443/",
        "https://xn--strae-oqa.example/",
        "https://[2606:4700:4700::1111]:443/",
    ],
)
def test_idna_authority_validation_preserves_original_request_url(monkeypatch, url):
    module = runtime()
    harness = DockerHarness(module)
    result = run_harness(monkeypatch, harness, url)
    assert result.failure == "sandbox_unverified"
    bodies = [json.loads(body) for _, body in harness.commands if body]
    assert bodies == [{"requested_url": url, "device": "mobile", "locale": "en"}]


def test_cleanup_attests_ownership_labels_before_removing_resources(monkeypatch):
    module = runtime()
    harness = DockerHarness(module)
    run_harness(monkeypatch, harness)
    commands = [command[5:] for command, _ in harness.commands]
    for index, command in enumerate(commands):
        if command[0] == "create" or command[:2] == ("volume", "create"):
            assert "--label" in command
        if "rm" in command:
            assert any(
                "inspect" in earlier and command[-1] in earlier for earlier in commands[:index]
            )


@pytest.mark.parametrize("kind", ["volume", "container"])
def test_owner_mismatch_never_removes_colliding_resource(monkeypatch, kind):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if argv[5:7] == (kind, "inspect"):
            body = {"Labels": {"org.ai-search-audit.owner": "foreign"}}
            if kind == "container":
                body = {"Config": body}
            return module.ProcessResult(0, json.dumps(body).encode())
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure == "cleanup_failed"
    assert not any(
        command[5:7] == ("volume", "rm") if kind == "volume" else command[5] == "rm"
        for command, _ in harness.commands
    )


def test_missing_runtime_does_not_launch_or_build(monkeypatch):
    module = runtime()
    monkeypatch.setattr(module, "_docker_available", lambda: False)
    monkeypatch.setattr(
        module, "_run_process", lambda *args, **kwargs: pytest.fail("process started")
    )
    request = module.LighthouseRequest(
        requested_url="https://perf.example/", device="mobile", locale="en"
    )
    result = module.LighthouseRuntime(
        module.LighthouseRuntimeConfig(image_id="sha256:" + "a" * 64)
    ).run(request)
    assert result.failure == "runtime_missing"


def test_runner_assets_require_sandbox_network_checks_before_navigation():
    module = runtime()
    asset = Path(module.__file__).with_name("lighthouse_assets")
    runner = (asset / "runner.mjs").read_text()
    assert "--no-sandbox" not in runner
    assert "userDataDir" in runner and "mkdtemp" in runner
    assert "pipe: true" in runner
    assert runner.index("await verifyIsolation") < runner.index("await lighthouse(")
    assert "--proxy-bypass-list=<-loopback>" in runner
    assert "chrome://sandbox" in runner
    assert '"lighthouse": "13.4.1"' in (asset / "package.json").read_text()


def test_isolation_checker_fails_closed_for_capabilities_routes_or_interfaces():
    import runpy

    module = runtime()
    asset = Path(module.__file__).with_name("lighthouse_assets") / "isolation.py"
    assert asset.is_file(), "kernel isolation checker is missing"
    checker = runpy.run_path(str(asset))["status_isolated"]
    status = "CapEff:\t0000000000000000\nNoNewPrivs:\t1\nSeccomp:\t2\n"
    assert checker(status, {"lo": 0x9}, "Iface Destination Gateway\n")
    assert not checker(status.replace("Seccomp:\t2", "Seccomp:\t0"), {"lo": 0x9}, "")
    assert not checker(status.replace("0000000000000000", "0000000000000001"), {"lo": 0x9}, "")
    assert not checker(status, {"lo": 0x9, "eth0": 0x1}, "")
    assert not checker(status, {"lo": 0x9}, "Iface Destination Gateway\neth0 00000000 01000000")


def test_isolation_accepts_real_down_tunnels_but_rejects_up_or_malformed_flags():
    import runpy

    checker = runpy.run_path(str(runtime().ASSETS / "isolation.py"))["status_isolated"]
    status = "CapEff:\t0000000000000000\nNoNewPrivs:\t1\nSeccomp:\t2\n"
    observed = {
        "lo": 0x9,
        "tunl0": 0x80,
        "gre0": 0x80,
        "gretap0": 0x1002,
        "erspan0": 0x1002,
        "ip_vti0": 0x80,
        "ip6_vti0": 0x80,
        "sit0": 0x80,
        "ip6tnl0": 0x80,
        "ip6gre0": 0x80,
    }
    assert checker(status, observed, "Iface Destination Gateway\n")
    for malformed in (
        {},
        {"lo": 0x8},
        {"lo": 0x1},
        {"lo": 0x9, "eth0": 0x1003},
        {"lo": 0x9, "gre0": None},
        {"lo": "0x9"},
        {"lo": 0x9, "tunl0": -1},
    ):
        assert not checker(status, malformed, "Iface Destination Gateway\n")


@pytest.mark.parametrize("name,flags", [("../bad", "0x80\n"), ("tunl0", "junk"), ("tunl0", None)])
def test_namespace_interface_enumeration_rejects_bad_names_or_flag_files(monkeypatch, name, flags):
    import runpy
    import socket

    namespace = runpy.run_path(str(runtime().ASSETS / "isolation.py"))
    assert "interface_flags" in namespace, "validated namespace interface enumeration missing"
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "lo"), (2, name)])

    def read_flags(path, *args, **kwargs):
        if path.parent.name == "lo":
            return "0x9\n"
        if flags is None:
            raise FileNotFoundError()
        return flags

    monkeypatch.setattr(Path, "read_text", read_flags)
    with pytest.raises((ValueError, OSError)):
        namespace["interface_flags"]()


def test_build_assets_pin_dependencies_and_match_installed_image_attestation():
    module = runtime()
    dockerfile = module.ASSETS / "Dockerfile"
    assert dockerfile.is_file(), "operator-only Dockerfile missing"
    content = dockerfile.read_text()
    assert "FROM node:22.19.0-bookworm-slim@sha256:" in content
    assert "chromium=152.0.7977.75-1~deb12u1" in content
    assert "npm ci --ignore-scripts" in content
    assert "USER 1000:1000" in content
    for label, digest in module._asset_labels().items():
        assert f'{label}="{digest}"' in content
    lock = json.loads((module.ASSETS / "package-lock.json").read_text())
    assert lock["packages"][""]["dependencies"] == {
        "lighthouse": "13.4.1",
        "puppeteer-core": "25.10.0",
    }
    for name, package in lock["packages"].items():
        if name:
            assert package["resolved"].startswith("https://registry.npmjs.org/")
            assert package["integrity"].startswith("sha512-")
    seccomp = json.loads((module.ASSETS / "seccomp.json").read_text())
    assert seccomp["defaultAction"] == "SCMP_ACT_ERRNO"
    assert seccomp["syscalls"][-1] == {
        "names": ["clone", "setns", "unshare", "chroot"],
        "action": "SCMP_ACT_ALLOW",
    }


def success_envelope():
    from tests.test_lighthouse_provider import lhr

    return {
        "status": "ok",
        "isolation": True,
        "versions": {
            "node": "22.19.0",
            "lighthouse": "13.4.1",
            "chrome": "152.0.7977.75",
            "puppeteer": "25.10.0",
        },
        "lhr": lhr(),
    }


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"isolation": False}, "runtime_unverified"),
        ({"isolation": 1}, "runtime_unverified"),
        ({"versions": {}}, "runtime_unverified"),
        ({"status": "unavailable", "reason": "network_unverified"}, "network_unverified"),
        ({"status": "unavailable", "reason": "runtime_unverified"}, "runtime_unverified"),
        ({"status": "unavailable", "reason": "SECRET"}, "runtime_error"),
    ],
)
def test_unverified_runtime_envelopes_are_fixed_failure_without_lhr(
    monkeypatch, mutation, expected
):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if "--attach" in argv:
            return module.ProcessResult(0, json.dumps(success_envelope() | mutation).encode())
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure == expected
    assert result.body == b"" and result.fingerprint is None
    assert "SECRET" not in repr(result)


def test_runner_output_limit_reason_is_preserved_without_raw_lhr(monkeypatch):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if "--attach" in argv:
            return module.ProcessResult(
                0, b'{"status":"unavailable","reason":"response_too_large"}'
            )
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure == "response_too_large"
    assert result.body == b"" and result.fingerprint is None


def test_success_returns_exact_provenance_only_after_cleanup(monkeypatch):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if "--attach" in argv:
            return module.ProcessResult(0, json.dumps(success_envelope()).encode())
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure is None
    assert result.fingerprint.image_id == "sha256:" + "a" * 64
    assert result.fingerprint.chrome_version == "152.0.7977.75"
    assert result.fingerprint.runner_sha256 == module._asset_labels()["org.ai-search-audit.runner"]
    assert json.loads(result.body) == success_envelope()["lhr"]
    assert harness.commands[-1][0][5:7] == ("volume", "rm")
    assert not any("perf.example" in str(command) for command, _ in harness.commands)
    assert any(body and b"perf.example" in body for _, body in harness.commands)


def test_launcher_policy_changes_invalidate_runtime_fingerprint(monkeypatch):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if "--attach" in argv:
            return module.ProcessResult(0, json.dumps(success_envelope()).encode())
        return result

    original = run_harness(monkeypatch, dispatch).fingerprint.policy_sha256
    sha = module._sha
    monkeypatch.setattr(
        module, "_sha", lambda path: "f" * 64 if path == Path(module.__file__) else sha(path)
    )
    changed = run_harness(monkeypatch, dispatch).fingerprint.policy_sha256
    assert original != changed


def test_cleanup_failure_discards_successful_lhr(monkeypatch):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if "--attach" in argv:
            return module.ProcessResult(0, json.dumps(success_envelope()).encode())
        if "rm" in argv:
            return module.ProcessResult(1, b"")
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure == "cleanup_failed" and result.body == b""


@pytest.mark.parametrize(
    "mutation", [{"Id": "sha256:" + "f" * 64}, {"Os": "windows"}, {"Config": {"Labels": {}}}]
)
def test_unverified_image_never_creates_browser(monkeypatch, mutation):
    module = runtime()
    harness = DockerHarness(module)

    def dispatch(argv, **kwargs):
        result = harness(argv, **kwargs)
        if argv[5:7] == ("image", "inspect"):
            return module.ProcessResult(
                0, json.dumps(json.loads(result.stdout) | mutation).encode()
            )
        return result

    result = run_harness(monkeypatch, dispatch)
    assert result.failure == "runtime_unverified"
    assert not any("create" in command for command, _ in harness.commands)
