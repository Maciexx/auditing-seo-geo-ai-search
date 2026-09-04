"""Node-only contract tests; deliberately no Chrome, npm install or target navigation."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).parents[1] / "src/ai_search_audit/lighthouse_assets/runner.mjs"


def node(expression, prelude=""):
    binary = shutil.which("node")
    if binary is None:
        pytest.skip("optional Node runtime is not installed")
    result = subprocess.run(
        [
            binary,
            "--input-type=module",
            "-e",
            prelude + f"const runner = await import({json.dumps(RUNNER.as_uri())});\n" + expression,
        ],
        capture_output=True,
        timeout=5,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr.decode()
    return json.loads(result.stdout)


def test_actual_chromium_sandbox_status_is_recognized_without_false_positives():
    status = (
        "Layer 1 Sandbox\tNamespace\nPID namespaces\tYes\nNetwork namespaces\tYes\n"
        "Seccomp-BPF sandbox\tYes\nSeccomp-BPF sandbox supports TSYNC\tYes"
    )
    assert (
        node(f"console.log(JSON.stringify(runner.sandboxVerified({json.dumps(status)})))") is True
    )
    for bad in (status.replace("Namespace", "None"), status.replace("Yes", "No", 1), ""):
        assert (
            node(f"console.log(JSON.stringify(runner.sandboxVerified({json.dumps(bad)})))") is False
        )


def test_runner_fixed_launch_options_never_inherit_environment():
    options = node("console.log(JSON.stringify(runner.launchOptions('/tmp/fresh-profile',12345)))")
    assert options["executablePath"] == "/usr/bin/chromium"
    assert options["pipe"] is True
    assert options["env"] == {"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "LANG": "C.UTF-8"}
    assert "--proxy-bypass-list=<-loopback>" in options["args"]
    assert "--no-sandbox" not in options["args"]


@pytest.mark.parametrize(
    "updates",
    [
        {"command": "evil"},
        {"device": "unknown"},
        {"locale": "fr"},
        {"requested_url": "file:///tmp/file"},
        {"requested_url": "https://user@perf.example/"},
        {"requested_url": "https://perf.example:444/"},
    ],
)
def test_runner_rejects_noncontract_input(updates):
    value = dict(requested_url="https://perf.example/", device="mobile", locale="en") | updates
    assert (
        node(
            f"let rejected=false;try{{runner.validateRequest({json.dumps(value)})}}"
            "catch{rejected=true}console.log(JSON.stringify(rejected))"
        )
        is True
    )


def test_browser_launch_rejection_is_runtime_unverified():
    assert (
        node(
            "let reason;try{await runner.launchBrowser({launch:async()=>{throw Error('SECRET')}},"
            "'/tmp/profile',12345)}catch(error){reason=error.message}console.log(JSON.stringify(reason))"
        )
        == "runtime_unverified"
    )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("exec", "network_unverified"),
        ("exec_timeout", "network_unverified"),
        ("goto", "sandbox_unverified"),
        ("evaluate", "sandbox_unverified"),
        ("newPage", "sandbox_unverified"),
    ],
)
def test_prenavigation_check_errors_are_fixed_unavailable_reasons(failure, expected):
    prelude = (
        "import cp from 'node:child_process';import {promisify} from 'node:util';"
        "import {syncBuiltinESMExports} from 'node:module';"
        "const stub=()=>{};stub[promisify.custom]=async()=>{"
        + (
            "throw Error('SECRET');"
            if failure.startswith("exec")
            else "return {stdout:JSON.stringify({network_isolated:true}),stderr:''};"
        )
        + "};cp.execFile=stub;syncBuiltinESMExports();"
    )
    expression = (
        f"const failed={json.dumps(failure)};const page={{"
        "goto:async()=>{if(failed==='goto')throw Error('SECRET')},"
        "evaluate:async()=>{throw Error('SECRET')},close:async()=>{}};"
        "const browser={newPage:async()=>{"
        "if(failed==='newPage')throw Error('SECRET');return page}};"
        "let reason;try{await runner.verifyIsolation(browser)}catch(error){reason=error.message}"
        "console.log(JSON.stringify(reason));"
    )
    assert node(expression, prelude) == expected
