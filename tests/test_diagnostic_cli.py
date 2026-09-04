import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_search_audit.cli import build_parser, main
from ai_search_audit.data_intake import IntakeOwnershipError, create_owned_intake_dir
from ai_search_audit.diagnostic_models import DiagnosticContract
from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.diagnostic_workflow import prepare_diagnostic_contract
from ai_search_audit.models import DataState
from tests.test_project_orchestrator import _create

ROOT = Path(__file__).parents[1]


def _hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def project(tmp_path):
    _create(tmp_path, domain="https://studio.example", client_name="Example Studio")
    return tmp_path / "clients/example"


def _command(root, name, *, source="public-v1"):
    return [
        "project",
        name,
        "project:example",
        "--source-version",
        source,
        "--clients-root",
        str(root.parent),
    ]


def _process(root, *, source="public-v1", installed=False, arguments=None):
    # Mock only HTTP/DNS; collection, CLI, owned intake, persistence and reload remain real.
    bootstrap = """
import httpx, runpy, socket, sys
from ai_search_audit.crawler import NativeCrawler
original = NativeCrawler.__init__
def init(self, *args, **kwargs):
    kwargs['transport'] = httpx.MockTransport(lambda request: httpx.Response(
        200, text='<html lang="en"><main><h1>Studio</h1><p>Service.</p></main></html>',
        headers={'content-type': 'text/html'}))
    kwargs['resolver'] = lambda *args: ['93.184.216.34']
    original(self, *args, **kwargs)
NativeCrawler.__init__ = init
def forbidden(*args, **kwargs):
    raise AssertionError('unexpected real network request')
socket.create_connection = forbidden
socket.socket.connect = forbidden
entry = sys.argv.pop(1)
if entry == 'module':
    runpy.run_module('ai_search_audit', run_name='__main__')
else:
    runpy.run_path(entry, run_name='__main__')
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if not installed:
        environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(Path(sys.executable).parent / "ai-search-audit") if installed else "module",
            *(
                arguments
                if arguments is not None
                else [
                    *_command(root, "diagnose", source=source),
                    "--intake-root",
                    str(root.parent.parent / "intake"),
                ]
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=root.parent.parent,
        env=environment,
    )


def _handshake(process):
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    assert line.startswith("OWNED_INTAKE_DIR="), line + process.communicate(timeout=30)[1]
    contract_line = process.stdout.readline().strip()
    assert contract_line.startswith("DIAGNOSTIC_CONTRACT="), contract_line
    assert process.poll() is None
    return Path(line.split("=", 1)[1]), DiagnosticContract.model_validate_json(
        contract_line.split("=", 1)[1]
    )


def test_diagnose_requires_explicit_source_and_owned_intake():
    args = build_parser().parse_args(
        [
            "project",
            "diagnose",
            "project:example",
            "--source-version",
            "public-v1",
            "--clients-root",
            "/tmp/synthetic-clients",
            "--intake-root",
            "/tmp/synthetic-intake",
        ]
    )
    assert args.source_version == "public-v1"
    assert args.project_ref == "project:example"


def test_validation_parser_accepts_explicit_diagnostic_pair():
    args = build_parser().parse_args(
        [
            "project",
            "validate",
            "project:example",
            "--clients-root",
            "/tmp/synthetic-clients",
            "--implementation-date",
            "2026-09-01",
            "--baseline-diagnostic-run",
            "public-v1/run-1",
            "--follow-up-diagnostic-run",
            "public-v1/run-2",
        ]
    )
    assert args.baseline_diagnostic_run == "public-v1/run-1"
    assert args.follow_up_diagnostic_run == "public-v1/run-2"


def test_finalization_parser_accepts_explicit_diagnostic_run():
    args = build_parser().parse_args(
        [
            "project",
            "finalize",
            "project:example",
            "--clients-root",
            "/tmp/synthetic-clients",
            "--version-id",
            "public-v1",
            "--markdown",
            "/tmp/synthetic.md",
            "--pdf",
            "/tmp/synthetic.pdf",
            "--reviewed-pdf-sha256",
            "0" * 64,
            "--no-hero-reason",
            "No image",
            "--diagnostic-run",
            "public-v1/run-1",
        ]
    )
    assert args.diagnostic_run == "public-v1/run-1"


@pytest.mark.parametrize("locale", ["pl", "en"])
def test_benchmark_prepare_installed_is_readonly_exact_localized_json(tmp_path, locale):
    _create(tmp_path, domain="https://studio.example", report_locale=locale)
    root = tmp_path / "clients/example"
    before = _hashes(root)
    expected = prepare_diagnostic_contract(
        "project:example", clients_root=root.parent, source_version="public-v1"
    ).worksheet
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "ai-search-audit"),
            *_command(root, "benchmark-prepare"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected.model_dump(mode="json")
    assert result.stderr == ""
    assert _hashes(root) == before


@pytest.mark.parametrize("installed", [False, True])
def test_diagnose_subprocess_consumes_owned_input_publishes_reloadable_run(project, installed):
    before = _hashes(project)
    process = _process(project, installed=installed)
    owned, contract = _handshake(process)
    (owned / "normalized-intake.json").write_text(
        json.dumps({"expected_binding": contract.source.binding.model_dump(mode="json")})
    )
    stdout, stderr = process.communicate("READY\n", timeout=30)
    assert process.returncode == 0, stderr
    destination = Path(stdout.strip())
    assert destination == project / "diagnostics/public-v1/run-1"
    loaded = DiagnosticStore(project).load("public-v1", "run-1")
    assert loaded.run.module_states.render_parity is DataState.UNAVAILABLE
    assert loaded.run.module_states.benchmark is DataState.UNAVAILABLE
    assert loaded.run.benchmark is None
    assert not owned.exists()
    assert before == {k: v for k, v in _hashes(project).items() if not k.startswith("diagnostics/")}


@pytest.mark.parametrize("failure", ["eof", "invalid-ready", "malformed", "source-mismatch"])
def test_diagnose_subprocess_rejection_cleans_intake_without_publication(project, failure):
    before = _hashes(project)
    process = _process(project)
    owned, contract = _handshake(process)
    binding = contract.source.binding.model_dump(mode="json")
    if failure == "source-mismatch":
        binding["project_id"] = "other"
    (owned / "normalized-intake.json").write_text(
        "{invalid" if failure == "malformed" else json.dumps({"expected_binding": binding})
    )
    stdout, stderr = process.communicate(
        "" if failure == "eof" else "NO\n" if failure == "invalid-ready" else "READY\n", timeout=30
    )
    assert process.returncode != 0
    assert stdout == ""
    assert stderr
    assert not owned.exists()
    assert before == _hashes(project)


def test_diagnose_invalid_source_never_opens_owned_intake(project):
    process = _process(project, source="missing-v99")
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode != 0
    assert "OWNED_INTAKE_DIR=" not in stdout
    assert "source version" in stderr
    assert not (project.parent.parent / "intake").exists()


def test_validate_cli_persists_selected_diagnostic_pair(project):
    from tests.test_diagnostic_workflow import _run

    _run(project)
    _run(project)
    process = _process(
        project,
        arguments=[
            "project",
            "validate",
            "project:example",
            "--clients-root",
            str(project.parent),
            "--implementation-date",
            "2026-09-01",
            "--baseline-diagnostic-run",
            "public-v1/run-1",
            "--follow-up-diagnostic-run",
            "public-v1/run-2",
        ],
    )
    stdout, stderr = process.communicate(timeout=45)
    assert process.returncode == 0, stderr
    bundle = Path(stdout.strip())
    projection = json.loads(
        (bundle / "aggregates/supplementary-diagnostic-comparison.json").read_text()
    )
    assert projection["baseline_reference"] == {"source_version": "public-v1", "run_id": "run-1"}
    assert projection["follow_up_reference"] == {"source_version": "public-v1", "run_id": "run-2"}
    assert projection["comparison"]["baseline_metrics"]["mention_rate"] is None


def test_validate_cli_unpaired_diagnostic_flag_fails_before_intake(project):
    process = _process(
        project,
        arguments=[
            "project",
            "validate",
            "project:example",
            "--clients-root",
            str(project.parent),
            "--implementation-date",
            "2026-09-01",
            "--baseline-diagnostic-run",
            "public-v1/run-1",
            "--intake-root",
            str(project.parent.parent / "intake"),
        ],
    )
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode != 0
    assert "together" in stderr
    assert "OWNED_INTAKE_DIR=" not in stdout
    assert not (project.parent.parent / "intake").exists()


def test_validate_unpaired_flag_cleans_previously_issued_owned_intake(tmp_path):
    owned = create_owned_intake_dir(tmp_path / "intake")
    payload = owned / "normalized-intake.json"
    payload.write_text('{"project_id":"example"}')
    with pytest.raises(ValueError, match="together"):
        main(
            [
                "project",
                "validate",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--implementation-date",
                "2026-09-01",
                "--baseline-diagnostic-run",
                "public-v1/run-1",
                "--intake-dir",
                str(owned),
                "--normalized-intake",
                str(payload),
            ]
        )
    assert not owned.exists()


def test_validate_unpaired_flag_preserves_unissued_directory(tmp_path):
    unowned = tmp_path / "unissued"
    unowned.mkdir()
    payload = unowned / "normalized-intake.json"
    payload.write_text('{"project_id":"example"}')
    with pytest.raises((ValueError, IntakeOwnershipError)):
        main(
            [
                "project",
                "validate",
                "project:example",
                "--clients-root",
                str(tmp_path / "clients"),
                "--implementation-date",
                "2026-09-01",
                "--baseline-diagnostic-run",
                "public-v1/run-1",
                "--intake-dir",
                str(unowned),
                "--normalized-intake",
                str(payload),
            ]
        )
    assert payload.read_text() == '{"project_id":"example"}'
