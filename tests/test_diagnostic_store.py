from __future__ import annotations

import hashlib
import json
import shutil
from importlib import import_module

import pytest
from pydantic import ValidationError

from tests.test_diagnostic_workflow import (
    _contract,
    _hashes,
    _intake,
    _load,
    _rendered,
    _run,
    _tampered_finding_payload,
    no_network,
    project,
)

__all__ = ["project", "no_network"]


def test_symlink_hidden_by_parent_traversal_rejects_before_normalization(project, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(project.parent, target_is_directory=True)
    store = import_module("ai_search_audit.diagnostic_store")
    with pytest.raises(ValueError, match="directory|traversal"):
        store.DiagnosticStore(alias / ".." / "clients" / "example")


def test_sequential_runs_have_exact_inventory_and_do_not_overwrite(project):
    first = _run(project)
    before = _hashes(first)
    second = _run(project)
    assert first.name == "run-1" and second.name == "run-2"
    assert _hashes(first) == before
    assert {p.name for p in first.iterdir()} == {
        "diagnostics.json",
        "evidence.jsonl",
        "manifest.json",
    }
    loaded = _load(project, first)
    assert loaded.manifest.run_number == 1
    assert (
        loaded.manifest_sha256 == hashlib.sha256((first / "manifest.json").read_bytes()).hexdigest()
    )
    assert loaded.run.binding == _contract(project).source.binding


def test_nested_run_and_manifest_are_immutable(project):
    loaded = _load(project, _run(project))
    with pytest.raises(ValidationError):
        loaded.run.binding.report_locale = "pl"
    with pytest.raises(ValidationError):
        loaded.run.captures[0].state = "FAILED"
    with pytest.raises(ValidationError):
        loaded.manifest.files[0].sha256 = "0" * 64
    assert isinstance(loaded.run.captures, tuple)
    assert isinstance(loaded.manifest.files, tuple)


def test_evidence_tamper_and_extra_files_reject(project):
    path = _run(project)
    ledger = path / "evidence.jsonl"
    original = ledger.read_bytes()
    ledger.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash|size"):
        _load(project, path)
    ledger.write_bytes(original)
    (path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        _load(project, path)


def test_unknown_manifest_schema_rejects(project):
    path = _run(project)
    file = path / "manifest.json"
    manifest = json.loads(file.read_bytes())
    manifest["schema_version"] = "99.0.0"
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _load(project, path)


def test_collision_does_not_overwrite_and_staging_is_removed(project, monkeypatch):
    path = _run(project)
    before = _hashes(path)
    store = import_module("ai_search_audit.diagnostic_store")

    def collision(source, destination):
        destination.mkdir()
        (destination / "sentinel").write_text("other writer")
        raise FileExistsError("collision")

    monkeypatch.setattr(store, "_rename_directory_no_replace", collision)
    with pytest.raises(FileExistsError):
        _run(project)
    assert _hashes(path) == before
    assert (path.parent / "run-2/sentinel").read_text() == "other writer"
    assert not list(path.parent.glob(".run-*"))


def test_interrupted_staging_is_not_loaded_as_a_run(project):
    first = _run(project)
    staging = first.parent / ".run-interrupted"
    staging.mkdir()
    (staging / "diagnostics.json").write_text("partial")
    assert _run(project).name == "run-2"
    assert staging.exists()
    store = import_module("ai_search_audit.diagnostic_store").DiagnosticStore(project)
    with pytest.raises(ValueError):
        store.load("public-v1", ".run-interrupted")


def test_source_version_mismatch_and_model_copy_rejected(project):
    run = _load(project, _run(project)).run
    store = import_module("ai_search_audit.diagnostic_store").DiagnosticStore(project)
    wrong = run.model_copy(
        update={"binding": run.binding.model_copy(update={"report_locale": "pl"})}
    )
    with pytest.raises(ValueError, match="binding"):
        store.publish(wrong)
    with pytest.raises(ValueError):
        store.load("public-v2", "run-1")


def test_symlink_ancestor_rejected_before_resolution(project, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(project.parent, target_is_directory=True)
    workflow = import_module("ai_search_audit.diagnostic_workflow")
    with pytest.raises(ValueError, match="directory"):
        workflow.prepare_diagnostic_contract(
            "project:example", clients_root=alias, source_version="public-v1"
        )
    store = import_module("ai_search_audit.diagnostic_store")
    with pytest.raises(ValueError, match="directory"):
        store.DiagnosticStore(alias / "example")


def test_symlink_evidence_rejects(project, tmp_path):
    path = _run(project)
    ledger = path / "evidence.jsonl"
    outside = tmp_path / "saved-evidence"
    ledger.rename(outside)
    ledger.symlink_to(outside)
    with pytest.raises(ValueError, match="regular"):
        _load(project, path)


def test_cross_project_baseline_cannot_escape_reference_schema(project):
    owned = _intake(project, baseline_run={"source_version": "../../other", "run_id": "run-1"})
    with pytest.raises(Exception, match="schema"):
        _run(project, owned=owned)
    assert not owned.exists()


def test_copied_cross_project_baseline_is_rejected_by_canonical_binding(project, tmp_path):
    from tests.test_project_orchestrator import _create

    original = _run(project)
    _create(
        tmp_path, project_id="other", domain="https://studio.example", client_name="Other Studio"
    )
    other = project.parent / "other"
    copied = other / "diagnostics/public-v1/run-1"
    shutil.copytree(original, copied)
    before = _hashes(other)
    owned = _intake(
        other,
        worksheet=_contract(other).worksheet.model_dump(mode="json"),
        baseline_run={"source_version": "public-v1", "run_id": "run-1"},
    )
    with pytest.raises(ValueError, match="binding"):
        _run(other, owned=owned)
    assert not owned.exists()
    assert _hashes(other) == before


@pytest.mark.parametrize(
    "relative", ["diagnostics", "diagnostics/public-v1", "diagnostics/public-v1/run-1"]
)
def test_all_diagnostic_directory_components_reject_symlinks(project, tmp_path, relative):
    path = _run(project)
    target = project / relative
    moved = tmp_path / "saved-directory"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        _load(project, path)


def test_interrupted_promotion_cleans_staging_without_promoting_partial_run(project, monkeypatch):
    store = import_module("ai_search_audit.diagnostic_store")
    before = _hashes(project)

    def interrupted(source, destination):
        raise InterruptedError("synthetic interruption before rename")

    monkeypatch.setattr(store, "_rename_directory_no_replace", interrupted)
    owned = _intake(project)
    with pytest.raises(InterruptedError):
        _run(project, owned=owned)
    assert not owned.exists()
    assert _hashes(project) == before
    assert not list(project.glob("diagnostics/public-v1/*"))


def test_rehashed_unknown_diagnostics_schema_is_rejected(project):
    path = _run(project)
    diagnostics_file = path / "diagnostics.json"
    diagnostics = json.loads(diagnostics_file.read_bytes())
    diagnostics["schema_version"] = "2.0.0"
    changed = json.dumps(diagnostics).encode()
    diagnostics_file.write_bytes(changed)
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    manifest["files"][0].update(byte_count=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        _load(project, path)


def test_stored_evidence_size_bound_is_enforced_before_json_parse(project):
    path = _run(project)
    (path / "evidence.jsonl").write_bytes(b" " * (8 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="size"):
        _load(project, path)


def test_canonical_project_manifest_symlink_is_not_followed(project, tmp_path):
    path = _run(project)
    manifest = project / "project.json"
    moved = project / "original-project.json"
    manifest.rename(moved)
    manifest.symlink_to(moved)
    with pytest.raises(ValueError, match="regular"):
        _contract(project)
    with pytest.raises(ValueError, match="regular"):
        _load(project, path)


def _benchmark_pair(project):
    worksheet = _contract(project).worksheet.model_dump(mode="json")
    first = _run(project, owned=_intake(project, worksheet=worksheet))
    return _run(
        project,
        owned=_intake(
            project,
            worksheet=worksheet,
            baseline_run={"source_version": "public-v1", "run_id": first.name},
        ),
    )


def test_recursive_load_validates_each_canonical_source_once_per_operation(project, monkeypatch):
    path = _benchmark_pair(project)
    module = import_module("ai_search_audit.diagnostic_store")
    original = module.load_diagnostic_source
    versions = []

    def counted(*args, **kwargs):
        versions.append(kwargs["source_version"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "load_diagnostic_source", counted)
    loaded = module.DiagnosticStore(project).load("public-v1", path.name)
    assert loaded.run.baseline is not None
    assert versions == ["public-v1"]


def test_source_cache_is_discarded_before_next_load(project):
    path = _benchmark_pair(project)
    store = import_module("ai_search_audit.diagnostic_store").DiagnosticStore(project)
    store.load("public-v1", path.name)
    canonical = project / "project.json"
    manifest = json.loads(canonical.read_bytes())
    manifest["report_locale"] = "pl"
    canonical.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="locale"):
        store.load("public-v1", path.name)


def _replace_diagnostics(path, payload):
    values = {key: value for key, value in payload.items() if key != "captures"}
    data = json.dumps(values).encode()
    (path / "diagnostics.json").write_bytes(data)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["algorithm_sha256"] = payload["algorithm_sha256"]
    manifest["files"][0].update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation", ["rule_id", "rule_statement", "expired_resolution", "status"])
@pytest.mark.parametrize("boundary", ["publish", "load"])
def test_public_store_rejects_self_hashed_forged_rules(project, mutation, boundary):
    from ai_search_audit.diagnostic_models import DiagnosticFinding

    path = _run(project, owned=_intake(project, rendered_captures=[_rendered(project)]))
    run = _load(project, path).run
    before = {
        key: value for key, value in _hashes(project).items() if not key.startswith("diagnostics/")
    }
    payload = _tampered_finding_payload(run, mutation)
    store = import_module("ai_search_audit.diagnostic_store").DiagnosticStore(project)
    if boundary == "publish":
        forged = run.model_copy(
            update={
                "findings": tuple(
                    DiagnosticFinding.model_validate(item) for item in payload["findings"]
                ),
                "algorithm_sha256": payload["algorithm_sha256"],
            }
        )
        with pytest.raises(ValueError, match="rule|status"):
            store.publish(forged)
        assert not (path.parent / "run-2").exists()
    else:
        _replace_diagnostics(path, payload)
        with pytest.raises(ValueError, match="rule|status"):
            store.load("public-v1", path.name)
    assert {
        key: value for key, value in _hashes(project).items() if not key.startswith("diagnostics/")
    } == before


@pytest.mark.parametrize("mutation", ["invented_predicate", "drop_finding", "duplicate_finding"])
@pytest.mark.parametrize("boundary", ["publish", "load"])
def test_public_store_rejects_finding_projection_and_inventory_tampering(
    project, mutation, boundary
):
    from ai_search_audit.diagnostic_models import DiagnosticFinding

    capture = _rendered(
        project, html="<main><h1>Studio</h1><p>Delivery.</p><h2>Service</h2><p>Details.</p></main>"
    )
    path = _run(project, owned=_intake(project, rendered_captures=[capture]))
    run = _load(project, path).run
    assert len(run.findings) == 2 and not any(f.assessments for f in run.findings)
    before = {
        key: value for key, value in _hashes(project).items() if not key.startswith("diagnostics/")
    }
    payload = _tampered_finding_payload(run, mutation)
    store = import_module("ai_search_audit.diagnostic_store").DiagnosticStore(project)
    if boundary == "publish":
        forged = run.model_copy(
            update={
                "findings": tuple(
                    DiagnosticFinding.model_validate(item) for item in payload["findings"]
                ),
                "algorithm_sha256": payload["algorithm_sha256"],
            }
        )
        with pytest.raises(ValueError, match="finding"):
            store.publish(forged)
        assert not (path.parent / "run-2").exists()
    else:
        _replace_diagnostics(path, payload)
        with pytest.raises(ValueError, match="finding"):
            store.load("public-v1", path.name)
    assert {
        key: value for key, value in _hashes(project).items() if not key.startswith("diagnostics/")
    } == before
