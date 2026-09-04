"""V3 uses the existing bounded atomic store and leaves old evidence bytes intact."""

import hashlib
import json

import pytest

from ai_search_audit.benchmark import canonical_hash
from ai_search_audit.diagnostic_store import DiagnosticStore
from tests.test_client_delivery import finalize, prepared
from tests.test_diagnostic_observations import api, observation
from tests.test_diagnostic_versions import performance_run
from tests.test_diagnostic_workflow import _contract, _hashes, _load, _run, no_network, project

__all__ = ["no_network", "project", "prepared"]


def test_existing_store_publishes_v3_attempt_jsonl_and_roundtrips(project):
    run = observation(_contract(project).source)
    store = DiagnosticStore(project)
    path = store.publish(run)
    hashes = _hashes(path)
    assert set(hashes) == {"diagnostics.json", "evidence.jsonl", "manifest.json"}
    document = json.loads((path / "diagnostics.json").read_bytes())
    assert not {"attempts", "captures", "collections"} & document.keys()
    evidence = [json.loads(line) for line in (path / "evidence.jsonl").read_bytes().splitlines()]
    assert evidence == [item.model_dump(mode="json") for item in run.attempts]
    loaded = store.load("public-v1", path.name)
    assert loaded.run == run
    assert loaded.manifest.schema_version == "3.0.0"
    assert _hashes(path) == hashes


def test_existing_schema_bytes_remain_unchanged_after_v3_publication(project):
    legacy = _run(project)
    performance = DiagnosticStore(project).publish(performance_run(project))
    hashes = {legacy: _hashes(legacy), performance: _hashes(performance)}
    DiagnosticStore(project).publish(observation(_contract(project).source))
    for path, expected in hashes.items():
        loaded = _load(project, path)
        assert loaded.run.schema_version in {"1.0.0", "2.0.0"}
        assert _hashes(path) == expected


def test_v3_atomic_failure_cleans_only_its_own_staging(project, monkeypatch):
    run = observation(_contract(project).source)
    store = DiagnosticStore(project)
    previous = store.publish(run)
    before = _hashes(previous)
    stale = previous.parent / ".run-existing"
    stale.mkdir()

    def fail(*args):
        raise OSError("synthetic atomic rename failure")

    monkeypatch.setattr("ai_search_audit.diagnostic_store._rename_directory_no_replace", fail)
    with pytest.raises(OSError, match="synthetic"):
        store.publish(run)
    assert set(p.name for p in previous.parent.iterdir()) == {previous.name, stale.name}
    assert _hashes(previous) == before


def rehash_files(path, values, evidence=None):
    (path / "diagnostics.json").write_text(json.dumps(values))
    if evidence is not None:
        (path / "evidence.jsonl").write_text("\n".join(json.dumps(a) for a in evidence) + "\n")
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["input_sha256"] = values["input_sha256"]
    for item in manifest["files"]:
        raw = (path / item["filename"]).read_bytes()
        item.update(byte_count=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (path / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("mutation", ["counts", "source", "inventory", "range", "model", "setup"])
def test_rehashed_forgery_rejects_not_just_file_digest_mismatch(project, mutation):
    store = DiagnosticStore(project)
    path = store.publish(observation(_contract(project).source))
    values = json.loads((path / "diagnostics.json").read_bytes())
    evidence = [json.loads(line) for line in (path / "evidence.jsonl").read_bytes().splitlines()]
    if mutation == "counts":
        values["sample"]["metrics"]["mention_rate"] = 0
    elif mutation == "source":
        for worksheet in (values["worksheet"], values["sample"]["worksheet"]):
            worksheet["prompts"][0]["text"] = "forged canonical prompt"
            worksheet["pack_content_hash"] = canonical_hash(
                {"pack_version": worksheet["pack_version"], "prompts": worksheet["prompts"]}
            )
        values["sample"]["responses"][0]["prompt_text"] = "forged canonical prompt"
    elif mutation == "inventory":
        evidence[0]["prompt_id"] = "foreign"
    elif mutation == "range":
        values["collection_range"]["end"] = "2026-09-05T00:00:00Z"
    elif mutation == "model":
        evidence[0]["returned_model"] = "other"
    else:
        values["setup"]["benchmark_setup"]["market"] = "other"
    values["input_sha256"] = canonical_hash(
        {
            "worksheet": values["worksheet"],
            "setup": values["setup"],
            "selected_prompt_ids": values["selected_prompt_ids"],
            "sample": values["sample"],
            "attempts": evidence,
            "price_provenance": values["price_provenance"],
        }
    )
    rehash_files(path, values, evidence)
    with pytest.raises(ValueError):
        store.load("public-v1", path.name)


@pytest.mark.parametrize("mutation", ["nan", "missing", "unknown", "binary"])
def test_unchecked_nested_attempts_reject_before_any_publication(project, mutation):
    run = observation(_contract(project).source)
    attempt = run.attempts[0]
    if mutation == "nan":
        attempt = attempt.model_copy(
            update={"usage": attempt.usage.model_copy(update={"total_tokens": float("nan")})}
        )
    elif mutation == "missing":
        values = attempt.model_dump(mode="python")
        del values["returned_model"]
        attempt = api().ObservationAttempt.model_construct(**values)
    elif mutation == "unknown":
        attempt = attempt.model_copy(update={"account_identity": "forbidden"})
    else:
        attempt = attempt.model_copy(update={"request_id": b"req_binary"})
    forged = run.model_copy(update={"attempts": (attempt,)})
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(forged)
    assert not (project / "diagnostics").exists()


def test_zero_attempt_stop_is_unavailable_not_zero_result(project):
    run = observation(_contract(project).source, status="failed")
    run = api().assemble_observation_run(
        _contract(project).source,
        run.worksheet,
        run.setup,
        run.selected_prompt_ids,
        run.sample,
        (),
        price_provenance=None,
    )
    path = DiagnosticStore(project).publish(run)
    assert (path / "evidence.jsonl").read_bytes() == b""
    assert _load(project, path).run.sample.metrics.mention_rate is None


def test_client_finalizer_rejects_v3_without_guarded_projection(prepared):
    root, _, args = prepared
    old = finalize(args)
    hashes = _hashes(old)
    path = DiagnosticStore(root).publish(observation(_contract(root).source))
    with pytest.raises(ValueError, match="observation.*projection"):
        finalize(dict(args, diagnostic_run_ref=f"public-v1/{path.name}"))
    assert _hashes(old) == hashes
    assert not (old.parent / "edition-2").exists()
