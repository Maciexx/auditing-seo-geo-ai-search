"""Versioned persistence uses synthetic source-bound provider observations."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest
from pydantic import ValidationError

from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.measurement_profile import MeasurementProfile, prepare_measurement_preflight
from ai_search_audit.performance_providers import PerformanceCollection
from tests.test_client_delivery import prepared
from tests.test_diagnostic_workflow import _contract, _hashes, _load, _run, no_network, project

__all__ = ["no_network", "project", "prepared"]


def performance_run(project):
    source = _contract(project).source
    preflight = prepare_measurement_preflight(source, MeasurementProfile())
    collections = []
    stamp = datetime(2026, 9, 4, tzinfo=UTC)
    for page in preflight.selected_pages:
        for device in preflight.devices:
            for provider in ("pagespeed_insights", "crux"):
                index = len(collections)
                collections.append(
                    PerformanceCollection.model_validate(
                        {
                            "attempts": [
                                {
                                    "attempt_id": f"attempt-{index}",
                                    "provider": provider,
                                    "requested_url": page.url,
                                    "device": device,
                                    "started_at": stamp + timedelta(seconds=index * 2),
                                    "ended_at": stamp + timedelta(seconds=index * 2 + 1),
                                    "state": "UNAVAILABLE",
                                    "reason": "missing_key",
                                    "http_status": None,
                                }
                            ],
                        }
                    )
                )
    module = import_module("ai_search_audit.diagnostic_workflow")
    return module.assemble_performance_run(source, preflight, tuple(collections))


def transient_run(project, *, version, include_local=True, early=False, eligible=True):
    source = _contract(project).source
    preflight = prepare_measurement_preflight(
        source, MeasurementProfile(schema_version=version, lighthouse_local=True)
    )
    collections = []
    for c in performance_run(project).collections:
        if c.attempts[0].provider != "pagespeed_insights":
            collections.append(c)
            continue
        payload = c.model_dump(mode="python")
        first = payload["attempts"][0]
        first.update(
            state="FAILED", reason="http_500" if eligible else "malformed_response", http_status=500
        )
        collections.append(PerformanceCollection.model_validate(payload))
        if include_local:
            local = dict(
                first,
                provider="lighthouse_local",
                state="UNAVAILABLE",
                reason="runtime_missing",
                http_status=None,
                attempt_id="local-" + first["attempt_id"],
                started_at=first["ended_at"] + timedelta(seconds=-2 if early else 1),
                ended_at=first["ended_at"] + timedelta(seconds=2),
            )
            collections.append(PerformanceCollection.model_validate({"attempts": [local]}))
    return import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        source, preflight, tuple(collections)
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
def test_performance_policy_derives_pipeline_and_fingerprint(project, version):
    from ai_search_audit.benchmark import canonical_hash

    run = transient_run(project, version=version, include_local=version == "1.1.0")
    assert run.schema_version == "2.0.0"
    assert run.pipeline_version == f"performance-{version}"
    assert run.algorithm_sha256 == canonical_hash(
        {"pipeline": f"performance-{version}", "normalization": "1.0.0"}
    )
    path = DiagnosticStore(project).publish(run)
    original = _hashes(path)
    assert _load(project, path).run == run
    assert _hashes(path) == original


@pytest.mark.parametrize("mutation", ["pipeline", "algorithm", "policy"])
def test_current_profile_rejects_mismatched_version_contract(project, mutation):
    from ai_search_audit.diagnostic_performance import DiagnosticRunV2, performance_algorithm_hash

    run = transient_run(project, version="1.1.0")
    payload = run.model_dump(mode="python")
    if mutation == "pipeline":
        payload["pipeline_version"] = "performance-1.0.0"
    elif mutation == "algorithm":
        payload["algorithm_sha256"] = performance_algorithm_hash()
    else:
        payload["preflight"]["profile"]["schema_version"] = "1.2.0"
    with pytest.raises(ValueError):
        DiagnosticRunV2.model_validate(payload)


@pytest.mark.parametrize(
    "version,include_local,early,eligible,message",
    [
        ("1.1.0", False, False, True, "inventory"),
        ("1.1.0", True, True, True, "chronology"),
        ("1.1.0", True, False, False, "inventory"),
        ("1.0.0", True, False, True, "inventory"),
    ],
)
def test_transient_fallback_inventory_and_chronology_are_enforced(
    project, version, include_local, early, eligible, message
):
    with pytest.raises(ValueError, match=message):
        transient_run(
            project, version=version, include_local=include_local, early=early, eligible=eligible
        )


def test_legacy_transient_artifact_bytes_remain_loadable_after_current_publication(project):
    from ai_search_audit.diagnostic_performance import performance_algorithm_hash

    legacy = transient_run(project, version="1.0.0", include_local=False)
    assert legacy.algorithm_sha256 == performance_algorithm_hash()
    store = DiagnosticStore(project)
    path = store.publish(legacy)
    original = _hashes(path)
    store.publish(transient_run(project, version="1.1.0"))
    assert _load(project, path).run == legacy
    assert _hashes(path) == original


@pytest.mark.parametrize("mutation", ["missing_local", "forged_local", "early_local"])
def test_persisted_current_fallback_rejects_tampering_even_with_rehashed_files(project, mutation):
    from ai_search_audit.diagnostic_performance import performance_collection_range

    run = transient_run(project, version="1.1.0")
    store = DiagnosticStore(project)
    path = store.publish(run)
    collections = [c.model_dump(mode="python") for c in run.collections]
    if mutation == "missing_local":
        collections = [c for c in collections if c["attempts"][0]["provider"] != "lighthouse_local"]
    elif mutation == "forged_local":
        for c in collections:
            if c["attempts"][0]["provider"] == "pagespeed_insights":
                c["attempts"][0]["reason"] = "malformed_response"
    else:
        for c in collections:
            if c["attempts"][0]["provider"] == "lighthouse_local":
                c["attempts"][0]["started_at"] -= timedelta(seconds=3)
    changed = tuple(PerformanceCollection.model_validate(c) for c in collections)
    payload = json.loads((path / "diagnostics.json").read_bytes())
    payload["collection_range"] = performance_collection_range(changed).model_dump(mode="json")
    rewrite(path, "diagnostics.json", payload)
    data = b"".join(
        (c.model_dump_json() + "\n").encode()
        for c in sorted(changed, key=lambda c: c.attempts[0].attempt_id)
    )
    (path / "evidence.jsonl").write_bytes(data)
    manifest = json.loads((path / "manifest.json").read_bytes())
    manifest["files"][1].update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
    rewrite(path, "manifest.json", manifest)
    with pytest.raises(
        ValueError, match="chronology" if mutation == "early_local" else "inventory"
    ):
        store.load("public-v1", path.name)


def test_original_v1_bytes_survive_v2_publication_and_loading(project):
    old = _run(project)
    original = _hashes(old)
    store = DiagnosticStore(project)
    new = store.publish(performance_run(project))
    assert new.name == "run-2"
    assert _load(project, old).run.schema_version == "1.0.0"
    loaded = _load(project, new)
    assert loaded.run.schema_version == loaded.manifest.schema_version == "2.0.0"
    assert _hashes(old) == original
    assert set(_hashes(new)) == {"diagnostics.json", "evidence.jsonl", "manifest.json"}
    assert "collections" not in json.loads((new / "diagnostics.json").read_bytes())
    assert loaded.run == performance_run(project)


def test_provider_run_records_actual_times_and_no_fictitious_intake_deletion(project):
    run = performance_run(project)
    assert run.collection_range.start == run.collections[0].attempts[0].started_at
    assert run.collection_range.end == run.collections[-1].attempts[-1].ended_at
    assert run.cleanup.status == "not_required"
    assert run.cleanup.method == "no_owned_intake"
    assert run.cleanup.completed_at is None
    with pytest.raises(ValidationError):
        run.preflight.profile.max_pages = 5
    with pytest.raises(ValidationError):
        run.collections[0].attempts[0].reason = "changed"


def rewrite(path, name, payload):
    data = (json.dumps(payload) + "\n").encode()
    (path / name).write_bytes(data)
    if name != "manifest.json":
        manifest = json.loads((path / "manifest.json").read_bytes())
        for item in manifest["files"]:
            if item["filename"] == name:
                item.update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
        (path / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("version", ["0.0.0", "3.0.0", None])
def test_unknown_v2_schemas_reject(project, version):
    path = DiagnosticStore(project).publish(performance_run(project))
    payload = json.loads((path / "manifest.json").read_bytes())
    payload["schema_version"] = version
    rewrite(path, "manifest.json", payload)
    with pytest.raises(ValueError):
        _load(project, path)


def corrupt(payload, mutation):
    first = payload["collections"][0]
    attempt = first["attempts"][0]
    if mutation == "success_without_measurement":
        attempt.update(state="AVAILABLE", reason=None, http_status=200)
    elif mutation == "duplicate_attempt":
        payload["collections"][1]["attempts"][0]["attempt_id"] = attempt["attempt_id"]
    elif mutation == "wrong_url":
        attempt["requested_url"] = "https://other.example/"
    elif mutation == "private_url":
        attempt["requested_url"] = "http://127.0.0.1/"
    elif mutation == "incomplete_inventory":
        payload["collections"].pop()
    elif mutation == "duplicate_collection":
        payload["collections"].append(first)
    elif mutation == "wrong_device":
        attempt["device"] = "desktop"
    elif mutation == "forged_preflight":
        payload["preflight"]["selected_pages"][0]["selection_reason"] = "Invented"
    elif mutation == "wrong_binding":
        payload["binding"]["source_sha256"] = "0" * 64
    elif mutation == "wrong_hash":
        payload["input_sha256"] = "0" * 64
    elif mutation == "wrong_algorithm":
        payload["algorithm_sha256"] = "0" * 64
    elif mutation == "wrong_range":
        payload["collection_range"]["end"] = "2025-01-01T00:00:00Z"
    elif mutation == "raw_payload":
        first["raw_response"] = "Do not persist"
    elif mutation == "credential_field":
        payload["preflight"]["profile"]["api_key"] = "synthetic-private-value"
    elif mutation == "unsafe_reason":
        attempt["reason"] = "Traceback: secret=value"
    elif mutation == "fake_cleanup":
        payload["cleanup"]["status"] = "deleted"
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "success_without_measurement",
        "duplicate_attempt",
        "wrong_url",
        "private_url",
        "incomplete_inventory",
        "duplicate_collection",
        "wrong_device",
        "forged_preflight",
        "wrong_binding",
        "wrong_hash",
        "wrong_algorithm",
        "wrong_range",
        "raw_payload",
        "credential_field",
        "unsafe_reason",
        "fake_cleanup",
    ],
)
@pytest.mark.parametrize("boundary", ["publish", "load"])
def test_v2_rejects_invalid_persisted_contract(project, mutation, boundary):
    run = performance_run(project)
    store = DiagnosticStore(project)
    payload = run.model_dump(mode="json")
    corrupt(payload, mutation)
    if boundary == "publish":
        with pytest.raises(ValueError):
            store.publish(run.model_copy(update=payload))
        assert not (project / "diagnostics").exists()
    else:
        path = store.publish(run)
        collections = payload.pop("collections")
        rewrite(path, "diagnostics.json", payload)
        data = b"".join(
            (json.dumps(item) + "\n").encode()
            for item in sorted(collections, key=lambda item: item["attempts"][0]["attempt_id"])
        )
        (path / "evidence.jsonl").write_bytes(data)
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["files"][1].update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
        rewrite(path, "manifest.json", manifest)
        with pytest.raises(ValueError):
            store.load("public-v1", path.name)


def test_manifest_and_run_must_agree_on_schema(project):
    path = DiagnosticStore(project).publish(performance_run(project))
    payload = json.loads((path / "diagnostics.json").read_bytes())
    payload["schema_version"] = "1.0.0"
    rewrite(path, "diagnostics.json", payload)
    with pytest.raises(ValueError, match="schema"):
        _load(project, path)


def test_unchecked_unknown_model_fields_are_not_silently_dropped(project):
    run = performance_run(project)
    fabricated_credential = "synthetic-secret"
    run = run.model_copy(update={"api_key": fabricated_credential})
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(run)


def measured_run(project):
    from tests.test_performance_models import configuration_fields

    run = performance_run(project)
    payload = run.collections[0].model_dump(mode="json")
    attempt = payload["attempts"][0]
    attempt.update(state="AVAILABLE", reason=None, http_status=200)
    payload["lab"] = {
        "binding": run.binding.model_dump(mode="json"),
        "evidence_id": "measurement-1",
        "attempt_id": attempt["attempt_id"],
        "provider": "pagespeed_insights",
        "requested_url": attempt["requested_url"],
        "final_url": attempt["requested_url"],
        "device": "mobile",
        "observed_at": "2026-09-03T12:00:00Z",
        "lighthouse_version": "13.0.0",
        "chrome_version": None,
        "configuration": configuration_fields(),
        "performance_score": 75.0,
        "metrics": [
            {"name": name, "value": 0.0, "unit": "unitless" if name == "cls" else "ms"}
            for name in ("lcp", "cls", "fcp", "tbt", "speed_index")
        ],
    }
    collection = PerformanceCollection.model_validate(payload)
    return import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        _contract(project).source, run.preflight, (collection, *run.collections[1:])
    )


def test_lab_measurement_roundtrips_with_actual_provider_timestamp(project):
    run = measured_run(project)
    path = DiagnosticStore(project).publish(run)
    loaded = _load(project, path).run
    assert loaded == run
    assert loaded.collection_range.start == datetime(2026, 9, 3, 12, tzinfo=UTC)
    assert loaded.collections[0].lab.metrics[0].value == 0.0


@pytest.mark.parametrize(
    "mutation", ["binding", "final_url", "orphan", "duplicate_id", "coverage", "nan"]
)
def test_lab_evidence_rejects_forgery_at_publish(project, mutation):
    run = measured_run(project)
    payload = run.model_dump(mode="json")
    lab = payload["collections"][0]["lab"]
    if mutation == "binding":
        lab["binding"]["source_sha256"] = "0" * 64
    elif mutation == "final_url":
        lab["final_url"] = "https://other.example/"
    elif mutation == "orphan":
        lab["attempt_id"] = "orphan"
    elif mutation == "duplicate_id":
        lab["evidence_id"] = payload["collections"][1]["attempts"][0]["attempt_id"]
    elif mutation == "coverage":
        lab["metrics"][0]["value"] = None
    else:
        lab["metrics"][0]["value"] = float("nan")
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(run.model_copy(update=payload))


@pytest.mark.parametrize(
    "mutation",
    ["over_budget", "foreign_history", "device_history", "reversed_history", "unjustified_retry"],
)
def test_attempt_history_obeys_frozen_profile_and_identity(project, mutation):
    run = performance_run(project)
    payload = run.model_dump(mode="json")
    collection = payload["collections"][0]
    first = collection["attempts"][0]
    retry = dict(first, attempt_id="retry")
    retry.update(started_at="2026-09-04T00:00:02Z", ended_at="2026-09-04T00:00:03Z")
    first.update(state="FAILED", reason="timeout")
    if mutation == "over_budget":
        payload["preflight"] = prepare_measurement_preflight(
            _contract(project).source, MeasurementProfile(retry_transient=False)
        ).model_dump(mode="json")
    elif mutation == "foreign_history":
        retry["requested_url"] = "https://other.example/"
    elif mutation == "device_history":
        retry["device"] = "desktop"
    elif mutation == "reversed_history":
        retry.update(started_at="2026-01-01T00:00:00Z", ended_at="2026-01-01T00:00:01Z")
    elif mutation == "unjustified_retry":
        first.update(state="UNAVAILABLE", reason="missing_key")
    collection["attempts"].append(retry)
    from ai_search_audit.benchmark import canonical_hash
    from ai_search_audit.diagnostic_performance import performance_collection_range

    payload["input_sha256"] = canonical_hash(payload["preflight"])
    payload["collection_range"] = performance_collection_range(
        tuple(PerformanceCollection.model_validate(c) for c in payload["collections"])
    ).model_dump(mode="json")
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(run.model_copy(update=payload))


def test_v2_fits_existing_explicit_finalization(prepared):
    import subprocess
    import sys
    from pathlib import Path

    from pypdf import PdfReader

    from ai_search_audit.measurement_report import (
        load_measurement_report,
        render_measurement_fragment,
    )
    from tests.test_client_delivery import digest, finalize

    root, _, args = prepared
    path = DiagnosticStore(root).publish(performance_run(root))
    projection = load_measurement_report(
        "project:example", clients_root=root.parent, diagnostic_run_ref=f"public-v1/{path.name}"
    )
    options = json.loads(PdfReader(args["pdf_path"]).metadata["/ClientEditionRenderOptions"])
    args["markdown_path"].write_text(
        args["markdown_path"].read_text() + "\n" + render_measurement_fragment(projection)
    )
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/render_client_pdf.py"),
        str(args["markdown_path"]),
        str(args["pdf_path"]),
    ]
    for key, value in options.items():
        command.extend([f"--{key}", value])
    command.extend(["--hero", str(args["hero_path"])])
    subprocess.run(command, capture_output=True, check=True, timeout=30)
    args["reviewed_pdf_sha256"] = digest(args["pdf_path"])
    destination = finalize({**args, "diagnostic_run_ref": f"public-v1/{path.name}"})
    record = json.loads((destination / "delivery.json").read_bytes())
    assert record["diagnostic_run"]["path"] == "diagnostics/public-v1/run-1"


@pytest.mark.parametrize("failure", ["schema", "hash", "measurement"])
def test_v2_finalization_rejects_invalid_saved_run(prepared, failure):
    from tests.test_client_delivery import finalize

    root, _, args = prepared
    path = DiagnosticStore(root).publish(performance_run(root))
    if failure == "schema":
        payload = json.loads((path / "manifest.json").read_bytes())
        payload["schema_version"] = "99.0.0"
        rewrite(path, "manifest.json", payload)
    elif failure == "hash":
        with (path / "evidence.jsonl").open("ab") as stream:
            stream.write(b" ")
    else:
        payload = json.loads((path / "diagnostics.json").read_bytes())
        payload["preflight"]["devices"] = ["mobile", "mobile"]
        rewrite(path, "diagnostics.json", payload)
    with pytest.raises(ValueError):
        finalize({**args, "diagnostic_run_ref": f"public-v1/{path.name}"})
    assert not (root / "reports/public-v1/edition-1").exists()


@pytest.mark.parametrize("version", [[], {}])
def test_malformed_schema_value_is_a_validation_error(project, version):
    run = performance_run(project)
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(run.model_copy(update={"schema_version": version}))


def test_unchecked_bytes_cannot_be_coerced_into_published_urls(project):
    run = performance_run(project)
    payload = run.model_dump(mode="python")
    attempt = payload["collections"][0]["attempts"][0]
    attempt["requested_url"] = attempt["requested_url"].encode()
    with pytest.raises(ValueError):
        DiagnosticStore(project).publish(run.model_copy(update=payload))


def test_assembly_canonicalizes_collection_order(project):
    run = performance_run(project)
    reordered = import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        _contract(project).source, run.preflight, tuple(reversed(run.collections))
    )
    assert reordered == run


def test_actual_crux_origin_fallback_is_persisted_as_field_not_page_lab(project):
    run = performance_run(project)
    payload = run.collections[1].model_dump(mode="json")
    first = payload["attempts"][0]
    first.update(reason="no_record", http_status=404)
    origin = first["requested_url"].rstrip("/")
    payload["attempts"].append(
        dict(
            first,
            attempt_id="origin-attempt",
            requested_url=origin,
            started_at="2026-09-04T00:00:04Z",
            ended_at="2026-09-04T00:00:05Z",
            state="AVAILABLE",
            reason=None,
            http_status=200,
        )
    )
    payload["field"] = {
        "binding": run.binding.model_dump(mode="json"),
        "evidence_id": "origin-evidence",
        "attempt_id": "origin-attempt",
        "requested_url": first["requested_url"],
        "record_key": origin,
        "scope": "origin",
        "device": "mobile",
        "observed_at": "2026-09-04T00:00:05Z",
        "period": {"first_date": "2026-08-01", "last_date": "2026-08-28"},
        "metrics": [
            {"name": name, "value": 0.0, "unit": "unitless" if name == "cls" else "ms"}
            for name in ("lcp", "inp", "cls")
        ],
    }
    collection = PerformanceCollection.model_validate(payload)
    run = import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        _contract(project).source,
        run.preflight,
        (run.collections[0], collection, *run.collections[2:]),
    )
    loaded = _load(project, DiagnosticStore(project).publish(run)).run
    assert loaded.collections[1].field.scope == "origin"
    assert loaded.collections[1].field.requested_url == first["requested_url"]
    assert loaded.collections[1].lab is None


def test_every_published_file_obeys_store_bound(project, monkeypatch):
    store = import_module("ai_search_audit.diagnostic_store")
    run = performance_run(project)
    monkeypatch.setattr(store, "_MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        DiagnosticStore(project).publish(run)
    assert not list(project.glob("diagnostics/public-v1/run-*"))


def test_enabled_local_fallback_cannot_disappear_from_inventory(project):
    run = performance_run(project)
    preflight = prepare_measurement_preflight(
        _contract(project).source, MeasurementProfile(lighthouse_local=True)
    )
    with pytest.raises(ValueError, match="inventory"):
        import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
            _contract(project).source, preflight, run.collections
        )


@pytest.mark.parametrize("early", [False, True])
def test_local_runtime_unavailability_survives_publication(project, early):
    run = performance_run(project)
    preflight = prepare_measurement_preflight(
        _contract(project).source, MeasurementProfile(lighthouse_local=True)
    )
    fallback = tuple(
        PerformanceCollection.model_validate(
            {
                "attempts": [
                    dict(
                        c.attempts[0].model_dump(mode="python"),
                        provider="lighthouse_local",
                        attempt_id="local-" + c.attempts[0].attempt_id,
                        reason="runtime_missing",
                        started_at=c.attempts[0].ended_at + timedelta(seconds=-2 if early else 1),
                        ended_at=c.attempts[0].ended_at + timedelta(seconds=2),
                    )
                ]
            }
        )
        for c in run.collections
        if c.attempts[0].provider == "pagespeed_insights"
    )
    if early:
        with pytest.raises(ValueError, match="chronology"):
            import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
                _contract(project).source, preflight, (*run.collections, *fallback)
            )
        return
    run = import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        _contract(project).source, preflight, (*run.collections, *fallback)
    )
    assert _load(project, DiagnosticStore(project).publish(run)).run == run


@pytest.mark.parametrize("reason", ["no_record", "no_data"])
def test_crux_url_no_data_requires_retained_origin_attempt(project, reason):
    run = performance_run(project)
    payload = run.collections[1].model_dump(mode="python")
    payload["attempts"][0].update(reason=reason, http_status=404 if reason == "no_record" else 200)
    collection = PerformanceCollection.model_validate(payload)
    with pytest.raises(ValueError, match="origin"):
        import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
            _contract(project).source,
            run.preflight,
            (run.collections[0], collection, *run.collections[2:]),
        )


def test_crux_origin_measurement_requires_actual_origin_attempt(project):
    run = performance_run(project)
    payload = run.collections[1].model_dump(mode="json")
    first = payload["attempts"][0]
    first.update(state="AVAILABLE", reason=None, http_status=200)
    payload["field"] = {
        "binding": run.binding.model_dump(mode="json"),
        "evidence_id": "origin-evidence",
        "attempt_id": first["attempt_id"],
        "requested_url": first["requested_url"],
        "record_key": first["requested_url"],
        "scope": "origin",
        "device": "mobile",
        "observed_at": first["ended_at"],
        "period": {"first_date": "2026-08-01", "last_date": "2026-08-28"},
        "metrics": [
            {"name": name, "value": 0.0, "unit": "unitless" if name == "cls" else "ms"}
            for name in ("lcp", "inp", "cls")
        ],
    }
    collection = PerformanceCollection.model_validate(payload)
    with pytest.raises(ValueError, match="scope"):
        import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
            _contract(project).source,
            run.preflight,
            (run.collections[0], collection, *run.collections[2:]),
        )


def normalized_crux_run(project):
    from ai_search_audit.performance_http import PerformanceRequest
    from ai_search_audit.performance_normalizers import normalize_crux
    from tests.test_performance_normalizers import crux_payload

    run = performance_run(project)
    attempt = run.collections[1].attempts[0]
    payload = crux_payload()
    payload["record"]["key"]["url"] = attempt.requested_url + "normalized"
    payload["urlNormalizationDetails"] = {
        "originalUrl": attempt.requested_url,
        "normalizedUrl": attempt.requested_url + "normalized",
        "ignoredProviderData": "must not persist",
    }
    field = normalize_crux(
        payload,
        request=PerformanceRequest(
            provider="crux", requested_url=attempt.requested_url, device="mobile", locale="en"
        ),
        binding=run.binding,
        attempt_id=attempt.attempt_id,
        evidence_id="normalized-field",
        observed_at=attempt.ended_at,
        page_url=attempt.requested_url,
    )
    collection = PerformanceCollection(
        attempts=(
            attempt.model_copy(
                update={
                    "state": "AVAILABLE",
                    "reason": None,
                    "http_status": 200,
                }
            ),
        ),
        field=field,
    )
    return import_module("ai_search_audit.diagnostic_workflow").assemble_performance_run(
        _contract(project).source,
        run.preflight,
        (run.collections[0], collection, *run.collections[2:]),
    )


def test_real_normalizer_retains_allowlisted_url_mapping_through_store(project):
    run = normalized_crux_run(project)
    field = run.collections[1].field
    assert field.url_normalization.original_url == field.requested_url
    assert field.url_normalization.normalized_url == field.record_key
    with pytest.raises(ValidationError):
        field.url_normalization.normalized_url = "https://other.example/"
    path = DiagnosticStore(project).publish(run)
    assert b"ignoredProviderData" not in (path / "evidence.jsonl").read_bytes()
    assert b"must not persist" not in (path / "evidence.jsonl").read_bytes()
    assert _load(project, path).run == run


@pytest.mark.parametrize("boundary", ["publish", "load"])
@pytest.mark.parametrize("mutation", ["missing", "original", "normalized", "origin"])
def test_crux_record_mapping_rejects_missing_or_forged_binding(project, boundary, mutation):
    run = normalized_crux_run(project)
    payload = run.model_dump(mode="json")
    field = payload["collections"][1]["field"]
    if mutation == "missing":
        field.pop("url_normalization", None)
    else:
        field["url_normalization"] = {
            "original_url": field["requested_url"],
            "normalized_url": field["record_key"],
        }
        if mutation == "original":
            field["url_normalization"]["original_url"] += "other"
        elif mutation == "normalized":
            field["url_normalization"]["normalized_url"] += "other"
        else:
            field["record_key"] = "https://other.example/normalized"
            field["url_normalization"]["normalized_url"] = field["record_key"]
    store = DiagnosticStore(project)
    if boundary == "publish":
        with pytest.raises(ValueError):
            store.publish(run.model_copy(update=payload))
    else:
        path = store.publish(run)
        lines = [json.loads(line) for line in (path / "evidence.jsonl").read_bytes().splitlines()]
        for line in lines:
            if line["attempts"][0]["attempt_id"] == field["attempt_id"]:
                line["field"] = field
        data = b"".join((json.dumps(line) + "\n").encode() for line in lines)
        (path / "evidence.jsonl").write_bytes(data)
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["files"][1].update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
        rewrite(path, "manifest.json", manifest)
        with pytest.raises(ValueError):
            store.load("public-v1", path.name)
