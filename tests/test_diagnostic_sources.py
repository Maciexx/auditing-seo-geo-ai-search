from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import ai_search_audit.project_orchestrator as project_orchestrator
from ai_search_audit.diagnostic_sources import load_diagnostic_source
from ai_search_audit.project_orchestrator import (
    enrich_project,
    update_project_context,
    validate_project,
)
from tests.test_crawler import public_resolver, transport
from tests.test_project_orchestrator import (
    NOW,
    _create,
    _owner_intake,
    _visibility_intake,
    _visibility_intake_for_period,
)


def _source_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("locale", ["pl", "en"])
def test_load_source_derives_binding_and_content_from_validated_bundle(tmp_path, locale):
    manifest = _create(tmp_path, report_locale=locale)
    project_root = tmp_path / "clients" / "example"
    version = manifest.versions[0]
    audit_path = project_root / version.relative_path / "engine" / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    before = _source_bytes(project_root)

    source = load_diagnostic_source(
        "project:example", clients_root=tmp_path / "clients", source_version="public-v1"
    )

    assert source.binding.project_id == manifest.project_id
    assert source.binding.source_version == version.version_id
    assert source.binding.audit_id == version.audit_id
    assert source.binding.report_locale == locale
    assert source.binding.domain == audit["site"]["domain"]
    assert source.binding.source_sha256 == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert source.canonical_domains == manifest.canonical_domains
    assert source.page_urls == tuple(page["url"] for page in audit["pages"])
    assert [prompt.model_dump(mode="json") for prompt in source.prompts] == audit["ai_prompts"]
    assert "report_status" not in source.binding.model_dump()
    assert _source_bytes(project_root) == before


def test_operation_cache_isolates_projects_with_same_version_and_audit_identity(tmp_path):
    from ai_search_audit.diagnostic_sources import _diagnostic_validation_operation

    first = _create(tmp_path, domain="https://studio.example")
    second = _create(tmp_path, domain="https://studio.example", project_id="other")
    assert first.latest_audit_id == second.latest_audit_id
    clients = tmp_path / "clients"
    alias = tmp_path / "linked-clients"
    alias.symlink_to(clients, target_is_directory=True)

    @_diagnostic_validation_operation
    def read_sources():
        left = load_diagnostic_source(
            "project:example", clients_root=clients, source_version="public-v1"
        )
        # The original caller path must be checked even when its real source is already cached.
        with pytest.raises(ValueError, match="real directory"):
            load_diagnostic_source(
                "project:example", clients_root=alias, source_version="public-v1"
            )
        right = load_diagnostic_source(
            "project:other", clients_root=clients, source_version="public-v1"
        )
        return left, right

    left, right = read_sources()
    assert left.binding.project_id == "example"
    assert right.binding.project_id == "other"
    assert left.binding.audit_id == right.binding.audit_id


def test_recursive_source_validation_rejects_cycle_and_resets_operation(tmp_path, monkeypatch):
    import ai_search_audit.diagnostic_sources as sources

    _create(tmp_path, domain="https://studio.example")
    clients = tmp_path / "clients"

    def recursive(*args, **kwargs):
        return load_diagnostic_source(
            "project:example", clients_root=clients, source_version="public-v1"
        )

    with monkeypatch.context() as patch:
        patch.setattr(sources, "_load_validated_source", recursive)
        with pytest.raises(ValueError, match="cyclic diagnostic source validation"):
            load_diagnostic_source(
                "project:example", clients_root=clients, source_version="public-v1"
            )
    assert (
        load_diagnostic_source(
            "project:example", clients_root=clients, source_version="public-v1"
        ).binding.project_id
        == "example"
    )


@pytest.mark.parametrize("field", ["domain", "locale", "audit_id", "project_id"])
def test_load_source_rejects_manifest_bundle_identity_mismatch_without_writes(tmp_path, field):
    _create(tmp_path)
    project_root = tmp_path / "clients" / "example"
    manifest_path = project_root / "project.json"
    manifest = json.loads(manifest_path.read_bytes())
    if field == "domain":
        manifest["canonical_domains"] = ["other.example"]
    elif field == "locale":
        manifest["report_locale"] = "pl"
    elif field == "audit_id":
        manifest["latest_audit_id"] = "other-audit"
        manifest["versions"][0]["audit_id"] = "other-audit"
    else:
        manifest["project_id"] = "other"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _source_bytes(project_root)

    with pytest.raises(ValueError, match="identity|locale|domain|ID"):
        load_diagnostic_source(
            "project:example", clients_root=tmp_path / "clients", source_version="public-v1"
        )

    assert _source_bytes(project_root) == before


@pytest.mark.parametrize(
    ("project_ref", "source_version", "message"),
    [
        ("project:example", "public-v99", "source version"),
        ("project:../example", "public-v1", "safe identifier"),
        ("project:example", "../public-v1", "safe identifier"),
        ("example", "public-v1", "project reference"),
    ],
)
def test_load_source_rejects_invalid_reference_without_writes(
    tmp_path, project_ref, source_version, message
):
    _create(tmp_path)
    clients_root = tmp_path / "clients"
    before = _source_bytes(clients_root)

    with pytest.raises(ValueError, match=message):
        load_diagnostic_source(
            project_ref, clients_root=clients_root, source_version=source_version
        )

    assert _source_bytes(clients_root) == before


def test_loaded_prompt_cannot_be_mutated_and_source_bytes_are_unchanged(tmp_path):
    _create(tmp_path)
    project_root = tmp_path / "clients" / "example"
    before = _source_bytes(project_root)
    source = load_diagnostic_source(
        "project:example", clients_root=tmp_path / "clients", source_version="public-v1"
    )

    with pytest.raises(ValidationError):
        source.prompts[0].text = "changed"
    with pytest.raises(TypeError):
        source.prompts[0].target_entities[0] = "changed"
    with pytest.raises(TypeError):
        source.prompts[0] = source.prompts[-1]

    assert _source_bytes(project_root) == before


def test_load_source_rejects_tampered_audit_instead_of_trusting_new_digest(tmp_path):
    manifest = _create(tmp_path)
    project_root = tmp_path / "clients" / "example"
    audit_path = project_root / manifest.versions[0].relative_path / "engine" / "audit.json"
    audit_path.write_bytes(audit_path.read_bytes() + b"\n")
    before = _source_bytes(project_root)

    with pytest.raises(ValueError, match="metadata|hash|bytes"):
        load_diagnostic_source(
            "project:example", clients_root=tmp_path / "clients", source_version="public-v1"
        )

    assert _source_bytes(project_root) == before


def test_load_source_selects_exact_historical_version_across_all_stages(tmp_path, monkeypatch):
    _create(tmp_path)
    owner_dir, owner_payload = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_payload,
        now=NOW,
    )
    visibility_dir, visibility_payload = _visibility_intake(tmp_path)
    enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=visibility_dir,
        normalized_intake=visibility_payload,
        now=NOW,
    )
    follow_up_time = datetime(2027, 8, 31, 10, tzinfo=UTC)
    follow_up_dir, follow_up_payload = _visibility_intake_for_period(
        tmp_path, period_start=follow_up_time, value=5
    )
    real_run = project_orchestrator.run_public_audit

    def local_public_run(*args, **kwargs):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_public_run)
    manifest = validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=follow_up_dir,
        normalized_intake=follow_up_payload,
        implementation_date=date(2026, 9, 1),
        now=follow_up_time,
    )
    project_root = tmp_path / "clients" / "example"
    before = _source_bytes(project_root)

    for version in manifest.versions:
        source = load_diagnostic_source(
            "project:example",
            clients_root=tmp_path / "clients",
            source_version=version.version_id,
        )
        assert source.binding.audit_id == version.audit_id
        assert source.binding.source_version == version.version_id
        audit_path = project_root / version.relative_path / "engine" / "audit.json"
        assert source.binding.source_sha256 == hashlib.sha256(audit_path.read_bytes()).hexdigest()

    assert _source_bytes(project_root) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("version_id", "context-v99"), ("report_status", "CLIENT_CONTEXT_DRAFT")],
)
def test_load_source_rejects_manifest_report_metadata_mismatch(tmp_path, field, value):
    _create(tmp_path)
    owner_dir, owner_payload = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_payload,
        now=NOW,
    )
    project_root = tmp_path / "clients" / "example"
    manifest_path = project_root / "project.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["versions"][-1][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _source_bytes(project_root)

    with pytest.raises(ValueError, match="metadata"):
        load_diagnostic_source(
            "project:example",
            clients_root=tmp_path / "clients",
            source_version=manifest["versions"][-1]["version_id"],
        )

    assert _source_bytes(project_root) == before
