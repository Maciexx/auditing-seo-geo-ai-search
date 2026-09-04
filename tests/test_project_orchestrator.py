from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import socket
import subprocess
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import ai_search_audit.orchestrator as orchestrator
import ai_search_audit.project_orchestrator as project_orchestrator
import ai_search_audit.prompts as prompts
from ai_search_audit.comparisons import ValidationComparison
from ai_search_audit.data_intake import (
    DateRange,
    FactApprovalState,
    OwnerFactInput,
    SourceArtifactDeclaration,
    VisibilityMetricPoint,
    VisibilityMetricSeriesInput,
    VisibilitySource,
    create_owned_intake_dir,
)
from ai_search_audit.data_requests import (
    DataRequestModule,
    DataRequestPack,
    EntityKind,
    render_data_request_markdown,
)
from ai_search_audit.models import AIPrompt, AuditRun, DataState, Entity, Page, Site
from ai_search_audit.owner_context import OwnerContext, OwnerFactField
from ai_search_audit.project_models import AuditStage, ReportStatus
from ai_search_audit.project_orchestrator import (
    create_project_audit,
    enrich_project,
    update_project_context,
    validate_project,
    validate_project_bundle,
)
from ai_search_audit.project_store import ProjectIdentityError, ProjectStore
from ai_search_audit.prompt_context import PromptTopic
from ai_search_audit.renderer import render_client_report
from ai_search_audit.report_models import (
    ClientReportData,
    ProjectReportMetadata,
    ValidationComparisonReportSection,
    report_context_digest,
)
from tests.test_crawler import public_resolver, transport

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


def _create(tmp_path: Path, **overrides: Any):
    values: dict[str, Any] = {
        "domain": "https://example.com",
        "clients_root": tmp_path / "clients",
        "project_id": "example",
        "client_name": "Example Client",
        "report_locale": "en",
        "max_pages": 2,
        "now": NOW,
        "crawler_transport": httpx.MockTransport(transport),
        "crawler_resolver": public_resolver,
    }
    values.update(overrides)
    return create_project_audit(**values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_project_persists_explicit_prompt_selection_and_validation_reuses_policy(
    tmp_path, monkeypatch
):
    from tests.test_selected_prompt_topics import _transport

    selection = PromptTopic(
        kind="category",
        locale="en",
        value="ceramics workshops",
        source_url="https://studio.example/",
        locator="content_text[29:47]",
        quote="ceramics workshops",
    )
    manifest = _create(
        tmp_path,
        domain="studio.example",
        selected_topics=(selection,),
        crawler_transport=_transport(),
    )
    source_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    source = AuditRun.model_validate_json((source_root / "engine" / "audit.json").read_text())
    assert source.configuration["prompt_topic_selection"] == [selection.model_dump(mode="json")]
    assert source.configuration["prompt_pack_version"] == "2.1.0"
    prompts.validate_automatic_prompt_pack(source)
    original = _bundle_hashes(source_root)
    real_run = project_orchestrator.run_public_audit

    def local_run(*args, **kwargs):
        return real_run(
            *args, **kwargs, crawler_transport=_transport(), crawler_resolver=public_resolver
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_run)
    updated = validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=datetime(2026, 9, 15, 10, tzinfo=UTC),
    )
    fresh_root = tmp_path / "clients" / "example" / updated.versions[-1].relative_path
    fresh = AuditRun.model_validate_json((fresh_root / "engine" / "audit.json").read_text())
    assert fresh.configuration["prompt_pack_version"] == "2.1.0"
    assert fresh.configuration["prompt_context_policy"] == "1.1.0"
    assert (
        fresh.configuration["prompt_topic_selection"]
        == source.configuration["prompt_topic_selection"]
    )
    prompts.validate_automatic_prompt_pack(fresh)
    assert _bundle_hashes(source_root) == original


@pytest.mark.parametrize("source_selected", [False, True])
def test_validation_can_explicitly_select_new_policy_or_reselect_changed_source(
    tmp_path, monkeypatch, source_selected
):
    from tests.test_selected_prompt_topics import _selection, _transport

    old_selection = PromptTopic(
        kind="category",
        locale="en",
        value="ceramics workshops",
        source_url="https://studio.example/",
        locator="content_text[29:47]",
        quote="ceramics workshops",
    )
    manifest = _create(
        tmp_path,
        domain="studio.example",
        crawler_transport=_transport(),
        selected_topics=(old_selection,) if source_selected else None,
    )
    source_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    source = AuditRun.model_validate_json((source_root / "engine" / "audit.json").read_text())
    original = _bundle_hashes(source_root)
    fresh_page = source.pages[0].model_copy(
        update={"content_text": "Updated " + source.pages[0].content_text}
    )
    new_selection = _selection(fresh_page)
    real_run = project_orchestrator.run_public_audit

    def local_run(*args, **kwargs):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=_transport("Updated "),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_run)
    options = dict(
        clients_root=tmp_path / "clients",
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=datetime(2026, 9, 15, 10, tzinfo=UTC),
    )
    if source_selected:
        with pytest.raises(ValueError, match="selection"):
            validate_project("project:example", **options)
        assert _bundle_hashes(source_root) == original
    updated = validate_project("project:example", selected_topics=(new_selection,), **options)
    assert len(updated.versions) == 2
    fresh_root = tmp_path / "clients" / "example" / updated.versions[-1].relative_path
    fresh = AuditRun.model_validate_json((fresh_root / "engine" / "audit.json").read_text())
    assert fresh.configuration["prompt_pack_version"] == "2.1.0"
    assert fresh.configuration["prompt_topic_selection"] == [new_selection.model_dump(mode="json")]
    prompts.validate_automatic_prompt_pack(fresh)
    assert _bundle_hashes(source_root) == original


def _bundle_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path) for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def offline_foundations(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("foundation integration tests must not make live network requests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _service_transport(request: httpx.Request, *, ecommerce: bool = False) -> httpx.Response:
    """Synthetic bilingual public site, also reusable for local report QA."""
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
    if request.url.path not in {"/", "/pl/"}:
        return httpx.Response(404, request=request)
    base = f"https://{request.url.host}"
    locale = "pl" if request.url.path == "/pl/" else "en"
    category = "Tworzenie oprogramowania" if locale == "pl" else "Software development"
    area = "Polska" if locale == "pl" else "Poland"
    product = "Niebieski kubek" if locale == "pl" else "Blue mug"
    offer = (
        {
            "@type": "Product",
            "name": product,
            "sku": "MUG-1",
            "offers": {"@type": "Offer", "price": "19.95", "priceCurrency": "PLN"},
        }
        if ecommerce
        else {"@type": "Service", "serviceType": category, "areaServed": area}
    )
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "OnlineStore" if ecommerce else "OnlineBusiness",
                "@id": f"{base}/#org",
                "name": "Example Studio",
                "url": f"{base}/",
            },
            {
                "@type": "WebSite",
                "name": "Example Studio website",
                "url": f"{base}/",
                "publisher": {"@id": f"{base}/#org"},
            },
            offer,
        ],
    }
    visible_offer = f"{product}. SKU MUG-1. 19.95 PLN." if ecommerce else f"{category}. {area}."
    return httpx.Response(
        200,
        text=(
            f"<html lang='{locale}'><head><title>Example Studio</title>"
            f"<link rel='canonical' href='{base}{request.url.path}'>"
            f'<script type="application/ld+json">{json.dumps(graph)}</script>'
            f"</head><body><main><h1>Example Studio</h1><p>{visible_offer}</p>"
            "<a href='/'>English</a><a href='/pl/'>Polski</a></main></body></html>"
        ),
        headers={"content-type": "text/html"},
        request=request,
    )


@pytest.mark.parametrize("ecommerce", [False, True])
def test_public_foundations_are_source_bound_and_bundle_reads_are_immutable(
    tmp_path, offline_foundations, ecommerce
):
    manifest = _create(
        tmp_path,
        domain="https://studio.example",
        crawler_transport=httpx.MockTransport(
            lambda request: _service_transport(request, ecommerce=ecommerce)
        ),
    )
    root = tmp_path / "clients/example" / manifest.versions[0].relative_path
    before = _bundle_hashes(root)
    audit = AuditRun.model_validate_json((root / "engine/audit.json").read_text())
    pack = json.loads((root / "engine/ai-prompts.json").read_text())
    request = DataRequestPack.model_validate_json(
        (root / "next-audit-data-request.json").read_text()
    )

    assert audit.configuration["entity_classification_policy"] == "2.0.0"
    assert audit.configuration["prompt_pack_version"] == pack["version"] == "2.0.0"
    assert audit.configuration["prompt_context_policy"] == "1.0.0"
    assert {prompt["pack_version"] for prompt in pack["prompts"]} == {"2.0.0"}
    assert pack["prompts"] == [prompt.model_dump(mode="json") for prompt in audit.ai_prompts]
    assert pack["observed_ai_visibility_state"] == "UNAVAILABLE"
    assert not audit.ai_observations
    measurement = next(score for score in audit.scores if score.name == "Measurement Maturity")
    assert measurement.state is DataState.UNAVAILABLE and measurement.value is None
    assert audit.entity is not None and audit.entity.brand == "Example Studio"
    assert audit.entity.type == ("OnlineStore" if ecommerce else "OnlineBusiness")
    assert request.entity_kind is (EntityKind.ECOMMERCE if ecommerce else EntityKind.GENERIC)
    assert (
        DataRequestModule.MERCHANT_CENTER in {item.module for item in request.items}
    ) == ecommerce
    assert {page.language for page in audit.pages} == {"pl", "en"}
    assert Counter(prompt.locale for prompt in audit.ai_prompts) == {"pl": 6, "en": 6}
    assert all(prompt.target_entities == ["Example Studio"] for prompt in audit.ai_prompts)
    context = audit.configuration["prompt_context"]
    if ecommerce:
        assert context == []
        assert not any(prompt.intent == "category_discovery" for prompt in audit.ai_prompts)
    else:
        assert {(topic["locale"], topic["kind"], topic["value"]) for topic in context} == {
            ("en", "category", "Software development"),
            ("en", "service_area", "Poland"),
            ("pl", "category", "Tworzenie oprogramowania"),
            ("pl", "service_area", "Polska"),
        }
        for topic in context:
            page = next(page for page in audit.pages if str(page.final_url) == topic["source_url"])
            field = "serviceType" if topic["kind"] == "category" else "areaServed"
            assert topic["locator"] == f"json_ld[0].@graph[2].{field}"
            assert topic["quote"] == topic["value"] == page.json_ld[0]["@graph"][2][field]
            assert topic["quote"] in page.content_text
        for locale, category, area in (
            ("en", "Software development", "Poland"),
            ("pl", "Tworzenie oprogramowania", "Polska"),
        ):
            discovery = next(
                prompt
                for prompt in audit.ai_prompts
                if prompt.locale == locale and prompt.intent == "category_discovery"
            )
            assert category in discovery.text and area in discovery.text
            assert discovery.query_themes == ["Example Studio", category, area]
            comparison = next(
                prompt
                for prompt in audit.ai_prompts
                if prompt.locale == locale and prompt.intent == "comparison"
            )
            assert category in comparison.text
            assert comparison.query_themes == ["Example Studio", category]
    prompts.validate_automatic_prompt_pack(audit)
    validate_project_bundle(root, expected_project_id="example")
    assert _bundle_hashes(root) == before


@pytest.mark.parametrize(
    ("policy", "value", "error"),
    [
        ("entity_classification_policy", "1.0.0", "data-request identity"),
        ("prompt_pack_version", "1.1.0", "mismatched prompt pack version"),
    ],
)
def test_foundation_policy_artifact_disagreement_rejects_after_external_rehash(
    tmp_path, offline_foundations, policy, value, error
):
    manifest = _create(
        tmp_path,
        domain="https://studio.example",
        crawler_transport=httpx.MockTransport(_service_transport),
    )
    root = tmp_path / "clients/example" / manifest.versions[0].relative_path
    audit_path = root / "engine/audit.json"
    payload = json.loads(audit_path.read_text())
    payload["configuration"][policy] = value
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    _rehash_output_manifest(root)
    before = _bundle_hashes(root)

    with pytest.raises(ValueError, match=error):
        validate_project_bundle(root, expected_project_id="example")

    assert _bundle_hashes(root) == before


def _legacy_example_prompts(site: Site, **_ignored: Any) -> list[AIPrompt]:
    """Frozen actual 1.1.0 output for _create's Example crawl, independent of today's generator."""
    assert site.domain == "example.com" and site.brand == "Example"
    rows = [
        ("en", "brand_discovery", "29c45a3a1469", "What is Example, and what is it known for?"),
        (
            "en",
            "category_discovery",
            "456e1ef59a9c",
            "Which Hotel options should I consider in its target market?",
        ),
        (
            "en",
            "comparison",
            "9003b0ca7ffa",
            "Compare Example with similar Hotel options in its target market.",
        ),
        (
            "en",
            "recommendation",
            "74b99d1b3e65",
            "Would you recommend Example for someone looking for Hotel in its target market?",
        ),
        (
            "en",
            "factual_verification",
            "23409c5a9df8",
            "What verified services, location details, and key facts are published about Example?",
        ),
        (
            "en",
            "location_service",
            "00165430f2d5",
            "Which Hotel services does Example provide in its target market?",
        ),
        (
            "pl",
            "brand_discovery",
            "125649af5da6",
            "Czym jest Example i z czego jest znana ta marka?",
        ),
        (
            "pl",
            "category_discovery",
            "b4b8d1bbf3e0",
            "Które oferty w kategorii Hotel warto rozważyć w docelowym rynku?",
        ),
        (
            "pl",
            "comparison",
            "f4aac4fec400",
            "Porównaj Example z podobnymi ofertami Hotel w docelowym rynku.",
        ),
        (
            "pl",
            "recommendation",
            "a59587ff38b3",
            "Czy warto wybrać Example, szukając Hotel w docelowym rynku?",
        ),
        (
            "pl",
            "factual_verification",
            "47d41843c065",
            "Jakie zweryfikowane usługi, dane lokalizacyjne i kluczowe fakty "
            "opublikowano o Example?",
        ),
        (
            "pl",
            "location_service",
            "e2e9d8f8fb80",
            "Jakie usługi Hotel oferuje Example w docelowym rynku?",
        ),
    ]
    return [
        AIPrompt(
            prompt_id=f"{intent}-{locale}-{digest}",
            pack_version="1.1.0",
            locale=locale,
            intent=intent,
            text=text,
            target_entities=["Example"],
            query_themes=["Example", "Hotel", "Private stays", "Example Hotel"],
            expected_evidence_needs=["official website", "independent authoritative source"],
            suggested_providers=[
                "openai-search",
                "gemini-grounding",
                "perplexity",
                "claude-search",
            ],
        )
        for locale, intent, digest, text in rows
    ]


def _create_legacy(tmp_path: Path, monkeypatch, **overrides: Any):
    """Create legacy artifacts once, with historical prompts and policy set before compilation."""
    compile_run = orchestrator.compile_audit_run

    def compile_legacy(run: AuditRun, **kwargs):
        run.configuration["entity_classification_policy"] = "1.0.0"
        run.configuration["prompt_pack_version"] = "1.1.0"
        run.configuration.pop("prompt_context_policy", None)
        run.configuration.pop("prompt_context", None)
        return compile_run(run, **kwargs)

    with monkeypatch.context() as legacy:
        legacy.setattr(orchestrator, "generate_prompt_pack", _legacy_example_prompts)
        legacy.setattr(orchestrator, "compile_audit_run", compile_legacy)
        return _create(tmp_path, **overrides)


def test_saved_bundle_prompt_version_survives_generator_upgrade(tmp_path, monkeypatch):
    manifest = _create_legacy(tmp_path, monkeypatch)
    root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    before = {path.relative_to(root): _sha256(path) for path in root.rglob("*") if path.is_file()}
    saved = json.loads((root / "engine" / "ai-prompts.json").read_text())
    assert saved["version"] == "1.1.0"
    assert {prompt["pack_version"] for prompt in saved["prompts"]} == {"1.1.0"}

    monkeypatch.setattr(prompts, "PROMPT_PACK_VERSION", "2.0.0")
    monkeypatch.setattr(project_orchestrator, "PROMPT_PACK_VERSION", "2.0.0", raising=False)

    validate_project_bundle(root, expected_project_id="example")

    after = {path.relative_to(root): _sha256(path) for path in root.rglob("*") if path.is_file()}
    assert after == before


def test_compile_saved_prompt_pack_preserves_version_after_generator_upgrade(tmp_path, monkeypatch):
    manifest = _create_legacy(tmp_path, monkeypatch)
    root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    run = AuditRun.model_validate_json((root / "engine" / "audit.json").read_text())
    saved = (root / "engine" / "ai-prompts.json").read_bytes()
    assert {prompt.pack_version for prompt in run.ai_prompts} == {"1.1.0"}
    monkeypatch.setattr(prompts, "PROMPT_PACK_VERSION", "2.0.0")
    monkeypatch.setattr(orchestrator, "PROMPT_PACK_VERSION", "2.0.0", raising=False)

    orchestrator.compile_audit_run(run, output_dir=tmp_path / "recompiled", report_locale="en")

    assert (tmp_path / "recompiled" / "ai-prompts.json").read_bytes() == saved


def test_frozen_legacy_lifecycle_preserves_baseline_and_new_crawl_is_noncomparable(
    tmp_path, monkeypatch, offline_foundations
):
    from ai_search_audit.benchmark import (
        compare_benchmarks,
        prepare_benchmark_worksheet,
        validate_benchmark_responses,
    )
    from ai_search_audit.client_delivery import finalize_client_report
    from ai_search_audit.comparisons import ComparisonLimitation
    from ai_search_audit.diagnostic_sources import load_diagnostic_source
    from ai_search_audit.diagnostic_store import DiagnosticStore
    from tests.test_benchmark import _setup
    from tests.test_client_pdf_script import render_edition
    from tests.test_diagnostic_workflow import _contract

    manifest = _create_legacy(tmp_path, monkeypatch, client_name="Example Studio")
    project_root = tmp_path / "clients/example"
    public_root = project_root / manifest.versions[0].relative_path
    original_files = _bundle_hashes(public_root)
    saved_pack = (public_root / "engine/ai-prompts.json").read_bytes()
    original = AuditRun.model_validate_json((public_root / "engine/audit.json").read_text())
    assert original.configuration["entity_classification_policy"] == "1.0.0"
    assert original.configuration["prompt_pack_version"] == "1.1.0"
    assert "prompt_context_policy" not in original.configuration
    assert "prompt_context" not in original.configuration
    assert original.ai_prompts == _legacy_example_prompts(original.site)

    def forbidden(*_args, **_kwargs):
        pytest.fail("saved legacy operations must not crawl or regenerate prompts")

    with monkeypatch.context() as historical:
        historical.setattr(project_orchestrator, "run_public_audit", forbidden)
        historical.setattr(orchestrator, "generate_prompt_pack", forbidden)
        historical.setattr(prompts, "generate_prompt_pack", forbidden)
        contract = _contract(project_root)
        worksheet = prepare_benchmark_worksheet(contract.source, _setup())
        assert worksheet.pack_version == "1.1.0"
        assert [prompt.model_dump(mode="json") for prompt in worksheet.prompts] == [
            prompt.model_dump(mode="json") for prompt in original.ai_prompts
        ]
        baseline_path = _benchmark_run(project_root)
        baseline_run = DiagnosticStore(project_root).load("public-v1", baseline_path.name)
        baseline = baseline_run.run.benchmark
        assert baseline is not None
        assert baseline_run.run.cleanup.status == "deleted"
        assert baseline.worksheet == worksheet
        baseline_before = baseline.model_dump_json()
        baseline_files = _bundle_hashes(baseline_path)
        assert _bundle_hashes(public_root) == original_files

        imported = prompts.import_observations(
            [
                {
                    "observation_id": "legacy-observation",
                    "prompt_id": worksheet.prompts[0].prompt_id,
                    "provider": "interactive-openai-search",
                    "observed_at": NOW.isoformat(),
                    "grounded": True,
                    "brand_mentioned": True,
                    "citations": ["https://example.com/"],
                }
            ]
        )
        assert imported[0].prompt_id == original.ai_prompts[0].prompt_id

        owned, intake = _owner_intake(tmp_path, entity="Example Studio")
        context = update_project_context(
            "project:example",
            clients_root=project_root.parent,
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )
        assert not owned.exists()
        context_version = context.versions[-1]
        context_root = project_root / context_version.relative_path
        context_files = _bundle_hashes(context_root)
        metrics_dir, metrics = _visibility_intake(tmp_path)
        enriched = enrich_project(
            "project:example",
            clients_root=project_root.parent,
            intake_dir=metrics_dir,
            normalized_intake=metrics,
            now=NOW.replace(hour=11),
        )
        assert not metrics_dir.exists()
        assert _bundle_hashes(context_root) == context_files
        for version in enriched.versions:
            root = project_root / version.relative_path
            run = AuditRun.model_validate_json((root / "engine/audit.json").read_text())
            assert original.configuration.items() <= run.configuration.items()
            assert run.ai_prompts == original.ai_prompts
            assert (root / "engine/ai-prompts.json").read_bytes() == saved_pack
            source = load_diagnostic_source(
                "project:example",
                clients_root=project_root.parent,
                source_version=version.version_id,
            )
            current = prepare_benchmark_worksheet(source, _setup())
            assert current.prompts == worksheet.prompts
            assert current.pack_content_hash == worksheet.pack_content_hash
            assert current.pack_version == "1.1.0"
        assert enriched.versions[-1].source_audit_id == context_version.audit_id
        visibility = json.loads(
            (
                project_root
                / enriched.versions[-1].relative_path
                / "aggregates/visibility-metrics.json"
            ).read_text()
        )
        assert visibility["metrics"][0]["value"] == 0
        assert visibility["metrics"][0]["state"] == "AVAILABLE"
        historical_files = _bundle_hashes(project_root / "audits")

        rendered, markdown, pdf, _, command = render_edition(tmp_path, hero=False)
        assert rendered.returncode == 0, rendered.stderr
        rendered = subprocess.run(
            command + ["--audit-id", original.audit_id], capture_output=True, text=True
        )
        assert rendered.returncode == 0, rendered.stderr
        edition = finalize_client_report(
            "project:example",
            clients_root=project_root.parent,
            version_id="public-v1",
            markdown_path=markdown,
            pdf_path=pdf,
            reviewed_pdf_sha256=_sha256(pdf),
            no_hero_reason="No suitable controlled image",
            diagnostic_run_ref=f"public-v1/{baseline_path.name}",
        )
        delivery = json.loads((edition / "delivery.json").read_text())
        assert delivery["audit_id"] == original.audit_id
        assert delivery["diagnostic_run"]["manifest_sha256"] == baseline_run.manifest_sha256
        assert delivery["report_status"] == "PUBLIC_EVIDENCE_DRAFT"
        assert _bundle_hashes(project_root / "audits") == historical_files
        assert _bundle_hashes(public_root) == original_files
        assert _bundle_hashes(baseline_path) == baseline_files

    validated = _diagnostic_validation(tmp_path, monkeypatch)
    fresh_version = validated.versions[-1]
    fresh_root = project_root / fresh_version.relative_path
    fresh = AuditRun.model_validate_json((fresh_root / "engine/audit.json").read_text())
    assert fresh_version.source_audit_id == enriched.latest_audit_id
    assert fresh.configuration["entity_classification_policy"] == "2.0.0"
    assert fresh.configuration["prompt_pack_version"] == "2.0.0"
    assert {prompt.pack_version for prompt in fresh.ai_prompts} == {"2.0.0"}
    assert [prompt.text for prompt in fresh.ai_prompts] != [
        prompt.text for prompt in original.ai_prompts
    ]
    comparison = ValidationComparison.model_validate_json(
        (fresh_root / "aggregates/validation-comparison.json").read_text()
    )
    assert comparison.ai_visibility is not None
    assert comparison.ai_visibility.state is DataState.UNKNOWN
    assert ComparisonLimitation.PROMPT_PACK_MISMATCH in comparison.ai_visibility.limitations
    assert comparison.ai_visibility.absolute_delta is None
    fresh_source = load_diagnostic_source(
        "project:example", clients_root=project_root.parent, source_version=fresh_version.version_id
    )
    fresh_worksheet = prepare_benchmark_worksheet(fresh_source, _setup())
    assert fresh_worksheet.pack_content_hash != worksheet.pack_content_hash
    assert fresh_worksheet.setup_fingerprint == worksheet.setup_fingerprint
    prompt = next(prompt for prompt in fresh_worksheet.prompts if prompt.locale == "en")
    follow_up = validate_benchmark_responses(
        fresh_source,
        fresh_worksheet,
        [
            dict(
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.text,
                observed_at=(NOW + timedelta(days=15)).isoformat(),
                response_text="No matching business was returned in this synthetic response.",
                complete=True,
                grounded=True,
                brand_mentioned=False,
                response_truncated=False,
                citations=[],
                citations_complete=True,
                inspection_scope="full_response",
            )
        ],
    )
    result = compare_benchmarks(
        baseline, follow_up, baseline_source=contract.source, follow_up_source=fresh_source
    )
    assert result.state is DataState.UNKNOWN
    assert result.limitations == ("Full prompt pack content differs.",)
    assert result.mention_delta is result.citation_delta is None
    assert result.baseline_metrics.measured == result.follow_up_metrics.measured == 1
    assert baseline.model_dump_json() == baseline_before
    reread = DiagnosticStore(project_root).load("public-v1", baseline_path.name)
    assert reread.run.benchmark == baseline
    assert reread.manifest_sha256 == baseline_run.manifest_sha256
    assert _bundle_hashes(baseline_path) == baseline_files
    assert all(
        _sha256(project_root / "audits" / path) == digest
        for path, digest in historical_files.items()
    )


@pytest.mark.parametrize("configuration", [{}, {"entity_classification_policy": "1.0.0"}])
@pytest.mark.parametrize(
    ("entity_type", "structured_data"),
    [
        ("OnlineBusiness", {}),
        ("CollectionPage", {}),
        ("Product", {}),
        ("Organization", {"@type": "OnlineBusiness"}),
        ("Organization", {"@type": "CollectionPage"}),
        ("Service", {"@type": "Service", "offers": {"@type": "Offer"}}),
    ],
)
def test_legacy_entity_policy_preserves_broad_ecommerce_classification(
    configuration, entity_type, structured_data
):
    run = AuditRun(
        audit_id="audit-example",
        audit_engine_version="0.2.0",
        site=Site(domain="studio.example", base_url="https://studio.example"),
        ruleset_version="test",
        ruleset_verified_date=date(2026, 9, 3),
        timestamp=NOW,
        scores=[],
        configuration=configuration,
        entity=Entity(brand="Studio", type=entity_type, domain="studio.example"),
        pages=[
            Page(
                url="https://studio.example",
                final_url="https://studio.example",
                status_code=200,
                json_ld=[structured_data],
            )
        ],
    )
    before = run.model_dump_json()

    assert project_orchestrator._entity_kind(run) is EntityKind.ECOMMERCE
    assert run.model_dump_json() == before


@pytest.mark.parametrize("marker", [None, "", "3.0.0", 1, True, [], {}])
def test_entity_classifier_rejects_unknown_policy(marker):
    run = AuditRun(
        audit_id="audit-example",
        audit_engine_version="0.2.0",
        site=Site(domain="studio.example", base_url="https://studio.example"),
        ruleset_version="test",
        ruleset_verified_date=date(2026, 9, 3),
        timestamp=NOW,
        scores=[],
        configuration={"entity_classification_policy": marker},
    )

    with pytest.raises(ValueError, match="unsupported entity classification policy"):
        project_orchestrator._entity_kind(run)


def _diagnostic_validation(tmp_path, monkeypatch, **options):
    real_run = project_orchestrator.run_public_audit

    def local_run(*args, **kwargs):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_run)
    return validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=datetime(2026, 9, 15, 10, tzinfo=UTC),
        **options,
    )


def _benchmark_run(root, *, mentioned=True, model="example-1.0", observed_at=NOW):
    from tests.test_benchmark import _setup
    from tests.test_diagnostic_workflow import _contract, _intake, _run

    contract = _contract(root)
    prompt = next(p for p in contract.worksheet.prompts if p.locale == "en")
    return _run(
        root,
        owned=_intake(
            root,
            worksheet=contract.worksheet.model_dump(mode="json"),
            setup=_setup(model_id=model).model_dump(mode="json"),
            responses=[
                dict(
                    prompt_id=prompt.prompt_id,
                    prompt_text=prompt.text,
                    observed_at=observed_at.isoformat(),
                    response_text="Studio service.",
                    complete=True,
                    grounded=True,
                    brand_mentioned=mentioned,
                    response_truncated=False,
                    citations=[f"https://{contract.source.binding.domain}/"] if mentioned else [],
                    citations_complete=True,
                    inspection_scope="full_response",
                )
            ],
        ),
    )


@pytest.mark.parametrize(
    "mode", ["absent", "missing", "comparable", "unknown-model", "different-model"]
)
def test_validation_persists_source_bound_supplementary_diagnostics(tmp_path, monkeypatch, mode):
    from ai_search_audit.diagnostic_store import DiagnosticStore
    from tests.test_diagnostic_workflow import _run

    initial = _create(tmp_path, domain="https://studio.example")
    root = tmp_path / "clients/example"
    options = {}
    if mode != "absent":
        if mode == "missing":
            _run(root)
            _run(root)
        else:
            _benchmark_run(root, mentioned=False)
            _benchmark_run(
                root,
                model=None
                if mode == "unknown-model"
                else "example-2.0"
                if mode == "different-model"
                else "example-1.0",
                observed_at=NOW + timedelta(days=1),
            )
        options = dict(
            baseline_diagnostic_run="public-v1/run-1", follow_up_diagnostic_run="public-v1/run-2"
        )
    manifest = _diagnostic_validation(tmp_path, monkeypatch, **options)
    version = manifest.versions[-1]
    bundle = root / version.relative_path
    snapshot = validate_project_bundle(
        bundle,
        expected_project_id="example",
        expected_version_number=2,
        expected_source_audit_id=initial.latest_audit_id,
        expect_owner_context=False,
        expect_validation_comparison=True,
        expected_stage=AuditStage.VALIDATION,
    )
    report = ClientReportData.model_validate_json(
        snapshot.files["engine/client-report-data.json"].content
    )
    field = "supplementary_diagnostic_comparison"
    if mode == "absent":
        assert getattr(report, field) is None
        assert field not in report.model_dump(mode="json")
        assert "aggregates/supplementary-diagnostic-comparison.json" not in snapshot.files
        # Pre-Task-9 serialization: no optional-null key may alter deterministic PDF IDs.
        payload = report.model_dump(mode="json")
        legacy = {
            "project": payload["project"],
            "owner_context": payload["owner_context"],
            "measurement": payload["measurement"],
            "validation_comparison": payload["validation_comparison"],
        }
        assert (
            report.context_digest
            == hashlib.sha256(
                json.dumps(
                    legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
    else:
        projected = getattr(report, field)
        canonical = json.loads(
            snapshot.files["aggregates/supplementary-diagnostic-comparison.json"].content
        )
        assert projected.model_dump(mode="json") == canonical
        left = DiagnosticStore(root).load("public-v1", "run-1")
        right = DiagnosticStore(root).load("public-v1", "run-2")
        assert projected.baseline_binding == left.run.binding
        assert projected.follow_up_binding == right.run.binding
        assert projected.baseline_manifest_sha256 == left.manifest_sha256
        assert projected.follow_up_manifest_sha256 == right.manifest_sha256
        assert projected.baseline_collection_range == left.run.collection_range
        assert projected.follow_up_collection_range == right.run.collection_range
        assert projected.comparison.mention_delta == (100 if mode == "comparable" else None)
        assert projected.comparison.citation_delta == (100 if mode == "comparable" else None)
        if mode == "missing":
            assert projected.comparison.baseline_metrics.state is DataState.UNAVAILABLE
            assert projected.comparison.baseline_metrics.mention_rate is None
        assert report.ai_search.observed_visibility_state is DataState.UNAVAILABLE
        assert report.audit_timestamp != projected.follow_up_collection_range.start
        assert version.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT


@pytest.mark.parametrize(
    "baseline,followup",
    [
        ("public-v1/run-1", None),
        (None, "public-v1/run-1"),
        ("../other/run-1", "public-v1/run-1"),
        ("latest/run-1", "public-v1/run-1"),
        ("public-v1/run-99", "public-v1/run-1"),
    ],
)
def test_validation_rejects_invalid_diagnostic_references_before_crawl(
    tmp_path, monkeypatch, baseline, followup
):
    _create(tmp_path, domain="https://studio.example")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid diagnostic references must fail before fresh crawl")

    monkeypatch.setattr(project_orchestrator, "run_public_audit", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=None,
            normalized_intake=None,
            implementation_date=date(2026, 9, 1),
            baseline_diagnostic_run=baseline,
            follow_up_diagnostic_run=followup,
        )
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


def test_diagnostic_comparison_rejects_symlinked_clients_ancestor(tmp_path):
    from ai_search_audit.diagnostic_models import DiagnosticRunReference
    from ai_search_audit.diagnostic_workflow import compare_diagnostic_runs
    from tests.test_diagnostic_workflow import _run

    _create(tmp_path, domain="https://studio.example")
    root = tmp_path / "clients/example"
    _run(root)
    link = tmp_path / "linked-clients"
    link.symlink_to(root.parent, target_is_directory=True)
    reference = DiagnosticRunReference(source_version="public-v1", run_id="run-1")
    with pytest.raises(ValueError, match="real directory"):
        compare_diagnostic_runs(
            "project:example",
            clients_root=link,
            baseline_reference=reference,
            follow_up_reference=reference,
        )


def test_validation_rejects_intact_cross_project_diagnostic_run(tmp_path, monkeypatch):
    from tests.test_diagnostic_workflow import _run

    _create(tmp_path, domain="https://studio.example")
    _create(tmp_path, domain="https://other.example", project_id="other")
    root = tmp_path / "clients/example"
    _run(root)
    original = _run(tmp_path / "clients/other")
    shutil.copytree(original, root / "diagnostics/public-v1/run-2")
    with pytest.raises(ValueError):
        _diagnostic_validation(
            tmp_path,
            monkeypatch,
            baseline_diagnostic_run="public-v1/run-1",
            follow_up_diagnostic_run="public-v1/run-2",
        )
    assert len(ProjectStore(root.parent).load("example").versions) == 1


def test_validate_diagnostics_rejects_original_symlink_before_project_read(tmp_path, monkeypatch):
    _create(tmp_path, domain="https://studio.example")
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "clients", target_is_directory=True)

    def forbidden(*args, **kwargs):
        pytest.fail("original diagnostic path must be checked before project reads")

    monkeypatch.setattr(ProjectStore, "load", forbidden)
    with pytest.raises(ValueError, match="real directory"):
        validate_project(
            "project:example",
            clients_root=link,
            intake_dir=None,
            normalized_intake=None,
            implementation_date=date(2026, 9, 1),
            baseline_diagnostic_run="public-v1/run-1",
            follow_up_diagnostic_run="public-v1/run-2",
        )


def test_nested_diagnostic_history_validates_each_source_once_per_operation(tmp_path, monkeypatch):
    from ai_search_audit.diagnostic_sources import load_diagnostic_source
    from ai_search_audit.diagnostic_workflow import run_diagnostics

    manifest = _create(tmp_path, domain="https://studio.example", max_pages=1)
    root = tmp_path / "clients/example"
    real_public = project_orchestrator.run_public_audit

    def public_run(*args, **kwargs):
        return real_public(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", public_run)
    for index in (1, 2):
        version = manifest.versions[-1]
        source = load_diagnostic_source(
            "project:example", clients_root=root.parent, source_version=version.version_id
        )
        owned = create_owned_intake_dir(tmp_path / "nested-intake")
        (owned / "normalized-intake.json").write_text(
            json.dumps({"expected_binding": source.binding.model_dump(mode="json")})
        )
        run = run_diagnostics(
            "project:example",
            clients_root=root.parent,
            source_version=version.version_id,
            owned_dir=owned,
            intake_root=owned.parent,
            now=NOW + timedelta(days=index),
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )
        reference = f"{version.version_id}/{run.name}"
        manifest = validate_project(
            "project:example",
            clients_root=root.parent,
            intake_dir=None,
            normalized_intake=None,
            implementation_date=NOW.date(),
            now=NOW + timedelta(days=index),
            baseline_diagnostic_run=reference,
            follow_up_diagnostic_run=reference,
        )

    calls = Counter()
    real_render = project_orchestrator.render_client_report

    def counted_render(report, destination):
        calls[report.audit_id] += 1
        return real_render(report, destination)

    monkeypatch.setattr(project_orchestrator, "render_client_report", counted_render)
    for _ in range(2):
        calls.clear()
        source = load_diagnostic_source(
            "project:example", clients_root=root.parent, source_version="validation-v3"
        )
        assert source.binding.audit_id == manifest.latest_audit_id
        assert calls == Counter({version.audit_id: 1 for version in manifest.versions})

    # A later operation must not trust a source that changed after the previous success.
    oldest = root / manifest.versions[0].relative_path / "engine/audit.json"
    original = oldest.read_bytes()
    oldest.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="metadata|hash|bytes"):
        load_diagnostic_source(
            "project:example", clients_root=root.parent, source_version="validation-v3"
        )
    oldest.write_bytes(original)
    calls.clear()
    load_diagnostic_source(
        "project:example", clients_root=root.parent, source_version="validation-v3"
    )
    assert calls == Counter({version.audit_id: 1 for version in manifest.versions})


@pytest.mark.parametrize(
    "tamper", ["metric", "date", "hash", "binding", "reference", "future-source"]
)
def test_supplementary_bundle_rejects_coherently_rehashed_projection(tmp_path, monkeypatch, tamper):
    from ai_search_audit.report_models import SupplementaryDiagnosticComparison

    initial = _create(tmp_path, domain="https://studio.example")
    root = tmp_path / "clients/example"
    _benchmark_run(root, mentioned=False)
    _benchmark_run(root, model=None)
    manifest = _diagnostic_validation(
        tmp_path,
        monkeypatch,
        baseline_diagnostic_run="public-v1/run-1",
        follow_up_diagnostic_run="public-v1/run-2",
    )
    version = manifest.versions[-1]
    bundle = root / version.relative_path
    aggregate = bundle / "aggregates/supplementary-diagnostic-comparison.json"
    payload = json.loads(aggregate.read_text())
    if tamper == "metric":
        payload["comparison"]["baseline_metrics"]["mention_rate"] = 50
    elif tamper == "date":
        payload["baseline_collection_range"]["start"] = "2026-01-01T10:00:00Z"
    elif tamper == "hash":
        payload["baseline_manifest_sha256"] = "0" * 64
    elif tamper == "binding":
        payload["baseline_binding"]["audit_id"] = "wrong-audit"
    elif tamper == "reference":
        payload["baseline_reference"]["run_id"] = "run-99"
    else:
        payload["baseline_reference"]["source_version"] = version.version_id
        payload["baseline_binding"]["source_version"] = version.version_id
    projection = SupplementaryDiagnosticComparison.model_validate(payload)
    aggregate.write_text(projection.model_dump_json())
    audit_file = bundle / "engine/audit.json"
    audit = json.loads(audit_file.read_text())
    audit["configuration"]["supplementary_diagnostic_comparison_sha256"] = _sha256(aggregate)
    audit_file.write_text(json.dumps(audit))
    report_file = bundle / "engine/client-report-data.json"
    report = ClientReportData.model_validate_json(report_file.read_text())
    updated = report.model_copy(
        update={
            "supplementary_diagnostic_comparison": projection,
            "context_digest": report_context_digest(
                report.project,
                report.owner_context,
                report.measurement,
                report.validation_comparison,
                projection,
            ),
        }
    )
    report_file.write_text(updated.model_dump_json())
    render_client_report(updated, next((bundle / "report").glob("*.pdf")))
    _rehash_output_manifest(bundle)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_project_bundle(
            bundle,
            expected_project_id="example",
            expected_version_number=2,
            expected_source_audit_id=initial.latest_audit_id,
            expect_owner_context=False,
            expect_validation_comparison=True,
            expected_stage=AuditStage.VALIDATION,
        )


def _owner_intake(
    tmp_path: Path,
    *,
    project_id: str = "example",
    canonical_domain: str = "example.com",
    entity: str = "Example Client",
    approval_state: FactApprovalState = FactApprovalState.APPROVED,
    conflict_ids: tuple[str, ...] = (),
) -> tuple[Path, dict[str, object]]:
    intake_root = tmp_path / "intake"
    owned = create_owned_intake_dir(intake_root, now=NOW)
    content = b'{"canonical_entity":"Example Client"}\n'
    source_path = owned / "owner-answers.json"
    source_path.write_bytes(content)
    payload: dict[str, object] = {
        "project_id": project_id,
        "canonical_domain": canonical_domain,
        "sources": [
            SourceArtifactDeclaration(
                source_id="owner-answer",
                filename="owner-answers.json",
                sha256=hashlib.sha256(content).hexdigest(),
                byte_count=len(content),
                platform=VisibilitySource.MANUAL,
                report_type="owner-context",
            ).model_dump(mode="json")
        ],
        "owner_facts": [
            OwnerFactInput(
                fact_id="fact-canonical-entity",
                field=OwnerFactField.CANONICAL_ENTITY.value,
                value=entity,
                source_id="owner-answer",
                approval_state=approval_state,
                conflict_ids=conflict_ids,
            ).model_dump(mode="json")
        ],
    }
    return owned, payload


def _visibility_intake(
    tmp_path: Path,
    *,
    project_id: str = "example",
    canonical_domain: str = "example.com",
) -> tuple[Path, dict[str, object]]:
    intake_root = tmp_path / "visibility-intake"
    owned = create_owned_intake_dir(intake_root, now=NOW)
    content = b"month,sessions\n2026-08,0\n"
    (owned / "ga4.csv").write_bytes(content)
    source = SourceArtifactDeclaration(
        source_id="ga4",
        filename="ga4.csv",
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        platform=VisibilitySource.GOOGLE_ANALYTICS,
        report_type="visibility-export",
        date_range=DateRange(start=NOW.date().replace(day=1), end=NOW.date()),
    )
    series = VisibilityMetricSeriesInput(
        metric_id="ga4-ai-sessions",
        source_id="ga4",
        metric="ga4.ai_assistant.sessions",
        unit="sessions",
        state=DataState.AVAILABLE,
        coverage=1,
        confidence=0.9,
        filters=("channel=AI Assistant",),
        points=(VisibilityMetricPoint(period_start=NOW.date().replace(day=1), value=0),),
    )
    return owned, {
        "project_id": project_id,
        "canonical_domain": canonical_domain,
        "sources": [source.model_dump(mode="json")],
        "metric_series": [series.model_dump(mode="json")],
    }


def _visibility_intake_for_period(
    tmp_path: Path,
    *,
    period_start: datetime,
    value: float,
    coverage: float = 1,
    confidence: float = 0.9,
) -> tuple[Path, dict[str, object]]:
    intake_root = tmp_path / f"visibility-intake-{period_start.date()}"
    owned = create_owned_intake_dir(intake_root, now=period_start)
    content = f"month,sessions\n{period_start:%Y-%m},{value}\n".encode()
    (owned / "ga4.csv").write_bytes(content)
    month_end = period_start.date().replace(day=31)
    source = SourceArtifactDeclaration(
        source_id="ga4-follow-up",
        filename="ga4.csv",
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        platform=VisibilitySource.GOOGLE_ANALYTICS,
        report_type="visibility-export",
        date_range=DateRange(start=period_start.date().replace(day=1), end=month_end),
    )
    series = VisibilityMetricSeriesInput(
        metric_id="ga4-ai-sessions",
        source_id=source.source_id,
        metric="ga4.ai_assistant.sessions",
        unit="sessions",
        state=DataState.AVAILABLE if coverage == 1 else DataState.PARTIAL,
        coverage=coverage,
        confidence=confidence,
        filters=("channel=AI Assistant",),
        points=(
            VisibilityMetricPoint(
                period_start=period_start.date().replace(day=1),
                value=value,
            ),
        ),
    )
    return owned, {
        "project_id": "example",
        "canonical_domain": "example.com",
        "sources": [source.model_dump(mode="json")],
        "metric_series": [series.model_dump(mode="json")],
    }


def _rehash_output_manifest(version_root: Path) -> None:
    output_path = version_root / "manifests" / "output-manifest.json"
    output = json.loads(output_path.read_text())
    for artifact in output["artifacts"]:
        artifact_path = version_root / artifact["relative_path"]
        artifact["bytes"] = artifact_path.stat().st_size
        artifact["sha256"] = _sha256(artifact_path)
    output_path.write_text(json.dumps(output), encoding="utf-8")


@pytest.mark.parametrize("relative_path", ("C:\\secret.txt", "folder\\file.txt"))
def test_artifact_manifest_paths_must_be_portable_posix(relative_path: str) -> None:
    with pytest.raises(ValueError, match="relative POSIX"):
        project_orchestrator.ArtifactMetadata(
            relative_path=relative_path,
            sha256="0" * 64,
            bytes=0,
        )


def test_create_project_audit_builds_immutable_public_v1_entirely_in_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run = project_orchestrator.run_public_audit
    observed_output: Path | None = None

    def recording_run(*args: Any, **kwargs: Any):
        nonlocal observed_output
        observed_output = Path(kwargs["output_dir"])
        assert ".staging" in observed_output.parts
        assert observed_output.name == "engine"
        assert not (tmp_path / "clients" / "example").exists()
        return real_run(*args, **kwargs)

    monkeypatch.setattr(project_orchestrator, "run_public_audit", recording_run)

    manifest = _create(tmp_path)

    assert observed_output is not None
    project_root = tmp_path / "clients" / "example"
    version_root = project_root / "audits" / "2026-08-31_public-v1"
    assert {path.name for path in version_root.iterdir()} == {
        "engine",
        "report",
        "manifests",
        "aggregates",
        "next-audit-data-request_en.md",
        "next-audit-data-request.json",
    }
    assert {path.name for path in (version_root / "engine").iterdir()} == {
        "audit.json",
        "evidence.jsonl",
        "implementation-backlog.csv",
        "ai-prompts.json",
        "report-draft.json",
        "client-report-data.json",
    }
    report = version_root / "report" / "Example_Client_AI_Search_SEO_Audit_EN_v1.pdf"
    assert report.is_file()
    assert not (version_root / "engine" / "client-report.pdf").exists()
    assert manifest.latest_audit_id == manifest.versions[0].audit_id
    assert manifest.versions[0].version_id == "public-v1"
    assert manifest.versions[0].stage is AuditStage.PUBLIC
    assert manifest.versions[0].report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    assert manifest.versions[0].source_audit_id is None
    assert manifest.versions[0].relative_path == "audits/2026-08-31_public-v1"
    public_report = ClientReportData.model_validate_json(
        (version_root / "engine" / "client-report-data.json").read_text()
    )
    assert public_report.project is not None
    assert public_report.project.project_id == "example"
    assert public_report.project.version_id == "public-v1"
    assert public_report.project.version_number == 1
    assert public_report.project.stage is AuditStage.PUBLIC
    assert public_report.project.report_status is ReportStatus.PUBLIC_EVIDENCE_DRAFT
    assert public_report.project.source_audit_id is None
    assert public_report.owner_context is None
    assert public_report.measurement is None


def test_persisted_audit_and_manifests_use_stable_relative_paths_and_hashes(
    tmp_path: Path,
) -> None:
    manifest = _create(tmp_path)
    version_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    audit = json.loads((version_root / "engine" / "audit.json").read_text())

    assert audit["output_paths"] == {
        "ai-prompts.json": "engine/ai-prompts.json",
        "audit.json": "engine/audit.json",
        "client-report-data.json": "engine/client-report-data.json",
        "client-report.pdf": "report/Example_Client_AI_Search_SEO_Audit_EN_v1.pdf",
        "evidence.jsonl": "engine/evidence.jsonl",
        "implementation-backlog.csv": "engine/implementation-backlog.csv",
        "report-draft.json": "engine/report-draft.json",
    }

    for name in ("input-manifest.json", "evidence-manifest.json", "output-manifest.json"):
        payload = json.loads((version_root / "manifests" / name).read_text())
        serialized = json.dumps(payload)
        assert payload["schema_version"] == "1.0.0"
        assert payload["version"] == "1.0.0"
        assert payload["project_id"] == "example"
        assert payload["audit_id"] == manifest.latest_audit_id
        assert str(tmp_path) not in serialized
        assert ".staging" not in serialized
        assert "raw-input" not in serialized

    input_manifest = json.loads((version_root / "manifests" / "input-manifest.json").read_text())
    assert input_manifest["raw_inputs"] == []
    assert input_manifest["public_target"]["canonical_domain"] == "example.com"
    output_manifest = json.loads((version_root / "manifests" / "output-manifest.json").read_text())
    assert "manifests/output-manifest.json" not in {
        artifact["relative_path"] for artifact in output_manifest["artifacts"]
    }
    for artifact in output_manifest["artifacts"]:
        path = version_root / artifact["relative_path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_public_bundle_rejects_coherent_wrong_project_report_metadata(
    tmp_path: Path,
) -> None:
    manifest = _create(tmp_path)
    version_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    report_path = version_root / "engine" / "client-report-data.json"
    payload = json.loads(report_path.read_text())
    payload["project"]["project_id"] = "other"
    project = ProjectReportMetadata.model_validate(payload["project"])
    payload["context_digest"] = report_context_digest(project, None, None)
    report = ClientReportData.model_validate(payload)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    pdf_path = next((version_root / "report").glob("*.pdf"))
    project_orchestrator.render_client_report(report, pdf_path)
    _rehash_output_manifest(version_root)

    with pytest.raises(ValueError, match="trusted version"):
        validate_project_bundle(version_root, expected_project_id="example")


def test_manifest_dtos_apply_canonical_project_id_validation(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    manifest_root = (
        tmp_path / "clients" / "example" / manifest.versions[0].relative_path / "manifests"
    )
    model_files = (
        (project_orchestrator.InputManifest, "input-manifest.json"),
        (project_orchestrator.EvidenceManifest, "evidence-manifest.json"),
        (project_orchestrator.OutputManifest, "output-manifest.json"),
    )

    for model, filename in model_files:
        payload = json.loads((manifest_root / filename).read_text())
        payload["project_id"] = "../escape"
        with pytest.raises(ValueError, match="safe identifier"):
            model.model_validate(payload)


@pytest.mark.parametrize(
    "tampered_documents",
    (
        ("input",),
        ("evidence",),
        ("output",),
        ("request",),
        ("input", "evidence", "output"),
        ("input", "evidence", "output", "request"),
    ),
)
def test_manifest_project_identity_tampering_is_rejected_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_documents: tuple[str, ...],
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def write_tampered_manifests(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        paths = {
            "input": version_root / "manifests" / "input-manifest.json",
            "evidence": version_root / "manifests" / "evidence-manifest.json",
            "request": version_root / "next-audit-data-request.json",
        }
        for name, path in paths.items():
            if name not in tampered_documents:
                continue
            payload = json.loads(path.read_text())
            payload["project_id"] = "other-project"
            path.write_text(json.dumps(payload), encoding="utf-8")

        output_path = version_root / "manifests" / "output-manifest.json"
        output = json.loads(output_path.read_text())
        if "output" in tampered_documents:
            output["project_id"] = "other-project"
        for artifact in output["artifacts"]:
            artifact_path = version_root / artifact["relative_path"]
            artifact["bytes"] = artifact_path.stat().st_size
            artifact["sha256"] = _sha256(artifact_path)
        output_path.write_text(json.dumps(output), encoding="utf-8")

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("tampered bundle reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", write_tampered_manifests)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="project identity"):
        _create(tmp_path)

    assert promoted is False
    assert not (tmp_path / "clients" / "example").exists()


def test_data_request_is_tailored_to_canonical_entity_without_unobserved_modules(
    tmp_path: Path,
) -> None:
    manifest = _create(tmp_path)
    version_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    request = DataRequestPack.model_validate_json(
        (version_root / "next-audit-data-request.json").read_text()
    )
    modules = {item.module for item in request.items}

    assert request.entity_kind.value == "local"
    assert request.observed_features == ()
    assert DataRequestModule.GOOGLE_BUSINESS_PROFILE in modules
    assert DataRequestModule.MERCHANT_CENTER not in modules
    assert DataRequestModule.BING_WEBMASTER_TOOLS not in modules
    assert DataRequestModule.SANITIZED_LOGS not in modules
    assert DataRequestModule.CRAWL_EXPORT not in modules
    assert DataRequestModule.AI_MONITORING_EXPORT not in modules
    markdown = version_root / "next-audit-data-request_en.md"
    assert markdown.is_file() and markdown.read_text(encoding="utf-8").startswith("# ")


@pytest.mark.parametrize(
    ("entity_type", "expected_kind", "expected_module", "excluded_module"),
    (
        ("Organization", "generic", None, DataRequestModule.MERCHANT_CENTER),
        (
            "Product",
            "generic",
            None,
            DataRequestModule.MERCHANT_CENTER,
        ),
    ),
)
def test_data_request_maps_non_local_canonical_entity_types_explicitly(
    tmp_path: Path,
    entity_type: str,
    expected_kind: str,
    expected_module: DataRequestModule | None,
    excluded_module: DataRequestModule,
) -> None:
    def entity_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nAllow: /\n",
                headers={"content-type": "text/plain"},
                request=request,
            )
        return httpx.Response(
            200,
            text=(
                "<html lang='en'><head><title>Example</title>"
                "<link rel='canonical' href='https://example.com/'>"
                f'<script type="application/ld+json">{{"@type":"{entity_type}",'
                '"name":"Example"}</script></head><body><h1>Example</h1></body></html>'
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    manifest = _create(
        tmp_path,
        crawler_transport=httpx.MockTransport(entity_transport),
        max_pages=1,
    )
    version_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    request = DataRequestPack.model_validate_json(
        (version_root / "next-audit-data-request.json").read_text()
    )
    modules = {item.module for item in request.items}

    assert request.entity_kind.value == expected_kind
    if expected_module is not None:
        assert expected_module in modules
    assert excluded_module not in modules


@pytest.mark.parametrize(
    ("entity_type", "offer_type", "expected_kind"),
    [
        ("OnlineStore", "Product", EntityKind.ECOMMERCE),
        ("OnlineStore", "Service", EntityKind.GENERIC),
        ("OnlineBusiness", "Service", EntityKind.GENERIC),
    ],
)
def test_data_request_requires_product_sales_evidence_for_merchant_center(
    tmp_path, entity_type, offer_type, expected_kind
):
    graph = {
        "@graph": [
            {"@type": entity_type, "name": "Example"},
            {
                "@type": offer_type,
                "name": "Blue mug",
                "sku": "MUG-1",
                "offers": {"@type": "Offer", "price": "19.95", "priceCurrency": "PLN"},
            },
        ]
    }

    def entity_transport(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        return httpx.Response(
            200,
            text=(
                "<html lang='en'><head><title>Example</title>"
                "<link rel='canonical' href='https://example.com/'>"
                f'<script type="application/ld+json">{json.dumps(graph)}</script>'
                "</head><body><h1>Example</h1><p>Blue mug</p></body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    manifest = _create(
        tmp_path, crawler_transport=httpx.MockTransport(entity_transport), max_pages=1
    )
    root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    request = DataRequestPack.model_validate_json(
        (root / "next-audit-data-request.json").read_text()
    )
    modules = {item.module for item in request.items}
    assert request.entity_kind is expected_kind
    assert (DataRequestModule.MERCHANT_CENTER in modules) == (expected_kind is EntityKind.ECOMMERCE)
    assert DataRequestModule.OWNER_CONTEXT in modules
    validate_project_bundle(root, expected_project_id="example")


def test_missing_authenticated_measurement_still_promotes_unavailable_not_zero(
    tmp_path: Path,
) -> None:
    manifest = _create(tmp_path)
    audit_path = (
        tmp_path
        / "clients"
        / "example"
        / manifest.versions[0].relative_path
        / "engine"
        / "audit.json"
    )
    scores = {item["name"]: item for item in json.loads(audit_path.read_text())["scores"]}

    assert scores["Measurement Maturity"]["state"] == DataState.UNAVAILABLE.value
    assert scores["Measurement Maturity"]["value"] is None
    assert list((audit_path.parent.parent / "report").glob("*.pdf"))


def test_existing_project_never_overwrites_and_identity_mismatch_is_explicit(
    tmp_path: Path,
) -> None:
    first = _create(tmp_path)
    project_root = tmp_path / "clients" / "example"
    hashes_before = {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in project_root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileExistsError):
        _create(tmp_path)
    with pytest.raises(ProjectIdentityError):
        _create(tmp_path, domain="https://different.example")
    with pytest.raises(ProjectIdentityError):
        _create(tmp_path, client_name="Different Client")

    hashes_after = {
        path.relative_to(project_root).as_posix(): _sha256(path)
        for path in project_root.rglob("*")
        if path.is_file()
    }
    assert hashes_after == hashes_before
    assert ProjectStore(tmp_path / "clients").load("example") == first


def test_real_crawl_failure_discards_pending_project(tmp_path: Path) -> None:
    def failed_crawl(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    with pytest.raises(RuntimeError, match="no usable HTML pages"):
        _create(tmp_path, crawler_transport=httpx.MockTransport(failed_crawl))

    assert not (tmp_path / "clients" / "example").exists()
    assert not list((tmp_path / "clients" / ".staging").iterdir())


def test_real_report_compilation_failure_discards_pending_project(tmp_path: Path) -> None:
    def invalidate_protected_findings(payload):
        return payload.model_copy(update={"findings": []})

    with pytest.raises(ValueError, match="protected finding changed"):
        _create(tmp_path, rewrite_provider=invalidate_protected_findings)

    assert not (tmp_path / "clients" / "example").exists()
    assert not list((tmp_path / "clients" / ".staging").iterdir())


@pytest.mark.parametrize(
    ("boundary", "message"),
    (
        ("run_public_audit", "crawl failed"),
        ("write_data_request", "request failed"),
        ("_relocate_pdf", "relocation failed"),
        ("validate_project_bundle", "manifest validation failed"),
    ),
)
def test_any_pipeline_failure_discards_only_owned_pending_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    message: str,
) -> None:
    clients_root = tmp_path / "clients"
    unrelated = clients_root / ".staging" / "unrelated-run"
    unrelated.mkdir(parents=True)
    marker = unrelated / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(project_orchestrator, boundary, fail)

    with pytest.raises(RuntimeError, match=message):
        _create(tmp_path)

    assert not (clients_root / "example").exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in (clients_root / ".staging").iterdir()} == {"unrelated-run"}


def test_artifacts_are_validated_before_project_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_validate = project_orchestrator.validate_project_bundle
    real_promote = ProjectStore.promote_new_project
    events: list[str] = []

    def recording_validate(version_root: Path, *, expected_project_id: str):
        snapshot = real_validate(version_root, expected_project_id=expected_project_id)
        events.append("validated")
        return snapshot

    def recording_promote(self: ProjectStore, pending, manifest):
        events.append("promoted")
        assert events == ["validated", "promoted"]
        return real_promote(self, pending, manifest)

    monkeypatch.setattr(project_orchestrator, "validate_project_bundle", recording_validate)
    monkeypatch.setattr(ProjectStore, "promote_new_project", recording_promote)

    _create(tmp_path)

    assert events == ["validated", "promoted"]


def test_bundle_validation_rejects_incomplete_output_manifest(tmp_path: Path) -> None:
    manifest = _create(tmp_path)
    version_root = tmp_path / "clients" / "example" / manifest.versions[0].relative_path
    output_path = version_root / "manifests" / "output-manifest.json"
    output = json.loads(output_path.read_text())
    output["artifacts"] = output["artifacts"][:-1]
    output_path.write_text(json.dumps(output), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact set"):
        project_orchestrator.validate_project_bundle(version_root, expected_project_id="example")


def test_invalid_pdf_is_rejected_and_cleaned_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_relocate = project_orchestrator._relocate_pdf

    def replace_with_non_pdf(source: Path, destination: Path) -> None:
        real_relocate(source, destination)
        destination.write_bytes(b"not a PDF")

    monkeypatch.setattr(project_orchestrator, "_relocate_pdf", replace_with_non_pdf)

    with pytest.raises(ValueError, match="PDF"):
        _create(tmp_path)

    assert not (tmp_path / "clients" / "example").exists()
    assert not list((tmp_path / "clients" / ".staging").iterdir())


@pytest.mark.parametrize(
    "bundle_entry",
    ("engine/audit.json", "report/pdf", "manifests/input-manifest.json", "data-request"),
)
def test_out_of_tree_identical_file_symlink_is_rejected_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle_entry: str,
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def replace_with_symlink(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        if bundle_entry == "report/pdf":
            source = next((version_root / "report").glob("*.pdf"))
        elif bundle_entry == "data-request":
            source = version_root / "next-audit-data-request.json"
        else:
            source = version_root / bundle_entry
        outside = tmp_path / f"outside-{source.name}"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("symlinked bundle reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", replace_with_symlink)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="symlink|regular"):
        _create(tmp_path)

    assert promoted is False
    assert not (tmp_path / "clients" / "example").exists()


def test_out_of_tree_directory_symlink_is_rejected_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def replace_engine_directory(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        engine = version_root / "engine"
        outside = tmp_path / "outside-engine"
        shutil.copytree(engine, outside)
        shutil.rmtree(engine)
        engine.symlink_to(outside, target_is_directory=True)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("symlinked ancestor reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", replace_engine_directory)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="symlink|directory"):
        _create(tmp_path)

    assert promoted is False


@pytest.mark.parametrize(
    ("relative_path", "add_to_output_manifest"),
    (
        ("raw-input.csv", True),
        ("screenshots/home.png", True),
        ("aggregates/output-manifest.json", False),
        ("aggregates/output-manifest.json", True),
    ),
)
def test_unexpected_bundle_file_is_rejected_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    add_to_output_manifest: bool,
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def add_unexpected_file(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        unexpected = version_root / relative_path
        unexpected.parent.mkdir(parents=True, exist_ok=True)
        unexpected.write_bytes(b"unexpected private input")
        if add_to_output_manifest:
            output_path = version_root / "manifests" / "output-manifest.json"
            output = json.loads(output_path.read_text())
            output["artifacts"].append(
                {
                    "relative_path": relative_path,
                    "sha256": _sha256(unexpected),
                    "bytes": unexpected.stat().st_size,
                }
            )
            output_path.write_text(json.dumps(output), encoding="utf-8")

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("unexpected file reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", add_unexpected_file)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="inventory|unexpected|artifact set"):
        _create(tmp_path)

    assert promoted is False


@pytest.mark.parametrize(
    "artifact",
    (
        "evidence",
        "report-draft",
        "client-report",
        "data-request-domain",
        "data-request-entity",
        "data-request-markdown",
        "audit-output-paths",
        "ai-prompts",
    ),
)
def test_cross_artifact_semantic_tampering_is_rejected_after_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def tamper_and_rehash(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        if artifact == "evidence":
            path = version_root / "engine" / "evidence.jsonl"
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            lines[0]["collector"] = "tampered-collector"
            path.write_text("".join(json.dumps(line) + "\n" for line in lines))
            evidence_manifest_path = version_root / "manifests" / "evidence-manifest.json"
            evidence_manifest = json.loads(evidence_manifest_path.read_text())
            evidence_manifest["artifact"]["bytes"] = path.stat().st_size
            evidence_manifest["artifact"]["sha256"] = _sha256(path)
            evidence_manifest_path.write_text(json.dumps(evidence_manifest), encoding="utf-8")
        elif artifact == "report-draft":
            path = version_root / "engine" / "report-draft.json"
            payload = json.loads(path.read_text())
            payload["target_domain"] = "different.example"
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif artifact == "client-report":
            path = version_root / "engine" / "client-report-data.json"
            payload = json.loads(path.read_text())
            payload["brand"] = "Different Brand"
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif artifact in {"data-request-domain", "data-request-entity"}:
            path = version_root / "next-audit-data-request.json"
            payload = json.loads(path.read_text())
            if artifact == "data-request-domain":
                payload["canonical_domain"] = "different.example"
            else:
                payload["entity_kind"] = "generic"
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif artifact == "data-request-markdown":
            path = version_root / "next-audit-data-request_en.md"
            path.write_text(path.read_text() + "\nTampered request.\n", encoding="utf-8")
        elif artifact == "audit-output-paths":
            path = version_root / "engine" / "audit.json"
            payload = json.loads(path.read_text())
            payload["output_paths"]["audit.json"] = "/tmp/.staging/audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path = version_root / "engine" / "ai-prompts.json"
            payload = json.loads(path.read_text())
            payload["prompts"][0]["text"] = "tampered prompt"
            path.write_text(json.dumps(payload), encoding="utf-8")
        _rehash_output_manifest(version_root)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("semantically inconsistent bundle reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", tamper_and_rehash)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="evidence|report|data-request|output paths|prompt"):
        _create(tmp_path)

    assert promoted is False


@pytest.mark.parametrize("invalid_pdf", (b"", b"%PDF-1.7\ntruncated"))
def test_structurally_invalid_pdf_is_rejected_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_pdf: bytes,
) -> None:
    real_relocate = project_orchestrator._relocate_pdf
    promoted = False

    def replace_pdf(source: Path, destination: Path) -> None:
        real_relocate(source, destination)
        destination.write_bytes(invalid_pdf)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("invalid PDF reached promotion")

    monkeypatch.setattr(project_orchestrator, "_relocate_pdf", replace_pdf)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="readable PDF"):
        _create(tmp_path)

    assert promoted is False


def test_changed_client_report_json_with_refreshed_hash_rejects_unchanged_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def change_report_json(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        report_data = version_root / "engine" / "client-report-data.json"
        payload = json.loads(report_data.read_text())
        payload["executive_summary"] = "Tampered but structurally valid summary."
        report_data.write_text(json.dumps(payload), encoding="utf-8")
        _rehash_output_manifest(version_root)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("report/PDF mismatch reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", change_report_json)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="PDF.*ClientReportData|ClientReportData.*PDF"):
        _create(tmp_path)

    assert promoted is False


def test_swapped_locale_pdf_with_refreshed_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_manifest = _create(tmp_path / "reference", report_locale="pl")
    reference_version = (
        tmp_path
        / "reference"
        / "clients"
        / "example"
        / reference_manifest.versions[0].relative_path
    )
    polish_pdf = next((reference_version / "report").glob("*.pdf")).read_bytes()
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def swap_pdf(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        next((version_root / "report").glob("*.pdf")).write_bytes(polish_pdf)
        _rehash_output_manifest(version_root)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("swapped locale PDF reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", swap_pdf)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="PDF.*ClientReportData|ClientReportData.*PDF"):
        _create(tmp_path / "target")

    assert promoted is False


@pytest.mark.parametrize("mutation", ("changed", "missing", "extra", "reordered"))
def test_backlog_projection_tampering_is_rejected_after_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def mutate_backlog(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        backlog_path = version_root / "engine" / "implementation-backlog.csv"
        rows = list(csv.reader(io.StringIO(backlog_path.read_text())))
        assert len(rows) >= 3
        if mutation == "changed":
            rows[1][2] = "Tampered title"
        elif mutation == "missing":
            rows.pop()
        elif mutation == "extra":
            rows.append(list(rows[-1]))
        else:
            rows[1], rows[2] = rows[2], rows[1]
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        backlog_path.write_text(buffer.getvalue(), encoding="utf-8")
        _rehash_output_manifest(version_root)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("tampered backlog reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", mutate_backlog)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="implementation backlog"):
        _create(tmp_path)

    assert promoted is False


def test_coherent_data_request_prose_tampering_is_rejected_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_manifests = project_orchestrator._write_manifests
    promoted = False

    def tamper_request_pair(**kwargs: Any) -> None:
        real_write_manifests(**kwargs)
        version_root = Path(kwargs["version_root"])
        request_json = version_root / "next-audit-data-request.json"
        payload = json.loads(request_json.read_text())
        payload["introduction"] = "Tampered but internally coherent request prose."
        pack = DataRequestPack.model_validate(payload)
        request_json.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
        request_markdown = version_root / "next-audit-data-request_en.md"
        request_markdown.write_text(render_data_request_markdown(pack), encoding="utf-8")
        _rehash_output_manifest(version_root)

    def reject_promotion(*_args: Any, **_kwargs: Any) -> None:
        nonlocal promoted
        promoted = True
        raise AssertionError("tampered request pack reached promotion")

    monkeypatch.setattr(project_orchestrator, "_write_manifests", tamper_request_pair)
    monkeypatch.setattr(ProjectStore, "promote_new_project", reject_promotion)

    with pytest.raises(ValueError, match="data-request.*canonical|canonical.*data-request"):
        _create(tmp_path)

    assert promoted is False


@pytest.mark.parametrize(
    ("project_id", "client_name", "locale", "max_pages"),
    (
        ("../escape", "Example", "en", 2),
        ("example", " ", "en", 2),
        ("example", "Example", "de", 2),
        ("example", "Example", "en", 0),
        ("example", "Example", "en", 501),
    ),
)
def test_invalid_inputs_fail_before_staging(
    tmp_path: Path,
    project_id: str,
    client_name: str,
    locale: str,
    max_pages: int,
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _create(
            tmp_path,
            project_id=project_id,
            client_name=client_name,
            report_locale=locale,
            max_pages=max_pages,
        )

    assert not (tmp_path / "clients" / ".staging").exists()


def test_update_project_context_reuses_persisted_audit_without_fresh_crawl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_manifest = _create(tmp_path)
    public_version = public_manifest.versions[0]
    public_audit_path = (
        tmp_path / "clients" / "example" / public_version.relative_path / "engine" / "audit.json"
    )
    public_audit_before = public_audit_path.read_bytes()
    owned, intake = _owner_intake(tmp_path)

    def forbidden_crawl(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("context update must not start a public crawl")

    monkeypatch.setattr(project_orchestrator, "run_public_audit", forbidden_crawl)

    manifest = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )

    assert not owned.exists()
    assert public_audit_path.read_bytes() == public_audit_before
    assert len(manifest.versions) == 2
    context_version = manifest.versions[-1]
    assert context_version.version_id == "context-v2"
    assert context_version.stage is AuditStage.CONTEXT
    assert context_version.source_audit_id == public_version.audit_id
    assert context_version.audit_id != public_version.audit_id
    assert context_version.report_status is ReportStatus.CLIENT_VALIDATED
    context_root = tmp_path / "clients" / "example" / context_version.relative_path
    assert (context_root / "aggregates" / "owner-context.json").is_file()
    assert (context_root / "report" / "Example_Client_AI_Search_SEO_Audit_EN_v2.pdf").is_file()
    assert (context_root / "next-audit-data-request_en.md").is_file()
    context_audit = json.loads((context_root / "engine" / "audit.json").read_text())
    public_configuration = json.loads(public_audit_before)["configuration"]
    assert public_configuration["entity_classification_policy"] == "2.0.0"
    assert public_configuration.items() <= context_audit["configuration"].items()
    assert context_audit["audit_id"] == context_version.audit_id
    assert context_audit["entity_consistency_matrix"]["canonical_entity"]["Example Client"]
    owner_evidence = [
        item for item in context_audit["evidence"] if item["source_type"] == "owner_fact"
    ]
    assert len(owner_evidence) == 1
    assert owner_evidence[0]["source_scope"] == "client-supplied"
    assert owner_evidence[0]["metadata"]["source"]["filename"] == "owner-answers.json"
    report = ClientReportData.model_validate_json(
        (context_root / "engine" / "client-report-data.json").read_text()
    )
    assert report.project is not None
    assert report.project.project_id == "example"
    assert report.project.version_id == context_version.version_id
    assert report.project.version_number == context_version.version_number
    assert report.project.stage is context_version.stage
    assert report.project.report_status is context_version.report_status
    assert report.project.source_audit_id == context_version.source_audit_id
    assert report.owner_context is not None
    assert report.owner_context.project_id == "example"
    assert report.owner_context.facts[0].fact_id == "fact-canonical-entity"
    assert report.measurement is None


@pytest.mark.parametrize("tampered_artifact", ("audit", "output-manifest"))
def test_update_rejects_untrusted_public_source_bundle_and_deletes_input(
    tmp_path: Path,
    tampered_artifact: str,
) -> None:
    public_manifest = _create(tmp_path)
    public_root = tmp_path / "clients" / "example" / public_manifest.versions[0].relative_path
    if tampered_artifact == "audit":
        audit_path = public_root / "engine" / "audit.json"
        audit_path.write_bytes(audit_path.read_bytes() + b" \n")
    else:
        output_path = public_root / "manifests" / "output-manifest.json"
        payload = json.loads(output_path.read_text())
        payload["artifacts"][0]["relative_path"] = "engine/report-draft.json"
        output_path.write_text(json.dumps(payload), encoding="utf-8")
    owned, intake = _owner_intake(tmp_path)

    with pytest.raises(ValueError, match="artifact|manifest|bundle"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


@pytest.mark.parametrize("invalid_content", ("empty", "metrics", "examples"))
def test_context_update_rejects_empty_or_mixed_intake_without_promotion(
    tmp_path: Path,
    invalid_content: str,
) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)
    if invalid_content == "empty":
        intake["owner_facts"] = []
    elif invalid_content == "metrics":
        intake["metric_series"] = [
            {
                "metric_id": "metric-1",
                "source_id": "owner-answer",
                "metric": "organic sessions",
                "unit": "sessions",
                "state": "UNAVAILABLE",
                "coverage": 0,
                "confidence": 0.5,
            }
        ]
    else:
        intake["cited_examples"] = [
            {
                "example_id": "example-1",
                "source_id": "owner-answer",
                "description": "Public citation",
                "citation": "https://example.com/citation",
                "non_sensitive": True,
            }
        ]

    with pytest.raises(RuntimeError, match="owner-context intake"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


def test_context_update_rejects_unreferenced_source_before_deletion(tmp_path: Path) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)
    unused_content = b"unused owner attachment\n"
    (owned / "unused.json").write_bytes(unused_content)
    sources = list(intake["sources"])
    sources.append(
        SourceArtifactDeclaration(
            source_id="unused-source",
            filename="unused.json",
            sha256=hashlib.sha256(unused_content).hexdigest(),
            byte_count=len(unused_content),
            platform=VisibilitySource.MANUAL,
            report_type="owner-context",
        ).model_dump(mode="json")
    )
    intake["sources"] = sources

    with pytest.raises(RuntimeError, match="referenced by an owner fact"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


@pytest.mark.parametrize("tampered_field", ("value", "approval", "conflicts", "provenance"))
def test_context_bundle_rejects_rehashed_owner_context_semantic_tampering(
    tmp_path: Path,
    tampered_field: str,
) -> None:
    public = _create(tmp_path).versions[0]
    owned, intake = _owner_intake(tmp_path)
    manifest = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )
    context = manifest.versions[-1]
    version_root = tmp_path / "clients" / "example" / context.relative_path
    owner_path = version_root / "aggregates" / "owner-context.json"
    owner_payload = json.loads(owner_path.read_text())
    fact = owner_payload["facts"][0]
    if tampered_field == "value":
        fact["value"] = "Tampered Client"
    elif tampered_field == "approval":
        fact["approval_state"] = FactApprovalState.UNKNOWN.value
    elif tampered_field == "conflicts":
        fact["conflict_ids"] = ["conflict-tampered"]
    else:
        fact["provenance"]["sha256"] = "b" * 64
        owner_payload["sources"][0]["sha256"] = "b" * 64
    owner_path.write_text(json.dumps(owner_payload), encoding="utf-8")

    input_path = version_root / "manifests" / "input-manifest.json"
    input_payload = json.loads(input_path.read_text())
    normalized = input_payload["normalized_inputs"][0]
    normalized["bytes"] = owner_path.stat().st_size
    normalized["sha256"] = _sha256(owner_path)
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")
    _rehash_output_manifest(version_root)

    with pytest.raises(ValueError, match="owner-context.*canonical audit"):
        validate_project_bundle(
            version_root,
            expected_project_id="example",
            expected_version_number=2,
            expected_source_audit_id=public.audit_id,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "evidence-omission",
        "evidence-extra",
        "matrix-omission",
        "matrix-extra",
        "report-matrix-omission",
        "report-evidence-omission",
    ),
)
def test_owner_context_projection_rejects_omissions_and_extras(
    tmp_path: Path,
    mutation: str,
) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)
    manifest = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )
    version_root = tmp_path / "clients" / "example" / manifest.versions[-1].relative_path
    owner_context = OwnerContext.model_validate_json(
        (version_root / "aggregates" / "owner-context.json").read_text()
    )
    audit = AuditRun.model_validate_json((version_root / "engine" / "audit.json").read_text())
    report_payload = json.loads((version_root / "engine" / "client-report-data.json").read_text())
    owner_evidence = next(item for item in audit.evidence if item.collector == "owner-context-v1")
    if mutation == "evidence-omission":
        audit.evidence.remove(owner_evidence)
    elif mutation == "evidence-extra":
        audit.evidence.append(
            owner_evidence.model_copy(update={"evidence_id": "evidence-owner-extra"})
        )
    elif mutation == "matrix-omission":
        audit.entity_consistency_matrix["canonical_entity"]["Example Client"].clear()
    elif mutation == "matrix-extra":
        audit.entity_consistency_matrix["canonical_entity"]["Example Client"].append(
            "evidence-owner-extra"
        )
    elif mutation == "report-matrix-omission":
        report_payload["entity_consistency"]["facts"] = [
            fact
            for fact in report_payload["entity_consistency"]["facts"]
            if fact["fact"] != "canonical_entity"
        ]
    else:
        report_payload["evidence_appendix"] = [
            item
            for item in report_payload["evidence_appendix"]
            if item["evidence_id"] != owner_evidence.evidence_id
        ]
    report = ClientReportData.model_validate(report_payload)

    with pytest.raises(ValueError, match="owner-context.*(audit|report)"):
        project_orchestrator._validate_owner_context_projection(
            owner_context,
            audit=audit,
            client_report=report,
        )


@pytest.mark.parametrize(
    ("intake_overrides", "message"),
    (
        ({"canonical_domain": "other.example"}, "domain"),
        ({"entity": "Different Company"}, "entity"),
    ),
)
def test_update_project_context_rejects_identity_mismatch_and_deletes_input(
    tmp_path: Path,
    intake_overrides: dict[str, str],
    message: str,
) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path, **intake_overrides)

    with pytest.raises(ProjectIdentityError, match=message):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    manifest = ProjectStore(tmp_path / "clients").load("example")
    assert len(manifest.versions) == 1


@pytest.mark.parametrize(
    ("approval_state", "conflict_ids"),
    (
        (FactApprovalState.UNKNOWN, ()),
        (FactApprovalState.APPROVED, ("conflict-entity",)),
    ),
)
def test_update_project_context_keeps_draft_for_unapproved_or_unresolved_facts(
    tmp_path: Path,
    approval_state: FactApprovalState,
    conflict_ids: tuple[str, ...],
) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(
        tmp_path,
        approval_state=approval_state,
        conflict_ids=conflict_ids,
    )

    manifest = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )

    assert manifest.versions[-1].report_status is ReportStatus.CLIENT_CONTEXT_DRAFT


def test_update_rejects_input_report_status_and_deletes_owned_input(tmp_path: Path) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)
    intake["report_status"] = ReportStatus.CLIENT_VALIDATED.value

    with pytest.raises(RuntimeError, match="schema validation"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


def test_update_failure_deletes_input_and_promotes_no_context_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create(tmp_path)
    owned, intake = _owner_intake(tmp_path)

    def fail_compile(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(project_orchestrator, "compile_audit_run", fail_compile)

    with pytest.raises(RuntimeError, match="compile failed"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()
    manifest = ProjectStore(tmp_path / "clients").load("example")
    assert len(manifest.versions) == 1


def test_update_requires_explicit_existing_project_reference_and_deletes_input(
    tmp_path: Path,
) -> None:
    owned, intake = _owner_intake(tmp_path)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        update_project_context(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=owned,
            normalized_intake=intake,
            now=NOW,
        )

    assert not owned.exists()


@pytest.mark.parametrize("policy", [None, "1.0.0", "2.0.0"])
def test_enrich_project_reuses_context_without_crawl_and_retains_visibility_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str | None
) -> None:
    real_compile = orchestrator.compile_audit_run

    def compile_with_source_policy(run, **kwargs):
        if policy is None:
            run.configuration.pop("entity_classification_policy", None)
        else:
            run.configuration["entity_classification_policy"] = policy
        return real_compile(run, **kwargs)

    with monkeypatch.context() as source_patch:
        source_patch.setattr(orchestrator, "compile_audit_run", compile_with_source_policy)
        _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    context_manifest = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    context_version = context_manifest.versions[-1]
    context_root = tmp_path / "clients" / "example" / context_version.relative_path
    owner_before = (context_root / "aggregates" / "owner-context.json").read_bytes()
    context_configuration = json.loads((context_root / "engine" / "audit.json").read_text())[
        "configuration"
    ]
    assert context_configuration.get("entity_classification_policy") == policy
    metric_dir, metric_intake = _visibility_intake(tmp_path)

    def forbidden_crawl(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("visibility enrichment must not crawl")

    monkeypatch.setattr(project_orchestrator, "run_public_audit", forbidden_crawl)
    manifest = enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=metric_dir,
        normalized_intake=metric_intake,
        now=NOW.replace(hour=11),
    )

    assert not metric_dir.exists()
    enriched = manifest.versions[-1]
    assert enriched.version_id == "context-v3"
    assert enriched.source_audit_id == context_version.audit_id
    assert enriched.report_status is context_version.report_status
    root = tmp_path / "clients" / "example" / enriched.relative_path
    assert (root / "aggregates" / "owner-context.json").read_bytes() == owner_before
    visibility = json.loads((root / "aggregates" / "visibility-metrics.json").read_text())
    assert visibility["metrics"][0]["value"] == 0
    assert visibility["metrics"][0]["state"] == "AVAILABLE"
    assert visibility["metrics"][0]["coverage"] == 1
    audit = json.loads((root / "engine" / "audit.json").read_text())
    assert audit["configuration"]["source_audit_id"] == context_version.audit_id
    assert audit["configuration"]["visibility_snapshot_sha256"]
    assert audit["configuration"].get("entity_classification_policy") == policy
    assert {
        key: value for key, value in context_configuration.items() if key != "source_audit_id"
    }.items() <= audit["configuration"].items()
    report = ClientReportData.model_validate_json(
        (root / "engine" / "client-report-data.json").read_text()
    )
    assert report.project is not None
    assert report.project.version_id == enriched.version_id
    assert report.project.report_status is enriched.report_status
    assert report.owner_context is not None
    assert report.owner_context.project_id == "example"
    assert report.measurement is not None
    assert report.measurement.project_id == "example"
    assert report.measurement.metrics[0].metric == "ga4.ai_assistant.sessions"
    assert report.measurement.metrics[0].state is DataState.AVAILABLE
    assert report.measurement.metrics[0].value == 0
    assert report.measurement.metrics[0].coverage == 1
    validate_project_bundle(
        root,
        expected_project_id="example",
        expected_version_number=3,
        expected_source_audit_id=context_version.audit_id,
        expect_owner_context=True,
        expect_visibility_metrics=True,
    )


def test_enrich_project_rejects_public_only_project_and_deletes_input(
    tmp_path: Path,
) -> None:
    _create(tmp_path)
    metric_dir, metric_intake = _visibility_intake(tmp_path)

    with pytest.raises(ValueError, match="context-v2"):
        enrich_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=metric_dir,
            normalized_intake=metric_intake,
            now=NOW,
        )

    assert not metric_dir.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 1


def test_visibility_input_cannot_upgrade_context_report_status(tmp_path: Path) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path, approval_state=FactApprovalState.UNKNOWN)
    context = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    ).versions[-1]
    assert context.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT
    metric_dir, metric_intake = _visibility_intake(tmp_path)

    enriched = enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=metric_dir,
        normalized_intake=metric_intake,
        now=NOW.replace(hour=11),
    ).versions[-1]

    assert enriched.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT
    root = tmp_path / "clients" / "example" / enriched.relative_path
    report = ClientReportData.model_validate_json(
        (root / "engine" / "client-report-data.json").read_text()
    )
    assert report.project is not None
    assert report.project.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT


def test_enrich_project_rejects_persisted_status_inconsistent_with_owner_context(
    tmp_path: Path,
) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    manifest_path = tmp_path / "clients" / "example" / "project.json"
    payload = json.loads(manifest_path.read_text())
    payload["versions"][-1]["report_status"] = ReportStatus.CLIENT_CONTEXT_DRAFT.value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    metric_dir, metric_intake = _visibility_intake(tmp_path)

    with pytest.raises(ValueError, match="report status"):
        enrich_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=metric_dir,
            normalized_intake=metric_intake,
            now=NOW.replace(hour=11),
        )

    assert not metric_dir.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 2


def test_enrich_project_rejects_identity_mismatch_and_deletes_input(
    tmp_path: Path,
) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    metric_dir, metric_intake = _visibility_intake(tmp_path, canonical_domain="other.example")

    with pytest.raises(ProjectIdentityError, match="domain"):
        enrich_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=metric_dir,
            normalized_intake=metric_intake,
            now=NOW.replace(hour=11),
        )

    assert not metric_dir.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 2


@pytest.mark.parametrize("mixed", ("owner", "example", "unused-source"))
def test_enrich_project_rejects_mixed_or_unreferenced_intake_without_promotion(
    tmp_path: Path,
    mixed: str,
) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    metric_dir, payload = _visibility_intake(tmp_path)
    if mixed == "owner":
        payload["owner_facts"] = [
            OwnerFactInput(
                fact_id="fact",
                field=OwnerFactField.MARKET.value,
                value="PL",
                source_id="ga4",
            ).model_dump(mode="json")
        ]
    elif mixed == "example":
        payload["cited_examples"] = [
            {
                "example_id": "example",
                "source_id": "ga4",
                "description": "Example",
                "citation": "example.com",
                "non_sensitive": True,
            }
        ]
    else:
        extra = b"extra"
        (metric_dir / "extra.csv").write_bytes(extra)
        payload["sources"] = [
            *payload["sources"],  # type: ignore[misc]
            SourceArtifactDeclaration(
                source_id="unused",
                filename="extra.csv",
                sha256=hashlib.sha256(extra).hexdigest(),
                byte_count=len(extra),
                platform=VisibilitySource.GOOGLE_ANALYTICS,
                report_type="visibility-export",
            ).model_dump(mode="json"),
        ]

    with pytest.raises(RuntimeError, match="visibility intake"):
        enrich_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=metric_dir,
            normalized_intake=payload,
            now=NOW.replace(hour=11),
        )

    assert not metric_dir.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 2


def test_enrich_failure_deletes_input_and_promotes_no_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    metric_dir, payload = _visibility_intake(tmp_path)
    monkeypatch.setattr(
        project_orchestrator,
        "compile_audit_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        enrich_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=metric_dir,
            normalized_intake=payload,
            now=NOW.replace(hour=11),
        )

    assert not metric_dir.exists()
    assert len(ProjectStore(tmp_path / "clients").load("example").versions) == 2


def test_visibility_bundle_rejects_rehashed_semantic_tampering(tmp_path: Path) -> None:
    _create(tmp_path)
    owner_dir, owner_intake = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_intake,
        now=NOW,
    )
    metric_dir, payload = _visibility_intake(tmp_path)
    manifest = enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=metric_dir,
        normalized_intake=payload,
        now=NOW.replace(hour=11),
    )
    latest = manifest.versions[-1]
    root = tmp_path / "clients" / "example" / latest.relative_path
    visibility_path = root / "aggregates" / "visibility-metrics.json"
    visibility = json.loads(visibility_path.read_text())
    visibility["metrics"][0]["value"] = 99
    visibility_path.write_text(json.dumps(visibility), encoding="utf-8")
    input_path = root / "manifests" / "input-manifest.json"
    inputs = json.loads(input_path.read_text())
    aggregate = next(
        item
        for item in inputs["normalized_inputs"]
        if item["relative_path"] == "aggregates/visibility-metrics.json"
    )
    aggregate["sha256"] = _sha256(visibility_path)
    aggregate["bytes"] = visibility_path.stat().st_size
    input_path.write_text(json.dumps(inputs), encoding="utf-8")
    _rehash_output_manifest(root)

    with pytest.raises(ValueError, match="visibility aggregate"):
        validate_project_bundle(
            root,
            expected_project_id="example",
            expected_version_number=3,
            expected_source_audit_id=latest.source_audit_id,
            expect_owner_context=True,
            expect_visibility_metrics=True,
        )


def test_validate_project_creates_linked_validation_version_with_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _create(tmp_path)
    owner_dir, owner_payload = _owner_intake(
        tmp_path,
        approval_state=FactApprovalState.APPROVED,
        conflict_ids=("conflict-entity",),
    )
    context = update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_payload,
        now=NOW,
    )
    baseline_dir, baseline_payload = _visibility_intake(tmp_path)
    baseline = enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=baseline_dir,
        normalized_intake=baseline_payload,
        now=NOW,
    )
    baseline_version = baseline.versions[-1]
    follow_up_time = datetime(2027, 8, 31, 10, tzinfo=UTC)
    follow_up_dir, follow_up_payload = _visibility_intake_for_period(
        tmp_path,
        period_start=follow_up_time,
        value=5,
    )
    real_run = project_orchestrator.run_public_audit

    def local_public_run(*args: Any, **kwargs: Any):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_public_run)

    validated = validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=follow_up_dir,
        normalized_intake=follow_up_payload,
        implementation_date=date(2026, 9, 1),
        now=follow_up_time,
    )

    assert len(public.versions) == 1
    assert len(context.versions) == 2
    version = validated.versions[-1]
    assert version.version_id == "validation-v4"
    assert version.stage is AuditStage.VALIDATION
    assert version.source_audit_id == baseline_version.audit_id
    assert version.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT
    assert not follow_up_dir.exists()
    root = tmp_path / "clients" / "example" / version.relative_path
    comparison = json.loads((root / "aggregates" / "validation-comparison.json").read_text())
    assert comparison["baseline_audit_id"] == baseline_version.audit_id
    assert comparison["follow_up_audit_id"] == version.audit_id
    assert comparison["causality"] == "NOT_ESTABLISHED"
    assert comparison["metrics"][0]["absolute_delta"] == 5
    assert comparison["metrics"][0]["basis"] == "DIRECT"
    report = ClientReportData.model_validate_json(
        (root / "engine" / "client-report-data.json").read_text()
    )
    assert report.validation_comparison is not None
    assert report.validation_comparison.model_dump(mode="json") == comparison
    validate_project_bundle(
        root,
        expected_project_id="example",
        expected_version_number=4,
        expected_source_audit_id=baseline_version.audit_id,
        expect_owner_context=True,
        expect_visibility_metrics=True,
        expect_validation_comparison=True,
        expected_stage=AuditStage.VALIDATION,
    )


def test_validate_project_records_early_state_without_fabricated_metric_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create(tmp_path)
    real_run = project_orchestrator.run_public_audit

    def local_public_run(*args: Any, **kwargs: Any):
        return real_run(
            *args,
            **kwargs,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    monkeypatch.setattr(project_orchestrator, "run_public_audit", local_public_run)
    validation_time = datetime(2026, 9, 15, 10, tzinfo=UTC)

    manifest = validate_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=validation_time,
    )

    version = manifest.versions[-1]
    assert version.version_id == "validation-v2"
    assert version.report_status is ReportStatus.CLIENT_CONTEXT_DRAFT
    root = tmp_path / "clients" / "example" / version.relative_path
    comparison = json.loads((root / "aggregates" / "validation-comparison.json").read_text())
    assert comparison["timing_state"] == "EARLY"
    assert comparison["timing_warning"]
    assert comparison["metrics"] == []
    assert comparison["causality"] == "NOT_ESTABLISHED"


def test_validation_bundle_rejects_rehashed_baseline_source_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = _create(tmp_path)
    real_run = project_orchestrator.run_public_audit

    def local_public_run(*args: Any, **kwargs: Any):
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
        intake_dir=None,
        normalized_intake=None,
        implementation_date=date(2026, 9, 1),
        now=datetime(2026, 12, 1, 10, tzinfo=UTC),
    )
    version = manifest.versions[-1]
    root = tmp_path / "clients" / "example" / version.relative_path
    tampered_baseline = "audit-tampered"

    comparison_path = root / "aggregates" / "validation-comparison.json"
    comparison_payload = json.loads(comparison_path.read_text())
    comparison_payload["baseline_audit_id"] = tampered_baseline
    comparison_path.write_text(json.dumps(comparison_payload), encoding="utf-8")

    audit_path = root / "engine" / "audit.json"
    audit_payload = json.loads(audit_path.read_text())
    audit_payload["configuration"]["baseline_audit_id"] = tampered_baseline
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

    report_path = root / "engine" / "client-report-data.json"
    report = ClientReportData.model_validate_json(report_path.read_text())
    comparison = ValidationComparisonReportSection.model_validate(comparison_payload)
    tampered_report = report.model_copy(
        update={
            "validation_comparison": comparison,
            "context_digest": report_context_digest(
                report.project,
                report.owner_context,
                report.measurement,
                comparison,
            ),
        }
    )
    report_path.write_text(tampered_report.model_dump_json(indent=2), encoding="utf-8")
    pdf_path = next((root / "report").glob("*.pdf"))
    render_client_report(tampered_report, pdf_path)
    _rehash_output_manifest(root)

    with pytest.raises(ValueError, match="baseline.*source"):
        validate_project_bundle(
            root,
            expected_project_id="example",
            expected_version_number=2,
            expected_source_audit_id=public.versions[-1].audit_id,
            expect_owner_context=False,
            expect_validation_comparison=True,
            expected_stage=AuditStage.VALIDATION,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "metric",
        "implementation-date",
        "observed-at",
        "finding",
        "ai-refs",
        "ai-setup",
        "ai-value",
        "coverage",
        "confidence",
    ),
)
def test_validation_bundle_rejects_rehashed_canonical_comparison_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _create(tmp_path)
    owner_dir, owner_payload = _owner_intake(tmp_path)
    update_project_context(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=owner_dir,
        normalized_intake=owner_payload,
        now=NOW,
    )
    baseline_dir, baseline_payload = _visibility_intake(tmp_path)
    baseline = enrich_project(
        "project:example",
        clients_root=tmp_path / "clients",
        intake_dir=baseline_dir,
        normalized_intake=baseline_payload,
        now=NOW,
    )
    baseline_version = baseline.versions[-1]
    follow_up_time = datetime(2027, 8, 31, 10, tzinfo=UTC)
    follow_up_dir, follow_up_payload = _visibility_intake_for_period(
        tmp_path,
        period_start=follow_up_time,
        value=5,
        coverage=0.2 if tamper in {"coverage", "confidence"} else 1,
        confidence=0.2 if tamper in {"coverage", "confidence"} else 0.9,
    )
    real_run = project_orchestrator.run_public_audit

    def local_public_run(*args: Any, **kwargs: Any):
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
    version = manifest.versions[-1]
    root = tmp_path / "clients" / "example" / version.relative_path
    comparison_path = root / "aggregates" / "validation-comparison.json"
    payload = json.loads(comparison_path.read_text())

    if tamper == "metric":
        metric = payload["metrics"][0]
        metric["follow_up_value"] = 9
        metric["absolute_delta"] = 9
    elif tamper in {"coverage", "confidence"}:
        payload["metrics"][0][tamper] = 0.99
    elif tamper == "implementation-date":
        implementation = date.fromisoformat(payload["implementation_date"]) + timedelta(days=1)
        payload["implementation_date"] = implementation.isoformat()
        payload["validation_target_date"] = (implementation + timedelta(days=90)).isoformat()
    elif tamper == "observed-at":
        payload["observed_at"] = (
            date.fromisoformat(payload["observed_at"]) + timedelta(days=1)
        ).isoformat()
    elif tamper == "finding":
        finding = next(item for item in payload["findings"] if item["follow_up_finding_ids"])
        finding["follow_up_finding_ids"] = ["finding-invented"]
    elif tamper == "ai-refs":
        payload["ai_visibility"]["follow_up_observation_ids"] = ["observation-invented"]
    elif tamper == "ai-setup":
        payload["ai_visibility"]["setup_fingerprint"] = "a" * 64
    else:
        ai = payload["ai_visibility"]
        prompts = ai["follow_up_canonical_prompt_ids"]
        ai.update(
            {
                "state": "AVAILABLE",
                "baseline_value": 0,
                "follow_up_value": 50,
                "absolute_delta": 50,
                "prompt_pack_version": "1.0.0",
                "setup_fingerprint": "a" * 64,
                "baseline_observation_ids": [
                    f"observation-baseline-{index}" for index, _ in enumerate(prompts)
                ],
                "follow_up_observation_ids": [
                    f"observation-follow-up-{index}" for index, _ in enumerate(prompts)
                ],
                "baseline_prompt_ids": prompts,
                "follow_up_prompt_ids": prompts,
                "baseline_measurable_prompt_ids": prompts,
                "follow_up_measurable_prompt_ids": prompts,
                "baseline_canonical_prompt_ids": prompts,
                "limitations": [],
            }
        )

    comparison = ValidationComparison.model_validate(payload)
    comparison_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    report_path = root / "engine" / "client-report-data.json"
    report = ClientReportData.model_validate_json(report_path.read_text())
    report_comparison = ValidationComparisonReportSection.model_validate(
        comparison.model_dump(mode="python")
    )
    tampered_report = report.model_copy(
        update={
            "validation_comparison": report_comparison,
            "context_digest": report_context_digest(
                report.project,
                report.owner_context,
                report.measurement,
                report_comparison,
            ),
        }
    )
    report_path.write_text(tampered_report.model_dump_json(indent=2), encoding="utf-8")
    render_client_report(tampered_report, next((root / "report").glob("*.pdf")))
    _rehash_output_manifest(root)

    with pytest.raises(ValueError, match="canonical validation comparison"):
        validate_project_bundle(
            root,
            expected_project_id="example",
            expected_version_number=version.version_number,
            expected_source_audit_id=baseline_version.audit_id,
            expect_owner_context=True,
            expect_visibility_metrics=True,
            expect_validation_comparison=True,
            expected_stage=AuditStage.VALIDATION,
        )


def test_validate_project_deletes_owned_intake_when_fresh_crawl_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _create(tmp_path)
    intake_dir, payload = _visibility_intake_for_period(
        tmp_path,
        period_start=datetime(2027, 8, 31, 10, tzinfo=UTC),
        value=5,
    )

    def fail_run(*args: Any, **kwargs: Any):
        raise RuntimeError("fresh crawl failed")

    monkeypatch.setattr(project_orchestrator, "run_public_audit", fail_run)

    with pytest.raises(RuntimeError, match="fresh crawl failed"):
        validate_project(
            "project:example",
            clients_root=tmp_path / "clients",
            intake_dir=intake_dir,
            normalized_intake=payload,
            implementation_date=date(2026, 9, 1),
            now=datetime(2027, 8, 31, 10, tzinfo=UTC),
        )

    assert not intake_dir.exists()
    assert ProjectStore(tmp_path / "clients").load("example") == before
