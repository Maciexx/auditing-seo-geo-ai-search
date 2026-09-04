import json
from datetime import UTC, date, datetime

import httpx
import pytest

from ai_search_audit import orchestrator, prompts
from ai_search_audit.artifact_policy import saved_prompt_version
from ai_search_audit.diagnostic_models import FrozenPrompt
from ai_search_audit.models import AuditRun
from ai_search_audit.prompt_context import PromptTopic, extract_prompt_topics
from ai_search_audit.prompts import generate_prompt_pack, import_observations
from tests.test_crawler import public_resolver
from tests.test_prompts import _service_page, _site

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


def _validate(run):
    validator = getattr(prompts, "validate_automatic_prompt_pack", None)
    assert callable(validator), "automatic execution requires a source-bound prompt policy gate"
    return validator(run)


def _warnings(pack, report_locale):
    warning_builder = getattr(prompts, "prompt_scope_warnings", None)
    assert callable(warning_builder), "neutral packs must disclose their discovery limitation"
    return warning_builder(pack, report_locale)


def _run(pages=None):
    pages = [_service_page()] if pages is None else pages
    site = _site()
    topics = extract_prompt_topics(site.domain, pages)
    return AuditRun(
        audit_id="audit-prompt-policy",
        site=site,
        audit_engine_version="0.2.0",
        ruleset_version="2026.08.11",
        ruleset_verified_date=date(2026, 8, 11),
        timestamp=NOW,
        pages=pages,
        ai_prompts=generate_prompt_pack(site, topics=topics),
        scores=[],
        configuration={
            "prompt_pack_version": "2.0.0",
            "prompt_context_policy": "1.0.0",
            "prompt_context": [topic.model_dump(mode="json") for topic in topics],
        },
    )


@pytest.mark.parametrize("pages", [None, [], [_service_page(), _service_page()]])
def test_automatic_gate_accepts_exact_recomputed_context_and_pack_without_mutation(pages):
    run = _run(pages)
    before = run.model_dump(mode="json")
    assert _validate(run) is None
    assert run.model_dump(mode="json") == before


@pytest.mark.parametrize("field", ["source_url", "locator", "quote", "value", "locale", "kind"])
def test_automatic_gate_rejects_fabricated_provenance_or_context(field):
    run = _run()
    changes = {
        "source_url": "https://foreign.example/en",
        "locator": "json_ld[999].serviceType",
        "quote": "Fabricated quote",
        "value": "Web design",
        "locale": "pl",
        "kind": "service_area",
    }
    run.configuration["prompt_context"][0][field] = changes[field]
    fake_topics = tuple(
        PromptTopic.model_validate(item) for item in run.configuration["prompt_context"]
    )
    run.ai_prompts = generate_prompt_pack(run.site, topics=fake_topics)
    before = run.model_dump(mode="json")
    with pytest.raises(ValueError, match="context"):
        _validate(run)
    assert run.model_dump(mode="json") == before


@pytest.mark.parametrize("change", ["missing", "reordered", "duplicate"])
def test_automatic_gate_requires_exact_serialized_context_observations(change):
    run = _run([_service_page(), _service_page("pl", "Tworzenie oprogramowania")])
    if change == "missing":
        del run.configuration["prompt_context"]
    elif change == "reordered":
        run.configuration["prompt_context"].reverse()
    else:
        run.configuration["prompt_context"].append(run.configuration["prompt_context"][0])
    with pytest.raises(ValueError, match="context"):
        _validate(run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "Changed text with the original ID"),
        ("target_entities", ["Different brand"]),
        ("query_themes", ["Unbound slogan"]),
        ("suggested_providers", ["invented-provider"]),
        ("expected_evidence_needs", ["No sources"]),
        ("prompt_id", "changed-id"),
        ("intent", "changed-intent"),
    ],
)
def test_automatic_gate_rejects_any_prompt_tampering(field, value):
    run = _run()
    original_id = run.ai_prompts[0].prompt_id
    setattr(run.ai_prompts[0], field, value)
    if field != "prompt_id":
        assert run.ai_prompts[0].prompt_id == original_id
    with pytest.raises(ValueError, match="prompt"):
        _validate(run)


@pytest.mark.parametrize("change", ["missing", "reordered", "extra", "site", "source"])
def test_automatic_gate_rejects_changed_pack_site_or_source(change):
    run = _run()
    if change == "missing":
        run.ai_prompts.pop()
    elif change == "reordered":
        run.ai_prompts.reverse()
    elif change == "extra":
        run.ai_prompts.append(run.ai_prompts[0])
    elif change == "site":
        run.site.brand = "Changed Brand"
    else:
        run.pages[0].content_text = "No longer supports the saved category"
    with pytest.raises(ValueError):
        _validate(run)


@pytest.mark.parametrize("version", ["1.1.0", "9.0.0"])
def test_automatic_gate_rejects_old_or_unknown_pack_versions(version):
    run = _run()
    run.configuration["prompt_pack_version"] = version
    for prompt in run.ai_prompts:
        prompt.pack_version = version
    with pytest.raises(ValueError, match="version"):
        _validate(run)


@pytest.mark.parametrize("policy", [None, "0.0.0", "2.0.0", 1, {}])
def test_automatic_gate_rejects_missing_or_unknown_context_policy(policy):
    run = _run()
    if policy is None:
        del run.configuration["prompt_context_policy"]
    else:
        run.configuration["prompt_context_policy"] = policy
    with pytest.raises(ValueError, match="policy"):
        _validate(run)


def test_old_prompt_snapshot_and_observation_import_remain_readable():
    run = _run()
    run.configuration = {"prompt_pack_version": "1.1.0"}
    legacy = run.ai_prompts[0].model_copy(
        update={
            "pack_version": "1.1.0",
            "prompt_id": "recommendation-en-74b99d1b3e65",
            "locale": "en",
            "intent": "recommendation",
            "text": (
                "Would you recommend Example for someone looking for Hotel in its target market?"
            ),
            "target_entities": ["Example"],
            "query_themes": ["Example", "Hotel", "Private stays", "Example Hotel"],
        }
    )
    run.ai_prompts = [legacy]
    with pytest.raises(ValueError, match="version"):
        _validate(run)
    assert saved_prompt_version(run) == "1.1.0"
    snapshot = FrozenPrompt.model_validate(legacy.model_dump(mode="json"))
    assert snapshot.text == legacy.text and snapshot.pack_version == "1.1.0"
    observation = {
        "observation_id": "legacy-observation",
        "prompt_id": snapshot.prompt_id,
        "provider": "interactive-openai-search",
        "observed_at": NOW.isoformat(),
        "grounded": True,
        "brand_mentioned": True,
        "citations": ["https://example.com/"],
    }
    assert import_observations([observation])[0].prompt_id == snapshot.prompt_id


@pytest.mark.parametrize(
    ("report_locale", "message"),
    [
        (
            "pl",
            "Brak potwierdzonej kategorii (en, pl): "
            "zestaw pytań nie bada odkrywania marki przez kategorię.",
        ),
        (
            "en",
            "No verified category (en, pl): "
            "the prompt pack does not test category-based discovery.",
        ),
    ],
)
def test_missing_discovery_warning_is_one_localized_message_with_sorted_locales(
    report_locale, message
):
    pack = generate_prompt_pack(_site())
    assert _warnings(pack, report_locale) == [message]


def test_scope_warning_only_lists_locales_without_category_discovery():
    topics = extract_prompt_topics("studio.example", [_service_page("pl", "Naprawa rowerów")])
    pack = generate_prompt_pack(_site(), topics=topics)
    assert _warnings(pack, "en") == [
        "No verified category (en): the prompt pack does not test category-based discovery."
    ]
    both = topics + extract_prompt_topics("studio.example", [_service_page()])
    assert _warnings(generate_prompt_pack(_site(), topics=both), "en") == []


def _transport(with_category):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path not in ("/", "/en"):
            return httpx.Response(404)
        locale = "en" if request.url.path == "/en" else "pl"
        category = "Software development" if locale == "en" else "Tworzenie oprogramowania"
        schema = [
            {
                "@type": "Organization",
                "name": "Canonical Studio",
                "address": {"addressLocality": "Office City"},
            }
        ]
        if with_category:
            schema.append({"@type": "Service", "serviceType": category})
        html = (
            f'<html lang="{locale}"><head><title>Canonical Studio</title>'
            f'<link rel="canonical" href="https://studio.example{request.url.path}">'
            f'<script type="application/ld+json">{json.dumps(schema)}</script></head>'
            "<body><h1>Odkryj niezapomniane chwile</h1><h2>Unbound heading</h2>"
            f"<p>{category if with_category else 'Unknown products'}</p>"
            '<a href="/en">English</a></body></html>'
        )
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("with_category", [False, True])
def test_orchestration_binds_context_and_canonical_brand_before_compilation(
    tmp_path, monkeypatch, with_category
):
    compiled = []

    def capture(run, **kwargs):
        compiled.append(run.model_dump(mode="json"))
        return run

    monkeypatch.setattr(orchestrator, "compile_audit_run", capture)
    run = orchestrator.run_public_audit(
        "studio.example",
        output_dir=tmp_path,
        max_pages=2,
        now=NOW,
        crawler_transport=_transport(with_category),
        crawler_resolver=public_resolver,
    )
    assert len(compiled) == 1
    assert run.site.brand == "Canonical Studio"
    assert run.configuration["entity_classification_policy"] == "2.0.0"
    assert run.configuration["prompt_pack_version"] == "2.0.0"
    assert run.configuration["prompt_context_policy"] == "1.0.0"
    topics = extract_prompt_topics(run.site.domain, run.pages)
    assert bool(topics) is with_category
    assert run.configuration["prompt_context"] == [
        topic.model_dump(mode="json") for topic in topics
    ]
    assert run.ai_prompts == generate_prompt_pack(run.site, topics=topics)
    assert all(prompt.target_entities == ["Canonical Studio"] for prompt in run.ai_prompts)
    assert all("Office City" not in prompt.text for prompt in run.ai_prompts)
    assert all(
        "Odkryj" not in prompt.text and "Unbound" not in prompt.text for prompt in run.ai_prompts
    )
    assert compiled[0]["configuration"] == run.configuration
    _validate(run)


@pytest.mark.parametrize(
    ("title", "expected_brand"),
    [("Example Studio", "Example Studio"), ("Odkryj niezapomniane chwile", "Example-Studio")],
)
def test_schema_free_audit_keeps_root_slogans_out_of_english_prompts(
    tmp_path, monkeypatch, title, expected_brand
):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path not in ("/", "/en"):
            return httpx.Response(404)
        locale = "en" if request.url.path == "/en" else "pl"
        html = (
            f'<html lang="{locale}"><head><title>{title}</title></head>'
            '<body><h1>Odkryj niezapomniane chwile</h1><a href="/en">English</a></body></html>'
        )
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    monkeypatch.setattr(orchestrator, "compile_audit_run", lambda run, **_: run)
    run = orchestrator.run_public_audit(
        "example-studio.example",
        output_dir=tmp_path,
        max_pages=2,
        now=NOW,
        crawler_transport=httpx.MockTransport(handler),
        crawler_resolver=public_resolver,
    )

    assert {page.language for page in run.pages} == {"en", "pl"}
    assert all(not page.json_ld for page in run.pages)
    english_prompts = [prompt for prompt in run.ai_prompts if prompt.locale == "en"]
    assert english_prompts
    assert all("Odkryj niezapomniane chwile" not in prompt.text for prompt in english_prompts)
    assert run.site.brand == expected_brand
    assert all(prompt.target_entities == [expected_brand] for prompt in run.ai_prompts)
    _validate(run)


@pytest.mark.parametrize("report_locale", ["pl", "en"])
def test_scope_warning_is_before_compile_without_findings_or_score_deduction(
    tmp_path, monkeypatch, report_locale
):
    compiled = []

    def capture(run, **kwargs):
        compiled.append(run.model_dump(mode="json"))
        return run

    monkeypatch.setattr(orchestrator, "compile_audit_run", capture)
    options = dict(
        domain="studio.example",
        output_dir=tmp_path,
        max_pages=2,
        now=NOW,
        report_locale=report_locale,
        crawler_resolver=public_resolver,
    )
    run = orchestrator.run_public_audit(**options, crawler_transport=_transport(False))
    expected = (
        "Brak potwierdzonej kategorii (en, pl): "
        "zestaw pytań nie bada odkrywania marki przez kategorię."
        if report_locale == "pl"
        else "No verified category (en, pl): "
        "the prompt pack does not test category-based discovery."
    )
    assert run.warnings.count(expected) == 1
    assert compiled[0]["warnings"].count(expected) == 1
    monkeypatch.setattr(orchestrator, "prompt_scope_warnings", lambda *_: [])
    without_warning = orchestrator.run_public_audit(**options, crawler_transport=_transport(False))
    assert without_warning.warnings == [warning for warning in run.warnings if warning != expected]
    assert run.findings == without_warning.findings
    assert run.scores == without_warning.scores
