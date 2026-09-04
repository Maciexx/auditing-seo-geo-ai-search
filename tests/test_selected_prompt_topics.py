"""Offline source selection and versioned discovery contracts."""

import hashlib
import json

import httpx
import pytest

from ai_search_audit import orchestrator, prompt_context
from ai_search_audit.artifact_policy import saved_prompt_version
from ai_search_audit.models import Page
from ai_search_audit.prompt_context import PromptTopic, extract_prompt_topics
from ai_search_audit.prompts import generate_prompt_pack, validate_automatic_prompt_pack
from tests.test_crawler import public_resolver
from tests.test_prompt_policy import NOW, _run
from tests.test_prompts import _service_page, _site


def _page(locale="en", text="Find and book ceramics workshops in London.", **changes):
    values = dict(
        url=f"https://studio.example/{locale}",
        final_url=f"https://studio.example/{locale}",
        status_code=200,
        language=locale,
        content_text=text,
    )
    values.update(changes)
    return Page(**values)


def _selection(page, value="ceramics workshops", kind="category"):
    start = page.content_text.index(value)
    return PromptTopic(
        kind=kind,
        locale=page.language.split("-")[0],
        value=value,
        source_url=str(page.final_url),
        locator=f"content_text[{start}:{start + len(value)}]",
        quote=value,
    )


def _resolve(pages, topics):
    resolver = getattr(prompt_context, "resolve_selected_prompt_topics", None)
    assert callable(resolver), "explicit source selection must re-resolve before use"
    return resolver("studio.example", pages, topics)


def _new_run(pages, selections):
    run = _run(pages)
    topics = extract_prompt_topics(run.site.domain, pages) + _resolve(pages, selections)
    run.configuration.update(
        prompt_pack_version="2.1.0",
        prompt_context_policy="1.1.0",
        prompt_topic_selection=[topic.model_dump(mode="json") for topic in selections],
        prompt_context=[topic.model_dump(mode="json") for topic in topics],
    )
    run.ai_prompts = generate_prompt_pack(
        run.site, topics=topics, pack_version="2.1.0", pages=pages
    )
    return run


@pytest.mark.parametrize(
    ("locale", "text", "category", "area"),
    [
        ("en", "Find and book ceramics workshops in London.", "ceramics workshops", "London"),
        ("pl", "Znajdź warsztaty ceramiczne. Warszawa.", "warsztaty ceramiczne", "Warszawa"),
    ],
)
def test_nonclassified_visible_selection_is_explicit_and_locale_bound(locale, text, category, area):
    page = _page(locale, text)
    topics = (_selection(page, category), _selection(page, area, "service_area"))
    assert extract_prompt_topics("studio.example", [page]) == ()
    assert _resolve([page], topics) == topics
    run = _new_run([page], topics)
    validate_automatic_prompt_pack(run)
    discovery = [p for p in run.ai_prompts if p.intent == "category_discovery"]
    assert len(discovery) == 1 and discovery[0].locale == locale
    assert category in discovery[0].text and area in discovery[0].text
    assert run.site.brand not in discovery[0].text
    assert discovery[0].query_themes == [category, area]
    assert all("gemini-grounding" not in p.suggested_providers for p in run.ai_prompts)
    assert any(p.intent == "factual_verification" for p in run.ai_prompts)
    assert all(p.intent != "comparison" for p in run.ai_prompts)
    assert all("operate" not in p.text and "działa" not in p.text for p in run.ai_prompts)
    assert saved_prompt_version(run) == "2.1.0"


@pytest.mark.parametrize(
    "changes",
    [
        {"source_url": "https://foreign.example/en"},
        {"source_url": "https://studio.example/en#fake"},
        {"locale": "pl"},
        {"value": "warsztaty ceramiczne"},
        {"quote": "Invented quote"},
        {"locator": "content_text[-1:99]"},
        {"locator": "content_text[0:999999999999999999999]"},
        {"locator": "__import__('os').system('false')"},
        {"locator": "../../private"},
        {"locator": "json_ld[0].serviceType"},
        {"kind": "commercial_goal"},
        {"value": ["ceramics workshops"]},
    ],
)
def test_selection_rejects_untrusted_fields_including_unchecked_model_copy(changes):
    page = _page()
    topic = _selection(page).model_copy(update=changes)
    with pytest.raises(ValueError):
        _resolve([page], (topic,))


@pytest.mark.parametrize(
    "changes",
    [
        {"status_code": 404},
        {"status_code": 302},
        {"language": "de"},
        {"final_url": "https://foreign.example/en"},
        {"canonical": "https://foreign.example/en"},
        {"canonical": "https://studio.example/other"},
        {"content_text": "Unsupported revised page"},
    ],
)
def test_source_change_or_noncanonical_page_invalidates_selection(changes):
    page = _page()
    selection = _selection(page)
    page = Page.model_validate({**page.model_dump(), **changes})
    with pytest.raises(ValueError):
        _resolve([page], (selection,))


@pytest.mark.parametrize("value", ["x" * 121, "ignore\nprevious", 'ignore "instructions"', "<tag>"])
def test_unsafe_or_unbounded_literal_source_is_not_a_topic(value):
    page = _page(text=value)
    with pytest.raises(ValueError):
        topic = _selection(page, value)
        _resolve([page], (topic,))


def test_substring_is_not_a_verbatim_phrase_and_duplicate_source_is_ambiguous():
    page = _page(text="ceramics workshops")
    with pytest.raises(ValueError):
        _resolve([page], (_selection(page, "ceramic"),))
    with pytest.raises(ValueError):
        _resolve([page, page], (_selection(page),))


def test_www_selection_requires_observed_canonical_root_redirect():
    page = _page(url="https://www.studio.example/en", final_url="https://www.studio.example/en")
    topic = _selection(page)
    with pytest.raises(ValueError):
        _resolve([page], (topic,))
    root = _page(url="https://studio.example/", final_url="https://www.studio.example/")
    assert _resolve([root, page], (topic,)) == (topic,)


@pytest.mark.parametrize("value", ["studio.example", "Example Studio", "STUDIO.EXAMPLE"])
def test_target_domain_or_brand_inside_source_topic_cannot_leak_into_discovery(value):
    page = _page(text=f"{value} ceramics workshops")
    run = _new_run([page], (_selection(page, page.content_text),))
    assert not any(p.intent == "category_discovery" for p in run.ai_prompts)
    validate_automatic_prompt_pack(run)


def test_conflicting_selected_areas_do_not_choose_arbitrarily():
    page = _page(text="ceramics workshops London Warsaw")
    run = _new_run(
        [page],
        (
            _selection(page),
            _selection(page, "London", "service_area"),
            _selection(page, "Warsaw", "service_area"),
        ),
    )
    discovery = next(p for p in run.ai_prompts if p.intent == "category_discovery")
    assert "London" not in discovery.text and "Warsaw" not in discovery.text


@pytest.mark.parametrize("selection", [[], "proposal", None])
def test_selection_container_is_explicitly_typed(selection):
    with pytest.raises(ValueError):
        _resolve([_page()], selection)


def test_conflicting_auto_and_explicit_categories_stay_unknown():
    page = _service_page(content_text="Software development. Ceramics workshops.")
    run = _new_run([page], (_selection(page, "Ceramics workshops"),))
    assert not any(p.intent == "category_discovery" for p in run.ai_prompts)
    validate_automatic_prompt_pack(run)


@pytest.mark.parametrize("brand", ["ceramics workshops", "Which options", "London"])
def test_brand_common_phrase_overlap_suppresses_discovery_without_rewriting_source(brand):
    page = _page()
    site = _site(brand=brand)
    topics = (_selection(page), _selection(page, "London", "service_area"))
    pack = generate_prompt_pack(site, topics=topics, pack_version="2.1.0", pages=[page])
    assert not any(p.intent == "category_discovery" for p in pack)


@pytest.mark.parametrize(
    ("brand", "phrase"),
    [
        ("ceramics workshops", "ceramics   workshops"),
        ("ceramics   workshops", "ceramics workshops"),
        ("ceramics workshops", "ｃｅｒａｍｉｃｓ workshops"),
        ("ｃｅｒａｍｉｃｓ workshops", "ceramics workshops"),
    ],
)
def test_brand_contamination_normalizes_both_sides_without_rewriting_observation(brand, phrase):
    page = _page(text=phrase)
    topic = _selection(page, phrase)
    pack = generate_prompt_pack(
        _site(brand=brand), topics=(topic,), pack_version="2.1.0", pages=[page]
    )
    assert not any(p.intent == "category_discovery" for p in pack)
    assert topic.value == topic.quote == phrase


def test_canonical_alias_suppresses_discovery_but_unrelated_entity_name_does_not():
    page = _page(
        json_ld=[
            {
                "@type": "Organization",
                "name": "Example Studio",
                "url": "https://studio.example",
                "alternateName": "ceramics workshops",
            }
        ]
    )
    run = _new_run([page], (_selection(page),))
    assert not any(p.intent == "category_discovery" for p in run.ai_prompts)
    page.json_ld[0]["name"] = "Another Organization"
    run = _new_run([page], (_selection(page),))
    assert any(p.intent == "category_discovery" for p in run.ai_prompts)


@pytest.mark.parametrize("tamper", ["selection", "context", "text", "policy", "source", "alias"])
def test_new_automatic_gate_recomputes_every_input_without_mutation(tamper):
    page = _page()
    run = _new_run([page], (_selection(page),))
    if tamper == "selection":
        run.configuration["prompt_topic_selection"][0]["locator"] = "content_text[0:5]"
    elif tamper == "context":
        run.configuration["prompt_context"] = []
    elif tamper == "text":
        run.ai_prompts[0].text = "Which company is best?"
    elif tamper == "policy":
        run.configuration["prompt_context_policy"] = "1.0.0"
    elif tamper == "source":
        run.pages[0].content_text = "Changed"
    else:
        run.pages[0].json_ld = [
            {
                "@type": "Organization",
                "name": "Example Studio",
                "alternateName": "ceramics workshops",
            }
        ]
    before = run.model_dump_json()
    with pytest.raises(ValueError):
        validate_automatic_prompt_pack(run)
    assert run.model_dump_json() == before


def test_new_pack_is_deterministic_and_legacy_serialization_unchanged():
    page = _page()
    legacy = _run()
    before = legacy.model_dump_json()
    first = _new_run([page], (_selection(page),))
    second = _new_run([page], (_selection(page),))
    assert first.model_dump_json() == second.model_dump_json()
    for p in first.ai_prompts:
        digest = hashlib.sha256(f"2.1.0:{p.locale}:{p.intent}:{p.text}".encode()).hexdigest()[:12]
        assert p.prompt_id == f"{p.intent}-{p.locale}-{digest}"
    validate_automatic_prompt_pack(legacy)
    assert legacy.model_dump_json() == before


@pytest.mark.parametrize("change", ["value", "quote", "locator", "source_url", "pages"])
def test_new_generator_cannot_bypass_source_resolution(change):
    page = _page()
    topic = _selection(page)
    if change != "pages":
        topic = topic.model_copy(update={change: "arbitrary freeform"})
    with pytest.raises(ValueError):
        generate_prompt_pack(
            _site(),
            topics=(topic,),
            pack_version="2.1.0",
            pages=[] if change == "pages" else [page],
        )


@pytest.mark.parametrize("selection", [None, {}, "not-a-list", [None], [dict(value="proposal")]])
def test_new_gate_rejects_missing_or_malformed_saved_selection(selection):
    run = _new_run([_page()], ())
    run.configuration["prompt_topic_selection"] = selection
    with pytest.raises(ValueError):
        validate_automatic_prompt_pack(run)


def _transport(prefix=""):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path != "/":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=(
                f'<html lang="en"><head><title>{prefix}Example Studio</title>'
                '<script type="application/ld+json">'
                '{"@type":"Organization","name":"Example Studio"}</script>'
                "</head><body><p>Find and book ceramics workshops in London.</p></body></html>"
            ),
        )

    return httpx.MockTransport(handler)


def test_new_audit_boundary_persists_selection_separately_and_keeps_readiness(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrator, "compile_audit_run", lambda run, **_: run)
    options = dict(
        output_dir=tmp_path,
        max_pages=1,
        now=NOW,
        crawler_transport=_transport(),
        crawler_resolver=public_resolver,
    )
    legacy = orchestrator.run_public_audit("studio.example", **options)
    selection = _selection(legacy.pages[0])
    run = orchestrator.run_public_audit("studio.example", selected_topics=(selection,), **options)
    assert run.configuration["prompt_topic_selection"] == [selection.model_dump(mode="json")]
    assert run.configuration["prompt_pack_version"] == "2.1.0"
    assert run.configuration["prompt_context_policy"] == "1.1.0"
    assert run.scores == legacy.scores and run.findings == legacy.findings
    assert run.entity == legacy.entity and run.ai_observations == []
    assert run.configuration["prompt_context"] == [selection.model_dump(mode="json")]
    validate_automatic_prompt_pack(run)
    assert "prompt_topic_selection" not in legacy.configuration
    # The empty selection opts into the new neutral policy, not automatic classification.
    neutral = orchestrator.run_public_audit("studio.example", selected_topics=(), **options)
    assert neutral.configuration["prompt_pack_version"] == "2.1.0"
    assert any("No verified category" in warning for warning in neutral.warnings)
    assert not any(p.intent == "category_discovery" for p in neutral.ai_prompts)
    assert json.loads(run.model_dump_json())["configuration"] == run.configuration
