import hashlib
from datetime import UTC, datetime

import pytest

from ai_search_audit.benchmark import canonical_hash
from ai_search_audit.models import DataState, Page, Site
from ai_search_audit.prompt_context import extract_prompt_topics
from ai_search_audit.prompts import PROMPT_PACK_VERSION, generate_prompt_pack, import_observations
from ai_search_audit.scoring import observed_ai_visibility


def test_public_generator_ignores_unbound_legacy_context() -> None:
    prompts = generate_prompt_pack(
        Site(
            domain="studio.example",
            base_url="https://studio.example",
            brand="Example Studio",
            languages=["pl", "en"],
        ),
        entity_type="WebSite",
        category="WebSite",
        location="Office City",
        content_themes=["Odkryj niezapomniane chwile"],
    )
    text = " ".join(prompt.text for prompt in prompts)
    for unbound in ("WebSite", "Office City", "Odkryj niezapomniane chwile"):
        assert unbound not in text
    assert {prompt.pack_version for prompt in prompts} == {"2.0.0"}


def test_prompt_pack_exists_without_grounded_provider() -> None:
    prompts = generate_prompt_pack(
        Site(
            domain="lakeside-hotel.example",
            base_url="https://lakeside-hotel.example",
            brand="Example Lakeside Hotel",
            languages=["pl", "en"],
        ),
        entity_type="hotel",
        location="Example Bay, Poland",
    )
    assert len(prompts) >= 10
    assert {prompt.locale for prompt in prompts} == {"pl", "en"}
    assert all(prompt.pack_version == PROMPT_PACK_VERSION for prompt in prompts)
    assert len({prompt.prompt_id for prompt in prompts}) == len(prompts)
    assert all(prompt.query_themes for prompt in prompts)
    polish = [prompt.text for prompt in prompts if prompt.locale == "pl"]
    english = [prompt.text for prompt in prompts if prompt.locale == "en"]
    assert any("Czym jest" in text for text in polish)
    assert any("What is" in text for text in english)
    assert observed_ai_visibility([]).state is DataState.UNAVAILABLE


def test_prompt_pack_is_domain_neutral_for_non_hospitality_entities() -> None:
    page = Page(
        url="https://flowforge.example/en",
        final_url="https://flowforge.example/en",
        status_code=200,
        language="en",
        content_text="Software development in Europe",
        json_ld=[
            {"@type": "Service", "serviceType": "Software development", "areaServed": "Europe"}
        ],
    )
    prompts = generate_prompt_pack(
        Site(
            domain="flowforge.example",
            base_url="https://flowforge.example",
            brand="FlowForge",
            languages=["en"],
        ),
        topics=extract_prompt_topics("flowforge.example", [page]),
    )
    text = " ".join(prompt.text for prompt in prompts).casefold()
    assert "software development" in text
    assert "europe" in text
    for leaked_assumption in ("premium private", "hotel", "rooms", "hospitality"):
        assert leaked_assumption not in text


def _site(languages=("pl", "en"), brand="Example Studio"):
    return Site(
        domain="studio.example",
        base_url="https://studio.example",
        brand=brand,
        languages=list(languages),
    )


def _service_page(locale="en", category="Software development", area=None, **changes):
    node = {"@type": "Service", "serviceType": category}
    if area is not None:
        node["areaServed"] = area
    values = dict(
        url=f"https://studio.example/{locale}",
        final_url=f"https://studio.example/{locale}",
        status_code=200,
        language=locale,
        content_text=f"{category} {area or ''}",
        json_ld=[node],
    )
    values.update(changes)
    return Page(**values)


_NEUTRAL = {
    "pl": {
        "brand_discovery": "Czym jest Example Studio i z czego jest znana ta marka?",
        "factual_verification": "Jakie informacje o Example Studio można potwierdzić w źródłach?",
        "offer_details": "Co oferuje Example Studio i gdzie opisano szczegóły oferty?",
        "service_area": "Gdzie działa Example Studio i jakie źródła to potwierdzają?",
        "conditions": "Jakie warunki korzystania z oferty Example Studio są publicznie dostępne?",
        "sources": "Które niezależne źródła opisują Example Studio?",
    },
    "en": {
        "brand_discovery": "What is Example Studio, and what is it known for?",
        "factual_verification": "Which facts about Example Studio can be verified from sources?",
        "offer_details": "What does Example Studio offer, and where are the details published?",
        "service_area": "Where does Example Studio operate, and which sources confirm this?",
        "conditions": (
            "Which terms for using the services or products of Example Studio "
            "are publicly available?"
        ),
        "sources": "Which independent sources describe Example Studio?",
    },
}


def test_missing_context_has_exactly_six_neutral_intents_per_locale():
    prompts = generate_prompt_pack(_site())
    assert len(prompts) == 12
    for locale in ("pl", "en"):
        localized = [prompt for prompt in prompts if prompt.locale == locale]
        assert {prompt.intent: prompt.text for prompt in localized} == _NEUTRAL[locale]
        assert all(prompt.query_themes == ["Example Studio"] for prompt in localized)


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        (["PL_pl", "EN-US", "pl", "en_GB", "fr"], ["pl", "en"]),
        ([" en ", "PL-PL", "EN"], ["en", "pl"]),
        (["de", "fr"], ["en"]),
        ([], ["en"]),
    ],
)
def test_requested_supported_locales_are_normalized_and_unique(languages, expected):
    prompts = generate_prompt_pack(_site(languages))
    assert [prompt.locale for prompt in prompts[::6]] == expected
    assert len(prompts) == 6 * len(expected)


def test_brand_falls_back_to_domain():
    prompts = generate_prompt_pack(_site(["en"], brand=None))
    assert all(prompt.target_entities == ["studio.example"] for prompt in prompts)
    assert all("studio.example" in prompt.text for prompt in prompts)


def test_source_bound_locales_replace_only_offer_and_conditions():
    pages = [
        _service_page("pl", "Tworzenie oprogramowania", "Polska"),
        _service_page("en", "Software development", "Europe"),
    ]
    topics = extract_prompt_topics("studio.example", pages)
    snapshot = [topic.model_dump(mode="json") for topic in topics]
    prompts = generate_prompt_pack(_site(), topics=topics)
    expected = {
        "pl": (
            "Tworzenie oprogramowania",
            "Polska",
            "Które firmy oferują usługę „Tworzenie oprogramowania”? Obszar działania: „Polska”.",
            "Porównaj ofertę Example Studio z innymi dostawcami usługi „Tworzenie oprogramowania”.",
        ),
        "en": (
            "Software development",
            "Europe",
            'Which providers offer the service "Software development"? Service area: "Europe".',
            'Compare Example Studio with other providers of "Software development".',
        ),
    }
    assert len(prompts) == 12
    for locale, (category, area, discovery, comparison) in expected.items():
        by_intent = {prompt.intent: prompt for prompt in prompts if prompt.locale == locale}
        assert by_intent["category_discovery"].text == discovery
        assert "Example Studio" not in discovery
        assert by_intent["category_discovery"].target_entities == ["Example Studio"]
        assert by_intent["category_discovery"].query_themes == ["Example Studio", category, area]
        assert by_intent["comparison"].text == comparison
        assert by_intent["comparison"].query_themes == ["Example Studio", category]
        for intent in ("brand_discovery", "factual_verification", "service_area", "sources"):
            assert by_intent[intent].text == _NEUTRAL[locale][intent]
            assert by_intent[intent].query_themes == ["Example Studio"]
        assert set(by_intent) == {
            "brand_discovery",
            "factual_verification",
            "category_discovery",
            "service_area",
            "comparison",
            "sources",
        }
    assert [topic.model_dump(mode="json") for topic in topics] == snapshot


@pytest.mark.parametrize(
    "pages",
    [
        [_service_page("en", "Tworzenie oprogramowania")],
        [_service_page("pl", "Odkryj niezapomniane chwile")],
        [_service_page("en", "bespoke ceramic products")],
        [_service_page("en", "Software development", json_ld=[{"@type": "Store"}])],
        [_service_page("de", "Software development")],
    ],
)
def test_wrong_locale_slogans_unknown_categories_and_stores_stay_neutral(pages):
    topics = extract_prompt_topics("studio.example", pages)
    assert generate_prompt_pack(_site(), topics=topics) == generate_prompt_pack(_site())


def test_single_locale_evidence_is_not_translated_or_shared():
    topics = extract_prompt_topics("studio.example", [_service_page("pl", "Naprawa rowerów")])
    prompts = generate_prompt_pack(_site(), topics=topics)
    assert any(prompt.intent == "category_discovery" for prompt in prompts if prompt.locale == "pl")
    english = [prompt for prompt in prompts if prompt.locale == "en"]
    assert {prompt.intent: prompt.text for prompt in english} == _NEUTRAL["en"]
    assert all(prompt.query_themes == ["Example Studio"] for prompt in english)


def test_multiple_distinct_categories_stay_neutral_without_selecting_a_source():
    topics = extract_prompt_topics(
        "studio.example",
        [_service_page("en", "Software development", "Europe"), _service_page("en", "Web design")],
    )
    assert generate_prompt_pack(_site(), topics=topics) == generate_prompt_pack(_site())


@pytest.mark.parametrize("areas", [(None,), ("Europe", "Poland")])
def test_missing_or_ambiguous_area_does_not_change_unambiguous_category(areas):
    topics = extract_prompt_topics("studio.example", [_service_page(area=area) for area in areas])
    prompts = generate_prompt_pack(_site(["en"]), topics=topics)
    discovery = next(prompt for prompt in prompts if prompt.intent == "category_discovery")
    assert discovery.text == 'Which providers offer the service "Software development"?'
    assert discovery.query_themes == ["Example Studio", "Software development"]


def test_duplicate_category_observations_are_not_ambiguous():
    topics = extract_prompt_topics("studio.example", [_service_page(), _service_page()])
    prompts = generate_prompt_pack(_site(["en"]), topics=topics)
    assert len(topics) == 2
    assert len([prompt for prompt in prompts if prompt.intent == "category_discovery"]) == 1


def test_foreign_service_area_cannot_complete_official_category():
    topics = extract_prompt_topics(
        "studio.example",
        [
            _service_page(),
            _service_page(area="Foreign City", final_url="https://foreign.example/en"),
        ],
    )
    prompts = generate_prompt_pack(_site(["en"]), topics=topics)
    assert any(prompt.intent == "category_discovery" for prompt in prompts)
    assert all(
        "Foreign City" not in prompt.text + " ".join(prompt.query_themes) for prompt in prompts
    )


def test_stable_ids_cover_exact_v2_text_and_full_content_hash_covers_metadata():
    prompts = generate_prompt_pack(_site())
    assert prompts == generate_prompt_pack(_site())
    for prompt in prompts:
        digest = hashlib.sha256(
            f"2.0.0:{prompt.locale}:{prompt.intent}:{prompt.text}".encode()
        ).hexdigest()[:12]
        assert prompt.prompt_id == f"{prompt.intent}-{prompt.locale}-{digest}"
        assert prompt.expected_evidence_needs == [
            "official website",
            "independent authoritative source",
        ]
        assert prompt.suggested_providers == [
            "openai-search",
            "gemini-grounding",
            "perplexity",
            "claude-search",
        ]
    content = [prompt.model_dump(mode="json") for prompt in prompts]
    changed = [dict(item) for item in content]
    changed[0]["query_themes"] = ["Tampered metadata"]
    assert canonical_hash(content) != canonical_hash(changed)
    changed[0] = {**content[0], "text": "Tampered text with unchanged ID"}
    assert canonical_hash(content) != canonical_hash(changed)


def test_interactive_observations_can_be_imported() -> None:
    observations = import_observations(
        [
            {
                "observation_id": "o1",
                "prompt_id": "p1",
                "provider": "interactive-openai-search",
                "observed_at": datetime(2026, 8, 11, tzinfo=UTC).isoformat(),
                "citations": ["https://example.com"],
                "brand_mentioned": True,
                "grounded": True,
            }
        ]
    )
    assert observations[0].provider == "interactive-openai-search"
    assert observed_ai_visibility(observations).value == 100
