from importlib.util import find_spec

import pytest
from pydantic import ValidationError

from ai_search_audit.models import Page


def _context():
    assert find_spec("ai_search_audit.prompt_context") is not None
    from ai_search_audit import prompt_context

    return prompt_context


def _page(phrase="Software development", *, language="en", path="/en", **changes):
    values = dict(
        url=f"https://studio.example{path}",
        final_url=f"https://studio.example{path}",
        status_code=200,
        language=language,
        content_text=phrase,
        json_ld=[{"@type": "Service", "serviceType": phrase}],
    )
    values.update(changes)
    return Page(**values)


def test_paired_locales_keep_independent_phrases_and_sources():
    pages = [
        _page("Tworzenie oprogramowania", language="pl", path="/pl"),
        _page(h1=["Discover unforgettable experiences"]),
    ]
    snapshots = [page.model_dump(mode="json") for page in pages]

    topics = _context().extract_prompt_topics("studio.example", pages)

    assert isinstance(topics, tuple)
    assert [topic.model_dump() for topic in topics] == [
        dict(
            kind="category",
            locale="pl",
            value="Tworzenie oprogramowania",
            source_url="https://studio.example/pl",
            locator="json_ld[0].serviceType",
            quote="Tworzenie oprogramowania",
        ),
        dict(
            kind="category",
            locale="en",
            value="Software development",
            source_url="https://studio.example/en",
            locator="json_ld[0].serviceType",
            quote="Software development",
        ),
    ]
    assert [page.model_dump(mode="json") for page in pages] == snapshots
    assert _context().extract_prompt_topics("studio.example", pages) == topics


@pytest.mark.parametrize(
    ("locale", "phrase"),
    [
        ("pl", "tworzenie oprogramowania"),
        ("pl", "projektowanie stron internetowych"),
        ("pl", "usługi księgowe"),
        ("pl", "usługi tłumaczeniowe"),
        ("pl", "naprawa rowerów"),
        ("pl", "usługi stomatologiczne"),
        ("en", "software development"),
        ("en", "web design"),
        ("en", "accounting services"),
        ("en", "translation services"),
        ("en", "bicycle repair"),
        ("en", "dental services"),
    ],
)
def test_each_reviewed_category_preserves_locale_quote_and_source(locale, phrase):
    page = _page(phrase, language=locale, path=f"/{locale}/offer")

    (topic,) = _context().extract_prompt_topics("studio.example", [page])

    assert (topic.kind, topic.locale, topic.value, topic.quote) == (
        "category",
        locale,
        phrase,
        phrase,
    )
    assert topic.source_url == str(page.final_url)
    assert topic.locator == "json_ld[0].serviceType"


@pytest.mark.parametrize("language", ["en-GB", "EN-us", "en_US", " en "])
def test_known_language_tags_are_normalized(language):
    (topic,) = _context().extract_prompt_topics("studio.example", [_page(language=language)])
    assert topic.locale == "en"


@pytest.mark.parametrize("language", [None, "", "de", "fr", "unknown", "english"])
def test_unknown_languages_do_not_supply_topics(language):
    assert _context().extract_prompt_topics("studio.example", [_page(language=language)]) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"final_url": "https://foreign.example/en"},
        {"final_url": "https://shop.studio.example/en"},
        {"final_url": "https://studio.example.evil.test/en"},
        {"status_code": 404},
        {"status_code": 201},
        {"content_text": "Unrelated content"},
        {"content_text": "software development"},
        {"content_text": "", "h1": ["Software development"]},
        {"json_ld": []},
    ],
)
def test_page_requires_official_successful_visible_evidence(changes):
    assert _context().extract_prompt_topics("studio.example", [_page(**changes)]) == ()


@pytest.mark.parametrize(
    ("phrase", "language"),
    [
        ("Tworzenie oprogramowania", "en"),
        ("Software development", "pl"),
        ("Odkryj niezapomniane chwile", "pl"),
        ("Example Studio", "en"),
        ("WebSite", "en"),
        ("Service", "en"),
        ("software development and web design", "en"),
    ],
)
def test_unreviewed_or_wrong_locale_phrases_are_not_categories(phrase, language):
    page = _page(phrase, language=language, h1=[phrase])
    assert _context().extract_prompt_topics("studio.example", [page]) == ()


@pytest.mark.parametrize(
    "node",
    [
        {"@type": "Organization", "serviceType": "Software development"},
        {"@type": "Service", "serviceType": ["Software development"]},
        {"@type": "Service", "serviceType": {"name": "Software development"}},
        {"@type": "Service"},
        {"@type": "Organization", "address": {"addressLocality": "Office City"}},
        {
            "@type": "WebPage",
            "mainEntity": {"@type": "Service", "serviceType": "Software development"},
        },
    ],
)
def test_only_literal_service_type_on_top_level_or_graph_is_evidence(node):
    page = _page(content_text="Software development Office City", json_ld=[node])
    assert _context().extract_prompt_topics("studio.example", [page]) == ()


def test_graph_nodes_and_type_lists_preserve_exact_locators():
    page = _page(
        json_ld=[
            {"@type": "WebSite"},
            {
                "@graph": [
                    None,
                    {"@type": ["Thing", "Service"], "serviceType": "Software development"},
                ]
            },
        ]
    )
    (topic,) = _context().extract_prompt_topics("studio.example", [page])
    assert topic.locator == "json_ld[1].@graph[1].serviceType"


def test_whitespace_normalization_preserves_phrase_case():
    page = _page("  Software\t development ", content_text="Our Software\n  development offer.")
    (topic,) = _context().extract_prompt_topics("studio.example", [page])
    assert topic.value == topic.quote == "Software development"


def test_multiple_categories_and_distinct_source_observations_are_retained():
    page = _page(
        content_text="Software development and web design",
        json_ld=[
            {"@type": "Service", "serviceType": "Software development"},
            {"@type": "Service", "serviceType": "web design"},
            {"@type": "Service", "serviceType": "Software development"},
        ],
    )
    topics = _context().extract_prompt_topics("studio.example", [page, _page(path="/en/other")])
    assert [topic.value for topic in topics] == [
        "Software development",
        "web design",
        "Software development",
        "Software development",
    ]
    assert len({(topic.source_url, topic.locator) for topic in topics}) == 4


@pytest.mark.parametrize("domain", ["studio.example", "www.studio.example"])
def test_observed_root_redirect_establishes_only_apex_www_alias(domain):
    alias = "www.studio.example" if domain == "studio.example" else "studio.example"
    page = _page(final_url=f"https://{alias}/en", url=f"https://{alias}/en")
    root = _page(url=f"https://{domain}/", final_url=f"https://{alias}/", json_ld=[])
    assert _context().extract_prompt_topics(domain, [page]) == ()
    (topic,) = _context().extract_prompt_topics(domain, [page, root])
    assert topic.source_url == f"https://{alias}/en"


@pytest.mark.parametrize("changes", [{"status_code": 404}, {"url": "https://studio.example/en"}])
def test_unobserved_root_alias_cannot_supply_evidence(changes):
    page = _page(url="https://www.studio.example/en", final_url="https://www.studio.example/en")
    root = _page(path="/", final_url="https://www.studio.example/", json_ld=[], **changes)
    assert _context().extract_prompt_topics("studio.example", [root, page]) == ()


def test_unknown_product_category_has_no_fake_service_context():
    page = _page(
        "handmade stationery",
        json_ld=[{"@type": "Product", "category": "handmade stationery", "offers": {"price": 20}}],
    )
    assert _context().extract_prompt_topics("studio.example", [page]) == ()


def _topic_values():
    return dict(
        kind="category",
        locale="en",
        value="Software development",
        source_url="https://studio.example/en",
        locator="json_ld[0].serviceType",
        quote="Software development",
    )


def test_topic_is_frozen_and_has_no_extra_fields():
    model = _context().PromptTopic
    topic = model(**_topic_values())
    with pytest.raises(ValidationError, match="frozen"):
        topic.value = "web design"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model(**_topic_values(), provider="unbound")
    assert model.model_validate_json(topic.model_dump_json()) == topic


@pytest.mark.parametrize(
    "changes",
    [
        {"value": ""},
        {"value": "x" * 121},
        {"locale": "de"},
        {"kind": "office"},
    ],
)
def test_topic_rejects_invalid_value_locale_and_kind(changes):
    with pytest.raises(ValidationError):
        _context().PromptTopic(**(_topic_values() | changes))


def test_topic_value_allows_exact_limit():
    topic = _context().PromptTopic(**(_topic_values() | {"value": "x" * 120}))
    assert len(topic.value) == 120


@pytest.mark.parametrize(
    ("locale", "phrase", "area"),
    [("pl", "Tworzenie oprogramowania", "Polska"), ("en", "Software development", "Poland")],
)
def test_service_area_uses_same_recognized_service_and_locale(locale, phrase, area):
    page = _page(
        phrase,
        language=locale,
        path=f"/{locale}",
        content_text=f"{phrase}. {area}.",
        json_ld=[{"@type": "Service", "serviceType": phrase, "areaServed": area}],
    )
    category, topic = _context().extract_prompt_topics("studio.example", [page])
    assert category.kind == "category"
    assert topic.model_dump() == dict(
        kind="service_area",
        locale=locale,
        value=area,
        source_url=f"https://studio.example/{locale}",
        locator="json_ld[0].areaServed",
        quote=area,
    )


def test_two_categories_and_two_service_areas_remain_observations():
    page = _page(
        content_text="Software development in Poland. web design in Germany.",
        json_ld=[
            {
                "@graph": [
                    {
                        "@type": "Service",
                        "serviceType": "Software development",
                        "areaServed": "Poland",
                    },
                    {
                        "@type": ["Service", "Thing"],
                        "serviceType": "web design",
                        "areaServed": "Germany",
                    },
                ]
            }
        ],
    )
    topics = _context().extract_prompt_topics("studio.example", [page])
    assert [(topic.kind, topic.value) for topic in topics] == [
        ("category", "Software development"),
        ("service_area", "Poland"),
        ("category", "web design"),
        ("service_area", "Germany"),
    ]
    assert [topic.locator for topic in topics if topic.kind == "service_area"] == [
        "json_ld[0].@graph[0].areaServed",
        "json_ld[0].@graph[1].areaServed",
    ]


@pytest.mark.parametrize(
    "area",
    [
        "",
        "   ",
        "x" * 121,
        "{Poland}",
        "Poland}",
        "<Poland>",
        "Poland>",
        "Poland\nGermany",
        "Poland\rGermany",
        "Poland\r\nGermany",
        "Poland\u2028Germany",
        "Poland\u2029Germany",
        "Poland\x85Germany",
        "Poland\vGermany",
        "Poland\fGermany",
        "Poland\x1cGermany",
        "Poland\x1dGermany",
        "Poland\x1eGermany",
        ["Poland"],
        {"@type": "Country", "name": "Poland"},
        None,
    ],
)
def test_unsafe_nonliteral_or_unbounded_area_is_rejected_before_whitespace_collapse(area):
    visible = " ".join(area.split()) if isinstance(area, str) else "Poland"
    page = _page(
        content_text=f"Software development. {visible}",
        json_ld=[{"@type": "Service", "serviceType": "Software development", "areaServed": area}],
    )
    topics = _context().extract_prompt_topics("studio.example", [page])
    assert [topic.kind for topic in topics] == ["category"]


@pytest.mark.parametrize("area", ["Poland", "x" * 120, "  New\t York  "])
def test_safe_visible_area_preserves_phrase_after_whitespace_normalization(area):
    normalized = " ".join(area.split())
    page = _page(
        content_text=f"Software development in {normalized}",
        json_ld=[{"@type": "Service", "serviceType": "Software development", "areaServed": area}],
    )
    _, topic = _context().extract_prompt_topics("studio.example", [page])
    assert topic.value == topic.quote == normalized


def test_area_must_be_visible_on_the_same_page():
    page = _page(
        json_ld=[
            {"@type": "Service", "serviceType": "Software development", "areaServed": "Poland"}
        ]
    )
    other = _page(path="/pl", language="pl", content_text="Poland", json_ld=[])
    topics = _context().extract_prompt_topics("studio.example", [page, other])
    assert [topic.kind for topic in topics] == ["category"]


@pytest.mark.parametrize(
    "node",
    [
        {"@type": "Service", "areaServed": "Poland"},
        {"@type": "Service", "serviceType": "Unreviewed consulting", "areaServed": "Poland"},
        {"@type": "Organization", "areaServed": "Poland"},
        {
            "@type": "Service",
            "serviceType": "Software development",
            "address": {"addressLocality": "Poland"},
        },
        {
            "@type": "Service",
            "serviceType": "Software development",
            "provider": {"areaServed": "Poland"},
        },
    ],
)
def test_unrecognized_other_node_and_office_areas_do_not_bind_to_recognized_service(node):
    page = _page(
        content_text="Software development. Unreviewed consulting. Poland.",
        json_ld=[{"@type": "Service", "serviceType": "Software development"}, node],
    )
    topics = _context().extract_prompt_topics("studio.example", [page])
    assert topics
    assert all(topic.kind == "category" for topic in topics)


def test_invisible_category_cannot_unlock_visible_area():
    page = _page(
        content_text="Poland",
        json_ld=[
            {"@type": "Service", "serviceType": "Software development", "areaServed": "Poland"}
        ],
    )
    assert _context().extract_prompt_topics("studio.example", [page]) == ()
