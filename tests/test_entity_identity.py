from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl

from ai_search_audit.analyzers import (
    _entity_facts,
    build_entity_consistency,
    select_canonical_entity,
)
from ai_search_audit.knowledge import load_registry
from ai_search_audit.models import Evidence, ExternalMention, Page, Site


def _site() -> Site:
    return Site(domain="studio.example", base_url="https://studio.example", brand="Example Studio")


def _root(*nodes: dict[str, object]) -> Page:
    return Page(
        url="https://studio.example/",
        final_url="https://studio.example/",
        status_code=200,
        json_ld=[{"@context": "https://schema.org", "@graph": list(nodes)}],
    )


def _online_business() -> dict[str, object]:
    return {
        "@type": "OnlineBusiness",
        "@id": "https://studio.example/#org",
        "name": "Example Studio",
        "url": "https://studio.example/",
        "telephone": "+48 123 456 789",
        "address": {"addressLocality": "Office City"},
    }


def _website(publisher: object = "https://studio.example/#org") -> dict[str, object]:
    return {
        "@type": "WebSite",
        "name": "Example website",
        "url": "https://studio.example/",
        "publisher": publisher,
    }


def _hotel() -> dict[str, object]:
    return {
        "@type": "Hotel",
        "@id": "https://studio.example/#hotel",
        "name": "Unrelated example hotel",
        "url": "https://studio.example/hotel",
    }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "publisher", ["https://studio.example/#org", {"@id": "https://studio.example/#org"}]
)
def test_root_publisher_binding_outranks_unrelated_same_domain_hotel(
    publisher: object, reverse: bool
) -> None:
    nodes = [_hotel(), _online_business(), _website(publisher)]
    page = _root(*reversed(nodes)) if reverse else _root(*nodes)

    entity = select_canonical_entity(_site(), [page])

    assert entity.brand == "Example Studio"
    assert entity.type == "OnlineBusiness"


@pytest.mark.parametrize("types", [["WebSite", "Organization"], ["Organization", "WebSite"]])
def test_website_type_membership_preserves_publisher_binding(types: list[str]) -> None:
    website = _website("#org")
    website["@type"] = types

    entity = select_canonical_entity(_site(), [_root(_hotel(), _online_business(), website)])

    assert entity.type == "OnlineBusiness"
    assert entity.brand == "Example Studio"
    assert entity.facts["telephone"] == ["+48123456789"]


@pytest.mark.parametrize(
    ("publisher", "node_id"),
    [
        ("#org", "https://studio.example/#org"),
        ({"@id": "/#org"}, "https://studio.example/#org"),
        ("https://studio.example/#org", "#org"),
    ],
)
def test_relative_publisher_and_candidate_ids_resolve_against_final_url(
    publisher: object, node_id: str
) -> None:
    business = _online_business()
    business["@id"] = node_id

    entity = select_canonical_entity(_site(), [_root(_hotel(), business, _website(publisher))])

    assert entity.brand == "Example Studio"
    assert entity.type == "OnlineBusiness"


@pytest.mark.parametrize(
    ("publisher", "node_id"),
    [
        ("https://STUDIO.EXAMPLE/#org", "https://studio.example/#org"),
        ("https://studio.example/#org", "https://STUDIO.EXAMPLE/#org"),
    ],
)
def test_publisher_binding_ignores_hostname_casing(publisher: str, node_id: str) -> None:
    business = _online_business()
    business["@id"] = node_id

    entity = select_canonical_entity(_site(), [_root(_hotel(), business, _website(publisher))])

    assert entity.brand == "Example Studio"
    assert entity.type == "OnlineBusiness"


@pytest.mark.parametrize("reverse", [False, True])
def test_root_publishers_differing_only_in_hostname_case_are_not_conflicting(
    reverse: bool,
) -> None:
    nodes = [_hotel(), _online_business(), _website(), _website("https://STUDIO.EXAMPLE/#org")]
    page = _root(*reversed(nodes)) if reverse else _root(*nodes)

    entity = select_canonical_entity(_site(), [page])

    assert entity.type == "OnlineBusiness"
    assert entity.facts["telephone"] == ["+48123456789"]


@pytest.mark.parametrize(
    "different_reference",
    [
        "https://studio.example/org?Kind=A#Node",
        "https://studio.example/Org?kind=A#Node",
        "https://studio.example/Org?Kind=A#node",
    ],
)
def test_hostname_normalization_preserves_path_query_and_fragment_case(
    different_reference: str,
) -> None:
    business = _online_business()
    business["@id"] = "https://studio.example/Org?Kind=A#Node"
    hotel = _hotel()
    hotel["@id"] = different_reference

    entity = select_canonical_entity(
        _site(), [_root(hotel, business, _website("https://STUDIO.EXAMPLE/Org?Kind=A#Node"))]
    )

    assert entity.type == "OnlineBusiness"
    assert entity.brand == "Example Studio"


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_root_publishers_fall_back_without_arbitrary_entity_facts(
    reverse: bool,
) -> None:
    nodes = [
        _hotel(),
        _online_business(),
        {
            "@type": "Organization",
            "@id": "https://studio.example/#second-org",
            "name": "Second example organization",
        },
        _website(),
        _website("#second-org"),
    ]
    page = _root(*reversed(nodes)) if reverse else _root(*nodes)

    entity = select_canonical_entity(_site(), [page])

    assert entity.type == "Organization"
    assert entity.brand == "Example Studio"
    assert entity.facts == {}
    assert entity.location is None


def test_online_business_is_selected_with_its_official_facts() -> None:
    page = _root(
        {
            "@type": "OnlineStore",
            "name": "Foreign partner",
            "url": "https://partner.example/",
            "telephone": "+48 987 654 321",
        },
        _online_business(),
        _website({"@id": "https://studio.example/#org"}),
    )

    entity = select_canonical_entity(_site(), [page])

    assert entity.type == "OnlineBusiness"
    assert entity.brand == "Example Studio"
    assert entity.facts["telephone"] == ["+48123456789"]
    assert entity.facts["address"] == ["office city"]
    assert entity.location == "Office City"
    assert "location" not in entity.facts
    assert "areaServed" not in entity.facts


def test_foreign_only_nodes_use_root_identity_without_foreign_facts() -> None:
    page = _root(
        {
            "@type": "Hotel",
            "name": "Foreign hotel",
            "url": "https://foreign.example/",
            "telephone": "+48 987 654 321",
            "address": {"addressLocality": "Foreign City"},
        }
    )
    page.title = "Example Studio official website"

    entity = select_canonical_entity(_site(), [page])

    assert entity.brand == "Example Studio"
    assert entity.type == "Organization"
    assert entity.facts == {}
    assert entity.location is None


@pytest.mark.parametrize("brand", ["Example Studio", None])
@pytest.mark.parametrize(
    ("headings", "title"),
    [
        (["Odkryj niezapomniane chwile"], "Example Studio"),
        (["Odkryj niezapomniane chwile"], "Odkryj niezapomniane chwile"),
        ([], "Odkryj niezapomniane chwile"),
    ],
)
def test_root_prose_cannot_replace_supplied_brand_or_domain(
    brand: str | None, headings: list[str], title: str
) -> None:
    site = _site()
    site.brand = brand
    page = _root()
    page.h1 = headings
    page.title = title

    entity = select_canonical_entity(site, [page])

    assert entity.brand == (brand or site.domain)
    assert entity.type == "Organization"
    assert entity.facts == {}


@pytest.mark.parametrize(
    ("headings", "title", "expected"),
    [
        (["EXAMPLE STUDIO"], "Unbound title", "EXAMPLE STUDIO"),
        (["Odkryj niezapomniane chwile"], "Our services | EXAMPLE STUDIO", "EXAMPLE STUDIO"),
        ([], "Our services | Example-Studio", "Example-Studio"),
    ],
)
def test_root_identity_accepts_only_matching_alphanumeric_brand(
    headings: list[str], title: str, expected: str
) -> None:
    page = _root()
    page.h1 = headings
    page.title = title

    assert select_canonical_entity(_site(), [page]).brand == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_secondary_website_publishers_do_not_outvote_root_binding(reverse: bool) -> None:
    root = _root(_online_business(), _website())
    secondary = _root(_hotel(), _website("https://studio.example/#hotel"))
    secondary.url = secondary.final_url = HttpUrl("https://studio.example/partners")
    pages = [secondary, root] if reverse else [root, secondary]

    entity = select_canonical_entity(_site(), pages)

    assert entity.brand == "Example Studio"
    assert entity.type == "OnlineBusiness"


def test_observed_root_redirect_allows_www_publisher_and_entity() -> None:
    business = _online_business()
    business["url"] = "https://www.studio.example/"
    business["@id"] = "https://www.studio.example/#org"
    website = _website("#org")
    website["url"] = "https://www.studio.example/"
    page = _root(_hotel(), business, website)
    page.final_url = HttpUrl("https://www.studio.example/")
    page.redirect_chain = ["https://studio.example/", "https://www.studio.example/"]

    entity = select_canonical_entity(_site(), [page])

    assert entity.brand == "Example Studio"
    assert entity.type == "OnlineBusiness"
    assert entity.domain == "studio.example"
    assert entity.facts["telephone"] == ["+48123456789"]


@pytest.mark.parametrize("host", ["www.studio.example", "partner.studio.example"])
def test_unobserved_host_variants_cannot_supply_canonical_facts(host: str) -> None:
    page = _root({"@type": "Hotel", "name": "Untrusted Hotel", "url": f"https://{host}/"})

    entity = select_canonical_entity(_site(), [page])

    assert entity.brand == "Example Studio"
    assert entity.facts == {}


def test_redirect_to_arbitrary_subdomain_does_not_expand_entity_trust() -> None:
    page = _root(
        {
            "@type": "Hotel",
            "name": "Untrusted Hotel",
            "url": "https://partner.studio.example/",
        }
    )
    page.final_url = HttpUrl("https://partner.studio.example/")
    page.redirect_chain = [str(page.url), str(page.final_url)]

    assert select_canonical_entity(_site(), [page]).brand == "Example Studio"


@pytest.mark.parametrize("publisher", ["https://[invalid/#org", {"@id": "//[invalid"}])
def test_malformed_publisher_reference_does_not_crash_selection(publisher: object) -> None:
    page = _root(_online_business(), _website(publisher))

    assert select_canonical_entity(_site(), [page]).type == "OnlineBusiness"


@pytest.mark.parametrize("node_id", ["https://[invalid/#org", "//[invalid"])
def test_malformed_candidate_id_does_not_crash_selection(node_id: str) -> None:
    business = _online_business()
    business["@id"] = node_id

    assert select_canonical_entity(_site(), [_root(business)]).type == "OnlineBusiness"


@pytest.mark.parametrize(
    "malformed_url", ["https://[invalid", "https://studio.example:invalid/", "https://"]
)
@pytest.mark.parametrize("node_id", ["https://studio.example/#org", "#org"])
def test_malformed_candidate_url_is_omitted_when_id_establishes_identity(
    malformed_url: str,
    node_id: str,
) -> None:
    business = _online_business()
    business["@id"] = node_id
    business["url"] = malformed_url

    entity = select_canonical_entity(_site(), [_root(business, _website())])

    assert entity.type == "OnlineBusiness"
    assert entity.brand == "Example Studio"
    assert entity.facts == {"telephone": ["+48123456789"], "address": ["office city"]}


@pytest.mark.parametrize(
    "malformed_url", ["https://[invalid", "https://studio.example:invalid/", "https://"]
)
def test_malformed_candidate_url_without_established_identity_uses_fallback(
    malformed_url: str,
) -> None:
    business = _online_business()
    business.pop("@id")
    business["url"] = malformed_url

    entity = select_canonical_entity(_site(), [_root(business)])

    assert entity.type == "Organization"
    assert entity.facts == {}


@pytest.mark.parametrize(
    ("node_id", "node_url"),
    [
        ("https://studio.example/#org", "https://foreign.example/"),
        ("https://foreign.example/#org", "https://studio.example/"),
        ("https://studio.example/#org", "//foreign.example/"),
        ("//foreign.example/#org", "https://studio.example/"),
    ],
)
def test_mixed_local_and_foreign_identity_references_cannot_supply_facts(
    node_id: str, node_url: str
) -> None:
    business = _online_business()
    business["@id"] = node_id
    business["url"] = node_url
    business["name"] = "Conflicting identity"

    entity = select_canonical_entity(_site(), [_root(business)])

    assert entity.type == "Organization"
    assert entity.brand == "Example Studio"
    assert entity.facts == {}


@pytest.mark.parametrize(
    "unsupported_url", ["ftp://foreign.example/", {"value": "https://foreign.example/"}]
)
def test_unsupported_url_input_cannot_bypass_identity_domain_checks(
    unsupported_url: object,
) -> None:
    business = _online_business()
    business["url"] = unsupported_url

    entity = select_canonical_entity(_site(), [_root(business)])

    assert entity.type == "Organization"
    assert entity.brand == "Example Studio"
    assert entity.facts == {}


@pytest.mark.parametrize(
    "unsupported_url", ["ftp://foreign.example/", {"value": "https://foreign.example/"}]
)
def test_unsupported_url_input_is_not_normalized_into_an_official_fact(
    unsupported_url: object,
) -> None:
    business = _online_business()
    business["url"] = unsupported_url

    facts = _entity_facts(business)

    assert "url" not in facts
    assert facts["telephone"] == ["+48123456789"]


@pytest.mark.parametrize("reference", ["https://studio.example:invalid/#org", "https://"])
def test_malformed_publisher_ids_cannot_override_type_priority(reference: str) -> None:
    business = _online_business()
    business["@id"] = reference

    entity = select_canonical_entity(_site(), [_root(_hotel(), business, _website(reference))])

    assert entity.type == "Hotel"


@pytest.mark.parametrize("reference", ["https://foreign.example/", "//foreign.example/"])
def test_foreign_website_cannot_supply_an_authoritative_publisher(reference: str) -> None:
    website = _website()
    website["url"] = reference
    page = _root(_hotel(), _online_business(), website)

    assert select_canonical_entity(_site(), [page]).type == "Hotel"


@pytest.mark.parametrize("reference", ["//foreign.example/#org", "https://foreign.example/#org"])
def test_foreign_publisher_is_ignored_even_when_candidate_uses_same_id(reference: str) -> None:
    business = _online_business()
    business["@id"] = reference
    page = _root(_hotel(), business, _website(reference))

    assert select_canonical_entity(_site(), [page]).type == "Hotel"


def test_scheme_relative_foreign_entity_cannot_supply_canonical_facts() -> None:
    page = _root({"@type": "Hotel", "name": "Foreign Hotel", "@id": "//foreign.example/#hotel"})

    entity = select_canonical_entity(_site(), [page])

    assert entity.brand == "Example Studio"
    assert entity.facts == {}


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (["Organization", "OnlineBusiness"], "OnlineBusiness"),
        (["OnlineBusiness", "Organization"], "OnlineBusiness"),
        (["OnlineBusiness", "OnlineStore"], "OnlineStore"),
        (["OnlineStore", "OnlineBusiness"], "OnlineStore"),
        (["Organization", "OnlineStore"], "OnlineStore"),
    ],
)
def test_type_lists_choose_the_most_specific_supported_type(
    types: list[str], expected: str
) -> None:
    node = _online_business()
    node["@type"] = types

    assert select_canonical_entity(_site(), [_root(node)]).type == expected


def test_online_business_official_facts_keep_evidence_and_conflict_urls() -> None:
    page = _root(_online_business(), _website())
    canonical = select_canonical_entity(_site(), [page])
    official = Evidence(
        evidence_id="official-jsonld",
        source_url=page.url,
        source_type="structured_data",
        collector="structured-data",
        observed_at=datetime(2026, 9, 3, tzinfo=UTC),
        observed_value=page.json_ld[0],
    )
    mention = ExternalMention(
        mention_id="directory",
        url="https://directory.example/studio",
        publisher="Example directory",
        source_class="directory",
        claims={"address": "Previous Office City", "telephone": "+48 (123) 456-789"},
        evidence_ids=["directory-address"],
        independent_source_key="directory",
    )

    profile, matrix, findings = build_entity_consistency(
        _site(),
        [mention],
        load_registry(Path(__file__).parents[1] / "knowledge"),
        as_of=date(2026, 9, 3),
        canonical_entity=canonical,
        canonical_evidence=[official],
    )

    assert profile.type == "OnlineBusiness"
    assert matrix["address"] == {
        "office city": ["official-jsonld"],
        "previous office city": ["directory-address"],
    }
    assert matrix["telephone"] == {"+48123456789": ["official-jsonld", "directory-address"]}
    assert "location" not in matrix
    assert len(findings) == 1
    assert findings[0].technical_title == "Conflicting public values for address"
    assert set(findings[0].evidence_ids) == {"official-jsonld", "directory-address"}
    assert set(findings[0].affected_urls) == {
        str(page.url),
        "https://directory.example/studio",
    }
