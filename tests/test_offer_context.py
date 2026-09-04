from copy import deepcopy
from datetime import UTC, date, datetime

import pytest

from ai_search_audit.data_requests import EntityKind
from ai_search_audit.models import AuditRun, Entity, Page, Site
from ai_search_audit.project_orchestrator import _entity_kind


def entity(kind="OnlineStore"):
    return Entity(brand="Example", type=kind, domain="example.com")


def product(**changes):
    return {
        "@type": "Product",
        "name": "Blue mug",
        "sku": "MUG-1",
        "offers": {"@type": "Offer", "price": "19.95", "priceCurrency": "PLN"},
        **changes,
    }


def page(node=None, **changes):
    return Page(
        **{
            "url": "https://example.com/mug",
            "final_url": "https://example.com/mug",
            "status_code": 200,
            "content_text": "Buy our BLUE   MUG today.",
            "json_ld": [product() if node is None else node],
            **changes,
        }
    )


def classify(canonical, pages):
    return _entity_kind(
        AuditRun(
            audit_id="audit-example",
            audit_engine_version="0.2.0",
            site=Site(domain="example.com", base_url="https://example.com"),
            ruleset_version="test",
            ruleset_verified_date=date(2026, 9, 3),
            timestamp=datetime(2026, 9, 3, tzinfo=UTC),
            scores=[],
            entity=canonical,
            pages=pages,
            configuration={"entity_classification_policy": "2.0.0"},
        )
    )


@pytest.mark.parametrize("kind", ["OnlineStore", "OnlineBusiness", "CollectionPage", "Product"])
def test_v2_lone_commerce_type_is_generic(kind):
    assert classify(entity(kind), [page({"@type": kind})]) is EntityKind.GENERIC


@pytest.mark.parametrize(
    "price", [-1, "-0.01", True, False, float("nan"), float("inf"), "NaN", "Infinity", None, {}, ""]
)
def test_v2_rejects_invalid_offer_price(price):
    node = product(offers={"@type": "Offer", "price": price, "priceCurrency": "PLN"})
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


@pytest.mark.parametrize("currency", [None, "", "PL", "PLNN", "PŁN", 123, True])
def test_v2_rejects_missing_or_malformed_currency(currency):
    node = product(offers={"@type": "Offer", "price": 1, "priceCurrency": currency})
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


@pytest.mark.parametrize(
    "node",
    [
        product(name=" "),
        product(name=None),
        product(name="Not visible"),
        product(sku=None),
        product(sku=" "),
        product(offers={"@id": "#missing"}),
        product(offers={"@type": "Offer", "priceCurrency": "PLN"}),
        {"@type": "Service", "price": 10},
        {"@type": "Product", "price": 10},
    ],
)
def test_v2_rejects_incomplete_product_evidence(node):
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


@pytest.mark.parametrize(
    "changes",
    [
        {"status_code": 404},
        {"content_text": ""},
        {"final_url": "https://foreign.example/mug"},
        {"final_url": "https://www.example.com/mug"},
    ],
)
def test_v2_requires_successful_official_visible_page(changes):
    assert classify(entity(), [page(**changes)]) is EntityKind.GENERIC


@pytest.mark.parametrize("kind", ["Product", "IndividualProduct", "ProductModel"])
@pytest.mark.parametrize("price", [0, "0", 19.95, "19.95"])
def test_online_store_with_visible_identified_priced_product_is_ecommerce(kind, price):
    node = product(**{"@type": kind, "offers": {"price": price, "priceCurrency": "pln"}})
    assert classify(entity(), [page(node)]) is EntityKind.ECOMMERCE


def test_local_entity_remains_local_without_product_sales():
    assert classify(entity("Hotel"), [page(product(offers=None))]) is EntityKind.LOCAL


def test_missing_canonical_entity_does_not_infer_commerce():
    assert classify(None, [page()]) is EntityKind.GENERIC


def test_sales_urls_are_sorted_unique_final_urls_and_inputs_are_unchanged():
    from ai_search_audit.offer_context import product_sales_urls

    canonical = entity()
    pages = [page(), page(final_url="https://example.com/a"), page()]
    before = deepcopy((canonical, pages))
    assert product_sales_urls(canonical, pages) == (
        "https://example.com/a",
        "https://example.com/mug",
    )
    assert (canonical, pages) == before


def test_graph_and_list_offers_resolve_local_offer_reference():
    from ai_search_audit.offer_context import product_sales_urls

    graph = {
        "@graph": [
            product(offers=[{"@id": "#missing"}, {"@id": "#offer"}]),
            {"@id": "#offer", "@type": "Offer", "price": 3, "priceCurrency": "EUR"},
        ]
    }
    assert product_sales_urls(entity(), [page(graph)]) == ("https://example.com/mug",)


def test_observed_official_root_redirect_allows_www_product_page():
    from ai_search_audit.offer_context import product_sales_urls

    root = page(
        {}, url="https://example.com/", final_url="https://www.example.com/", content_text=""
    )
    product_page = page(url="https://www.example.com/mug", final_url="https://www.example.com/mug")
    assert product_sales_urls(entity(), [root, product_page]) == ("https://www.example.com/mug",)


@pytest.mark.parametrize("field", ["name", "category", "additionalType"])
@pytest.mark.parametrize(
    "marker",
    [
        "voucher",
        "gift card",
        "gift certificate",
        "bon podarunkowy",
        "karta podarunkowa",
        "reservation",
        "booking",
        "rezerwacja",
        "Service",
        "https://schema.org/GiftCard",
    ],
)
def test_priced_sku_product_voucher_on_online_store_remains_generic(field, marker):
    node = product(**{field: marker})
    assert classify(entity(), [page(node, content_text=f"Blue mug {marker}")]) is EntityKind.GENERIC


@pytest.mark.parametrize("marker", ["Service", "GiftCard"])
def test_service_or_giftcard_type_cannot_be_hidden_by_product_type(marker):
    assert (
        classify(entity(), [page(product(**{"@type": ["Product", marker]}))]) is EntityKind.GENERIC
    )


def test_service_offer_does_not_prove_product_sales():
    offer = {"@type": "Service", "price": 19.95, "priceCurrency": "PLN"}
    assert classify(entity(), [page(product(offers=offer))]) is EntityKind.GENERIC


def shipped_product(**changes):
    return product(
        gtin13="5901234123457",
        offers={
            "@type": "Offer",
            "price": "19.95",
            "priceCurrency": "PLN",
            "shippingDetails": {"shippingDestination": {"addressCountry": "PL"}},
        },
        **changes,
    )


def test_organization_qualifies_with_visible_gtin_price_and_same_offer_delivery_country():
    assert classify(entity("Organization"), [page(shipped_product())]) is EntityKind.ECOMMERCE


def test_organization_resolves_shipping_details_on_same_priced_offer():
    node = shipped_product()
    node["offers"]["shippingDetails"] = {"@id": "#shipping"}
    graph = {
        "@graph": [
            node,
            {"@id": "#shipping", "shippingDestination": {"addressCountry": {"name": "Poland"}}},
        ]
    }
    assert classify(entity("Organization"), [page(graph)]) is EntityKind.ECOMMERCE


@pytest.mark.parametrize("missing", ["gtin13", "shippingDetails", "addressCountry"])
def test_organization_needs_both_gtin_and_delivery_country(missing):
    node = shipped_product()
    if missing == "gtin13":
        node.pop(missing)
    elif missing == "shippingDetails":
        node["offers"].pop(missing)
    else:
        node["offers"]["shippingDetails"]["shippingDestination"].pop(missing)
    assert classify(entity("Organization"), [page(node)]) is EntityKind.GENERIC


def test_organization_does_not_join_price_and_shipping_from_different_offers():
    node = shipped_product()
    shipping = node["offers"].pop("shippingDetails")
    node["offers"] = [node["offers"], {"@type": "Offer", "shippingDetails": shipping}]
    assert classify(entity("Organization"), [page(node)]) is EntityKind.GENERIC


@pytest.mark.parametrize("country", [None, "", "  ", True, {}, {"@id": "#unresolved"}])
def test_organization_rejects_missing_or_malformed_delivery_country(country):
    node = shipped_product()
    node["offers"]["shippingDetails"]["shippingDestination"]["addressCountry"] = country
    assert classify(entity("Organization"), [page(node)]) is EntityKind.GENERIC


def test_graph_with_malformed_offer_reference_does_not_crash_or_prove_commerce():
    graph = {
        "@graph": [
            product(offers={"@id": "https://[invalid"}),
            {"@id": "https://[invalid", "price": 10, "priceCurrency": "PLN"},
        ]
    }
    assert classify(entity(), [page(graph)]) is EntityKind.GENERIC


def test_conflicting_duplicate_offer_ids_do_not_prove_commerce():
    graph = {
        "@graph": [
            product(offers={"@id": "#offer"}),
            {"@id": "#offer", "@type": "Service"},
            {"@id": "#offer", "@type": "Offer", "price": 10, "priceCurrency": "PLN"},
        ]
    }
    assert classify(entity(), [page(graph)]) is EntityKind.GENERIC


@pytest.mark.parametrize("kind", ["OnlineBusiness", "CollectionPage", "Hotel", "Product"])
def test_strong_product_proof_does_not_require_a_particular_canonical_type(kind):
    assert classify(entity(kind), [page(shipped_product())]) is EntityKind.ECOMMERCE


@pytest.mark.parametrize("kind", ["OnlineBusiness", "CollectionPage", "Product"])
def test_non_store_type_with_only_sku_and_price_remains_generic(kind):
    assert classify(entity(kind), [page()]) is EntityKind.GENERIC


@pytest.mark.parametrize("price", ["1_0", "1,00", "one", "--2"])
def test_malformed_numeric_strings_do_not_prove_commerce(price):
    node = product(offers={"@type": "Offer", "price": price, "priceCurrency": "PLN"})
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


def test_reference_veto_is_not_lost_when_resolving_offer():
    graph = {
        "@graph": [
            product(offers={"@id": "#offer", "category": "voucher"}),
            {"@id": "#offer", "@type": "Offer", "price": 10, "priceCurrency": "PLN"},
        ]
    }
    assert classify(entity(), [page(graph)]) is EntityKind.GENERIC


def test_resolved_non_offer_type_does_not_prove_commerce():
    graph = {
        "@graph": [
            product(offers={"@id": "#offer"}),
            {"@id": "#offer", "@type": "CollectionPage", "price": 10, "priceCurrency": "PLN"},
        ]
    }
    assert classify(entity(), [page(graph)]) is EntityKind.GENERIC


@pytest.mark.parametrize("conflict", [{"@type": "GiftCard"}, {"category": "voucher"}])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_duplicate_product_identity_does_not_prove_commerce(conflict, reverse):
    nodes = [product(**{"@id": "#product"}), {"@id": "#product", **conflict}]
    if reverse:
        nodes.reverse()
    assert classify(entity(), [page({"@graph": nodes})]) is EntityKind.GENERIC


@pytest.mark.parametrize("conflict", [{"@type": "GiftCard"}, {"category": "voucher"}])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_inline_offer_identity_does_not_prove_commerce(conflict, reverse):
    offers = [
        {"@id": "#offer", "@type": "Offer", "price": 10, "priceCurrency": "PLN"},
        {"@id": "#offer", **conflict},
    ]
    if reverse:
        offers.reverse()
    assert classify(entity(), [page(product(offers=offers))]) is EntityKind.GENERIC


@pytest.mark.parametrize("reference", ["https://foreign.example/product", "https://[invalid"])
def test_foreign_or_malformed_product_identity_does_not_prove_commerce(reference):
    assert classify(entity(), [page(product(**{"@id": reference}))]) is EntityKind.GENERIC


@pytest.mark.parametrize("port", ["bad", "99999", "-1"])
@pytest.mark.parametrize("target", ["product", "offer"])
def test_malformed_identity_port_does_not_prove_commerce(port, target):
    node = product()
    target_node = node if target == "product" else node["offers"]
    target_node["@id"] = f"https://example.com:{port}/{target}"
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


def test_unique_official_product_and_inline_offer_identities_still_prove_commerce():
    node = product(**{"@id": "#product"})
    node["offers"]["@id"] = "#offer"
    assert classify(entity(), [page(node)]) is EntityKind.ECOMMERCE


@pytest.mark.parametrize("reference", ["https:///product", "https:#product", "https:/product"])
@pytest.mark.parametrize("target", ["product", "offer"])
def test_malformed_explicit_scheme_identity_is_not_repaired(reference, target):
    node = product()
    target_node = node if target == "product" else node["offers"]
    target_node["@id"] = reference
    assert classify(entity(), [page(node)]) is EntityKind.GENERIC


@pytest.mark.parametrize("reverse", [False, True])
def test_host_case_cannot_hide_conflicting_product_identity(reverse):
    nodes = [
        product(**{"@id": "https://example.com/mug#product"}),
        {"@id": "https://EXAMPLE.COM/mug#product", "category": "voucher"},
    ]
    if reverse:
        nodes.reverse()
    assert classify(entity(), [page({"@graph": nodes})]) is EntityKind.GENERIC


def test_offer_reference_resolves_across_host_case_only():
    nodes = [
        product(offers={"@id": "https://EXAMPLE.COM/mug#offer"}),
        {"@id": "https://example.com/mug#offer", "price": 10, "priceCurrency": "PLN"},
    ]
    assert classify(entity(), [page({"@graph": nodes})]) is EntityKind.ECOMMERCE


@pytest.mark.parametrize(
    "different_id", ["https://example.com/Mug#product", "https://example.com/mug#Product"]
)
def test_identity_normalization_preserves_path_and_fragment_case(different_id):
    nodes = [
        product(**{"@id": "https://example.com/mug#product"}),
        {"@id": different_id, "category": "voucher"},
    ]
    assert classify(entity(), [page({"@graph": nodes})]) is EntityKind.ECOMMERCE
