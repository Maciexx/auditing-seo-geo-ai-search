"""Bounded public product-sales evidence, not Merchant Center eligibility."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urljoin, urlsplit

from pydantic import HttpUrl

from .analyzers import _entity_domains
from .models import Entity, Page, Site

_PRODUCT_TYPES = frozenset({"product", "individualproduct", "productmodel"})
_GTIN_FIELDS = ("gtin", "gtin8", "gtin12", "gtin13", "gtin14")
_NON_PRODUCT = re.compile(
    r"\b(?:service|giftcard|voucher|gift card|gift certificate|bon podarunkowy|"
    r"karta podarunkowa|reservation|booking|rezerwacja)\b"
)


def _nodes(value: dict[str, Any]) -> list[dict[str, Any]]:
    graph = value.get("@graph")
    return (
        [value, *[item for item in graph if isinstance(item, dict)]]
        if isinstance(graph, list)
        else [value]
    )


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _types(node: dict[str, Any]) -> set[str]:
    value = node.get("@type")
    values = value if isinstance(value, list) else [value]
    return {item.casefold() for item in values if isinstance(item, str)}


def _veto(node: dict[str, Any]) -> bool:
    for key in ("@type", "additionalType", "category", "name"):
        raw = node.get(key)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, str) and _NON_PRODUCT.search(" ".join(value.casefold().split())):
                return True
    return False


def _priced_offer(offer: dict[str, Any]) -> bool:
    value, currency = offer.get("price"), offer.get("priceCurrency")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, str))
        or re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", str(value))
        is None
        or not isinstance(currency, str)
        or re.fullmatch(r"[A-Za-z]{3}", currency) is None
        or ("@type" in offer and "offer" not in _types(offer))
    ):
        return False
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return False
    return price.is_finite() and price >= 0


def _reference(value: object, page: Page) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        original = urlsplit(value)
        if original.scheme and not original.hostname:
            return None
        resolved = urljoin(str(page.final_url), value)
        parsed = urlsplit(resolved)
        _ = parsed.port  # urlsplit defers malformed-port validation until access.
    except ValueError:
        return None
    if parsed.scheme not in {"https", "http"} or parsed.hostname != page.final_url.host:
        return None
    userinfo, separator, host_port = parsed.netloc.rpartition("@")
    return parsed._replace(netloc=userinfo + separator + host_port.lower()).geturl()


def _node_index(nodes: list[dict[str, Any]], page: Page) -> dict[str, dict[str, Any] | None]:
    definitions = list(nodes)
    for node in nodes:
        raw = node.get("offers")
        values = raw if isinstance(raw, list) else [raw]
        definitions.extend(value for value in values if isinstance(value, dict))
    by_id: dict[str, dict[str, Any] | None] = {}
    for node in definitions:
        if len(node) == 1:
            continue  # An @id-only reference is not a second definition.
        reference = _reference(node.get("@id"), page)
        if reference is not None:
            by_id[reference] = node if reference not in by_id else None
    return by_id


def _resolved_values(
    raw: object, by_id: dict[str, dict[str, Any] | None], page: Page
) -> list[dict[str, Any]]:
    values = raw if isinstance(raw, list) else [raw]
    resolved: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or _veto(value):
            continue
        if "@id" in value:
            reference = _reference(value["@id"], page)
            if reference is None:
                continue
            value = by_id.get(reference, value)
        if value is not None:
            resolved.append(value)
    return resolved


def _has_delivery_country(
    offer: dict[str, Any], by_id: dict[str, dict[str, Any] | None], page: Page
) -> bool:
    for shipping in _resolved_values(offer.get("shippingDetails"), by_id, page):
        for destination in _resolved_values(shipping.get("shippingDestination"), by_id, page):
            country = destination.get("addressCountry")
            if _nonblank(country):
                return True
            if any(_nonblank(item.get("name")) for item in _resolved_values(country, by_id, page)):
                return True
    return False


def product_sales_urls(entity: Entity | None, pages: list[Page]) -> tuple[str, ...]:
    """Return sorted, unique final URLs with visible, identified product offers."""
    if entity is None:
        return ()
    site = Site(domain=entity.domain, base_url=HttpUrl(f"https://{entity.domain}"))
    domains = _entity_domains(site, pages)
    evidence_urls: set[str] = set()
    for page in pages:
        if page.status_code != 200 or page.final_url.host not in domains:
            continue
        nodes = [node for value in page.json_ld for node in _nodes(value)]
        by_id = _node_index(nodes, page)
        visible = " ".join(page.content_text.casefold().split())
        for node in _resolved_values(nodes, by_id, page):
            name = node.get("name")
            if (
                not _PRODUCT_TYPES.intersection(_types(node))
                or _veto(node)
                or not isinstance(name, str)
                or not name.strip()
                or " ".join(name.casefold().split()) not in visible
                or not any(_nonblank(node.get(key)) for key in ("sku", *_GTIN_FIELDS))
            ):
                continue
            has_gtin = any(_nonblank(node.get(key)) for key in _GTIN_FIELDS)
            for offer in _resolved_values(node.get("offers"), by_id, page):
                if _veto(offer) or not _priced_offer(offer):
                    continue
                if entity.type.casefold() == "onlinestore" or (
                    has_gtin and _has_delivery_country(offer, by_id, page)
                ):
                    evidence_urls.add(str(page.final_url))
    return tuple(sorted(evidence_urls))
