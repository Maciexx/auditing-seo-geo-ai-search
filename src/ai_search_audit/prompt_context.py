"""Conservative locale-bound observations from public Service facts."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .analyzers import (
    _entity_domains,
    _entity_node_is_relevant,
    _json_ld_nodes,
    _supported_entity_type,
)
from .models import Page, Site


class PromptTopic(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["category", "service_area"]
    locale: Literal["pl", "en"]
    value: str = Field(min_length=1, max_length=120)
    source_url: str
    locator: str
    quote: str


# Reviewed literal source phrases, not a general industry classifier.
_CATEGORIES = {
    "pl": frozenset(
        {
            "tworzenie oprogramowania",
            "projektowanie stron internetowych",
            "usługi księgowe",
            "usługi tłumaczeniowe",
            "naprawa rowerów",
            "usługi stomatologiczne",
        }
    ),
    "en": frozenset(
        {
            "software development",
            "web design",
            "accounting services",
            "translation services",
            "bicycle repair",
            "dental services",
        }
    ),
}


def extract_prompt_topics(domain: str, pages: list[Page]) -> tuple[PromptTopic, ...]:
    """Keep source observations without translating or choosing an ambiguous profile."""
    site = Site(domain=domain, base_url=HttpUrl(f"https://{domain}"))
    domains = _entity_domains(site, pages)
    topics: list[PromptTopic] = []
    for page in pages:
        language = (page.language or "").strip().casefold().replace("_", "-").split("-", 1)[0]
        if language not in _CATEGORIES or page.status_code != 200:
            continue
        if page.final_url.host not in domains:
            continue
        locale: Literal["pl", "en"] = "pl" if language == "pl" else "en"
        visible = " ".join(page.content_text.split())
        for index, item in enumerate(page.json_ld):
            nodes = [(f"json_ld[{index}]", item)]
            graph = item.get("@graph")
            if isinstance(graph, list):
                nodes.extend(
                    (f"json_ld[{index}].@graph[{i}]", node)
                    for i, node in enumerate(graph)
                    if isinstance(node, dict)
                )
            for locator, node in nodes:
                raw_types = node.get("@type")
                types = raw_types if isinstance(raw_types, list) else [raw_types]
                if "Service" not in types:
                    continue
                phrase = node.get("serviceType")
                if not isinstance(phrase, str):
                    continue
                phrase = " ".join(phrase.split())
                if phrase.casefold() not in _CATEGORIES[locale] or phrase not in visible:
                    continue
                topics.append(
                    PromptTopic(
                        kind="category",
                        locale=locale,
                        value=phrase,
                        source_url=str(page.final_url),
                        locator=f"{locator}.serviceType",
                        quote=phrase,
                    )
                )
                area = node.get("areaServed")
                if not isinstance(area, str) or any(
                    character in area for character in "{}<>\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
                ):
                    continue
                area = " ".join(area.split())
                if not 0 < len(area) <= 120 or area not in visible:
                    continue
                topics.append(
                    PromptTopic(
                        kind="service_area",
                        locale=locale,
                        value=area,
                        source_url=str(page.final_url),
                        locator=f"{locator}.areaServed",
                        quote=area,
                    )
                )
    return tuple(topics)


_CONTENT_SPAN = re.compile(r"content_text\[(0|[1-9][0-9]{0,7}):(0|[1-9][0-9]{0,7})\]")


def _canonical_topic_page(page: Page, domains: set[str]) -> bool:
    return (
        page.status_code == 200
        and page.url.host in domains
        and page.final_url.host in domains
        and (
            page.canonical is None
            or urljoin(str(page.final_url), page.canonical) == str(page.final_url)
        )
    )


def resolve_selected_prompt_topics(
    domain: str, pages: list[Page], selected_topics: tuple[PromptTopic, ...]
) -> tuple[PromptTopic, ...]:
    """Re-resolve explicit observations, not inferred or client-approved business facts.

    Policy 1.1.0 locators are zero-based Python character spans in the saved,
    unmodified content_text. No CSS, JSON paths, normalization or translation is
    applied to selected text. A changed crawl must explicitly select again.
    """
    if not isinstance(selected_topics, tuple) or len(selected_topics) > 16:
        raise ValueError("prompt topic selection must be a bounded tuple")
    site = Site(domain=domain, base_url=HttpUrl(f"https://{domain}"))
    domains = _entity_domains(site, pages)
    resolved: list[PromptTopic] = []
    for candidate in selected_topics:
        # Revalidate even frozen objects: model_copy/model_construct skip validation.
        raw = (
            {field: getattr(candidate, field, None) for field in PromptTopic.model_fields}
            if isinstance(candidate, PromptTopic)
            else candidate
        )
        topic = PromptTopic.model_validate(raw)
        if (
            topic.value != topic.quote
            or topic.value != topic.value.strip()
            or any(
                unicodedata.category(char).startswith("C")
                or char in '{}<>\n\r\v\f\x85\u2028\u2029"“”„;:!?'
                for char in topic.value
            )
        ):
            raise ValueError("prompt topic selection must contain a safe verbatim phrase")
        match = _CONTENT_SPAN.fullmatch(topic.locator)
        if match is None:
            raise ValueError("unsupported prompt topic selection locator")
        matching = [page for page in pages if str(page.final_url) == topic.source_url]
        if len(matching) != 1 or not _canonical_topic_page(matching[0], domains):
            raise ValueError("prompt topic selection requires a unique canonical successful page")
        page = matching[0]
        language = (page.language or "").strip().casefold().replace("_", "-").split("-", 1)[0]
        start, end = (int(value) for value in match.groups())
        text = page.content_text
        if (
            language != topic.locale
            or not 0 <= start < end <= len(text)
            or text[start:end] != topic.quote
            or (start > 0 and text[start - 1].isalnum() and text[start].isalnum())
            or (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())
        ):
            raise ValueError("prompt topic selection does not match its source observation")
        if topic in resolved:
            raise ValueError("duplicate prompt topic selection")
        resolved.append(topic)
    return tuple(resolved)


def normalize_prompt_identity(value: str) -> str:
    """Normalize only identity comparisons; never rewrite a source observation."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prompt_brand_terms(site: Site, pages: list[Page]) -> tuple[str, ...]:
    """Conservative exclusion terms from canonical identity, never topic candidates."""
    domains = _entity_domains(site, pages)
    names = {site.brand or site.domain, *domains}
    brand = normalize_prompt_identity(site.brand or site.domain)
    for page in pages:
        if not _canonical_topic_page(page, domains):
            continue
        for item in page.json_ld:
            for node in _json_ld_nodes(item):
                name = node.get("name")
                if (
                    not isinstance(name, str)
                    or normalize_prompt_identity(name) != brand
                    or _supported_entity_type(node.get("@type")) is None
                    or not _entity_node_is_relevant(domains, page, node)
                ):
                    continue
                for key in ("alternateName", "legalName"):
                    value = node.get(key)
                    values = value if isinstance(value, list) else [value]
                    names.update(value.strip() for value in values if isinstance(value, str))
    return tuple(sorted(normalize_prompt_identity(name) for name in names if name.strip()))
