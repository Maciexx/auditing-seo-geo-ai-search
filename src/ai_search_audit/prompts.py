from __future__ import annotations

import hashlib
from typing import Any

from .models import AIObservation, AIPrompt, Site

PROMPT_PACK_VERSION = "1.1.0"

_TEMPLATES = {
    "en": {
        "brand_discovery": "What is {brand}, and what is it known for?",
        "category_discovery": "Which {category} options should I consider in {location}?",
        "comparison": "Compare {brand} with similar {category} options in {location}.",
        "recommendation": (
            "Would you recommend {brand} for someone looking for {content_theme} in {location}?"
        ),
        "factual_verification": (
            "What verified services, location details, and key facts are published about {brand}?"
        ),
        "location_service": "Which {category} services does {brand} provide in {location}?",
    },
    "pl": {
        "brand_discovery": "Czym jest {brand} i z czego jest znana ta marka?",
        "category_discovery": "Które oferty w kategorii {category} warto rozważyć w {location}?",
        "comparison": "Porównaj {brand} z podobnymi ofertami {category} w {location}.",
        "recommendation": ("Czy warto wybrać {brand}, szukając {content_theme} w {location}?"),
        "factual_verification": (
            "Jakie zweryfikowane usługi, dane lokalizacyjne i kluczowe fakty "
            "opublikowano o {brand}?"
        ),
        "location_service": "Jakie usługi {category} oferuje {brand} w {location}?",
    },
}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(value.split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def generate_prompt_pack(
    site: Site,
    *,
    entity_type: str = "business",
    category: str | None = None,
    location: str | None = None,
    content_themes: list[str] | None = None,
) -> list[AIPrompt]:
    brand = site.brand or site.domain
    requested_locales = [locale.split("-", 1)[0].casefold() for locale in site.languages]
    locales = _unique([locale for locale in requested_locales if locale in _TEMPLATES]) or ["en"]
    category_text = category or entity_type
    location_by_locale = {
        "en": location or "its target market",
        "pl": location or "docelowym rynku",
    }
    theme_values = _unique([brand, category_text, location or "", *(content_themes or [])])
    prompts: list[AIPrompt] = []
    for locale in locales:
        content_theme = next(
            (item for item in content_themes or [] if item.casefold() != brand.casefold()),
            category_text,
        )
        for intent, template in _TEMPLATES[locale].items():
            text = template.format(
                brand=brand,
                category=category_text,
                location=location_by_locale[locale],
                content_theme=content_theme,
            )
            digest = hashlib.sha256(
                f"{PROMPT_PACK_VERSION}:{locale}:{intent}:{text}".encode()
            ).hexdigest()[:12]
            prompts.append(
                AIPrompt(
                    prompt_id=f"{intent}-{locale}-{digest}",
                    pack_version=PROMPT_PACK_VERSION,
                    locale=locale,
                    intent=intent,
                    text=text,
                    target_entities=[brand],
                    query_themes=theme_values,
                    expected_evidence_needs=[
                        "official website",
                        "independent authoritative source",
                    ],
                    suggested_providers=[
                        "openai-search",
                        "gemini-grounding",
                        "perplexity",
                        "claude-search",
                    ],
                )
            )
    return prompts


def import_observations(items: list[dict[str, Any]]) -> list[AIObservation]:
    return [AIObservation.model_validate(item) for item in items]
