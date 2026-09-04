from __future__ import annotations

import hashlib
from typing import Any

from .artifact_policy import saved_prompt_version
from .models import AIObservation, AIPrompt, AuditRun, Page, Site
from .prompt_context import (
    PromptTopic,
    extract_prompt_topics,
    normalize_prompt_identity,
    prompt_brand_terms,
    resolve_selected_prompt_topics,
)
from .report_models import ReportLocale

PROMPT_PACK_VERSION = "2.0.0"
PROMPT_CONTEXT_POLICY_VERSION = "1.0.0"
SELECTED_PROMPT_PACK_VERSION = "2.1.0"
SELECTED_PROMPT_CONTEXT_POLICY_VERSION = "1.1.0"

_TEMPLATES = {
    "en": {
        "brand_discovery": "What is {brand}, and what is it known for?",
        "factual_verification": "Which facts about {brand} can be verified from sources?",
        "offer_details": "What does {brand} offer, and where are the details published?",
        "service_area": "Where does {brand} operate, and which sources confirm this?",
        "conditions": (
            "Which terms for using the services or products of {brand} are publicly available?"
        ),
        "sources": "Which independent sources describe {brand}?",
    },
    "pl": {
        "brand_discovery": "Czym jest {brand} i z czego jest znana ta marka?",
        "factual_verification": "Jakie informacje o {brand} można potwierdzić w źródłach?",
        "offer_details": "Co oferuje {brand} i gdzie opisano szczegóły oferty?",
        "service_area": "Gdzie działa {brand} i jakie źródła to potwierdzają?",
        "conditions": "Jakie warunki korzystania z oferty {brand} są publicznie dostępne?",
        "sources": "Które niezależne źródła opisują {brand}?",
    },
}


_CATEGORY_TEMPLATES = {
    "en": {
        "category_discovery": 'Which providers offer the service "{category}"?',
        "comparison": 'Compare {brand} with other providers of "{category}".',
    },
    "pl": {
        "category_discovery": "Które firmy oferują usługę „{category}”?",
        "comparison": "Porównaj ofertę {brand} z innymi dostawcami usługi „{category}”.",
    },
}
_AREA_SUFFIX = {"en": ' Service area: "{area}".', "pl": " Obszar działania: „{area}”."}

_NEUTRAL_TEMPLATES = {
    "en": {
        **_TEMPLATES["en"],
        "service_area": "Which locations are associated with {brand}, according to sources?",
        "conditions": "Which publicly available terms relate to {brand}?",
    },
    "pl": {
        **_TEMPLATES["pl"],
        "service_area": "Jakie lokalizacje są powiązane z {brand} według źródeł?",
        "conditions": "Jakie publicznie dostępne warunki dotyczą {brand}?",
    },
}
_NEUTRAL_DISCOVERY = {
    "en": 'Which options are available for "{category}", and which sources describe them?',
    "pl": "Jakie możliwości są dostępne dla tematu „{category}” i jakie źródła je opisują?",
}
_NEUTRAL_AREA = {"en": ' Location: "{area}".', "pl": " Lokalizacja: „{area}”."}


def generate_prompt_pack(
    site: Site,
    *,
    entity_type: str = "business",
    category: str | None = None,
    location: str | None = None,
    content_themes: list[str] | None = None,
    topics: tuple[PromptTopic, ...] = (),
    pack_version: str = PROMPT_PACK_VERSION,
    pages: list[Page] | None = None,
) -> list[AIPrompt]:
    """Generate locale-bound prompts; legacy free-form context is intentionally ignored."""
    if pack_version not in (PROMPT_PACK_VERSION, SELECTED_PROMPT_PACK_VERSION):
        raise ValueError("unsupported generated prompt pack version")
    selected_policy = pack_version == SELECTED_PROMPT_PACK_VERSION
    if selected_policy:
        automatic = extract_prompt_topics(site.domain, pages or [])
        resolve_selected_prompt_topics(
            site.domain, pages or [], tuple(topic for topic in topics if topic not in automatic)
        )
    brand = site.brand or site.domain
    brand_terms = prompt_brand_terms(site, pages or []) if selected_policy else ()
    requested_locales = [
        locale.strip().casefold().replace("_", "-").split("-", 1)[0] for locale in site.languages
    ]
    locales = list(
        dict.fromkeys(locale for locale in requested_locales if locale in _TEMPLATES)
    ) or ["en"]
    prompts: list[AIPrompt] = []
    for locale in locales:
        categories = {
            topic.value for topic in topics if topic.locale == locale and topic.kind == "category"
        }
        areas = {
            topic.value
            for topic in topics
            if topic.locale == locale and topic.kind == "service_area"
        }
        category_text = next(iter(categories)) if len(categories) == 1 else None
        area_text = next(iter(areas)) if len(areas) == 1 else None
        templates = _NEUTRAL_TEMPLATES if selected_policy else _TEMPLATES
        for intent, template in templates[locale].items():
            theme_values = [brand]
            if selected_policy and category_text is not None and intent == "offer_details":
                discovery = _NEUTRAL_DISCOVERY[locale].format(category=category_text)
                if area_text is not None:
                    discovery += _NEUTRAL_AREA[locale].format(area=area_text)
                # A common-phrase brand is still a brand. Do not erase or paraphrase
                # the source observation to manufacture an unbranded question.
                if not any(term in normalize_prompt_identity(discovery) for term in brand_terms):
                    intent = "category_discovery"
                    template = discovery
                    theme_values = [category_text, *([area_text] if area_text else [])]
            elif (
                not selected_policy
                and category_text is not None
                and intent in {"offer_details", "conditions"}
            ):
                intent = "category_discovery" if intent == "offer_details" else "comparison"
                template = _CATEGORY_TEMPLATES[locale][intent]
                theme_values.append(category_text)
            text = (
                template
                if selected_policy and intent == "category_discovery"
                else template.format(brand=brand, category=category_text)
            )
            if not selected_policy and intent == "category_discovery" and area_text is not None:
                text += _AREA_SUFFIX[locale].format(area=area_text)
                theme_values.append(area_text)
            digest = hashlib.sha256(
                f"{pack_version}:{locale}:{intent}:{text}".encode()
            ).hexdigest()[:12]
            prompts.append(
                AIPrompt(
                    prompt_id=f"{intent}-{locale}-{digest}",
                    pack_version=pack_version,
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
                        *([] if selected_policy else ["gemini-grounding"]),
                        "perplexity",
                        "claude-search",
                    ],
                )
            )
    return prompts


def prompt_scope_warnings(prompts: list[AIPrompt], report_locale: ReportLocale) -> list[str]:
    locales = {prompt.locale for prompt in prompts}
    discovery_locales = {
        prompt.locale for prompt in prompts if prompt.intent == "category_discovery"
    }
    missing = sorted(locales - discovery_locales)
    if not missing:
        return []
    label = ", ".join(missing)
    if report_locale == "pl":
        return [
            f"Brak potwierdzonej kategorii ({label}): "
            "zestaw pytań nie bada odkrywania marki przez kategorię."
        ]
    return [
        f"No verified category ({label}): the prompt pack does not test category-based discovery."
    ]


def validate_automatic_prompt_pack(run: AuditRun) -> None:
    """Gate future automatic execution, never reinterpret historical worksheets."""
    version = saved_prompt_version(run)
    policies = {
        PROMPT_PACK_VERSION: PROMPT_CONTEXT_POLICY_VERSION,
        SELECTED_PROMPT_PACK_VERSION: SELECTED_PROMPT_CONTEXT_POLICY_VERSION,
    }
    if version not in policies:
        raise ValueError("unsupported automatic prompt pack version")
    if run.configuration.get("prompt_context_policy") != policies[version]:
        raise ValueError("unsupported automatic prompt context policy")
    topics = extract_prompt_topics(run.site.domain, run.pages)
    if version == SELECTED_PROMPT_PACK_VERSION:
        selection = run.configuration.get("prompt_topic_selection")
        if not isinstance(selection, list):
            raise ValueError("missing or invalid explicit prompt topic selection")
        selected_topics = tuple(PromptTopic.model_validate(value) for value in selection)
        resolved = resolve_selected_prompt_topics(run.site.domain, run.pages, selected_topics)
        if selection != [topic.model_dump(mode="json") for topic in resolved]:
            raise ValueError("saved prompt topic selection is not canonical")
        topics += resolved
    if run.configuration.get("prompt_context") != [
        topic.model_dump(mode="json") for topic in topics
    ]:
        raise ValueError("saved prompt context does not match source observations")
    if run.ai_prompts != generate_prompt_pack(
        run.site, topics=topics, pack_version=version, pages=run.pages
    ):
        raise ValueError("saved prompts do not match the source-bound prompt pack")


def import_observations(items: list[dict[str, Any]]) -> list[AIObservation]:
    return [AIObservation.model_validate(item) for item in items]
