from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import ai_search_audit.data_requests as data_requests_module
from ai_search_audit.data_requests import (
    DataRequestContext,
    EntityKind,
    ObservedFeature,
    RequestImportance,
    build_data_request,
    render_data_request_markdown,
    write_data_request,
)


def request_context(
    *,
    entity_kind: EntityKind = EntityKind.GENERIC,
    locale: str = "en",
    observed_features: tuple[ObservedFeature, ...] = (),
) -> DataRequestContext:
    return DataRequestContext(
        project_id="north-star",
        canonical_domain="north-star.example",
        entity_kind=entity_kind,
        locale=locale,
        observed_features=observed_features,
        detected_languages=("en", "pl"),
        detected_markets=("GB", "PL"),
    )


def locale_payload(locale: str) -> dict[str, object]:
    resource = data_requests_module.resources.files("ai_search_audit").joinpath(
        "templates", "data-request", f"{locale}.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


@pytest.mark.parametrize("locale", ("en", "pl"))
@pytest.mark.parametrize(
    "entity_kind",
    (EntityKind.GENERIC, EntityKind.ECOMMERCE, EntityKind.LOCAL),
)
def test_every_pack_contains_owner_gsc_and_separate_ga4_requests(
    locale: str, entity_kind: EntityKind
) -> None:
    pack = build_data_request(request_context(entity_kind=entity_kind, locale=locale))
    modules = [item.module for item in pack.items]

    assert modules[0] == "owner_context"
    assert modules.count("google_search_console") == 5
    assert modules.count("ga4_organic_search") == 1
    assert modules.count("ga4_ai_assistant") == 1


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_gsc_request_uses_supported_window_and_discloses_ui_export_limit(locale: str) -> None:
    pack = build_data_request(request_context(locale=locale))
    item = next(item for item in pack.items if item.module == "google_search_console")
    searchable = " ".join(
        (
            item.date_range,
            item.rationale,
            *item.export_instructions,
        )
    ).lower()

    assert "16" in item.date_range
    assert "1,000" in searchable or "1000" in searchable
    assert "representative" in searchable or "reprezentatywn" in searchable
    assert "search console" in item.source.lower()
    assert {"query", "page"}.issubset({value.lower() for value in item.dimensions}) or {
        "zapytanie",
        "strona",
    }.issubset({value.lower() for value in item.dimensions})


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_gsc_requests_five_distinct_reports_with_exact_coverage(locale: str) -> None:
    pack = build_data_request(request_context(locale=locale))
    items = [item for item in pack.items if item.module == "google_search_console"]
    by_id = {item.item_id: item for item in items}

    assert tuple(by_id) == (
        "google_search_console",
        "google_search_console_page_indexing",
        "google_search_console_sitemaps",
        "google_search_console_core_web_vitals",
        "google_search_console_crawl_stats",
    )
    expected_report_terms = {
        "google_search_console": ("search results", "wyniki wyszukiwania"),
        "google_search_console_page_indexing": ("page indexing", "indeksowanie stron"),
        "google_search_console_sitemaps": ("sitemaps", "mapy witryny"),
        "google_search_console_core_web_vitals": (
            "core web vitals",
            "podstawowe wskaźniki internetowe",
        ),
        "google_search_console_crawl_stats": (
            "crawl stats",
            "statystyki indeksowania",
        ),
    }
    for item_id, terms in expected_report_terms.items():
        assert any(term in by_id[item_id].exact_report.lower() for term in terms)

    performance = by_id["google_search_console"]
    searchable_dimensions = " ".join(performance.dimensions).lower()
    for alternatives in (
        ("query", "zapytanie"),
        ("page", "strona"),
        ("country", "kraj"),
        ("device", "urządzenie"),
        ("search appearance", "wygląd w wyszukiwarce"),
    ):
        assert any(term in searchable_dimensions for term in alternatives)
    searchable_filters = " ".join(performance.filters).lower()
    assert "web" in searchable_filters or "sieć" in searchable_filters
    assert "other relevant search types" in searchable_filters or (
        "inne istotne typy wyszukiwania" in searchable_filters
    )


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_owner_context_explicitly_requests_all_audit_facts(locale: str) -> None:
    pack = build_data_request(request_context(locale=locale))
    owner = next(item for item in pack.items if item.module == "owner_context")
    searchable = " ".join(
        (
            owner.exact_report,
            *owner.dimensions,
            *owner.filters,
            *owner.export_instructions,
            owner.rationale,
        )
    ).lower()

    for alternatives in (
        ("canonical facts", "dane kanoniczne"),
        ("markets", "rynki"),
        ("languages", "języki"),
        ("seasonality", "sezonowość"),
        ("controlled profiles", "kontrolowane profile"),
        ("competitors", "konkurenci"),
        ("unresolved contradictions", "nierozstrzygnięte sprzeczności"),
        ("known implementation dates", "znane daty wdrożeń"),
    ):
        assert any(term in searchable for term in alternatives)


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_ga4_requests_only_visibility_metrics_and_accept_zero_ai_baseline(locale: str) -> None:
    pack = build_data_request(request_context(locale=locale))
    organic = next(item for item in pack.items if item.module == "ga4_organic_search")
    ai = next(item for item in pack.items if item.module == "ga4_ai_assistant")

    assert "16" in organic.date_range
    assert "16" in ai.date_range
    assert any("Organic Search" in value for value in organic.filters)
    assert any("AI Assistant" in value for value in ai.filters)
    assert organic.filters != ai.filters
    assert set(organic.metrics) == set(ai.metrics)
    assert len(organic.metrics) == 2
    assert all(
        any(term in metric.lower() for term in ("session", "user", "sesj", "użytkowni"))
        for metric in organic.metrics
    )
    assert any(term in " ".join(ai.export_instructions).lower() for term in ("zero", "zerow"))
    assert any(
        term in " ".join(ai.export_instructions).lower()
        for term in ("valid baseline", "prawidłow", "poprawn")
    )
    assert any(
        "landing" in dimension.lower() or "stron" in dimension.lower()
        for dimension in organic.dimensions
    )


def test_generic_request_does_not_leak_irrelevant_modules() -> None:
    pack = build_data_request(request_context(entity_kind=EntityKind.GENERIC))
    assert "merchant_center" not in {item.module for item in pack.items}
    assert "google_business_profile" not in {item.module for item in pack.items}


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_merchant_center_is_ecommerce_only_and_requests_visibility_diagnostics(
    locale: str,
) -> None:
    ecommerce = build_data_request(request_context(entity_kind=EntityKind.ECOMMERCE, locale=locale))
    other_modules = {
        item.module
        for kind in (EntityKind.GENERIC, EntityKind.LOCAL)
        for item in build_data_request(request_context(entity_kind=kind, locale=locale)).items
    }
    item = next(item for item in ecommerce.items if item.module == "merchant_center")
    searchable = " ".join(
        (
            item.exact_report,
            *item.metrics,
            *item.dimensions,
            *item.filters,
            *item.export_instructions,
        )
    ).lower()

    assert "merchant_center" not in other_modules
    for concept in ("impression", "click", "ctr"):
        assert concept in searchable
    assert "paid" in searchable or "płatn" in searchable
    assert "organic" in searchable or "bezpłatn" in searchable
    assert "product issue" in searchable or "problem z produkt" in searchable
    assert "account issue" in searchable or "problem z kont" in searchable


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_google_business_profile_is_local_only_and_requests_profile_visibility(
    locale: str,
) -> None:
    local = build_data_request(request_context(entity_kind=EntityKind.LOCAL, locale=locale))
    other_modules = {
        item.module
        for kind in (EntityKind.GENERIC, EntityKind.ECOMMERCE)
        for item in build_data_request(request_context(entity_kind=kind, locale=locale)).items
    }
    item = next(item for item in local.items if item.module == "google_business_profile")
    searchable = " ".join(
        (
            item.exact_report,
            *item.metrics,
            *item.dimensions,
            *item.export_instructions,
        )
    ).lower()

    assert "google_business_profile" not in other_modules
    for alternatives in (
        ("profile facts", "dane profilu"),
        ("categories", "kategorie"),
        ("locations", "lokalizacje"),
        ("google updates", "aktualizacje google"),
        ("views", "wyświetlenia"),
        ("searches", "wyszukiwania"),
        ("website clicks", "kliknięcia w witrynę"),
    ):
        assert any(term in searchable for term in alternatives)


_CONDITIONAL_FEATURES = (
    (ObservedFeature.BING_WEBMASTER_TOOLS, "bing_webmaster_tools"),
    (ObservedFeature.SANITIZED_LOGS, "sanitized_logs"),
    (ObservedFeature.FULL_CRAWL_EXPORT, "crawl_export"),
    (ObservedFeature.AI_VISIBILITY_MONITORING, "ai_monitoring_export"),
)


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_conditional_sources_are_absent_without_observed_evidence(locale: str) -> None:
    modules = {item.module for item in build_data_request(request_context(locale=locale)).items}

    assert modules.isdisjoint(module for _, module in _CONDITIONAL_FEATURES)


@pytest.mark.parametrize("locale", ("en", "pl"))
@pytest.mark.parametrize("entity_kind", tuple(EntityKind))
@pytest.mark.parametrize(("feature", "expected_module"), _CONDITIONAL_FEATURES)
def test_each_conditional_source_is_selected_independently_when_observed(
    locale: str,
    entity_kind: EntityKind,
    feature: ObservedFeature,
    expected_module: str,
) -> None:
    pack = build_data_request(
        request_context(
            locale=locale,
            entity_kind=entity_kind,
            observed_features=(feature,),
        )
    )
    selected_modules = {
        item.module
        for item in pack.items
        if item.module in {module for _, module in _CONDITIONAL_FEATURES}
    }

    assert selected_modules == {expected_module}
    assert pack.observed_features == (feature,)


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_selected_conditional_sources_are_optional_and_scoped(locale: str) -> None:
    pack = build_data_request(
        request_context(
            locale=locale,
            observed_features=tuple(feature for feature, _ in _CONDITIONAL_FEATURES),
        )
    )
    by_module = {item.module: item for item in pack.items}

    for module in (
        "bing_webmaster_tools",
        "sanitized_logs",
        "crawl_export",
        "ai_monitoring_export",
    ):
        item = by_module[module]
        assert item.importance is RequestImportance.OPTIONAL
        assert item.condition != ""

    bing = by_module["bing_webmaster_tools"]
    searchable = " ".join(
        (
            bing.exact_report,
            *bing.metrics,
            *bing.dimensions,
            *bing.export_instructions,
        )
    ).lower()
    for alternatives in (
        ("performance", "skuteczność"),
        ("queries", "zapytania"),
        ("pages", "strony"),
        ("indexing", "indeksowa"),
        ("sitemaps", "mapy witryny"),
        ("crawl issues", "problemy z indeksowaniem", "problemy crawlowania"),
    ):
        assert any(term in searchable for term in alternatives)


@pytest.mark.parametrize("locale", ("en", "pl"))
@pytest.mark.parametrize(
    "entity_kind",
    (EntityKind.GENERIC, EntityKind.ECOMMERCE, EntityKind.LOCAL),
)
def test_items_have_complete_handoff_metadata(locale: str, entity_kind: EntityKind) -> None:
    pack = build_data_request(
        request_context(
            entity_kind=entity_kind,
            locale=locale,
            observed_features=tuple(ObservedFeature),
        )
    )

    assert pack.deletion_policy
    for item in pack.items:
        assert item.source
        assert item.exact_report
        assert item.date_range
        assert item.dimensions
        assert item.filters
        assert item.preferred_formats
        assert item.rationale
        assert item.export_instructions
        assert item.anonymization
        assert item.deletion_notice == pack.deletion_policy


def test_required_items_support_named_claims_not_report_generation() -> None:
    for locale in ("en", "pl"):
        pack = build_data_request(request_context(locale=locale))
        required = [item for item in pack.items if item.importance is RequestImportance.REQUIRED]
        assert required
        assert all(item.claim_supported for item in required)
        searchable = " ".join(
            (
                pack.importance_definitions.required,
                *(item.rationale for item in required),
                *(item.claim_supported or "" for item in required),
            )
        ).lower()
        for forbidden in (
            "required to generate",
            "required to create the report",
            "wymagane do wygenerowania",
            "wymagane do utworzenia raportu",
        ):
            assert forbidden not in searchable


@pytest.mark.parametrize("locale", ("en", "pl"))
@pytest.mark.parametrize(
    "entity_kind",
    (EntityKind.GENERIC, EntityKind.ECOMMERCE, EntityKind.LOCAL),
)
def test_request_never_asks_for_business_outcomes_or_account_access(
    locale: str, entity_kind: EntityKind
) -> None:
    pack = build_data_request(
        request_context(
            entity_kind=entity_kind,
            locale=locale,
            observed_features=tuple(ObservedFeature),
        )
    )
    searchable = json.dumps(pack.model_dump(mode="json"), ensure_ascii=False).lower()

    for forbidden in (
        "lead",
        "formular",
        "conversion",
        "konwersj",
        "revenue",
        "przych",
        "crm",
        "credential",
        "login",
        "oauth",
        "api",
        "mcp",
        "plugin access",
        "direct account access",
        "bezpośredni dostęp do konta",
    ):
        assert forbidden not in searchable


def test_models_are_deeply_frozen_and_reject_required_without_claim() -> None:
    context = request_context()
    pack = build_data_request(context)

    with pytest.raises(ValidationError, match="frozen"):
        context.detected_markets = ("US",)
    with pytest.raises(ValidationError, match="frozen"):
        context.observed_features = (ObservedFeature.SANITIZED_LOGS,)
    with pytest.raises(ValidationError, match="frozen"):
        pack.items[0].module = "changed"
    with pytest.raises(ValidationError, match="named claim"):
        pack.items[0].model_copy(
            update={"importance": RequestImportance.REQUIRED, "claim_supported": None}
        ).model_validate(pack.items[0].model_dump() | {"claim_supported": None})


@pytest.mark.parametrize(
    "detected_languages",
    (
        ("",),
        ("   ",),
        ("en\nPL",),
        ("english language",),
        ("x" * 64,),
    ),
)
def test_context_rejects_invalid_detected_language_tags(
    detected_languages: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="language"):
        DataRequestContext(
            project_id="north-star",
            canonical_domain="north-star.example",
            entity_kind=EntityKind.GENERIC,
            locale="en",
            detected_languages=detected_languages,
        )


@pytest.mark.parametrize(
    "detected_markets",
    (
        ("",),
        ("   ",),
        ("Poland\nInjected heading",),
        ("Poland\u2028Injected heading",),
        ("Poland\x00Hidden",),
        ("x" * 81,),
    ),
)
def test_context_rejects_unsafe_detected_market_labels(
    detected_markets: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="market"):
        DataRequestContext(
            project_id="north-star",
            canonical_domain="north-star.example",
            entity_kind=EntityKind.GENERIC,
            locale="en",
            detected_markets=detected_markets,
        )


def test_context_canonicalizes_deduplicates_and_sorts_detected_metadata() -> None:
    context = DataRequestContext(
        project_id="north-star",
        canonical_domain="north-star.example",
        entity_kind=EntityKind.GENERIC,
        locale="en",
        detected_languages=("pl", "EN-us", "en-US", "de"),
        detected_markets=("Poland", "United Kingdom", "poland", "Poland", "Austria"),
    )

    assert context.detected_languages == ("de", "en-US", "pl")
    assert context.detected_markets == ("Austria", "Poland", "United Kingdom")


def test_context_normalizes_market_labels_to_nfc_before_deduplication() -> None:
    context = DataRequestContext(
        project_id="north-star",
        canonical_domain="north-star.example",
        entity_kind=EntityKind.GENERIC,
        locale="en",
        detected_markets=("Cafe\u0301", "Café"),
    )

    assert context.detected_markets == ("Café",)


def test_markdown_escapes_context_derived_metacharacters() -> None:
    context = DataRequestContext(
        project_id="north-star",
        canonical_domain="north-star.example",
        entity_kind=EntityKind.GENERIC,
        locale="en",
        detected_languages=("en",),
        detected_markets=("North [Pilot](https://evil.example) | *priority* #1",),
    )

    markdown = render_data_request_markdown(build_data_request(context))

    assert "[Pilot](https://evil.example)" not in markdown
    assert "*priority*" not in markdown
    assert r"North \[Pilot\]\(https://evil\.example\) \| \*priority\* \#1" in markdown


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_json_and_markdown_outputs_are_deterministic_from_frozen_input(
    tmp_path: Path, locale: str
) -> None:
    context = request_context(entity_kind=EntityKind.ECOMMERCE, locale=locale)
    pack = build_data_request(context)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_markdown, first_json = write_data_request(pack, first_root)
    second_markdown, second_json = write_data_request(pack, second_root)

    assert first_markdown.name == f"next-audit-data-request_{locale}.md"
    assert first_json.name == "next-audit-data-request.json"
    assert first_markdown.read_bytes() == second_markdown.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_text(encoding="utf-8") == render_data_request_markdown(pack)
    assert first_markdown.read_bytes().endswith(b"\n")
    assert first_json.read_bytes().endswith(b"\n")
    assert json.loads(first_json.read_text(encoding="utf-8")) == pack.model_dump(mode="json")


def test_fixed_content_and_headings_are_localized_with_shared_schema() -> None:
    english = build_data_request(request_context(locale="en"))
    polish = build_data_request(request_context(locale="pl"))
    english_markdown = render_data_request_markdown(english)
    polish_markdown = render_data_request_markdown(polish)

    assert english.locale == "en"
    assert polish.locale == "pl"
    assert [item.module for item in english.items] == [item.module for item in polish.items]
    assert "# Next-audit data request" in english_markdown
    assert "## Requested files" in english_markdown
    assert "# Prośba o dane do kolejnego audytu" in polish_markdown
    assert "## Proszę przygotować" in polish_markdown
    assert "Next-audit data request" not in polish_markdown
    assert "Requested files" not in polish_markdown
    assert english.items[0].rationale != polish.items[0].rationale


def test_locale_resource_validation_fails_clearly_for_malformed_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "templates" / "data-request"
    resource.mkdir(parents=True)
    (resource / "en.json").write_text('{"schema_version":"1.0.0"}', encoding="utf-8")
    monkeypatch.setattr(data_requests_module.resources, "files", lambda _package: tmp_path)

    with pytest.raises(RuntimeError, match="invalid data-request locale resource.*en"):
        data_requests_module._load_locale_resource("en")


def test_locale_resource_validation_fails_clearly_for_missing_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_requests_module.resources, "files", lambda _package: tmp_path)

    with pytest.raises(RuntimeError, match="invalid data-request locale resource.*pl"):
        data_requests_module._load_locale_resource("pl")


def test_locale_resource_wraps_resource_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_discovery(_package: str) -> Path:
        raise OSError("resource discovery failed")

    monkeypatch.setattr(data_requests_module.resources, "files", fail_discovery)

    with pytest.raises(
        RuntimeError,
        match="invalid data-request locale resource.*en.*resource discovery failed",
    ):
        data_requests_module._load_locale_resource("en")


def test_locale_resource_rejects_whitespace_only_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = locale_payload("en")
    items = payload["items"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    first["title"] = "   "
    resource = tmp_path / "templates" / "data-request"
    resource.mkdir(parents=True)
    (resource / "en.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(data_requests_module.resources, "files", lambda _package: tmp_path)

    with pytest.raises(RuntimeError, match="invalid data-request locale resource.*en"):
        data_requests_module._load_locale_resource("en")


@pytest.mark.parametrize("locale", ("en", "pl"))
def test_locale_items_contain_no_behavior_metadata(locale: str) -> None:
    payload = locale_payload(locale)
    items = payload["items"]
    assert isinstance(items, list)

    for item in items:
        assert isinstance(item, dict)
        assert {"module", "importance", "applicable_entity_kinds", "required_feature"}.isdisjoint(
            item
        )


def test_locale_item_cannot_override_canonical_feature_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = locale_payload("en")
    items = payload["items"]
    assert isinstance(items, list)
    conditional = next(
        item
        for item in items
        if isinstance(item, dict) and item.get("item_id") == "bing_webmaster_tools"
    )
    conditional["module"] = "owner_context"
    resource = tmp_path / "templates" / "data-request"
    resource.mkdir(parents=True)
    (resource / "en.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(data_requests_module.resources, "files", lambda _package: tmp_path)

    with pytest.raises(RuntimeError, match="invalid data-request locale resource.*en"):
        data_requests_module._load_locale_resource("en")


def test_failed_second_promotion_restores_previous_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = build_data_request(request_context(locale="en"))
    markdown_path, json_path = write_data_request(pack, tmp_path)
    markdown_path.write_text("old markdown\n", encoding="utf-8")
    json_path.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = os.replace
    promotions = 0

    def fail_second_promotion(source: str | Path, destination: str | Path) -> None:
        nonlocal promotions
        source_path = Path(source)
        destination_path = Path(destination)
        is_promotion = (
            source_path.parent.name.startswith(".data-request-")
            and destination_path.parent == tmp_path
        )
        if is_promotion:
            promotions += 1
            if promotions == 2:
                raise OSError("simulated second promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(data_requests_module.os, "replace", fail_second_promotion)

    with pytest.raises(OSError, match="simulated second promotion failure"):
        write_data_request(pack, tmp_path)

    assert markdown_path.read_text(encoding="utf-8") == "old markdown\n"
    assert json_path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not list(tmp_path.glob(".data-request-*"))


def test_failed_second_promotion_leaves_no_partial_new_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = build_data_request(request_context(locale="pl"))
    real_replace = os.replace
    promotions = 0

    def fail_second_promotion(source: str | Path, destination: str | Path) -> None:
        nonlocal promotions
        source_path = Path(source)
        destination_path = Path(destination)
        is_promotion = (
            source_path.parent.name.startswith(".data-request-")
            and destination_path.parent == tmp_path
        )
        if is_promotion:
            promotions += 1
            if promotions == 2:
                raise OSError("simulated second promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(data_requests_module.os, "replace", fail_second_promotion)

    with pytest.raises(OSError, match="simulated second promotion failure"):
        write_data_request(pack, tmp_path)

    assert not (tmp_path / "next-audit-data-request_pl.md").exists()
    assert not (tmp_path / "next-audit-data-request.json").exists()
    assert not list(tmp_path.glob(".data-request-*"))


def test_keyboard_interrupt_during_second_promotion_restores_previous_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = build_data_request(request_context(locale="en"))
    markdown_path, json_path = write_data_request(pack, tmp_path)
    markdown_path.write_text("old markdown\n", encoding="utf-8")
    json_path.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = os.replace

    def interrupt_second_promotion(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name == "next-audit-data-request.json"
            and source_path.parent.name.startswith(".data-request-")
            and destination_path == json_path
        ):
            raise KeyboardInterrupt
        real_replace(source, destination)

    monkeypatch.setattr(data_requests_module.os, "replace", interrupt_second_promotion)

    with pytest.raises(KeyboardInterrupt):
        write_data_request(pack, tmp_path)

    assert markdown_path.read_text(encoding="utf-8") == "old markdown\n"
    assert json_path.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not list(tmp_path.glob(".data-request-*"))


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
def test_interrupt_after_backup_move_restores_exact_previous_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
) -> None:
    pack = build_data_request(request_context(locale="en"))
    markdown_path, json_path = write_data_request(pack, tmp_path)
    old_markdown = b"old markdown bytes\n"
    old_json = b'{"old": "json bytes"}\n'
    markdown_path.write_bytes(old_markdown)
    json_path.write_bytes(old_json)
    real_replace = os.replace

    def interrupt_after_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        real_replace(source, destination)
        if source_path == markdown_path and destination_path.name == "previous.md":
            raise interrupt_type("after Markdown backup move")

    monkeypatch.setattr(data_requests_module.os, "replace", interrupt_after_replace)

    with pytest.raises(interrupt_type, match="after Markdown backup move"):
        write_data_request(pack, tmp_path)

    assert markdown_path.read_bytes() == old_markdown
    assert json_path.read_bytes() == old_json
    assert not list(tmp_path.glob(".data-request-*"))


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
def test_interrupt_after_first_new_promotion_leaves_no_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
) -> None:
    pack = build_data_request(request_context(locale="pl"))
    markdown_path = tmp_path / "next-audit-data-request_pl.md"
    json_path = tmp_path / "next-audit-data-request.json"
    real_replace = os.replace

    def interrupt_after_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        real_replace(source, destination)
        if (
            source_path.name == markdown_path.name
            and source_path.parent.name.startswith(".data-request-")
            and destination_path == markdown_path
        ):
            raise interrupt_type("after first new promotion")

    monkeypatch.setattr(data_requests_module.os, "replace", interrupt_after_replace)

    with pytest.raises(interrupt_type, match="after first new promotion"):
        write_data_request(pack, tmp_path)

    assert not markdown_path.exists()
    assert not json_path.exists()
    assert not list(tmp_path.glob(".data-request-*"))


def test_rollback_failure_removes_new_outputs_and_preserves_recovery_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = build_data_request(request_context(locale="pl"))
    markdown_path, json_path = write_data_request(pack, tmp_path)
    markdown_path.write_text("old markdown\n", encoding="utf-8")
    json_path.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = os.replace

    def fail_promotion_and_json_restore(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name == "next-audit-data-request.json" and destination_path == json_path:
            raise OSError("second promotion failed")
        if source_path.name == "previous.json" and destination_path == json_path:
            raise OSError("json rollback failed")
        real_replace(source, destination)

    monkeypatch.setattr(
        data_requests_module.os,
        "replace",
        fail_promotion_and_json_restore,
    )

    with pytest.raises(RuntimeError) as captured:
        write_data_request(pack, tmp_path)

    message = str(captured.value)
    assert "second promotion failed" in message
    assert "json rollback failed" in message
    assert markdown_path.read_text(encoding="utf-8") == "old markdown\n"
    assert not json_path.exists()
    staging = list(tmp_path.glob(".data-request-*"))
    assert len(staging) == 1
    assert (staging[0] / "previous.json").read_text(encoding="utf-8") == '{"old": true}\n'
