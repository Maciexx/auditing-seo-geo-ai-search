from __future__ import annotations

import html
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ai_search_audit.project_models import normalize_canonical_domain, validate_project_id

Locale = Literal["en", "pl"]
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_MARKDOWN_METACHARACTER = re.compile(r"([\\`*_[\]{}()#+.!|>\-])")


def _require_nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("localized prose must not be blank")
    return value


NonBlankText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=10_000),
    AfterValidator(_require_nonblank),
]


class FrozenDataRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class EntityKind(StrEnum):
    GENERIC = "generic"
    ECOMMERCE = "ecommerce"
    LOCAL = "local"


class ObservedFeature(StrEnum):
    BING_WEBMASTER_TOOLS = "bing_webmaster_tools"
    SANITIZED_LOGS = "sanitized_logs"
    FULL_CRAWL_EXPORT = "full_crawl_export"
    AI_VISIBILITY_MONITORING = "ai_visibility_monitoring"


class RequestImportance(StrEnum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class DataRequestModule(StrEnum):
    OWNER_CONTEXT = "owner_context"
    GOOGLE_SEARCH_CONSOLE = "google_search_console"
    GA4_ORGANIC_SEARCH = "ga4_organic_search"
    GA4_AI_ASSISTANT = "ga4_ai_assistant"
    MERCHANT_CENTER = "merchant_center"
    GOOGLE_BUSINESS_PROFILE = "google_business_profile"
    BING_WEBMASTER_TOOLS = "bing_webmaster_tools"
    SANITIZED_LOGS = "sanitized_logs"
    CRAWL_EXPORT = "crawl_export"
    AI_MONITORING_EXPORT = "ai_monitoring_export"


@dataclass(frozen=True, slots=True)
class _CanonicalItem:
    module: DataRequestModule
    importance: RequestImportance
    applicable_entity_kinds: tuple[EntityKind, ...] = ()
    required_feature: ObservedFeature | None = None


_ITEM_CATALOG: Mapping[str, _CanonicalItem] = MappingProxyType(
    {
        "owner_context": _CanonicalItem(
            DataRequestModule.OWNER_CONTEXT, RequestImportance.REQUIRED
        ),
        "google_search_console": _CanonicalItem(
            DataRequestModule.GOOGLE_SEARCH_CONSOLE, RequestImportance.REQUIRED
        ),
        "google_search_console_page_indexing": _CanonicalItem(
            DataRequestModule.GOOGLE_SEARCH_CONSOLE, RequestImportance.REQUIRED
        ),
        "google_search_console_sitemaps": _CanonicalItem(
            DataRequestModule.GOOGLE_SEARCH_CONSOLE, RequestImportance.REQUIRED
        ),
        "google_search_console_core_web_vitals": _CanonicalItem(
            DataRequestModule.GOOGLE_SEARCH_CONSOLE, RequestImportance.REQUIRED
        ),
        "google_search_console_crawl_stats": _CanonicalItem(
            DataRequestModule.GOOGLE_SEARCH_CONSOLE, RequestImportance.REQUIRED
        ),
        "ga4_organic_search": _CanonicalItem(
            DataRequestModule.GA4_ORGANIC_SEARCH, RequestImportance.REQUIRED
        ),
        "ga4_ai_assistant": _CanonicalItem(
            DataRequestModule.GA4_AI_ASSISTANT, RequestImportance.REQUIRED
        ),
        "merchant_center": _CanonicalItem(
            DataRequestModule.MERCHANT_CENTER,
            RequestImportance.RECOMMENDED,
            (EntityKind.ECOMMERCE,),
        ),
        "google_business_profile": _CanonicalItem(
            DataRequestModule.GOOGLE_BUSINESS_PROFILE,
            RequestImportance.RECOMMENDED,
            (EntityKind.LOCAL,),
        ),
        "bing_webmaster_tools": _CanonicalItem(
            DataRequestModule.BING_WEBMASTER_TOOLS,
            RequestImportance.OPTIONAL,
            required_feature=ObservedFeature.BING_WEBMASTER_TOOLS,
        ),
        "sanitized_logs": _CanonicalItem(
            DataRequestModule.SANITIZED_LOGS,
            RequestImportance.OPTIONAL,
            required_feature=ObservedFeature.SANITIZED_LOGS,
        ),
        "crawl_export": _CanonicalItem(
            DataRequestModule.CRAWL_EXPORT,
            RequestImportance.OPTIONAL,
            required_feature=ObservedFeature.FULL_CRAWL_EXPORT,
        ),
        "ai_monitoring_export": _CanonicalItem(
            DataRequestModule.AI_MONITORING_EXPORT,
            RequestImportance.OPTIONAL,
            required_feature=ObservedFeature.AI_VISIBILITY_MONITORING,
        ),
    }
)


class DataRequestContext(FrozenDataRequestModel):
    project_id: str
    canonical_domain: str
    entity_kind: EntityKind
    locale: Locale
    observed_features: tuple[ObservedFeature, ...] = ()
    detected_languages: tuple[str, ...] = ()
    detected_markets: tuple[str, ...] = ()

    @field_validator("project_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("canonical_domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        return normalize_canonical_domain(value)

    @field_validator("observed_features")
    @classmethod
    def normalize_observed_features(
        cls, value: tuple[ObservedFeature, ...]
    ) -> tuple[ObservedFeature, ...]:
        return tuple(feature for feature in ObservedFeature if feature in value)

    @field_validator("detected_languages")
    @classmethod
    def normalize_detected_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: set[str] = set()
        for tag in value:
            if len(tag) > 63 or not _LANGUAGE_TAG.fullmatch(tag):
                raise ValueError("detected language must be a bounded BCP47-like tag")
            parts = tag.split("-")
            canonical_parts = [parts[0].lower()]
            for part in parts[1:]:
                if len(part) == 2 and part.isalpha():
                    canonical_parts.append(part.upper())
                elif len(part) == 4 and part.isalpha():
                    canonical_parts.append(part.title())
                else:
                    canonical_parts.append(part.lower())
            normalized.add("-".join(canonical_parts))
        return tuple(sorted(normalized, key=str.casefold))

    @field_validator("detected_markets")
    @classmethod
    def normalize_detected_markets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: dict[str, str] = {}
        for raw_market in value:
            market = unicodedata.normalize("NFC", raw_market)
            has_unsafe_character = any(
                unicodedata.category(character).startswith("C")
                or unicodedata.category(character) in {"Zl", "Zp"}
                for character in market
            )
            if not market or market != market.strip() or len(market) > 80 or has_unsafe_character:
                raise ValueError("detected market must be bounded, nonblank safe text")
            deduplication_key = market.casefold()
            existing = normalized.get(deduplication_key)
            if existing is None or market < existing:
                normalized[deduplication_key] = market
        return tuple(sorted(normalized.values(), key=lambda item: (item.casefold(), item)))


class LocalizedImportance(FrozenDataRequestModel):
    required: NonBlankText
    recommended: NonBlankText
    optional: NonBlankText


class LocalizedEntityLabels(FrozenDataRequestModel):
    generic: NonBlankText
    ecommerce: NonBlankText
    local: NonBlankText


class DataRequestHeadings(FrozenDataRequestModel):
    title: NonBlankText
    project: NonBlankText
    domain: NonBlankText
    entity: NonBlankText
    languages: NonBlankText
    markets: NonBlankText
    importance_guide: NonBlankText
    requested_files: NonBlankText
    source: NonBlankText
    exact_report: NonBlankText
    importance: NonBlankText
    date_range: NonBlankText
    metrics: NonBlankText
    dimensions: NonBlankText
    filters: NonBlankText
    preferred_formats: NonBlankText
    rationale: NonBlankText
    export_instructions: NonBlankText
    anonymization: NonBlankText
    condition: NonBlankText
    claim_supported: NonBlankText
    deletion_policy: NonBlankText


class DataRequestItem(FrozenDataRequestModel):
    item_id: str = Field(min_length=1)
    module: DataRequestModule
    title: NonBlankText
    source: NonBlankText
    exact_report: NonBlankText
    importance: RequestImportance
    date_range: NonBlankText
    metrics: tuple[NonBlankText, ...] = Field(min_length=1)
    dimensions: tuple[NonBlankText, ...] = Field(min_length=1)
    filters: tuple[NonBlankText, ...] = Field(min_length=1)
    preferred_formats: tuple[NonBlankText, ...] = Field(min_length=1)
    rationale: NonBlankText
    export_instructions: tuple[NonBlankText, ...] = Field(min_length=1)
    anonymization: NonBlankText
    deletion_notice: NonBlankText
    condition: NonBlankText
    claim_supported: NonBlankText | None = None

    @model_validator(mode="after")
    def required_items_name_the_supported_claim(self) -> DataRequestItem:
        if self.importance is RequestImportance.REQUIRED and not self.claim_supported:
            raise ValueError("required data must support a named claim or comparison")
        return self


class DataRequestPack(FrozenDataRequestModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    version: Literal["1.0.0"] = "1.0.0"
    project_id: str
    canonical_domain: str
    entity_kind: EntityKind
    entity_label: NonBlankText
    locale: Locale
    observed_features: tuple[ObservedFeature, ...]
    detected_languages: tuple[str, ...]
    detected_markets: tuple[str, ...]
    introduction: NonBlankText
    headings: DataRequestHeadings
    importance_labels: LocalizedImportance
    importance_definitions: LocalizedImportance
    items: tuple[DataRequestItem, ...] = Field(min_length=1)
    deletion_policy: NonBlankText


class _LocaleItem(FrozenDataRequestModel):
    item_id: str = Field(min_length=1)
    title: NonBlankText
    source: NonBlankText
    exact_report: NonBlankText
    date_range: NonBlankText
    metrics: tuple[NonBlankText, ...] = Field(min_length=1)
    dimensions: tuple[NonBlankText, ...] = Field(min_length=1)
    filters: tuple[NonBlankText, ...] = Field(min_length=1)
    preferred_formats: tuple[NonBlankText, ...] = Field(min_length=1)
    rationale: NonBlankText
    export_instructions: tuple[NonBlankText, ...] = Field(min_length=1)
    anonymization: NonBlankText
    condition: NonBlankText
    claim_supported: NonBlankText | None = None


class _LocaleResource(FrozenDataRequestModel):
    schema_version: Literal["1.0.0"]
    locale: Locale
    introduction: NonBlankText
    headings: DataRequestHeadings
    importance_labels: LocalizedImportance
    importance_definitions: LocalizedImportance
    entity_labels: LocalizedEntityLabels
    deletion_policy: NonBlankText
    items: tuple[_LocaleItem, ...]

    @model_validator(mode="after")
    def validate_complete_catalog(self) -> _LocaleResource:
        item_ids = tuple(item.item_id for item in self.items)
        duplicate_ids = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
        unknown_ids = sorted(set(item_ids).difference(_ITEM_CATALOG))
        missing_ids = sorted(set(_ITEM_CATALOG).difference(item_ids))
        if duplicate_ids or unknown_ids or missing_ids:
            raise ValueError(
                "locale item IDs must match the canonical catalog exactly; "
                f"duplicates={duplicate_ids}, unknown={unknown_ids}, missing={missing_ids}"
            )
        by_id = {item.item_id: item for item in self.items}
        missing_claims = [
            item_id
            for item_id, behavior in _ITEM_CATALOG.items()
            if behavior.importance is RequestImportance.REQUIRED
            and not by_id[item_id].claim_supported
        ]
        if missing_claims:
            raise ValueError(f"required locale items need named claims: {missing_claims}")
        return self


def _load_locale_resource(locale: Locale) -> _LocaleResource:
    try:
        resource = resources.files("ai_search_audit").joinpath(
            "templates", "data-request", f"{locale}.json"
        )
        raw = resource.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        localized = _LocaleResource.model_validate(parsed)
        if localized.locale != locale:
            raise ValueError("resource locale does not match requested locale")
    except Exception as exc:
        raise RuntimeError(f"invalid data-request locale resource for {locale!r}: {exc}") from exc
    return localized


def _entity_label(resource: _LocaleResource, entity_kind: EntityKind) -> str:
    return {
        EntityKind.GENERIC: resource.entity_labels.generic,
        EntityKind.ECOMMERCE: resource.entity_labels.ecommerce,
        EntityKind.LOCAL: resource.entity_labels.local,
    }[entity_kind]


def _item_applies(item_id: str, context: DataRequestContext) -> bool:
    behavior = _ITEM_CATALOG.get(item_id)
    if behavior is None:
        return False
    entity_applies = (
        not behavior.applicable_entity_kinds
        or context.entity_kind in behavior.applicable_entity_kinds
    )
    feature_applies = (
        behavior.required_feature is None or behavior.required_feature in context.observed_features
    )
    return entity_applies and feature_applies


def build_data_request(context: DataRequestContext) -> DataRequestPack:
    """Build a deterministic, locally rendered request pack for the next audit."""
    resource = _load_locale_resource(context.locale)
    localized_by_id = {item.item_id: item for item in resource.items}
    built_items: list[DataRequestItem] = []
    for item_id, behavior in _ITEM_CATALOG.items():
        template = localized_by_id.get(item_id)
        if template is None:
            raise RuntimeError(f"locale resource is missing canonical item {item_id!r}")
        if not _item_applies(item_id, context):
            continue
        built_items.append(
            DataRequestItem(
                item_id=item_id,
                module=behavior.module,
                title=template.title,
                source=template.source,
                exact_report=template.exact_report,
                importance=behavior.importance,
                date_range=template.date_range,
                metrics=template.metrics,
                dimensions=template.dimensions,
                filters=template.filters,
                preferred_formats=template.preferred_formats,
                rationale=template.rationale,
                export_instructions=template.export_instructions,
                anonymization=template.anonymization,
                deletion_notice=resource.deletion_policy,
                condition=template.condition,
                claim_supported=template.claim_supported,
            )
        )
    items = tuple(built_items)
    return DataRequestPack(
        project_id=context.project_id,
        canonical_domain=context.canonical_domain,
        entity_kind=context.entity_kind,
        entity_label=_entity_label(resource, context.entity_kind),
        locale=context.locale,
        observed_features=context.observed_features,
        detected_languages=context.detected_languages,
        detected_markets=context.detected_markets,
        introduction=resource.introduction,
        headings=resource.headings,
        importance_labels=resource.importance_labels,
        importance_definitions=resource.importance_definitions,
        items=items,
        deletion_policy=resource.deletion_policy,
    )


def _joined(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _escape_markdown_context(value: str) -> str:
    escaped = _MARKDOWN_METACHARACTER.sub(r"\\\1", value)
    return html.escape(escaped, quote=False)


def render_data_request_markdown(pack: DataRequestPack) -> str:
    """Render a request pack without embedding locale-specific prose in code."""
    headings = pack.headings
    lines = [
        f"# {headings.title}",
        "",
        pack.introduction,
        "",
        f"- {headings.project}: `{_escape_markdown_context(pack.project_id)}`",
        f"- {headings.domain}: `{_escape_markdown_context(pack.canonical_domain)}`",
        f"- {headings.entity}: {pack.entity_label}",
        f"- {headings.languages}: "
        f"{_joined(tuple(_escape_markdown_context(item) for item in pack.detected_languages))}",
        f"- {headings.markets}: "
        f"{_joined(tuple(_escape_markdown_context(item) for item in pack.detected_markets))}",
        "",
        f"## {headings.importance_guide}",
        "",
    ]
    for importance in RequestImportance:
        label = getattr(pack.importance_labels, importance.value)
        definition = getattr(pack.importance_definitions, importance.value)
        lines.append(f"- **{label}:** {definition}")
    lines.extend(("", f"## {headings.requested_files}", ""))

    for index, item in enumerate(pack.items, start=1):
        importance_label = getattr(pack.importance_labels, item.importance.value)
        claim = item.claim_supported or ""
        lines.extend(
            (
                f"### {index}. {item.title}",
                "",
                f"- **{headings.source}:** {item.source}",
                f"- **{headings.exact_report}:** {item.exact_report}",
                f"- **{headings.importance}:** {importance_label}",
                f"- **{headings.date_range}:** {item.date_range}",
                f"- **{headings.metrics}:** {_joined(item.metrics)}",
                f"- **{headings.dimensions}:** {_joined(item.dimensions)}",
                f"- **{headings.filters}:** {_joined(item.filters)}",
                f"- **{headings.preferred_formats}:** {_joined(item.preferred_formats)}",
                f"- **{headings.rationale}:** {item.rationale}",
                f"- **{headings.export_instructions}:** {' '.join(item.export_instructions)}",
                f"- **{headings.anonymization}:** {item.anonymization}",
                f"- **{headings.condition}:** {item.condition}",
            )
        )
        if claim:
            lines.append(f"- **{headings.claim_supported}:** {claim}")
        lines.append("")

    lines.extend(
        (
            f"## {headings.deletion_policy}",
            "",
            pack.deletion_policy,
            "",
        )
    )
    return "\n".join(lines)


def _json_text(pack: DataRequestPack) -> str:
    return (
        json.dumps(
            pack.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    )


def _remove_promoted_file(path: Path) -> None:
    if os.path.lexists(path):
        if not path.is_file() and not path.is_symlink():
            raise RuntimeError(f"data-request output path is not a file: {path}")
        path.unlink()


def _error_description(error: BaseException) -> str:
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _move_completed(source: Path, destination: Path, *, intended: bool) -> bool:
    return intended and not os.path.lexists(source) and os.path.lexists(destination)


def write_data_request(pack: DataRequestPack, version_root: Path) -> tuple[Path, Path]:
    """Stage and promote both deterministic outputs, restoring the old pair on failure."""
    version_root = Path(version_root)
    version_root.mkdir(parents=True, exist_ok=True)
    markdown_path = version_root / f"next-audit-data-request_{pack.locale}.md"
    json_path = version_root / "next-audit-data-request.json"
    destinations = (markdown_path, json_path)

    stage_root = Path(tempfile.mkdtemp(prefix=".data-request-", dir=version_root))
    staged = (stage_root / markdown_path.name, stage_root / json_path.name)
    backups = (stage_root / "previous.md", stage_root / "previous.json")
    had_previous = tuple(os.path.lexists(destination) for destination in destinations)
    backup_intended = [False, False]
    promotion_intended = [False, False]

    try:
        staged[0].write_text(render_data_request_markdown(pack), encoding="utf-8")
        staged[1].write_text(_json_text(pack), encoding="utf-8")

        for index, (destination, backup) in enumerate(zip(destinations, backups, strict=True)):
            if had_previous[index]:
                if not destination.is_file() and not destination.is_symlink():
                    raise RuntimeError(f"data-request output path is not a file: {destination}")
                backup_intended[index] = True
                os.replace(destination, backup)

        for index, (source, destination) in enumerate(zip(staged, destinations, strict=True)):
            promotion_intended[index] = True
            os.replace(source, destination)
    except BaseException as primary_error:
        backup_present = [
            backup_intended[index] and os.path.lexists(backup)
            for index, backup in enumerate(backups)
        ]
        new_at_destination = [
            _move_completed(source, destination, intended=promotion_intended[index])
            for index, (source, destination) in enumerate(zip(staged, destinations, strict=True))
        ]
        rollback_errors: list[BaseException] = []
        for index in range(len(destinations) - 1, -1, -1):
            destination = destinations[index]
            backup = backups[index]
            removal_error: BaseException | None = None
            try:
                if new_at_destination[index]:
                    _remove_promoted_file(destination)
                    new_at_destination[index] = False
            except BaseException as exc:
                if os.path.lexists(destination):
                    removal_error = exc
                else:
                    new_at_destination[index] = False

            restore_intended = backup_present[index] and os.path.lexists(backup)
            try:
                if restore_intended:
                    os.replace(backup, destination)
                    backup_present[index] = False
                    new_at_destination[index] = False
                    removal_error = None
            except BaseException as exc:
                restore_completed = _move_completed(
                    backup,
                    destination,
                    intended=restore_intended,
                )
                if restore_completed:
                    backup_present[index] = False
                    new_at_destination[index] = False
                    removal_error = None
                else:
                    rollback_errors.append(exc)
            if removal_error is not None:
                rollback_errors.append(removal_error)

        for index, destination in enumerate(destinations):
            if had_previous[index] and not os.path.lexists(destination):
                rollback_errors.append(
                    RuntimeError(f"previous output was not restored: {destination.name}")
                )
            if not had_previous[index] and os.path.lexists(destination):
                new_at_destination[index] = True

        if rollback_errors:
            cleanup_errors: list[BaseException] = []
            for index, destination in enumerate(destinations):
                if not new_at_destination[index]:
                    continue
                try:
                    _remove_promoted_file(destination)
                    new_at_destination[index] = False
                except BaseException as exc:
                    cleanup_errors.append(exc)
            details = "; ".join(_error_description(error) for error in rollback_errors)
            message = (
                f"data-request write failed: {_error_description(primary_error)}; "
                f"rollback failed: {details}; "
                f"recovery files preserved at {stage_root}"
            )
            if cleanup_errors:
                cleanup_details = "; ".join(_error_description(error) for error in cleanup_errors)
                message += f"; final-output cleanup failed: {cleanup_details}"
            raise RuntimeError(message) from rollback_errors[0]

        shutil.rmtree(stage_root)
        raise
    else:
        shutil.rmtree(stage_root)

    return markdown_path, json_path
