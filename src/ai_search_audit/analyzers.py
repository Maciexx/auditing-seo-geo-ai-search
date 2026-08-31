from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from .knowledge import KnowledgeRegistry, ResolvedKnowledgeRule
from .models import (
    ClaimModality,
    Entity,
    Evidence,
    ExternalMention,
    FactualClaim,
    Finding,
    FindingStatus,
    Page,
    Priority,
    RuleState,
    Severity,
    Site,
    SitemapState,
)


def _finding_id(rule_id: str, value: str) -> str:
    return f"finding-{hashlib.sha256(f'{rule_id}:{value}'.encode()).hexdigest()[:12]}"


class _RuleFindingFields(TypedDict):
    rule_evidence_level: Literal["A", "B", "C", "D", "E"]
    rule_confidence: float
    rule_scoring_weight: float
    rule_state: RuleState
    rule_expires_at: date


def _rule_fields(rule: ResolvedKnowledgeRule) -> _RuleFindingFields:
    return {
        "rule_evidence_level": rule.evidence_level,
        "rule_confidence": rule.confidence,
        "rule_scoring_weight": rule.scoring_weight,
        "rule_state": rule.state,
        "rule_expires_at": rule.expires_at,
    }


def _status_for_rule(status: FindingStatus, rule: ResolvedKnowledgeRule) -> FindingStatus:
    if rule.state is RuleState.REQUIRES_VERIFICATION:
        return FindingStatus.REQUIRES_VERIFICATION
    return status


def analyze_pages(
    pages: list[Page],
    evidence: list[Evidence],
    registry: KnowledgeRegistry,
    *,
    as_of: date,
    sitemap_state: SitemapState = SitemapState.UNAVAILABLE,
) -> list[Finding]:
    evidence_by_url: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        if item.source_url:
            evidence_by_url[str(item.source_url)].append(item.evidence_id)
    findings: list[Finding] = []
    for page in pages:
        url = str(page.final_url)
        evidence_ids = evidence_by_url.get(url, [])
        if "noindex" in page.robots_directives:
            rule = registry.resolve("google-noindex-001", as_of=as_of)
            findings.append(
                Finding(
                    finding_id=_finding_id("google-noindex-001", url),
                    category="indexability",
                    severity=Severity.HIGH,
                    status=_status_for_rule(FindingStatus.CONFIRMED, rule),
                    rule_id="google-noindex-001",
                    technical_title="Indexable-content candidate contains noindex",
                    technical_description=f"{url} contains a noindex robots directive.",
                    client_title="A page is hidden from search",
                    client_explanation=(
                        "The page tells search systems not to include it in results."
                    ),
                    business_impact=(
                        "The page cannot be discovered through search while the directive remains."
                    ),
                    implementation=(
                        "Confirm intent, then remove noindex if this page should be discoverable."
                    ),
                    priority=Priority.P1,
                    affected_urls=[url],
                    evidence_ids=evidence_ids,
                    confidence=min(0.99, rule.confidence),
                    factual_claims=[
                        FactualClaim(
                            claim_id=f"{_finding_id('google-noindex-001', url)}-directive",
                            predicate="page.robots_directive",
                            value="noindex",
                            evidence_ids=evidence_ids,
                            modality=ClaimModality.OBSERVED,
                            meaning=f"{url} declares the noindex robots directive.",
                        )
                    ],
                    **_rule_fields(rule),
                )
            )
        if page.status_code == 200 and page.canonical is None:
            rule = registry.resolve("google-canonical-001", as_of=as_of)
            findings.append(
                Finding(
                    finding_id=_finding_id("google-canonical-001", url),
                    category="canonicalization",
                    severity=Severity.MEDIUM,
                    status=_status_for_rule(FindingStatus.INFERRED, rule),
                    rule_id="google-canonical-001",
                    technical_title="No HTML canonical annotation observed",
                    technical_description=f"No rel=canonical annotation was found on {url}.",
                    client_title="The preferred page version is not stated",
                    client_explanation=(
                        "Search systems can still select a preferred URL, "
                        "but the page does not state one."
                    ),
                    business_impact="This may make duplicate-page consolidation less predictable.",
                    implementation=(
                        "Add a consistent self-referencing canonical when it matches "
                        "the publishing strategy."
                    ),
                    priority=Priority.P2,
                    affected_urls=[url],
                    evidence_ids=evidence_ids,
                    confidence=min(0.8, rule.confidence),
                    factual_claims=[
                        FactualClaim(
                            claim_id=f"{_finding_id('google-canonical-001', url)}-canonical",
                            predicate="page.html_canonical",
                            value=None,
                            evidence_ids=evidence_ids,
                            modality=ClaimModality.OBSERVED,
                            negated=True,
                            meaning=f"No HTML canonical annotation was observed on {url}.",
                        ),
                        FactualClaim(
                            claim_id=f"{_finding_id('google-canonical-001', url)}-impact",
                            predicate="duplicate_consolidation.predictability",
                            value="may decrease",
                            evidence_ids=evidence_ids,
                            modality=ClaimModality.POSSIBLE,
                            meaning=(
                                "Missing canonical guidance may reduce consolidation "
                                "predictability."
                            ),
                        ),
                    ],
                    **_rule_fields(rule),
                )
            )
    findings.extend(_analyze_json_ld(pages, evidence, registry, as_of=as_of))
    findings.extend(_analyze_content_citability(pages, evidence, registry, as_of=as_of))
    findings.extend(_analyze_sitemap_state(sitemap_state, evidence, registry, as_of=as_of))
    findings.extend(analyze_search_access(evidence, registry, as_of=as_of))
    findings.extend(analyze_ai_search_access(evidence, registry, as_of=as_of))
    return findings


def _page_evidence_ids(page: Page, evidence: list[Evidence]) -> list[str]:
    url = str(page.final_url)
    return [
        item.evidence_id
        for item in evidence
        if item.source_type == "web_page" and item.source_url and str(item.source_url) == url
    ]


def _analyze_json_ld(
    pages: list[Page],
    evidence: list[Evidence],
    registry: KnowledgeRegistry,
    *,
    as_of: date,
) -> list[Finding]:
    invalid_pages = [page for page in pages if page.json_ld_errors]
    rule = registry.resolve("schema-org-jsonld-001", as_of=as_of)
    findings: list[Finding] = []
    if invalid_pages:
        affected_urls = [str(page.final_url) for page in invalid_pages]
        evidence_ids = sorted(
            {
                evidence_id
                for page in invalid_pages
                for evidence_id in _page_evidence_ids(page, evidence)
            }
        )
        messages = sorted(
            {error.message for page in invalid_pages for error in page.json_ld_errors}
        )
        count = len(invalid_pages)
        total = len(pages)
        finding_id = _finding_id("schema-org-jsonld-001", "invalid-json-ld")
        findings.append(
            Finding(
                finding_id=finding_id,
                category="structured_data_validity",
                severity=Severity.MEDIUM,
                status=_status_for_rule(FindingStatus.CONFIRMED, rule),
                rule_id="schema-org-jsonld-001",
                technical_title="Invalid JSON-LD observed across audited pages",
                technical_description=(
                    f"Invalid JSON-LD was observed on {count}/{total} audited pages. "
                    f"Parser errors: {', '.join(messages)}."
                ),
                client_title="Structured data cannot be parsed on audited pages",
                client_explanation=(
                    f"The audit found JSON-LD parser errors on {count} of {total} pages."
                ),
                business_impact=(
                    "Invalid markup may prevent search and AI systems from using the affected "
                    "structured facts. This audit does not measure downstream use."
                ),
                implementation=(
                    "Validate each affected JSON-LD block, remove trailing or duplicate JSON "
                    "objects, and retest the corrected markup."
                ),
                priority=Priority.P1,
                affected_urls=affected_urls,
                evidence_ids=evidence_ids,
                confidence=min(0.98, rule.confidence),
                factual_claims=[
                    FactualClaim(
                        claim_id=f"{finding_id}-parse-errors",
                        predicate="structured_data.json_ld.parse_errors",
                        value=f"{count}/{total} pages: {', '.join(messages)}",
                        evidence_ids=evidence_ids,
                        numbers=sorted({str(count), str(total)}),
                        modality=ClaimModality.OBSERVED,
                        meaning=(
                            f"JSON-LD parser errors were observed on {count} of {total} "
                            f"audited pages: {', '.join(messages)}."
                        ),
                    ),
                    FactualClaim(
                        claim_id=f"{finding_id}-impact",
                        predicate="machine_readable_facts.usability",
                        value="may decrease",
                        evidence_ids=evidence_ids,
                        modality=ClaimModality.POSSIBLE,
                        meaning=(
                            "Invalid JSON-LD may prevent systems from using the affected "
                            "structured facts."
                        ),
                    ),
                ],
                **_rule_fields(rule),
            )
        )
    missing_pages = [page for page in pages if not page.json_ld and not page.json_ld_errors]
    if not missing_pages:
        return findings
    affected_urls = [str(page.final_url) for page in missing_pages]
    evidence_ids = sorted(
        {
            evidence_id
            for page in missing_pages
            for evidence_id in _page_evidence_ids(page, evidence)
        }
    )
    count = len(missing_pages)
    total = len(pages)
    finding_id = _finding_id("schema-org-jsonld-001", "not-observed")
    findings.append(
        Finding(
            finding_id=finding_id,
            category="structured_data_presence",
            severity=Severity.LOW,
            status=_status_for_rule(FindingStatus.INFERRED, rule),
            rule_id="schema-org-jsonld-001",
            technical_title="Parseable JSON-LD not observed on audited pages",
            technical_description=(
                f"No parseable JSON-LD was observed on {count}/{total} audited pages."
            ),
            client_title="Some audited pages lack machine-readable entity markup",
            client_explanation=(
                f"The crawl did not observe parseable JSON-LD on {count} of {total} audited pages."
            ),
            business_impact=(
                "Adding accurate markup may make selected public facts easier to interpret, "
                "but it does not guarantee visibility or citations."
            ),
            implementation=(
                "Add valid Schema.org JSON-LD only for facts already visible and supportable "
                "on the page."
            ),
            priority=Priority.P2,
            affected_urls=affected_urls,
            evidence_ids=evidence_ids,
            confidence=min(0.8, rule.confidence),
            factual_claims=[
                FactualClaim(
                    claim_id=f"{finding_id}-presence",
                    predicate="structured_data.json_ld.observed",
                    value=False,
                    evidence_ids=evidence_ids,
                    numbers=sorted({str(count), str(total)}),
                    modality=ClaimModality.OBSERVED,
                    negated=True,
                    meaning=(
                        f"No parseable JSON-LD was observed on {count} of {total} audited pages."
                    ),
                )
            ],
            **_rule_fields(rule),
        )
    )
    return findings


def _analyze_content_citability(
    pages: list[Page],
    evidence: list[Evidence],
    registry: KnowledgeRegistry,
    *,
    as_of: date,
) -> list[Finding]:
    failures: dict[str, list[str]] = {}
    for page in pages:
        missing: list[str] = []
        if not page.title:
            missing.append("descriptive title")
        if not page.meta_description:
            missing.append("page summary")
        if not page.h1 or not page.h2:
            missing.append("heading structure")
        if len(page.content_text) < 100:
            missing.append("substantive public text")
        if missing:
            failures[str(page.final_url)] = missing
    if not failures:
        return []
    rule = registry.resolve("content-citability-001", as_of=as_of)
    evidence_ids = sorted(
        {
            evidence_id
            for page in pages
            if str(page.final_url) in failures
            for evidence_id in _page_evidence_ids(page, evidence)
        }
    )
    missing_signals = sorted({signal for values in failures.values() for signal in values})
    finding_id = _finding_id("content-citability-001", "assessed-content-gaps")
    return [
        Finding(
            finding_id=finding_id,
            category="content_citability",
            severity=Severity.LOW,
            status=_status_for_rule(
                FindingStatus.CONFIRMED if evidence_ids else FindingStatus.INFERRED,
                rule,
            ),
            rule_id="content-citability-001",
            technical_title="Observable content-citability checks are incomplete",
            technical_description=(
                f"{len(failures)} audited pages miss one or more assessed signals: "
                f"{', '.join(missing_signals)}."
            ),
            client_title="Some audited pages lack clear citation-ready context",
            client_explanation=(
                "The assessed pages do not consistently provide titles, summaries, headings, "
                "and substantive public text."
            ),
            business_impact=(
                "Clearer page context may make public facts easier to interpret and quote; "
                "it does not predict citation."
            ),
            implementation=(
                "Add accurate titles, concise summaries, descriptive headings, and concrete "
                "supporting facts to the affected pages."
            ),
            priority=Priority.P2,
            affected_urls=sorted(failures),
            evidence_ids=evidence_ids,
            confidence=min(0.85, rule.confidence),
            factual_claims=[
                FactualClaim(
                    claim_id=f"{finding_id}-checks",
                    predicate="content_citability.assessed_signals",
                    value=", ".join(missing_signals),
                    evidence_ids=evidence_ids,
                    numbers=[str(len(failures))],
                    modality=ClaimModality.OBSERVED,
                    meaning=(
                        f"{len(failures)} pages miss assessed content signals: "
                        f"{', '.join(missing_signals)}."
                    ),
                )
            ],
            **_rule_fields(rule),
        )
    ]


def _analyze_sitemap_state(
    state: SitemapState,
    evidence: list[Evidence],
    registry: KnowledgeRegistry,
    *,
    as_of: date,
) -> list[Finding]:
    if state not in {SitemapState.CONFIRMED_ABSENT, SitemapState.NOT_DISCOVERED}:
        return []
    rule = registry.resolve("sitemap-protocol-001", as_of=as_of)
    relevant = [item for item in evidence if item.source_type in {"robots_txt", "sitemap"}]
    evidence_ids = [item.evidence_id for item in relevant]
    affected_urls = sorted(
        {str(item.source_url) for item in relevant if item.source_url is not None}
    )
    confirmed = state is SitemapState.CONFIRMED_ABSENT
    finding_id = _finding_id("sitemap-protocol-001", state.value)
    return [
        Finding(
            finding_id=finding_id,
            category="sitemap_discovery",
            severity=Severity.LOW,
            status=_status_for_rule(
                FindingStatus.CONFIRMED if confirmed else FindingStatus.INFERRED,
                rule,
            ),
            rule_id="sitemap-protocol-001",
            technical_title=(
                "Conventional sitemap endpoint was confirmed absent"
                if confirmed
                else "Sitemap was not discovered in the assessed methods"
            ),
            technical_description=(
                "The assessed conventional sitemap endpoint returned absence evidence."
                if confirmed
                else "A sitemap was not discovered through the assessed public methods."
            ),
            client_title=(
                "The conventional sitemap endpoint is absent"
                if confirmed
                else "A sitemap was not discovered"
            ),
            client_explanation=(
                "The conventional endpoint returned an absence response."
                if confirmed
                else "The checked public discovery methods did not reveal a sitemap."
            ),
            business_impact=(
                "A discoverable sitemap may help search crawlers find canonical URLs, but its "
                "presence does not guarantee indexing."
            ),
            implementation=(
                "Publish a valid XML sitemap for canonical URLs and declare it in robots.txt."
            ),
            priority=Priority.P2,
            affected_urls=affected_urls,
            evidence_ids=evidence_ids,
            confidence=min(0.9 if confirmed else 0.75, rule.confidence),
            factual_claims=[
                FactualClaim(
                    claim_id=f"{finding_id}-state",
                    predicate="sitemap.discovery_state",
                    value=state.value,
                    evidence_ids=evidence_ids,
                    modality=(ClaimModality.OBSERVED if confirmed else ClaimModality.INFERRED),
                    meaning=(
                        "The conventional sitemap endpoint was confirmed absent."
                        if confirmed
                        else "A sitemap was not discovered through the assessed methods."
                    ),
                )
            ],
            **_rule_fields(rule),
        )
    ]


def analyze_search_access(
    evidence: list[Evidence], registry: KnowledgeRegistry, *, as_of: date
) -> list[Finding]:
    robots_evidence = next((item for item in evidence if item.source_type == "robots_txt"), None)
    if robots_evidence is None or not isinstance(robots_evidence.observed_value, dict):
        return []
    policies = robots_evidence.observed_value.get("policies")
    if not isinstance(policies, dict):
        return []
    findings: list[Finding] = []
    for product in registry.crawler_products:
        if product.purpose not in {"generic", "traditional_search"}:
            continue
        policy = policies.get(product.token)
        if not isinstance(policy, dict) or policy.get("allowed") is not False:
            continue
        rule = registry.resolve(product.source_rule, as_of=as_of)
        finding_id = _finding_id(product.source_rule, product.identifier)
        findings.append(
            Finding(
                finding_id=finding_id,
                category="search_access",
                severity=Severity.HIGH,
                status=_status_for_rule(FindingStatus.CONFIRMED, rule),
                rule_id=product.source_rule,
                technical_title=f"{product.token} is blocked at the site root",
                technical_description=f"robots.txt blocks {product.token} from '/'.",
                client_title=f"{product.vendor} search crawling is blocked",
                client_explanation=f"The public robots policy blocks {product.token}.",
                business_impact=(
                    "The block may prevent the affected search crawler from requesting public "
                    "pages."
                ),
                implementation=(
                    f"Confirm intent and update the robots.txt group for {product.token} if "
                    "search crawling is wanted."
                ),
                priority=Priority.P0,
                affected_urls=(
                    [str(robots_evidence.source_url)] if robots_evidence.source_url else []
                ),
                evidence_ids=[robots_evidence.evidence_id],
                confidence=min(0.98, rule.confidence),
                factual_claims=[
                    FactualClaim(
                        claim_id=f"{finding_id}-access",
                        predicate=f"crawler.{product.identifier}.robots_access",
                        value="blocked",
                        evidence_ids=[robots_evidence.evidence_id],
                        modality=ClaimModality.OBSERVED,
                        negated=True,
                        meaning=f"robots.txt blocks {product.token} from the site root.",
                    )
                ],
                **_rule_fields(rule),
            )
        )
    return findings


def analyze_ai_search_access(
    evidence: list[Evidence], registry: KnowledgeRegistry, *, as_of: date
) -> list[Finding]:
    robots_evidence = next((item for item in evidence if item.source_type == "robots_txt"), None)
    if robots_evidence is None or not isinstance(robots_evidence.observed_value, dict):
        return []
    policies = robots_evidence.observed_value.get("policies")
    if not isinstance(policies, dict):
        return []
    findings: list[Finding] = []
    for product in registry.crawler_products:
        if product.purpose != "search_citation" or not product.robots_txt_respected:
            continue
        policy = policies.get(product.token)
        if not isinstance(policy, dict) or policy.get("allowed") is not False:
            continue
        rule = registry.resolve(product.source_rule, as_of=as_of)
        source_url = str(robots_evidence.source_url) if robots_evidence.source_url else None
        finding_id = _finding_id(product.source_rule, product.identifier)
        findings.append(
            Finding(
                finding_id=finding_id,
                category="ai_search_access",
                severity=Severity.MEDIUM,
                status=_status_for_rule(FindingStatus.CONFIRMED, rule),
                rule_id=product.source_rule,
                technical_title=f"{product.token} is blocked at the site root",
                technical_description=(
                    f"The observed robots.txt policy blocks {product.token} from '/'."
                ),
                client_title=f"{product.vendor} search access is blocked",
                client_explanation=(
                    f"The public robots policy prevents {product.token} from crawling the site."
                ),
                business_impact=(
                    "This may limit the vendor's ability to crawl content for search or citation; "
                    "it does not measure observed AI visibility."
                ),
                implementation=(
                    "Confirm that the block is intentional. If search access is wanted, update "
                    f"the robots.txt group for {product.token}."
                ),
                priority=Priority.P1,
                affected_urls=[source_url] if source_url else [],
                evidence_ids=[robots_evidence.evidence_id],
                confidence=min(0.95, rule.confidence),
                factual_claims=[
                    FactualClaim(
                        claim_id=f"{finding_id}-access",
                        predicate=f"crawler.{product.identifier}.robots_access",
                        value="blocked",
                        evidence_ids=[robots_evidence.evidence_id],
                        modality=ClaimModality.OBSERVED,
                        negated=True,
                        meaning=f"robots.txt blocks {product.token} from the site root.",
                    ),
                    FactualClaim(
                        claim_id=f"{finding_id}-impact",
                        predicate="ai_search_crawlability",
                        value="may decrease",
                        evidence_ids=[robots_evidence.evidence_id],
                        modality=ClaimModality.POSSIBLE,
                        meaning=(
                            "Blocking a search/citation crawler may reduce vendor crawl access."
                        ),
                    ),
                ],
                **_rule_fields(rule),
            )
        )
    return findings


def validate_findings_rules(
    findings: list[Finding], registry: KnowledgeRegistry, *, as_of: date
) -> None:
    for finding in findings:
        try:
            rule = registry.resolve(finding.rule_id, as_of=as_of)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        expected = _rule_fields(rule)
        for field, value in expected.items():
            if getattr(finding, field) != value:
                raise ValueError(
                    f"finding {finding.finding_id} has inconsistent {field} for {finding.rule_id}"
                )


_ENTITY_TYPE_PRIORITY = {"Hotel": 0, "LocalBusiness": 1, "Organization": 2, "WebSite": 3}
_FACT_KEY_ALIASES = {
    "numberofrooms": "room_count",
    "number_of_rooms": "room_count",
    "rooms": "room_count",
    "room_count": "room_count",
    "phone": "telephone",
    "telephone": "telephone",
    "address": "address",
    "location": "location",
    "url": "url",
    "website": "url",
}


def _json_ld_nodes(value: dict[str, object]) -> list[dict[str, object]]:
    nodes = [value]
    graph = value.get("@graph")
    if isinstance(graph, list):
        nodes.extend(item for item in graph if isinstance(item, dict))
    return nodes


def _supported_entity_type(value: object) -> str | None:
    candidates = value if isinstance(value, list) else [value]
    supported = [
        item for item in candidates if isinstance(item, str) and item in _ENTITY_TYPE_PRIORITY
    ]
    return min(supported, key=_ENTITY_TYPE_PRIORITY.__getitem__) if supported else None


def _entity_node_is_relevant(site: Site, node: dict[str, object]) -> bool:
    references = [node.get("url"), node.get("@id")]
    explicit_urls = [item for item in references if isinstance(item, str) and "://" in item]
    if not explicit_urls:
        return True
    return any(urlsplit(item).hostname == site.domain for item in explicit_urls)


def _normalize_fact_key(key: str) -> str:
    return _FACT_KEY_ALIASES.get(key.casefold(), key.casefold())


def _normalize_fact_value(key: str, value: object) -> str | None:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if not isinstance(value, (str, int, float)):
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    if key == "room_count":
        match = re.search(r"\d+", text)
        return str(int(match.group())) if match else text.casefold()
    if key == "telephone":
        digits = re.sub(r"\D", "", text)
        return f"+{digits}" if text.startswith("+") and digits else digits or None
    if key == "url":
        parts = urlsplit(text)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme.casefold()}://{parts.netloc.casefold()}{parts.path.rstrip('/')}"
    return text.casefold()


def _entity_facts(node: dict[str, object]) -> dict[str, list[str]]:
    facts: dict[str, list[str]] = {}
    for raw_key in ("numberOfRooms", "rooms", "room_count", "telephone", "phone", "url"):
        if raw_key not in node:
            continue
        key = _normalize_fact_key(raw_key)
        normalized = _normalize_fact_value(key, node[raw_key])
        if normalized is not None:
            facts.setdefault(key, []).append(normalized)
    address = node.get("address")
    if isinstance(address, dict):
        parts = [
            address.get(key)
            for key in ("streetAddress", "postalCode", "addressLocality", "addressCountry")
        ]
        normalized = _normalize_fact_value("address", " ".join(str(item) for item in parts if item))
        if normalized:
            facts["address"] = [normalized]
    elif address is not None:
        normalized = _normalize_fact_value("address", address)
        if normalized:
            facts["address"] = [normalized]
    return facts


def select_canonical_entity(site: Site, pages: list[Page]) -> Entity:
    candidates: list[tuple[int, int, int, dict[str, object], str]] = []
    order = 0
    for page in pages:
        for item in page.json_ld:
            for node in _json_ld_nodes(item):
                entity_type = _supported_entity_type(node.get("@type"))
                name = node.get("name")
                if entity_type is None or not isinstance(name, str) or not name.strip():
                    continue
                candidates.append(
                    (
                        0 if _entity_node_is_relevant(site, node) else 1,
                        _ENTITY_TYPE_PRIORITY[entity_type],
                        order,
                        node,
                        entity_type,
                    )
                )
                order += 1
    if not candidates:
        page_brand = next(
            (page.h1[0].strip() for page in pages if page.h1 and page.h1[0].strip()),
            None,
        )
        return Entity(
            brand=page_brand or site.brand or site.domain,
            type="Organization",
            domain=site.domain,
            languages=site.languages,
        )
    _, _, _, node, entity_type = min(candidates, key=lambda item: item[:3])
    facts = _entity_facts(node)
    location: str | None = None
    address = node.get("address")
    if isinstance(address, dict) and isinstance(address.get("addressLocality"), str):
        location = " ".join(address["addressLocality"].split())
    return Entity(
        brand=str(node["name"]).strip(),
        type=entity_type,
        domain=site.domain,
        location=location,
        languages=site.languages,
        facts=facts,
    )


def _canonical_fact_sources(
    entity: Entity, evidence: list[Evidence]
) -> dict[str, tuple[list[str], list[str]]]:
    sources: dict[str, tuple[list[str], list[str]]] = {}
    for item in evidence:
        if item.source_type != "structured_data" or not isinstance(item.observed_value, dict):
            continue
        for node in _json_ld_nodes(item.observed_value):
            name = node.get("name")
            node_type = _supported_entity_type(node.get("@type"))
            if (
                node_type != entity.type
                or not isinstance(name, str)
                or " ".join(name.split()).casefold() != entity.brand.casefold()
            ):
                continue
            for key, values in _entity_facts(node).items():
                if not set(values).intersection(entity.facts.get(key, [])):
                    continue
                evidence_ids, urls = sources.setdefault(key, ([], []))
                evidence_ids.append(item.evidence_id)
                if item.source_url is not None:
                    urls.append(str(item.source_url))
    return sources


def build_entity_consistency(
    site: Site,
    mentions: list[ExternalMention],
    registry: KnowledgeRegistry,
    *,
    as_of: date,
    entity_type: str = "Organization",
    canonical_entity: Entity | None = None,
    canonical_evidence: list[Evidence] | None = None,
) -> tuple[Entity, dict[str, dict[str, list[str]]], list[Finding]]:
    matrix: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    fact_urls: dict[str, set[str]] = defaultdict(set)
    if canonical_entity is not None:
        official_sources = _canonical_fact_sources(canonical_entity, canonical_evidence or [])
        for key, canonical_values in canonical_entity.facts.items():
            evidence_ids, urls = official_sources.get(key, ([], []))
            if not evidence_ids:
                continue
            for canonical_value in canonical_values:
                matrix[key][canonical_value].extend(evidence_ids)
            fact_urls[key].update(urls)
    for mention in mentions:
        for raw_key, raw_value in mention.claims.items():
            key = _normalize_fact_key(raw_key)
            normalized_value = _normalize_fact_value(key, raw_value)
            if normalized_value is not None:
                matrix[key][normalized_value].extend(mention.evidence_ids)
                fact_urls[key].add(str(mention.url))
    plain_matrix = {key: dict(values) for key, values in matrix.items()}
    facts = dict(canonical_entity.facts) if canonical_entity else {}
    facts.update({key: list(values) for key, values in plain_matrix.items()})
    location_values = facts.get("location", [])
    profile = Entity(
        brand=canonical_entity.brand if canonical_entity else site.brand or site.domain,
        type=canonical_entity.type if canonical_entity else entity_type,
        domain=site.domain,
        location=location_values[0] if len(location_values) == 1 else None,
        languages=canonical_entity.languages if canonical_entity else site.languages,
        facts=facts,
    )
    findings: list[Finding] = []
    for fact, value_map in plain_matrix.items():
        if len(value_map) < 2:
            continue
        rule = registry.resolve("entity-consistency-001", as_of=as_of)
        evidence_ids = sorted({item for ids in value_map.values() for item in ids})
        numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", " ".join(value_map))))
        findings.append(
            Finding(
                finding_id=_finding_id("entity-consistency-001", fact),
                category="entity_consistency",
                severity=Severity.MEDIUM,
                status=_status_for_rule(FindingStatus.CONFIRMED, rule),
                rule_id="entity-consistency-001",
                technical_title=f"Conflicting public values for {fact}",
                technical_description=(
                    f"Independent sources report {len(value_map)} values for {fact}: "
                    f"{', '.join(value_map)}."
                ),
                client_title=f"Public sources disagree about {fact}",
                client_explanation=(
                    "Independent sources present different facts about the business."
                ),
                business_impact="The conflict may make automated entity summaries less reliable.",
                implementation=(
                    "Confirm the canonical fact and update owned profiles and major "
                    "independent listings."
                ),
                priority=Priority.P1,
                affected_urls=sorted(fact_urls[fact]),
                evidence_ids=evidence_ids,
                confidence=min(0.95, rule.confidence),
                factual_claims=[
                    FactualClaim(
                        claim_id=f"{_finding_id('entity-consistency-001', fact)}-conflict",
                        predicate=f"entity.{fact}.conflict",
                        value=" | ".join(value_map),
                        evidence_ids=evidence_ids,
                        numbers=numbers,
                        modality=ClaimModality.OBSERVED,
                        meaning=(
                            f"Independent sources report conflicting {fact} values: "
                            f"{', '.join(value_map)}."
                        ),
                    ),
                    FactualClaim(
                        claim_id=f"{_finding_id('entity-consistency-001', fact)}-impact",
                        predicate="automated_entity_summary.reliability",
                        value="may decrease",
                        evidence_ids=evidence_ids,
                        modality=ClaimModality.POSSIBLE,
                        meaning=(
                            "The public fact conflict may reduce automated-summary reliability."
                        ),
                    ),
                ],
                **_rule_fields(rule),
            )
        )
    return profile, plain_matrix, findings
