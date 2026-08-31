from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import HttpUrl

from . import __version__
from .adapters import (
    AdapterContext,
    GeoOptimizerAdapter,
    NativeCrawlerAdapter,
    PublicResearchAdapter,
    PublicResearchItem,
    StructuredDataAdapter,
)
from .analyzers import (
    analyze_pages,
    build_entity_consistency,
    select_canonical_entity,
    validate_findings_rules,
)
from .config import AuditConfig
from .crawler import Resolver
from .knowledge import KnowledgeRegistry, load_registry
from .models import AuditRun, DataState, Finding, ScoreResult, Site, SitemapState
from .prompts import PROMPT_PACK_VERSION, generate_prompt_pack
from .renderer import build_renderer_metadata, render_client_report
from .report_models import ReportLocale
from .reports import (
    RewriteProvider,
    apply_anti_slop,
    build_client_report_data,
    build_protected_claims_manifest,
    build_report_draft,
)
from .scoring import (
    ReadinessCheck,
    build_rule_backed_check,
    calculate_readiness,
    observed_ai_visibility,
    validate_score_explainability,
)

_FINAL_REPORT_ARTIFACTS = ("client-report-data.json", "client-report.pdf")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _clear_final_report_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in _FINAL_REPORT_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _rule_check(
    *,
    name: str,
    registry: KnowledgeRegistry,
    rule_id: str,
    as_of: datetime,
    state: DataState,
    score: float | None,
    confidence: float,
    explanation_finding_ids: list[str] | None = None,
    unavailable_reason: str | None = None,
) -> ReadinessCheck:
    return build_rule_backed_check(
        name=name,
        registry=registry,
        rule_id=rule_id,
        as_of=as_of.date(),
        state=state,
        score=score,
        confidence=confidence,
        explanation_finding_ids=explanation_finding_ids,
        unavailable_reason=unavailable_reason,
    )


def _finding_ids(
    findings: list[Finding],
    *,
    rule_id: str | None = None,
    categories: set[str] | None = None,
) -> list[str]:
    return sorted(
        finding.finding_id
        for finding in findings
        if (rule_id is None or finding.rule_id == rule_id)
        and (categories is None or finding.category in categories)
    )


def _robots_policies(run: AuditRun) -> dict[str, object] | None:
    robots = next((item for item in run.evidence if item.source_type == "robots_txt"), None)
    if robots is None or not isinstance(robots.observed_value, dict):
        return None
    policies = robots.observed_value.get("policies")
    if not isinstance(policies, dict):
        return {}
    return policies


def _robots_access_score(
    run: AuditRun, registry: KnowledgeRegistry
) -> tuple[DataState, float | None, str | None]:
    policies = _robots_policies(run)
    if policies is None:
        return DataState.UNAVAILABLE, None, "robots.txt policy evidence was not available."
    if not policies:
        return DataState.UNKNOWN, None, "robots.txt policies could not be normalized."
    allowed: list[bool] = []
    for product in registry.crawler_products:
        if product.purpose not in {"generic", "traditional_search"}:
            continue
        policy = policies.get(product.token)
        if isinstance(policy, dict) and isinstance(policy.get("allowed"), bool):
            allowed.append(policy["allowed"])
    if not allowed:
        return DataState.UNKNOWN, None, "Search crawler access could not be determined."
    return DataState.AVAILABLE, sum(allowed) / len(allowed) * 100, None


def _ai_search_access_checks(
    run: AuditRun, registry: KnowledgeRegistry, as_of: datetime
) -> list[ReadinessCheck]:
    policies = _robots_policies(run)
    checks: list[ReadinessCheck] = []
    for product in registry.crawler_products:
        if product.purpose != "search_citation" or not product.robots_txt_respected:
            continue
        policy = policies.get(product.token) if policies else None
        if policies is None:
            state, score = DataState.UNAVAILABLE, None
            unavailable_reason = "robots.txt policy evidence was not available."
        elif not isinstance(policy, dict) or not isinstance(policy.get("allowed"), bool):
            state, score = DataState.UNKNOWN, None
            unavailable_reason = f"{product.token} access could not be determined."
        else:
            state = DataState.AVAILABLE
            score = 100 if policy["allowed"] else 0
            unavailable_reason = None
        checks.append(
            _rule_check(
                name=f"{product.token} search/citation access",
                registry=registry,
                rule_id=product.source_rule,
                as_of=as_of,
                state=state,
                score=score,
                confidence=0.9,
                explanation_finding_ids=(
                    _finding_ids(
                        run.findings,
                        rule_id=product.source_rule,
                        categories={"ai_search_access"},
                    )
                    if score is not None and score < 100
                    else []
                ),
                unavailable_reason=unavailable_reason,
            )
        )
    return checks


def _default_scores(
    run: AuditRun,
    structured_state: DataState,
    research_state: DataState,
    registry: KnowledgeRegistry,
    as_of: datetime,
) -> list[ScoreResult]:
    assessable_pages = [page for page in run.pages if 200 <= page.status_code < 300]
    page_count = len(assessable_pages)
    page_check_state = DataState.AVAILABLE if page_count else DataState.UNAVAILABLE
    page_check_reason = None if page_count else "No 2xx HTML pages were available to assess."
    indexable = sum(1 for page in assessable_pages if page.indexable)
    canonical = sum(1 for page in assessable_pages if page.canonical)
    robots_state, robots_score, robots_reason = _robots_access_score(run, registry)
    noindex_findings = _finding_ids(run.findings, rule_id="google-noindex-001")
    canonical_findings = _finding_ids(run.findings, rule_id="google-canonical-001")
    robots_findings = _finding_ids(
        run.findings,
        rule_id="robots-rfc9309-001",
        categories={"search_access"},
    )
    sitemap_findings = _finding_ids(run.findings, rule_id="sitemap-protocol-001")
    if run.sitemap_state is SitemapState.AVAILABLE:
        sitemap_score_state, sitemap_score, sitemap_reason = DataState.AVAILABLE, 100.0, None
    elif run.sitemap_state in {
        SitemapState.CONFIRMED_ABSENT,
        SitemapState.NOT_DISCOVERED,
    }:
        sitemap_score_state, sitemap_score, sitemap_reason = DataState.AVAILABLE, 0.0, None
    elif run.sitemap_state is SitemapState.FETCH_FAILED:
        sitemap_score_state, sitemap_score, sitemap_reason = (
            DataState.FAILED,
            None,
            "A known sitemap could not be fetched or parsed.",
        )
    else:
        sitemap_score_state, sitemap_score, sitemap_reason = (
            DataState.UNAVAILABLE,
            None,
            "Sitemap discovery evidence was insufficient.",
        )
    technical = calculate_readiness(
        "Technical Search Readiness",
        [
            _rule_check(
                name="public crawl and indexability",
                registry=registry,
                rule_id="google-noindex-001",
                as_of=as_of,
                state=page_check_state,
                score=(indexable / page_count * 100) if page_count else None,
                confidence=0.9,
                explanation_finding_ids=(noindex_findings if indexable < page_count else []),
                unavailable_reason=page_check_reason,
            ),
            _rule_check(
                name="canonical annotations",
                registry=registry,
                rule_id="google-canonical-001",
                as_of=as_of,
                state=page_check_state,
                score=(canonical / page_count * 100) if page_count else None,
                confidence=0.9,
                explanation_finding_ids=(canonical_findings if canonical < page_count else []),
                unavailable_reason=page_check_reason,
            ),
            _rule_check(
                name="robots access for search crawlers",
                registry=registry,
                rule_id="robots-rfc9309-001",
                as_of=as_of,
                state=robots_state,
                score=robots_score,
                confidence=0.9,
                explanation_finding_ids=(
                    robots_findings if robots_score is not None and robots_score < 100 else []
                ),
                unavailable_reason=robots_reason,
            ),
            _rule_check(
                name="sitemap discovery",
                registry=registry,
                rule_id="sitemap-protocol-001",
                as_of=as_of,
                state=sitemap_score_state,
                score=sitemap_score,
                confidence=0.9,
                explanation_finding_ids=(
                    sitemap_findings if sitemap_score is not None and sitemap_score < 100 else []
                ),
                unavailable_reason=sitemap_reason,
            ),
            *_ai_search_access_checks(run, registry, as_of),
        ],
    )
    structured_ready_pages = sum(
        1 for page in assessable_pages if page.json_ld and not page.json_ld_errors
    )
    structured_findings = _finding_ids(run.findings, rule_id="schema-org-jsonld-001")
    structured_score = round(structured_ready_pages / page_count * 100, 2) if page_count else 0.0
    research_available = bool(run.external_mentions)
    research_reason = (
        None
        if research_available
        else "Independent public research inputs were not supplied for this audit."
    )
    entity = calculate_readiness(
        "Entity & Machine Understanding",
        [
            _rule_check(
                name="structured data observed",
                registry=registry,
                rule_id="schema-org-jsonld-001",
                as_of=as_of,
                state=structured_state if page_count else DataState.UNAVAILABLE,
                score=structured_score if page_count else None,
                confidence=0.85,
                explanation_finding_ids=(structured_findings if structured_score < 100 else []),
                unavailable_reason=page_check_reason,
            ),
            _rule_check(
                name="independent public entity evidence",
                registry=registry,
                rule_id="entity-consistency-001",
                as_of=as_of,
                state=DataState.AVAILABLE if research_available else research_state,
                score=100 if research_available else None,
                confidence=0.75,
                unavailable_reason=research_reason,
            ),
        ],
    )
    authority = calculate_readiness(
        "Authority & Trust",
        [
            _rule_check(
                name="independent public sources",
                registry=registry,
                rule_id="entity-consistency-001",
                as_of=as_of,
                state=DataState.AVAILABLE if research_available else research_state,
                score=100 if research_available else None,
                confidence=0.7,
                unavailable_reason=research_reason,
            ),
            ReadinessCheck(
                name="backlink provider",
                state=DataState.UNAVAILABLE,
                weight=2,
                confidence=0,
                unavailable_reason="Backlink provider access was not supplied.",
            ),
        ],
    )
    titled = sum(1 for page in assessable_pages if page.title)
    described = sum(1 for page in assessable_pages if page.meta_description)
    headed = sum(1 for page in assessable_pages if page.h1 and page.h2)
    substantive = sum(1 for page in assessable_pages if len(page.content_text) >= 100)
    content_findings = _finding_ids(run.findings, rule_id="content-citability-001")
    content = calculate_readiness(
        "Content Citability",
        [
            _rule_check(
                name=name,
                registry=registry,
                rule_id="content-citability-001",
                as_of=as_of,
                state=page_check_state,
                score=(count / page_count * 100) if page_count else None,
                confidence=0.8,
                explanation_finding_ids=(content_findings if count < page_count else []),
                unavailable_reason=page_check_reason,
            )
            for name, count in (
                ("descriptive page titles", titled),
                ("page summaries", described),
                ("heading structure", headed),
                ("substantive public text", substantive),
            )
        ],
    )
    measurement = calculate_readiness(
        "Measurement Maturity",
        [
            ReadinessCheck(
                name="Google Search Console",
                state=DataState.UNAVAILABLE,
                unavailable_reason="Google Search Console access was not supplied.",
            ),
            ReadinessCheck(
                name="Bing Webmaster Tools",
                state=DataState.UNAVAILABLE,
                unavailable_reason="Bing Webmaster Tools access was not supplied.",
            ),
            ReadinessCheck(
                name="analytics",
                state=DataState.UNAVAILABLE,
                unavailable_reason="Analytics access was not supplied.",
            ),
            ReadinessCheck(
                name="server logs",
                state=DataState.UNAVAILABLE,
                unavailable_reason="Server log access was not supplied.",
            ),
        ],
    )
    return [
        technical,
        entity,
        authority,
        content,
        measurement,
        observed_ai_visibility(run.ai_observations),
    ]


def run_public_audit(
    domain: str,
    *,
    output_dir: Path | str = Path("audit-output"),
    max_pages: int = 50,
    crawler_transport: httpx.BaseTransport | None = None,
    crawler_resolver: Resolver | None = None,
    research_items: list[PublicResearchItem] | None = None,
    rewrite_provider: RewriteProvider | None = None,
    geo_optimizer_command: str | None = None,
    report_locale: ReportLocale = "en",
    now: datetime | None = None,
) -> AuditRun:
    output_dir = Path(output_dir)
    _clear_final_report_artifacts(output_dir)
    config = AuditConfig(
        domain=domain,
        max_pages=max_pages,
        output_dir=output_dir,
        geo_optimizer_command=geo_optimizer_command,
    )
    registry_root = Path(__file__).resolve().parents[2] / "knowledge"
    registry = load_registry(registry_root)
    timestamp = now or datetime.now(UTC)
    parsed_domain = httpx.URL(config.domain).host
    site = Site(
        domain=parsed_domain,
        base_url=HttpUrl(config.domain),
        brand=parsed_domain.split(".")[0].title(),
    )
    context = AdapterContext(site=site, config=config)

    crawl = NativeCrawlerAdapter(
        registry.crawler_products,
        transport=crawler_transport,
        resolver=crawler_resolver,
    ).collect(context)
    if not crawl.pages:
        raise RuntimeError("public audit collected no usable HTML pages")
    languages = sorted({page.language for page in crawl.pages if page.language})
    site.languages = languages
    canonical_entity = select_canonical_entity(site, crawl.pages)
    site.brand = canonical_entity.brand
    context.pages = crawl.pages
    structured = StructuredDataAdapter().collect(context)
    research = PublicResearchAdapter().collect(context, items=research_items)
    geo = GeoOptimizerAdapter(geo_optimizer_command).collect(context)
    evidence = [*crawl.evidence, *structured.evidence, *research.evidence, *geo.evidence]
    sitemap_state = crawl.sitemap_state or SitemapState.UNAVAILABLE
    findings = analyze_pages(
        crawl.pages,
        evidence,
        registry,
        as_of=timestamp.date(),
        sitemap_state=sitemap_state,
    )
    entity, entity_matrix, entity_findings = build_entity_consistency(
        site,
        research.external_mentions,
        registry,
        as_of=timestamp.date(),
        entity_type=canonical_entity.type,
        canonical_entity=canonical_entity,
        canonical_evidence=structured.evidence,
    )
    findings.extend(entity_findings)
    validate_findings_rules(findings, registry, as_of=timestamp.date())
    content_themes = []
    for page in crawl.pages:
        content_themes.extend([*page.h1, *page.h2])
    prompts = generate_prompt_pack(
        site,
        entity_type=entity.type,
        category=entity.type,
        location=entity.location,
        content_themes=content_themes[:8],
    )
    audit_id = (
        "audit-"
        + hashlib.sha256(f"{parsed_domain}:{timestamp.isoformat()}".encode()).hexdigest()[:16]
    )
    run = AuditRun(
        audit_id=audit_id,
        site=site,
        audit_engine_version=__version__,
        ruleset_version=registry.version,
        ruleset_verified_date=registry.verified_date,
        timestamp=timestamp,
        adapter_versions={
            item.adapter_id: item.adapter_version for item in (crawl, structured, research, geo)
        },
        adapter_states={item.adapter_id: item.state for item in (crawl, structured, research, geo)},
        sitemap_state=sitemap_state,
        pages=crawl.pages,
        entity=entity,
        evidence=evidence,
        findings=findings,
        ai_prompts=prompts,
        external_mentions=research.external_mentions,
        entity_consistency_matrix=entity_matrix,
        scores=[],
        warnings=[*crawl.warnings, *structured.warnings, *research.warnings, *geo.warnings],
        configuration={"max_pages": max_pages, "target": config.domain},
    )
    run.scores = _default_scores(run, structured.state, research.state, registry, timestamp)
    validate_score_explainability(run.scores, run.findings, run.evidence, run.ai_observations)
    paths = {
        name: str(output_dir / name)
        for name in (
            "audit.json",
            "evidence.jsonl",
            "implementation-backlog.csv",
            "report-draft.json",
            "client-report-data.json",
            "client-report.pdf",
            "ai-prompts.json",
        )
    }
    run.output_paths = paths
    draft = build_report_draft(run, report_locale=report_locale)
    _atomic_text(output_dir / "report-draft.json", _json(draft.model_dump(mode="json")))
    manifest = build_protected_claims_manifest(draft)
    rewritten = apply_anti_slop(draft, manifest, provider=rewrite_provider)
    client_report = build_client_report_data(
        run,
        rewritten,
        manifest,
        renderer_metadata=build_renderer_metadata(),
    )

    _atomic_text(output_dir / "audit.json", _json(run.model_dump(mode="json")))
    evidence_lines = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n" for item in run.evidence
    )
    _atomic_text(output_dir / "evidence.jsonl", evidence_lines)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "priority",
            "finding_id",
            "client_title",
            "implementation",
            "affected_urls",
            "evidence_ids",
        ]
    )
    for finding in sorted(rewritten.draft.findings, key=lambda item: item.priority.value):
        writer.writerow(
            [
                finding.priority.value,
                finding.finding_id,
                finding.client_title,
                finding.implementation,
                " | ".join(finding.affected_urls),
                " | ".join(finding.evidence_ids),
            ]
        )
    _atomic_text(output_dir / "implementation-backlog.csv", buffer.getvalue())
    _atomic_text(
        output_dir / "ai-prompts.json",
        _json(
            {
                "version": PROMPT_PACK_VERSION,
                "observed_ai_visibility_state": observed_ai_visibility(
                    run.ai_observations
                ).state.value,
                "prompts": [prompt.model_dump(mode="json") for prompt in run.ai_prompts],
            }
        ),
    )
    with tempfile.TemporaryDirectory(prefix=".report-run.", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        staged_data = staging / "client-report-data.json"
        staged_pdf = staging / "client-report.pdf"
        _atomic_text(staged_data, _json(client_report.model_dump(mode="json")))
        render_client_report(client_report, staged_pdf)
        try:
            os.replace(staged_pdf, output_dir / "client-report.pdf")
            os.replace(staged_data, output_dir / "client-report-data.json")
        except BaseException:
            _clear_final_report_artifacts(output_dir)
            raise
    return run
