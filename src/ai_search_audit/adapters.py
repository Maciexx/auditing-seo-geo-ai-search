from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, HttpUrl

from .config import AuditConfig
from .crawler import NativeCrawler, Resolver
from .knowledge import CrawlerProduct
from .models import AIObservation, DataState, Evidence, ExternalMention, Page, Site, SitemapState

FUTURE_CAPABILITIES = {
    "screaming-frog",
    "lighthouse",
    "pagespeed-insights",
    "crux",
    "gsc",
    "gsc-generative-ai-reporting",
    "bing-webmaster-tools",
    "bing-ai-performance",
    "ga4",
    "google-ads",
    "microsoft-advertising",
    "google-business-profile",
    "server-logs",
    "structured-data-validator",
    "ahrefs",
    "semrush",
    "majestic",
    "openai-grounded-search",
    "gemini-grounding",
    "perplexity",
    "claude-search",
}


class AdapterContext(BaseModel):
    site: Site
    config: AuditConfig
    pages: list[Page] = Field(default_factory=list)


class AdapterResult(BaseModel):
    adapter_id: str
    adapter_version: str
    state: DataState
    evidence: list[Evidence] = Field(default_factory=list)
    observations: list[AIObservation] = Field(default_factory=list)
    external_mentions: list[ExternalMention] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    sitemap_state: SitemapState | None = None
    warnings: list[str] = Field(default_factory=list)


class AuditAdapter(Protocol):
    adapter_id: str
    version: str

    def collect(self, context: AdapterContext) -> AdapterResult: ...


class NativeCrawlerAdapter:
    adapter_id = "native-crawler"
    version = NativeCrawler.version

    def __init__(
        self,
        products: list[CrawlerProduct],
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self.products = products
        self.transport = transport
        self.resolver = resolver

    def collect(self, context: AdapterContext) -> AdapterResult:
        crawl = NativeCrawler(
            context.config,
            self.products,
            transport=self.transport,
            resolver=self.resolver,
        ).crawl()
        state = DataState.AVAILABLE if crawl.pages else DataState.FAILED
        return AdapterResult(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            state=state,
            evidence=crawl.evidence,
            pages=crawl.pages,
            sitemap_state=crawl.sitemap_state,
            warnings=crawl.warnings,
        )


class StructuredDataAdapter:
    adapter_id = "structured-data"
    version = "0.1.0"

    def collect(self, context: AdapterContext) -> AdapterResult:
        evidence: list[Evidence] = []
        now = datetime.now(UTC)
        for page in context.pages:
            for index, item in enumerate(page.json_ld):
                digest = hashlib.sha256(
                    f"{page.final_url}:{index}:{json.dumps(item, sort_keys=True)}".encode()
                ).hexdigest()[:16]
                evidence.append(
                    Evidence(
                        evidence_id=f"jsonld-{digest}",
                        source_url=page.final_url,
                        source_type="structured_data",
                        collector=self.adapter_id,
                        observed_at=now,
                        observed_value=item,
                    )
                )
        return AdapterResult(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            state=DataState.AVAILABLE,
            evidence=evidence,
        )


class GeoOptimizerAdapter:
    adapter_id = "geo-optimizer"
    version = "cli-json-v1"

    def __init__(self, command: str | None) -> None:
        self.command = command

    def collect(self, context: AdapterContext) -> AdapterResult:
        if not self.command:
            return AdapterResult(
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                state=DataState.UNAVAILABLE,
                warnings=["GEO Optimizer command is not configured"],
            )
        argv = shlex.split(self.command)
        if not argv or shutil.which(argv[0]) is None:
            return AdapterResult(
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                state=DataState.UNAVAILABLE,
                warnings=["GEO Optimizer executable is unavailable"],
            )
        try:
            completed = subprocess.run(
                [
                    *argv,
                    "audit",
                    "--url",
                    str(context.site.base_url),
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            data = json.loads(completed.stdout)
            evidence = Evidence(
                evidence_id=f"geo-{hashlib.sha256(completed.stdout.encode()).hexdigest()[:16]}",
                source_url=context.site.base_url,
                source_type="third_party_diagnostic",
                collector=self.adapter_id,
                observed_at=datetime.now(UTC),
                observed_value=data,
                confidence=0.7,
                metadata={"attribution": "Auriti-Labs/geo-optimizer-skill (MIT)"},
            )
            return AdapterResult(
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                state=DataState.AVAILABLE,
                evidence=[evidence],
            )
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            return AdapterResult(
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                state=DataState.FAILED,
                warnings=[f"GEO Optimizer failed: {exc}"],
            )


class PublicResearchItem(BaseModel):
    url: str
    publisher: str
    source_class: str
    independent_source_key: str
    claims: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class PublicResearchAdapter:
    adapter_id = "public-research"
    version = "0.1.0"

    def collect(
        self, context: AdapterContext, items: list[PublicResearchItem] | None = None
    ) -> AdapterResult:
        if not items:
            return AdapterResult(
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                state=DataState.UNAVAILABLE,
                warnings=["no external public research inputs were supplied"],
            )
        evidence: list[Evidence] = []
        mentions: list[ExternalMention] = []
        warnings: list[str] = []
        seen: set[str] = set()
        now = datetime.now(UTC)
        for item in items:
            if item.independent_source_key in seen:
                warnings.append(f"syndicated duplicate skipped: {item.url}")
                continue
            seen.add(item.independent_source_key)
            digest = hashlib.sha256(item.url.encode()).hexdigest()[:16]
            evidence_id = f"external-{digest}"
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    source_url=HttpUrl(item.url),
                    source_type=item.source_class,
                    source_scope="external",
                    collector=self.adapter_id,
                    observed_at=now,
                    observed_value=item.claims,
                    confidence=item.confidence,
                    metadata={
                        "publisher": item.publisher,
                        "independent_source_key": item.independent_source_key,
                    },
                )
            )
            mentions.append(
                ExternalMention(
                    mention_id=f"mention-{digest}",
                    url=HttpUrl(item.url),
                    publisher=item.publisher,
                    source_class=item.source_class,
                    claims=item.claims,
                    evidence_ids=[evidence_id],
                    independent_source_key=item.independent_source_key,
                )
            )
        return AdapterResult(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            state=DataState.AVAILABLE if mentions else DataState.UNAVAILABLE,
            evidence=evidence,
            external_mentions=mentions,
            warnings=warnings,
        )
