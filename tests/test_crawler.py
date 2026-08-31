from datetime import UTC
from pathlib import Path

import httpx
import pytest

from ai_search_audit.config import AuditConfig
from ai_search_audit.crawler import (
    NativeCrawler,
    UnsafeTargetError,
    parse_robots_policies,
    validate_public_url,
)
from ai_search_audit.knowledge import load_registry
from ai_search_audit.models import SitemapState

ROOT = Path(__file__).parents[1]
PRODUCTS = load_registry(ROOT / "knowledge").crawler_products


def public_resolver(_hostname: str) -> list[str]:
    return ["8.8.8.8"]


def test_crawler_parses_products_supplied_by_registry_without_vendor_constants() -> None:
    registry = load_registry(ROOT / "knowledge")
    policies = parse_robots_policies(
        "User-agent: OAI-SearchBot\nDisallow: /\n",
        registry.crawler_products,
    )
    by_token = {policy.user_agent: policy for policy in policies}
    assert by_token["OAI-SearchBot"].allowed is False
    assert by_token["OAI-SearchBot"].source_rule == "openai-oai-searchbot-001"
    crawler_source = (ROOT / "src" / "ai_search_audit" / "crawler.py").read_text()
    for vendor_token in ("OAI-SearchBot", "PerplexityBot", "Claude-SearchBot", "anthropic-ai"):
        assert vendor_token not in crawler_source


def transport(request: httpx.Request) -> httpx.Response:
    pages = {
        "/robots.txt": "User-agent: *\nAllow: /\nSitemap: https://example.com/sitemap.xml\n",
        "/sitemap.xml": (
            '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://example.com/</loc></url>"
            "<url><loc>https://example.com/en</loc></url></urlset>"
        ),
        "/": """<html lang="pl"><head><title>Hotel Example</title>
            <meta name="description" content="Private hotel">
            <link rel="canonical" href="https://example.com/">
            <link rel="alternate" hreflang="en" href="https://example.com/en">
            <script type="application/ld+json">{"@type":"Hotel","name":"Example"}</script>
            </head><body><h1>Hotel</h1><h2>Private stays</h2>
            <a href="/en">English</a><a href="https://outside.test/x">Outside</a></body></html>""",
        "/en": """<html lang="en"><head><title>Example Hotel</title>
            <meta name="robots" content="noindex, follow">
            <link rel="canonical" href="https://example.com/en"></head>
            <body><h1>Example Hotel</h1></body></html>""",
    }
    if request.url.path not in pages:
        return httpx.Response(404, request=request)
    content_type = "application/xml" if request.url.path.endswith(".xml") else "text/html"
    if request.url.path.endswith(".txt"):
        content_type = "text/plain"
    return httpx.Response(
        200, text=pages[request.url.path], headers={"content-type": content_type}, request=request
    )


def test_crawler_discovers_and_extracts_required_page_fields() -> None:
    config = AuditConfig(domain="example.com", max_pages=5)
    result = NativeCrawler(
        config, PRODUCTS, transport=httpx.MockTransport(transport), resolver=public_resolver
    ).crawl()
    assert result.robots_url == "https://example.com/robots.txt"
    assert result.sitemap_urls == ["https://example.com/sitemap.xml"]
    assert result.sitemap_state is SitemapState.AVAILABLE
    assert len(result.pages) == 2
    home = result.pages[0]
    assert home.title == "Hotel Example"
    assert home.meta_description == "Private hotel"
    assert home.h1 == ["Hotel"] and home.h2 == ["Private stays"]
    assert home.canonical == "https://example.com/"
    assert home.hreflang == {"en": "https://example.com/en"}
    assert home.language == "pl"
    assert home.json_ld[0]["@type"] == "Hotel"
    assert home.internal_links == ["https://example.com/en"]
    assert home.depth == 0 and home.indexable is True
    assert result.pages[1].depth == 1
    assert result.pages[1].indexable is False
    assert "meta robots noindex" in result.pages[1].indexability_reasons
    assert result.evidence and result.evidence[0].observed_at.tzinfo == UTC
    noindex_evidence = next(
        item for item in result.evidence if str(item.source_url).rstrip("/").endswith("/en")
    )
    assert noindex_evidence.observed_value["robots_directives"] == ["noindex", "follow"]


def test_invalid_json_ld_is_recorded_in_page_evidence() -> None:
    def invalid_json_ld_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        return httpx.Response(
            200,
            text=(
                '<html><head><title>Example</title><script type="application/ld+json">'
                '{"a":1}{"b":2}</script></head><body><h1>Example</h1></body></html>'
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(invalid_json_ld_transport),
        resolver=public_resolver,
    ).crawl()

    error = result.pages[0].json_ld_errors[0]
    assert error.message == "Extra data"
    assert error.line == 1 and error.column > 1
    assert error.excerpt == '{"a":1}{"b":2}'
    page_evidence = next(item for item in result.evidence if item.source_type == "web_page")
    assert page_evidence.observed_value["json_ld_errors"][0]["message"] == "Extra data"


def test_sitemap_not_discovered_is_distinct_from_confirmed_absence() -> None:
    def no_directive(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            text="<html><head><title>Example</title></head><body></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(no_directive),
        resolver=public_resolver,
    ).crawl()
    assert result.sitemap_state is SitemapState.NOT_DISCOVERED


def test_robots_without_directive_probes_conventional_sitemap() -> None:
    requested_paths: list[str] = []

    def conventional_sitemap(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    '<?xml version="1.0"?><urlset '
                    'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    "<url><loc>https://example.com/</loc></url></urlset>"
                ),
                headers={"content-type": "application/xml"},
                request=request,
            )
        return httpx.Response(
            200,
            text="<html><head><title>Example</title></head><body><h1>Example</h1></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(conventional_sitemap),
        resolver=public_resolver,
    ).crawl()

    assert "/sitemap.xml" in requested_paths
    assert result.sitemap_urls == ["https://example.com/sitemap.xml"]
    assert result.sitemap_state is SitemapState.AVAILABLE


def test_missing_robots_and_conventional_sitemap_confirm_absence() -> None:
    def absent(request: httpx.Request) -> httpx.Response:
        if request.url.path in {"/robots.txt", "/sitemap.xml"}:
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            text="<html><head><title>Example</title></head><body></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(absent),
        resolver=public_resolver,
    ).crawl()
    assert result.sitemap_state is SitemapState.CONFIRMED_ABSENT
    sitemap_evidence = next(item for item in result.evidence if item.source_type == "sitemap")
    assert sitemap_evidence.observed_value["status_code"] == 404


def test_known_sitemap_fetch_failure_is_not_score_zero_state() -> None:
    def failed_sitemap(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="Sitemap: https://example.com/sitemap.xml\n",
                request=request,
            )
        if request.url.path == "/sitemap.xml":
            raise httpx.ConnectError("sitemap offline", request=request)
        return httpx.Response(
            200,
            text="<html><head><title>Example</title></head><body></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(failed_sitemap),
        resolver=public_resolver,
    ).crawl()
    assert result.sitemap_state is SitemapState.FETCH_FAILED
    assert any("sitemap offline" in warning for warning in result.warnings)


def test_crawler_enforces_page_limit_and_same_origin() -> None:
    config = AuditConfig(domain="example.com", max_pages=1)
    result = NativeCrawler(
        config, PRODUCTS, transport=httpx.MockTransport(transport), resolver=public_resolver
    ).crawl()
    assert len(result.pages) == 1
    assert all("outside.test" not in str(page.url) for page in result.pages)


def test_crawler_records_fetch_failures_as_warnings() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    result = NativeCrawler(
        AuditConfig(domain="example.com"),
        PRODUCTS,
        transport=httpx.MockTransport(failing),
        resolver=public_resolver,
    ).crawl()
    assert result.pages == []
    assert any("offline" in warning for warning in result.warnings)


def test_robots_policy_distinguishes_search_user_fetch_and_training_bots() -> None:
    robots = """User-agent: *
Allow: /

User-agent: OAI-SearchBot
Disallow: /

User-agent: PerplexityBot
Allow: /

User-agent: Claude-SearchBot
Disallow: /

User-agent: Claude-User
Allow: /

User-agent: GPTBot
Disallow: /
"""

    def robots_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots, request=request)
        return httpx.Response(
            200,
            text="<html><head><title>Example</title></head><body><h1>Example</h1></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(robots_transport),
        resolver=public_resolver,
    ).crawl()
    policies = {policy.user_agent: policy for policy in result.robots_policies}
    assert policies["*"].allowed is True
    assert policies["OAI-SearchBot"].allowed is False
    assert policies["OAI-SearchBot"].purpose == "search_citation"
    assert policies["PerplexityBot"].allowed is True
    assert policies["Claude-SearchBot"].allowed is False
    assert policies["Claude-User"].allowed is True
    assert policies["Claude-User"].purpose == "user_fetch"
    assert policies["GPTBot"].allowed is False
    assert policies["GPTBot"].purpose == "training"
    assert policies["Googlebot"].purpose == "traditional_search"
    robots_evidence = next(item for item in result.evidence if item.source_type == "robots_txt")
    assert "User-agent: OAI-SearchBot" in robots_evidence.observed_value["raw"]
    assert robots_evidence.observed_value["policies"]["OAI-SearchBot"]["allowed"] is False


def test_sitemap_indexes_are_followed_recursively_within_bounds() -> None:
    documents = {
        "/robots.txt": "Sitemap: https://example.com/sitemap-index.xml\n",
        "/sitemap-index.xml": (
            "<sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
            "<sitemap><loc>https://example.com/nested-index.xml</loc></sitemap>"
            "</sitemapindex>"
        ),
        "/nested-index.xml": (
            "<sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
            "<sitemap><loc>https://example.com/pages.xml</loc></sitemap>"
            "</sitemapindex>"
        ),
        "/pages.xml": (
            "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
            "<url><loc>https://example.com/from-sitemap</loc></url></urlset>"
        ),
        "/": "<html><head><title>Home</title></head><body></body></html>",
        "/from-sitemap": "<html><head><title>Nested</title></head><body></body></html>",
    }

    def sitemap_transport(request: httpx.Request) -> httpx.Response:
        value = documents.get(request.url.path)
        if value is None:
            return httpx.Response(404, request=request)
        content_type = "application/xml" if request.url.path.endswith(".xml") else "text/html"
        return httpx.Response(
            200, text=value, headers={"content-type": content_type}, request=request
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=5, max_sitemaps=3),
        PRODUCTS,
        transport=httpx.MockTransport(sitemap_transport),
        resolver=public_resolver,
    ).crawl()
    assert result.sitemap_urls == [
        "https://example.com/sitemap-index.xml",
        "https://example.com/nested-index.xml",
        "https://example.com/pages.xml",
    ]
    assert any(str(page.final_url).rstrip("/").endswith("/from-sitemap") for page in result.pages)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.0.2.1/",
    ],
)
def test_ssrf_guard_rejects_non_public_literal_targets(url: str) -> None:
    with pytest.raises(UnsafeTargetError):
        validate_public_url(url, resolver=public_resolver)


def test_ssrf_guard_rejects_hostnames_resolving_to_private_ips() -> None:
    with pytest.raises(UnsafeTargetError, match="private.test"):
        validate_public_url("https://private.test/", resolver=lambda _hostname: ["10.0.0.8"])


def test_ssrf_guard_validates_redirect_targets_before_following() -> None:
    requested_hosts: list[str] = []

    def redirect_transport(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"}, request=request)

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(redirect_transport),
        resolver=public_resolver,
    ).crawl()
    assert result.pages == []
    assert "127.0.0.1" not in requested_hosts
    assert any("unsafe target" in warning.lower() for warning in result.warnings)


def test_crawler_follows_apex_www_and_http_https_redirects_within_bounds() -> None:
    requested: list[str] = []

    def canonical_redirect_transport(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "example.com" and request.url.path == "/robots.txt":
            return httpx.Response(
                301,
                headers={"location": "https://www.example.com/robots.txt"},
                request=request,
            )
        if request.url.host == "www.example.com" and request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="Sitemap: https://www.example.com/sitemap.xml\n",
                headers={"content-type": "text/plain"},
                request=request,
            )
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                    "<url><loc>https://www.example.com/</loc></url>"
                    "<url><loc>https://www.example.com/about</loc></url>"
                    "</urlset>"
                ),
                headers={"content-type": "application/xml"},
                request=request,
            )
        if request.url.host == "example.com" and request.url.path == "/":
            return httpx.Response(
                301,
                headers={"location": "https://www.example.com/"},
                request=request,
            )
        if request.url.path == "/":
            return httpx.Response(
                200,
                text="<html><head><title>Home</title></head>"
                "<body><a href='/about'>About</a></body></html>",
                headers={"content-type": "text/html"},
                request=request,
            )
        if request.url.path == "/about":
            return httpx.Response(
                200,
                text="<html><head><title>About</title></head><body></body></html>",
                headers={"content-type": "text/html"},
                request=request,
            )
        return httpx.Response(404, request=request)

    result = NativeCrawler(
        AuditConfig(domain="http://example.com", max_pages=2, max_sitemaps=1),
        PRODUCTS,
        transport=httpx.MockTransport(canonical_redirect_transport),
        resolver=public_resolver,
    ).crawl()

    assert {str(page.final_url) for page in result.pages} == {
        "https://www.example.com/",
        "https://www.example.com/about",
    }
    home = next(page for page in result.pages if str(page.final_url).endswith(".com/"))
    assert home.redirect_chain == ["http://example.com/"]
    assert len(requested) == 6


def test_crawler_rejects_redirect_to_unrelated_public_host() -> None:
    requested_hosts: list[str] = []

    def unrelated_redirect_transport(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(
            302,
            headers={"location": "https://attacker.example/landing"},
            request=request,
        )

    result = NativeCrawler(
        AuditConfig(domain="example.com", max_pages=1),
        PRODUCTS,
        transport=httpx.MockTransport(unrelated_redirect_transport),
        resolver=public_resolver,
    ).crawl()

    assert result.pages == []
    assert "attacker.example" not in requested_hosts
    assert any("canonical host" in warning.lower() for warning in result.warnings)
