from __future__ import annotations

import codecs
import hashlib
import ipaddress
import json
import re
import socket
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from email.message import Message
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, HttpUrl

from .config import AuditConfig
from .diagnostic_models import MAX_CAPTURE_BYTES, PageCaptureResult, is_html_content_type
from .knowledge import CrawlerControlType, CrawlerProduct, CrawlerPurpose
from .models import DataState, Evidence, JsonLdParseError, Page, SitemapState

Resolver = Callable[[str], Sequence[str]]


class UnsafeTargetError(ValueError):
    pass


class _UncertainHtmlCharset(ValueError):
    pass


def _http_charsets(content_type: str) -> list[str]:
    message = Message()
    message["content-type"] = content_type
    return [
        value if isinstance(value, str) else ""
        for key, value in (message.get_params() or [])
        if key.casefold() == "charset"
    ]


class _HtmlCharsetDeclarations(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.labels: list[str] = []

    def handle_pi(self, data: str) -> None:
        if re.match(r"xml\s", data, re.IGNORECASE) and re.search(r"\bencoding\b", data):
            # XML-declared encodings are outside this bounded HTML decoder.
            # Do not silently fall back to UTF-8 for an unsupported declaration.
            self.labels.append("")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        self.labels.extend(value or "" for name, value in attrs if name == "charset")
        if any(
            name == "http-equiv" and (value or "").casefold() == "content-type"
            for name, value in attrs
        ):
            for name, value in attrs:
                if name == "content":
                    self.labels.extend(_http_charsets(value or ""))


def _decode_capture_html(body: bytes, content_type: str) -> str:
    """Decode declared text without guessing; ambiguous/unsupported HTML stays unknown."""
    unknown = "HTML charset is unsupported or invalid; content usability is unknown."
    invalid = "HTML bytes do not decode strictly; content usability is unknown."
    # UTF-32 is not a supported HTML encoding; check before its UTF-16-like prefix.
    if body.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        raise _UncertainHtmlCharset(unknown)
    bom_encoding = None
    payload = body
    for marker, encoding in (
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if body.startswith(marker):
            bom_encoding = encoding
            payload = body[len(marker) :]
            break
    try:
        # Latin-1 is only a byte-preserving view of ASCII markup, never source text.
        preview = payload.decode(bom_encoding or "latin-1", errors="strict")
    except UnicodeError as exc:
        raise _UncertainHtmlCharset(invalid) from exc
    declarations = _HtmlCharsetDeclarations()
    declarations.feed(preview)
    declarations.close()
    encodings = {bom_encoding} if bom_encoding is not None else set()
    for label in _http_charsets(content_type) + declarations.labels:
        label = label.strip(" \t\r\n\f").lower()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
            raise _UncertainHtmlCharset(unknown)
        try:
            encoding = codecs.lookup(label).name
        except (LookupError, ValueError) as exc:
            raise _UncertainHtmlCharset(unknown) from exc
        try:
            b"\x00".decode(encoding)  # Nonempty input forces the text-codec safety check.
        except UnicodeError:
            pass  # A text codec may need more bytes; the actual body is decoded strictly below.
        if encoding == "utf-16" and bom_encoding in {"utf-16-le", "utf-16-be"}:
            encoding = bom_encoding
        # Accept only this explicit browser-label subset, not Python's forgiving aliases.
        if label not in {
            "ascii",
            "us-ascii",
            "utf-8",
            "utf8",
            "utf-16",
            "utf-16le",
            "utf-16be",
        } and not re.fullmatch(r"(?:cp|windows-)125[0-8]", label):
            raise _UncertainHtmlCharset(unknown)
        encodings.add(encoding)
    if len(encodings) > 1:
        raise _UncertainHtmlCharset(
            "HTML charset declarations conflict; content usability is unknown."
        )
    encoding = next(iter(encodings), "utf-8")
    if encoding.startswith("utf-16") and bom_encoding is None:
        raise _UncertainHtmlCharset(unknown)
    try:
        html = payload.decode(encoding, errors="strict")
    except UnicodeError as exc:
        raise _UncertainHtmlCharset(invalid) from exc
    if "\x00" in html:
        raise _UncertainHtmlCharset(invalid)
    return html


class RobotsPolicy(BaseModel):
    identifier: str
    user_agent: str
    vendor: str
    purpose: CrawlerPurpose
    control_type: CrawlerControlType
    robots_txt_respected: bool
    source_rule: str
    verified_at: date
    allowed: bool | None
    matched_directive: str | None = None


class CrawlResult(BaseModel):
    pages: list[Page] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    robots_url: str | None = None
    sitemap_urls: list[str] = Field(default_factory=list)
    sitemap_state: SitemapState = SitemapState.UNAVAILABLE
    robots_policies: list[RobotsPolicy] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _normalized_url(value: str) -> str:
    parts = urlsplit(value)
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _canonical_host(hostname: str | None) -> str:
    host = (hostname or "").casefold().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _effective_port(url: str) -> int | None:
    parts = urlsplit(url)
    if parts.port is not None:
        return parts.port
    return {"http": 80, "https": 443}.get(parts.scheme.casefold())


def _canonical_host_variant(left: str, right: str) -> bool:
    a, b = urlsplit(left), urlsplit(right)
    return bool(a.hostname and b.hostname) and _canonical_host(a.hostname) == _canonical_host(
        b.hostname
    )


def _canonical_scope(base: str, candidate: str) -> bool:
    if not _canonical_host_variant(base, candidate):
        return False
    base_parts, candidate_parts = urlsplit(base), urlsplit(candidate)
    base_scheme = base_parts.scheme.casefold()
    candidate_scheme = candidate_parts.scheme.casefold()
    try:
        if base_scheme == candidate_scheme:
            return _effective_port(base) == _effective_port(candidate)
        return (
            base_scheme == "http"
            and candidate_scheme == "https"
            and _effective_port(base) == 80
            and _effective_port(candidate) == 443
        )
    except ValueError:
        return False


def _canonical_redirect_allowed(source: str, target: str) -> bool:
    return _canonical_scope(source, target)


def _system_resolver(hostname: str) -> list[str]:
    try:
        return sorted(
            {
                str(item[4][0])
                for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        )
    except socket.gaierror as exc:
        raise UnsafeTargetError(f"could not resolve target host {hostname}: {exc}") from exc


def validate_public_url(url: str, *, resolver: Resolver | None = None) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise UnsafeTargetError(f"unsafe target URL: {url}")
    if parts.username is not None or parts.password is not None:
        raise UnsafeTargetError(f"credentials are not allowed in target URL: {url}")
    try:
        _ = parts.port
    except ValueError as exc:
        raise UnsafeTargetError(f"unsafe target port: {url}") from exc
    hostname = parts.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeTargetError(f"unsafe target host: {hostname}")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        addresses = (resolver or _system_resolver)(hostname)
        if not addresses:
            raise UnsafeTargetError(f"target host resolved to no addresses: {hostname}") from None
        parsed_addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        try:
            parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
        except ValueError as exc:
            raise UnsafeTargetError(
                f"invalid address resolved for target host: {hostname}"
            ) from exc
    else:
        parsed_addresses = [literal]
    if any(not address.is_global for address in parsed_addresses):
        raise UnsafeTargetError(f"unsafe target host or resolved address: {hostname}")


def _robots_groups(text: str) -> list[tuple[list[str], list[tuple[str, str]]]]:
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    agents: list[str] = []
    directives: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            if agents and directives:
                groups.append((agents, directives))
                agents, directives = [], []
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key, value = key.strip().casefold(), value.strip()
        if key == "user-agent":
            if directives:
                groups.append((agents, directives))
                agents, directives = [], []
            agents.append(value.casefold())
        elif key in {"allow", "disallow"} and agents:
            directives.append((key, value))
    if agents:
        groups.append((agents, directives))
    return groups


def _robots_path_matches(pattern: str, path: str = "/") -> bool:
    if not pattern:
        return False
    end_anchored = pattern.endswith("$")
    body = pattern[:-1] if end_anchored else pattern
    expression = "^" + re.escape(body).replace(r"\*", ".*")
    if end_anchored:
        expression += "$"
    return re.search(expression, path) is not None


def parse_robots_policies(text: str, products: list[CrawlerProduct]) -> list[RobotsPolicy]:
    groups = _robots_groups(text)
    policies: list[RobotsPolicy] = []
    for product in products:
        token = product.token.casefold()
        exact = [directives for agents, directives in groups if token in agents and token != "*"]
        applicable = exact or [directives for agents, directives in groups if "*" in agents]
        matching = [
            (key, value)
            for directives in applicable
            for key, value in directives
            if _robots_path_matches(value)
        ]
        if matching:
            key, value = max(
                matching,
                key=lambda item: (len(item[1].rstrip("$")), item[0] == "allow"),
            )
            allowed = key == "allow" if product.robots_txt_respected else None
            matched = f"{key.title()}: {value}"
        else:
            allowed = True if product.robots_txt_respected else None
            matched = None
        policies.append(
            RobotsPolicy(
                identifier=product.identifier,
                user_agent=product.token,
                vendor=product.vendor,
                purpose=product.purpose,
                control_type=product.control_type,
                robots_txt_respected=product.robots_txt_respected,
                source_rule=product.source_rule,
                verified_at=product.verified_at,
                allowed=allowed,
                matched_directive=matched,
            )
        )
    return policies


class NativeCrawler:
    version = "0.1.1"

    def __init__(
        self,
        config: AuditConfig,
        products: list[CrawlerProduct],
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self.config = config
        self.products = products
        self.transport = transport
        self.resolver = resolver

    def _validate_request(self, request: httpx.Request) -> None:
        validate_public_url(str(request.url), resolver=self.resolver)

    def _validate_redirect(self, response: httpx.Response) -> None:
        if not response.is_redirect:
            return
        location = response.headers.get("location")
        if not location:
            return
        source = str(response.request.url)
        target = _normalized_url(urljoin(source, location))
        validate_public_url(target, resolver=self.resolver)
        if not _canonical_redirect_allowed(source, target):
            raise UnsafeTargetError(
                f"redirect target is outside the canonical host family: {target}"
            )

    def _get(self, client: httpx.Client, url: str, warnings: list[str]) -> httpx.Response | None:
        try:
            response = client.get(url)
            if len(response.content) > self.config.max_response_bytes:
                warnings.append(f"response exceeded size limit: {url}")
                return None
            return response
        except (httpx.HTTPError, UnsafeTargetError) as exc:
            warnings.append(f"fetch failed for {url}: {exc}")
            return None

    def capture_page(self, url: str) -> PageCaptureResult:
        """Collect one fresh public page, bounding each redirect body and final response."""
        requested_url: str | None = None
        final_url: str | None = None
        status_code: int | None = None
        content_type: str | None = None
        truncated = False
        failure = "HTTP collection failed."
        try:
            validate_public_url(url, resolver=self.resolver)
            if not _canonical_scope(self.config.domain, url):
                raise UnsafeTargetError("initial capture target is outside canonical scope")
            httpx.URL(url)  # Validate HTTP syntax before retaining requested-URL metadata.
            requested_url = url
            current = _normalized_url(url)
            limit = min(self.config.max_response_bytes, MAX_CAPTURE_BYTES)
            # Follow explicitly so httpx cannot consume an unbounded redirect body.
            # Both existing security hooks still run on every HTTP exchange.
            with httpx.Client(
                transport=self.transport,
                headers={"user-agent": self.config.user_agent},
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
                max_redirects=self.config.max_redirects,
                event_hooks={
                    "request": [self._validate_request],
                    "response": [self._validate_redirect],
                },
            ) as client:
                for redirect_count in range(self.config.max_redirects + 1):
                    with client.stream("GET", current) as response:
                        candidate = _normalized_url(str(response.url))
                        if not _canonical_scope(self.config.domain, candidate):
                            raise UnsafeTargetError(
                                "final capture target is outside canonical scope"
                            )
                        final_url = candidate
                        status_code = response.status_code
                        content_type = response.headers.get("content-type")
                        body = bytearray()
                        for chunk in response.iter_bytes():
                            if len(body) + len(chunk) > limit:
                                truncated = True
                                failure = (
                                    "Response exceeded capture size limit; "
                                    "no partial body retained."
                                )
                                raise ValueError(failure)
                            body.extend(chunk)
                        if response.is_redirect and response.headers.get("location"):
                            if redirect_count >= self.config.max_redirects:
                                failure = "HTTP collection exceeded configured redirect limit."
                                raise ValueError(failure)
                            current = _normalized_url(
                                urljoin(candidate, response.headers["location"])
                            )
                            continue
                        html = None
                        decoding_limitation = None
                        if is_html_content_type(content_type):
                            try:
                                assert content_type is not None
                                html = _decode_capture_html(bytes(body), content_type)
                            except _UncertainHtmlCharset as exc:
                                decoding_limitation = str(exc)
                            except LookupError as exc:
                                failure = "HTML decoding failed: unsupported text charset."
                                raise ValueError(failure) from exc
                            if html is not None and len(html.encode()) > MAX_CAPTURE_BYTES:
                                truncated = True
                                failure = (
                                    "Decoded HTML exceeded capture size limit; "
                                    "no partial body retained."
                                )
                                raise ValueError(failure)
                        state: Literal[DataState.AVAILABLE, DataState.UNKNOWN, DataState.FAILED] = (
                            DataState.AVAILABLE if response.is_success else DataState.UNKNOWN
                        )
                        limitations: tuple[str, ...] = (
                            ()
                            if response.is_success
                            else (f"HTTP {status_code} is not successful content evidence.",)
                        )
                        if html is None:
                            state = DataState.UNKNOWN
                            limitations += (
                                decoding_limitation
                                or (
                                    "Missing or unsupported content type; "
                                    "HTML usability is unknown."
                                ),
                            )
                        if response.headers.get("cf-mitigated", "").casefold() == "challenge":
                            state = DataState.UNKNOWN
                            limitations += ("HTTP response identifies a challenge page.",)
                        return PageCaptureResult(
                            url=requested_url,
                            final_url=final_url,
                            observed_at=datetime.now(UTC),
                            collector=f"native-crawler/{self.version}",
                            status_code=status_code,
                            content_type=content_type,
                            body=bytes(body),
                            html=html,
                            complete=True,
                            truncated=False,
                            state=state,
                            limitations=limitations,
                        )
        except UnsafeTargetError:
            failure = "Public-target safety check failed; capture was not collected."
        except httpx.InvalidURL:
            failure = "Invalid HTTP target URL; capture was not collected."
        except httpx.HTTPError as exc:
            failure = f"HTTP collection failed: {type(exc).__name__}."
        except ValueError:
            # Do not echo arbitrary input URLs, credentials or server error details.
            pass
        return PageCaptureResult(
            url=requested_url,
            final_url=final_url,
            observed_at=datetime.now(UTC),
            collector=f"native-crawler/{self.version}",
            status_code=status_code,
            content_type=content_type,
            html=None,
            complete=False,
            truncated=truncated,
            state=DataState.FAILED,
            limitations=(failure,),
        )

    def crawl(self) -> CrawlResult:
        base = _normalized_url(self.config.domain)
        result = CrawlResult(robots_url=urljoin(base, "/robots.txt"))
        headers = {"user-agent": self.config.user_agent}
        with httpx.Client(
            transport=self.transport,
            headers=headers,
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            max_redirects=self.config.max_redirects,
            event_hooks={
                "request": [self._validate_request],
                "response": [self._validate_redirect],
            },
        ) as client:
            robots_url = result.robots_url
            assert robots_url is not None
            robots = self._get(client, robots_url, result.warnings)
            probe_conventional_sitemap = False
            conventional_absence_is_confirmed = False
            if robots is not None and robots.status_code == 200:
                result.robots_policies = parse_robots_policies(robots.text, self.products)
                for line in robots.text.splitlines():
                    key, separator, value = line.partition(":")
                    if separator and key.strip().lower() == "sitemap":
                        sitemap = value.strip()
                        if sitemap and sitemap not in result.sitemap_urls:
                            result.sitemap_urls.append(sitemap)
                robots_fingerprint = hashlib.sha256(robots.content).hexdigest()
                result.evidence.append(
                    Evidence(
                        evidence_id=f"robots-{robots_fingerprint[:16]}",
                        source_url=HttpUrl(str(robots.url)),
                        source_type="robots_txt",
                        collector="native-crawler",
                        observed_at=datetime.now(UTC),
                        observed_value={
                            "raw": robots.text,
                            "sitemaps": result.sitemap_urls,
                            "policies": {
                                policy.user_agent: policy.model_dump(mode="json")
                                for policy in result.robots_policies
                            },
                        },
                        content_fingerprint=robots_fingerprint,
                    )
                )
                if not result.sitemap_urls:
                    result.sitemap_urls.append(urljoin(base, "/sitemap.xml"))
                    probe_conventional_sitemap = True
            elif robots is not None and robots.status_code in {404, 410}:
                result.sitemap_urls.append(urljoin(base, "/sitemap.xml"))
                probe_conventional_sitemap = True
                conventional_absence_is_confirmed = True

            seeds = [base]
            sitemap_queue: deque[str] = deque(result.sitemap_urls)
            known_sitemaps = set(result.sitemap_urls)
            result.sitemap_urls = []
            parsed_sitemaps = 0
            failed_sitemaps = 0
            absent_sitemaps = 0
            while sitemap_queue and len(result.sitemap_urls) < self.config.max_sitemaps:
                sitemap_url = sitemap_queue.popleft()
                result.sitemap_urls.append(sitemap_url)
                response = self._get(client, sitemap_url, result.warnings)
                if response is None:
                    failed_sitemaps += 1
                    continue
                sitemap_fingerprint = hashlib.sha256(response.content).hexdigest()
                observation: dict[str, Any] = {"status_code": response.status_code}
                if response.status_code != 200:
                    failed_sitemaps += 1
                    if response.status_code in {404, 410}:
                        absent_sitemaps += 1
                    result.evidence.append(
                        Evidence(
                            evidence_id=f"sitemap-{sitemap_fingerprint[:16]}",
                            source_url=HttpUrl(str(response.url)),
                            source_type="sitemap",
                            collector="native-crawler",
                            observed_at=datetime.now(UTC),
                            observed_value=observation,
                            content_fingerprint=sitemap_fingerprint,
                        )
                    )
                    continue
                try:
                    root = ElementTree.fromstring(response.content)
                    root_name = root.tag.rsplit("}", 1)[-1].casefold()
                    observation["document_type"] = root_name
                    locations = [
                        loc.text.strip()
                        for loc in root.findall(".//{*}loc")
                        if loc.text and _canonical_scope(base, loc.text.strip())
                    ]
                    if root_name == "sitemapindex":
                        parsed_sitemaps += 1
                        for location in locations:
                            normalized = _normalized_url(location)
                            if normalized not in known_sitemaps:
                                known_sitemaps.add(normalized)
                                sitemap_queue.append(normalized)
                    elif root_name == "urlset":
                        parsed_sitemaps += 1
                        seeds.extend(_normalized_url(location) for location in locations)
                    else:
                        failed_sitemaps += 1
                        result.warnings.append(
                            f"unsupported sitemap document type {root_name}: {sitemap_url}"
                        )
                except ElementTree.ParseError as exc:
                    failed_sitemaps += 1
                    observation["parse_error"] = str(exc)
                    result.warnings.append(f"invalid sitemap {sitemap_url}: {exc}")
                result.evidence.append(
                    Evidence(
                        evidence_id=f"sitemap-{sitemap_fingerprint[:16]}",
                        source_url=HttpUrl(str(response.url)),
                        source_type="sitemap",
                        collector="native-crawler",
                        observed_at=datetime.now(UTC),
                        observed_value=observation,
                        content_fingerprint=sitemap_fingerprint,
                    )
                )

            if parsed_sitemaps:
                result.sitemap_state = SitemapState.AVAILABLE
            elif result.sitemap_urls:
                if probe_conventional_sitemap and absent_sitemaps == len(result.sitemap_urls):
                    result.sitemap_state = (
                        SitemapState.CONFIRMED_ABSENT
                        if conventional_absence_is_confirmed
                        else SitemapState.NOT_DISCOVERED
                    )
                elif failed_sitemaps:
                    result.sitemap_state = SitemapState.FETCH_FAILED

            queue: deque[tuple[str, int]] = deque(
                (seed, 0 if seed == base else 1) for seed in seeds
            )
            seen: set[str] = set()
            while queue and len(result.pages) < self.config.max_pages:
                url, depth = queue.popleft()
                url = _normalized_url(url)
                if url in seen or not _canonical_scope(base, url):
                    continue
                seen.add(url)
                response = self._get(client, url, result.warnings)
                if response is None:
                    continue
                final_url = _normalized_url(str(response.url))
                if final_url in seen and final_url != url:
                    continue
                seen.add(final_url)
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type:
                    result.warnings.append(f"unsupported content type for {url}: {content_type}")
                    continue
                page, links, parse_warnings = self._parse_page(response, depth, result.sitemap_urls)
                result.pages.append(page)
                result.warnings.extend(parse_warnings)
                fingerprint = hashlib.sha256(response.content).hexdigest()
                result.evidence.append(
                    Evidence(
                        evidence_id=f"page-{fingerprint[:16]}",
                        source_url=page.final_url,
                        source_type="web_page",
                        collector="native-crawler",
                        observed_at=datetime.now(UTC),
                        observed_value={
                            "status_code": page.status_code,
                            "title": page.title,
                            "canonical": page.canonical,
                            "indexable": page.indexable,
                            "robots_directives": page.robots_directives,
                            "indexability_reasons": page.indexability_reasons,
                            "json_ld_errors": [
                                item.model_dump(mode="json") for item in page.json_ld_errors
                            ],
                        },
                        content_fingerprint=fingerprint,
                    )
                )
                for link in links:
                    if link not in seen:
                        queue.append((link, depth + 1))
                if self.config.delay_seconds:
                    time.sleep(self.config.delay_seconds)
        return result

    def _parse_page(
        self, response: httpx.Response, depth: int, sitemap_urls: list[str]
    ) -> tuple[Page, list[str], list[str]]:
        soup = BeautifulSoup(response.text, "html.parser")
        final_url = str(response.url)
        warnings: list[str] = []
        canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
        canonical = urljoin(final_url, str(canonical_tag.get("href"))) if canonical_tag else None
        hreflang: dict[str, str] = {}
        for tag in soup.find_all("link", rel=lambda value: value and "alternate" in value):
            language, href = tag.get("hreflang"), tag.get("href")
            if language and href:
                hreflang[str(language)] = urljoin(final_url, str(href))
        robots_directives: list[str] = []
        for meta in soup.find_all("meta"):
            if str(meta.get("name", "")).lower() in {"robots", "googlebot", "bingbot"}:
                robots_directives.extend(
                    item.strip().lower()
                    for item in str(meta.get("content", "")).split(",")
                    if item.strip()
                )
        x_robots_tag = response.headers.get("x-robots-tag")
        if x_robots_tag:
            robots_directives.extend(
                item.strip().lower() for item in x_robots_tag.split(",") if item.strip()
            )
        json_ld: list[dict[str, Any]] = []
        json_ld_errors: list[JsonLdParseError] = []
        for script in soup.find_all("script", type="application/ld+json"):
            raw_json_ld = script.get_text().strip()
            try:
                value = json.loads(raw_json_ld)
                if isinstance(value, dict):
                    json_ld.append(value)
                elif isinstance(value, list):
                    json_ld.extend(item for item in value if isinstance(item, dict))
            except json.JSONDecodeError as exc:
                json_ld_errors.append(
                    JsonLdParseError(
                        message=exc.msg,
                        line=exc.lineno,
                        column=exc.colno,
                        excerpt=raw_json_ld[:240],
                    )
                )
                warnings.append(f"invalid JSON-LD on {final_url}: {exc.msg}")
        internal_links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            link = _normalized_url(urljoin(final_url, str(anchor["href"])))
            if _canonical_scope(final_url, link) and link not in internal_links:
                internal_links.append(link)
        reasons: list[str] = []
        status_code = response.status_code
        indexable = 200 <= status_code < 300
        if not indexable:
            reasons.append(f"HTTP status {status_code}")
        if "noindex" in robots_directives:
            indexable = False
            reasons.append("meta robots noindex")
        if indexable:
            reasons.append("successful HTML response without noindex")
        title = soup.title.get_text(" ", strip=True) if soup.title else None

        def is_description(value: str | None) -> bool:
            return value is not None and value.lower() == "description"

        description_tag = soup.find("meta", attrs={"name": is_description})
        content_text = soup.get_text(" ", strip=True)[: self.config.max_text_chars]
        redirect_chain = [str(item.url) for item in response.history]
        page = Page(
            url=HttpUrl(redirect_chain[0] if redirect_chain else final_url),
            final_url=HttpUrl(final_url),
            status_code=status_code,
            redirect_chain=redirect_chain,
            title=title,
            meta_description=str(description_tag.get("content")) if description_tag else None,
            h1=[tag.get_text(" ", strip=True) for tag in soup.find_all("h1")],
            h2=[tag.get_text(" ", strip=True) for tag in soup.find_all("h2")],
            canonical=canonical,
            hreflang=hreflang,
            robots_directives=robots_directives,
            internal_links=internal_links,
            sitemap_references=sitemap_urls,
            language=str(soup.html.get("lang")) if soup.html and soup.html.get("lang") else None,
            json_ld=json_ld,
            json_ld_errors=json_ld_errors,
            content_text=content_text,
            indexable=indexable,
            indexability_reasons=reasons,
            depth=depth,
        )
        return page, internal_links, warnings
