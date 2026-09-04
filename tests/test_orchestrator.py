import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ai_search_audit import orchestrator
from ai_search_audit.adapters import PublicResearchItem
from ai_search_audit.knowledge import default_registry_root, load_registry
from ai_search_audit.models import DataState
from ai_search_audit.orchestrator import run_public_audit
from ai_search_audit.prompts import PROMPT_PACK_VERSION
from tests.test_crawler import public_resolver, transport


def _ai_policy_transport(*, block_training: bool) -> httpx.MockTransport:
    training_directive = "Disallow: /" if block_training else "Allow: /"
    robots = f"""User-agent: *
Allow: /

User-agent: OAI-SearchBot
Disallow: /

User-agent: PerplexityBot
Disallow: /

User-agent: Claude-SearchBot
Disallow: /

User-agent: GPTBot
{training_directive}

User-agent: ClaudeBot
{training_directive}
"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text=robots,
                headers={"content-type": "text/plain"},
                request=request,
            )
        return httpx.Response(
            200,
            text=(
                "<html><head><title>Example</title>"
                "<link rel='canonical' href='https://example.com/'></head>"
                "<body><h1>Example</h1><h2>Services</h2></body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    return httpx.MockTransport(handler)


def test_public_audit_writes_all_required_artifacts(tmp_path: Path) -> None:
    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
        research_items=[
            PublicResearchItem(
                url="https://press.example/story",
                publisher="Press",
                source_class="press",
                independent_source_key="press",
                claims={"brand": "Example", "rooms": "24"},
            )
        ],
    )
    expected = {
        "audit.json",
        "evidence.jsonl",
        "implementation-backlog.csv",
        "report-draft.json",
        "client-report-data.json",
        "client-report.pdf",
        "ai-prompts.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert run.audit_engine_version == "0.3.0"
    assert run.configuration["entity_classification_policy"] == "2.0.0"
    assert run.ruleset_version == load_registry(default_registry_root()).version
    assert run.adapter_versions["public-research"] == "0.1.0"
    scores = {score.name: score for score in run.scores}
    assert scores["Measurement Maturity"].state is DataState.UNAVAILABLE
    assert scores["Measurement Maturity"].value is None
    assert scores["Observed AI Visibility"].state is DataState.UNAVAILABLE
    assert scores["Technical Search Readiness"].total_checks >= 3
    assert scores["Content Citability"].total_checks >= 3
    assert scores["Technical Search Readiness"].confidence <= 1
    assert all(score.coverage <= 1 for score in scores.values())
    prompt_pack = json.loads((tmp_path / "ai-prompts.json").read_text())
    assert prompt_pack["version"] == PROMPT_PACK_VERSION
    assert prompt_pack["prompts"]
    report = json.loads((tmp_path / "client-report-data.json").read_text())
    assert report["report_schema_version"] == "1.2.0"
    assert report["report_template_version"] == "1.2.0"
    assert report["renderer"]["renderer_version"] == "69.0"
    assert report["evidence_appendix"]
    assert (tmp_path / "client-report.pdf").stat().st_mtime_ns >= (
        tmp_path / "client-report-data.json"
    ).stat().st_mtime_ns


def test_report_draft_is_persisted_before_anti_slop_provider_runs(tmp_path: Path) -> None:
    def provider(payload):
        persisted = json.loads((tmp_path / "report-draft.json").read_text())
        assert persisted["audit_id"].startswith("audit-")
        assert set(payload.model_fields_set) == {"executive_summary", "findings"}
        return payload

    run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
        rewrite_provider=provider,
    )


def test_failed_compilation_removes_stale_final_report_artifacts(tmp_path: Path) -> None:
    (tmp_path / "client-report.pdf").write_bytes(b"stale pdf")
    (tmp_path / "client-report-data.json").write_text('{"stale": true}')

    def remove_narrative_item(payload):
        return payload.model_copy(update={"findings": []})

    with pytest.raises(ValueError, match="protected finding changed"):
        run_public_audit(
            "example.com",
            output_dir=tmp_path,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
            rewrite_provider=remove_narrative_item,
        )
    assert (tmp_path / "report-draft.json").exists()
    assert not (tmp_path / "client-report-data.json").exists()
    assert not (tmp_path / "client-report.pdf").exists()


def test_failed_render_leaves_no_final_or_staging_report_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "client-report.pdf").write_bytes(b"stale pdf")
    (tmp_path / "client-report-data.json").write_text('{"stale": true}')

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(orchestrator, "render_client_report", fail_render)

    with pytest.raises(RuntimeError, match="renderer failed"):
        run_public_audit(
            "example.com",
            output_dir=tmp_path,
            crawler_transport=httpx.MockTransport(transport),
            crawler_resolver=public_resolver,
        )

    assert not (tmp_path / "client-report-data.json").exists()
    assert not (tmp_path / "client-report.pdf").exists()
    assert not list(tmp_path.glob(".report-run.*"))


def test_report_records_deterministic_anti_slop_fallback(tmp_path: Path) -> None:
    run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
    )
    report = json.loads((tmp_path / "client-report-data.json").read_text())
    assert report["anti_slop"]["route"] == "deterministic-fallback"
    assert report["anti_slop"]["agent_skill_state"] == "UNAVAILABLE"


def test_client_report_and_backlog_use_rewritten_finding_content(tmp_path: Path) -> None:
    def rewrite_provider(payload):
        payload.findings[0].client_title = "Rewritten client-facing title"
        payload.findings[0].implementation = "Rewritten implementation guidance."
        return payload

    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(transport),
        crawler_resolver=public_resolver,
        rewrite_provider=rewrite_provider,
    )
    report = json.loads((tmp_path / "client-report-data.json").read_text())
    backlog = (tmp_path / "implementation-backlog.csv").read_text()
    assert run.findings[0].client_title != "Rewritten client-facing title"
    assert report["findings"][0]["client_title"] == "Rewritten client-facing title"
    assert "Rewritten implementation guidance." in backlog


def test_blocked_ai_search_products_create_findings_without_training_score_effect(
    tmp_path: Path,
) -> None:
    timestamp = datetime(2026, 8, 11, tzinfo=UTC)
    training_allowed = run_public_audit(
        "example.com",
        output_dir=tmp_path / "training-allowed",
        crawler_transport=_ai_policy_transport(block_training=False),
        crawler_resolver=public_resolver,
        now=timestamp,
    )
    training_blocked = run_public_audit(
        "example.com",
        output_dir=tmp_path / "training-blocked",
        crawler_transport=_ai_policy_transport(block_training=True),
        crawler_resolver=public_resolver,
        now=timestamp,
    )
    findings = [
        finding for finding in training_allowed.findings if finding.category == "ai_search_access"
    ]
    assert {finding.rule_id for finding in findings} == {
        "openai-oai-searchbot-001",
        "perplexity-bots-001",
        "anthropic-crawler-001",
    }
    assert {
        claim.predicate
        for finding in findings
        for claim in finding.factual_claims
        if claim.predicate.endswith("robots_access")
    } == {
        "crawler.openai-search.robots_access",
        "crawler.perplexity-search.robots_access",
        "crawler.anthropic-search.robots_access",
    }
    robots_evidence = next(
        item for item in training_allowed.evidence if item.source_type == "robots_txt"
    )
    assert all(finding.evidence_ids == [robots_evidence.evidence_id] for finding in findings)
    assert all(finding.affected_urls == [str(robots_evidence.source_url)] for finding in findings)
    assert not any("training" in finding.finding_id for finding in findings)

    allowed_score = next(
        score for score in training_allowed.scores if score.name == "Technical Search Readiness"
    )
    blocked_score = next(
        score for score in training_blocked.scores if score.name == "Technical Search Readiness"
    )
    assert allowed_score.value == blocked_score.value
    assert allowed_score.coverage == blocked_score.coverage
    check_names = {check.name for check in allowed_score.checks}
    assert {
        "OAI-SearchBot search/citation access",
        "PerplexityBot search/citation access",
        "Claude-SearchBot search/citation access",
    } <= check_names
    assert all("GPTBot" not in name and "ClaudeBot" not in name for name in check_names)


def test_sitemap_fetch_failure_remains_non_numeric_and_explained(tmp_path: Path) -> None:
    def sitemap_failure(request: httpx.Request) -> httpx.Response:
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
            text=(
                "<html><head><title>Example</title><meta name='description' content='Example'>"
                "<link rel='canonical' href='https://example.com/'></head>"
                "<body><h1>Example</h1><h2>Services</h2>"
                + ("Public facts and details. " * 8)
                + "</body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(sitemap_failure),
        crawler_resolver=public_resolver,
    )
    technical = next(score for score in run.scores if score.name == "Technical Search Readiness")
    sitemap = next(check for check in technical.checks if check.name == "sitemap discovery")
    assert sitemap.state is DataState.FAILED
    assert sitemap.score is None
    assert sitemap.unavailable_reason and "fetch" in sitemap.unavailable_reason.casefold()


def test_conventional_sitemap_probe_avoids_deduction(tmp_path: Path) -> None:
    def conventional_sitemap(request: httpx.Request) -> httpx.Response:
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
            text=(
                "<html><head><title>Example</title><meta name='description' content='Example'>"
                "<link rel='canonical' href='https://example.com/'></head>"
                "<body><h1>Example</h1><h2>Services</h2>"
                + ("Public facts and details. " * 8)
                + "</body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(conventional_sitemap),
        crawler_resolver=public_resolver,
    )

    assert run.sitemap_state.value == "AVAILABLE"
    assert not any(finding.category == "sitemap_discovery" for finding in run.findings)
    sitemap = next(
        check
        for score in run.scores
        if score.name == "Technical Search Readiness"
        for check in score.checks
        if check.name == "sitemap discovery"
    )
    assert sitemap.score == 100


def test_non_2xx_html_page_is_excluded_from_page_quality_denominators(
    tmp_path: Path,
) -> None:
    def page_with_404(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    '<?xml version="1.0"?><urlset '
                    'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    "<url><loc>https://example.com/</loc></url>"
                    "<url><loc>https://example.com/missing</loc></url></urlset>"
                ),
                headers={"content-type": "application/xml"},
                request=request,
            )
        if request.url.path == "/missing":
            return httpx.Response(
                404,
                text="<html><head><title>Missing</title></head><body><h1>Missing</h1></body></html>",
                headers={"content-type": "text/html"},
                request=request,
            )
        return httpx.Response(
            200,
            text=(
                "<html><head><title>Example</title><meta name='description' content='Example'>"
                "<link rel='canonical' href='https://example.com/'>"
                '<script type="application/ld+json">'
                '{"@type":"Organization","name":"Example"}</script></head>'
                "<body><h1>Example</h1><h2>Services</h2>"
                + ("Public facts and details. " * 8)
                + "</body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(page_with_404),
        crawler_resolver=public_resolver,
    )

    assert {page.status_code for page in run.pages} == {200, 404}
    technical = next(score for score in run.scores if score.name == "Technical Search Readiness")
    checks = {check.name: check for check in technical.checks}
    assert checks["public crawl and indexability"].score == 100
    assert checks["canonical annotations"].score == 100
    content = next(score for score in run.scores if score.name == "Content Citability")
    assert all(check.score == 100 for check in content.checks)


def test_partial_invalid_json_ld_scores_proportionally(tmp_path: Path) -> None:
    def mixed_json_ld(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                text=(
                    '<?xml version="1.0"?><urlset '
                    'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    "<url><loc>https://example.com/</loc></url>"
                    "<url><loc>https://example.com/valid</loc></url>"
                    "<url><loc>https://example.com/invalid</loc></url></urlset>"
                ),
                headers={"content-type": "application/xml"},
                request=request,
            )
        json_ld = (
            '{"@type":"Organization","name":"Example"}'
            if request.url.path != "/invalid"
            else '{"@type":"Organization"}{"name":"Example"}'
        )
        canonical = f"https://example.com{request.url.path}"
        return httpx.Response(
            200,
            text=(
                "<html><head><title>Example</title><meta name='description' content='Example'>"
                f"<link rel='canonical' href='{canonical}'>"
                f'<script type="application/ld+json">{json_ld}</script></head>'
                "<body><h1>Example</h1><h2>Services</h2>"
                + ("Public facts and details. " * 8)
                + "</body></html>"
            ),
            headers={"content-type": "text/html"},
            request=request,
        )

    run = run_public_audit(
        "example.com",
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(mixed_json_ld),
        crawler_resolver=public_resolver,
    )

    invalid = next(
        finding for finding in run.findings if finding.category == "structured_data_validity"
    )
    structured = next(
        check
        for score in run.scores
        if score.name == "Entity & Machine Understanding"
        for check in score.checks
        if check.name == "structured data observed"
    )
    assert structured.score == pytest.approx(66.67, abs=0.01)
    assert structured.explanation_finding_ids == [invalid.finding_id]
