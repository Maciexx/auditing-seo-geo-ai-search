from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai_search_audit.analyzers import analyze_pages, validate_findings_rules
from ai_search_audit.knowledge import RuleState, load_registry
from ai_search_audit.models import Evidence, Finding, FindingStatus, Page, SitemapState

REGISTRY = load_registry(Path(__file__).parents[1] / "knowledge")


def test_page_analyzer_creates_evidence_backed_noindex_finding() -> None:
    page = Page(
        url="https://example.com/hidden",
        final_url="https://example.com/hidden",
        status_code=200,
        robots_directives=["noindex"],
        indexable=False,
        indexability_reasons=["meta robots noindex"],
    )
    evidence = Evidence(
        evidence_id="page-1",
        source_url=page.url,
        source_type="web_page",
        collector="native-crawler",
        observed_at=datetime.now(UTC),
        observed_value={"indexable": False},
    )
    findings = analyze_pages([page], [evidence], REGISTRY, as_of=datetime(2026, 8, 11).date())
    finding = next(item for item in findings if item.rule_id == "google-noindex-001")
    assert finding.status is FindingStatus.CONFIRMED
    assert finding.evidence_ids == ["page-1"]
    assert finding.rule_evidence_level == "A"
    assert finding.rule_confidence == 0.99
    assert finding.rule_scoring_weight == 1
    assert finding.rule_state is RuleState.CURRENT


def test_page_analyzer_marks_missing_canonical_as_inferred() -> None:
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        indexable=True,
    )
    evidence = Evidence(
        evidence_id="page-2",
        source_url=page.url,
        source_type="web_page",
        collector="native-crawler",
        observed_at=datetime.now(UTC),
        observed_value={"canonical": None},
    )
    finding = next(
        item
        for item in analyze_pages([page], [evidence], REGISTRY, as_of=datetime(2026, 8, 11).date())
        if "canonical" in item.rule_id
    )
    assert finding.status is FindingStatus.INFERRED
    assert "may" in finding.business_impact.lower()


def test_stale_rule_downgrades_finding_to_requires_verification() -> None:
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        indexable=True,
    )
    evidence = Evidence(
        evidence_id="page-3",
        source_url=page.url,
        source_type="web_page",
        collector="native-crawler",
        observed_at=datetime.now(UTC),
        observed_value={"canonical": None},
    )
    finding = analyze_pages([page], [evidence], REGISTRY, as_of=datetime(2030, 1, 1).date())[0]
    assert finding.status is FindingStatus.REQUIRES_VERIFICATION
    assert finding.rule_state is RuleState.REQUIRES_VERIFICATION


def test_findings_referencing_nonexistent_rules_fail_validation() -> None:
    finding = Finding.model_validate(
        {
            "finding_id": "bad",
            "category": "test",
            "severity": "LOW",
            "status": "INFERRED",
            "rule_id": "missing-rule",
            "technical_title": "Bad",
            "technical_description": "Bad",
            "client_title": "Bad",
            "client_explanation": "Bad",
            "business_impact": "May be bad",
            "implementation": "Check",
            "priority": "P3",
            "evidence_ids": ["e1"],
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="missing-rule"):
        validate_findings_rules([finding], REGISTRY, as_of=datetime(2026, 8, 11).date())


def test_invalid_json_ld_generates_one_aggregated_evidence_backed_finding() -> None:
    pages = [
        Page(
            url=f"https://example.com/page-{index}",
            final_url=f"https://example.com/page-{index}",
            status_code=200,
            canonical=f"https://example.com/page-{index}",
            json_ld_errors=[
                {
                    "message": "Extra data",
                    "line": 1,
                    "column": 8,
                    "excerpt": '{"a":1}{"b":2}',
                }
            ],
            indexable=True,
        )
        for index in range(2)
    ]
    evidence = [
        Evidence(
            evidence_id=f"page-{index}",
            source_url=page.final_url,
            source_type="web_page",
            collector="native-crawler",
            observed_at=datetime.now(UTC),
            observed_value={
                "json_ld_errors": [item.model_dump(mode="json") for item in page.json_ld_errors]
            },
        )
        for index, page in enumerate(pages)
    ]

    findings = analyze_pages(pages, evidence, REGISTRY, as_of=datetime(2026, 8, 11).date())
    invalid = [item for item in findings if item.category == "structured_data_validity"]

    assert len(invalid) == 1
    assert invalid[0].affected_urls == [str(page.final_url) for page in pages]
    assert invalid[0].evidence_ids == ["page-0", "page-1"]
    assert invalid[0].factual_claims[0].numbers == ["2"]
    assert "Extra data" in invalid[0].technical_description
    assert "may" in invalid[0].business_impact.casefold()


def test_partial_json_ld_absence_generates_evidence_backed_finding() -> None:
    pages = [
        Page(
            url="https://example.com/with-json-ld",
            final_url="https://example.com/with-json-ld",
            status_code=200,
            json_ld=[{"@type": "Organization", "name": "Example"}],
            indexable=True,
        ),
        Page(
            url="https://example.com/without-json-ld",
            final_url="https://example.com/without-json-ld",
            status_code=200,
            indexable=True,
        ),
    ]
    evidence = [
        Evidence(
            evidence_id=f"page-{index}",
            source_url=page.final_url,
            source_type="web_page",
            collector="native-crawler",
            observed_at=datetime.now(UTC),
            observed_value={"json_ld": page.json_ld},
        )
        for index, page in enumerate(pages)
    ]

    findings = analyze_pages(pages, evidence, REGISTRY, as_of=datetime(2026, 8, 11).date())
    missing = next(item for item in findings if item.category == "structured_data_presence")

    assert missing.affected_urls == ["https://example.com/without-json-ld"]
    assert missing.evidence_ids == ["page-1"]
    assert missing.factual_claims[0].numbers == ["1", "2"]


def test_not_discovered_sitemap_wording_does_not_claim_confirmed_absence() -> None:
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        canonical="https://example.com/",
        indexable=True,
    )
    evidence = Evidence(
        evidence_id="robots-1",
        source_url="https://example.com/robots.txt",
        source_type="robots_txt",
        collector="native-crawler",
        observed_at=datetime.now(UTC),
        observed_value={"sitemaps": []},
    )
    finding = next(
        item
        for item in analyze_pages(
            [page],
            [evidence],
            REGISTRY,
            as_of=datetime(2026, 8, 11).date(),
            sitemap_state=SitemapState.NOT_DISCOVERED,
        )
        if item.category == "sitemap_discovery"
    )
    wording = " ".join(
        [finding.client_title, finding.client_explanation, finding.technical_description]
    ).casefold()
    assert "does not have a sitemap" not in wording
    assert "not discovered" in wording
