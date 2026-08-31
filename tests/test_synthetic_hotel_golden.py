import json
import subprocess
from pathlib import Path

import httpx

from ai_search_audit.adapters import PublicResearchItem
from ai_search_audit.orchestrator import run_public_audit
from tests.test_crawler import public_resolver

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic-hotel"


def test_synthetic_hotel_golden_audit_is_evidence_rich(tmp_path: Path) -> None:
    site = json.loads((FIXTURE / "site.json").read_text())
    pages = json.loads((FIXTURE / "pages.json").read_text())
    expected = json.loads((FIXTURE / "expected.json").read_text())
    external = [
        PublicResearchItem.model_validate(item)
        for item in json.loads((FIXTURE / "external-evidence.json").read_text())
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        item = pages.get(request.url.path)
        if item is None:
            return httpx.Response(404, request=request)
        return httpx.Response(
            item["status"],
            text=item["body"],
            headers={"content-type": item["content_type"]},
            request=request,
        )

    run = run_public_audit(
        site["domain"],
        output_dir=tmp_path,
        crawler_transport=httpx.MockTransport(handler),
        crawler_resolver=public_resolver,
        research_items=external,
    )
    assert run.site.languages == sorted(expected["required_languages"])
    assert {item.source_class for item in run.external_mentions} >= set(
        expected["required_source_classes"]
    )
    conflict = next(
        item
        for item in run.findings
        if item.category == "entity_consistency"
        and item.factual_claims[0].predicate == f"entity.{expected['required_conflict']}.conflict"
    )
    assert set(conflict.factual_claims[0].numbers) == set(expected["required_conflict_values"])
    assert len(conflict.evidence_ids) == 2
    assert conflict.rule_state and conflict.rule_state.value == "CURRENT"
    assert run.entity and run.entity.brand == "Example Lakeside Hotel"
    assert run.entity and run.entity.facts["positioning"] == ["boutique"]
    assert run.entity.facts["seasonality"] == ["seasonal demand"]
    assert run.entity.facts["differentiator"] == ["guided nature stays"]
    assert not any(page.json_ld for page in run.pages)
    assert all(page.json_ld_errors for page in run.pages)
    assert len(run.ai_prompts) >= expected["minimum_prompt_count"]
    assert all(prompt.pack_version == expected["prompt_pack_version"] for prompt in run.ai_prompts)
    scores = {score.name: score for score in run.scores}
    assert expected["forbidden_score"] not in scores
    assert scores["Technical Search Readiness"].total_checks >= expected["minimum_technical_checks"]
    assert scores["Content Citability"].total_checks >= expected["minimum_content_checks"]
    for score_name in expected["unavailable_scores"]:
        assert scores[score_name].state.value == "UNAVAILABLE"
        assert scores[score_name].value is None
    assert all(0 <= score.coverage <= 1 and 0 <= score.confidence <= 1 for score in run.scores)

    invalid_json_ld = [
        finding for finding in run.findings if finding.category == "structured_data_validity"
    ]
    assert len(invalid_json_ld) == 1
    invalid = invalid_json_ld[0]
    assert len(invalid.affected_urls) == expected["invalid_json_ld_affected_pages"]
    assert expected["invalid_json_ld_error"] in invalid.technical_description
    assert len(invalid.evidence_ids) == expected["invalid_json_ld_affected_pages"]
    structured_score = next(
        check
        for score in run.scores
        if score.name == "Entity & Machine Understanding"
        for check in score.checks
        if check.name == "structured data observed"
    )
    assert structured_score.score == 0
    assert structured_score.explanation_finding_ids == [invalid.finding_id]

    report_path = tmp_path / "client-report-data.json"
    report = json.loads(report_path.read_text())
    assert report["report_schema_version"] == expected["report_schema_version"]
    assert report["report_template_version"] == expected["report_template_version"]
    report_conflict = next(
        item for item in report["findings"] if item["finding_id"] == conflict.finding_id
    )
    assert report_conflict["factual_claims"] == [
        claim.model_dump(mode="json") for claim in conflict.factual_claims
    ]
    assert any(
        term in report_conflict["business_impact"].casefold()
        for term in expected["uncertainty_terms"]
    )
    assert all("coverage" in score and "confidence" in score for score in report["scores"])
    report_text = report_path.read_text()
    assert "add llms.txt" not in report_text.lower()
    backlog = (tmp_path / "implementation-backlog.csv").read_text()
    assert invalid.finding_id in backlog
    assert "Validate each affected JSON-LD block" in backlog

    pdf_path = tmp_path / "client-report.pdf"
    assert pdf_path.exists() and pdf_path.stat().st_size > 10_000
    pdf_text = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for section in expected["required_pdf_sections"]:
        assert section in pdf_text
    assert "Measurement" in pdf_text and "Maturity" in pdf_text
    assert "UNAVAILABLE" in pdf_text
