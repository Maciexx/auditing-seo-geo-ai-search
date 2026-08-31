from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_skill_defines_complete_audit_workflow_contract() -> None:
    text = (ROOT / "SKILL.md").read_text()
    assert text.startswith("---\nname: auditing-seo-geo-ai-search\n")
    for section in (
        "## Core principle",
        "## Operating contract",
        "## Evidence model",
        "## Audit workflow",
        "## Measurement plan",
        "## Owner questions",
        "## Client-ready output contract",
        "## Quality gate",
    ):
        assert section in text
    assert "references/client-pdf-spec.md" in text
    assert "scripts/render_client_pdf.py" in text
    assert "visually verified client PDF" in text
    assert "Do not make CMS, analytics, Search Console" in text
    assert "without promising rankings, citations, recommendations, traffic, or revenue" in text


def test_skill_exposes_bundled_engine_without_confusing_it_with_client_report() -> None:
    text = (ROOT / "SKILL.md").read_text()
    for token in (
        "## Bundled audit engine",
        "ai-search-audit audit https://example.com",
        "audit.json",
        "client-report.pdf",
        "does not replace the broader client edition",
        "bounded public crawl",
    ):
        assert token in text


def test_agent_metadata_matches_skill() -> None:
    data = yaml.safe_load((ROOT / "agents" / "openai.yaml").read_text())
    interface = data["interface"]
    assert interface["display_name"] == "SEO, GEO & AI Search Audit"
    assert interface["short_description"] == "Evidence-led audits with client-ready PDF"
    assert "$auditing-seo-geo-ai-search" in interface["default_prompt"]


def test_geo_optimizer_attribution_is_present() -> None:
    attribution = (ROOT / "LICENSES" / "geo-optimizer-attribution.md").read_text()
    assert "Auriti-Labs/geo-optimizer-skill" in attribution
    assert "MIT" in attribution


def test_bundled_noto_sans_has_separate_ofl_license() -> None:
    license_text = (ROOT / "LICENSES" / "OFL-NotoSans.txt").read_text()
    font = (
        ROOT
        / "src"
        / "ai_search_audit"
        / "templates"
        / "client-report"
        / "v1"
        / "assets"
        / "fonts"
        / "NotoSans-Variable.ttf"
    )
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
    assert font.exists() and font.stat().st_size > 1_000_000
