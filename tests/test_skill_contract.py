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


def test_skill_defines_the_local_versioned_project_lifecycle() -> None:
    text = (ROOT / "SKILL.md").read_text()
    normalized = " ".join(text.split())

    for invocation in (
        "$auditing-seo-geo-ai-search https://example.com",
        "$auditing-seo-geo-ai-search update project:example",
        "$auditing-seo-geo-ai-search enrich project:example",
        "$auditing-seo-geo-ai-search validate project:example",
    ):
        assert invocation in text

    for invariant in (
        "public-v1",
        "explicit domain and entity identity check",
        "next-audit-data-request_<LOCALE>.md",
        "owned temporary intake directory",
        "keep the coordinating CLI process open",
        "--intake-root",
        "never copy raw attachments into the client project",
        "delete the owned intake directory on success or failure",
        "reattach the source files and rerun",
        "90 days",
        "does not establish causation",
        "stable public checkout remains untouched",
    ):
        assert invariant.casefold() in normalized.casefold()


def test_skill_keeps_optional_first_party_data_inside_visibility_scope() -> None:
    text = (ROOT / "SKILL.md").read_text()
    normalized = " ".join(text.split())

    for required in (
        "free public audit",
        "optional deeper first-party visibility diagnostic",
        "GA4 Organic Search",
        "GA4 AI Assistant",
        "visits and sessions",
    ):
        assert required in normalized

    local_workflow = text.split("## Local versioned project workflow", maxsplit=1)[1].split(
        "\n## ", maxsplit=1
    )[0]
    assert "leads, revenue, CRM" in local_workflow
    assert "pricing or sales copy" in local_workflow


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


def test_skill_has_executable_client_edition_handoff_contract():
    skill = (ROOT / "SKILL.md").read_text()
    reference = (ROOT / "references/client-pdf-spec.md").read_text()
    assert "project finalize" in skill
    assert "not complete" in skill
    assert "--reviewed-pdf-sha256" in reference
    assert "--audit-id" in reference
    assert "--locale" in reference
    assert "reports/<audit-version>/edition-N/" in reference
    assert "--no-hero-reason" in reference


def test_release_version_is_updated_consistently():
    import tomllib

    from ai_search_audit import __version__

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert __version__ == project["project"]["version"] == "0.2.0"
