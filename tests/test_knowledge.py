from datetime import date
from pathlib import Path

import pytest

from ai_search_audit.knowledge import KnowledgeRegistry, RuleState, load_registry

ROOT = Path(__file__).parents[1]


def test_registry_loads_all_required_files() -> None:
    registry = load_registry(ROOT / "knowledge")
    assert isinstance(registry, KnowledgeRegistry)
    assert registry.version == "2026.09.04"
    assert registry.verified_date.isoformat() == "2026-09-04"
    assert len(registry.rules) >= 9


def test_rule_ids_are_unique_and_sources_are_primary() -> None:
    registry = load_registry(ROOT / "knowledge")
    ids = [rule.rule_id for rule in registry.rules]
    assert len(ids) == len(set(ids))
    assert all(str(rule.source_url).startswith("https://") for rule in registry.rules)
    assert all(rule.evidence_level in {"A", "B", "C", "D", "E"} for rule in registry.rules)


def test_emerging_conventions_are_experimental_and_non_scoring() -> None:
    registry = load_registry(ROOT / "knowledge")
    emerging = [rule for rule in registry.rules if "llms" in rule.rule_id]
    assert emerging
    assert all(rule.evidence_level == "E" and rule.scoring_weight == 0 for rule in emerging)


def test_invalid_rule_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "registry.yaml").write_text(
        "version: x\nverified_date: 2026-08-11\nfiles: [bad.yaml]\n", encoding="utf-8"
    )
    (tmp_path / "bad.yaml").write_text(
        "rules:\n  - rule_id: bad\n    vendor: x\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_registry(tmp_path)


def test_registry_resolves_current_rule_with_operational_metadata() -> None:
    registry = load_registry(ROOT / "knowledge")
    resolved = registry.resolve("google-noindex-001", as_of=date(2026, 8, 11))
    assert resolved.state is RuleState.CURRENT
    assert resolved.evidence_level == "A"
    assert resolved.confidence == 0.99
    assert resolved.scoring_weight == 1


def test_stale_rule_requires_verification() -> None:
    registry = load_registry(ROOT / "knowledge")
    resolved = registry.resolve("openai-oai-searchbot-001", as_of=date(2027, 8, 11))
    assert resolved.state is RuleState.REQUIRES_VERIFICATION
    assert resolved.expires_at < date(2027, 8, 11)


def test_registry_rejects_unknown_rule_id() -> None:
    registry = load_registry(ROOT / "knowledge")
    with pytest.raises(KeyError, match="missing-rule"):
        registry.resolve("missing-rule", as_of=date(2026, 8, 11))


@pytest.mark.parametrize(
    "rule_id,url",
    [
        (
            "content-render-parity-001",
            "https://developers.google.com/search/docs/appearance/ai-features",
        ),
        (
            "content-section-context-001",
            "https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents",
        ),
    ],
)
def test_content_diagnostic_rules_are_explicit_non_scoring_inferences(
    rule_id: str, url: str
) -> None:
    registry = load_registry(ROOT / "knowledge")
    rule = registry.resolve(rule_id, as_of=date(2026, 9, 3))
    assert str(rule.source_url) == url
    assert rule.source_type == "official_vendor_documentation_with_audit_inference"
    assert rule.scoring_weight == 0
    assert rule.verified_at == date(2026, 9, 2)
    assert rule.review_interval_days > 0
    assert 0 < rule.confidence <= 1
    assert rule.state is RuleState.CURRENT
    assert "audit" in rule.statement.lower()
    assert "ranking" in rule.statement.lower()
    assert (
        registry.resolve(rule_id, as_of=date(2028, 1, 1)).state is RuleState.REQUIRES_VERIFICATION
    )
    assert all(
        other.verified_at == date(2026, 8, 11)
        for other in registry.rules
        if other.rule_id
        not in {
            "content-render-parity-001",
            "content-section-context-001",
            "performance-psi-score-001",
            "performance-lcp-thresholds-001",
        }
    )


def test_openai_search_crawler_rule_uses_current_publishers_faq() -> None:
    registry = load_registry(ROOT / "knowledge")
    rule = registry.resolve("openai-oai-searchbot-001", as_of=date(2026, 8, 11))
    assert str(rule.source_url) == (
        "https://help.openai.com/en/articles/12627856-publishers-and-developers-faq"
    )
    assert "GPTBot" in rule.statement
    assert "training" in rule.statement


def test_registry_owns_active_crawler_product_metadata() -> None:
    registry = load_registry(ROOT / "knowledge")
    products = {product.token: product for product in registry.crawler_products}
    assert set(products) == {
        "*",
        "OAI-SearchBot",
        "GPTBot",
        "PerplexityBot",
        "Perplexity-User",
        "ClaudeBot",
        "Claude-User",
        "Claude-SearchBot",
        "Googlebot",
        "Bingbot",
    }
    assert products["OAI-SearchBot"].purpose == "search_citation"
    assert products["GPTBot"].purpose == "training"
    assert products["Perplexity-User"].purpose == "user_fetch"
    assert products["Perplexity-User"].robots_txt_respected is False
    assert products["Claude-User"].robots_txt_respected is True
    assert all(product.identifier and product.control_type for product in products.values())
    assert all(product.verified_at == date(2026, 8, 11) for product in products.values())
    for product in products.values():
        assert registry.resolve(product.source_rule, as_of=date(2026, 8, 11))


def test_registry_rejects_crawler_product_with_missing_source_rule(tmp_path: Path) -> None:
    (tmp_path / "registry.yaml").write_text(
        "version: x\nverified_date: 2026-08-11\nfiles: [products.yaml]\n",
        encoding="utf-8",
    )
    (tmp_path / "products.yaml").write_text(
        """crawler_products:
  - identifier: example-search
    token: ExampleBot
    vendor: example
    purpose: search_citation
    control_type: robots_txt
    robots_txt_respected: true
    source_rule: missing-rule
    verified_at: 2026-08-11
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing source rule"):
        load_registry(tmp_path)


def test_registry_rejects_duplicate_crawler_tokens(tmp_path: Path) -> None:
    (tmp_path / "registry.yaml").write_text(
        "version: x\nverified_date: 2026-08-11\nfiles: [products.yaml]\n",
        encoding="utf-8",
    )
    (tmp_path / "products.yaml").write_text(
        """rules:
  - rule_id: example-rule
    vendor: example
    statement: Example crawler rule.
    source_url: https://example.com/crawlers
    source_type: official_vendor_documentation
    evidence_level: A
    verified_at: 2026-08-11
    review_interval_days: 90
    confidence: 1
    scoring_weight: 1
crawler_products:
  - identifier: example-one
    token: ExampleBot
    vendor: example
    purpose: search_citation
    control_type: robots_txt
    robots_txt_respected: true
    source_rule: example-rule
    verified_at: 2026-08-11
  - identifier: example-two
    token: ExampleBot
    vendor: example
    purpose: training
    control_type: robots_txt
    robots_txt_respected: true
    source_rule: example-rule
    verified_at: 2026-08-11
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokens must be unique"):
        load_registry(tmp_path)
