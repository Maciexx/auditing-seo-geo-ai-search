from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from .models import RuleState

CrawlerPurpose = Literal[
    "generic", "search_citation", "user_fetch", "training", "traditional_search"
]
CrawlerControlType = Literal["robots_txt", "request_user_agent"]


class KnowledgeRule(BaseModel):
    rule_id: str
    vendor: str
    statement: str
    source_url: HttpUrl
    source_type: str
    evidence_level: Literal["A", "B", "C", "D", "E"]
    verified_at: date
    review_interval_days: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    scoring_weight: float = Field(default=1, ge=0)


class ResolvedKnowledgeRule(KnowledgeRule):
    state: RuleState
    expires_at: date


class CrawlerProduct(BaseModel):
    identifier: str
    token: str
    vendor: str
    purpose: CrawlerPurpose
    control_type: CrawlerControlType
    robots_txt_respected: bool
    source_rule: str
    verified_at: date


class KnowledgeRegistry(BaseModel):
    version: str
    verified_date: date
    rules: list[KnowledgeRule]
    crawler_products: list[CrawlerProduct]

    def resolve(self, rule_id: str, *, as_of: date) -> ResolvedKnowledgeRule:
        rule = next((item for item in self.rules if item.rule_id == rule_id), None)
        if rule is None:
            raise KeyError(f"knowledge rule does not exist: {rule_id}")
        expires_at = rule.verified_at + timedelta(days=rule.review_interval_days)
        state = RuleState.REQUIRES_VERIFICATION if as_of > expires_at else RuleState.CURRENT
        return ResolvedKnowledgeRule(
            **rule.model_dump(),
            state=state,
            expires_at=expires_at,
        )


def load_registry(root: Path) -> KnowledgeRegistry:
    try:
        metadata = yaml.safe_load((root / "registry.yaml").read_text(encoding="utf-8"))
        rules: list[KnowledgeRule] = []
        crawler_products: list[CrawlerProduct] = []
        for relative in metadata["files"]:
            data = yaml.safe_load((root / relative).read_text(encoding="utf-8")) or {}
            rules.extend(KnowledgeRule.model_validate(item) for item in data.get("rules", []))
            crawler_products.extend(
                CrawlerProduct.model_validate(item) for item in data.get("crawler_products", [])
            )
        ids = [rule.rule_id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("knowledge rule IDs must be unique")
        product_ids = [product.identifier for product in crawler_products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("crawler product identifiers must be unique")
        product_tokens = [product.token.casefold() for product in crawler_products]
        if len(product_tokens) != len(set(product_tokens)):
            raise ValueError("crawler product tokens must be unique")
        missing_rules = sorted(
            {product.source_rule for product in crawler_products if product.source_rule not in ids}
        )
        if missing_rules:
            raise ValueError(f"crawler product has missing source rule: {', '.join(missing_rules)}")
        return KnowledgeRegistry(
            version=str(metadata["version"]),
            verified_date=metadata["verified_date"],
            rules=rules,
            crawler_products=crawler_products,
        )
    except (KeyError, OSError, TypeError, ValidationError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid Knowledge Registry: {exc}") from exc
