from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    AIPrompt,
    AuditRun,
    DataState,
    Evidence,
    Finding,
    Recommendation,
    ScoreResult,
    SitemapState,
)
from .report_models import (
    REPORT_SCHEMA_VERSION,
    REPORT_TEMPLATE_VERSION,
    AISearchReportSection,
    AntiSlopReportMetadata,
    ClientReportData,
    EntityConsistencyReportSection,
    EntityFact,
    EntityFactValue,
    EvidenceAppendixItem,
    RendererMetadata,
    ReportFinding,
    ReportLocale,
    ReportRecommendation,
    ReportScore,
    validate_report_compatibility,
)
from .report_models import ReportCompatibilityError as ReportCompatibilityError
from .scoring import observed_ai_visibility, validate_score_explainability


class ClaimGuardError(ValueError):
    pass


class AntiSlopUnavailable(RuntimeError):
    pass


class ReportDraft(BaseModel):
    report_schema_version: str
    report_template_version: str
    report_locale: ReportLocale
    audit_id: str
    target_domain: str
    brand: str
    audit_timestamp: datetime
    audit_engine_version: str
    ruleset_version: str
    sitemap_state: SitemapState
    executive_summary: str
    findings: list[Finding]
    scores: list[ScoreResult]
    evidence: list[Evidence]
    recommendations: list[Recommendation]
    ai_prompts: list[AIPrompt]
    entity_consistency_matrix: dict[str, dict[str, list[str]]]
    methodology: list[str]
    limitations: list[str]


class NarrativeFindingRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: str = Field(frozen=True)
    client_title: str
    client_explanation: str
    business_impact: str
    implementation: str


class NarrativeRewritePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    executive_summary: str
    findings: list[NarrativeFindingRewrite]


class ProtectedClaimsManifest(BaseModel):
    manifest_version: str = "2.1.0"
    draft_hash: str
    audit_id: str
    target_domain: str
    findings: dict[str, dict[str, object]]
    evidence_ids: list[str]
    evidence_hashes: dict[str, str]
    scores: list[dict[str, object]]
    numeric_tokens: list[str] = Field(default_factory=list)
    narrative_certainty: dict[str, dict[str, object]] = Field(default_factory=dict)


class RewriteResult(BaseModel):
    draft: ReportDraft
    route: str
    agent_skill_state: DataState


RewriteProvider = Callable[[NarrativeRewritePayload], NarrativeRewritePayload]

_NARRATIVE_FIELDS = (
    "client_title",
    "client_explanation",
    "business_impact",
    "implementation",
)
_STRONG_CERTAINTY = re.compile(
    r"\b(?:will|definitely|certainly|guarantees?|guaranteed|always|never|must|proves?|"
    r"na pewno|zdecydowanie|zawsze|nigdy|musi|gwarantuje|udowadnia|spowoduje|zapewni)\b",
    re.IGNORECASE,
)
_CAUTIOUS_CERTAINTY = re.compile(
    r"\b(?:may|might|could|can|likely|perhaps|possibly|possible|potentially|"
    r"appears?|seems?|suggests?|może|mogą|mógłby|mogłaby|prawdopodobnie|"
    r"być może|potencjalnie|wydaje się|sugeruje)\b",
    re.IGNORECASE,
)


class _DraftCopy(TypedDict):
    summary: str
    methodology: tuple[str, ...]
    limitations: tuple[str, ...]


_DRAFT_COPY: dict[ReportLocale, _DraftCopy] = {
    "en": {
        "summary": (
            "The audit found {confirmed} confirmed and {inferred} inferred findings. "
            "Priorities are based on collected evidence and stated confidence."
        ),
        "methodology": (
            "Public evidence was collected within configured crawl and research limits.",
            "Readiness scores include state, assessed coverage, confidence, and finding links.",
            "Unavailable inputs are excluded rather than converted to zero.",
        ),
        "limitations": (
            "The audit covers public evidence and supplied public-research inputs only.",
            "Observed AI Visibility remains unavailable without grounded observations.",
            "Readiness findings do not guarantee search ranking, AI visibility, or citation.",
        ),
    },
    "pl": {
        "summary": (
            "Audyt wykazał {confirmed} potwierdzonych i {inferred} wnioskowanych ustaleń. "
            "Priorytety wynikają z zebranych dowodów i określonego poziomu pewności."
        ),
        "methodology": (
            "Dowody publiczne zebrano w skonfigurowanych granicach crawlu i researchu.",
            "Oceny gotowości obejmują stan, zakres oceny, pewność i odnośniki do ustaleń.",
            "Niedostępne dane są wyłączane z oceny, a nie zamieniane na zero.",
        ),
        "limitations": (
            "Audyt obejmuje wyłącznie dowody publiczne i przekazane dane public research.",
            "Widoczność w AI pozostaje niedostępna bez ugruntowanych obserwacji.",
            "Ustalenia gotowości nie gwarantują pozycji, widoczności AI ani cytowania.",
        ),
    },
}


def hash_draft(draft: ReportDraft) -> str:
    payload = draft.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_report_draft(run: AuditRun, *, report_locale: ReportLocale = "en") -> ReportDraft:
    confirmed = sum(1 for finding in run.findings if finding.status.value == "CONFIRMED")
    inferred = sum(1 for finding in run.findings if finding.status.value == "INFERRED")
    copy = _DRAFT_COPY[report_locale]
    summary = copy["summary"].format(confirmed=confirmed, inferred=inferred)
    return ReportDraft(
        report_schema_version=REPORT_SCHEMA_VERSION,
        report_template_version=REPORT_TEMPLATE_VERSION,
        report_locale=report_locale,
        audit_id=run.audit_id,
        target_domain=run.site.domain,
        brand=run.entity.brand if run.entity else run.site.brand or run.site.domain,
        audit_timestamp=run.timestamp,
        audit_engine_version=run.audit_engine_version,
        ruleset_version=run.ruleset_version,
        sitemap_state=run.sitemap_state,
        executive_summary=summary,
        findings=[finding.model_copy(deep=True) for finding in run.findings],
        scores=[score.model_copy(deep=True) for score in run.scores],
        evidence=[item.model_copy(deep=True) for item in run.evidence],
        recommendations=[item.model_copy(deep=True) for item in run.recommendations],
        ai_prompts=[item.model_copy(deep=True) for item in run.ai_prompts],
        entity_consistency_matrix={
            fact: {value: list(ids) for value, ids in values.items()}
            for fact, values in run.entity_consistency_matrix.items()
        },
        methodology=list(copy["methodology"]),
        limitations=list(copy["limitations"]),
    )


def _protected_finding(finding: Finding) -> dict[str, object]:
    return {
        "category": finding.category,
        "rule_id": finding.rule_id,
        "severity": finding.severity.value,
        "status": finding.status.value,
        "priority": finding.priority.value,
        "confidence": finding.confidence,
        "affected_urls": finding.affected_urls,
        "evidence_ids": finding.evidence_ids,
        "rule_evidence_level": finding.rule_evidence_level,
        "rule_confidence": finding.rule_confidence,
        "rule_scoring_weight": finding.rule_scoring_weight,
        "rule_state": finding.rule_state.value if finding.rule_state else None,
        "rule_expires_at": (
            finding.rule_expires_at.isoformat() if finding.rule_expires_at else None
        ),
        "technical_title": finding.technical_title,
        "technical_description": finding.technical_description,
        "factual_claims": [claim.model_dump(mode="json") for claim in finding.factual_claims],
        "numbers": sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", finding.model_dump_json()))),
    }


def _evidence_hash(item: Evidence) -> str:
    return hashlib.sha256(item.model_dump_json(exclude_none=False).encode()).hexdigest()


def _protected_scores(scores: list[ScoreResult]) -> list[dict[str, object]]:
    return [score.model_dump(mode="json") for score in scores]


def _certainty_rank(text: str) -> int:
    if _STRONG_CERTAINTY.search(text):
        return 2
    if _CAUTIOUS_CERTAINTY.search(text):
        return 0
    return 1


def _structured_certainty_ceiling(finding: Finding) -> int:
    if any(claim.modality.value in {"POSSIBLE", "INFERRED"} for claim in finding.factual_claims):
        return 0
    structured_text = " ".join(
        f"{claim.value or ''} {claim.meaning}" for claim in finding.factual_claims
    )
    return 2 if _certainty_rank(structured_text) == 2 else 1


def _narrative_certainty(finding: Finding) -> dict[str, object]:
    return {
        "structured_ceiling": _structured_certainty_ceiling(finding),
        "fields": {
            field: _certainty_rank(str(getattr(finding, field))) for field in _NARRATIVE_FIELDS
        },
    }


def build_protected_claims_manifest(draft: ReportDraft) -> ProtectedClaimsManifest:
    serialized = draft.model_dump_json(exclude_none=False)
    return ProtectedClaimsManifest(
        draft_hash=hash_draft(draft),
        audit_id=draft.audit_id,
        target_domain=draft.target_domain,
        findings={finding.finding_id: _protected_finding(finding) for finding in draft.findings},
        evidence_ids=sorted(item.evidence_id for item in draft.evidence),
        evidence_hashes={item.evidence_id: _evidence_hash(item) for item in draft.evidence},
        scores=_protected_scores(draft.scores),
        numeric_tokens=sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", serialized))),
        narrative_certainty={
            finding.finding_id: _narrative_certainty(finding) for finding in draft.findings
        },
    )


def claim_guard(draft: ReportDraft, manifest: ProtectedClaimsManifest) -> None:
    if draft.audit_id != manifest.audit_id or draft.target_domain != manifest.target_domain:
        raise ClaimGuardError("protected report identity changed after rewriting")
    current = {finding.finding_id: _protected_finding(finding) for finding in draft.findings}
    if current != manifest.findings:
        changed = sorted(set(current) | set(manifest.findings))
        raise ClaimGuardError(f"protected finding changed after rewriting: {', '.join(changed)}")
    evidence_ids = sorted(item.evidence_id for item in draft.evidence)
    if evidence_ids != manifest.evidence_ids:
        raise ClaimGuardError("protected evidence IDs changed after rewriting")
    evidence_hashes = {item.evidence_id: _evidence_hash(item) for item in draft.evidence}
    if evidence_hashes != manifest.evidence_hashes:
        raise ClaimGuardError("protected evidence content changed after rewriting")
    if _protected_scores(draft.scores) != manifest.scores:
        raise ClaimGuardError("protected score status or values changed after rewriting")
    current_numbers = sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", draft.model_dump_json())))
    if current_numbers != manifest.numeric_tokens:
        raise ClaimGuardError("protected numeric claims changed after rewriting")
    narrative_certainty_guard(draft, manifest)


def narrative_certainty_guard(draft: ReportDraft, manifest: ProtectedClaimsManifest) -> None:
    for finding in draft.findings:
        protected = manifest.narrative_certainty[finding.finding_id]
        ceiling_value = protected["structured_ceiling"]
        if not isinstance(ceiling_value, int):
            raise ClaimGuardError("invalid narrative-certainty manifest")
        ceiling = ceiling_value
        baselines = protected["fields"]
        if not isinstance(baselines, dict):
            raise ClaimGuardError("invalid narrative-certainty manifest")
        for field in _NARRATIVE_FIELDS:
            baseline_value = baselines[field]
            if not isinstance(baseline_value, int):
                raise ClaimGuardError("invalid narrative-certainty manifest")
            baseline = baseline_value
            if _certainty_rank(str(getattr(finding, field))) > max(baseline, ceiling):
                raise ClaimGuardError(
                    "narrative certainty strengthened after rewriting: "
                    f"{finding.finding_id}.{field}"
                )


def evidence_validator(draft: ReportDraft) -> None:
    available = {item.evidence_id for item in draft.evidence}
    for finding in draft.findings:
        missing = set(finding.evidence_ids) - available
        if missing:
            raise ClaimGuardError(
                f"finding {finding.finding_id} references missing evidence: {sorted(missing)}"
            )
        for claim in finding.factual_claims:
            missing_claim_evidence = set(claim.evidence_ids) - available
            if missing_claim_evidence:
                raise ClaimGuardError(
                    f"claim {claim.claim_id} references missing evidence: "
                    f"{sorted(missing_claim_evidence)}"
                )


def _narrative_payload(draft: ReportDraft) -> NarrativeRewritePayload:
    return NarrativeRewritePayload(
        executive_summary=draft.executive_summary,
        findings=[
            NarrativeFindingRewrite(
                finding_id=finding.finding_id,
                client_title=finding.client_title,
                client_explanation=finding.client_explanation,
                business_impact=finding.business_impact,
                implementation=finding.implementation,
            )
            for finding in draft.findings
        ],
    )


def _apply_narrative_payload(draft: ReportDraft, payload: NarrativeRewritePayload) -> ReportDraft:
    expected_ids = [finding.finding_id for finding in draft.findings]
    returned_ids = [finding.finding_id for finding in payload.findings]
    if returned_ids != expected_ids:
        raise ClaimGuardError("protected finding changed after rewriting: narrative item IDs")
    rewritten = draft.model_copy(deep=True)
    rewritten.executive_summary = payload.executive_summary
    for finding, narrative in zip(rewritten.findings, payload.findings, strict=True):
        for field in _NARRATIVE_FIELDS:
            setattr(finding, field, getattr(narrative, field))
    return rewritten


def _deterministic_anti_slop(payload: NarrativeRewritePayload) -> NarrativeRewritePayload:
    cleaned = payload.model_copy(deep=True)
    text = cleaned.executive_summary
    for phrase in ("Moreover, ", "Furthermore, ", "incredibly ", "robust ", "seamless "):
        text = text.replace(phrase, "")
    cleaned.executive_summary = " ".join(text.split())
    return cleaned


def apply_anti_slop(
    draft: ReportDraft,
    manifest: ProtectedClaimsManifest,
    *,
    provider: RewriteProvider | None = None,
) -> RewriteResult:
    if manifest.draft_hash != hash_draft(draft):
        raise ClaimGuardError("protected-claims manifest does not match the pre-rewrite draft")
    payload = _narrative_payload(draft)
    if provider is not None:
        try:
            rewritten_payload = provider(payload.model_copy(deep=True))
            if not isinstance(rewritten_payload, NarrativeRewritePayload):
                raise ClaimGuardError("Anti-Slop returned an invalid narrative payload")
            rewritten = _apply_narrative_payload(draft, rewritten_payload)
            claim_guard(rewritten, manifest)
            evidence_validator(rewritten)
            return RewriteResult(
                draft=rewritten,
                route="installed-anti-slop-skill",
                agent_skill_state=DataState.AVAILABLE,
            )
        except AntiSlopUnavailable:
            pass
    rewritten = _apply_narrative_payload(draft, _deterministic_anti_slop(payload))
    claim_guard(rewritten, manifest)
    evidence_validator(rewritten)
    return RewriteResult(
        draft=rewritten,
        route="deterministic-fallback",
        agent_skill_state=DataState.UNAVAILABLE,
    )


def report_data(draft: ReportDraft, rewrite: RewriteResult) -> dict[str, object]:
    return {
        "audit_id": draft.audit_id,
        "target_domain": draft.target_domain,
        "audience": "executive_with_selected_technical_detail",
        "executive_summary": rewrite.draft.executive_summary,
        "scores": [score.model_dump(mode="json") for score in rewrite.draft.scores],
        "findings": [finding.model_dump(mode="json") for finding in rewrite.draft.findings],
        "anti_slop": {
            "route": rewrite.route,
            "agent_skill_state": rewrite.agent_skill_state.value,
        },
    }


def _recommendations_from_findings(
    findings: list[Finding], canonical: list[Recommendation]
) -> list[Recommendation]:
    by_finding = {
        finding_id: recommendation
        for recommendation in canonical
        for finding_id in recommendation.finding_ids
    }
    recommendations: list[Recommendation] = []
    for finding in findings:
        existing = by_finding.get(finding.finding_id)
        recommendations.append(
            Recommendation(
                recommendation_id=(
                    existing.recommendation_id
                    if existing
                    else f"recommendation-{finding.finding_id}"
                ),
                finding_ids=[finding.finding_id],
                title=finding.client_title,
                implementation=finding.implementation,
                priority=finding.priority,
                confidence=min(
                    finding.confidence,
                    existing.confidence if existing else finding.confidence,
                ),
            )
        )
    return recommendations


def _evidence_appendix(evidence: list[Evidence]) -> list[EvidenceAppendixItem]:
    appendix: list[EvidenceAppendixItem] = []
    for item in evidence:
        if isinstance(item.observed_value, str):
            observation = " ".join(item.observed_value.split())
        else:
            observation = json.dumps(
                item.observed_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        publisher = item.metadata.get("publisher")
        appendix.append(
            EvidenceAppendixItem(
                evidence_id=item.evidence_id,
                source_type=item.source_type,
                source_scope=item.source_scope,
                source=publisher if isinstance(publisher, str) else item.collector,
                url=str(item.source_url) if item.source_url else None,
                observation=observation[:800],
                collected_at=item.observed_at,
                confidence=item.confidence,
            )
        )
    return sorted(appendix, key=lambda item: item.evidence_id)


def _frozen_entity_facts(
    matrix: dict[str, dict[str, list[str]]],
) -> tuple[EntityFact, ...]:
    return tuple(
        EntityFact(
            fact=fact,
            values=tuple(
                EntityFactValue(value=value, evidence_ids=tuple(evidence_ids))
                for value, evidence_ids in sorted(values.items())
            ),
        )
        for fact, values in sorted(matrix.items())
    )


def build_client_report_data(
    run: AuditRun,
    rewrite: RewriteResult,
    manifest: ProtectedClaimsManifest,
    *,
    renderer_metadata: RendererMetadata,
) -> ClientReportData:
    final_draft = rewrite.draft
    validate_report_compatibility(
        schema_version=final_draft.report_schema_version,
        template_version=final_draft.report_template_version,
        renderer=renderer_metadata,
    )
    claim_guard(final_draft, manifest)
    evidence_validator(final_draft)
    narrative_certainty_guard(final_draft, manifest)
    validate_score_explainability(
        final_draft.scores,
        final_draft.findings,
        final_draft.evidence,
        run.ai_observations,
    )

    prompt_versions = sorted({prompt.pack_version for prompt in final_draft.ai_prompts})
    ai_state = observed_ai_visibility(run.ai_observations).state
    recommendations = _recommendations_from_findings(
        final_draft.findings, final_draft.recommendations
    )
    return ClientReportData(
        report_schema_version=final_draft.report_schema_version,
        report_template_version=final_draft.report_template_version,
        report_locale=final_draft.report_locale,
        audit_id=final_draft.audit_id,
        target_domain=final_draft.target_domain,
        brand=final_draft.brand,
        audit_timestamp=final_draft.audit_timestamp,
        audit_engine_version=final_draft.audit_engine_version,
        ruleset_version=final_draft.ruleset_version,
        sitemap_state=final_draft.sitemap_state,
        audience="executive_with_selected_technical_detail",
        executive_summary=final_draft.executive_summary,
        scores=tuple(
            ReportScore.model_validate(score.model_dump(mode="python"))
            for score in final_draft.scores
        ),
        findings=tuple(
            ReportFinding.model_validate(finding.model_dump(mode="python"))
            for finding in final_draft.findings
        ),
        recommendations=tuple(
            ReportRecommendation.model_validate(item.model_dump(mode="python"))
            for item in recommendations
        ),
        ai_search=AISearchReportSection(
            access_finding_ids=tuple(
                finding.finding_id
                for finding in final_draft.findings
                if finding.category == "ai_search_access"
            ),
            prompt_count=len(final_draft.ai_prompts),
            prompt_pack_version=prompt_versions[0] if len(prompt_versions) == 1 else None,
            observed_visibility_state=ai_state,
        ),
        entity_consistency=EntityConsistencyReportSection(
            facts=_frozen_entity_facts(final_draft.entity_consistency_matrix),
            finding_ids=tuple(
                finding.finding_id
                for finding in final_draft.findings
                if finding.category == "entity_consistency"
            ),
        ),
        evidence_appendix=tuple(_evidence_appendix(final_draft.evidence)),
        methodology=tuple(final_draft.methodology),
        limitations=tuple(final_draft.limitations),
        anti_slop=AntiSlopReportMetadata(
            route=rewrite.route,
            agent_skill_state=rewrite.agent_skill_state,
        ),
        renderer=renderer_metadata,
    )
