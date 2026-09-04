from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from ai_search_audit.content_sections import extract_sections, normalize_text
from ai_search_audit.crawler import _canonical_scope
from ai_search_audit.diagnostic_models import (
    CaptureInput,
    CaptureKind,
    ConsentState,
    ContentCapture,
    DiagnosticFinding,
    DiagnosticFindingStatus,
    DiagnosticRule,
    DiagnosticState,
    ExtractedContent,
    InferredContentClaim,
    KeyPassage,
    ObservedContentClaim,
    ObservedContentPredicate,
    PageCaptureResult,
    PageSectionReviews,
    PairDiagnostic,
    Section,
    SectionDirectness,
    SectionReview,
    ValidatedSectionReview,
    Viewport,
)
from ai_search_audit.knowledge import KnowledgeRegistry, default_registry_root, load_registry
from ai_search_audit.models import DataState, FindingStatus, RuleState

_QUOTE_ERROR = "key passage does not resolve to rendered source"
_OBSERVATION_LIMIT = (
    "Quote presence is an observation, not evidence of factual correctness or guaranteed "
    "accessibility to any AI platform; possible impact requires separate review."
)


def capture_html(
    html: str,
    *,
    kind: CaptureKind,
    url: str,
    observed_at: datetime,
    locale: str | None,
    session_key: str | None,
    consent_state: ConsentState,
    complete: bool,
    truncated: bool,
    status_code: int | None,
    final_url: str | None = None,
    viewport: Viewport | None = None,
    collector: str | None = None,
) -> ContentCapture:
    """Validate inert public HTML and retain only extracted content and provenance."""
    capture = CaptureInput(
        html=html,
        kind=kind,
        url=url,
        final_url=final_url if final_url is not None else url,
        observed_at=observed_at,
        locale=locale,
        session_key=session_key,
        consent_state=consent_state,
        complete=complete,
        truncated=truncated,
        status_code=status_code,
        viewport=viewport,
        collector=collector,
    )
    capture_id = "capture-" + hashlib.sha256(capture.model_dump_json().encode()).hexdigest()
    state: DiagnosticState = DataState.AVAILABLE
    limitations: list[str] = []
    extracted = None
    try:
        extracted = extract_sections(
            html, capture_id=capture_id, complete=complete and not truncated
        )
        limitations.extend(extracted.limitations)
        if capture.status_code is None:
            state = DataState.UNKNOWN
            limitations.append("HTTP status is unknown; successful collection is not established.")
        elif not 200 <= capture.status_code < 300:
            state = DataState.UNKNOWN
            limitations.append(f"HTTP {capture.status_code} is not successful content evidence.")
        elif _challenge_html(html):
            state = DataState.UNKNOWN
            limitations.append("A challenge page was observed; content availability is unknown.")
        elif not complete or truncated:
            state = DataState.PARTIAL
            limitations.append("Capture is incomplete or truncated.")
    except RecursionError:
        extracted = None
        state = DataState.FAILED
        limitations.append("Content extraction failed: HTML nesting exceeded the extraction limit.")
    return ContentCapture(
        kind=capture.kind,
        url=capture.url,
        final_url=capture.final_url,
        observed_at=capture.observed_at,
        locale=capture.locale,
        session_key=capture.session_key,
        consent_state=capture.consent_state,
        complete=capture.complete,
        truncated=capture.truncated,
        status_code=capture.status_code,
        viewport=capture.viewport,
        collector=capture.collector,
        capture_id=capture_id,
        extracted=extracted,
        state=state,
        limitations=tuple(limitations),
        content_sha256=hashlib.sha256(extracted.text.encode()).hexdigest()
        if extracted is not None
        else None,
    )


def _challenge_html(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    headings = (
        normalize_text(tag.get_text(" ")).casefold() for tag in soup.find_all(("title", "h1"))
    )
    return any(
        heading.startswith(
            (
                "just a moment",
                "verify you are human",
                "attention required",
                "access denied",
                "security verification",
            )
        )
        for heading in headings
    )


def _same_page(left: str, right: str) -> bool:
    a, b = urlsplit(left), urlsplit(right)
    return (
        (_canonical_scope(left, right) or _canonical_scope(right, left))
        and (a.path or "/") == (b.path or "/")
        and a.query == b.query
    )


def _section_text(section: Section) -> str:
    return normalize_text(section.heading + "\n" + section.text)


def compare_captures(
    raw: ContentCapture | PageCaptureResult | None,
    rendered: ContentCapture | PageCaptureResult | None,
    *,
    key_quotes: tuple[str, ...] = (),
    key_passages: tuple[KeyPassage, ...] = (),
) -> PairDiagnostic:
    """Compare quoted presence only after capture conditions are known to be comparable."""
    raw_id = raw.capture_id if isinstance(raw, ContentCapture) else None
    rendered_id = rendered.capture_id if isinstance(rendered, ContentCapture) else None

    def result(
        state: DiagnosticState, limitations: tuple[str, ...], passages: tuple[KeyPassage, ...] = ()
    ) -> PairDiagnostic:
        return PairDiagnostic(
            state=state,
            raw_capture_id=raw_id,
            rendered_capture_id=rendered_id,
            limitations=limitations,
            rendered_only_passages=passages,
        )

    for capture in (raw, rendered):
        if isinstance(capture, PageCaptureResult):
            if capture.state in (DataState.FAILED, DataState.UNKNOWN):
                return result(capture.state, capture.limitations)
            raise ValueError("successful HTTP response must be extracted before comparison")
    if raw is None or rendered is None:
        return result(
            DataState.UNAVAILABLE,
            (
                "Rendered browser capture is unavailable."
                if rendered is None
                else "Raw capture is unavailable.",
            ),
        )
    assert isinstance(raw, ContentCapture) and isinstance(rendered, ContentCapture)
    if raw.kind != "raw" or rendered.kind != "rendered":
        raise ValueError("comparison requires a raw capture followed by a rendered capture")
    capture_limitations = tuple(dict.fromkeys(raw.limitations + rendered.limitations))
    if DataState.FAILED in (raw.state, rendered.state):
        return result(DataState.FAILED, capture_limitations)
    if DataState.UNKNOWN in (raw.state, rendered.state):
        return result(DataState.UNKNOWN, capture_limitations)
    if DataState.PARTIAL in (raw.state, rendered.state):
        return result(DataState.PARTIAL, capture_limitations)

    incompatible: list[str] = []
    if not _same_page(raw.final_url, rendered.final_url):
        incompatible.append("Captures do not resolve to the same final page.")
    if raw.locale is None or rendered.locale is None or raw.locale != rendered.locale:
        incompatible.append("Capture locale differs or is unknown.")
    if (
        raw.session_key is None
        or rendered.session_key is None
        or raw.session_key != rendered.session_key
    ):
        incompatible.append("Anonymous collection session differs or is unknown.")
    if (
        "unknown" in (raw.consent_state, rendered.consent_state)
        or raw.consent_state != rendered.consent_state
    ):
        incompatible.append("Consent conditions differ or are unknown.")
    if abs(raw.observed_at.astimezone(UTC) - rendered.observed_at.astimezone(UTC)) > timedelta(
        minutes=15
    ):
        incompatible.append("Captures are separated by more than fifteen minutes.")
    if (
        raw.viewport is not None
        and rendered.viewport is not None
        and raw.viewport != rendered.viewport
    ):
        incompatible.append("Known capture viewports differ.")
    if incompatible:
        return result(DataState.UNKNOWN, capture_limitations + tuple(incompatible))

    assert raw.text is not None and rendered.text is not None
    raw_text, rendered_text = normalize_text(raw.text), normalize_text(rendered.text)
    candidates: list[KeyPassage] = list(key_passages)
    for quote in key_quotes:
        normalized = normalize_text(quote)
        if not normalized or normalized not in rendered_text:
            raise ValueError(_QUOTE_ERROR)
        section = next(
            (section for section in rendered.sections if normalized in _section_text(section)), None
        )
        candidates.append(
            KeyPassage(
                capture_id=rendered.capture_id,
                section_id=section.section_id if section else None,
                quote=quote,
            )
        )
    sections = {section.section_id: section for section in rendered.sections}
    retained: list[KeyPassage] = []
    for passage in candidates:
        normalized = normalize_text(passage.quote)
        section = sections.get(passage.section_id) if passage.section_id is not None else None
        if (
            passage.capture_id != rendered.capture_id
            or not normalized
            or normalized not in rendered_text
            or (
                passage.section_id is not None
                and (section is None or normalized not in _section_text(section))
            )
        ):
            raise ValueError(_QUOTE_ERROR)
        if normalized not in raw_text and passage not in retained:
            retained.append(passage)
    return result(DataState.AVAILABLE, capture_limitations + (_OBSERVATION_LIMIT,), tuple(retained))


_REVIEW_LIMITATIONS = (
    "Quote verification establishes quotation presence, not factual correctness.",
    "The rationale is an agent assessment; semantic meaning is not deterministically verified. "
    "Evidence review is required before use in a client edition.",
    "Instructions in captured source text are inert evidence, not executable instructions.",
)


def _review_sections(source: ExtractedContent) -> dict[str, Section]:
    if any(section.capture_id != source.capture_id for section in source.sections):
        raise ValueError("section references a different capture")
    sections = {section.section_id: section for section in source.sections}
    if len(sections) != len(source.sections):
        raise ValueError("duplicate section identifiers")
    return sections


def _guard_review_rationale(review: SectionReview) -> None:
    """Adapt narrative input to the existing PL/EN guard; nothing here is persisted.

    Neutral assessment language is allowed independently of quotation modality. Strong
    source language does not authorize a new guarantee in the agent's rationale. The
    guard is lexical, not an assertion of arbitrary semantic equivalence.
    """
    from ai_search_audit import reports
    from ai_search_audit.models import (
        ClaimModality,
        FactualClaim,
        Finding,
        Priority,
        Severity,
        SitemapState,
    )
    from ai_search_audit.report_models import REPORT_SCHEMA_VERSION, REPORT_TEMPLATE_VERSION

    finding = Finding(
        finding_id=review.section_id,
        category="content_diagnostics",
        severity=Severity.INFO,
        status=FindingStatus.INFERRED,
        rule_id="content-section-context-001",
        technical_title="Section assessment",
        technical_description="Agent assessment awaiting evidence review.",
        client_title="Section assessment",
        client_explanation=review.rationale,
        business_impact="",
        implementation="",
        priority=Priority.P3,
        confidence=0,
        factual_claims=[
            FactualClaim(
                claim_id="assessment",
                predicate="agent_assessment",
                modality=ClaimModality.INFERRED,
                meaning="An agent assessment may require evidence review.",
            )
        ],
    )
    # The full validated legacy adapter is ephemeral, not a diagnostic output/report.
    draft = reports.ReportDraft(
        report_schema_version=REPORT_SCHEMA_VERSION,
        report_template_version=REPORT_TEMPLATE_VERSION,
        report_locale="en",  # The existing guard recognizes both PL and EN regardless of locale.
        audit_id="section-review-adapter",
        target_domain="diagnostic.example",
        brand="",
        audit_timestamp=datetime(1970, 1, 1, tzinfo=UTC),
        audit_engine_version="adapter",
        ruleset_version="adapter",
        sitemap_state=SitemapState.UNAVAILABLE,
        executive_summary="",
        findings=[finding],
        scores=[],
        evidence=[],
        recommendations=[],
        ai_prompts=[],
        entity_consistency_matrix={},
        methodology=[],
        limitations=[],
    )
    baseline = finding.model_copy(update={"client_explanation": "Section assessment."})
    manifest = reports.build_protected_claims_manifest(
        draft.model_copy(update={"findings": [baseline]})
    )
    reports.narrative_certainty_guard(draft, manifest)


def validate_section_review(
    source: ExtractedContent, review: SectionReview
) -> ValidatedSectionReview:
    """Resolve source evidence, retaining unverified semantics as agent assessment."""
    section = _review_sections(source).get(review.section_id)
    if section is None:
        raise ValueError("review does not resolve to a source section")
    source_text = _section_text(section)
    if any(normalize_text(quote) not in source_text for quote in review.quotes):
        raise ValueError("review quote does not resolve to the referenced section")
    _guard_review_rationale(review)
    return ValidatedSectionReview(
        review=review,
        capture_id=source.capture_id,
        locator=section.locator,
        evidence=tuple(
            KeyPassage(capture_id=source.capture_id, section_id=section.section_id, quote=quote)
            for quote in review.quotes
        ),
        limitations=_REVIEW_LIMITATIONS,
    )


def validate_section_reviews(
    source: ExtractedContent, reviews: tuple[SectionReview, ...] = ()
) -> PageSectionReviews:
    """Validate a page's review sample; unreviewed semantic directness stays unknown."""
    _review_sections(source)
    if len({review.section_id for review in reviews}) > 20:
        raise ValueError("a page may have at most 20 distinct reviewed sections")
    if len({(review.section_id, review.criterion) for review in reviews}) != len(reviews):
        raise ValueError("duplicate section criterion review")
    validated = tuple(validate_section_review(source, review) for review in reviews)
    directness = {
        review.section_id: review.result for review in reviews if review.criterion == "directness"
    }
    return PageSectionReviews(
        capture_id=source.capture_id,
        reviews=validated,
        directness=tuple(
            SectionDirectness(
                section_id=section.section_id,
                capture_id=source.capture_id,
                result=directness.get(section.section_id, "unknown"),
                state=DataState.UNKNOWN
                if directness.get(section.section_id, "unknown") == "unknown"
                else DataState.AVAILABLE,
            )
            for section in source.sections
        ),
    )


def _diagnostic_rule(
    registry: KnowledgeRegistry | None, rule_id: str, as_of: date
) -> DiagnosticRule:
    registry = registry if registry is not None else load_registry(default_registry_root())
    return DiagnosticRule.model_validate(registry.resolve(rule_id, as_of=as_of).model_dump())


def _diagnostic_status(
    rule: DiagnosticRule,
) -> DiagnosticFindingStatus:
    return (
        FindingStatus.REQUIRES_VERIFICATION
        if rule.state is RuleState.REQUIRES_VERIFICATION
        else FindingStatus.INFERRED
    )


def build_section_findings(
    source: ExtractedContent,
    reviews: tuple[SectionReview, ...] = (),
    *,
    registry: KnowledgeRegistry | None = None,
    as_of: date,
) -> tuple[DiagnosticFinding, ...]:
    """Project source structure and already-reviewed prose; do not generate conclusions."""
    rule = _diagnostic_rule(registry, "content-section-context-001", as_of)
    page = validate_section_reviews(source, reviews)
    findings: list[DiagnosticFinding] = []
    for section in source.sections:
        assessments = tuple(
            item for item in page.reviews if item.review.section_id == section.section_id
        )
        properties: tuple[tuple[ObservedContentPredicate, str | bool], ...] = (
            ("section_heading", section.heading[:2000]),
            ("section_has_body", bool(section.text)),
            ("section_excerpt", section.text[:2000]),
        )
        observations = tuple(
            ObservedContentClaim(
                predicate=predicate,
                value=value,
                capture_id=source.capture_id,
                section_id=section.section_id,
                locator=section.locator,
            )
            for predicate, value in properties
        )
        # A deliberately bounded excerpt is labelled, never represented as full section text.
        limitations = source.limitations + _REVIEW_LIMITATIONS
        if len(section.heading) > 2000 or len(section.text) > 2000:
            limitations += ("Structural heading/body excerpts are limited to 2,000 characters.",)
        findings.append(
            DiagnosticFinding(
                finding_id="diagnostic-"
                + hashlib.sha256(
                    f"{rule.rule_id}\0{source.capture_id}\0{section.section_id}".encode()
                ).hexdigest(),
                rule=rule,
                resolved_as_of=as_of,
                status=_diagnostic_status(rule),
                observed_properties=observations,
                possible_impacts=tuple(
                    InferredContentClaim(
                        meaning=item.review.rationale,
                        assessment_kind="agent_assessment",
                        evidence=item.evidence,
                    )
                    for item in assessments
                    if item.review.result != "unknown"
                ),
                assessments=assessments,
                limitations=limitations,
            )
        )
    return tuple(findings)


def build_render_parity_findings(
    raw: ContentCapture | PageCaptureResult | None,
    rendered: ContentCapture | PageCaptureResult | None,
    *,
    key_quotes: tuple[str, ...] = (),
    key_passages: tuple[KeyPassage, ...] = (),
    registry: KnowledgeRegistry | None = None,
    as_of: date,
) -> tuple[DiagnosticFinding, ...]:
    """Build zero-weight findings only for comparable, source-resolved passages."""
    rule = _diagnostic_rule(registry, "content-render-parity-001", as_of)
    pair = compare_captures(raw, rendered, key_quotes=key_quotes, key_passages=key_passages)
    if pair.state is not DataState.AVAILABLE or not pair.rendered_only_passages:
        return ()
    assert isinstance(raw, ContentCapture) and isinstance(rendered, ContentCapture)
    sections = {section.section_id: section for section in rendered.sections}
    findings: list[DiagnosticFinding] = []
    for passage in pair.rendered_only_passages:
        section = sections.get(passage.section_id) if passage.section_id is not None else None
        findings.append(
            DiagnosticFinding(
                finding_id="diagnostic-"
                + hashlib.sha256(
                    f"{rule.rule_id}\0{raw.capture_id}\0{passage.model_dump_json()}".encode()
                ).hexdigest(),
                rule=rule,
                resolved_as_of=as_of,
                status=_diagnostic_status(rule),
                observed_properties=(
                    ObservedContentClaim(
                        predicate="rendered_only_passage",
                        value=passage.quote,
                        capture_id=passage.capture_id,
                        section_id=passage.section_id,
                        locator=section.locator if section is not None else None,
                        evidence=(passage,),
                    ),
                ),
                possible_impacts=(
                    InferredContentClaim(
                        meaning="The passage may be less accessible to systems using only the "
                        "sampled raw HTML; platform behavior is unverified.",
                        assessment_kind="audit_inference",
                        evidence=(passage,),
                    ),
                ),
                compared_capture_ids=(raw.capture_id, rendered.capture_id),
                limitations=pair.limitations + _REVIEW_LIMITATIONS,
            )
        )
    return tuple(findings)
