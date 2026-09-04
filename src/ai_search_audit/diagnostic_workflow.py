from __future__ import annotations

import hashlib
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from ai_search_audit.benchmark import (
    canonical_hash,
    compare_benchmarks,
    prepare_benchmark_worksheet,
    validate_benchmark_responses,
    validate_benchmark_worksheet,
)
from ai_search_audit.config import AuditConfig
from ai_search_audit.content_diagnostics import (
    _diagnostic_rule,
    _diagnostic_status,
    _guard_review_rationale,
    _same_page,
    build_render_parity_findings,
    build_section_findings,
    capture_html,
    compare_captures,
    validate_section_reviews,
)
from ai_search_audit.content_sections import normalize_text
from ai_search_audit.crawler import NativeCrawler, Resolver, _canonical_host_variant
from ai_search_audit.data_intake import consume_owned_payload
from ai_search_audit.diagnostic_intake import read_diagnostic_payload
from ai_search_audit.diagnostic_models import (
    BenchmarkComparison,
    BenchmarkSample,
    BenchmarkWorksheet,
    CaptureInput,
    CaptureMetadata,
    ContentCapture,
    DiagnosticAttemptEvidence,
    DiagnosticBaseline,
    DiagnosticBinding,
    DiagnosticCaptureEvidence,
    DiagnosticCleanup,
    DiagnosticCollectionRange,
    DiagnosticContract,
    DiagnosticFinding,
    DiagnosticIntake,
    DiagnosticModuleStates,
    DiagnosticRun,
    DiagnosticRunReference,
    DiagnosticSectionEvidence,
    DiagnosticSelectedPage,
    DiagnosticSource,
    DiagnosticState,
    DiagnosticTextEvidence,
    InferredContentClaim,
    KeyPassage,
    ObservedContentClaim,
    PageCaptureResult,
    PageSectionReviews,
    PairDiagnostic,
    ValidatedSectionReview,
)
from ai_search_audit.diagnostic_performance import (
    DiagnosticRunV2,
    ProviderRunCleanup,
    ordered_collections,
    performance_algorithm_hash,
    performance_collection_range,
    reject_binary_values,
    validate_performance_source,
)
from ai_search_audit.diagnostic_sources import (
    _diagnostic_validation_operation,
    load_diagnostic_source,
)
from ai_search_audit.knowledge import default_registry_root, load_registry
from ai_search_audit.measurement_policy import performance_pipeline_version
from ai_search_audit.measurement_profile import MeasurementPreflight, _reject_unserialized_fields
from ai_search_audit.models import DataState
from ai_search_audit.performance_providers import PerformanceCollection
from ai_search_audit.project_store import ProjectStore
from ai_search_audit.report_models import SupplementaryDiagnosticComparison


def assemble_performance_run(
    source: DiagnosticSource,
    preflight: MeasurementPreflight,
    collections: tuple[PerformanceCollection, ...],
) -> DiagnosticRunV2:
    """Pure assembly of normalized evidence; no intake deletion, network or publication."""
    for value in (source, preflight, collections):
        _reject_unserialized_fields(value)
        reject_binary_values(value)
    preflight = MeasurementPreflight.model_validate(preflight.model_dump(mode="python"))
    collections = tuple(
        PerformanceCollection.model_validate(item.model_dump(mode="python", serialize_as_any=True))
        for item in collections
    )
    run = DiagnosticRunV2(
        binding=source.binding,
        pipeline_version=performance_pipeline_version(preflight.profile.schema_version),
        input_sha256=canonical_hash(preflight.model_dump(mode="json")),
        algorithm_sha256=performance_algorithm_hash(
            policy_version=preflight.profile.schema_version
        ),
        collection_range=performance_collection_range(collections),
        preflight=preflight,
        collections=ordered_collections(preflight, collections),
        cleanup=ProviderRunCleanup(),
    )
    validate_performance_source(run, source)
    return run


def parse_diagnostic_run_reference(value: str) -> DiagnosticRunReference:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError("diagnostic run reference must be source-version/run-N")
    return DiagnosticRunReference(source_version=parts[0], run_id=parts[1])


@_diagnostic_validation_operation
def compare_diagnostic_runs(
    project_ref: str,
    *,
    clients_root: str | Path,
    baseline_reference: DiagnosticRunReference,
    follow_up_reference: DiagnosticRunReference,
) -> SupplementaryDiagnosticComparison:
    """Reload and recompute solely from real validated samples and explicit source versions."""
    from ai_search_audit.diagnostic_store import DiagnosticStore

    clients_root = _real_directory_path(clients_root)
    project_root = _project_directory(ProjectStore(clients_root).resolve(project_ref))
    store = DiagnosticStore(project_root)
    before = store.load(baseline_reference.source_version, baseline_reference.run_id)
    after = store.load(follow_up_reference.source_version, follow_up_reference.run_id)
    if not isinstance(before.run, DiagnosticRun) or not isinstance(after.run, DiagnosticRun):
        raise ValueError("benchmark comparison requires benchmark diagnostic runs")
    baseline_source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=baseline_reference.source_version
    )
    follow_up_source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=follow_up_reference.source_version
    )
    # An empty, validated sample preserves UNAVAILABLE rates; no observation is invented.
    baseline = before.run.benchmark or validate_benchmark_responses(
        baseline_source, before.run.worksheet, ()
    )
    follow_up = after.run.benchmark or validate_benchmark_responses(
        follow_up_source, after.run.worksheet, ()
    )
    return SupplementaryDiagnosticComparison(
        baseline_reference=baseline_reference,
        follow_up_reference=follow_up_reference,
        baseline_manifest_sha256=before.manifest_sha256,
        follow_up_manifest_sha256=after.manifest_sha256,
        baseline_binding=before.run.binding,
        follow_up_binding=after.run.binding,
        baseline_collection_range=before.run.collection_range,
        follow_up_collection_range=after.run.collection_range,
        comparison=compare_benchmarks(
            baseline, follow_up, baseline_source=baseline_source, follow_up_source=follow_up_source
        ),
    )


def _real_directory_path(path: str | Path) -> Path:
    """Check the caller's original spelling before any Path.resolve can hide links."""
    if ".." in Path(path).parts:
        raise ValueError("parent traversal is not allowed in diagnostic directory paths")
    absolute = Path(os.path.abspath(path))
    for component in (*reversed(absolute.parents), absolute):
        mode = component.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"directory component is not a real directory: {component}")
    return absolute


def _project_directory(path: str | Path) -> Path:
    root = _real_directory_path(path)
    if not stat.S_ISREG((root / "project.json").lstat().st_mode):
        raise ValueError("canonical project manifest must be a regular file")
    return root


def _audited_urls(source: DiagnosticSource) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            url
            for url in source.page_urls
            if _canonical_host_variant(f"https://{source.binding.domain}/", url)
        )
    )


def _selected_pages(
    source: DiagnosticSource, selected_pages: tuple[DiagnosticSelectedPage, ...]
) -> tuple[DiagnosticSelectedPage, ...]:
    urls = _audited_urls(source)
    if selected_pages:
        selected_pages = tuple(
            DiagnosticSelectedPage.model_validate(page.model_dump()) for page in selected_pages
        )
        if len(selected_pages) > 10 or len({page.url for page in selected_pages}) != len(
            selected_pages
        ):
            raise ValueError("selected page sample must be unique and contain at most ten pages")
        if any(page.url not in urls for page in selected_pages):
            raise ValueError("selected page must be a validated audited site URL")
        return selected_pages
    home = next(
        (url for url in urls if urlsplit(url).path in {"", "/"} and not urlsplit(url).query), None
    )
    ordered = ((home,) if home else ()) + tuple(url for url in urls if url != home)
    return tuple(
        DiagnosticSelectedPage(
            url=url,
            reason="Canonical homepage" if url == home else "Audited page in canonical audit order",
        )
        for url in ordered[:5]
    )


def prepare_diagnostic_contract(
    project_ref: str,
    *,
    clients_root: str | Path,
    source_version: str,
    selected_pages: tuple[DiagnosticSelectedPage, ...] = (),
) -> DiagnosticContract:
    """Prepare exact source-bound inputs, without changing the canonical project."""
    clients = _real_directory_path(clients_root)
    _project_directory(ProjectStore(clients).resolve(project_ref))
    source = load_diagnostic_source(
        project_ref, clients_root=clients, source_version=source_version
    )
    pages = _selected_pages(source, selected_pages)
    session_key = _session_key(source.binding)
    instructions = (
        (
            "Zbierz aktualne strony w nowej anonimowej sesji bez logowania i bez zgody cookies. "
            "Zapisz rzeczywisty język strony, czas, viewport i warunki zgody."
            if source.binding.report_locale == "pl"
            else "Collect fresh pages in an anonymous browser without login or cookie consent. "
            "Record actual page language, time, viewport and consent conditions."
        ),
        (
            "session_key oznacza zadeklarowane warunki nowego anonimowego zbierania, a nie wspólne "
            "cookies, unikalne wykonanie lub dowód wspólnej sesji przeglądarki. "
            "Użyj etykiety tylko "
            "gdy rzeczywiste zbieranie spełnia te warunki. Maksymalny odstęp pary: 15 minut."
            if source.binding.report_locale == "pl"
            else "session_key labels declared fresh anonymous collection conditions, "
            "not shared cookies, "
            "a unique execution or proof of a shared browser session. Use this label only when the "
            "actual browser collection followed those conditions. "
            "Maximum pairing interval: 15 minutes."
        ),
    )
    return DiagnosticContract(
        source=source,
        worksheet=prepare_benchmark_worksheet(source),
        selected_pages=pages,
        session_key=session_key,
        instructions=instructions,
    )


def _session_key(binding: DiagnosticBinding) -> str:
    return (
        "anonymous-"
        + canonical_hash(
            {
                "binding": binding.model_dump(mode="json"),
                "anonymous_conditions_policy": "1.0.0",
            }
        )[:24]
    )


def _text_evidence(text: str) -> DiagnosticTextEvidence:
    excerpt = text[:2000]
    return DiagnosticTextEvidence(
        excerpt=excerpt,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
        source_length=len(text),
        truncated=len(text) > 2000,
    )


def _attempt_evidence(result: PageCaptureResult) -> DiagnosticAttemptEvidence:
    return DiagnosticAttemptEvidence(
        **result.model_dump(exclude={"body", "html"}),
        body_sha256=hashlib.sha256(result.body).hexdigest() if result.body is not None else None,
    )


def _capture_evidence(
    url: str,
    capture: ContentCapture | PageCaptureResult,
    *,
    attempt: PageCaptureResult | None = None,
    passages: tuple[KeyPassage, ...] = (),
) -> DiagnosticCaptureEvidence:
    original = _attempt_evidence(attempt) if attempt is not None else None
    if isinstance(capture, PageCaptureResult):
        assert original is not None
        return DiagnosticCaptureEvidence(
            evidence_id="attempt-" + canonical_hash(original.model_dump(mode="json")),
            capture_id=None,
            kind="raw",
            url=url,
            observed_at=capture.observed_at,
            metadata=None,
            attempt=original,
            state=capture.state,
            content_sha256=None,
            text=None,
            sections=(),
            passages=(),
            limitations=capture.limitations,
        )
    return DiagnosticCaptureEvidence(
        evidence_id=capture.capture_id,
        capture_id=capture.capture_id,
        kind=capture.kind,
        url=url,
        observed_at=capture.observed_at,
        metadata=CaptureMetadata.model_validate(
            capture.model_dump(include=set(CaptureMetadata.model_fields))
        ),
        attempt=original,
        state=capture.state,
        content_sha256=capture.content_sha256,
        text=_text_evidence(capture.text) if capture.text is not None else None,
        sections=tuple(
            DiagnosticSectionEvidence(
                section_id=section.section_id,
                capture_id=section.capture_id,
                locator=section.locator,
                level=section.level,
                heading=_text_evidence(section.heading),
                body=_text_evidence(section.text),
                heading_path=tuple(_text_evidence(part) for part in section.heading_path),
                block_kinds=section.block_kinds,
            )
            for section in capture.sections
        ),
        passages=tuple(sorted(set(passages), key=lambda p: (p.section_id or "", p.quote))),
        limitations=capture.limitations,
    )


def _import_rendered(capture: CaptureInput) -> ContentCapture:
    return capture_html(
        capture.html,
        kind="rendered",
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
    )


def _raw_capture(result: PageCaptureResult, session_key: str) -> ContentCapture | PageCaptureResult:
    # In particular, HTTP 200 challenge headers must not become successful evidence.
    if result.state is not DataState.AVAILABLE:
        return result
    assert result.html is not None and result.url is not None
    document = BeautifulSoup(result.html, "html.parser")
    html = document.find("html")
    language = html.get("lang") if html is not None else None
    locale = language.strip() if isinstance(language, str) and language.strip() else None
    if locale is not None and len(locale) > 64:
        locale = None  # Unsupported metadata remains unknown, never truncated to a known locale.
    return capture_html(
        result.html,
        kind="raw",
        url=result.url,
        final_url=result.final_url,
        observed_at=result.observed_at,
        locale=locale,
        session_key=session_key,
        consent_state="none",
        complete=result.complete,
        truncated=result.truncated,
        status_code=result.status_code,
        viewport=None,
        collector=result.collector,
    )


def _validate_passage(source: ContentCapture, passage: KeyPassage) -> None:
    quote = normalize_text(passage.quote)
    if (
        source.capture_id != passage.capture_id
        or source.text is None
        or quote not in normalize_text(source.text)
    ):
        raise ValueError("key passage does not resolve to rendered source")
    if passage.section_id is not None:
        section = next((s for s in source.sections if s.section_id == passage.section_id), None)
        if section is None or quote not in normalize_text(section.heading + "\n" + section.text):
            raise ValueError("key passage does not resolve to rendered section")


def _aggregate(states: tuple[DiagnosticState, ...]) -> DiagnosticState:
    if not states:
        return DataState.UNAVAILABLE
    for state in (DataState.FAILED, DataState.UNKNOWN, DataState.PARTIAL):
        if state in states:
            return state
    if all(state is DataState.UNAVAILABLE for state in states):
        return DataState.UNAVAILABLE
    return DataState.PARTIAL if DataState.UNAVAILABLE in states else DataState.AVAILABLE


def _module_states(
    captures: tuple[DiagnosticCaptureEvidence, ...],
    pairs: tuple[PairDiagnostic, ...],
    reviews: tuple[PageSectionReviews, ...],
    benchmark_state: DiagnosticState,
) -> DiagnosticModuleStates:
    return DiagnosticModuleStates(
        raw_capture=_aggregate(tuple(c.state for c in captures if c.kind == "raw")),
        render_parity=_aggregate(tuple(pair.state for pair in pairs)),
        section_reviews=_aggregate(
            tuple(item.state for group in reviews for item in group.directness)
        ),
        benchmark=benchmark_state,
    )


def _collection_range(
    captures: tuple[DiagnosticCaptureEvidence, ...],
    responses: tuple[datetime, ...],
) -> DiagnosticCollectionRange:
    timestamps = tuple(c.observed_at.astimezone(UTC) for c in captures) + responses
    return DiagnosticCollectionRange(
        start=min(timestamps) if timestamps else None,
        end=max(timestamps) if timestamps else None,
        reason=None if timestamps else "No collection observations are available.",
    )


def _algorithm_hash(findings: tuple[DiagnosticFinding, ...]) -> str:
    rules = {f.rule.rule_id: f.rule.model_dump(mode="json") for f in findings}
    return canonical_hash(
        {
            "pipeline": "1.0.0",
            "extraction": "1.0.0",
            "benchmark": "1.0.0",
            "crawler": NativeCrawler.version,
            "rules": rules,
        }
    )


def _expected_pair_state(
    raw: DiagnosticCaptureEvidence,
    rendered: DiagnosticCaptureEvidence | None,
) -> DiagnosticState:
    """Task 3's compatibility gate over persisted metadata, without inventing full text."""
    if raw.attempt is not None and raw.attempt.state is not DataState.AVAILABLE:
        return raw.attempt.state
    if rendered is None:
        return DataState.UNAVAILABLE
    for state in (DataState.FAILED, DataState.UNKNOWN, DataState.PARTIAL):
        if state in (raw.state, rendered.state):
            return state
    left, right = raw.metadata, rendered.metadata
    if left is None or right is None:
        raise ValueError("pair requires observed capture metadata")
    if (
        not _same_page(left.final_url, right.final_url)
        or left.locale is None
        or right.locale is None
        or left.locale != right.locale
        or left.session_key is None
        or right.session_key is None
        or left.session_key != right.session_key
        or "unknown" in (left.consent_state, right.consent_state)
        or left.consent_state != right.consent_state
        or abs(left.observed_at.astimezone(UTC) - right.observed_at.astimezone(UTC))
        > timedelta(minutes=15)
        or (
            left.viewport is not None
            and right.viewport is not None
            and left.viewport != right.viewport
        )
    ):
        return DataState.UNKNOWN
    return DataState.AVAILABLE


def _validate_run_integrity(run: DiagnosticRun) -> None:
    """Revalidate persisted internal references and derived values, not provider metrics."""
    if run.worksheet.binding != run.binding:
        raise ValueError("run worksheet binding mismatch")
    if run.session_key != _session_key(run.binding):
        raise ValueError("run anonymous collection conditions label mismatch")
    if run.benchmark is not None and run.benchmark.worksheet != run.worksheet:
        raise ValueError("benchmark worksheet mismatch")
    if run.algorithm_sha256 != _algorithm_hash(run.findings):
        raise ValueError("algorithm fingerprint mismatch")
    if run.findings:
        registry = load_registry(default_registry_root())
        for finding in run.findings:
            try:
                rule = _diagnostic_rule(registry, finding.rule.rule_id, finding.resolved_as_of)
            except KeyError as exc:
                raise ValueError(
                    "diagnostic finding rule is not present in the trusted registry"
                ) from exc
            if finding.rule != rule:
                raise ValueError(
                    "diagnostic finding rule snapshot does not match its recorded date"
                )
            if finding.status != _diagnostic_status(rule):
                raise ValueError("diagnostic finding status contradicts its trusted rule state")
    if run.collection_range != _collection_range(
        run.captures, tuple(r.observed_at for r in run.benchmark.responses) if run.benchmark else ()
    ):
        raise ValueError("collection range does not match actual observations")
    if run.module_states != _module_states(
        run.captures,
        run.pairs,
        run.section_reviews,
        run.benchmark.metrics.state if run.benchmark else DataState.UNAVAILABLE,
    ):
        raise ValueError("module states do not match observations")
    urls = tuple(page.url for page in run.selected_pages)
    if len(set(urls)) != len(urls):
        raise ValueError("duplicate selected pages")
    raw = tuple(c for c in run.captures if c.kind == "raw")
    rendered = tuple(c for c in run.captures if c.kind == "rendered")
    if any(
        c.metadata is not None
        and (
            c.metadata.session_key != run.session_key
            or c.metadata.consent_state != "none"
            or c.metadata.viewport is not None
        )
        for c in raw
    ):
        raise ValueError("raw capture must retain actual fresh anonymous collection conditions")
    if tuple(c.url for c in raw) != urls or len(run.pairs) != len(urls):
        raise ValueError("raw collection and pair inventory must match selected pages")
    if len({c.url for c in rendered}) != len(rendered) or any(c.url not in urls for c in rendered):
        raise ValueError("rendered evidence must belong to selected pages")
    if len({c.evidence_id for c in run.captures}) != len(run.captures):
        raise ValueError("duplicate capture evidence IDs")
    captures = {c.capture_id: c for c in run.captures if c.capture_id is not None}
    for raw_evidence, pair in zip(raw, run.pairs, strict=True):
        browser = next((c for c in rendered if c.url == raw_evidence.url), None)
        if pair.state != _expected_pair_state(raw_evidence, browser):
            raise ValueError("pair state contradicts actual collection conditions")
        if pair.raw_capture_id != raw_evidence.capture_id or pair.rendered_capture_id != (
            browser.capture_id if browser is not None else None
        ):
            raise ValueError("pair references a different page or capture")
        for passage in pair.rendered_only_passages:
            if browser is None or passage not in browser.passages:
                raise ValueError("pair passage is absent from evidence")
    expected_review_ids = {capture.capture_id for capture in rendered if capture.text is not None}
    review_ids = tuple(group.capture_id for group in run.section_reviews)
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != expected_review_ids:
        raise ValueError("review group inventory must match every extracted rendered capture")
    for group in run.section_reviews:
        capture = captures.get(group.capture_id)
        if capture is None or capture.kind != "rendered":
            raise ValueError("review references a different capture")
        sections = {s.section_id: s for s in capture.sections}
        if tuple(d.section_id for d in group.directness) != tuple(sections):
            raise ValueError("review directness inventory mismatch")
        for item in group.reviews:
            _guard_review_rationale(item.review)
            section = sections.get(item.review.section_id)
            if (
                section is None
                or item.capture_id != group.capture_id
                or item.locator != section.locator
            ):
                raise ValueError("review section reference mismatch")
            if any(p not in capture.passages for p in item.evidence):
                raise ValueError("review quotes are absent from normalized evidence")
            if item.evidence != tuple(
                KeyPassage(
                    capture_id=group.capture_id,
                    section_id=item.review.section_id,
                    quote=quote,
                )
                for quote in item.review.quotes
            ):
                raise ValueError("review quotes contradict retained validated evidence")
        directness = {
            item.review.section_id: item.review.result
            for item in group.reviews
            if item.review.criterion == "directness"
        }
        for directness_item in group.directness:
            expected_result = directness.get(directness_item.section_id, "unknown")
            expected_state = (
                DataState.UNKNOWN if expected_result == "unknown" else DataState.AVAILABLE
            )
            if (
                directness_item.result != expected_result
                or directness_item.state != expected_state
                or directness_item.capture_id != group.capture_id
            ):
                raise ValueError("review directness is not derived from actual assessments")
    _validate_finding_projections(run, rendered)
    if (run.comparison is None) != (run.baseline is None):
        raise ValueError("comparison requires an explicit baseline")
    if run.baseline is not None and (
        run.baseline.binding.project_id != run.binding.project_id
        or run.baseline.binding.report_locale != run.binding.report_locale
        or run.baseline.reference.source_version != run.baseline.binding.source_version
    ):
        raise ValueError("baseline source binding mismatch")


def _validate_finding_projections(
    run: DiagnosticRun, rendered: tuple[DiagnosticCaptureEvidence, ...]
) -> None:
    """Verify Task 4's exact retained-evidence projections, without rebuilding full text."""
    actual = {finding.finding_id: finding for finding in run.findings}
    if len(actual) != len(run.findings):
        raise ValueError("diagnostic finding inventory contains duplicate identifiers")
    expected_ids: set[str] = set()

    def check(
        finding_id: str,
        rule_id: str,
        observations: tuple[ObservedContentClaim, ...],
        impacts: tuple[InferredContentClaim, ...],
        assessments: tuple[ValidatedSectionReview, ...] = (),
        compared_ids: tuple[str, ...] = (),
    ) -> None:
        if finding_id in expected_ids:
            raise ValueError("diagnostic finding inventory contains duplicate source projections")
        expected_ids.add(finding_id)
        finding = actual.get(finding_id)
        if finding is None:
            raise ValueError("diagnostic finding inventory is missing a source projection")
        if (
            finding.rule.rule_id != rule_id
            or finding.observed_properties != observations
            or finding.possible_impacts != impacts
            or finding.assessments != assessments
            or finding.compared_capture_ids != compared_ids
        ):
            raise ValueError("diagnostic finding does not match its canonical source projection")

    reviews = {group.capture_id: group.reviews for group in run.section_reviews}
    for capture in rendered:
        for section in capture.sections:
            rule_id = "content-section-context-001"
            finding_id = (
                "diagnostic-"
                + hashlib.sha256(
                    f"{rule_id}\0{capture.capture_id}\0{section.section_id}".encode()
                ).hexdigest()
            )
            assessments = tuple(
                item
                for item in reviews[section.capture_id]
                if item.review.section_id == section.section_id
            )
            observations: tuple[ObservedContentClaim, ...] = (
                ObservedContentClaim(
                    predicate="section_heading",
                    value=section.heading.excerpt,
                    capture_id=section.capture_id,
                    section_id=section.section_id,
                    locator=section.locator,
                ),
                ObservedContentClaim(
                    predicate="section_has_body",
                    value=section.body.source_length > 0,
                    capture_id=section.capture_id,
                    section_id=section.section_id,
                    locator=section.locator,
                ),
                ObservedContentClaim(
                    predicate="section_excerpt",
                    value=section.body.excerpt,
                    capture_id=section.capture_id,
                    section_id=section.section_id,
                    locator=section.locator,
                ),
            )
            impacts = tuple(
                InferredContentClaim(
                    meaning=item.review.rationale,
                    assessment_kind="agent_assessment",
                    evidence=item.evidence,
                )
                for item in assessments
                if item.review.result != "unknown"
            )
            check(finding_id, rule_id, observations, impacts, assessments)

    by_capture = {capture.capture_id: capture for capture in rendered}
    for pair in run.pairs:
        if pair.state is not DataState.AVAILABLE:
            continue
        assert pair.raw_capture_id is not None and pair.rendered_capture_id is not None
        capture = by_capture[pair.rendered_capture_id]
        sections = {section.section_id: section for section in capture.sections}
        for passage in pair.rendered_only_passages:
            rule_id = "content-render-parity-001"
            finding_id = (
                "diagnostic-"
                + hashlib.sha256(
                    f"{rule_id}\0{pair.raw_capture_id}\0{passage.model_dump_json()}".encode()
                ).hexdigest()
            )
            parity_section = (
                sections.get(passage.section_id) if passage.section_id is not None else None
            )
            observations = (
                ObservedContentClaim(
                    predicate="rendered_only_passage",
                    value=passage.quote,
                    capture_id=passage.capture_id,
                    section_id=passage.section_id,
                    locator=parity_section.locator if parity_section is not None else None,
                    evidence=(passage,),
                ),
            )
            impacts = (
                InferredContentClaim(
                    meaning="The passage may be less accessible to systems using only the "
                    "sampled raw HTML; platform behavior is unverified.",
                    assessment_kind="audit_inference",
                    evidence=(passage,),
                ),
            )
            check(
                finding_id,
                rule_id,
                observations,
                impacts,
                compared_ids=(pair.raw_capture_id, pair.rendered_capture_id),
            )
    if set(actual) != expected_ids:
        raise ValueError("diagnostic finding inventory contains an unexpected source projection")


def run_diagnostics(
    project_ref: str,
    *,
    clients_root: str | Path,
    source_version: str,
    owned_dir: Path,
    intake_root: Path,
    now: datetime | None = None,
    crawler_transport: httpx.BaseTransport | None = None,
    crawler_resolver: Resolver | None = None,
) -> Path:
    """Consume owned input and publish only after successful verified deletion.

    ``now`` injects the coordinator clock for reproducible offline processing; HTTP
    capture and AI observation timestamps always come from their actual collection.
    """
    from ai_search_audit.diagnostic_store import DiagnosticStore

    def process(descriptor: int) -> dict[str, object]:
        intake = read_diagnostic_payload(descriptor)
        contract = prepare_diagnostic_contract(
            project_ref,
            clients_root=clients_root,
            source_version=source_version,
            selected_pages=intake.selected_pages,
        )
        if intake.expected_binding != contract.source.binding:
            raise ValueError("diagnostic input binding does not match canonical source")
        rendered = {c.url: _import_rendered(c) for c in intake.rendered_captures}
        if any(url not in contract.selected_page_urls for url in rendered):
            raise ValueError("rendered URL is not in the audited selected page sample")
        captures_by_id = {c.capture_id: c for c in rendered.values()}
        for passage in intake.key_passages:
            capture = captures_by_id.get(passage.capture_id)
            if capture is None:
                raise ValueError("key passage capture reference does not resolve")
            _validate_passage(capture, passage)
        for group in intake.section_reviews:
            capture = captures_by_id.get(group.capture_id)
            if capture is None or capture.extracted is None:
                raise ValueError("section review capture reference does not resolve")
            validate_section_reviews(capture.extracted, group.reviews)
        worksheet = contract.worksheet
        if intake.worksheet is not None:
            worksheet = validate_benchmark_worksheet(contract.source, intake.worksheet)
        if intake.setup is not None:
            worksheet = prepare_benchmark_worksheet(contract.source, intake.setup)
        benchmark = (
            validate_benchmark_responses(contract.source, worksheet, intake.responses)
            if (intake.worksheet is not None or intake.responses)
            else None
        )
        comparison = None
        baseline = None
        if intake.baseline_run is not None:
            if benchmark is None:
                raise ValueError("baseline comparison requires a benchmark sample")
            store = DiagnosticStore(Path(clients_root) / contract.source.binding.project_id)
            loaded = store.load(intake.baseline_run.source_version, intake.baseline_run.run_id)
            if not isinstance(loaded.run, DiagnosticRun) or loaded.run.benchmark is None:
                raise ValueError("baseline run contains no benchmark sample")
            baseline_source = load_diagnostic_source(
                project_ref,
                clients_root=clients_root,
                source_version=intake.baseline_run.source_version,
            )
            comparison = compare_benchmarks(
                loaded.run.benchmark,
                benchmark,
                baseline_source=baseline_source,
                follow_up_source=contract.source,
            )
            baseline = DiagnosticBaseline(
                reference=intake.baseline_run,
                binding=loaded.run.binding,
                manifest_sha256=loaded.manifest_sha256,
            )
        return _assemble(
            intake,
            contract,
            rendered,
            worksheet=worksheet,
            benchmark=benchmark,
            comparison=comparison,
            baseline=baseline,
            now=now,
            crawler_transport=crawler_transport,
            crawler_resolver=crawler_resolver,
        )

    values = consume_owned_payload(owned_dir, intake_root=intake_root, processor=process)
    # Construct this fact only after the owned lifecycle returns successfully.
    values["cleanup"] = DiagnosticCleanup(completed_at=now or datetime.now(UTC))
    run = DiagnosticRun.model_validate(values)
    return DiagnosticStore(Path(clients_root) / run.binding.project_id).publish(run)


def _assemble(
    intake: DiagnosticIntake,
    contract: DiagnosticContract,
    rendered: dict[str, ContentCapture],
    *,
    worksheet: BenchmarkWorksheet,
    benchmark: BenchmarkSample | None,
    comparison: BenchmarkComparison | None,
    baseline: DiagnosticBaseline | None,
    now: datetime | None,
    crawler_transport: httpx.BaseTransport | None,
    crawler_resolver: Resolver | None,
) -> dict[str, object]:
    evidence: list[DiagnosticCaptureEvidence] = []
    pairs: list[PairDiagnostic] = []
    findings: list[DiagnosticFinding] = []
    reviews: list[PageSectionReviews] = []
    as_of = (now or datetime.now(UTC)).date()
    for url in contract.selected_page_urls:
        # Use the actual audited URL, retaining HTTP-only origins and explicit ports.
        crawler = NativeCrawler(
            AuditConfig(domain=url), [], transport=crawler_transport, resolver=crawler_resolver
        )
        attempt = crawler.capture_page(url)
        raw = _raw_capture(attempt, contract.session_key)
        browser = rendered.get(url)
        passages = tuple(
            p
            for p in intake.key_passages
            if browser is not None and p.capture_id == browser.capture_id
        )
        pair = compare_captures(raw, browser, key_passages=passages)
        pairs.append(pair)
        evidence.append(_capture_evidence(url, raw, attempt=attempt))
        findings.extend(
            build_render_parity_findings(raw, browser, key_passages=passages, as_of=as_of)
        )
        if browser is not None:
            retained = list(passages)
            if browser.extracted is not None:
                submitted = next(
                    (
                        group.reviews
                        for group in intake.section_reviews
                        if group.capture_id == browser.capture_id
                    ),
                    (),
                )
                group = validate_section_reviews(browser.extracted, submitted)
                reviews.append(group)
                retained.extend(p for review in group.reviews for p in review.evidence)
                findings.extend(build_section_findings(browser.extracted, submitted, as_of=as_of))
            evidence.append(_capture_evidence(url, browser, passages=tuple(retained)))
    captures = tuple(evidence)
    frozen_findings = tuple(findings)
    return {
        "binding": contract.source.binding,
        "input_sha256": canonical_hash(
            {
                "intake": intake.model_dump(mode="json"),
                "raw_attempts": [
                    c.attempt.model_dump(mode="json") for c in captures if c.attempt is not None
                ],
            }
        ),
        "algorithm_sha256": _algorithm_hash(frozen_findings),
        "collection_range": _collection_range(
            captures, tuple(r.observed_at for r in benchmark.responses) if benchmark else ()
        ),
        "selected_pages": contract.selected_pages,
        "session_key": contract.session_key,
        "worksheet": worksheet,
        "captures": captures,
        "pairs": tuple(pairs),
        "section_reviews": tuple(reviews),
        "findings": frozen_findings,
        "benchmark": benchmark,
        "comparison": comparison,
        "baseline": baseline,
        "module_states": _module_states(
            captures,
            tuple(pairs),
            tuple(reviews),
            benchmark.metrics.state if benchmark else DataState.UNAVAILABLE,
        ),
    }
