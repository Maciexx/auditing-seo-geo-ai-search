from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import httpx
import pytest

from ai_search_audit.content_diagnostics import capture_html
from ai_search_audit.data_intake import IntakeCleanupError, create_owned_intake_dir
from ai_search_audit.diagnostic_models import DiagnosticRun, DiagnosticSelectedPage
from ai_search_audit.models import DataState
from tests.test_crawler import public_resolver
from tests.test_diagnostic_intake import rendered
from tests.test_project_orchestrator import _create


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic tests must not perform live network requests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_preparation_does_not_rewrite_canonical_project(tmp_path):
    manifest = _create(tmp_path, domain="https://studio.example", client_name="Example Studio")
    root = tmp_path / "clients/example"
    before = _hashes(root)
    workflow = import_module("ai_search_audit.diagnostic_workflow")
    contract = workflow.prepare_diagnostic_contract(
        "project:example", clients_root=root.parent, source_version="public-v1"
    )
    assert contract.source.binding.audit_id == manifest.latest_audit_id
    assert len(contract.selected_page_urls) <= 5
    assert contract.selected_page_urls[0] == "https://studio.example/"
    assert contract.worksheet.prompts == contract.source.prompts
    assert before == _hashes(root)


@pytest.fixture
def project(tmp_path):
    _create(tmp_path, domain="https://studio.example", client_name="Example Studio")
    return tmp_path / "clients/example"


def _contract(root, **kwargs):
    return import_module("ai_search_audit.diagnostic_workflow").prepare_diagnostic_contract(
        f"project:{root.name}", clients_root=root.parent, source_version="public-v1", **kwargs
    )


def _intake(root, **changes):
    contract = _contract(root)
    owned = create_owned_intake_dir(root.parent.parent / "intake")
    payload = {"expected_binding": contract.source.binding.model_dump(mode="json"), **changes}
    (owned / "normalized-intake.json").write_text(json.dumps(payload))
    return owned


def _run(root, *, owned=None, handler=None, **kwargs):
    if owned is None:
        owned = _intake(root)
    if handler is None:

        def handler(request):
            return httpx.Response(
                200,
                text='<html lang="en"><main><h1>Studio</h1><p>Service.</p></main></html>',
                headers={"content-type": "text/html"},
            )

    workflow = import_module("ai_search_audit.diagnostic_workflow")
    return workflow.run_diagnostics(
        f"project:{root.name}",
        clients_root=root.parent,
        source_version="public-v1",
        owned_dir=owned,
        intake_root=owned.parent,
        crawler_transport=httpx.MockTransport(handler),
        crawler_resolver=public_resolver,
        **kwargs,
    )


def _load(root, path):
    return (
        import_module("ai_search_audit.diagnostic_store")
        .DiagnosticStore(root)
        .load("public-v1", path.name)
    )


def _rendered(root, **changes):
    return rendered(
        observed_at=datetime.now(UTC).isoformat(),
        session_key=_contract(root).session_key,
        **changes,
    )


def test_missing_optional_collection_and_source_status_are_preserved(project):
    before = _hashes(project)
    owned = _intake(project)
    path = _run(project, owned=owned)
    loaded = _load(project, path)
    assert not owned.exists()
    assert loaded.run.module_states.render_parity is DataState.UNAVAILABLE
    assert loaded.run.module_states.benchmark is DataState.UNAVAILABLE
    assert loaded.run.benchmark is None
    assert loaded.run.collection_range.start is not None
    assert loaded.run.cleanup.status == "deleted"
    assert before == {k: v for k, v in _hashes(project).items() if not k.startswith("diagnostics/")}


def test_fresh_raw_uses_actual_locale_and_does_not_copy_browser_metadata(project):
    capture = _rendered(project, viewport=[800, 600])
    path = _run(project, owned=_intake(project, rendered_captures=[capture]))
    run = _load(project, path).run
    raw = next(item for item in run.captures if item.kind == "raw")
    assert raw.metadata.locale == "en"
    assert raw.metadata.viewport is None
    assert raw.metadata.consent_state == "none"
    assert run.pairs[0].state is DataState.AVAILABLE


@pytest.mark.parametrize(
    "consent,html,headers",
    [
        ("accepted", '<html lang="en"><main>Service</main></html>', {}),
        ("none", "<html><main>Service</main></html>", {}),
        ("none", '<html lang="en"><main>Service</main></html>', {"cf-mitigated": "challenge"}),
    ],
)
def test_ambiguous_pair_remains_unknown(project, consent, html, headers):
    owned = _intake(project, rendered_captures=[_rendered(project, consent_state=consent)])
    path = _run(
        project,
        owned=owned,
        handler=lambda request: httpx.Response(
            200, text=html, headers={"content-type": "text/html", **headers}
        ),
    )
    run = _load(project, path).run
    assert run.pairs[0].state is DataState.UNKNOWN
    assert not run.pairs[0].rendered_only_passages
    if headers:
        assert run.captures[0].attempt.state is DataState.UNKNOWN


def test_collector_failure_is_failed_evidence(project):
    def fail(request):
        raise httpx.ConnectError("offline")

    run = _load(project, _run(project, handler=fail)).run
    assert run.module_states.raw_capture is DataState.FAILED
    assert run.captures[0].attempt.state is DataState.FAILED
    assert run.captures[0].content_sha256 is None


@pytest.mark.parametrize(
    "changes",
    [
        {"key_passages": [{"capture_id": "missing", "quote": "invented"}]},
        {"section_reviews": [{"capture_id": "missing", "reviews": []}]},
        {"selected_pages": [{"url": "https://studio.example/not-audited", "reason": "important"}]},
    ],
)
def test_invalid_references_reject_and_clean_owned_input(project, changes):
    before = _hashes(project)
    owned = _intake(project, **changes)
    with pytest.raises(ValueError):
        _run(project, owned=owned)
    assert not owned.exists()
    assert _hashes(project) == before


def test_invented_quote_is_rejected_even_when_pair_is_not_comparable(project):
    raw_input = _rendered(project, consent_state="accepted")
    source = capture_html(
        **{key: value for key, value in raw_input.items() if key != "account_state"}
    )
    owned = _intake(
        project,
        rendered_captures=[raw_input],
        key_passages=[{"capture_id": source.capture_id, "quote": "Invented passage."}],
    )
    with pytest.raises(ValueError, match="passage"):
        _run(project, owned=owned)
    assert not owned.exists()


def test_cleanup_failure_never_publishes(project, monkeypatch):
    import ai_search_audit.data_intake as intake

    def fail(*args, **kwargs):
        raise IntakeCleanupError("synthetic cleanup failure")

    monkeypatch.setattr(intake, "_delete_verified_owned_directory", fail)
    owned = _intake(project)
    with pytest.raises(IntakeCleanupError):
        _run(project, owned=owned)
    assert not (project / "diagnostics").exists()


def test_wrong_locale_binding_rejects_and_cleans(project):
    wrong = _contract(project).source.binding.model_dump(mode="json")
    wrong["report_locale"] = "pl"
    owned = _intake(project, expected_binding=wrong)
    with pytest.raises(ValueError, match="binding"):
        _run(project, owned=owned)
    assert not owned.exists()


def test_explicit_selection_rejects_unseen_urls(project):
    with pytest.raises(ValueError, match="audited"):
        _contract(
            project,
            selected_pages=(
                DiagnosticSelectedPage(
                    url="https://studio.example/unknown", reason="Important service"
                ),
            ),
        )


def test_http_only_audited_site_preserves_scheme_and_fresh_collection(tmp_path):
    _create(tmp_path, domain="http://studio.example", client_name="Example Studio")
    root = tmp_path / "clients/example"
    contract = _contract(root)
    assert contract.selected_page_urls[0] == "http://studio.example/"
    assert all(url.startswith("http://") for url in contract.selected_page_urls)
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(
            200,
            text='<html lang="en"><main>Service</main></html>',
            headers={"content-type": "text/html"},
        )

    _run(root, handler=handler)
    assert requested == list(contract.selected_page_urls)


@pytest.fixture(scope="module")
def reviewed_run(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("diagnostic-reviewed")
    _create(root_dir, domain="https://studio.example", client_name="Example Studio")
    root = root_dir / "clients/example"
    html = '<html lang="en"><main><h1>Studio</h1><h2>Service</h2><h3>Delivery</h3><p>'
    html += "Context. " * 260 + "Delivery may take 2 days.</p><ul><li>No guarantee.</li></ul>"
    html += "<table><tr><td>Days</td><td>2</td></tr></table></main></html>"
    capture_input = _rendered(root, html=html, consent_state="accepted")
    capture = capture_html(
        **{key: value for key, value in capture_input.items() if key != "account_state"}
    )
    section = next(s for s in capture.sections if s.heading == "Delivery")
    review = {
        "section_id": section.section_id,
        "criterion": "directness",
        "result": "needs_review",
        "quotes": ["Delivery may take 2 days."],
        "rationale": "Clarify when the delivery period begins.",
    }
    owned = _intake(
        root,
        rendered_captures=[capture_input],
        section_reviews=[{"capture_id": capture.capture_id, "reviews": [review]}],
    )
    return _load(root, _run(root, owned=owned)).run


def test_persisted_sections_keep_heading_ancestry_and_block_relationships(reviewed_run):
    rendered_evidence = next(c for c in reviewed_run.captures if c.kind == "rendered")
    section = next(s for s in rendered_evidence.sections if s.heading.excerpt == "Delivery")
    assert tuple(part.excerpt for part in section.heading_path) == ("Studio", "Service", "Delivery")
    assert {"paragraph", "list_item", "table_row"}.issubset(section.block_kinds)
    assert section.body.truncated and len(section.body.excerpt) == 2000
    assert section.body.source_sha256 != section.body.excerpt_sha256
    assert any(p.quote == "Delivery may take 2 days." for p in rendered_evidence.passages)
    serialized = reviewed_run.model_dump_json()
    assert "<html" not in serialized and 'html":' not in serialized
    assert "Context. " * 260 not in serialized


@pytest.mark.parametrize(
    "field,value",
    [
        ("status_code", 201),
        ("final_url", "https://studio.example/other"),
        ("collector", "not-the-observed-collector"),
        ("url", "https://studio.example/other"),
    ],
)
def test_raw_attempt_and_extracted_metadata_cannot_disagree(reviewed_run, field, value):
    payload = reviewed_run.model_dump(mode="json")
    raw = next(c for c in payload["captures"] if c["kind"] == "raw")
    raw["attempt"][field] = value
    with pytest.raises(ValueError, match="attempt|metadata"):
        DiagnosticRun.model_validate(payload)


def test_pair_state_cannot_upgrade_noncomparable_metadata(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    assert payload["pairs"][0]["state"] == "UNKNOWN"
    payload["pairs"][0]["state"] = "AVAILABLE"
    payload["module_states"]["render_parity"] = "PARTIAL"
    with pytest.raises(ValueError, match="pair"):
        DiagnosticRun.model_validate(payload)


def test_review_directness_cannot_be_upgraded_without_review(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    item = next(d for d in payload["section_reviews"][0]["directness"] if d["state"] == "UNKNOWN")
    item.update(state="AVAILABLE", result="adequate")
    with pytest.raises(ValueError, match="directness"):
        DiagnosticRun.model_validate(payload)


def test_finding_assessment_prose_must_come_from_validated_review(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    finding = next(f for f in payload["findings"] if f["possible_impacts"])
    finding["possible_impacts"][0]["meaning"] = "An unrelated new conclusion."
    with pytest.raises(ValueError, match="review|finding"):
        DiagnosticRun.model_validate(payload)


def test_review_quote_cannot_diverge_from_retained_validated_evidence(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    payload["section_reviews"][0]["reviews"][0]["review"]["quotes"] = ["An invented quotation."]
    with pytest.raises(ValueError, match="quote|review"):
        DiagnosticRun.model_validate(payload)


def test_fixed_collector_and_clock_produce_identical_analytical_payload(project, monkeypatch):
    from ai_search_audit.crawler import NativeCrawler
    from ai_search_audit.diagnostic_models import PageCaptureResult

    moment = datetime(2026, 9, 3, 9, 30, tzinfo=UTC)
    html = '<html lang="en"><main>Same content.</main></html>'

    def captured(self, url):
        return PageCaptureResult(
            url=url,
            final_url=url,
            observed_at=moment,
            collector="fixture",
            status_code=200,
            content_type="text/html",
            body=html.encode(),
            html=html,
            complete=True,
            truncated=False,
            state=DataState.AVAILABLE,
        )

    monkeypatch.setattr(NativeCrawler, "capture_page", captured)
    first = _load(project, _run(project, now=moment))
    second = _load(project, _run(project, now=moment))
    assert first.run == second.run
    assert first.manifest.run_number != second.manifest.run_number


def test_pl_report_language_does_not_replace_actual_english_page_locale(tmp_path):
    _create(
        tmp_path, domain="https://studio.example", client_name="Example Studio", report_locale="pl"
    )
    root = tmp_path / "clients/example"
    owned = _intake(root, rendered_captures=[_rendered(root)])
    run = _load(root, _run(root, owned=owned)).run
    assert run.binding.report_locale == "pl"
    assert run.pairs[0].state is DataState.AVAILABLE
    assert run.captures[0].metadata.locale == "en"


def test_whole_capture_passage_can_span_section_boundaries(project):
    capture_input = _rendered(
        project, html="<main><h1>Studio</h1><p>Delivery may take 2 days.</p></main>"
    )
    capture = capture_html(
        **{key: value for key, value in capture_input.items() if key != "account_state"}
    )
    owned = _intake(
        project,
        rendered_captures=[capture_input],
        key_passages=[
            {"capture_id": capture.capture_id, "quote": "Studio Delivery may take 2 days."}
        ],
    )
    run = _load(project, _run(project, owned=owned)).run
    assert run.pairs[0].rendered_only_passages[0].section_id is None


def test_benchmark_setup_rebuilds_exact_worksheet_and_compares_explicit_baseline(project):
    from tests.test_benchmark import _setup
    from tests.test_diagnostic_intake import response

    worksheet = _contract(project).worksheet
    prompt = worksheet.prompts[0]
    supplied = dict(
        worksheet=worksheet.model_dump(mode="json"),
        setup=_setup().model_dump(mode="json"),
        responses=[
            response(
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.text,
                response_text="Observed answer. " * 150,
            )
        ],
    )
    first = _run(project, owned=_intake(project, **supplied))
    supplied["responses"][0]["observed_at"] = "2026-09-04T09:30:00+00:00"
    second = _run(
        project,
        owned=_intake(
            project, **supplied, baseline_run={"source_version": "public-v1", "run_id": first.name}
        ),
    )
    baseline = _load(project, first)
    result = _load(project, second).run
    assert result.benchmark.worksheet.prompts == worksheet.prompts
    assert result.benchmark.worksheet.setup == _setup()
    assert result.baseline.manifest_sha256 == baseline.manifest_sha256
    assert result.comparison.mention_delta == 0
    assert result.comparison.causality == "NOT_ESTABLISHED"
    assert len(result.benchmark.responses[0].response_excerpt) == 2000
    assert result.benchmark.responses[0].excerpt_truncated


def test_explicit_page_selection_keeps_prepared_anonymous_conditions_label(project):
    contract = _contract(project)
    selected = (DiagnosticSelectedPage(url=contract.selected_page_urls[0], reason="Core offering"),)
    custom = _contract(project, selected_pages=selected)
    assert custom.session_key == contract.session_key
    owned = _intake(
        project,
        selected_pages=[page.model_dump(mode="json") for page in selected],
        rendered_captures=[_rendered(project)],
    )
    run = _load(project, _run(project, owned=owned)).run
    assert run.pairs[0].state is DataState.AVAILABLE


@pytest.mark.parametrize("state", ["PARTIAL", "UNAVAILABLE", "FAILED"])
def test_capture_state_cannot_contradict_successful_extraction(reviewed_run, state):
    from ai_search_audit.diagnostic_models import DiagnosticCaptureEvidence

    payload = reviewed_run.model_dump(mode="json")
    payload["captures"][0]["state"] = state
    with pytest.raises(ValueError, match="capture|extraction"):
        DiagnosticCaptureEvidence.model_validate(payload["captures"][0])


def test_missing_capture_text_is_not_a_valid_successful_capture(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    payload["captures"][0].update(text=None, content_sha256=None, sections=[])
    with pytest.raises(ValueError, match="capture|extraction"):
        DiagnosticRun.model_validate(payload)


def test_empty_review_group_cannot_drop_unknown_section_states(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    payload["section_reviews"] = []
    payload["module_states"]["section_reviews"] = "UNAVAILABLE"
    with pytest.raises(ValueError, match="review"):
        DiagnosticRun.model_validate(payload)


@pytest.mark.parametrize(
    "html",
    [
        "<main><h1>Studio</h1><p>Delivery may take 2 days.</p></main>",
        "<main></main>",
    ],
)
@pytest.mark.parametrize("mutation", ["remove", "duplicate"])
def test_unreviewed_capture_requires_exact_review_group_inventory(project, html, mutation):
    owned = _intake(project, rendered_captures=[_rendered(project, html=html)])
    run = _load(project, _run(project, owned=owned)).run
    assert len(run.section_reviews) == 1
    assert run.section_reviews[0].reviews == ()
    if run.section_reviews[0].directness:
        assert run.module_states.section_reviews is DataState.UNKNOWN
    else:
        assert run.module_states.section_reviews is DataState.UNAVAILABLE
    payload = run.model_dump(mode="json")
    if mutation == "remove":
        payload["section_reviews"] = []
        payload["module_states"]["section_reviews"] = "UNAVAILABLE"
    else:
        payload["section_reviews"] *= 2
    with pytest.raises(ValueError, match="review.*inventory"):
        DiagnosticRun.model_validate(payload)


def test_capture_text_hash_is_not_evidence_for_invented_short_passage(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    raw = payload["captures"][0]
    raw["passages"] = [{"capture_id": raw["capture_id"], "quote": "Invented text."}]
    with pytest.raises(ValueError, match="passage"):
        DiagnosticRun.model_validate(payload)


def test_source_bindings_reject_report_status_fields(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    payload["report_status"] = "final"
    with pytest.raises(ValueError, match="report_status"):
        DiagnosticRun.model_validate(payload)


def test_changed_raw_input_changes_input_fingerprint_with_same_intake(project, monkeypatch):
    from ai_search_audit.crawler import NativeCrawler
    from ai_search_audit.diagnostic_models import PageCaptureResult

    moment = datetime(2026, 9, 3, 9, 30, tzinfo=UTC)
    body = ["First response"]

    def captured(self, url):
        html = f'<html lang="en"><main>{body[0]}</main></html>'
        return PageCaptureResult(
            url=url,
            final_url=url,
            observed_at=moment,
            collector="fixture",
            status_code=200,
            content_type="text/html",
            body=html.encode(),
            html=html,
            complete=True,
            truncated=False,
            state=DataState.AVAILABLE,
        )

    monkeypatch.setattr(NativeCrawler, "capture_page", captured)
    first = _load(project, _run(project, now=moment)).run
    body[0] = "Second response"
    second = _load(project, _run(project, now=moment)).run
    assert first.input_sha256 != second.input_sha256


def test_overlong_actual_html_language_stays_unknown_without_losing_content(project):
    owned = _intake(project, rendered_captures=[_rendered(project)])
    html = '<html lang="' + "x" * 65 + '"><main>Service</main></html>'
    path = _run(
        project,
        owned=owned,
        handler=lambda request: httpx.Response(
            200, text=html, headers={"content-type": "text/html"}
        ),
    )
    run = _load(project, path).run
    assert run.captures[0].state is DataState.AVAILABLE
    assert run.captures[0].metadata.locale is None
    assert run.captures[0].text.excerpt == "Service"
    assert run.pairs[0].state is DataState.UNKNOWN
    assert any("locale" in reason.casefold() for reason in run.pairs[0].limitations)


def test_mutated_review_and_finding_cannot_bypass_existing_certainty_guard(reviewed_run):
    payload = reviewed_run.model_dump(mode="json")
    meaning = "This guarantees that AI systems will cite this page."
    payload["section_reviews"][0]["reviews"][0]["review"]["rationale"] = meaning
    finding = next(f for f in payload["findings"] if f["assessments"])
    finding["assessments"][0]["review"]["rationale"] = meaning
    finding["possible_impacts"][0]["meaning"] = meaning
    with pytest.raises(ValueError):
        DiagnosticRun.model_validate(payload)


def test_default_selection_is_audited_distinct_home_first_and_bounded():
    from ai_search_audit.diagnostic_workflow import _selected_pages
    from tests.test_benchmark import _source

    source = _source().model_copy(
        update={
            "page_urls": (
                "https://studio.example/?campaign=x",
                "https://studio.example/service",
                "https://studio.example/",
                "https://studio.example/service",
                *(f"https://studio.example/page-{i}" for i in range(12)),
                "https://unrelated.example/",
            )
        }
    )
    pages = _selected_pages(source, ())
    assert [p.url for p in pages] == [
        "https://studio.example/",
        "https://studio.example/?campaign=x",
        "https://studio.example/service",
        "https://studio.example/page-0",
        "https://studio.example/page-1",
    ]
    eleven = tuple(
        DiagnosticSelectedPage(url=f"https://studio.example/page-{i}", reason="Offering")
        for i in range(11)
    )
    with pytest.raises(ValueError, match="ten"):
        _selected_pages(source, eleven)
    assert len(_selected_pages(source, eleven[:10])) == 10


@pytest.mark.parametrize(
    "final_url",
    [
        "https://outside.example/",
        "http://127.0.0.1/",
        "https://user:secret@studio.example/",
    ],
)
def test_unextracted_attempt_urls_still_obey_public_canonical_scope(final_url):
    from ai_search_audit.diagnostic_models import DiagnosticAttemptEvidence

    with pytest.raises(ValueError):
        DiagnosticAttemptEvidence(
            url="https://studio.example/",
            final_url=final_url,
            observed_at=datetime(2026, 9, 3, tzinfo=UTC),
            collector="fixture",
            status_code=200,
            content_type="application/json",
            body_sha256="a" * 64,
            complete=True,
            truncated=False,
            state=DataState.UNKNOWN,
            limitations=("Not an HTML response.",),
        )


def _tampered_finding_payload(run, mutation):
    from ai_search_audit.diagnostic_models import DiagnosticFinding
    from ai_search_audit.diagnostic_workflow import _algorithm_hash

    payload = run.model_dump(mode="json")
    finding = payload["findings"][0]
    if mutation == "rule_id":
        finding["rule"]["rule_id"] = "invented-diagnostic-rule"
    elif mutation == "rule_statement":
        finding["rule"]["statement"] = "An invented platform policy."
    elif mutation == "expired_resolution":
        finding["resolved_as_of"] = "2027-09-03"
    elif mutation == "status":
        finding["status"] = (
            "REQUIRES_VERIFICATION" if finding["status"] == "INFERRED" else "INFERRED"
        )
    elif mutation == "invented_predicate":
        finding["observed_properties"][0].update(
            predicate="rendered_only_passage",
            value="Invented passage with no evidence.",
            evidence=[],
        )
    elif mutation == "locator":
        finding["observed_properties"][0]["locator"] = "invented-location"
    elif mutation == "missing_section_ref":
        finding["observed_properties"][0]["section_id"] = None
    elif mutation == "drop_property":
        finding["observed_properties"].pop()
    elif mutation == "duplicate_property":
        finding["observed_properties"].append(finding["observed_properties"][0])
    elif mutation == "empty_properties":
        finding["observed_properties"] = []
    elif mutation == "drop_finding":
        payload["findings"].pop(0)
    elif mutation == "duplicate_finding":
        payload["findings"].append(finding)
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")
    payload["algorithm_sha256"] = _algorithm_hash(
        tuple(DiagnosticFinding.model_validate(item) for item in payload["findings"])
    )
    return payload


@pytest.mark.parametrize("mutation", ["rule_id", "rule_statement", "expired_resolution", "status"])
def test_finding_rule_snapshot_must_match_registry_at_recorded_date(reviewed_run, mutation):
    payload = _tampered_finding_payload(reviewed_run, mutation)
    with pytest.raises(ValueError, match="rule|status"):
        DiagnosticRun.model_validate(payload)


def test_run_rule_validation_loads_registry_once(reviewed_run, monkeypatch):
    from ai_search_audit import diagnostic_workflow as workflow
    from ai_search_audit.knowledge import load_registry

    calls = []

    def counted(root):
        calls.append(root)
        return load_registry(root)

    monkeypatch.setattr(workflow, "load_registry", counted, raising=False)
    DiagnosticRun.model_validate(reviewed_run.model_dump(mode="json"))
    assert len(reviewed_run.findings) > 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "invented_predicate",
        "locator",
        "missing_section_ref",
        "drop_property",
        "duplicate_property",
        "empty_properties",
        "drop_finding",
        "duplicate_finding",
    ],
)
def test_section_finding_projection_and_inventory_are_exact(reviewed_run, mutation):
    payload = _tampered_finding_payload(reviewed_run, mutation)
    with pytest.raises(ValueError, match="finding"):
        DiagnosticRun.model_validate(payload)


@pytest.fixture(scope="module")
def parity_run(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("diagnostic-parity")
    _create(root_dir, domain="https://studio.example", client_name="Example Studio")
    root = root_dir / "clients/example"
    capture_input = _rendered(root)
    capture = capture_html(
        **{key: value for key, value in capture_input.items() if key != "account_state"}
    )
    passages = [
        {
            "capture_id": capture.capture_id,
            "section_id": capture.sections[0].section_id,
            "quote": "Delivery may take 2 days.",
        },
        {"capture_id": capture.capture_id, "quote": "Studio Delivery may take 2 days."},
    ]
    owned = _intake(root, rendered_captures=[capture_input], key_passages=passages)
    return _load(root, _run(root, owned=owned)).run


@pytest.mark.parametrize(
    "mutation", ["quote", "evidence", "locator", "compared_ids", "drop", "duplicate"]
)
def test_parity_findings_match_exact_available_pair_passages(parity_run, mutation):
    payload = parity_run.model_dump(mode="json")
    finding = next(
        f for f in payload["findings"] if f["rule"]["rule_id"] == "content-render-parity-001"
    )
    if mutation == "quote":
        finding["observed_properties"][0]["value"] = "Invented passage."
    elif mutation == "evidence":
        finding["observed_properties"][0]["evidence"] = []
    elif mutation == "locator":
        finding["observed_properties"][0]["locator"] = "invented-location"
    elif mutation == "compared_ids":
        finding["compared_capture_ids"].reverse()
    elif mutation == "drop":
        payload["findings"].remove(finding)
    else:
        payload["findings"].append(finding)
    with pytest.raises(ValueError, match="finding"):
        DiagnosticRun.model_validate(payload)


@pytest.mark.parametrize("mutation", ["drop_impacts", "drop_assessment", "cross_section"])
def test_section_findings_preserve_all_section_specific_assessments(reviewed_run, mutation):
    payload = reviewed_run.model_dump(mode="json")
    original = next(f for f in payload["findings"] if f["assessments"])
    if mutation == "drop_impacts":
        original["possible_impacts"] = []
    else:
        if mutation == "cross_section":
            other = next(f for f in payload["findings"] if not f["assessments"])
            other["assessments"] = original["assessments"]
            other["possible_impacts"] = original["possible_impacts"]
        original["assessments"] = []
        original["possible_impacts"] = []
    with pytest.raises(ValueError, match="finding"):
        DiagnosticRun.model_validate(payload)


@pytest.mark.parametrize("copies", [0, 2])
def test_parity_finding_requires_exactly_one_fixed_inference(parity_run, copies):
    payload = parity_run.model_dump(mode="json")
    finding = next(
        f for f in payload["findings"] if f["rule"]["rule_id"] == "content-render-parity-001"
    )
    finding["possible_impacts"] *= copies
    with pytest.raises(ValueError, match="finding"):
        DiagnosticRun.model_validate(payload)
