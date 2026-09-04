from __future__ import annotations

import json
import os
import signal
import socket
from importlib import import_module
from pathlib import Path

import pytest
from pydantic import ValidationError

import ai_search_audit.diagnostic_models as models
from ai_search_audit.benchmark import prepare_benchmark_worksheet
from ai_search_audit.data_intake import (
    IntakeValidationError,
    consume_owned_payload,
    create_owned_intake_dir,
)
from ai_search_audit.diagnostic_models import DiagnosticBinding, DiagnosticSource, FrozenPrompt


def binding() -> DiagnosticBinding:
    return DiagnosticBinding(
        project_id="studio-example",
        source_version="public-v1",
        audit_id="audit-1",
        report_locale="en",
        domain="studio.example",
        source_sha256="a" * 64,
    )


def rendered(**changes):
    return {
        "kind": "rendered",
        "url": "https://studio.example/",
        "final_url": "https://studio.example/",
        "observed_at": "2026-09-03T09:30:00+00:00",
        "locale": "en",
        "session_key": "anonymous-1",
        "consent_state": "none",
        "account_state": "anonymous",
        "viewport": [1280, 720],
        "collector": "browser",
        "status_code": 200,
        "complete": True,
        "truncated": False,
        "html": "<main><h1>Studio</h1><p>Delivery may take 2 days.</p></main>",
        **changes,
    }


def review(**changes):
    return {
        "section_id": "section-1",
        "criterion": "directness",
        "result": "needs_review",
        "quotes": ["Delivery may take 2 days."],
        "rationale": "Clarify when the period begins.",
        **changes,
    }


def response(**changes):
    return {
        "prompt_id": "p1",
        "prompt_text": "Which studio delivers?",
        "observed_at": "2026-09-03T09:30:00+00:00",
        "grounded": True,
        "brand_mentioned": False,
        "citations": [],
        "citations_complete": True,
        "complete": True,
        "response_truncated": False,
        "inspection_scope": "full_response",
        "response_text": "No studio was found.",
        **changes,
    }


def worksheet(setup=None):
    source = DiagnosticSource(
        binding=binding(),
        prompts=(
            FrozenPrompt(
                prompt_id="p1",
                pack_version="1.0.0",
                locale="en",
                intent="discovery",
                text="Which studio delivers?",
            ),
        ),
        page_urls=("https://studio.example/",),
        canonical_domains=("studio.example",),
    )
    return prepare_benchmark_worksheet(source, setup).model_dump(mode="json")


def payload(**changes):
    return {"schema_version": "1.0.0", "expected_binding": binding().model_dump(), **changes}


def test_diagnostic_intake_is_separate_frozen_input_contract():
    data = payload(
        selected_pages=[{"url": "https://studio.example/", "reason": "Canonical homepage"}],
        rendered_captures=[rendered()],
        key_passages=[{"capture_id": "capture-1", "quote": "Delivery may take 2 days."}],
        section_reviews=[{"capture_id": "capture-1", "reviews": [review()]}],
        worksheet=worksheet(),
        responses=[response()],
        baseline_run={"source_version": "public-v1", "run_id": "run-1"},
    )
    result = models.DiagnosticIntake.model_validate(data)
    data["selected_pages"][0]["reason"] = "changed"
    data["rendered_captures"][0]["viewport"][0] = 1
    data["section_reviews"][0]["reviews"][0]["quotes"].append("invented")

    assert result.selected_pages[0].reason == "Canonical homepage"
    assert result.rendered_captures[0].viewport == (1280, 720)
    assert result.section_reviews[0].reviews[0].quotes == ("Delivery may take 2 days.",)
    assert result.responses[0].response_text == "No studio was found."
    assert result.baseline_run.run_id == "run-1"
    with pytest.raises(ValidationError, match="frozen"):
        result.selected_pages[0].reason = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        result.expected_binding.audit_id = "changed"


def test_minimal_diagnostic_intake_supports_unavailable_collection():
    result = models.DiagnosticIntake.model_validate(payload())
    assert result.selected_pages == result.rendered_captures == result.responses == ()
    assert result.worksheet is result.setup is result.baseline_run is None


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0.0"},
        {"status": "CONFIRMED"},
        {"metrics": {"mention_rate": 100}},
        {"sources": [{"filename": "nested/source.html"}]},
        {"rendered_captures": [rendered(kind="raw")]},
        {"rendered_captures": [rendered(html_path="nested/source.html")]},
        {"rendered_captures": [rendered(account_state="signed_in")]},
        {"selected_pages": [{"url": "https://studio.example/", "reason": " "}]},
        {"selected_pages": [{"url": "https://studio.example/", "reason": "x" * 501}]},
        {"selected_pages": [{"url": "../source.html", "reason": "Offering"}]},
        {"selected_pages": [{"url": "https://studio.example/", "reason": "Home", "type": "fake"}]},
        {"section_reviews": [{"capture_id": "capture-1", "reviews": [review(score=100)]}]},
        {"responses": [response()]},
        {"worksheet": worksheet(), "responses": [{"state": "AVAILABLE", "metrics": {}}]},
    ],
)
def test_diagnostic_intake_rejects_non_input_fields_and_invalid_declarations(changes):
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(**changes))


@pytest.mark.parametrize(
    "baseline",
    [
        "public-v1/run-1",
        {"source_version": "../public-v1", "run_id": "run-1"},
        {"source_version": "latest", "run_id": "run-1"},
        {"source_version": "public-v1", "run_id": "latest"},
        {"source_version": "public-v1", "run_id": "run-0"},
        {"source_version": "public-v1", "run_id": "run-01"},
        {"source_version": "public-v1", "run_id": "nested/run-1"},
        {"source_version": "public-v1", "run_id": "run-1", "path": "/tmp/run"},
    ],
)
def test_baseline_requires_explicit_safe_source_version_and_run_id(baseline):
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(baseline_run=baseline))


def test_page_and_review_workload_caps_are_explicit():
    pages = [{"url": f"https://studio.example/{i}", "reason": "Offering"} for i in range(11)]
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(selected_pages=pages))
    assert (
        len(
            models.DiagnosticIntake.model_validate(
                payload(selected_pages=pages[:10])
            ).selected_pages
        )
        == 10
    )
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(selected_pages=pages[:1] * 2))
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(rendered_captures=[rendered()] * 2))
    captures = [rendered(url=p["url"], final_url=p["url"]) for p in pages]
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(rendered_captures=captures))
    reviews = [review(section_id=f"section-{i}") for i in range(21)]
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(
            payload(section_reviews=[{"capture_id": "c1", "reviews": reviews}])
        )
    accepted = models.DiagnosticIntake.model_validate(
        payload(
            section_reviews=[
                {
                    "capture_id": "c1",
                    "reviews": reviews[:20]
                    + [review(section_id="section-0", criterion="conditions")],
                }
            ]
        )
    )
    assert len(accepted.section_reviews[0].reviews) == 21
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(
            payload(section_reviews=[{"capture_id": "c1", "reviews": [review(), review()]}])
        )
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(
            payload(section_reviews=[{"capture_id": "c1", "reviews": []}] * 2)
        )


def test_worksheet_binding_and_setup_must_not_contradict_intake():
    setup = models.BenchmarkSetup(provider="Example AI")
    sheet = worksheet(setup)
    result = models.DiagnosticIntake.model_validate(
        payload(worksheet=sheet, setup=setup.model_dump())
    )
    assert result.setup == result.worksheet.setup
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(
            payload(worksheet=sheet, setup={"provider": "Other AI"})
        )
    changed = binding().model_dump()
    changed["audit_id"] = "different-audit"
    with pytest.raises(ValidationError):
        models.DiagnosticIntake.model_validate(payload(worksheet=sheet, expected_binding=changed))
    # A worksheet with no recorded setup can be paired with newly observed settings.
    assert (
        models.DiagnosticIntake.model_validate(payload(worksheet=worksheet(), setup=setup)).setup
        == setup
    )


def consume_payload(owned: Path):
    reader = import_module("ai_search_audit.diagnostic_intake").read_diagnostic_payload
    return consume_owned_payload(owned, intake_root=owned.parent, processor=reader)


def write_payload(tmp_path: Path, data: bytes) -> Path:
    owned = create_owned_intake_dir(tmp_path / "intake")
    (owned / "normalized-intake.json").write_bytes(data)
    return owned


def test_reader_returns_validated_payload_after_cleanup_without_accessing_html_references(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic intake must not fetch HTML references")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    text = '<script>fetch("https://external.example/")</script><img src="file:///private.txt">'
    owned = write_payload(
        tmp_path, json.dumps(payload(rendered_captures=[rendered(html=text)])).encode()
    )
    (owned / "source.txt").write_text("synthetic attachment is deleted, not retained")
    result = consume_payload(owned)
    assert result.rendered_captures[0].html == text
    assert not owned.exists()
    assert tuple(owned.parent.iterdir()) == ()


@pytest.mark.parametrize(
    "data",
    [
        b"{",
        b"\xff",
        b"[]",
        b"null",
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        b'{"expected_binding":{"audit_id":"a","audit_id":"b"}}',
        b'{"expected_binding":NaN}',
        b'{"expected_binding":Infinity}',
        b'{"expected_binding":-Infinity}',
        b"[" * 2000 + b"0" + b"]" * 2000,
        json.dumps(payload())
        .replace(
            '"schema_version": "1.0.0"',
            '"schema_version": "1.0.0", "schema_version": "1.0.0"',
        )
        .encode(),
        json.dumps(payload())
        .replace('"audit_id": "audit-1"', '"audit_id": "different", "audit_id": "audit-1"')
        .encode(),
        json.dumps(payload(html_path="nested/source.html")).encode(),
        json.dumps(payload(rendered_captures=[rendered(filename="nested/source.html")])).encode(),
    ],
)
def test_reader_rejects_ambiguous_invalid_or_unknown_input_and_deletes_owned_directory(
    tmp_path, data
):
    owned = write_payload(tmp_path, data)
    with pytest.raises(IntakeValidationError):
        consume_payload(owned)
    assert not owned.exists()


@pytest.mark.parametrize("overflow", [0, 1, 20000])
def test_reader_observes_exact_total_byte_limit(tmp_path, monkeypatch, overflow):
    limit = 2 * 1024 * 1024
    base = json.dumps(payload()).encode()
    owned = write_payload(tmp_path, base + b" " * (limit + overflow - len(base)))
    actual_read = os.read
    bytes_read = 0

    def counting_read(fd, count):
        nonlocal bytes_read
        result = actual_read(fd, count)
        bytes_read += len(result)
        return result

    # Count only payload reads; ownership verification has separate marker reads.
    def processor(fd):
        reader = import_module("ai_search_audit.diagnostic_intake").read_diagnostic_payload
        with monkeypatch.context() as context:
            context.setattr(os, "read", counting_read)
            return reader(fd)

    if overflow:
        with pytest.raises(IntakeValidationError, match="2 MiB"):
            consume_owned_payload(owned, intake_root=owned.parent, processor=processor)
    else:
        assert (
            consume_owned_payload(
                owned, intake_root=owned.parent, processor=processor
            ).expected_binding
            == binding()
        )
    assert bytes_read <= limit + 1
    assert not owned.exists()


def test_reader_rejects_symlink_without_reading_or_deleting_its_target(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(payload()))
    owned = create_owned_intake_dir(tmp_path / "intake")
    (owned / "normalized-intake.json").symlink_to(outside)
    with pytest.raises(IntakeValidationError):
        consume_payload(owned)
    assert outside.read_text() == json.dumps(payload())
    assert not owned.exists()


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo"])
def test_reader_requires_a_regular_file_without_blocking_on_fifo(tmp_path, kind):
    owned = create_owned_intake_dir(tmp_path / "intake")
    entry = owned / "normalized-intake.json"
    if kind == "directory":
        entry.mkdir()
    elif kind == "fifo":
        os.mkfifo(entry)

    def alarm(signum, frame):
        raise TimeoutError("reader blocked on a non-regular input")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        with pytest.raises(IntakeValidationError):
            consume_payload(owned)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert not owned.exists()


def test_reader_opens_only_fixed_name_relative_to_supplied_descriptor(tmp_path, monkeypatch):
    owned = write_payload(tmp_path, json.dumps(payload()).encode())
    actual_open = os.open
    accesses = []

    def processor(fd):
        def observed_open(path, flags, mode=0o777, *, dir_fd=None):
            accesses.append((path, dir_fd, bool(flags & os.O_NOFOLLOW)))
            return actual_open(path, flags, mode, dir_fd=dir_fd)

        reader = import_module("ai_search_audit.diagnostic_intake").read_diagnostic_payload
        with monkeypatch.context() as context:
            context.setattr(os, "open", observed_open)
            result = reader(fd)
        assert accesses == [("normalized-intake.json", fd, True)]
        return result

    assert (
        consume_owned_payload(owned, intake_root=owned.parent, processor=processor).expected_binding
        == binding()
    )
    assert not owned.exists()
