from datetime import UTC, datetime, timedelta
from importlib.util import find_spec

import pytest
from pydantic import ValidationError

from ai_search_audit.models import DataState


def test_rendered_passage_is_an_observation_not_a_score() -> None:
    assert find_spec("ai_search_audit.content_diagnostics") is not None
    from ai_search_audit.content_diagnostics import capture_html, compare_captures

    common = dict(
        url="https://studio.example/",
        observed_at=datetime(2026, 9, 2, tzinfo=UTC),
        locale="pl",
        session_key="anonymous-1",
        consent_state="none",
        complete=True,
        truncated=False,
        status_code=200,
    )
    raw = capture_html("<main><h1>Studio</h1></main>", kind="raw", **common)
    dom = capture_html(
        "<main><h1>Studio</h1><p>Wsparcie może potrwać 2 dni.</p></main>",
        kind="rendered",
        **common,
    )
    result = compare_captures(raw, dom, key_quotes=("Wsparcie może potrwać 2 dni.",))
    assert result.state is DataState.AVAILABLE
    assert result.rendered_only_quotes == ("Wsparcie może potrwać 2 dni.",)
    assert "score" not in result.model_dump()


def sample(html="<main><h1>Studio</h1><p>Support may take 2 days.</p></main>", **changes):
    from ai_search_audit.content_diagnostics import capture_html

    metadata = dict(
        kind="raw",
        url="https://studio.example/",
        observed_at=datetime(2026, 9, 2, tzinfo=UTC),
        locale="en",
        session_key="anonymous-1",
        consent_state="none",
        complete=True,
        truncated=False,
        status_code=200,
    )
    metadata.update(changes)
    return capture_html(html, **metadata)


def test_static_parity_and_menu_changes_do_not_create_findings():
    from ai_search_audit.content_diagnostics import compare_captures

    html = "<main><h1>Studio</h1><p>Support may take 2 days.</p></main>"
    raw = sample(html + "<nav>Home</nav>")
    rendered = sample(html + "<nav>" + "More navigation " * 1000 + "</nav>", kind="rendered")
    result = compare_captures(raw, rendered, key_quotes=("Support may take 2 days.",))
    assert result.state is DataState.AVAILABLE
    assert result.rendered_only_quotes == ()
    assert raw.text == rendered.text


@pytest.mark.parametrize("padding", [1, 1000])
def test_rendered_only_observation_has_no_percentage_cutoff(padding):
    from ai_search_audit.content_diagnostics import compare_captures

    common = "<h1>Studio</h1><p>" + "Static text. " * padding + "</p>"
    raw = sample("<main>" + common + "</main>")
    rendered = sample("<main>" + common + "<p>New passage.</p></main>", kind="rendered")
    result = compare_captures(raw, rendered, key_quotes=("New passage.",))
    assert result.state is DataState.AVAILABLE
    assert result.rendered_only_quotes == ("New passage.",)
    assert "severity" not in result.model_dump()


@pytest.mark.parametrize(
    ("raw_quote", "rendered_quote"),
    [
        ("Price is 30 PLN.", "Price is 300 PLN."),
        ("Length is 2 cm.", "Length is 2 m."),
        ("It is available.", "It is not available."),
        ("It will take 2 days.", "It may take 2 days."),
    ],
)
def test_numbers_units_negation_and_modality_are_preserved(raw_quote, rendered_quote):
    from ai_search_audit.content_diagnostics import compare_captures

    raw = sample(f"<main><p>{raw_quote}</p></main>")
    rendered = sample(f"<main><p>{rendered_quote}</p></main>", kind="rendered")
    result = compare_captures(raw, rendered, key_quotes=(rendered_quote,))
    assert result.rendered_only_quotes == (rendered_quote,)
    assert raw_quote in raw.text
    assert rendered_quote in rendered.text


def test_quote_matching_uses_only_nfc_and_whitespace_and_retains_original():
    from ai_search_audit.content_diagnostics import compare_captures

    raw = sample("<main><p>Intro</p></main>")
    rendered = sample("<main><p>Café may take 2 days.</p></main>", kind="rendered")
    quote = "Cafe\u0301   may\n take 2 days."
    assert compare_captures(raw, rendered, key_quotes=(quote,)).rendered_only_quotes == (quote,)
    raw = sample("<main><p>Cafe\u0301   may\n take 2 days.</p></main>")
    assert compare_captures(raw, rendered, key_quotes=(quote,)).rendered_only_quotes == ()


@pytest.mark.parametrize("quote", ["", "  \n", "Invented", "support may take 2 days."])
def test_missing_or_nonexact_quote_rejects(quote):
    from ai_search_audit.content_diagnostics import compare_captures

    with pytest.raises(ValueError, match="key passage does not resolve to rendered source"):
        compare_captures(sample(), sample(kind="rendered"), key_quotes=(quote,))


@pytest.mark.parametrize(
    ("changes", "state", "reason"),
    [
        ({"locale": "pl"}, DataState.UNKNOWN, "locale"),
        ({"locale": None}, DataState.UNKNOWN, "locale"),
        ({"session_key": "anonymous-2"}, DataState.UNKNOWN, "session"),
        ({"session_key": None}, DataState.UNKNOWN, "session"),
        ({"consent_state": "accepted"}, DataState.UNKNOWN, "consent"),
        ({"consent_state": "unknown"}, DataState.UNKNOWN, "consent"),
        (
            {"observed_at": datetime(2026, 9, 2, 0, 15, 1, tzinfo=UTC)},
            DataState.UNKNOWN,
            "fifteen",
        ),
        ({"complete": False}, DataState.PARTIAL, "incomplete"),
        ({"complete": False, "truncated": True}, DataState.PARTIAL, "truncated"),
        ({"status_code": 403}, DataState.UNKNOWN, "403"),
        ({"status_code": 429}, DataState.UNKNOWN, "429"),
        ({"status_code": 500}, DataState.UNKNOWN, "500"),
        ({"final_url": "https://studio.example/other"}, DataState.UNKNOWN, "page"),
        ({"final_url": "https://studio.example/?region=other"}, DataState.UNKNOWN, "page"),
    ],
)
def test_incompatible_metadata_precedes_quote_validation(changes, state, reason):
    from ai_search_audit.content_diagnostics import compare_captures

    result = compare_captures(sample(), sample(kind="rendered", **changes), key_quotes=("",))
    assert result.state is state
    assert reason in " ".join(result.limitations).lower()
    assert result.rendered_only_quotes == ()


def test_exact_fifteen_minutes_and_canonical_www_upgrade_are_comparable():
    from ai_search_audit.content_diagnostics import compare_captures

    raw = sample(url="http://studio.example/service")
    rendered = sample(
        kind="rendered",
        url="https://www.studio.example/service",
        observed_at=raw.observed_at + timedelta(minutes=15),
    )
    assert compare_captures(raw, rendered).state is DataState.AVAILABLE


def test_dst_fold_uses_elapsed_minutes_and_matches_json_roundtrip():
    from zoneinfo import ZoneInfo

    from ai_search_audit.content_diagnostics import compare_captures

    warsaw = ZoneInfo("Europe/Warsaw")
    raw = sample(observed_at=datetime(2026, 10, 25, 2, 5, tzinfo=warsaw, fold=0))
    rendered = sample(
        kind="rendered", observed_at=datetime(2026, 10, 25, 2, 5, tzinfo=warsaw, fold=1)
    )
    original = compare_captures(raw, rendered)
    reloaded = compare_captures(
        type(raw).model_validate_json(raw.model_dump_json()),
        type(rendered).model_validate_json(rendered.model_dump_json()),
    )
    assert original.state is DataState.UNKNOWN
    assert original == reloaded
    assert "fifteen" in " ".join(original.limitations)


def test_missing_browser_is_unavailable_not_zero():
    from ai_search_audit.content_diagnostics import compare_captures

    result = compare_captures(sample(), None)
    assert result.state is DataState.UNAVAILABLE
    assert "browser" in " ".join(result.limitations).lower()
    assert "score" not in result.model_dump()


@pytest.mark.parametrize(
    "html",
    [
        "<html><title>Just a moment...</title><body>Checking your browser</body></html>",
        "<main><h1>Verify you are human</h1><p>Complete the CAPTCHA.</p></main>",
    ],
)
def test_challenge_is_unknown_not_javascript_evidence(html):
    from ai_search_audit.content_diagnostics import compare_captures

    result = compare_captures(sample(html), sample(kind="rendered"), key_quotes=("",))
    assert result.state is DataState.UNKNOWN
    assert "challenge" in " ".join(result.limitations).lower()


def test_extraction_recursion_error_becomes_explicit_failure():
    from ai_search_audit.content_diagnostics import compare_captures

    raw = sample("<main>" + "<div>" * 1100 + "deep" + "</div>" * 1100 + "</main>")
    assert raw.state is DataState.FAILED
    assert raw.extracted is None
    assert raw.text is None
    assert "extraction" in " ".join(raw.limitations).lower()
    result = compare_captures(raw, sample(kind="rendered"))
    assert result.state is DataState.FAILED
    assert result.rendered_only_quotes == ()


@pytest.mark.parametrize("complete", [False, True])
def test_unknown_http_status_does_not_invent_a_collection_failure(complete):
    from ai_search_audit.content_diagnostics import compare_captures

    raw = sample(status_code=None, complete=complete)
    assert raw.state is DataState.UNKNOWN
    assert raw.text is not None
    assert compare_captures(raw, sample(kind="rendered")).state is DataState.UNKNOWN


def test_explicit_collection_failure_remains_failed_in_pair():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import PageCaptureResult

    failure = PageCaptureResult(
        url="https://studio.example/",
        final_url=None,
        observed_at=datetime(2026, 9, 2, tzinfo=UTC),
        collector="native-crawler/0.1.1",
        status_code=None,
        html=None,
        content_type=None,
        complete=False,
        truncated=False,
        state=DataState.FAILED,
        limitations=("HTTP collection failed: ReadTimeout.",),
    )
    assert compare_captures(failure, sample(kind="rendered")).state is DataState.FAILED


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": datetime(2026, 9, 2)},
        {"complete": True, "truncated": True},
        {"kind": "browser"},
        {"status_code": 99},
        {"viewport": (0, 800)},
        {"url": "https://user:secret@studio.example/"},
        {"url": "file:///tmp/page.html"},
        {"url": "http://localhost/"},
        {"url": "http://127.0.0.1/"},
        {"url": "http://169.254.169.254/"},
        {"url": "http://192.0.2.1/"},
        {"url": "http://10.0.0.1/"},
        {"final_url": "https://unrelated.example/"},
        {"locale": ""},
        {"session_key": "person@example.com"},
    ],
)
def test_invalid_capture_metadata_is_rejected(changes):
    with pytest.raises(ValidationError):
        sample(**changes)


def test_intake_rejects_oversized_utf8_without_silent_truncation():
    with pytest.raises(ValueError, match="2 MiB"):
        sample("ą" * (1024 * 1024 + 1))


def test_unknown_browser_metadata_is_not_fabricated_and_hashes_are_trusted():
    import hashlib

    capture = sample()
    assert capture.url == capture.final_url == "https://studio.example/"
    assert capture.viewport is None and capture.collector is None
    assert capture.account_state == "anonymous"
    assert "html" not in capture.model_dump()
    assert capture.content_sha256 == hashlib.sha256(capture.text.encode()).hexdigest()
    assert sample() == capture
    assert sample(kind="rendered").capture_id != capture.capture_id
    with pytest.raises(ValidationError):
        capture.extracted.sections[0].text = "rewritten"
    with pytest.raises(ValidationError):
        type(capture).model_validate({**capture.model_dump(), "content_sha256": "0" * 64})
    with pytest.raises(ValidationError):
        type(capture).model_validate({**capture.model_dump(), "account_state": "authenticated"})


def test_input_boundary_rejects_private_account_state_and_oversize():
    from ai_search_audit.diagnostic_models import CaptureInput

    metadata = sample().model_dump(
        exclude={"extracted", "capture_id", "content_sha256", "state", "limitations"}
    )
    with pytest.raises(ValidationError):
        CaptureInput(**{**metadata, "account_state": "authenticated"}, html="<main>Private</main>")
    with pytest.raises(ValidationError, match="2 MiB"):
        CaptureInput(**metadata, html="x" * (2 * 1024 * 1024 + 1))


def test_explicit_key_passage_resolves_to_its_exact_capture_and_section():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import KeyPassage

    raw = sample("<main><h1>Studio</h1></main>")
    rendered = sample(kind="rendered")
    passage = KeyPassage(
        capture_id=rendered.capture_id,
        section_id=rendered.sections[0].section_id,
        quote="Support may take 2 days.",
    )
    result = compare_captures(raw, rendered, key_passages=(passage,))
    assert result.rendered_only_passages == (passage,)
    for changes in [
        {"capture_id": raw.capture_id},
        {"section_id": raw.sections[0].section_id},
        {"quote": "Invented claim"},
    ]:
        invalid = KeyPassage(**{**passage.model_dump(), **changes})
        with pytest.raises(ValueError, match="key passage does not resolve to rendered source"):
            compare_captures(raw, rendered, key_passages=(invalid,))


def test_key_passage_cannot_claim_text_from_another_section():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import KeyPassage

    rendered = sample(
        "<main><h1>First</h1><p>One.</p><h2>Second</h2><p>Two.</p></main>", kind="rendered"
    )
    passage = KeyPassage(
        capture_id=rendered.capture_id, section_id=rendered.sections[0].section_id, quote="Two."
    )
    with pytest.raises(ValueError, match="key passage does not resolve to rendered source"):
        compare_captures(sample(), rendered, key_passages=(passage,))


def test_retained_quote_length_is_bounded_and_includes_original_whitespace():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import KeyPassage

    rendered = sample("<main><p>" + "x" * 2000 + "</p></main>", kind="rendered")
    assert compare_captures(sample(), rendered, key_quotes=("x" * 2000,)).rendered_only_quotes == (
        "x" * 2000,
    )
    with pytest.raises(ValueError):
        compare_captures(sample(), rendered, key_quotes=("x" * 2000 + " ",))
    with pytest.raises(ValidationError):
        KeyPassage(
            capture_id=rendered.capture_id,
            section_id=rendered.sections[0].section_id,
            quote=" " * 20,
        )


def test_pair_dto_rejects_noncomparable_passages_and_mutable_metadata_shells():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import PairDiagnostic

    result = compare_captures(
        sample("<main>Intro</main>"),
        sample(kind="rendered"),
        key_quotes=("Support may take 2 days.",),
    )
    with pytest.raises(ValidationError):
        PairDiagnostic(**{**result.model_dump(), "state": DataState.UNKNOWN})
    assert isinstance(result.rendered_only_passages, tuple)
    assert isinstance(result.limitations, tuple)


def test_capture_dto_rejects_cross_capture_extraction():
    capture = sample()
    other = sample(kind="rendered")
    with pytest.raises(ValidationError):
        type(capture).model_validate(
            {**capture.model_dump(), "extracted": other.extracted.model_dump()}
        )


def test_reversed_raw_rendered_kinds_reject():
    from ai_search_audit.content_diagnostics import compare_captures

    with pytest.raises(ValueError, match="raw.*rendered"):
        compare_captures(sample(kind="rendered"), sample())


def test_quote_spanning_sections_resolves_to_whole_capture_not_an_arbitrary_section():
    from ai_search_audit.content_diagnostics import compare_captures
    from ai_search_audit.diagnostic_models import KeyPassage

    rendered = sample(
        "<main><h1>First</h1><p>One.</p><h2>Second</h2><p>Two.</p></main>", kind="rendered"
    )
    quote = "One. Second Two."
    result = compare_captures(sample(), rendered, key_quotes=(quote,))
    assert result.rendered_only_quotes == (quote,)
    passage = result.rendered_only_passages[0]
    assert passage.capture_id == rendered.capture_id
    assert passage.section_id is None
    assert compare_captures(sample(), rendered, key_passages=(passage,)).rendered_only_quotes == (
        quote,
    )
    for changes in [{"quote": "Invented quote"}, {"capture_id": sample().capture_id}]:
        invalid = KeyPassage(**{**passage.model_dump(), **changes})
        with pytest.raises(ValueError, match="key passage does not resolve to rendered source"):
            compare_captures(sample(), rendered, key_passages=(invalid,))


@pytest.mark.parametrize("state", [DataState.UNAVAILABLE, DataState.PARTIAL])
def test_capture_boundary_rejects_impossible_state_combinations(state):
    capture = sample()
    with pytest.raises(ValidationError):
        type(capture).model_validate(
            {**capture.model_dump(), "state": state, "limitations": ("Reason",)}
        )


def test_known_different_viewports_are_not_comparable():
    from ai_search_audit.content_diagnostics import compare_captures

    result = compare_captures(
        sample(viewport=(800, 600)), sample(kind="rendered", viewport=(1200, 800))
    )
    assert result.state is DataState.UNKNOWN
    assert "viewport" in " ".join(result.limitations).lower()


def test_inert_capture_never_loads_remote_assets(monkeypatch):
    import httpx

    def forbidden(*args, **kwargs):
        pytest.fail("inert capture tried to load remote content")

    monkeypatch.setattr(httpx.Client, "request", forbidden)
    capture = sample(
        '<main><p>Source</p><img src="https://other.example/image.png"></main>'
        '<script src="https://other.example/code.js"></script>'
        '<link href="https://other.example/style.css" rel="stylesheet">'
    )
    assert capture.text == "Source"
