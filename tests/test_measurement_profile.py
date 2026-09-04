"""Pure, source-bound measurement preparation: no execution or credentials."""

import hashlib
import importlib
import importlib.util
import json

import pytest
from pydantic import ValidationError

from ai_search_audit.benchmark import prepare_benchmark_worksheet
from ai_search_audit.diagnostic_models import DiagnosticBinding, DiagnosticSource, FrozenPrompt
from ai_search_audit.performance_http import HTTPRequestLimits
from tests.test_prompt_policy import _run

FABRICATED_CREDENTIAL = "synthetic-value"


def api():
    assert importlib.util.find_spec("ai_search_audit.measurement_profile") is not None, (
        "measurement profile and pure preflight must exist"
    )
    return importlib.import_module("ai_search_audit.measurement_profile")


def test_profile_defaults_are_bounded_and_do_not_authorize_paid_execution():
    profile = api().MeasurementProfile()
    assert profile.schema_version == "1.1.0"
    assert profile.max_pages == 3
    assert profile.max_prompts == 12
    assert profile.gemini is False
    assert profile.paid_use_consent is False
    assert profile.gemini_model is None
    assert 1 <= profile.http_limits.timeout_seconds <= 120
    assert 1 <= profile.http_limits.max_response_bytes <= 8388608


def test_legacy_profile_retains_version_and_identical_fields():
    legacy = api().MeasurementProfile(schema_version="1.0.0")
    current = api().MeasurementProfile()
    assert legacy.schema_version == "1.0.0"
    assert legacy.model_dump(exclude={"schema_version"}) == current.model_dump(
        exclude={"schema_version"}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0.0"},
        {"pagespeed_insights": 1},
        {"crux": "false"},
        {"lighthouse_local": "true"},
        {"gemini": 1},
        {"retry_transient": "true"},
        {"paid_use_consent": "false"},
        {"max_pages": 0},
        {"max_pages": 6},
        {"max_pages": True},
        {"max_prompts": 0},
        {"max_prompts": 13},
        {"max_prompts": "12"},
        {"http_limits": {"timeout_seconds": 121.0}},
        {"http_limits": {"timeout_seconds": float("nan")}},
        {"http_limits": {"max_response_bytes": 8388609}},
        {"http_limits": {"max_response_bytes": True}},
        {"gemini_model": ""},
        {"gemini_model": " "},
        {"gemini_model": b"gemini-2.5-flash"},
        {"gemini_model": "auto"},
        {"gemini_model": "LATEST"},
        {"gemini_model": "gemini-flash-latest"},
        {"gemini_model": "gemini-auto"},
        {"gemini_model": "https://api.example/models/gemini-2.5-flash"},
        {"gemini_model": "projects/account/models/gemini-2.5-flash"},
        {"gemini": True},
        {"pagespeed_insights": False, "lighthouse_local": True},
        {"api_key": FABRICATED_CREDENTIAL},
        {"account_id": "synthetic-account"},
        {"command": "arbitrary-command"},
        {"endpoint": "https://foreign.example"},
        {"repetitions": 3},
    ],
)
def test_profile_rejects_invalid_or_unsupported_configuration(changes):
    with pytest.raises(ValidationError):
        api().MeasurementProfile.model_validate(changes)


def test_profile_and_request_limits_are_frozen():
    profile = api().MeasurementProfile(gemini_model="gemini-2.5-flash")
    with pytest.raises(ValidationError):
        profile.max_pages = 5
    with pytest.raises(ValidationError):
        profile.http_limits.timeout_seconds = 3.0


def source(urls=("https://studio.example/",), *, prompts=()):
    return DiagnosticSource(
        binding=DiagnosticBinding(
            project_id="studio",
            source_version="v1",
            audit_id="audit-prompt-policy",
            report_locale="pl",
            domain="studio.example",
            source_sha256="a" * 64,
        ),
        prompts=prompts,
        page_urls=urls,
        canonical_domains=("studio.example",),
    )


def prepare(src=None, **settings):
    method = getattr(api(), "prepare_measurement_preflight", None)
    assert callable(method), "pure deterministic preflight must exist"
    return method(source() if src is None else src, api().MeasurementProfile(**settings))


def canonical_source(run=None):
    run = _run() if run is None else run
    raw = run.model_dump_json(indent=2).encode()
    src = source(
        tuple(str(page.url) for page in run.pages),
        prompts=tuple(FrozenPrompt.model_validate(p.model_dump()) for p in run.ai_prompts),
    )
    src = src.model_copy(
        update={
            "binding": src.binding.model_copy(
                update={"source_sha256": hashlib.sha256(raw).hexdigest()}
            )
        }
    )
    return src, raw


def test_page_selection_prioritizes_home_offer_information_then_audited_order():
    urls = (
        "https://studio.example/other",
        "https://studio.example/contact",
        "https://studio.example/services/design",
        "https://studio.example/",
        "https://studio.example/last",
    )
    result = prepare(source(urls))
    assert tuple(page.url for page in result.selected_pages) == (urls[3], urls[2], urls[1])
    assert "homepage" in result.selected_pages[0].selection_reason
    assert "URL-path heuristic" in result.selected_pages[1].selection_reason
    assert "URL-path heuristic" in result.selected_pages[2].selection_reason
    assert all(page.locale is None for page in result.selected_pages)
    assert tuple(page.url for page in prepare(source(urls), max_pages=5).selected_pages) == (
        urls[3],
        urls[2],
        urls[1],
        urls[0],
        urls[4],
    )


def test_selection_without_role_candidates_preserves_order_and_does_not_invent_home():
    urls = ("https://studio.example/en/page-b", "https://studio.example/pl/page-a")
    result = prepare(source(urls))
    assert tuple(p.url for p in result.selected_pages) == urls
    assert all(
        p.selection_reason == "Existing audited URL order; page role unknown."
        for p in result.selected_pages
    )
    assert all(p.locale is None for p in result.selected_pages)


def test_selection_deduplicates_normalized_identity_but_keeps_exact_first_audited_url():
    urls = (
        "HTTPS://STUDIO.EXAMPLE:443/services?a=1",
        "https://studio.example/services?a=1",
        "https://studio.example/services?a=2",
    )
    result = prepare(source(urls))
    assert tuple(p.url for p in result.selected_pages) == (urls[0], urls[2])


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://foreign.example/",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/",
        "https://user:password@studio.example/",
        "file:///etc/hosts",
        "https://studio.example/#fragment",
        "https://studio.example\\@foreign.example/",
    ],
)
def test_unsafe_or_foreign_audited_urls_are_not_selected(unsafe):
    result = prepare(source((unsafe, "https://studio.example/")))
    assert tuple(p.url for p in result.selected_pages) == ("https://studio.example/",)


def test_no_audited_pages_means_no_invented_pages_or_performance_attempts():
    result = prepare(source(()))
    assert result.selected_pages == ()
    assert result.maximum_attempts.total == 0


@pytest.mark.parametrize("retry,expected", [(True, (12, 24, 6, 42)), (False, (6, 12, 6, 24))])
def test_maximum_attempt_budget_counts_provider_scopes_and_devices(retry, expected):
    result = prepare(
        source(tuple(f"https://studio.example/{n}" for n in range(3))),
        retry_transient=retry,
        lighthouse_local=True,
    )
    budget = result.maximum_attempts
    assert (
        budget.pagespeed_insights,
        budget.crux,
        budget.lighthouse_local,
        budget.total,
    ) == expected
    assert budget.gemini == 0
    assert result.devices == ("mobile", "desktop")


def test_disabled_modules_contribute_zero():
    result = prepare(pagespeed_insights=False, crux=False)
    assert result.maximum_attempts.total == 0


def test_preflight_is_deterministic_deeply_frozen_and_detached():
    src, _ = canonical_source()
    result = prepare(src)
    assert result == prepare(src)
    assert result.binding == src.binding and result.binding is not src.binding
    assert result.prompts[0] == src.prompts[0] and result.prompts[0] is not src.prompts[0]
    for obj, field, value in (
        (result, "devices", ("mobile",)),
        (result.binding, "domain", "foreign.example"),
        (result.selected_pages[0], "locale", "pl"),
        (result.prompts[0], "text", "changed"),
        (result.profile, "max_pages", 5),
        (result.profile.http_limits, "timeout_seconds", 3.0),
        (result.maximum_attempts, "gemini", 999),
    ):
        with pytest.raises(ValidationError):
            setattr(obj, field, value)
    assert isinstance(result.prompts[0].query_themes, tuple)


@pytest.mark.parametrize("change", ["profile", "limits", "source", "binding", "prompt"])
def test_preparation_revalidates_unchecked_model_copy_changes(change):
    src, _ = canonical_source()
    profile = api().MeasurementProfile()
    if change == "profile":
        profile = profile.model_copy(update={"max_pages": 99})
    elif change == "limits":
        profile = profile.model_copy(
            update={
                "http_limits": HTTPRequestLimits().model_copy(update={"timeout_seconds": 999.0})
            }
        )
    elif change == "source":
        src = src.model_copy(update={"unexpected": "value"})
    elif change == "binding":
        src = src.model_copy(
            update={"binding": src.binding.model_copy(update={"source_sha256": "wrong"})}
        )
    else:
        src = src.model_copy(
            update={"prompts": (src.prompts[0].model_copy(update={"text": None}),)}
        )
    with pytest.raises(ValueError):
        api().prepare_measurement_preflight(src, profile)


def test_preparation_rejects_unchecked_bytes_model_without_json_type_coercion():
    profile = api().MeasurementProfile().model_copy(update={"gemini_model": b"gemini-2.5-flash"})
    with pytest.raises(ValidationError, match="gemini_model"):
        api().prepare_measurement_preflight(source(), profile)


def test_preparation_rejects_binding_outside_canonical_domains():
    src = source().model_copy(update={"canonical_domains": ("foreign.example",)})
    with pytest.raises(ValueError, match="canonical"):
        prepare(src)


@pytest.mark.parametrize(
    "consent,attempts,reason", [(False, 0, "paid_use_not_authorized"), (True, 4, None)]
)
def test_gemini_subset_is_stable_and_full_pack_denominator_is_preserved(consent, attempts, reason):
    src, raw = canonical_source()
    profile = api().MeasurementProfile(
        gemini=True, gemini_model="gemini-2.5-flash", paid_use_consent=consent, max_prompts=4
    )
    result = api().prepare_measurement_preflight(src, profile, audit_json=raw)
    assert result.prompts == src.prompts[:4]
    assert result.full_prompt_count == len(src.prompts) == 12
    assert result.maximum_attempts.gemini == attempts
    assert result.gemini_blocked_reason == reason


def test_larger_legacy_pack_subset_does_not_rewrite_or_shrink_denominator_when_disabled():
    src, _ = canonical_source()
    extra = tuple(p.model_copy(update={"prompt_id": f"extra-{p.prompt_id}"}) for p in src.prompts)
    src = src.model_copy(update={"prompts": src.prompts + extra})
    before = src.model_dump_json()
    result = prepare(src)
    assert result.prompts == src.prompts[:12]
    assert result.full_prompt_count == 24
    assert src.model_dump_json() == before


def test_legacy_prompt_version_remains_usable_for_performance_only():
    src, _ = canonical_source()
    src = src.model_copy(
        update={
            "prompts": tuple(p.model_copy(update={"pack_version": "1.1.0"}) for p in src.prompts)
        }
    )
    result = prepare(src)
    assert result.prompts == src.prompts
    assert result.maximum_attempts.pagespeed_insights > 0
    assert result.maximum_attempts.gemini == 0


@pytest.mark.parametrize("version", ["1.1.0", "2.0.0"])
def test_preflight_full_pack_hash_matches_existing_benchmark_worksheet(version):
    src, _ = canonical_source()
    src = src.model_copy(
        update={
            "prompts": tuple(p.model_copy(update={"pack_version": version}) for p in src.prompts)
        }
    )
    result = prepare(src, max_prompts=4)
    assert result.full_prompt_pack_hash == prepare_benchmark_worksheet(src).pack_content_hash


@pytest.mark.parametrize("reason", ["", "   ", "x" * 201])
def test_page_selection_reason_requires_bounded_nonblank_text(reason):
    with pytest.raises(ValidationError):
        api().MeasurementSelectedPage(url="https://studio.example/", selection_reason=reason)


@pytest.mark.parametrize(
    "change",
    ["missing", "hash", "identity", "pages", "prompts", "quality", "duplicate", "nonfinite"],
)
def test_gemini_requires_exact_canonical_bytes_and_automatic_prompt_quality(change):
    run = _run()
    src, raw = canonical_source(run)
    if change == "missing":
        raw = None
    elif change == "hash":
        raw += b" "
    elif change in {"identity", "pages", "prompts"}:
        if change == "identity":
            src = src.model_copy(
                update={"binding": src.binding.model_copy(update={"audit_id": "another-audit"})}
            )
        elif change == "pages":
            src = src.model_copy(update={"page_urls": ("https://studio.example/other",)})
        else:
            src = src.model_copy(update={"prompts": src.prompts[:-1]})
    else:
        if change == "quality":
            run.ai_prompts[0].text = "Unapproved changed question"
            src, raw = canonical_source(run)
        elif change == "duplicate":
            raw = raw.replace(b'"audit_id":', b'"audit_id": "ignored", "audit_id":', 1)
        else:
            payload = json.loads(raw)
            payload["configuration"]["bad"] = float("nan")
            raw = json.dumps(payload).encode()
        src = src.model_copy(
            update={
                "binding": src.binding.model_copy(
                    update={"source_sha256": hashlib.sha256(raw).hexdigest()}
                )
            }
        )
    with pytest.raises(ValueError):
        api().prepare_measurement_preflight(
            src,
            api().MeasurementProfile(
                gemini=True, gemini_model="gemini-2.5-flash", paid_use_consent=True
            ),
            audit_json=raw,
        )


def test_preparation_does_not_read_credentials_network_run_processes_or_write(monkeypatch):
    import builtins
    import os
    import socket
    import subprocess

    src, raw = canonical_source()
    profile = api().MeasurementProfile(gemini=True, gemini_model="gemini-2.5-flash")

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must have no I/O or credential lookup")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    # Pydantic dynamically compiles PublicURL and checks its own plugin switch.
    # Permit only that library setting, never an operator credential lookup.
    monkeypatch.setattr(
        os,
        "getenv",
        lambda name, *args: None if name == "PYDANTIC_DISABLE_PLUGINS" else forbidden(),
    )
    result = api().prepare_measurement_preflight(src, profile, audit_json=raw)
    assert result.maximum_attempts.gemini == 0
