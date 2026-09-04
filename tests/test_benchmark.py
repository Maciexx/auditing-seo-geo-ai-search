import hashlib
import json
import socket

import pytest
from pydantic import BaseModel, ValidationError

from ai_search_audit.diagnostic_models import DiagnosticBinding, DiagnosticSource, FrozenPrompt


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    from ai_search_audit.prompts import generate_prompt_pack

    def forbidden(*args, **kwargs):
        pytest.fail("benchmark preparation must not use network or regenerate prompts")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("ai_search_audit.prompts.generate_prompt_pack", forbidden)
    yield generate_prompt_pack


def _source(locale="pl", business="studio"):
    return DiagnosticSource(
        binding=DiagnosticBinding(
            project_id="example",
            source_version="public-v1",
            audit_id="audit-1",
            report_locale=locale,
            domain=f"{business}.example",
            source_sha256="a" * 64,
        ),
        prompts=tuple(
            FrozenPrompt(
                prompt_id=f"p-{language}",
                pack_version="1.1.0",
                locale=language,
                intent="discovery",
                text=text,
                target_entities=(business,),
                query_themes=(business,),
                expected_evidence_needs=("public source",),
                suggested_providers=("manual",),
            )
            for language, text in (("pl", f"Co oferuje {business}?"), ("en", f"Which {business}?"))
        ),
        page_urls=(f"https://{business}.example/",),
        canonical_domains=(f"{business}.example",),
    )


def _setup(**changes):
    from ai_search_audit.diagnostic_models import BenchmarkSetup

    fields = dict(
        provider="Example AI",
        product="Example Search",
        model_id="example-1.0",
        interface="consumer_ui",
        search_mode="enabled",
        locale="en",
        market="PL",
        account_state="anonymous",
        reset_method="new_conversation",
    )
    fields.update(changes)
    return BenchmarkSetup(**fields)


def test_canonical_hash_covers_text_and_ignores_mapping_order():
    from ai_search_audit.benchmark import canonical_hash

    left = {"version": "1.1.0", "prompts": [{"id": "p1", "text": "Which studio?"}]}
    right = {"version": "1.1.0", "prompts": [{"id": "p1", "text": "Which shop?"}]}

    assert canonical_hash(left) != canonical_hash(right)
    assert canonical_hash(left) == canonical_hash(dict(reversed(list(left.items()))))


def test_missing_prompts_do_not_count_as_negative_answers():
    from ai_search_audit.benchmark import sample_metrics

    result = sample_metrics(expected=4, mentioned=(True, False), cited=(False, False))
    assert result.mention_rate == 50.0
    assert result.citation_rate == 0.0
    assert result.coverage == 0.5
    assert result.measured == 2
    assert result.expected == 4


@pytest.mark.parametrize(
    "mentioned,cited,state,mention_rate,citation_rate,citation_count",
    [
        ((), (), "UNAVAILABLE", None, None, 0),
        ((False, False), (False, False), "AVAILABLE", 0.0, 0.0, 2),
        ((True, False), (False, None), "PARTIAL", 50.0, 0.0, 1),
        ((True,), (None,), "PARTIAL", 100.0, None, 0),
    ],
)
def test_sample_metrics_counts_only_known_observations(
    mentioned, cited, state, mention_rate, citation_rate, citation_count
):
    from ai_search_audit.benchmark import sample_metrics

    result = sample_metrics(expected=2, mentioned=mentioned, cited=cited)
    assert result.state == state
    assert result.mention_rate == mention_rate
    assert result.citation_rate == citation_rate
    assert result.citation_measured == citation_count
    assert result.citation_coverage == citation_count / 2
    assert result.confidence == result.coverage == len(mentioned) / 2
    assert result.limitations == (
        ("Confidence describes this observed sample only.",)
        if mentioned
        else ("No grounded completed responses.",)
    )


@pytest.mark.parametrize(
    "expected,mentioned,cited",
    [
        (0, (), ()),
        (1, (True,), ()),
        (1, (True, False), (True, False)),
        (True, (), ()),
        (1.0, (), ()),
        (1, (1,), (False,)),
        (1, (True,), (0,)),
    ],
)
def test_sample_metrics_rejects_invalid_denominators_and_flags(expected, mentioned, cited):
    from ai_search_audit.benchmark import sample_metrics

    with pytest.raises(ValueError):
        sample_metrics(expected=expected, mentioned=mentioned, cited=cited)


@pytest.mark.parametrize("state", ["UNKNOWN", "UNAVAILABLE", "FAILED"])
@pytest.mark.parametrize("rate", ["mention_rate", "citation_rate"])
def test_nonnumeric_benchmark_metrics_cannot_contain_rates(state, rate):
    from ai_search_audit.diagnostic_models import BenchmarkMetrics

    with pytest.raises(ValidationError):
        BenchmarkMetrics.model_validate(
            dict(
                state=state,
                expected=2,
                measured=0,
                citation_measured=0,
                coverage=0,
                citation_coverage=0,
                confidence=0,
                **{rate: 0.0},
            )
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), object(), {1, 2}])
def test_canonical_hash_rejects_non_json_values(value):
    from ai_search_audit.benchmark import canonical_hash

    with pytest.raises((TypeError, ValueError)):
        canonical_hash({"value": value})


def test_canonical_hash_uses_exact_utf8_canonical_serialization():
    from ai_search_audit.benchmark import canonical_hash

    assert (
        canonical_hash({"z": "żółć", "a": [1]})
        == hashlib.sha256('{"a":[1],"z":"żółć"}'.encode()).hexdigest()
    )


@pytest.mark.parametrize("locale", ["pl", "en"])
@pytest.mark.parametrize("business", ["studio", "shop"])
def test_worksheet_copies_exact_bilingual_prompts_without_credentials(locale, business):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    source = _source(locale, business)
    before = source.model_dump_json()
    worksheet = prepare_benchmark_worksheet(source)

    assert worksheet.schema_version == "1.0.0"
    assert worksheet.binding == source.binding
    assert worksheet.prompts == source.prompts
    assert {prompt.locale for prompt in worksheet.prompts} == {"pl", "en"}
    assert worksheet.pack_version == "1.1.0"
    assert worksheet.setup is None
    assert worksheet.setup_fingerprint is None
    assert "observations" not in worksheet.model_dump()
    assert "score" not in worksheet.model_dump()
    assert source.model_dump_json() == before


def test_fixed_instructions_follow_report_locale_only():
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    polish = prepare_benchmark_worksheet(_source("pl"))
    english = prepare_benchmark_worksheet(_source("en"))
    assert polish.prompts == english.prompts
    assert polish.pack_content_hash == english.pack_content_hash
    assert polish.instructions != english.instructions
    assert "prompt" in " ".join(polish.instructions).lower()
    assert "prompt" in " ".join(english.instructions).lower()
    assert polish.instructions == prepare_benchmark_worksheet(_source("pl", "shop")).instructions


def test_setup_fingerprint_is_computed_from_all_validated_settings():
    from ai_search_audit.benchmark import canonical_hash, prepare_benchmark_worksheet

    setup = _setup()
    worksheet = prepare_benchmark_worksheet(_source(), setup)
    assert worksheet.setup_fingerprint == canonical_hash(setup.model_dump(mode="json"))
    assert (
        worksheet.setup_fingerprint
        != prepare_benchmark_worksheet(_source(), _setup(interface="api")).setup_fingerprint
    )
    assert (
        worksheet.setup_fingerprint
        == prepare_benchmark_worksheet(_source("en", "shop"), setup).setup_fingerprint
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "Other AI"),
        ("product", "Other Search"),
        ("model_id", "example-2.0"),
        ("interface", "api"),
        ("search_mode", "disabled"),
        ("locale", "pl"),
        ("market", "GB"),
        ("account_state", "signed_in_paid"),
        ("reset_method", "temporary_conversation"),
    ],
)
def test_every_critical_dimension_changes_the_setup_fingerprint(field, value):
    from ai_search_audit.benchmark import benchmark_setup_fingerprint

    assert benchmark_setup_fingerprint(_setup()) != benchmark_setup_fingerprint(
        _setup(**{field: value})
    )


def test_all_settings_can_be_unknown_without_inventing_model_or_failed_observation():
    from ai_search_audit.benchmark import benchmark_setup_fingerprint, prepare_benchmark_worksheet
    from ai_search_audit.diagnostic_models import BenchmarkSetup

    setup = BenchmarkSetup()
    assert all(value is None for value in setup.model_dump().values())
    assert benchmark_setup_fingerprint(setup) is None
    worksheet = prepare_benchmark_worksheet(_source(), setup)
    assert worksheet.setup.model_id is None
    assert worksheet.setup_fingerprint is None
    assert "state" not in worksheet.model_dump()


@pytest.mark.parametrize(
    "field",
    [
        "provider",
        "product",
        "model_id",
        "interface",
        "search_mode",
        "locale",
        "market",
        "account_state",
        "reset_method",
    ],
)
def test_any_unknown_critical_setup_field_prevents_comparable_fingerprint(field):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    worksheet = prepare_benchmark_worksheet(_source(), _setup(**{field: None}))
    assert worksheet.setup_fingerprint is None
    assert getattr(worksheet.setup, field) is None


@pytest.mark.parametrize(
    "field", ["provider", "product", "model_id", "locale", "market", "reset_method"]
)
def test_blank_critical_settings_are_rejected_not_hashed_as_known(field):
    with pytest.raises(ValidationError):
        _setup(**{field: "   "})


@pytest.mark.parametrize(
    "field", ["provider", "product", "model_id", "locale", "market", "reset_method"]
)
@pytest.mark.parametrize("placeholder", ["unknown", "UNAVAILABLE", " N/A "])
def test_explicit_unknown_placeholders_require_null_not_a_comparable_setting(field, placeholder):
    from ai_search_audit.benchmark import benchmark_setup_fingerprint

    assert benchmark_setup_fingerprint(_setup(**{field: None})) is None
    with pytest.raises(ValidationError, match="null"):
        benchmark_setup_fingerprint(_setup(**{field: placeholder}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_state", "someone@studio.example"),
        ("account_state", "user-123"),
        ("account_state", "unknown"),
        ("interface", "browser"),
        ("search_mode", "unknown"),
        ("settings", {"email": "someone@studio.example"}),
        ("account_id", "user-123"),
        ("setup_fingerprint", "a" * 64),
        ("report_status", "CLIENT_VALIDATED"),
        ("project_id", "other"),
        ("observed_at", "2026-09-03"),
        ("response", "answer"),
    ],
)
def test_setup_rejects_identity_arbitrary_settings_and_untrusted_hashes(field, value):
    with pytest.raises(ValidationError):
        _setup(**{field: value})


@pytest.mark.parametrize("change", ["duplicate", "mixed_versions", "empty"])
def test_preparation_rejects_invalid_prompt_packs(change):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    source = _source()
    payload = source.model_dump(mode="json")
    if change == "duplicate":
        payload["prompts"][1]["prompt_id"] = payload["prompts"][0]["prompt_id"]
    elif change == "mixed_versions":
        payload["prompts"][1]["pack_version"] = "1.2.0"
    else:
        payload["prompts"] = []
    with pytest.raises(ValueError, match="prompt|pack"):
        prepare_benchmark_worksheet(DiagnosticSource.model_validate(payload))


@pytest.mark.parametrize("field", ["prompt_id", "pack_version", "locale", "intent", "text"])
def test_preparation_rejects_blank_required_prompt_values(field):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    payload = _source().model_dump(mode="json")
    payload["prompts"] = payload["prompts"][:1]
    for prompt in payload["prompts"]:
        prompt[field] = "  "
    with pytest.raises(ValueError, match="prompt|pack"):
        prepare_benchmark_worksheet(DiagnosticSource.model_validate(payload))


def test_changed_text_with_same_ids_and_version_changes_pack_hash():
    from ai_search_audit.benchmark import prepare_benchmark_worksheet

    source = _source()
    payload = source.model_dump(mode="json")
    payload["prompts"][0]["text"] = "Changed canonical question?"
    changed = DiagnosticSource.model_validate(payload)
    assert (
        prepare_benchmark_worksheet(source).pack_content_hash
        != prepare_benchmark_worksheet(changed).pack_content_hash
    )


def test_outputs_are_deeply_frozen_and_json_round_trips():
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet
    from ai_search_audit.diagnostic_models import BenchmarkWorksheet

    source = _source()
    worksheet = prepare_benchmark_worksheet(source, _setup())

    def frozen_tree(value):
        assert not isinstance(value, (dict, list, set))
        if isinstance(value, BaseModel):
            assert value.model_config["frozen"]
            assert value.model_config["extra"] == "forbid"
            for field in type(value).model_fields:
                frozen_tree(getattr(value, field))
        elif isinstance(value, tuple):
            for item in value:
                frozen_tree(item)

    frozen_tree(worksheet)
    assert BenchmarkWorksheet.model_validate_json(worksheet.model_dump_json()) == worksheet
    assert (
        validate_benchmark_worksheet(source, json.loads(worksheet.model_dump_json())) == worksheet
    )
    with pytest.raises(ValidationError):
        worksheet.setup.model_id = "changed"
    with pytest.raises(ValidationError):
        worksheet.prompts[0].text = "changed"
    with pytest.raises(TypeError):
        worksheet.prompts[0].target_entities[0] = "changed"
    with pytest.raises(TypeError):
        worksheet.instructions[0] = "changed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("pack_content_hash", "b" * 64),
        ("setup_fingerprint", "b" * 64),
        ("schema_version", "2.0.0"),
        ("report_status", "CLIENT_VALIDATED"),
        ("project_id", "other"),
        ("settings", {}),
    ],
)
def test_stored_worksheet_rejects_forged_hashes_and_injected_fields(field, value):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet

    source = _source()
    payload = json.loads(prepare_benchmark_worksheet(source, _setup()).model_dump_json())
    payload[field] = value
    with pytest.raises(ValueError):
        validate_benchmark_worksheet(source, payload)


@pytest.mark.parametrize(
    "field",
    [
        "project_id",
        "source_version",
        "audit_id",
        "report_locale",
        "domain",
        "source_sha256",
    ],
)
def test_import_rejects_binding_mutation_even_with_self_consistent_hashes(field):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet

    source = _source()
    changed = source.model_dump(mode="json")
    changed["binding"][field] = {
        "report_locale": "en",
        "source_sha256": "b" * 64,
        "domain": "other.example",
    }.get(field, "other")
    forged = prepare_benchmark_worksheet(DiagnosticSource.model_validate(changed))
    with pytest.raises(ValueError, match="canonical|source|binding"):
        validate_benchmark_worksheet(source, json.loads(forged.model_dump_json()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "Which shop?"),
        ("locale", "de"),
        ("intent", "changed"),
        ("target_entities", ["other"]),
        ("prompt_id", "other"),
    ],
)
def test_import_rejects_changed_prompt_content_even_with_recomputed_hash(field, value):
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet

    source = _source()
    changed = source.model_dump(mode="json")
    changed["prompts"][0][field] = value
    forged = prepare_benchmark_worksheet(DiagnosticSource.model_validate(changed))
    with pytest.raises(ValueError, match="canonical|source|prompt"):
        validate_benchmark_worksheet(source, json.loads(forged.model_dump_json()))


def test_import_rejects_mutated_stored_prompt_and_instructions():
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet

    source = _source()
    worksheet = prepare_benchmark_worksheet(source)
    changed_prompt = json.loads(worksheet.model_dump_json())
    changed_prompt["prompts"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_benchmark_worksheet(source, changed_prompt)
    changed_instructions = json.loads(worksheet.model_dump_json())
    changed_instructions["instructions"] = ["Declare success without testing."]
    with pytest.raises(ValueError, match="instructions"):
        validate_benchmark_worksheet(source, changed_instructions)


def test_validation_rechecks_model_copy_instead_of_trusting_frozen_instance():
    from ai_search_audit.benchmark import (
        benchmark_setup_fingerprint,
        prepare_benchmark_worksheet,
        validate_benchmark_worksheet,
    )

    source = _source()
    worksheet = prepare_benchmark_worksheet(source, _setup())
    bypassed = worksheet.model_copy(update={"setup_fingerprint": "b" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        validate_benchmark_worksheet(source, bypassed)
    with pytest.raises(ValidationError):
        benchmark_setup_fingerprint(_setup().model_copy(update={"account_state": "user-123"}))


def test_preparation_keeps_validated_project_files_and_identity_unchanged(
    tmp_path, monkeypatch, no_network
):
    import ai_search_audit.prompts as prompts
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_worksheet
    from ai_search_audit.diagnostic_sources import load_diagnostic_source
    from tests.test_project_orchestrator import _create

    # The existing synthetic project fixture generates its canonical pack before this task.
    with monkeypatch.context() as context:
        context.setattr(prompts, "generate_prompt_pack", no_network)
        context.setattr("ai_search_audit.orchestrator.generate_prompt_pack", no_network)
        _create(tmp_path, domain="https://studio.example", report_locale="pl")
    root = tmp_path / "clients"
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    source = load_diagnostic_source(
        "project:example", clients_root=root, source_version="public-v1"
    )
    worksheet = prepare_benchmark_worksheet(source)
    assert validate_benchmark_worksheet(source, worksheet) == worksheet
    assert worksheet.binding == source.binding
    assert worksheet.prompts == source.prompts
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def _response(source=None, prompt_index=0, **changes):
    from ai_search_audit.diagnostic_models import BenchmarkResponseInput

    source = source or _source()
    prompt = source.prompts[prompt_index]
    fields = dict(
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.text,
        observed_at="2026-09-03T12:00:00Z",
        response_text="The studio offers public examples.",
        grounded=True,
        brand_mentioned=True,
        citations=(),
        citations_complete=True,
        complete=True,
        response_truncated=False,
        inspection_scope="full_response",
    )
    fields.update(changes)
    return BenchmarkResponseInput.model_validate(fields)


def _sample(source=None, responses=None, setup=None):
    from ai_search_audit.benchmark import (
        prepare_benchmark_worksheet,
        validate_benchmark_responses,
    )

    source = source or _source()
    if responses is None:
        responses = (_response(source),)
    return validate_benchmark_responses(
        source, prepare_benchmark_worksheet(source, setup), responses
    )


@pytest.mark.parametrize("locale", ["pl", "en"])
@pytest.mark.parametrize("business", ["studio", "shop"])
def test_validated_sample_uses_full_pack_and_separates_mentions_citations(locale, business):
    source = _source(locale, business)
    result = _sample(source, (_response(source),))
    assert result.metrics.expected == 2
    assert result.metrics.measured == 1
    assert result.metrics.mention_rate == 100.0
    assert result.metrics.citation_rate == 0.0
    assert result.metrics.coverage == 0.5
    assert result.measurable_prompt_ids == ("p-pl",)
    assert result.citation_measurable_prompt_ids == ("p-pl",)
    assert result.worksheet.binding == source.binding


def test_validated_missing_sample_is_unavailable_not_zero():
    result = _sample(responses=())
    assert result.metrics.state == "UNAVAILABLE"
    assert result.metrics.mention_rate is result.metrics.citation_rate is None
    assert result.metrics.expected == 2
    assert result.metrics.measured == 0
    assert result.responses == ()


@pytest.mark.parametrize(
    "changes",
    [{"grounded": False}, {"grounded": None}, {"brand_mentioned": None}, {"complete": False}],
)
def test_nonmeasurable_response_remains_evidence_without_becoming_a_negative(changes):
    result = _sample(responses=(_response(**changes),))
    assert len(result.responses) == 1
    assert result.metrics.measured == 0
    assert result.metrics.mention_rate is result.metrics.citation_rate is None


def test_completed_grounded_brand_absence_is_a_measured_zero():
    result = _sample(responses=(_response(brand_mentioned=False),))
    assert result.metrics.measured == 1
    assert result.metrics.mention_rate == result.metrics.citation_rate == 0.0


@pytest.mark.parametrize(
    "citations,complete,rate,count",
    [
        ((), False, None, 0),
        ((), True, 0.0, 1),
        (("https://STUDIO.example./source",), True, 100.0, 1),
        (("https://studio.example/source",), False, 100.0, 1),
        (("https://studio.example.evil.example/",), True, 0.0, 1),
        (("https://notstudio.example/",), True, 0.0, 1),
        (("https://www.studio.example/",), True, 0.0, 1),
        (("https://elsewhere.example/studio.example",), False, None, 0),
    ],
)
def test_citation_measurement_requires_exact_approved_host_or_completed_inspection(
    citations, complete, rate, count
):
    result = _sample(responses=(_response(citations=citations, citations_complete=complete),))
    assert result.metrics.citation_rate == rate
    assert result.metrics.citation_measured == count


def test_citation_domains_are_normalized_idna_and_explicitly_approved():
    changed = _source().model_dump(mode="json")
    changed["canonical_domains"] = ["studio.example", "ŻÓŁĆ.example."]
    source = DiagnosticSource.model_validate(changed)
    result = _sample(source, (_response(source, citations=("http://żółć.example/path",)),))
    assert result.metrics.citation_rate == 100.0


@pytest.mark.parametrize("approved", ["straße.example", "xn--strae-oqa.example"])
@pytest.mark.parametrize(
    "citation,rate",
    [
        ("https://straße.example/source", 100.0),
        ("https://xn--strae-oqa.example/source", 100.0),
        ("https://strasse.example/source", 0.0),
        ("https://www.straße.example/source", 0.0),
    ],
)
def test_citation_idna_policy_matches_unicode_punycode_but_not_distinct_ascii(
    approved, citation, rate
):
    from ai_search_audit.diagnostic_models import BenchmarkSample

    changed = _source().model_dump(mode="json")
    changed["canonical_domains"] = [approved]
    source = DiagnosticSource.model_validate(changed)
    result = _sample(source, (_response(source, citations=(citation,)),))
    assert result.approved_domains == ("xn--strae-oqa.example",)
    assert result.metrics.citation_rate == rate
    assert BenchmarkSample.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "domain",
    [
        "https://studio.example",
        "studio.example/path",
        "studio.example?query",
        "studio.example#fragment",
        "user@studio.example",
        "@studio.example",
        "studio.example:443",
        "studio.example\\path",
        " studio.example",
        "%73tudio.example",
    ],
)
def test_approved_citation_domains_accept_only_bare_hosts(domain):
    changed = _source().model_dump(mode="json")
    changed["canonical_domains"] = [domain]
    source = DiagnosticSource.model_validate(changed)
    with pytest.raises(ValueError, match="host|domain"):
        _sample(source)


@pytest.mark.parametrize(
    "url",
    [
        "studio.example",
        "//studio.example/",
        "ftp://studio.example/",
        "https://",
        "https://user:secret@studio.example/",
        "https://user@studio.example/",
        "https://@studio.example/",
        "https://:@studio.example/",
        "https://studio.example:99999/",
        "https://studio.example:bad/",
        "https://studio.example\\@evil.example/",
        " https://studio.example/",
        "https://studio.example/\nsource",
    ],
)
def test_response_rejects_malformed_and_credential_citation_urls(url):
    with pytest.raises(ValidationError):
        _response(citations=(url,))


def test_response_rejects_duplicate_prompt_records():
    response = _response()
    with pytest.raises(ValueError, match="duplicate"):
        _sample(responses=(response, response))


@pytest.mark.parametrize("changes", [{"prompt_id": "missing"}, {"prompt_text": "Other prompt"}])
def test_response_must_resolve_exact_canonical_prompt(changes):
    with pytest.raises(ValueError, match="prompt"):
        _sample(responses=(_response(**changes),))


def test_responses_require_source_bound_worksheet_not_merely_self_consistent_hashes():
    from ai_search_audit.benchmark import prepare_benchmark_worksheet, validate_benchmark_responses

    changed = _source().model_dump(mode="json")
    changed["prompts"][0]["text"] = "Forged prompt"
    forged = DiagnosticSource.model_validate(changed)
    with pytest.raises(ValueError, match="canonical|source|prompt"):
        validate_benchmark_responses(
            _source(), prepare_benchmark_worksheet(forged), (_response(forged),)
        )


@pytest.mark.parametrize("hash_value", ["a" * 64, "b" * 64])
def test_processing_rejects_forged_digest_even_for_a_frozen_input(hash_value):
    response = _response().model_copy(update={"response_hash": hash_value})
    with pytest.raises(ValueError, match="hash"):
        _sample(responses=(response,))


def test_response_hashes_are_derived_from_actual_text_and_tampering_fails():
    response = _response()
    actual_hash = hashlib.sha256(response.response_text.encode()).hexdigest()
    result = _sample(responses=(response.model_copy(update={"response_hash": actual_hash}),))
    observation = result.responses[0]
    assert observation.response_hash == observation.source_response_hash == actual_hash
    assert observation.excerpt_hash == actual_hash
    assert observation.response_excerpt == response.response_text
    assert observation.excerpt_truncated is False
    with pytest.raises(ValueError, match="hash"):
        _sample(
            responses=(
                response.model_copy(
                    update={
                        "response_hash": actual_hash,
                        "response_text": "Tampered response",
                    }
                ),
            )
        )


def test_long_response_retains_bounded_excerpt_distinct_hashes_and_recorded_inspection():
    from ai_search_audit.diagnostic_models import MAX_BENCHMARK_EXCERPT_CHARS

    text = "A" * (MAX_BENCHMARK_EXCERPT_CHARS + 50)
    result = _sample(responses=(_response(response_text=text, brand_mentioned=False),))
    observation = result.responses[0]
    assert observation.response_excerpt == text[:MAX_BENCHMARK_EXCERPT_CHARS]
    assert (
        observation.response_hash
        == observation.source_response_hash
        == hashlib.sha256(text.encode()).hexdigest()
    )
    assert (
        observation.excerpt_hash
        == hashlib.sha256(observation.response_excerpt.encode()).hexdigest()
    )
    assert observation.excerpt_hash != observation.source_response_hash
    assert observation.excerpt_truncated is True
    assert observation.inspection_scope == "full_response"
    assert result.metrics.mention_rate == 0.0
    assert "response_text" not in observation.model_dump()
    assert text not in result.model_dump_json()


def test_retained_benchmark_excerpt_has_literal_2000_character_boundary():
    observation = _sample(responses=(_response(response_text="a" * 2001),)).responses[0]
    assert len(observation.response_excerpt) == 2000
    assert observation.excerpt_truncated is True


def test_benchmark_raw_capture_limit_counts_utf8_bytes_not_unicode_characters():
    assert len(_response(response_text="ż" * (1024 * 1024)).response_text) == 1024 * 1024
    with pytest.raises(ValidationError, match="2 MiB"):
        _response(response_text="ż" * (1024 * 1024 + 1))


def test_whitespace_only_response_is_not_actual_retained_evidence():
    with pytest.raises(ValidationError, match="blank"):
        _response(response_text=" \n\t ")


@pytest.mark.parametrize("mentioned,measured", [(False, 0), (True, 1), (None, 0)])
def test_truncated_capture_cannot_prove_absence_from_an_excerpt(mentioned, measured):
    result = _sample(
        responses=(
            _response(
                response_text="Captured excerpt",
                response_truncated=True,
                inspection_scope="excerpt",
                brand_mentioned=mentioned,
            ),
        )
    )
    observation = result.responses[0]
    assert observation.source_response_hash is None
    assert observation.response_hash == observation.excerpt_hash
    assert observation.excerpt_truncated is True
    assert result.metrics.measured == measured


def test_full_capture_does_not_invent_full_inspection_from_available_text():
    result = _sample(
        responses=(
            _response(
                response_text="A" * 5000,
                inspection_scope="excerpt",
                brand_mentioned=False,
            ),
        )
    )
    assert result.responses[0].inspection_scope == "excerpt"
    assert result.metrics.measured == 0


def test_truncated_input_cannot_claim_full_response_inspection_or_use_full_digest():
    with pytest.raises(ValidationError, match="inspection|truncated"):
        _response(response_truncated=True, inspection_scope="full_response")
    with pytest.raises(ValueError, match="hash"):
        _sample(
            responses=(
                _response(
                    response_text="Excerpt",
                    response_truncated=True,
                    inspection_scope="excerpt",
                    response_hash=hashlib.sha256(b"Excerpt and an unseen remainder").hexdigest(),
                ),
            )
        )


@pytest.mark.parametrize(
    "field", ["grounded", "brand_mentioned", "complete", "citations_complete", "response_truncated"]
)
@pytest.mark.parametrize("value", [1, "true", "false"])
def test_response_flags_are_strict_not_coerced(field, value):
    with pytest.raises(ValidationError):
        _response(**{field: value})


@pytest.mark.parametrize("value", ["2026-09-03", "2026-09-03T12:00:00", 1788436800, "1788436800"])
def test_response_timestamp_requires_explicit_aware_datetime(value):
    with pytest.raises(ValidationError):
        _response(observed_at=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("setup_fingerprint", "a" * 64),
        ("model_id", "forged"),
        ("project_id", "other"),
        ("metrics", {"mention_rate": 100}),
        ("report_status", "CLIENT_VALIDATED"),
    ],
)
def test_provider_response_cannot_inject_trusted_settings_or_results(field, value):
    with pytest.raises(ValidationError):
        _response(**{field: value})


def test_response_revalidates_frozen_input_with_bypassed_strict_flags():
    with pytest.raises(ValidationError):
        _sample(responses=(_response().model_copy(update={"grounded": "true"}),))


def test_stored_sample_round_trip_is_deeply_immutable_and_checks_excerpt_hash():
    from ai_search_audit.diagnostic_models import BenchmarkSample

    original = _sample(responses=(_response(citations=("https://studio.example/",)),))
    loaded = BenchmarkSample.model_validate_json(original.model_dump_json())
    assert loaded == original
    assert isinstance(loaded.responses, tuple)
    assert isinstance(loaded.responses[0].citations, tuple)
    assert isinstance(loaded.approved_domains, tuple)
    assert isinstance(loaded.metrics.limitations, tuple)
    with pytest.raises(ValidationError):
        loaded.responses[0].brand_mentioned = False
    with pytest.raises(ValidationError):
        loaded.metrics.mention_rate = 0
    damaged = json.loads(original.model_dump_json())
    damaged["responses"][0]["response_excerpt"] = "Tampered"
    with pytest.raises(ValueError, match="hash"):
        BenchmarkSample.model_validate(damaged)
    damaged = json.loads(original.model_dump_json())
    damaged["metrics"]["mention_rate"] = 0
    with pytest.raises(ValueError, match="metrics"):
        BenchmarkSample.model_validate(damaged)


def test_legacy_observation_json_loads_unchanged_without_upgrading_missing_metadata():
    from ai_search_audit.diagnostic_models import BenchmarkResponseInput
    from ai_search_audit.models import AIObservation

    payload = dict(
        observation_id="old-1",
        prompt_id="p-pl",
        provider="Example AI",
        observed_at="2026-09-03T12:00:00",
        response_excerpt="Legacy text",
    )
    legacy = AIObservation.model_validate_json(json.dumps(payload))
    assert legacy.grounded is legacy.brand_mentioned is None
    assert legacy.citations == []
    assert "complete" not in legacy.model_dump()
    assert "model_id" not in legacy.model_dump()
    assert AIObservation.model_validate_json(legacy.model_dump_json()) == legacy
    with pytest.raises(ValidationError):
        BenchmarkResponseInput.model_validate(legacy.model_dump())
