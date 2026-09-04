"""Fabricated Responses transports only; these tests never authorize paid requests."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from importlib import import_module

import httpx
import pytest
from pydantic import SecretStr

from tests.test_benchmark import _source, no_network

__all__ = ["no_network"]
MODEL = "gpt-5.4-mini-2026-03-17"
NOW = datetime(2026, 9, 4, tzinfo=UTC)
KEY = "sk-not-a-real-observation-test-only"


def profile(**changes):
    values = dict(
        model_id=MODEL,
        selected_prompt_ids=("p-pl", "p-en"),
        operational_allowance_usd=Decimal("2"),
    )
    values.update(changes)
    return import_module("ai_search_audit.observation_profile").ObservationProfile(**values)


def payload(text="Visit studio.example for its services."):
    return {
        "id": "resp_fixture",
        "object": "response",
        "status": "completed",
        "model": MODEL,
        "service_tier": "default",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_fixture",
                "status": "completed",
                "action": {
                    "type": "search",
                    "queries": ["studio"],
                    "sources": [{"type": "url", "url": "https://consulted.example/"}],
                },
            },
            message(text),
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 40, "cache_creation_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }


def message(text, annotations=None):
    return {
        "type": "message",
        "id": "msg_fixture",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": annotations
                if annotations is not None
                else [
                    {
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": 5,
                        "url": "https://studio.example/",
                        "title": "Studio",
                    }
                ],
            }
        ],
    }


def collect(data=None, *, selected=None, source=None, handler=None, **kwargs):
    calls = []

    def respond(request):
        calls.append(request)
        if handler:
            return handler(request)
        body = json.loads(json.dumps(data if data is not None else payload()))
        for item in body.get("output", []):
            if item.get("type") == "web_search_call":
                item["id"] += str(len(calls))
        return httpx.Response(
            200,
            stream=httpx.ByteStream(json.dumps(body).encode()),
            headers={"x-request-id": "req_fixture"},
        )

    module = import_module("ai_search_audit.openai_observations")
    runner = module.OpenAIObservations(
        transport=httpx.MockTransport(respond),
        now=lambda: NOW,
        **kwargs.pop("runner_kwargs", {}),
    )
    credential = kwargs.pop("api_key", SecretStr(KEY))
    result = runner.collect(
        source or _source(),
        profile=selected or profile(),
        api_key=credential,
        paid_authorized=kwargs.pop("paid_authorized", True),
        **kwargs,
    )
    return result, calls


def test_success_binds_exact_prompts_search_citations_and_accounting():
    result, calls = collect()
    assert len(calls) == len(result.run.attempts) == 2
    assert result.stop_reason is None
    assert result.run.sample.metrics.measured == 2
    assert result.run.sample.metrics.mention_rate == 100
    assert result.run.sample.metrics.citation_rate == 100
    for index, request in enumerate(calls):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body = json.loads(request.content)
        assert body["input"] == _source().prompts[index].text
        assert body["tools"] == [
            {"type": "web_search", "search_context_size": "low", "external_web_access": True}
        ]
        assert body["tool_choice"] == "required"
        assert body["include"] == ["web_search_call.action.sources"]
        assert body["store"] is False
        assert body["service_tier"] == "default"
        assert body["model"] == MODEL
        assert body["max_tool_calls"] == profile().max_tool_calls
        assert body["max_output_tokens"] == profile().max_output_tokens
        assert not {"previous_response_id", "conversation", "metadata"} & body.keys()
        assert KEY not in request.content.decode()
    attempt = result.run.attempts[0]
    assert str(attempt.search_actions[0].consulted_sources[0]) == "https://consulted.example/"
    assert str(attempt.citation_annotations[0].url) == "https://studio.example/"
    assert attempt.usage.total_tokens == 120
    assert result.estimates[0].amount_usd == Decimal("0.010138")
    assert result.estimates[0].search_calls == 1
    assert result.run.setup.benchmark_setup.account_state is None
    assert result.run.setup.benchmark_setup.market is None
    assert KEY not in result.model_dump_json()


@pytest.mark.parametrize(
    "gate,key,reason",
    [
        (False, SecretStr(KEY), "paid_not_authorized"),
        (True, None, "missing_key"),
        (True, SecretStr("bad key"), "invalid_key"),
    ],
)
def test_preflight_never_fabricates_attempts(gate, key, reason):
    result, calls = collect(paid_authorized=gate, api_key=key)
    assert calls == []
    assert result.run.attempts == ()
    assert result.stop_reason == reason
    assert result.run.sample.metrics.measured == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": "auto"},
        {"model_id": "gpt-5.4-mini"},
        {"selected_prompt_ids": ()},
        {"selected_prompt_ids": ("p-en", "p-pl")},
        {"selected_prompt_ids": ("p-pl", "p-pl")},
        {"selected_prompt_ids": ("foreign",)},
        {"max_tool_calls": 33},
        {"max_tool_calls": True},
        {"max_output_tokens": 15},
        {"max_response_bytes": 0},
        {"timeout_seconds": 0.5},
        {"operational_allowance_usd": "NaN"},
        {"base_url": "https://evil.example/"},
    ],
)
def test_invalid_profile_or_selection_cannot_request(changes):
    with pytest.raises(ValueError):
        collect(selected=profile(**changes))


def test_discovery_identity_metadata_never_enters_request():
    source = _source()
    source = source.model_copy(
        update={
            "prompts": tuple(
                p.model_copy(update={"text": "Which services are available?"})
                for p in source.prompts
            )
        }
    )
    _, calls = collect(source=source)
    for request in calls:
        assert "studio" not in request.content.decode().casefold()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("studio is a common word", None),
        ("Try https://studio.example/services", True),
        ("Try studio.example", True),
        ("Try https://studio.example.evil.test", None),
        ("Try https://evil.test/?next=studio.example", None),
        ("Try https://evil.test/#studio.example", None),
        ("Try notstudio.example", None),
        ("No suitable options found", False),
    ],
)
def test_mentions_use_only_inspected_text_and_conservative_aliases(text, expected):
    result, _ = collect(payload(text))
    assert result.run.sample.responses[0].brand_mentioned is expected


def test_configured_tool_and_prose_citations_do_not_establish_grounding():
    data = payload()
    data["output"] = data["output"][1:]
    result, _ = collect(data)
    assert result.run.sample.responses[0].grounded is False
    assert result.run.sample.metrics.measured == 0
    assert result.estimates[0].search_calls == 0


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_noncomplete_provider_status_has_no_benchmark_answer(status):
    data = payload()
    data["status"] = status
    result, _ = collect(data)
    assert result.run.sample.responses == ()
    assert result.run.attempts[0].status in {"incomplete", "failed"}


def test_refusal_in_any_message_disqualifies_entire_answer():
    data = payload()
    data["output"].append(message("hidden"))
    data["output"][-1]["content"] = [{"type": "refusal", "refusal": "Cannot answer"}]
    result, _ = collect(data)
    assert result.run.attempts[0].status == "refused"
    assert result.run.sample.responses == ()


def test_multimessage_unicode_offsets_and_full_inspection_beyond_excerpt():
    data = payload("😀" + "x" * 2100)
    data["output"][1] = message("😀" + "x" * 2100, [])
    data["output"].append(message("studio.example"))
    result, _ = collect(data)
    response = result.run.sample.responses[0]
    assert response.brand_mentioned is True
    assert response.captured_text_length == 2116
    assert response.inspection_scope == "full_response"
    assert response.excerpt_truncated is True
    annotation = result.run.attempts[0].citation_annotations[0]
    assert annotation.start_index == 2102


@pytest.mark.parametrize(
    "mutation",
    [
        "unsafe_url",
        "credentials",
        "offset",
        "float_offset",
        "late_message",
        "unknown_tool",
        "tool_bound",
        "output_bound",
        "invalid_usage",
        "cache_write",
    ],
)
def test_untrusted_shapes_and_resource_violations_fail_closed(mutation):
    data = payload()
    annotation = data["output"][1]["content"][0]["annotations"][0]
    if mutation == "unsafe_url":
        annotation["url"] = "file:///etc/passwd"
    elif mutation == "credentials":
        annotation["url"] = "https://user:pass@studio.example/"
    elif mutation == "offset":
        annotation["end_index"] = 999
    elif mutation == "float_offset":
        annotation["end_index"] = 5.0
    elif mutation == "late_message":
        data["output"].append({"type": "message", "content": "bad"})
    elif mutation == "unknown_tool":
        data["output"].append({"type": "function_call"})
    elif mutation == "tool_bound":
        data["output"] = [dict(data["output"][0], id=f"ws_{i}") for i in range(33)]
    elif mutation == "output_bound":
        data["usage"]["output_tokens"] = 99999
        data["usage"]["total_tokens"] = 100099
    elif mutation == "invalid_usage":
        data["usage"]["input_tokens"] = True
    else:
        data["usage"]["input_tokens_details"]["cache_creation_tokens"] = 3
    result, calls = collect(data)
    assert len(calls) == 1
    assert result.run.attempts[0].status == "failed"
    assert result.run.sample.responses == ()
    assert result.estimates[0].amount_usd is None
    assert result.stop_reason == "unknown_cost"


def test_missing_usage_stops_without_zero_cost_or_missing_samples_as_negatives():
    data = payload()
    del data["usage"]
    result, calls = collect(data)
    assert len(calls) == 1
    assert result.run.attempts[0].usage is None
    assert result.run.sample.metrics.measured == 1
    assert result.estimates[0].amount_usd is None
    assert result.stop_reason == "unknown_cost"


def test_missing_citation_and_consulted_metadata_remain_unknown():
    data = payload()
    del data["output"][0]["action"]["sources"]
    del data["output"][1]["content"][0]["annotations"]
    result, _ = collect(data)
    assert result.run.attempts[0].search_actions[0].consulted_sources is None
    assert result.run.attempts[0].citation_metadata_complete is False
    assert result.run.sample.metrics.citation_rate is None


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("model", "different-model", "model_mismatch"),
        ("service_tier", "priority", "tier_mismatch"),
    ],
)
def test_actual_model_tier_mismatch_retains_usage_but_cannot_be_priced(field, value, reason):
    data = payload()
    data[field] = value
    result, calls = collect(data)
    assert len(calls) == 1
    assert result.run.attempts[0].error_category == reason
    assert result.run.attempts[0].usage.total_tokens == 120
    assert result.estimates[0].amount_usd is None


@pytest.mark.parametrize(
    "status,category",
    [
        (429, "http_429"),
        (500, "http_5xx"),
        (401, "invalid_key"),
        (400, "http_error"),
        (302, "http_error"),
    ],
)
def test_http_errors_and_redirects_are_single_attempt_and_private(status, category, caplog):
    result, calls = collect(
        handler=lambda _: httpx.Response(
            status,
            text=KEY,
            headers={"location": f"https://evil.example/{KEY}"},
            extensions={"reason_phrase": KEY.encode(), "http_version": KEY.encode()},
        )
    )
    assert len(calls) == 1
    assert result.run.attempts[0].error_category == category
    assert KEY not in result.model_dump_json() + caplog.text


@pytest.mark.parametrize("failure", ["timeout", "transport_error"])
def test_uncertain_transport_outcomes_never_retry(failure):
    def fail(_):
        raise (httpx.ReadTimeout if failure == "timeout" else httpx.ConnectError)(KEY)

    result, calls = collect(handler=fail)
    assert len(calls) == 1
    assert result.run.attempts[0].error_category == failure
    assert KEY not in result.model_dump_json()


@pytest.mark.parametrize(
    "text",
    [
        KEY,
        "%73%6b-not-a-real-observation-test-only",
        "\\u0073\\u006b-not-a-real-observation-test-only",
    ],
)
def test_sensitive_provider_output_is_not_retained(text):
    result, _ = collect(payload(text))
    assert result.run.attempts[0].error_category == "sensitive_response"
    assert KEY not in result.model_dump_json()


def test_response_byte_and_elapsed_time_bounds():
    result, _ = collect(selected=profile(max_response_bytes=16))
    assert result.run.attempts[0].error_category == "response_too_large"
    times = iter([0.0, 121.0, 122.0, 123.0])
    result, _ = collect(runner_kwargs={"clock": lambda: next(times)})
    assert result.run.attempts[0].error_category == "timeout"


def test_pricing_expiry_and_insufficient_operational_allowance_stop_preflight():
    result, calls = collect(runner_kwargs={"pricing_date": NOW.date() + timedelta(days=31)})
    assert not calls
    assert result.stop_reason == "pricing_unavailable"
    result, calls = collect(selected=profile(operational_allowance_usd=Decimal("0.00001")))
    assert not calls
    assert result.stop_reason == "allowance_exhausted"


@pytest.mark.parametrize("status", [429, 500])
def test_received_request_id_survives_http_failure(status):
    result, _ = collect(
        handler=lambda _: httpx.Response(status, headers={"x-request-id": "req_failure"})
    )
    assert result.run.attempts[0].request_id == "req_failure"


def test_received_request_id_survives_stream_failure():
    result, _ = collect(selected=profile(max_response_bytes=16))
    assert result.run.attempts[0].request_id == "req_fixture"


def test_duplicate_call_id_across_requests_retains_prior_paid_attempt():
    result, calls = collect(
        handler=lambda _: httpx.Response(
            200, stream=httpx.ByteStream(json.dumps(payload()).encode())
        )
    )
    assert len(calls) == 2
    assert len(result.run.sample.responses) == 1
    assert result.run.attempts[1].error_category == "malformed_response"
    assert result.run.attempts[1].search_actions[0].call_id is None
    assert result.run.attempts[1].usage.total_tokens == 120
    assert result.stop_reason == "unknown_cost"


@pytest.mark.parametrize("mutation", ["oversized_later", "invalid_locale", "domain"])
def test_all_selected_inputs_are_checked_before_first_paid_request(mutation):
    source = _source()
    if mutation == "oversized_later":
        source = source.model_copy(
            update={
                "prompts": (
                    source.prompts[0],
                    source.prompts[1].model_copy(update={"text": "x" * 70000}),
                )
            }
        )
    elif mutation == "invalid_locale":
        source = source.model_copy(
            update={
                "prompts": (
                    source.prompts[0],
                    source.prompts[1].model_copy(update={"locale": "bad locale"}),
                )
            }
        )
    else:
        source = source.model_copy(update={"canonical_domains": ("bad domain",)})

    def never(_):
        pytest.fail("preflight must validate every selected prompt before any request")

    with pytest.raises(ValueError, match="invalid observation configuration"):
        collect(source=source, handler=never)


@pytest.mark.parametrize("action_type", ["open_page", "find_in_page"])
def test_nonsearch_actions_and_query_strings_do_not_add_search_fees(action_type):
    data = payload()
    data["output"][0]["action"]["queries"] = ["a", "b", "c"]
    data["output"].insert(
        1,
        {
            "type": "web_search_call",
            "id": "ws_other",
            "status": "completed",
            "action": {"type": action_type},
        },
    )
    result, _ = collect(data)
    assert len(result.run.attempts[0].search_actions) == 2
    assert result.estimates[0].search_calls == 1
    assert result.estimates[0].amount_usd == Decimal("0.010138")


@pytest.mark.parametrize("status", ["in_progress", "searching", "failed", "incomplete"])
def test_noncompleted_search_does_not_ground_and_has_unknown_cost(status):
    data = payload()
    data["output"][0]["status"] = status
    result, calls = collect(data)
    assert len(calls) == 1
    assert result.run.sample.responses[0].grounded is False
    assert result.estimates[0].amount_usd is None


def test_partial_citation_inventory_validates_every_message():
    data = payload()
    second = message("Second visible response", [])
    del second["content"][0]["annotations"]
    data["output"].append(second)
    result, _ = collect(data)
    assert result.run.attempts[0].citation_metadata_complete is False
    assert len(result.run.attempts[0].citation_annotations) == 1
    assert result.run.sample.responses[0].citations_complete is False


@pytest.mark.parametrize("mutation", ["no_text", "incomplete_message", "empty_sources"])
def test_empty_and_partial_output_are_not_negative_answers(mutation):
    data = payload()
    if mutation == "no_text":
        data["output"] = data["output"][:1]
    elif mutation == "incomplete_message":
        data["output"][1]["status"] = "incomplete"
    else:
        data["output"][0]["action"]["sources"] = []
    result, _ = collect(data)
    if mutation == "empty_sources":
        assert result.run.attempts[0].search_actions[0].consulted_sources == ()
    else:
        assert result.run.sample.responses == ()
        assert result.run.attempts[0].status in {"incomplete", "no_text"}


def test_untrusted_snapshot_and_loaded_invalid_usage_cannot_be_priced():
    usage = import_module("ai_search_audit.observation_usage")
    result, _ = collect()
    snapshot = result.run.price_provenance.model_copy(update={"input_per_million": Decimal(0)})
    assert (
        usage.estimate_attempt(result.run.attempts[0], snapshot, on=NOW.date()).amount_usd is None
    )
    data = payload()
    data["usage"]["input_tokens_details"]["cache_creation_tokens"] = 10
    invalid, _ = collect(data)
    attempt = type(invalid.run.attempts[0]).model_validate_json(
        invalid.run.attempts[0].model_dump_json()
    )
    assert (
        usage.estimate_attempt(attempt, usage.trusted_price_snapshot(), on=NOW.date()).amount_usd
        is None
    )


@pytest.mark.parametrize("space", ["\n", "\t", "\u00a0", "   "])
def test_multitoken_canonical_entity_whitespace_is_unknown_not_negative(space):
    source = _source(business="blue-wing")
    source = source.model_copy(
        update={
            "prompts": tuple(
                p.model_copy(update={"target_entities": ("Blue Wing",)}) for p in source.prompts
            )
        }
    )
    result, _ = collect(payload(f"Blue{space}Wing offers services"), source=source)
    assert result.run.sample.responses[0].brand_mentioned is None


@pytest.mark.parametrize("mutation", ["incomplete_details", "object"])
def test_contradictory_or_foreign_root_metadata_cannot_prove_completed_answer(mutation):
    data = payload()
    if mutation == "incomplete_details":
        data["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        data["object"] = "chat.completion"
    result, calls = collect(data)
    assert len(calls) == 1
    assert result.run.sample.responses == ()
    assert result.run.attempts[0].error_category == "unsupported_response"


def test_network_boundary_disables_env_proxy_redirects_and_transport_retries(monkeypatch):
    seen = []
    original = httpx.Client.__init__

    def inspect(self, *args, **kwargs):
        seen.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", inspect)
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-inherit.invalid:80")
    collect()
    assert all(item["trust_env"] is False and item["follow_redirects"] is False for item in seen)
    assert all(item["timeout"].read == profile().timeout_seconds for item in seen)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
def test_missing_individual_usage_counts_remain_null_and_stop(field):
    data = payload()
    del data["usage"][field]
    result, calls = collect(data)
    assert len(calls) == 1
    assert getattr(result.run.attempts[0].usage, field) is None
    assert result.estimates[0].amount_usd is None


def test_absent_cached_usage_cannot_be_priced_as_zero_cache():
    data = payload()
    del data["usage"]["input_tokens_details"]
    result, _ = collect(data)
    assert result.run.attempts[0].usage.cached_input_tokens is None
    assert result.estimates[0].amount_usd is None


@pytest.mark.parametrize("body", [b'{"output":[],"output":[]}', b'{"value":NaN}', b"[]"])
def test_invalid_json_payloads_fail_with_safe_error(body):
    result, calls = collect(handler=lambda _: httpx.Response(200, stream=httpx.ByteStream(body)))
    assert len(calls) == 1
    assert result.run.attempts[0].error_category == "malformed_response"


def test_stream_timeout_preserves_received_request_id():
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"output":'
            raise httpx.ReadTimeout(KEY)

    result, calls = collect(
        handler=lambda _: httpx.Response(
            200, stream=BrokenStream(), headers={"x-request-id": "req_before_timeout"}
        )
    )
    assert len(calls) == 1
    assert result.run.attempts[0].request_id == "req_before_timeout"
    assert result.run.attempts[0].error_category == "timeout"


@pytest.mark.parametrize(
    "canonical,visible,expected",
    [
        ("strasse.example", "https://straße.example/", None),
        ("straße.example", "https://strasse.example/", None),
        ("straße.example", "https://straße.example/", True),
        ("straße.example", "https://xn--strae-oqa.example/", True),
        ("xn--strae-oqa.example", "https://straße.example/", True),
        ("strasse.example", "https://STRASSE.EXAMPLE/", True),
        ("straße.example", "HTTPS://STRAẞE.EXAMPLE/", True),
        ("straße.example", "straße.example", True),
        ("straße.example", "xn--strae-oqa.example", True),
        ("strasse.example", "straße.example", None),
        ("straße.example", "strasse.example", None),
        ("strasse.example", "STRASSE.EXAMPLE", True),
        ("studio.example", "https://STUDIO.EXAMPLE/services", True),
    ],
)
def test_canonical_host_matching_preserves_modern_idna(canonical, visible, expected):
    source = _source()
    source = source.model_copy(
        update={
            "canonical_domains": (canonical,),
            "prompts": tuple(
                p.model_copy(update={"target_entities": ("Straße", "Strasse")})
                for p in source.prompts
            ),
        }
    )
    module = import_module("ai_search_audit.openai_observations")
    assert module._mentions(f"Try {visible}", source, grounded=True) is expected


@pytest.mark.parametrize("prefix", ["", "https://"])
@pytest.mark.parametrize(
    "host",
    [
        "foo\u0301studio.example",
        "studio.example\u0301",
        "\u0301studio.example",
        "foo\u0338studio.example",
        "studio.example\u0338",
        "\u0338studio.example",
        "user@studio.example",
        "notstudio.example",
        "studio.example.evil.test",
        "evil.example/?next=studio.example",
        "evil.example/#studio.example",
    ],
)
def test_combining_marks_and_lookalikes_cannot_split_into_canonical_host(prefix, host):
    module = import_module("ai_search_audit.openai_observations")
    assert module._mentions(f"Try {prefix}{host}", _source(), grounded=True) is not True


@pytest.mark.parametrize(
    "visible",
    [
        "cafe\u0301.example",
        "https://cafe\u0301.example/",
        "CAFÉ.EXAMPLE",
        "https://xn--caf-dma.example/",
    ],
)
def test_decomposed_complete_host_matches_its_canonical_idna(visible):
    source = _source().model_copy(update={"canonical_domains": ("café.example",)})
    module = import_module("ai_search_audit.openai_observations")
    assert module._mentions(f"Try {visible}", source, grounded=True) is True


@pytest.mark.parametrize(
    "mutation,expected_actions",
    [
        ("invalid_citation", 1),
        ("late_message", 1),
        ("invalid_later_action", 1),
        ("duplicate_action", 1),
        ("excess_actions", 3),
    ],
)
def test_later_parse_errors_preserve_bounded_valid_search_action_prefix(mutation, expected_actions):
    data = payload()
    if mutation == "invalid_citation":
        data["output"][1]["content"][0]["annotations"][0]["end_index"] = 999
    elif mutation == "late_message":
        data["output"].append(
            {"type": "message", "role": "assistant", "status": "completed", "content": "invalid"}
        )
    elif mutation == "invalid_later_action":
        data["output"].append(
            {
                "type": "web_search_call",
                "id": "ws_invalid",
                "status": "completed",
                "action": {"type": "unknown"},
            }
        )
    elif mutation == "duplicate_action":
        data["output"].append(data["output"][0].copy())
    else:
        data["output"] = [dict(data["output"][0], id=f"ws_{index}") for index in range(4)]
    result, calls = collect(data)
    assert len(calls) == 1
    attempt = result.run.attempts[0]
    assert attempt.status == "failed"
    assert attempt.usage.total_tokens == 120
    assert len(attempt.search_actions) == expected_actions
    assert all(
        action.action == "search" and action.status == "completed"
        for action in attempt.search_actions
    )
    assert len({action.call_id for action in attempt.search_actions}) == expected_actions
    assert attempt.citation_metadata_complete is False
    assert result.run.sample.responses == ()
    assert result.estimates[0].amount_usd is None
    assert result.estimates[0].search_calls is None
    assert result.stop_reason == "unknown_cost"
