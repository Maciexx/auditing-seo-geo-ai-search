"""Read-only, frozen client facts and technical accounting, never strategic prose."""

import base64
import hashlib
import json
import re
from importlib import import_module, util

import httpx
import pytest

from ai_search_audit.cli import main
from ai_search_audit.diagnostic_store import DiagnosticStore
from tests.test_diagnostic_workflow import _contract, _hashes, no_network, project
from tests.test_openai_observations import _source, collect, message, payload, profile

__all__ = ["no_network", "project"]


def module():
    assert util.find_spec("ai_search_audit.observation_report"), "observation projection missing"
    return import_module("ai_search_audit.observation_report")


def test_literal_projection_is_explicit_and_legacy_fragment_bytes_do_not_change():
    report = report_for(data=payload(r"**Example Studio** may help. literal \u002a"))
    legacy = module().render_observation_fragment(report)
    body = re.sub(r"audit-measurement:[a-f0-9]{64}", "audit-measurement:HASH", legacy)
    assert hashlib.sha256(body.encode()).hexdigest() == (
        "72af22b6deff97ef3e0316c2918b7efe1c3d39b818f70cc7b2f63373e30cb9c9"
    )
    assert module().render_observation_fragment(report, projection_version="1.0.0") == legacy
    modern = module().render_observation_fragment(report, projection_version="1.1.0")
    assert modern != legacy and "<!-- audit-literal-v1:" in modern
    assert "**Example Studio**" in module().render_observation_technical_fragment(report)
    with pytest.raises(ValueError, match="projection"):
        module().render_observation_fragment(report, projection_version="9.0.0")


def test_literal_projection_preserves_unicode_offsets_and_repeated_citations():
    text = "Żółć studio.example " + "word " * 140
    data = payload(text)
    spans = [
        {
            "type": "url_citation",
            "start_index": index,
            "end_index": index + 5,
            "url": "https://studio.example/",
        }
        for index in [0, 3, 30, 40, 50, 60, 70, 80, 590, 598]
    ]
    data["output"][1] = message(text, spans)
    report = report_for(data=data)
    before = report.model_dump_json()
    client = import_module("ai_search_audit.observation_client").build_observation_client(
        report, projection_version="1.1.0"
    )
    for excerpt in client.excerpts:
        encoded = excerpt.text.removeprefix("<!-- audit-literal-v1:").removesuffix(" -->")
        literal = json.loads(base64.b64decode(encoded))
        assert literal["text"] == text[:598]
        assert literal["urls"] == ["https://studio.example/"]
        assert literal["citations"] == [[a["start_index"], a["end_index"], 0] for a in spans[:-1]]
    assert len(client.excerpts) == 2
    assert report.model_dump_json() == before


def report_for(*, source=None, data=None, **kwargs):
    source = source or _source()
    run, _ = collect(
        data,
        source=source,
        selected=profile(selected_prompt_ids=tuple(p.prompt_id for p in source.prompts[:2])),
        **kwargs,
    )
    return module().ObservationReport(
        reference={"source_version": source.binding.source_version, "run_id": "run-1"},
        manifest_sha256="a" * 64,
        run=run.run,
    )


@pytest.mark.parametrize("locale", ["en", "pl"])
def test_client_actual_sample_locale_citations_without_operator_ids(locale):
    source = _source()
    source = source.model_copy(
        update={"binding": source.binding.model_copy(update={"report_locale": locale})}
    )
    report = report_for(source=source)
    text = module().render_observation_fragment(report)
    assert "OpenAI API web search" in text
    assert "2026-09-04" in text
    assert "pl" in text and "en" in text
    assert ("Próbka" if locale == "pl" else "Sample") in text
    assert ("markę" if locale == "pl" else "Branded") in text
    assert "2/2" in text
    assert "[studio.example](https://studio.example/)" in text
    assert "Visit studio.example" in text
    assert "consulted.example" not in text
    assert "req_fixture" not in text and "diagnostics/" not in text
    assert "ChatGPT" in text and "Google AI Overviews" in text
    assert ("nie" if locale == "pl" else "not") in text.lower()
    assert text == module().render_observation_fragment(report)


@pytest.mark.parametrize(
    "version,expected", [("1.1.0", "branded"), ("2.0.0", "branded"), ("2.1.0", "discovery")]
)
def test_scope_requires_new_pack_not_discovery_name(version, expected):
    source = _source()
    source = source.model_copy(
        update={
            "prompts": tuple(
                p.model_copy(
                    update={
                        "pack_version": version,
                        "intent": "category_discovery",
                        "text": "Which providers?",
                    }
                )
                for p in source.prompts
            )
        }
    )
    client = (
        import_module("ai_search_audit.observation_client")
        if util.find_spec("ai_search_audit.observation_client")
        else None
    )
    assert client is not None, "frozen client DTO missing"
    dto = client.build_observation_client(report_for(source=source))
    assert {row.scope for row in dto.groups} == {expected}


def test_unknown_usage_and_zero_eligible_are_not_negative_results():
    data = payload("ambiguous studio phrase")
    data["usage"] = None
    report = report_for(data=data)
    text = module().render_observation_fragment(report)
    assert "0/2" in text
    assert "niedostępne" in text.lower()
    assert "0%" not in text
    assert "Brak danych nie jest wynikiem negatywnym" in text
    technical = module().render_observation_technical_fragment(report)
    assert "req_fixture" in technical and '"amount_usd": null' in technical
    assert "source_derived_estimate_not_invoice" in technical


def test_excerpt_keeps_only_overlapping_annotations_and_escapes_injection():
    data = payload()
    data["output"][1] = message(
        "studio.example @@TOKEN0@@ [evil](https://evil.example/) " + "x" * 2100,
        [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 14,
                "url": "https://studio.example/a_(safe)?x=[z]",
            },
            {
                "type": "url_citation",
                "start_index": 2100,
                "end_index": 2110,
                "url": "https://outside.example/",
            },
        ],
    )
    text = module().render_observation_fragment(report_for(data=data))
    assert "outside.example" not in text
    assert "@@TOKEN" not in text
    assert "[evil](https://evil.example/)" not in text
    assert "https://studio.example/a_%28safe%29?x=%5Bz%5D" in text
    assert "fragment" in text.lower()


@pytest.mark.parametrize("change", ["prose", "report", "extra"])
def test_guard_rejects_unchecked_projection_changes(change):
    report = report_for()
    client = import_module("ai_search_audit.observation_client")
    dto = client.build_observation_client(report)
    if change == "prose":
        dto = dto.model_copy(update={"introduction": ("Guaranteed ChatGPT visibility",)})
    elif change == "report":
        dto = dto.model_copy(
            update={"report": report.model_copy(update={"manifest_sha256": "b" * 64})}
        )
    else:
        dto = dto.model_copy(update={"strategy": "unsupported strategic thesis"})
    with pytest.raises(ValueError):
        client.validate_observation_client(dto, report=report)


def test_report_cli_read_only_binds_published_source(project, capsys):
    module()
    report = report_for(source=_contract(project).source)
    path = DiagnosticStore(project).publish(report.run)
    before = _hashes(project)
    args = [
        "project",
        "observation-report",
        "project:example",
        "--clients-root",
        str(project.parent),
        "--diagnostic-run",
        f"public-v1/{path.name}",
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    loaded = module().load_observation_report(
        "project:example", clients_root=project.parent, diagnostic_run_ref=f"public-v1/{path.name}"
    )
    assert text == module().render_observation_fragment(loaded)
    assert main(args + ["--projection-version", "1.1.0"]) == 0
    assert capsys.readouterr().out == module().render_observation_fragment(
        loaded, projection_version="1.1.0"
    )
    assert main(args + ["--view", "technical"]) == 0
    assert "req_fixture" in capsys.readouterr().out
    assert _hashes(project) == before


def test_exact_fragment_cannot_be_duplicate_stale_or_unbound():
    report = report_for()
    fragment = module().render_observation_fragment(report)
    require = module().require_observation_fragment
    require("## Evidence\n\n" + fragment, fragment)
    for text, expected in (
        (fragment, fragment),
        ("## Evidence\n" + fragment * 2, fragment),
        ("## Evidence\n" + fragment, None),
        ("## Evidence\n" + fragment.replace("2/2", "1/2"), fragment),
    ):
        with pytest.raises(ValueError):
            require(text, expected)


def test_whole_pack_denominator_survives_selected_subset(project):
    report = report_for(source=_contract(project).source)
    assert report.run.sample.metrics.expected > len(report.run.selected_prompt_ids)
    fragment = module().render_observation_fragment(report)
    assert f"2/{report.run.sample.metrics.expected} whole-pack questions" in fragment


@pytest.mark.parametrize("mutation", ["nested-count", "raw-error", "nan", "unserialized"])
def test_nested_frozen_dto_revalidation_rejects_mutation(mutation):
    report = report_for()
    client = import_module("ai_search_audit.observation_client")
    dto = client.build_observation_client(report)
    if mutation == "nested-count":
        dto = dto.model_copy(
            update={"groups": (dto.groups[0].model_copy(update={"eligible": 999}),)}
        )
    elif mutation == "raw-error":
        dto = dto.model_copy(
            update={"excerpts": (dto.excerpts[0].model_copy(update={"text": "provider error"}),)}
        )
    else:
        a = dto.report.run.attempts[0]
        a = a.model_copy(
            update={"unexpected": "private"}
            if mutation == "unserialized"
            else {"usage": a.usage.model_copy(update={"input_tokens": float("nan")})}
        )
        dto = dto.model_copy(
            update={
                "report": dto.report.model_copy(
                    update={"run": dto.report.run.model_copy(update={"attempts": (a,)})}
                )
            }
        )
    with pytest.raises(ValueError):
        client.validate_observation_client(dto, report=report)


def test_partial_citation_metadata_has_separate_denominator():
    data = payload()
    del data["output"][1]["content"][0]["annotations"]
    report = report_for(data=data)
    client = import_module("ai_search_audit.observation_client").build_observation_client(report)
    assert sum(g.eligible for g in client.groups) == 2
    assert sum(g.citation_eligible for g in client.groups) == 0
    assert not any(excerpt.sources for excerpt in client.excerpts)


def test_excerpts_cannot_inject_headings_or_hidden_renderer_metadata():
    data = payload("## Injected heading @@TOKEN0@@ **claim** studio.example")
    text = module().render_observation_fragment(report_for(data=data))
    assert "\n## Injected" not in text
    assert "**claim**" not in text
    assert "@@TOKEN" not in text


@pytest.mark.parametrize("overlap", [False, True])
def test_dense_citations_bound_excerpt_without_detaching_sources_or_cherry_picking(overlap):
    data = payload("studio.example " + "word " * 90)
    data["output"][1]["content"][0]["annotations"] = [
        {
            "type": "url_citation",
            "start_index": 0 if overlap else index * 20,
            "end_index": 10 if overlap else index * 20 + 10,
            "url": f"https://source-{index}.example/",
        }
        for index in range(12)
    ]
    report = report_for(data=data)
    client = import_module("ai_search_audit.observation_client").build_observation_client(report)
    assert len(report.run.attempts[0].citation_annotations) == 12
    assert all(len(excerpt.sources) <= 5 for excerpt in client.excerpts)
    if overlap:
        assert not client.excerpts
        assert "pominięto" in " ".join(client.notes)
    else:
        assert len(client.excerpts) == 2
        for excerpt in client.excerpts:
            assert len(excerpt.sources) == 5
            assert "source-5.example" not in " ".join(excerpt.sources)
            assert len(excerpt.text) <= 102


def test_annotation_crossing_excerpt_cut_is_not_partially_displayed():
    data = payload("studio.example " + "word " * 400)
    data["output"][1]["content"][0]["annotations"] = [
        {"type": "url_citation", "start_index": 0, "end_index": 650, "url": "https://span.example/"}
    ]
    client = import_module("ai_search_audit.observation_client").build_observation_client(
        report_for(data=data)
    )
    assert not client.excerpts
    assert "pominięto" in " ".join(client.notes)


def test_omitted_early_excerpts_are_not_replaced_with_later_favorable_answer():
    source = _source()
    source = source.model_copy(
        update={
            "prompts": (
                *source.prompts,
                source.prompts[0].model_copy(update={"prompt_id": "p-extra"}),
            )
        }
    )
    calls = 0

    def response(request):
        nonlocal calls
        calls += 1
        data = payload("studio.example later favorable answer")
        if calls < 3:
            data["output"][1]["content"][0]["annotations"] = [
                {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 10,
                    "url": f"https://source-{index}.example/",
                }
                for index in range(6)
            ]
        data["output"][0]["id"] = f"ws_fixture_{calls}"
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps(data).encode()))

    result, _ = collect(
        source=source,
        handler=response,
        selected=profile(selected_prompt_ids=tuple(p.prompt_id for p in source.prompts)),
    )
    report = module().ObservationReport(
        reference={"source_version": "public-v1", "run_id": "run-1"},
        manifest_sha256="a" * 64,
        run=result.run,
    )
    client = import_module("ai_search_audit.observation_client").build_observation_client(report)
    assert len(result.run.sample.responses) == 3
    assert not client.excerpts
