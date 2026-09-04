"""Explicit local workflow; fabricated transport and socket denial only."""

import json
from importlib import import_module, util

import httpx
import pytest

from ai_search_audit.cli import main
from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.openai_observations import OpenAIObservations
from tests.test_diagnostic_workflow import _contract, _hashes, no_network, project
from tests.test_openai_observations import KEY, NOW, payload, profile

__all__ = ["no_network", "project"]


def module():
    assert util.find_spec("ai_search_audit.observation_workflow"), "observation workflow missing"
    return import_module("ai_search_audit.observation_workflow")


def command(root, **changes):
    ids = tuple(p.prompt_id for p in _contract(root).source.prompts[:2])
    file = root.parent.parent / "observation-profile.json"
    file.write_text(profile(selected_prompt_ids=ids, **changes).model_dump_json())
    return [
        "project",
        "observe-ai",
        "project:example",
        "--clients-root",
        str(root.parent),
        "--source-version",
        "public-v1",
        "--profile",
        str(file),
    ]


def harness(monkeypatch, capsys, *, fail_second=False):
    workflow = module()
    calls = []

    def respond(request):
        assert "OBSERVATION_PREFLIGHT=" in capsys.readouterr().out or calls
        calls.append(request)
        if fail_second and len(calls) == 2:
            raise httpx.ReadTimeout("synthetic private provider body")
        data = payload()
        data["output"][0]["id"] += str(len(calls))
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps(data).encode()))

    monkeypatch.setattr(
        workflow,
        "OpenAIObservations",
        lambda: OpenAIObservations(transport=httpx.MockTransport(respond), now=lambda: NOW),
    )
    monkeypatch.setenv("AUDIT_OPENAI_API_KEY", KEY)
    return calls


@pytest.mark.parametrize("flag", [[], ["--paid-authorized"], ["--preflight-only"]])
def test_no_authority_or_no_purpose_key_never_publishes(project, monkeypatch, capsys, flag):
    module()
    monkeypatch.delenv("AUDIT_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    before = _hashes(project)
    assert main(command(project) + flag) == 0
    output = capsys.readouterr().out
    assert '"published": false' in output
    assert '"state": "UNAVAILABLE"' in output
    assert KEY not in output
    assert _hashes(project) == before


def test_preflight_exact_selection_policy_prices_before_transport(project, monkeypatch, capsys):
    workflow = module()
    source = _contract(project).source
    selected = profile(selected_prompt_ids=tuple(p.prompt_id for p in source.prompts[:2]))
    value = workflow.prepare_observation_preflight(source, selected)
    assert value.selected_prompt_ids == selected.selected_prompt_ids
    assert value.model_id == selected.model_id
    assert [(p.prompt_id, p.locale, p.scope) for p in value.prompts] == [
        (p.prompt_id, p.locale, "branded") for p in source.prompts[:2]
    ]
    assert value.price_basis.as_of == NOW.date()
    assert value.max_output_tokens == selected.max_output_tokens
    assert value.max_tool_calls == selected.max_tool_calls
    assert value.operational_allowance_usd == selected.operational_allowance_usd
    assert "not" in " ".join(value.limitations).lower()
    calls = harness(monkeypatch, capsys)
    assert main(command(project) + ["--paid-authorized"]) == 0
    assert len(calls) == 2
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    assert len(run.attempts) == run.sample.metrics.measured == 2


def test_partial_actual_attempts_publish_only_existing_store(project, monkeypatch, capsys):
    calls = harness(monkeypatch, capsys, fail_second=True)
    before = _hashes(project)
    assert main(command(project) + ["--paid-authorized"]) == 0
    output = capsys.readouterr().out
    assert '"state": "PARTIAL"' in output
    assert "synthetic private provider body" not in output
    assert len(calls) == 2
    run = DiagnosticStore(project).load("public-v1", "run-1").run
    assert len(run.attempts) == 2 and run.sample.metrics.measured == 1
    assert all(_hashes(project)[key] == value for key, value in before.items())


@pytest.mark.parametrize("failure", ["duplicate", "extra", "oversize", "symlink", "key-in-id"])
def test_profile_failures_do_not_echo_or_publish(project, monkeypatch, capsys, failure):
    module()
    monkeypatch.setenv("AUDIT_OPENAI_API_KEY", KEY)
    args = command(project)
    from pathlib import Path

    file = Path(args[-1])
    if failure == "duplicate":
        file.write_text('{"model_id":"' + KEY + '","model_id":"other"}')
    elif failure == "extra":
        values = json.loads(file.read_text())
        values["api_key"] = KEY
        file.write_text(json.dumps(values))
    elif failure == "oversize":
        file.write_text(KEY * 4000)
    elif failure == "symlink":
        link = file.with_suffix(".link")
        link.symlink_to(file)
        args[-1] = str(link)
    else:
        values = json.loads(file.read_text())
        values["selected_prompt_ids"] = [KEY]
        file.write_text(json.dumps(values))
    assert main(args + ["--paid-authorized"]) == 2
    output = capsys.readouterr()
    assert KEY not in output.out + output.err
    assert "OBSERVATION_PREFLIGHT=" not in output.out
    assert not (project / "diagnostics").exists()
