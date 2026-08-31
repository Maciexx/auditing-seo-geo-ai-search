import subprocess

import ai_search_audit.adapters as adapters_module
from ai_search_audit.adapters import (
    FUTURE_CAPABILITIES,
    AdapterContext,
    GeoOptimizerAdapter,
    StructuredDataAdapter,
)
from ai_search_audit.config import AuditConfig
from ai_search_audit.models import DataState, Page, Site


def context() -> AdapterContext:
    return AdapterContext(
        site=Site(domain="example.com", base_url="https://example.com"),
        config=AuditConfig(domain="example.com"),
        pages=[
            Page(
                url="https://example.com/",
                final_url="https://example.com/",
                status_code=200,
                json_ld=[{"@type": "Organization", "name": "Example"}],
            )
        ],
    )


def test_geo_optimizer_is_explicitly_unavailable_without_command() -> None:
    result = GeoOptimizerAdapter(command=None).collect(context())
    assert result.state is DataState.UNAVAILABLE
    assert "not configured" in result.warnings[0]


def test_geo_optimizer_uses_documented_json_audit_command(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(adapters_module.shutil, "which", lambda _command: "/usr/local/bin/geo")

    def successful_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"score": 84}', stderr="")

    monkeypatch.setattr(adapters_module.subprocess, "run", successful_run)
    result = GeoOptimizerAdapter(command="geo").collect(context())
    assert calls == [["geo", "audit", "--url", "https://example.com/", "--format", "json"]]
    assert result.state is DataState.AVAILABLE
    assert result.evidence[0].observed_value == {"score": 84}


def test_structured_data_adapter_normalizes_json_ld() -> None:
    result = StructuredDataAdapter().collect(context())
    assert result.state is DataState.AVAILABLE
    assert result.evidence[0].observed_value["@type"] == "Organization"


def test_future_adapter_capabilities_are_explicit() -> None:
    assert {
        "screaming-frog",
        "lighthouse",
        "gsc",
        "bing-ai-performance",
        "ga4",
        "server-logs",
        "ahrefs",
        "semrush",
        "openai-grounded-search",
        "gemini-grounding",
        "perplexity",
        "claude-search",
    } <= FUTURE_CAPABILITIES
