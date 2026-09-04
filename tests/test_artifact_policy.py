from datetime import UTC, date, datetime
from typing import Any

import pytest

from ai_search_audit.artifact_policy import classification_version, saved_prompt_version
from ai_search_audit.models import AIPrompt, AuditRun, Site


def _run(
    configuration: dict[str, Any] | None = None,
    prompt_versions: tuple[str, ...] = (),
) -> AuditRun:
    return AuditRun(
        audit_id="audit-example",
        audit_engine_version="0.2.0",
        site=Site(domain="studio.example", base_url="https://studio.example"),
        ruleset_version="test",
        ruleset_verified_date=date(2026, 9, 3),
        timestamp=datetime(2026, 9, 3, tzinfo=UTC),
        scores=[],
        configuration=configuration or {},
        ai_prompts=[
            AIPrompt(
                prompt_id=f"prompt-{index}",
                pack_version=version,
                locale="en",
                intent="discovery",
                text="Which design studios offer this service?",
            )
            for index, version in enumerate(prompt_versions)
        ],
    )


@pytest.mark.parametrize(
    ("configuration", "expected"),
    [
        ({}, "1.0.0"),
        ({"entity_classification_policy": "1.0.0"}, "1.0.0"),
        ({"entity_classification_policy": "2.0.0"}, "2.0.0"),
    ],
)
def test_classification_version_reads_saved_policy(configuration, expected):
    run = _run(configuration)
    before = run.model_dump_json()

    assert classification_version(run) == expected
    assert run.model_dump_json() == before


@pytest.mark.parametrize("marker", [None, "", "3.0.0", "1.1.0", 1, True, [], {}])
def test_classification_version_rejects_invalid_explicit_marker(marker):
    with pytest.raises(ValueError, match="unsupported entity classification policy"):
        classification_version(_run({"entity_classification_policy": marker}))


@pytest.mark.parametrize(
    ("configuration", "versions", "expected"),
    [
        ({}, (), "1.1.0"),
        ({}, ("1.1.0",), "1.1.0"),
        ({}, ("2.0.0",), "2.0.0"),
        ({}, ("1.1.0", "1.1.0"), "1.1.0"),
        ({}, ("2.0.0", "2.0.0"), "2.0.0"),
        ({"prompt_pack_version": "1.1.0"}, (), "1.1.0"),
        ({"prompt_pack_version": "2.0.0"}, (), "2.0.0"),
        ({"prompt_pack_version": "1.1.0"}, ("1.1.0",), "1.1.0"),
        ({"prompt_pack_version": "2.0.0"}, ("2.0.0",), "2.0.0"),
    ],
)
def test_saved_prompt_version_reads_artifact_not_generator(configuration, versions, expected):
    run = _run(configuration, versions)
    before = run.model_dump_json()

    assert saved_prompt_version(run) == expected
    assert run.model_dump_json() == before


@pytest.mark.parametrize("versions", [(), ("1.1.0",), ("2.0.0",)])
@pytest.mark.parametrize("marker", [None, "", "3.0.0", "1.0.0", 1, True, [], {}])
def test_saved_prompt_version_rejects_invalid_explicit_marker(marker, versions):
    with pytest.raises(ValueError, match="prompt"):
        saved_prompt_version(_run({"prompt_pack_version": marker}, versions))


@pytest.mark.parametrize(
    ("configuration", "versions"),
    [
        ({}, ("1.1.0", "2.0.0")),
        ({"prompt_pack_version": "1.1.0"}, ("1.1.0", "2.0.0")),
        ({"prompt_pack_version": "1.1.0"}, ("2.0.0",)),
        ({"prompt_pack_version": "2.0.0"}, ("1.1.0",)),
        ({}, ("",)),
        ({}, ("3.0.0",)),
        ({}, ("1.0.0",)),
    ],
)
def test_saved_prompt_version_rejects_mixed_unknown_or_mismatched_versions(configuration, versions):
    with pytest.raises(ValueError, match="prompt"):
        saved_prompt_version(_run(configuration, versions))
