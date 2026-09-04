"""Offline regressions for the existing narrative-only callback boundary.

These tests do not provide a Gemini synthesis route or authenticate free-text provenance.
"""

import pytest
from pydantic import ConfigDict, HttpUrl, ValidationError

from ai_search_audit.diagnostic_models import BenchmarkResponseInput
from ai_search_audit.measurement_profile import MeasurementProfile
from ai_search_audit.models import DataState, ScoreResult, Site
from ai_search_audit.reports import (
    ClaimGuardError,
    NarrativeFindingRewrite,
    NarrativeRewritePayload,
    apply_anti_slop,
    build_protected_claims_manifest,
    build_report_draft,
)
from tests.test_reports import run_fixture


@pytest.fixture
def draft():
    run = run_fixture()
    run.site = Site(domain="studio.example", base_url="https://studio.example")
    run.evidence[0].source_url = HttpUrl("https://studio.example")
    run.scores = [
        ScoreResult(
            name="visibility", state=DataState.UNAVAILABLE, unavailable_reason="Not observed"
        )
    ]
    return build_report_draft(run)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("grounded_answer", "Restricted provider-derived answer"),
        ("search_suggestions", ["Restricted provider-derived suggestion"]),
        ("grounding_links", ["https://source.example/answer"]),
        ("derived_visibility_score", 90),
        ("evidence", [{"source_class": "press", "claims": {"claim": "Relabeled analysis"}}]),
        ("observedAIresponse", "Generated analysis"),
        ("observations", [{"provider": "gemini-grounding", "grounded": True}]),
        ("scores", [{"name": "visibility", "value": 90}]),
        ("report_status", "FINAL"),
        ("target_domain", "other.example"),
    ],
)
def test_callback_rejects_unchecked_non_narrative_fields(draft, nested, field, value):
    before = draft.model_dump_json()
    manifest = build_protected_claims_manifest(draft)

    def provider(payload):
        if nested:
            payload.findings[0] = payload.findings[0].model_copy(update={field: value})
            return payload
        return payload.model_copy(update={field: value})

    with pytest.raises(ClaimGuardError, match="invalid narrative payload"):
        apply_anti_slop(draft, manifest, provider=provider)
    assert draft.model_dump_json() == before


@pytest.mark.parametrize("nested", [False, True])
def test_callback_rejects_subclass_extras_without_dropping_them(draft, nested):
    class ExtendedPayload(NarrativeRewritePayload):
        model_config = ConfigDict(extra="allow")

    class ExtendedFinding(NarrativeFindingRewrite):
        model_config = ConfigDict(extra="allow")

    def provider(payload):
        if nested:
            payload.findings[0] = ExtendedFinding(
                **payload.findings[0].model_dump(), evidence="Unsupported source"
            )
            return payload
        return ExtendedPayload(**payload.model_dump(), evidence="Unsupported source")

    with pytest.raises(ClaimGuardError, match="invalid narrative payload"):
        apply_anti_slop(draft, build_protected_claims_manifest(draft), provider=provider)


@pytest.mark.parametrize("nested", [False, True])
def test_callback_revalidates_unchecked_narrative_types(draft, nested):
    def provider(payload):
        if nested:
            payload.findings[0] = payload.findings[0].model_copy(
                update={"client_explanation": {"source_class": "analysis"}}
            )
            return payload
        return payload.model_copy(update={"executive_summary": {"source_class": "analysis"}})

    with pytest.raises(ClaimGuardError, match="invalid narrative payload"):
        apply_anti_slop(draft, build_protected_claims_manifest(draft), provider=provider)


def test_narrative_result_is_not_an_observed_response_or_score(draft):
    manifest = build_protected_claims_manifest(draft)

    def provider(payload):
        payload.findings[0].client_title = "Verify the public room information"
        return payload

    result = apply_anti_slop(draft, manifest, provider=provider)
    assert result.draft.audit_id == draft.audit_id
    assert result.draft.target_domain == draft.target_domain
    assert result.draft.scores == draft.scores
    assert result.draft.evidence == draft.evidence
    assert result.draft.findings[0].status == draft.findings[0].status
    assert result.draft.findings[0].factual_claims == draft.findings[0].factual_claims
    with pytest.raises(ValidationError):
        BenchmarkResponseInput.model_validate(result)
    with pytest.raises(ValidationError):
        BenchmarkResponseInput.model_validate(result.model_dump(mode="python"))


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0"])
@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3.6-flash"])
def test_gemini_stays_unavailable_before_any_io_even_with_consent(
    tmp_path, monkeypatch, version, model
):
    from ai_search_audit import measurement_workflow

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Gemini must stop before source, credential, or provider I/O")

    monkeypatch.setattr(measurement_workflow, "load_diagnostic_source", forbidden)
    monkeypatch.setattr(measurement_workflow.os.environ, "get", forbidden)
    profile = MeasurementProfile(
        schema_version=version,
        gemini=True,
        gemini_model=model,
        paid_use_consent=True,
    )
    serialized = profile.model_dump_json()
    with pytest.raises(measurement_workflow.UnsupportedMeasurementProfile, match="Gemini"):
        measurement_workflow.run_measurements(
            "project:studio",
            clients_root=tmp_path,
            source_version="public-v1",
            profile=profile,
            on_preflight=forbidden,
        )
    assert MeasurementProfile.model_validate_json(serialized).model_dump_json() == serialized
    assert list(tmp_path.iterdir()) == []
