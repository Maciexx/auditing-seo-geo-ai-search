import pytest

from ai_search_audit.models import ClaimModality
from ai_search_audit.reports import (
    AntiSlopUnavailable,
    ClaimGuardError,
    apply_anti_slop,
    build_protected_claims_manifest,
    build_report_draft,
)
from tests.test_reports import run_fixture


def test_available_installed_skill_provider_is_attempted_first() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)
    calls: list[str] = []

    def provider(current):
        calls.append("installed-skill")
        return current

    result = apply_anti_slop(draft, manifest, provider=provider)
    assert calls == ["installed-skill"]
    assert result.route == "installed-anti-slop-skill"


def test_installed_skill_can_rewrite_client_prose_while_claims_remain_invariant() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)

    def provider(current):
        current.findings[0].client_title = "Verify the room total shown publicly"
        current.findings[0].client_explanation = "One listing needs a fact check."
        current.findings[
            0
        ].business_impact = "The discrepancy could make automated summaries less dependable."
        current.findings[0].implementation = "Confirm the total and update that listing."
        return current

    result = apply_anti_slop(draft, manifest, provider=provider)
    assert result.draft.findings[0].client_title == "Verify the room total shown publicly"
    assert result.draft.findings[0].factual_claims == draft.findings[0].factual_claims


def test_deterministic_fallback_runs_only_when_skill_unavailable() -> None:
    draft = build_report_draft(run_fixture())
    draft.executive_summary = "Moreover, this is an incredibly robust result."
    manifest = build_protected_claims_manifest(draft)

    def unavailable(current):
        raise AntiSlopUnavailable("agent skill unavailable in headless mode")

    result = apply_anti_slop(draft, manifest, provider=unavailable)
    assert result.route == "deterministic-fallback"
    assert "Moreover" not in result.draft.executive_summary
    assert result.agent_skill_state == "UNAVAILABLE"


def test_installed_skill_cannot_strengthen_narrative_certainty() -> None:
    draft = build_report_draft(run_fixture())
    manifest = build_protected_claims_manifest(draft)

    def provider(current):
        current.findings[
            0
        ].business_impact = "This discrepancy will definitely make summaries unreliable."
        return current

    with pytest.raises(ClaimGuardError, match="narrative certainty strengthened"):
        apply_anti_slop(draft, manifest, provider=provider)


def test_installed_skill_can_use_direct_wording_when_structured_claim_supports_it() -> None:
    draft = build_report_draft(run_fixture())
    impact_claim = draft.findings[0].factual_claims[1]
    impact_claim.modality = ClaimModality.ASSERTED
    impact_claim.value = "decreases"
    impact_claim.meaning = "The discrepancy reduces automated-summary reliability."
    manifest = build_protected_claims_manifest(draft)

    def provider(current):
        current.findings[
            0
        ].business_impact = "The discrepancy makes automated summaries less dependable."
        return current

    result = apply_anti_slop(draft, manifest, provider=provider)
    assert result.draft.findings[0].business_impact.startswith("The discrepancy makes")


def test_explicit_structured_certainty_can_authorize_equivalent_narrative() -> None:
    draft = build_report_draft(run_fixture())
    impact_claim = draft.findings[0].factual_claims[1]
    impact_claim.modality = ClaimModality.ASSERTED
    impact_claim.value = "will decrease"
    impact_claim.meaning = "The discrepancy will reduce automated-summary reliability."
    manifest = build_protected_claims_manifest(draft)

    def provider(current):
        current.findings[
            0
        ].business_impact = "The discrepancy will make automated summaries less dependable."
        return current

    result = apply_anti_slop(draft, manifest, provider=provider)
    assert "will" in result.draft.findings[0].business_impact


def test_installed_skill_receives_narrative_fields_without_brand_or_sitemap_state() -> None:
    draft = build_report_draft(run_fixture())
    draft.brand = "Protected Brand"
    manifest = build_protected_claims_manifest(draft)
    observed_keys: set[str] = set()

    def provider(payload):
        observed_keys.update(payload.model_fields_set)
        assert not hasattr(payload, "brand")
        assert not hasattr(payload, "sitemap_state")
        with pytest.raises((AttributeError, ValueError)):
            payload.brand = "Changed Brand"
        with pytest.raises((AttributeError, ValueError)):
            payload.sitemap_state = "CONFIRMED_ABSENT"
        return payload

    result = apply_anti_slop(draft, manifest, provider=provider)

    assert observed_keys == {"executive_summary", "findings"}
    assert result.draft.brand == "Protected Brand"
    assert result.draft.sitemap_state == draft.sitemap_state


def test_polish_cautious_language_cannot_be_rewritten_as_certain() -> None:
    draft = build_report_draft(run_fixture(), report_locale="pl")
    draft.findings[
        0
    ].business_impact = "Ta rozbieżność może obniżyć wiarygodność automatycznych podsumowań."
    manifest = build_protected_claims_manifest(draft)

    def provider(payload):
        payload.findings[
            0
        ].business_impact = "Ta rozbieżność na pewno obniży wiarygodność automatycznych podsumowań."
        return payload

    with pytest.raises(ClaimGuardError, match="narrative certainty strengthened"):
        apply_anti_slop(draft, manifest, provider=provider)
