import pytest
from pydantic import ValidationError

from ai_search_audit.diagnostic_models import DiagnosticBinding
from ai_search_audit.models import AIPrompt


def _binding_fields():
    return dict(
        project_id="example",
        source_version="public-v1",
        audit_id="audit-1",
        report_locale="pl",
        domain="studio.example",
        source_sha256="a" * 64,
    )


def test_binding_is_frozen_and_forbids_status_from_input():
    fields = _binding_fields()
    binding = DiagnosticBinding(**fields)
    with pytest.raises(ValidationError):
        binding.domain = "other.example"
    with pytest.raises(ValidationError):
        DiagnosticBinding(**fields, report_status="CLIENT_VALIDATED")


@pytest.mark.parametrize("field", ["project_id", "source_version", "audit_id"])
@pytest.mark.parametrize("value", ["../escape", "", "a/b", "CON"])
def test_binding_rejects_unsafe_identifiers(field, value):
    fields = _binding_fields()
    fields[field] = value
    with pytest.raises(ValidationError, match="safe identifier"):
        DiagnosticBinding(**fields)


def test_binding_normalizes_domain():
    fields = _binding_fields()
    fields["domain"] = "https://STUDIO.example/"
    assert DiagnosticBinding(**fields).domain == "studio.example"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "https://user:secret@studio.example"),
        ("report_locale", "de"),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("source_sha256", "z" * 64),
    ],
)
def test_binding_rejects_invalid_identity_values(field, value):
    fields = _binding_fields()
    fields[field] = value
    with pytest.raises(ValidationError):
        DiagnosticBinding(**fields)


def test_frozen_prompt_preserves_all_legacy_fields_without_mutable_aliases():
    from ai_search_audit.diagnostic_models import FrozenPrompt

    prompt = AIPrompt(
        prompt_id="prompt-1",
        pack_version="pack-1",
        locale="pl",
        intent="discovery",
        text="Which studio should I choose?",
        target_entities=["Studio"],
        query_themes=["studios"],
        expected_evidence_needs=["source"],
        suggested_providers=["manual"],
    )
    frozen = FrozenPrompt.model_validate(prompt.model_dump())

    assert set(FrozenPrompt.model_fields) == set(AIPrompt.model_fields)
    assert frozen.model_dump(mode="json") == prompt.model_dump(mode="json")
    for field in (
        "target_entities",
        "query_themes",
        "expected_evidence_needs",
        "suggested_providers",
    ):
        value = getattr(frozen, field)
        assert isinstance(value, tuple)
        getattr(prompt, field).append("changed")
        assert "changed" not in value
        with pytest.raises(TypeError):
            value[0] = "changed"
    with pytest.raises(ValidationError):
        frozen.text = "changed"


def test_source_copies_mutable_inputs_into_deeply_frozen_values():
    from ai_search_audit.diagnostic_models import DiagnosticSource

    prompts = [
        dict(
            prompt_id="prompt-1",
            pack_version="pack-1",
            locale="pl",
            intent="discovery",
            text="Which studio?",
            target_entities=["Studio"],
        )
    ]
    page_urls = ["https://studio.example/"]
    canonical_domains = ["studio.example"]
    source = DiagnosticSource(
        binding=_binding_fields(),
        prompts=prompts,
        page_urls=page_urls,
        canonical_domains=canonical_domains,
    )
    prompts[0]["target_entities"].append("changed")
    page_urls.append("https://other.example/")
    canonical_domains.append("other.example")

    assert source.prompts[0].target_entities == ("Studio",)
    assert source.page_urls == ("https://studio.example/",)
    assert source.canonical_domains == ("studio.example",)
    assert isinstance(source.prompts, tuple)
    with pytest.raises(ValidationError):
        source.binding.domain = "other.example"
    with pytest.raises(ValidationError):
        source.prompts[0].text = "changed"
    with pytest.raises(ValidationError):
        source.page_urls = ()
