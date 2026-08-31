from datetime import UTC, datetime

from ai_search_audit.models import DataState, Site
from ai_search_audit.prompts import PROMPT_PACK_VERSION, generate_prompt_pack, import_observations
from ai_search_audit.scoring import observed_ai_visibility


def test_prompt_pack_exists_without_grounded_provider() -> None:
    prompts = generate_prompt_pack(
        Site(
            domain="lakeside-hotel.example",
            base_url="https://lakeside-hotel.example",
            brand="Example Lakeside Hotel",
            languages=["pl", "en"],
        ),
        entity_type="hotel",
        location="Example Bay, Poland",
    )
    assert len(prompts) >= 10
    assert {prompt.locale for prompt in prompts} == {"pl", "en"}
    assert all(prompt.pack_version == PROMPT_PACK_VERSION for prompt in prompts)
    assert len({prompt.prompt_id for prompt in prompts}) == len(prompts)
    assert all(prompt.query_themes for prompt in prompts)
    polish = [prompt.text for prompt in prompts if prompt.locale == "pl"]
    english = [prompt.text for prompt in prompts if prompt.locale == "en"]
    assert any("Czym jest" in text for text in polish)
    assert any("What is" in text for text in english)
    assert observed_ai_visibility([]).state is DataState.UNAVAILABLE


def test_prompt_pack_is_domain_neutral_for_non_hospitality_entities() -> None:
    prompts = generate_prompt_pack(
        Site(
            domain="flowforge.example",
            base_url="https://flowforge.example",
            brand="FlowForge",
            languages=["en"],
        ),
        entity_type="SoftwareApplication",
        category="workflow automation software",
        location="Europe",
        content_themes=["workflow automation", "security"],
    )
    text = " ".join(prompt.text for prompt in prompts).casefold()
    assert "workflow automation" in text
    assert "europe" in text
    for leaked_assumption in ("premium private", "hotel", "rooms", "hospitality"):
        assert leaked_assumption not in text


def test_interactive_observations_can_be_imported() -> None:
    observations = import_observations(
        [
            {
                "observation_id": "o1",
                "prompt_id": "p1",
                "provider": "interactive-openai-search",
                "observed_at": datetime(2026, 8, 11, tzinfo=UTC).isoformat(),
                "citations": ["https://example.com"],
                "brand_mentioned": True,
                "grounded": True,
            }
        ]
    )
    assert observations[0].provider == "interactive-openai-search"
    assert observed_ai_visibility(observations).value == 100
