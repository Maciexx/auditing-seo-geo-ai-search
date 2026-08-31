from ai_search_audit.adapters import AdapterContext, PublicResearchAdapter, PublicResearchItem
from ai_search_audit.config import AuditConfig
from ai_search_audit.models import DataState, Site


def test_public_research_normalizes_external_independent_sources() -> None:
    context = AdapterContext(
        site=Site(domain="lakeside-hotel.example", base_url="https://lakeside-hotel.example"),
        config=AuditConfig(domain="lakeside-hotel.example"),
    )
    items = [
        PublicResearchItem(
            url="https://press.example/sample-hotel",
            publisher="Travel Press",
            source_class="press",
            independent_source_key="travel-press",
            claims={"rooms": "24", "positioning": "boutique"},
        ),
        PublicResearchItem(
            url="https://ota.example/sample-hotel",
            publisher="OTA",
            source_class="ota",
            independent_source_key="ota",
            claims={"rooms": "21"},
        ),
    ]
    result = PublicResearchAdapter().collect(context, items=items)
    assert result.state is DataState.AVAILABLE
    assert {mention.source_class for mention in result.external_mentions} == {"press", "ota"}
    assert all(evidence.source_scope == "external" for evidence in result.evidence)


def test_public_research_deduplicates_syndicated_sources() -> None:
    context = AdapterContext(
        site=Site(domain="example.com", base_url="https://example.com"),
        config=AuditConfig(domain="example.com"),
    )
    items = [
        PublicResearchItem(
            url=f"https://copy{index}.example/story",
            publisher=f"Copy {index}",
            source_class="press",
            independent_source_key="wire-story-1",
            claims={"address": "Main Street"},
        )
        for index in range(2)
    ]
    result = PublicResearchAdapter().collect(context, items=items)
    assert len(result.external_mentions) == 1
    assert "syndicated duplicate" in result.warnings[0]


def test_public_research_without_inputs_is_unavailable_not_failed() -> None:
    context = AdapterContext(
        site=Site(domain="example.com", base_url="https://example.com"),
        config=AuditConfig(domain="example.com"),
    )
    result = PublicResearchAdapter().collect(context, items=[])
    assert result.state is DataState.UNAVAILABLE
