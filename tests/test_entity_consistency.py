from datetime import UTC, date, datetime
from pathlib import Path

from ai_search_audit.analyzers import build_entity_consistency, select_canonical_entity
from ai_search_audit.knowledge import load_registry
from ai_search_audit.models import Evidence, ExternalMention, FindingStatus, Page, Site


def test_entity_consistency_detects_conflict_across_independent_sources() -> None:
    mentions = [
        ExternalMention(
            mention_id="m1",
            url="https://press.example/story",
            publisher="Press",
            source_class="press",
            claims={"rooms": "24", "location": "Example Bay, Poland"},
            evidence_ids=["e1"],
            independent_source_key="press",
        ),
        ExternalMention(
            mention_id="m2",
            url="https://ota.example/listing",
            publisher="OTA",
            source_class="ota",
            claims={"rooms": "21", "location": "Example Bay, Poland"},
            evidence_ids=["e2"],
            independent_source_key="ota",
        ),
    ]
    profile, matrix, findings = build_entity_consistency(
        Site(
            domain="lakeside-hotel.example",
            base_url="https://lakeside-hotel.example",
            brand="Example Lakeside Hotel",
        ),
        mentions,
        load_registry(Path(__file__).parents[1] / "knowledge"),
        as_of=date(2026, 8, 11),
        entity_type="Hotel",
    )
    assert profile.brand == "Example Lakeside Hotel"
    assert matrix["room_count"] == {"24": ["e1"], "21": ["e2"]}
    conflict = findings[0]
    assert conflict.status is FindingStatus.CONFIRMED
    assert set(conflict.evidence_ids) == {"e1", "e2"}


def test_canonical_entity_prefers_relevant_supported_node_from_graph() -> None:
    site = Site(domain="example.com", base_url="https://example.com", brand="Example")
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        json_ld=[
            {
                "@context": "https://schema.org",
                "@graph": [
                    {
                        "@type": "Hotel",
                        "name": "Unrelated Hotel",
                        "url": "https://unrelated.example/",
                    },
                    {
                        "@type": "Organization",
                        "name": "Example Incorporated",
                        "url": "https://example.com/",
                        "telephone": "+48 123 456 789",
                    },
                    {"@type": "WebSite", "name": "Example website"},
                ],
            },
            {"@type": "BreadcrumbList", "name": "Last arbitrary JSON-LD name"},
        ],
    )
    entity = select_canonical_entity(site, [page])
    assert entity.brand == "Example Incorporated"
    assert entity.type == "Organization"
    assert entity.facts["telephone"] == ["+48123456789"]


def test_entity_fact_normalization_avoids_format_only_conflicts() -> None:
    mentions = [
        ExternalMention(
            mention_id="one",
            url="https://directory.example/one",
            publisher="Directory",
            source_class="directory",
            claims={"numberOfRooms": "10 rooms", "telephone": "+48 123 456 789"},
            evidence_ids=["e1"],
            independent_source_key="one",
        ),
        ExternalMention(
            mention_id="two",
            url="https://press.example/two",
            publisher="Press",
            source_class="press",
            claims={"room_count": "10", "telephone": "+48 (123) 456-789"},
            evidence_ids=["e2"],
            independent_source_key="two",
        ),
    ]
    _, matrix, findings = build_entity_consistency(
        Site(domain="example.com", base_url="https://example.com", brand="Example"),
        mentions,
        load_registry(Path(__file__).parents[1] / "knowledge"),
        as_of=date(2026, 8, 11),
    )
    assert matrix["room_count"] == {"10": ["e1", "e2"]}
    assert matrix["telephone"] == {"+48123456789": ["e1", "e2"]}
    assert findings == []


def test_official_entity_fact_conflicts_with_one_independent_source_and_keeps_urls() -> None:
    site = Site(domain="example.com", base_url="https://example.com", brand="Example")
    official_node = {
        "@type": "Hotel",
        "name": "Example Hotel",
        "url": "https://example.com/",
        "numberOfRooms": "24 rooms",
    }
    page = Page(
        url="https://example.com/",
        final_url="https://example.com/",
        status_code=200,
        json_ld=[{"@context": "https://schema.org", "@graph": [official_node]}],
    )
    canonical_entity = select_canonical_entity(site, [page])
    structured_evidence = Evidence(
        evidence_id="jsonld-official",
        source_url="https://example.com/",
        source_type="structured_data",
        collector="structured-data",
        observed_at=datetime(2026, 8, 11, tzinfo=UTC),
        observed_value={"@context": "https://schema.org", "@graph": [official_node]},
    )
    mention = ExternalMention(
        mention_id="ota",
        url="https://ota.example/listing",
        publisher="OTA",
        source_class="ota",
        claims={"rooms": "21"},
        evidence_ids=["external-room-count"],
        independent_source_key="ota",
    )

    _, matrix, findings = build_entity_consistency(
        site,
        [mention],
        load_registry(Path(__file__).parents[1] / "knowledge"),
        as_of=date(2026, 8, 11),
        entity_type="Hotel",
        canonical_entity=canonical_entity,
        canonical_evidence=[structured_evidence],
    )

    assert matrix["room_count"] == {
        "24": ["jsonld-official"],
        "21": ["external-room-count"],
    }
    conflict = findings[0]
    assert set(conflict.evidence_ids) == {"jsonld-official", "external-room-count"}
    assert set(conflict.affected_urls) == {
        "https://example.com/",
        "https://ota.example/listing",
    }
