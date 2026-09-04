import socket
import urllib.request

import pytest
from pydantic import ValidationError

from ai_search_audit.content_sections import extract_sections, normalize_text
from ai_search_audit.diagnostic_models import ExtractedContent, Section


def test_sections_preserve_qualifications_and_ignore_navigation():
    html = (
        "<nav>Menu 999</nav><main><h2>Warunki</h2>"
        "<p>Usługa może potrwać 2 dni. Nie jest gwarantowana.</p>"
        "<h3>Zakres</h3><ul><li>Obsługa w Polsce</li></ul></main>"
    )
    parsed = extract_sections(html, capture_id="capture-1")
    assert "Menu" not in parsed.text
    assert "może potrwać 2 dni" in parsed.text
    assert "Nie jest gwarantowana" in parsed.text
    assert [item.level for item in parsed.sections] == [2, 3]
    assert parsed.sections[1].heading_path == ("Warunki", "Zakres")
    assert parsed == extract_sections(html, capture_id="capture-1")


def test_normalize_text_preserves_case_numbers_units_negation_and_modality():
    assert normalize_text("  NIE\u0301  może\ttrwać  2 kg. ") == "NIÉ może trwać 2 kg."


def test_all_top_level_articles_are_extracted_without_double_counting_nested_content():
    html = """
        <article><h2>Oferta</h2><p>Usługa A</p></article>
        <article><h2>Pomoc</h2><article><p>Zagnieżdżony szczegół</p></article></article>
    """
    parsed = extract_sections(html, capture_id="capture-articles")

    assert [section.locator for section in parsed.sections] == [
        "root-0/heading-0",
        "root-1/heading-0",
    ]
    assert parsed.text.count("Zagnieżdżony szczegół") == 1
    assert parsed.sections[1].text == "Zagnieżdżony szczegół"
    assert not parsed.fallback_used
    assert parsed.limitations == ()


def test_nested_headings_lists_and_table_cells_keep_context_and_boundaries():
    html = """
        <main>
          <h1>Oferta</h1><p>Wstęp</p>
          <h2>Zakres</h2><ul><li>Polska<ul><li>Warszawa</li></ul></li></ul>
          <h3>Cennik</h3><table><tr><th>Plan</th><th>Czas</th></tr>
          <tr><td>Start</td><td>2 dni</td></tr></table>
        </main>
    """
    parsed = extract_sections(html, capture_id="capture-structure")

    assert [section.heading_path for section in parsed.sections] == [
        ("Oferta",),
        ("Oferta", "Zakres"),
        ("Oferta", "Zakres", "Cennik"),
    ]
    assert parsed.sections[1].text.splitlines() == ["Polska", "Warszawa"]
    assert parsed.sections[2].text.splitlines() == ["Plan | Czas", "Start | 2 dni"]
    assert parsed.sections[2].block_kinds == ("table_row", "table_row")


def test_hidden_content_is_excluded_and_body_fallback_is_explicit():
    html = """
        <html><body><p>Visible</p><p hidden>Hidden attribute</p>
        <p aria-hidden="true">ARIA hidden</p><p style="display: none">CSS hidden</p>
        <p style="visibility:hidden">Also hidden</p></body></html>
    """
    parsed = extract_sections(html, capture_id="capture-fallback")

    assert parsed.text == "Visible"
    assert parsed.sections[0].level == 0
    assert parsed.fallback_used
    assert parsed.limitations == ("No main or article container found; extracted body fallback.",)


def test_unheaded_direct_text_is_retained_before_the_first_heading():
    parsed = extract_sections(
        "<main>Unheaded intro <span>text</span><h2>Title</h2><p>Body</p></main>",
        capture_id="capture-intro",
    )

    assert parsed.sections[0].level == 0
    assert parsed.sections[0].text == "Unheaded intro text"
    assert parsed.text == "Unheaded intro text\nTitle\nBody"


def test_inline_markup_does_not_invent_whitespace_inside_a_number():
    parsed = extract_sections(
        "<main><h2>Cennik</h2><p>Cost: 1<span>2</span> EUR</p></main>",
        capture_id="capture-inline-number",
    )

    assert parsed.sections[0].text == "Cost: 12 EUR"


def test_generic_block_boundaries_and_line_breaks_preserve_word_boundaries():
    parsed = extract_sections(
        "<main><div>A</div><div>B</div><p>not<br>guaranteed</p></main>",
        capture_id="capture-block-boundaries",
    )

    assert parsed.text == "A\nB\nnot guaranteed"


def test_nested_generic_blocks_keep_their_document_order_boundaries():
    parsed = extract_sections(
        "<main><div><div>A</div><div>B</div></div></main>",
        capture_id="capture-nested-block-boundaries",
    )

    assert parsed.text == "A\nB"


def test_list_item_line_break_preserves_a_word_boundary():
    parsed = extract_sections(
        "<main><h2>Warunki</h2><ul><li>not<br>guaranteed</li></ul></main>",
        capture_id="capture-list-break",
    )

    assert parsed.sections[0].text == "not guaranteed"


def test_list_text_before_and_after_a_nested_list_stays_in_document_order():
    parsed = extract_sections(
        "<main><ul><li>Before<ul><li>Inside</li></ul>After</li></ul></main>",
        capture_id="capture-nested-list-order",
    )

    assert parsed.text == "Before\nInside\nAfter"
    assert parsed.sections[0].text == "Before\nInside\nAfter"


def test_loose_text_line_break_preserves_a_word_boundary():
    parsed = extract_sections(
        "<main>not<br>guaranteed</main>",
        capture_id="capture-loose-break",
    )

    assert parsed.text == "not guaranteed"


def test_nested_table_cells_keep_inner_cell_boundaries_explicit():
    parsed = extract_sections(
        "<main><table><tr><td>A<table><tr><td>1</td><td>2</td></tr></table>Z</td>"
        "<td>B</td></tr></table></main>",
        capture_id="capture-nested-table",
    )

    assert parsed.text == "A / 1 | 2 / Z | B"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<main><p>not</p><p>guaranteed</p></main>", "not\nguaranteed"),
        (
            "<main><a href='/offer'><div>not</div><div>guaranteed</div></a></main>",
            "not\nguaranteed",
        ),
        (
            "<main><table><tr><td><p>not</p><p>guaranteed</p></td><td>kg</td></tr></table></main>",
            "not / guaranteed | kg",
        ),
        ("<main><a href='/offer'><p>1<span>2</span> kg</p></a></main>", "12 kg"),
    ],
)
def test_shared_block_boundary_policy_preserves_inline_adjacency(html, expected):
    parsed = extract_sections(html, capture_id="capture-shared-boundaries")

    assert parsed.text == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            "<main><section><article>1</article><article>2</article> kg</section></main>",
            "1\n2\nkg",
        ),
        (
            "<main><div><dl><dt>Plan</dt><dd>2 kg</dd></dl></div></main>",
            "Plan\n2 kg",
        ),
    ],
)
def test_standard_text_only_block_elements_keep_boundaries_through_wrappers(html, expected):
    parsed = extract_sections(html, capture_id="capture-standard-blocks")

    assert parsed.text == expected


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("/* display:none; */ color:red", "Visible\nSentinel"),
        ("--display:none; color:red", "Visible\nSentinel"),
        ("display:none; display:block", "Visible\nSentinel"),
        ("display:none; display:inline flex", "Visible\nSentinel"),
        ("display:none; display:var(--layout)", "Visible\nSentinel"),
        ("visibility:hidden; visibility:visible", "Visible\nSentinel"),
        ("display:none !important; display:block", "Sentinel"),
        ("display:none !important; display:block !important", "Visible\nSentinel"),
        ("display:none", "Sentinel"),
    ],
)
def test_hidden_style_detection_parses_only_effective_inline_declarations(style, expected):
    parsed = extract_sections(
        f'<main><p style="{style}">Visible</p><p>Sentinel</p></main>',
        capture_id="capture-style",
    )

    assert parsed.text == expected


@pytest.mark.parametrize(
    "style",
    [
        "--note:';display:none;'; color:red",
        "content:';visibility:hidden;'; color:red",
        "background-image:url('https://assets.example/;display:none;')",
        "--theme:{display:block;display:none;};color:red",
        "color:red; /* comment;display:none",
        "dis/**/play:none",
    ],
)
def test_ambiguous_inline_style_never_proves_content_hidden(style):
    parsed = extract_sections(
        f'<main><p style="{style}">Visible</p><p>Sentinel</p></main>',
        capture_id="capture-ambiguous-style",
    )

    assert parsed.text == "Visible\nSentinel"


def test_comments_are_not_treated_as_visible_source_content():
    parsed = extract_sections(
        "<main><!-- hidden editorial note --><p>Visible</p></main>",
        capture_id="capture-comment",
    )

    assert parsed.text == "Visible"


def test_hidden_parent_is_removed_without_touching_detached_descendants():
    parsed = extract_sections(
        "<main><div hidden><p>Private child</p></div><p>Public content</p></main>",
        capture_id="capture-hidden-parent",
    )

    assert parsed.text == "Public content"


def test_table_rows_preserve_empty_cell_boundaries():
    parsed = extract_sections(
        "<main><h2>Table</h2><table><tr><td>1</td><td></td><td>3</td></tr></table></main>",
        capture_id="capture-table-boundaries",
    )

    assert parsed.sections[0].text == "1 |  | 3"


def test_identical_headings_heading_only_sections_and_partial_input_remain_distinct():
    parsed = extract_sections(
        "<main><h2>Kontakt</h2><h2>Kontakt</h2><p>Odpowiedź</p></main>",
        capture_id="capture-partial",
        complete=False,
    )

    assert [section.heading for section in parsed.sections] == ["Kontakt", "Kontakt"]
    assert [section.locator for section in parsed.sections] == [
        "root-0/heading-0",
        "root-0/heading-1",
    ]
    assert parsed.sections[0].text == ""
    assert parsed.sections[0].block_kinds == ()
    assert parsed.sections[0].section_id != parsed.sections[1].section_id
    assert parsed.limitations == ("Input capture is explicitly partial.",)


def test_imported_markup_is_inert_and_never_requests_remote_resources(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("extraction must not make a network request")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    parsed = extract_sections(
        "<main><script>window.location='https://attacker.example/'</script>"
        "<img src='https://assets.example/pixel.png'><h2>Bezpieczne</h2><p>Treść</p></main>",
        capture_id="capture-inert",
    )

    assert parsed.text == "Bezpieczne\nTreść"
    assert "attacker" not in parsed.text


def test_extracted_models_are_deeply_immutable():
    section = Section(
        section_id="section-1",
        capture_id="capture-1",
        locator="root-0/heading-0",
        level=2,
        heading="Warunki",
        heading_path=["Warunki"],
        text="Treść",
        block_kinds=["paragraph"],
    )
    extracted = ExtractedContent(
        capture_id="capture-1",
        text="Warunki\nTreść",
        sections=[section],
        fallback_used=False,
        limitations=[],
    )

    assert isinstance(extracted.sections, tuple)
    assert isinstance(section.heading_path, tuple)
    with pytest.raises(ValidationError):
        extracted.text = "changed"
    with pytest.raises(TypeError):
        section.heading_path[0] = "changed"
