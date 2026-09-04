import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "encoded",
    ["!!!", "/w==", "eA==" * 900, base64.b64encode(b"x" * 603).decode()],
    ids=["malformed", "invalid-utf8", "invalid-padding", "invalid-json"],
)
def test_invalid_literal_paragraph_rejects_render(tmp_path, encoded):
    _, source, _, _, command = render_edition(tmp_path, hero=False)
    command += ["--observation-projection-version", "1.1.0"]
    source.write_text("## Evidence\n\n<!-- audit-literal-v1:" + encoded + " -->\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "literal paragraph" in result.stderr


@pytest.mark.parametrize(
    "change",
    [
        "text-bound",
        "url-scheme",
        "url-userinfo",
        "url-control",
        "dangling",
        "bool-offset",
        "outside",
        "out-of-order",
        "too-many-spans",
        "too-many-urls",
        "extra-field",
    ],
)
def test_literal_paragraph_enforces_bounded_typed_citation_inventory(change):
    value = {"text": "Example", "urls": ["https://studio.example/"], "citations": [[0, 7, 0]]}
    if change == "text-bound":
        value["text"] = "x" * 601
    elif change.startswith("url-"):
        value["urls"] = [
            {
                "url-scheme": "javascript:alert(1)",
                "url-userinfo": "https://user@studio.example/",
                "url-control": "https://studio.example/\n",
            }[change]
        ]
    elif change == "dangling":
        value["citations"] = [[0, 7, 1]]
    elif change == "bool-offset":
        value["citations"] = [[False, 7, 0]]
    elif change == "outside":
        value["citations"] = [[0, 8, 0]]
    elif change == "out-of-order":
        value["citations"] = [[0, 7, 0], [0, 1, 0]]
    elif change == "too-many-spans":
        value["citations"] = [[0, 7, 0]] * 201
    elif change == "too-many-urls":
        value["urls"] = [f"https://source-{n}.example/" for n in range(6)]
    else:
        value["extra"] = "untrusted"
    encoded = base64.b64encode(json.dumps(value).encode()).decode()
    render = run_path(str(ROOT / "scripts/render_client_pdf.py"))["literal_paragraph_markup"]
    with pytest.raises(ValueError, match="literal paragraph"):
        render("<!-- audit-literal-v1:" + encoded + " -->")


def test_literal_marker_and_backslashes_are_inert_without_opt_in(tmp_path):
    _, source, output, _, command = render_edition(tmp_path, hero=False)
    source.write_text("## Evidence\n\n<!-- audit-literal-v1:!!! -->\n\nliteral \\u002a\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = " ".join(page.extract_text() for page in PdfReader(output).pages)
    assert "audit-literal-v1:!!!" in text and r"literal \u002a" in text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("**may help, not guaranteed**", "<b>may help, not guaranteed</b>"),
        ("`x**2 ** literal`", '<font name="Courier">x**2 ** literal</font>'),
        ("`**literal**`", '<font name="Courier">**literal**</font>'),
        (r"\**literal**", r"\**literal**"),
        ("x**2 ** `", "x**2 ** `"),
        ("<b>not HTML</b> @@TOKEN0@@", "&lt;b&gt;not HTML&lt;/b&gt; @@TOKEN0@@"),
        ("[other](https://other.example/)", "[other](https://other.example/)"),
    ],
)
def test_literal_typography_preserves_math_escaped_delimiters_and_code(text, expected):
    render = run_path(str(ROOT / "scripts/render_client_pdf.py"))["literal_inline_text"]
    assert render(text) == expected


def test_measurement_binding_marker_is_inert_but_other_comments_remain_visible(tmp_path):
    _, source, output, _, command = render_edition(tmp_path, hero=False)
    digest = "abcdef0123456789" * 4
    source.write_text(
        "## Measurements\nBefore marker.\n"
        f"<!-- audit-measurement:{digest} -->\nAfter marker.\n\n"
        "<!-- ordinary-visible -->\n\n<!-- audit-measurement:malformed -->\n"
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = " ".join(page.extract_text() for page in PdfReader(output).pages)
    assert digest not in text
    assert "Before marker." in text and "After marker." in text
    assert "ordinary-visible" in text and "audit-measurement:malformed" in text


@pytest.mark.parametrize(
    "variant",
    ["nonbreaking_hyphen", "nbsp", "ordinary", "uppercase", "inline"],
)
def test_only_original_exact_measurement_marker_can_disappear_from_pdf(tmp_path, variant):
    _, source, output, _, command = render_edition(tmp_path, hero=False)
    canonical = "<!-- audit-measurement:" + "a" * 64 + " -->"
    malformed = "<!-- audit-measurement:" + "b" * 64 + " -->"
    malformed = {
        "nonbreaking_hyphen": malformed.replace("audit-measurement", "audit\u2011measurement"),
        "nbsp": malformed.replace("<!-- ", "<!--\u00a0"),
        "ordinary": malformed.replace("audit-measurement", "ordinary-comment"),
        "uppercase": malformed.replace("audit-measurement", "AUDIT-MEASUREMENT"),
        "inline": "Visible prefix " + malformed + " visible suffix",
    }[variant]
    source.write_text(
        "## Measurements\nBefore marker.\n"
        + canonical
        + "\nAfter marker.\n"
        + malformed
        + "\nAfter malformed marker.\n"
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = " ".join(page.extract_text() for page in PdfReader(output).pages)
    text = " ".join(text.split())
    assert "a" * 64 not in text
    assert "b" * 64 in text
    assert "Before marker." in text and "After marker." in text
    assert "After malformed marker." in text
    if variant == "inline":
        assert "Visible prefix" in text and "visible suffix" in text


def render_edition(tmp_path: Path, *, locale: str = "en", hero: bool = True):
    source = tmp_path / "client.md"
    source.write_text(
        "# Example Studio\n\n## Executive decision\n\n"
        "Observation: Example Studio provides software services.\n\n"
        "## Sources\n\nhttps://studio.example\n",
        encoding="utf-8",
    )
    output = tmp_path / "client.pdf"
    image = tmp_path / "hero.png"
    if hero:
        Image.new("RGB", (1200, 800), "#173D34").save(image)
    command = [
        sys.executable,
        str(ROOT / "scripts/render_client_pdf.py"),
        str(source),
        str(output),
        "--client",
        "Example Studio",
        "--date",
        "2026-09-02",
        "--version",
        "public-v1",
        "--locale",
        locale,
    ]
    command += (
        ["--hero", str(image)] if hero else ["--no-hero-reason", "No suitable controlled image"]
    )
    return subprocess.run(command, capture_output=True, text=True), source, output, image, command


@pytest.mark.parametrize(
    "locale,label,absent",
    [("en", "DIGITAL VISIBILITY AUDIT", "Wersja"), ("pl", "AUDYT WIDOCZNOŚCI CYFROWEJ", "Version")],
)
def test_editorial_renderer_binds_source_image_and_locale(tmp_path, locale, label, absent):
    result, source, output, hero, command = render_edition(tmp_path, locale=locale)
    assert result.returncode == 0, result.stderr
    pdf = PdfReader(output)
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert label in text
    assert absent not in text
    assert len(pdf.pages[0].images) == 1
    assert pdf.metadata["/ClientEditionTemplate"] == "editorial-v1"
    assert (
        pdf.metadata["/ClientEditionSourceSHA256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert pdf.metadata["/ClientEditionHeroSHA256"] == hashlib.sha256(hero.read_bytes()).hexdigest()
    assert pdf.metadata["/ClientEditionLocale"] == locale
    assert pdf.metadata["/ClientEditionVersion"] == "public-v1"
    first = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    assert output.read_bytes() == first


def test_missing_explicit_cover_image_does_not_silently_render_typographic_cover(tmp_path):
    source = tmp_path / "client.md"
    source.write_text("# Example\n\n## Decision\nText\n")
    output = tmp_path / "client.pdf"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_client_pdf.py"),
            str(source),
            str(output),
            "--client",
            "Example",
            "--date",
            "2026-09-02",
            "--hero",
            str(tmp_path / "missing.png"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_explicit_typographic_cover_has_no_image_and_records_reason(tmp_path):
    result, _, output, _, _ = render_edition(tmp_path, hero=False)
    assert result.returncode == 0, result.stderr
    pdf = PdfReader(output)
    assert not pdf.pages[0].images
    assert pdf.metadata["/ClientEditionNoHeroReason"] == "No suitable controlled image"


def test_renderer_records_explicit_audit_identity(tmp_path):
    _, _, output, _, command = render_edition(tmp_path)
    result = subprocess.run(
        command + ["--audit-id", "audit-example"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert PdfReader(output).metadata["/ClientEditionAuditID"] == "audit-example"


def test_client_markdown_renderer_creates_readable_pdf(tmp_path: Path) -> None:
    source = tmp_path / "client-audit.md"
    output = tmp_path / "client-audit.pdf"
    source.write_text(
        """# Example Lakeside Hotel: audit

## Executive decision

Observation: The hotel's official website clearly identifies the property and location.

Source: https://example.com/hotel
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_client_pdf.py",
            str(source),
            str(output),
            "--client",
            "Example Lakeside Hotel",
            "--date",
            "2026-08-31",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 5_000

    from pypdf import PdfReader

    reader = PdfReader(output)
    assert len(reader.pages) >= 2
    assert reader.metadata is not None
    assert reader.metadata.creator == "auditing-seo-geo-ai-search"

    text = "\n".join(page.extract_text() or "" for page in reader.pages).casefold()
    assert "Example Lakeside Hotel".casefold() in text
    assert "Executive decision".casefold() in text
