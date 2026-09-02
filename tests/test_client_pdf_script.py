import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).parents[1]


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
