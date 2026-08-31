import subprocess
import sys
from pathlib import Path


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
