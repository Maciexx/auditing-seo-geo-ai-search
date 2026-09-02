from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field
from weasyprint import CSS, HTML, default_url_fetcher  # type: ignore[import-untyped]
from weasyprint import __version__ as weasyprint_version
from weasyprint.text.ffi import pango  # type: ignore[import-untyped]

from .report_models import (
    ClientReportData,
    RendererMetadata,
    validate_report_compatibility,
)

TEMPLATE_VERSION = "1.2.0"
TEMPLATE_DIRECTORY_BY_VERSION = {TEMPLATE_VERSION: "v1"}
TEMPLATE_DIRECTORY = TEMPLATE_DIRECTORY_BY_VERSION[TEMPLATE_VERSION]
TEMPLATE_ROOT = Path(__file__).parent / "templates" / "client-report" / TEMPLATE_DIRECTORY
FONT_ROOT = TEMPLATE_ROOT / "assets" / "fonts"


class RenderSecurityError(ValueError):
    pass


class RendererEnvironmentError(ValueError):
    pass


class RenderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path
    sha256: str = Field(min_length=64, max_length=64)
    pdf_identifier: str = Field(min_length=64, max_length=64)
    byte_count: int = Field(gt=0)


def _digest_files(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _template_digest() -> str:
    paths = [
        path
        for path in TEMPLATE_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".html", ".css", ".json"}
    ]
    return _digest_files(paths, root=TEMPLATE_ROOT)


def _font_digest() -> str:
    return _digest_files(list(FONT_ROOT.glob("*.ttf")), root=FONT_ROOT)


def _environment_fingerprint() -> str:
    environment = {
        "machine": platform.machine(),
        "pango_version": int(pango.pango_version()),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "weasyprint_version": weasyprint_version,
    }
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_renderer_metadata() -> RendererMetadata:
    return RendererMetadata(
        renderer="WeasyPrint",
        renderer_version=weasyprint_version,
        template_version=TEMPLATE_VERSION,
        template_digest=_template_digest(),
        font_asset_digest=_font_digest(),
        environment_fingerprint=_environment_fingerprint(),
    )


def local_asset_fetcher(url: str, *args: object, **kwargs: object) -> dict[str, Any]:
    parts = urlsplit(url)
    if parts.scheme.casefold() != "file":
        raise RenderSecurityError(f"report renderer rejected non-local resource: {url}")
    path = Path(unquote(parts.path)).resolve()
    root = TEMPLATE_ROOT.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RenderSecurityError(f"report renderer rejected uncontrolled local resource: {url}")
    return cast(dict[str, Any], default_url_fetcher(url, *args, **kwargs))


def _jinja_environment() -> Environment:
    environment = Environment(
        loader=PackageLoader("ai_search_audit", f"templates/client-report/{TEMPLATE_DIRECTORY}"),
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["format_number"] = _format_number
    return environment


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".12g")


def _translations(locale: str) -> dict[str, Any]:
    path = TEMPLATE_ROOT / "locales" / f"{locale}.json"
    if not path.is_file():
        raise ValueError(f"unsupported report locale: {locale}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid report locale resource: {locale}")
    return cast(dict[str, Any], value)


def render_report_html(data: ClientReportData) -> str:
    if not isinstance(data, ClientReportData):
        raise TypeError("renderer accepts ClientReportData only")
    template = _jinja_environment().get_template("report.html")
    return template.render(data=data, t=_translations(data.report_locale))


def _localized_page_css(data: ClientReportData) -> CSS:
    translations = _translations(data.report_locale)
    footer = json.dumps(translations["footer_label"], ensure_ascii=False)
    page = json.dumps(translations["page"], ensure_ascii=False)
    return CSS(
        string=(
            "@page {"
            f"@bottom-left {{ content: {footer}; }}"
            f'@bottom-right {{ content: {page} " " counter(page) " / " counter(pages); }}'
            "}"
            "@page:first { @bottom-left { content: none; } "
            "@bottom-right { content: none; } }"
        )
    )


def _canonical_report_json(data: ClientReportData) -> bytes:
    return json.dumps(
        data.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _pdf_identifier(data: ClientReportData) -> bytes:
    payload = b"\0".join(
        (
            _canonical_report_json(data),
            data.renderer.template_digest.encode(),
            data.renderer.environment_fingerprint.encode(),
        )
    )
    return hashlib.sha256(payload).digest()


def _atomic_pdf(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def render_client_report(data: ClientReportData, output_path: Path) -> RenderResult:
    if not isinstance(data, ClientReportData):
        raise TypeError("renderer accepts ClientReportData only")
    validate_report_compatibility(
        schema_version=data.report_schema_version,
        template_version=data.report_template_version,
        renderer=data.renderer,
    )
    current_metadata = build_renderer_metadata()
    if data.renderer != current_metadata:
        raise RendererEnvironmentError(
            "ClientReportData renderer metadata does not match the active renderer environment"
        )
    html = render_report_html(data)
    identifier = _pdf_identifier(data)
    pdf = HTML(
        string=html,
        base_url=TEMPLATE_ROOT.as_uri(),
        url_fetcher=local_asset_fetcher,
    ).write_pdf(
        stylesheets=[
            CSS(
                filename=TEMPLATE_ROOT / "styles.css",
                url_fetcher=local_asset_fetcher,
            ),
            _localized_page_css(data),
        ],
        pdf_identifier=identifier,
        full_fonts=False,
        hinting=True,
    )
    if not isinstance(pdf, bytes):
        raise RuntimeError("WeasyPrint did not return PDF bytes")
    output_path = Path(output_path)
    _atomic_pdf(output_path, pdf)
    return RenderResult(
        output_path=output_path,
        sha256=hashlib.sha256(pdf).hexdigest(),
        pdf_identifier=identifier.hex(),
        byte_count=len(pdf),
    )
