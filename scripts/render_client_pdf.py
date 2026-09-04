#!/usr/bin/env python3
"""Render a concise Markdown audit as a client-ready A4 PDF."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import html
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    ListFlowable,
    ListItem,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, PAGE_HEIGHT = A4
FOREST = colors.HexColor("#173D34")
FOREST_LIGHT = colors.HexColor("#2A5B4E")
GOLD = colors.HexColor("#A78849")
IVORY = colors.HexColor("#F4F0E7")
WARM_WHITE = colors.HexColor("#FBFAF7")
PALE_GREEN = colors.HexColor("#EEF3F0")
PALE_RED = colors.HexColor("#F6E9E6")
TEXT = colors.HexColor("#2E3331")
MUTED = colors.HexColor("#68706C")
GRID = colors.HexColor("#D5D8D4")

LABELS = {
    "pl": {
        "cover": "AUDYT WIDOCZNOŚCI CYFROWEJ",
        "version": "Wersja",
        "page": "Strona",
        "section": "SEKCJA",
        "scope": "Audyt zewnętrzny oparty na dostępnych dowodach",
        "subject": "Audyt widoczności w Google i wyszukiwarkach generatywnych",
        "author": "Audyt zewnętrzny",
        "confidentiality": "Poufne - materiał dla właścicieli",
    },
    "en": {
        "cover": "DIGITAL VISIBILITY AUDIT",
        "version": "Version",
        "page": "Page",
        "section": "SECTION",
        "scope": "External audit based on available evidence",
        "subject": "Visibility in Google and generative search",
        "author": "External audit",
        "confidentiality": "Confidential - for the owners",
    },
}


def first_existing(candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def register_fonts() -> dict[str, str]:
    serif = first_existing(
        [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ]
    )
    serif_bold = first_existing(
        [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        ]
    )
    sans = first_existing(
        [
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    sans_bold = first_existing(
        [
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    )

    fonts = {
        "serif": "Times-Roman",
        "serif_bold": "Times-Bold",
        "sans": "Helvetica",
        "sans_bold": "Helvetica-Bold",
        "mono": "Courier",
    }
    if serif:
        pdfmetrics.registerFont(TTFont("ClientSerif", serif))
        fonts["serif"] = "ClientSerif"
    if serif_bold:
        pdfmetrics.registerFont(TTFont("ClientSerifBold", serif_bold))
        fonts["serif_bold"] = "ClientSerifBold"
    if sans:
        pdfmetrics.registerFont(TTFont("ClientSans", sans))
        fonts["sans"] = "ClientSans"
    if sans_bold:
        pdfmetrics.registerFont(TTFont("ClientSansBold", sans_bold))
        fonts["sans_bold"] = "ClientSansBold"
    return fonts


def sanitize_text(value: str) -> str:
    return (
        value.replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )


def strip_markdown(value: str) -> str:
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*_`]", "", value)
    return sanitize_text(value).strip()


def inline_markup(value: str, fonts: dict[str, str]) -> str:
    value = sanitize_text(value)
    tokens: list[str] = []

    def hold_link(match: re.Match[str]) -> str:
        label = html.escape(strip_markdown(match.group(1)))
        url = html.escape(match.group(2), quote=True)
        tokens.append(f'<link href="{url}" color="#2A5B4E">{label}</link>')
        return f"@@TOKEN{len(tokens) - 1}@@"

    value = re.sub(r"\[([^]]+)]\((https?://[^)]+)\)", hold_link, value)
    value = html.escape(value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", rf'<font name="{fonts["mono"]}">\1</font>', value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    for index, token in enumerate(tokens):
        value = value.replace(f"@@TOKEN{index}@@", token)
    return value


def literal_inline_text(text: str) -> str:
    """Only paired typographic emphasis/code, never links, tags, tokens or layout."""
    pattern = re.compile(
        r"(?<![\w`\\])`([^`\n]+)(?<!\\)`(?![\w`])"
        r"|(?<![\w*\\])\*\*(?=\S)([^*\n]+?)(?<=\S)(?<!\\)\*\*(?![\w*])"
    )
    chunks = []
    position = 0
    for match in pattern.finditer(text):
        chunks.append(html.escape(text[position : match.start()]))
        if match.group(1) is not None:
            chunks.append('<font name="Courier">' + html.escape(match.group(1)) + "</font>")
        else:
            chunks.append("<b>" + html.escape(match.group(2)) + "</b>")
        position = match.end()
    chunks.append(html.escape(text[position:]))
    return "".join(chunks)


def literal_paragraph_markup(line: str) -> str:
    """Decode opt-in bounded evidence text only at the terminal text stage.

    No decoded character passes through general Markdown, layout or token substitution.
    Citation offsets refer to untouched Unicode text, including overlaps.
    """
    match = re.fullmatch(r"<!-- audit-literal-v1:([A-Za-z0-9+/=]{1,65536}) -->", line)
    try:
        if match is None:
            raise ValueError
        raw = base64.b64decode(match.group(1), validate=True)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or set(data) != {"text", "urls", "citations"}:
            raise ValueError
        text, urls, citations = data["text"], data["urls"], data["citations"]
        if not isinstance(text, str) or not 1 <= len(text) <= 600:
            raise ValueError
        if not isinstance(urls, list) or len(urls) > 5:
            raise ValueError
        for url in urls:
            if (
                not isinstance(url, str)
                or len(url) > 2083
                or "\\" in url
                or any(c.isspace() or ord(c) < 32 for c in url)
            ):
                raise ValueError
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError
            if parts.username is not None or parts.password is not None:
                raise ValueError
            _ = parts.port
        if len(set(urls)) != len(urls):
            raise ValueError
        if not isinstance(citations, list) or len(citations) > 200:
            raise ValueError
        for citation in citations:
            if (
                not isinstance(citation, list)
                or len(citation) != 3
                or any(type(v) is not int for v in citation)
                or not 0 <= citation[0] < citation[1] <= len(text)
                or not 0 <= citation[2] < len(urls)
            ):
                raise ValueError
        if citations != sorted(citations, key=lambda c: (c[1], c[0], c[2])) or {
            c[2] for c in citations
        } != set(range(len(urls))):
            raise ValueError
    except (ValueError, TypeError, UnicodeError, binascii.Error) as exc:
        raise ValueError("invalid observation literal paragraph") from exc
    chunks = ['"']
    position = 0
    for index, (start, end, source) in enumerate(citations):
        span = text[start:end]
        if span.startswith("(["):  # Conventional parenthesized provider citation.
            span = span[1:-1] if span.endswith("))") else span
        link = re.fullmatch(r"\[([^]\n]+)]\((https?://\S+)\)", span)
        overlaps = any(
            other != index and s < end and start < e for other, (s, e, _) in enumerate(citations)
        )
        if start >= position and not overlaps and link and link.group(2) == urls[source]:
            chunks.append(literal_inline_text(text[position:start]))
            label = html.escape(link.group(1))
        else:
            chunks.append(literal_inline_text(text[position:end]))
            label = f" [{source + 1}]"
        chunks.append(
            f'<link href="{html.escape(urls[source], quote=True)}" color="#2A5B4E">{label}</link>'
        )
        position = end
    chunks.extend([literal_inline_text(text[position:]), '"'])
    return "".join(chunks)


def make_styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_label": ParagraphStyle(
            "CoverLabel",
            parent=base["Normal"],
            fontName=fonts["sans_bold"],
            fontSize=8.2,
            leading=10,
            textColor=GOLD,
            spaceAfter=5 * mm,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=fonts["serif_bold"],
            fontSize=25,
            leading=29,
            textColor=FOREST,
            alignment=TA_LEFT,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=fonts["serif"],
            fontSize=12,
            leading=17,
            textColor=FOREST,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName=fonts["serif"],
            fontSize=24,
            leading=28,
            textColor=FOREST,
            spaceBefore=1 * mm,
            spaceAfter=7 * mm,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName=fonts["sans_bold"],
            fontSize=12.2,
            leading=15,
            textColor=FOREST_LIGHT,
            spaceBefore=4.5 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=fonts["sans"],
            fontSize=8.5,
            leading=12.2,
            textColor=TEXT,
            spaceAfter=2.8 * mm,
            allowWidows=0,
            allowOrphans=0,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold",
            parent=base["BodyText"],
            fontName=fonts["sans_bold"],
            fontSize=8.5,
            leading=12.2,
            textColor=FOREST,
            spaceAfter=1.2 * mm,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName=fonts["sans"],
            fontSize=8.7,
            leading=12.5,
            textColor=TEXT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName=fonts["sans_bold"],
            fontSize=7.2,
            leading=9.2,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=fonts["sans"],
            fontSize=7.1,
            leading=9.4,
            textColor=TEXT,
        ),
        "list": ParagraphStyle(
            "List",
            parent=base["BodyText"],
            fontName=fonts["sans"],
            fontSize=8.4,
            leading=11.7,
            textColor=TEXT,
            leftIndent=1 * mm,
            spaceAfter=0.8 * mm,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName=fonts["sans_bold"],
            fontSize=7.2,
            leading=9,
            textColor=GOLD,
            spaceAfter=2.5 * mm,
        ),
    }


def draw_cover_image(
    canvas, image_path: Path, x: float, y: float, width: float, height: float
) -> None:
    reader = ImageReader(str(image_path))
    image_width, image_height = reader.getSize()
    scale = max(width / image_width, height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    canvas.saveState()
    path = canvas.beginPath()
    path.rect(x, y, width, height)
    canvas.clipPath(path, stroke=0, fill=0)
    canvas.drawImage(reader, draw_x, draw_y, draw_width, draw_height, mask="auto")
    canvas.setFillColor(colors.Color(0.04, 0.15, 0.12, alpha=0.18))
    canvas.rect(x, y, width, height, stroke=0, fill=1)
    canvas.restoreState()


class CoverPage(Flowable):
    def __init__(
        self,
        client: str,
        title: str,
        subtitle: str,
        date: str,
        version: str,
        confidentiality: str,
        hero: Path | None,
        fonts: dict[str, str],
        styles: dict[str, ParagraphStyle],
        locale: str = "pl",
    ) -> None:
        super().__init__()
        self.width = PAGE_WIDTH
        self.height = PAGE_HEIGHT
        self.client = client
        self.title = title
        self.subtitle = subtitle
        self.date = date
        self.version = version
        self.confidentiality = confidentiality
        self.hero = hero
        self.fonts = fonts
        self.styles = styles
        self.labels = LABELS[locale]

    def wrap(self, avail_width, avail_height):
        return avail_width, avail_height

    def draw(self):
        canvas = self.canv
        origin_x = -self._frame._leftPadding
        origin_y = -self._frame._bottomPadding
        top_height = 515

        canvas.saveState()
        canvas.setFillColor(IVORY)
        canvas.rect(origin_x, origin_y, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        if self.hero and self.hero.is_file():
            draw_cover_image(
                canvas,
                self.hero,
                origin_x,
                PAGE_HEIGHT - top_height + origin_y,
                PAGE_WIDTH,
                top_height,
            )
        else:
            canvas.setFillColor(FOREST)
            canvas.rect(
                origin_x,
                PAGE_HEIGHT - top_height + origin_y,
                PAGE_WIDTH,
                top_height,
                stroke=0,
                fill=1,
            )
            canvas.setStrokeColor(colors.Color(1, 1, 1, alpha=0.08))
            canvas.setLineWidth(0.5)
            for offset in range(-200, 900, 70):
                canvas.line(
                    origin_x + offset,
                    PAGE_HEIGHT - top_height + origin_y,
                    origin_x + offset + 240,
                    PAGE_HEIGHT + origin_y,
                )

        canvas.setFont(self.fonts["sans_bold"], 10)
        canvas.setFillColor(colors.white)
        canvas.drawString(origin_x + 39, PAGE_HEIGHT - 38 + origin_y, self.client.upper())
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.2)
        canvas.line(
            origin_x + 39, PAGE_HEIGHT - 50 + origin_y, origin_x + 140, PAGE_HEIGHT - 50 + origin_y
        )

        label = Paragraph(self.labels["cover"], self.styles["cover_label"])
        label.wrapOn(canvas, 470, 30)
        label.drawOn(canvas, origin_x + 39, 251 + origin_y)

        title = Paragraph(inline_markup(self.title, self.fonts), self.styles["cover_title"])
        _, title_height = title.wrap(485, 100)
        title.drawOn(canvas, origin_x + 39, 220 - title_height + origin_y)
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.3)
        canvas.line(
            origin_x + 39,
            208 - title_height + origin_y,
            origin_x + 160,
            208 - title_height + origin_y,
        )

        subtitle = Paragraph(
            inline_markup(self.subtitle, self.fonts), self.styles["cover_subtitle"]
        )
        _, subtitle_height = subtitle.wrap(480, 60)
        subtitle.drawOn(canvas, origin_x + 39, 182 - title_height - subtitle_height + origin_y)

        canvas.setFont(self.fonts["sans"], 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            origin_x + 39,
            42 + origin_y,
            f"{self.labels['scope']} | {self.labels['version']} {self.version} | {self.date}",
        )
        canvas.drawRightString(origin_x + PAGE_WIDTH - 39, 42 + origin_y, self.confidentiality)
        canvas.restoreState()


class AuditDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        client: str,
        date: str,
        version: str,
        fonts: dict[str, str],
        locale: str = "pl",
        **kwargs,
    ):
        super().__init__(filename, pagesize=A4, **kwargs)
        self.client = client
        self.report_date = date
        self.version = version
        self.fonts = fonts
        self.labels = LABELS[locale]

        cover_frame = Frame(
            0,
            0,
            PAGE_WIDTH,
            PAGE_HEIGHT,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="cover",
        )
        content_frame = Frame(
            18 * mm, 20 * mm, PAGE_WIDTH - 36 * mm, PAGE_HEIGHT - 42 * mm, id="content"
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover_frame]),
                PageTemplate(id="content", frames=[content_frame], onPage=self.draw_content_chrome),
            ]
        )

    def draw_content_chrome(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(WARM_WHITE)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.setFont(self.fonts["sans_bold"], 6.5)
        canvas.setFillColor(FOREST)
        canvas.drawString(18 * mm, PAGE_HEIGHT - 12.5 * mm, self.client.upper())
        client_width = pdfmetrics.stringWidth(self.client.upper(), self.fonts["sans_bold"], 6.5)
        canvas.setFont(self.fonts["sans"], 6.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            18 * mm + client_width + 4 * mm, PAGE_HEIGHT - 12.5 * mm, "|  AI SEARCH & SEO AUDIT"
        )
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(0.6)
        canvas.line(18 * mm, PAGE_HEIGHT - 16 * mm, PAGE_WIDTH - 18 * mm, PAGE_HEIGHT - 16 * mm)

        canvas.setFont(self.fonts["sans"], 6.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            18 * mm, 10 * mm, f"{self.labels['version']} {self.version} - {self.report_date}"
        )
        canvas.drawCentredString(PAGE_WIDTH / 2, 10 * mm, f"{self.labels['page']} {doc.page - 1}")
        canvas.restoreState()


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_flowable(
    rows: list[list[str]], styles: dict[str, ParagraphStyle], fonts: dict[str, str], width: float
) -> LongTable:
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        data.append([Paragraph(inline_markup(cell, fonts), style) for cell in row])

    if column_count == 3:
        widths = [width * 0.30, width * 0.36, width * 0.34]
    elif column_count == 2:
        widths = [width * 0.35, width * 0.65]
    elif column_count == 4:
        widths = [width * 0.16, width * 0.30, width * 0.30, width * 0.24]
    else:
        widths = [width / column_count] * column_count

    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), FOREST),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(data)):
        commands.append(
            ("BACKGROUND", (0, row_index), (-1, row_index), WARM_WHITE if row_index % 2 else IVORY)
        )
    table.setStyle(TableStyle(commands))
    return table


def callout_flowable(
    text: str, styles: dict[str, ParagraphStyle], fonts: dict[str, str], width: float
) -> Table:
    paragraph = Paragraph(inline_markup(text, fonts), styles["callout"])
    background = PALE_GREEN
    line_color = FOREST_LIGHT
    if "ryzy" in text.lower() or text.lower().startswith("hipoteza"):
        background = PALE_RED
        line_color = colors.HexColor("#A75F52")
    box = Table([[paragraph]], colWidths=[width])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, line_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def markdown_to_story(
    markdown: str,
    styles: dict[str, ParagraphStyle],
    fonts: dict[str, str],
    content_width: float,
    locale: str = "pl",
    observation_projection_version: str = "1.0.0",
):
    raw_lines = markdown.splitlines()
    marker_lines = {
        index
        for index, line in enumerate(raw_lines)
        if re.fullmatch(r"<!-- audit-measurement:[0-9a-f]{64} -->", line)
    }
    lines = [sanitize_text(line) for line in raw_lines]
    story = []
    index = 0
    section_index = 0
    first_section = True

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped or index in marker_lines:
            index += 1
            continue
        if stripped.startswith("# "):
            index += 1
            continue
        if first_section and not stripped.startswith("## "):
            index += 1
            continue
        if observation_projection_version == "1.1.0" and stripped.startswith(
            "<!-- audit-literal-v1:"
        ):
            story.append(Paragraph(literal_paragraph_markup(stripped), styles["body"]))
            index += 1
            continue
        if stripped.startswith("## "):
            if not first_section:
                story.append(PageBreak())
            first_section = False
            section_index += 1
            title = strip_markdown(stripped[3:])
            story.append(
                Paragraph(f"{LABELS[locale]['section']} {section_index:02d}", styles["kicker"])
            )
            story.append(Paragraph(inline_markup(title, fonts), styles["section"]))
            index += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:], fonts), styles["subsection"]))
            index += 1
            continue
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and is_table_separator(lines[index + 1])
        ):
            rows = [split_table_row(stripped)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index]))
                index += 1
            story.extend([table_flowable(rows, styles, fonts, content_width), Spacer(1, 3.5 * mm)])
            continue
        if re.match(r"^[-*] ", stripped):
            items = []
            while index < len(lines) and re.match(r"^\s*[-*] ", lines[index]):
                item_text = re.sub(r"^\s*[-*] ", "", lines[index]).strip()
                items.append(
                    ListItem(
                        Paragraph(inline_markup(item_text, fonts), styles["list"]),
                        leftIndent=3 * mm,
                    )
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=5 * mm,
                    bulletFontName=fonts["sans"],
                    bulletFontSize=5.5,
                )
            )
            story.append(Spacer(1, 2.2 * mm))
            continue
        if re.match(r"^\d+\. ", stripped):
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\. ", lines[index]):
                item_text = re.sub(r"^\s*\d+\. ", "", lines[index]).strip()
                items.append(
                    ListItem(
                        Paragraph(inline_markup(item_text, fonts), styles["list"]),
                        leftIndent=3 * mm,
                    )
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    leftIndent=5 * mm,
                    bulletFontName=fonts["sans_bold"],
                    bulletFontSize=7.5,
                )
            )
            story.append(Spacer(1, 2.2 * mm))
            continue
        if stripped.startswith("> "):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            story.extend(
                [
                    callout_flowable(" ".join(quote_lines), styles, fonts, content_width),
                    Spacer(1, 3.5 * mm),
                ]
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                candidate.startswith(("#", ">", "|"))
                or (
                    observation_projection_version == "1.1.0"
                    and candidate.startswith("<!-- audit-literal-v1:")
                )
                or index in marker_lines
                or re.match(r"^[-*] ", candidate)
                or re.match(r"^\d+\. ", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph_text = " ".join(paragraph_lines)
        lower = strip_markdown(paragraph_text).lower()
        if lower.startswith(
            ("fakt zweryfikowany:", "obserwacja:", "hipoteza:", "nieznane:", "p0:", "p1:", "p2:")
        ):
            story.extend(
                [
                    callout_flowable(paragraph_text, styles, fonts, content_width),
                    Spacer(1, 3.5 * mm),
                ]
            )
        else:
            story.append(Paragraph(inline_markup(paragraph_text, fonts), styles["body"]))
    return story


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_markdown", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--client", required=True)
    parser.add_argument("--title", default="AI Search & SEO Audit")
    parser.add_argument("--subtitle")
    parser.add_argument("--date", required=True)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--audit-id", default="")
    parser.add_argument("--locale", choices=("pl", "en"), default="pl")
    cover = parser.add_mutually_exclusive_group()
    cover.add_argument("--hero", type=Path)
    cover.add_argument("--no-hero-reason")
    parser.add_argument("--author")
    parser.add_argument("--confidentiality")
    parser.add_argument(
        "--observation-projection-version", choices=("1.0.0", "1.1.0"), default="1.0.0"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hero is not None and not args.hero.is_file():
        raise ValueError("explicit hero image does not exist")
    if args.no_hero_reason is not None and not args.no_hero_reason.strip():
        raise ValueError("no-hero reason cannot be empty")
    if args.output_pdf.resolve() in {
        args.input_markdown.resolve(),
        args.hero.resolve() if args.hero else args.input_markdown.resolve(),
    }:
        raise ValueError("output PDF must not replace an input")
    markdown = args.input_markdown.read_text(encoding="utf-8")
    fonts = register_fonts()
    styles = make_styles(fonts)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".client-render-", dir=args.output_pdf.parent
    ) as temporary:
        raw_pdf = Path(temporary) / "render.pdf"
        _render(args, markdown, fonts, styles, raw_pdf)
        reader = PdfReader(raw_pdf)
        writer = PdfWriter(clone_from=reader)
        writer.add_metadata(
            {
                "/ClientEditionTemplate": "editorial-v1",
                "/ClientEditionSourceSHA256": hashlib.sha256(
                    args.input_markdown.read_bytes()
                ).hexdigest(),
                "/ClientEditionHeroSHA256": hashlib.sha256(args.hero.read_bytes()).hexdigest()
                if args.hero
                else "none",
                "/ClientEditionNoHeroReason": args.no_hero_reason or "",
                "/ClientEditionLocale": args.locale,
                "/ClientEditionClient": args.client,
                "/ClientEditionVersion": args.version,
                "/ClientEditionAuditID": args.audit_id,
                "/ClientEditionRenderOptions": json.dumps(
                    {
                        "client": args.client,
                        "title": args.title,
                        "subtitle": args.subtitle or LABELS[args.locale]["subject"],
                        "date": args.date,
                        "version": args.version,
                        "locale": args.locale,
                        "audit-id": args.audit_id,
                        "author": args.author or LABELS[args.locale]["author"],
                        "confidentiality": args.confidentiality
                        or LABELS[args.locale]["confidentiality"],
                        **(
                            {"observation-projection-version": "1.1.0"}
                            if args.observation_projection_version == "1.1.0"
                            else {}
                        ),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "/ClientEditionRendererSHA256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        writer.root_object[NameObject("/Lang")] = TextStringObject(args.locale)
        staged = Path(temporary) / "client.pdf"
        writer.write(staged)
        os.replace(staged, args.output_pdf)


def _render(args, markdown, fonts, styles, output):
    labels = LABELS[args.locale]

    document = AuditDocTemplate(
        str(output),
        client=args.client,
        date=args.date,
        version=args.version,
        fonts=fonts,
        title=args.title,
        author=args.author or labels["author"],
        subject=labels["subject"],
        creator="auditing-seo-geo-ai-search",
        locale=args.locale,
        invariant=1,
    )
    cover = CoverPage(
        client=args.client,
        title=args.title,
        subtitle=args.subtitle or labels["subject"],
        date=args.date,
        version=args.version,
        confidentiality=args.confidentiality or labels["confidentiality"],
        hero=args.hero,
        fonts=fonts,
        styles=styles,
        locale=args.locale,
    )
    story = [cover, NextPageTemplate("content"), PageBreak()]
    story.extend(
        markdown_to_story(
            markdown,
            styles,
            fonts,
            PAGE_WIDTH - 36 * mm,
            args.locale,
            args.observation_projection_version,
        )
    )
    document.build(story)


if __name__ == "__main__":
    main()
