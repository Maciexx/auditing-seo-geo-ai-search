from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from ai_search_audit.diagnostic_models import ExtractedContent, Section

_REMOVED_TAGS = ("script", "style", "nav", "header", "footer")
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_SEMANTIC_BLOCKS = _HEADINGS | {"p", "li", "tr"}
_BLOCK_CONTAINERS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "form",
        "hgroup",
        "hr",
        "menu",
        "ol",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "ul",
    }
)
_STYLE_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STYLE_DECLARATION = re.compile(r"^\s*(display|visibility)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_IMPORTANT_SUFFIX = re.compile(r"\s*!\s*important\s*$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def extract_sections(
    html: str,
    *,
    capture_id: str,
    complete: bool = True,
) -> ExtractedContent:
    """Extract structural content from inert HTML without resolving external resources."""
    soup = BeautifulSoup(html, "html.parser")
    _remove_excluded_content(soup)
    roots, fallback_used, limitations = _select_roots(soup)
    if not complete:
        limitations.append("Input capture is explicitly partial.")

    sections: list[Section] = []
    document_blocks: list[str] = []
    for root_index, root in enumerate(roots):
        root_sections, root_blocks = _extract_root(root, capture_id, root_index)
        sections.extend(root_sections)
        document_blocks.extend(root_blocks)

    return ExtractedContent(
        capture_id=capture_id,
        text="\n".join(document_blocks),
        sections=tuple(sections),
        fallback_used=fallback_used,
        limitations=tuple(limitations),
    )


def _remove_excluded_content(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_REMOVED_TAGS):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.parent is not None and _is_explicitly_hidden(tag):
            tag.decompose()


def _is_explicitly_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    if tag.name == "input" and str(tag.get("type", "")).lower() == "hidden":
        return True
    return _style_hides(str(tag.get("style", "")))


def _style_hides(style: str) -> bool:
    effective: dict[str, tuple[str, bool]] = {}
    uncommented = _STYLE_COMMENT.sub(" ", style)
    # This extractor recognizes only simple inline declarations. Any remaining complex
    # syntax can hide semicolons or join tokens, so it cannot prove content is hidden.
    if any(
        character in uncommented
        for character in ("'", '"', "\\", "(", ")", "{", "}", "[", "]", "/")
    ):
        return False
    for declaration in uncommented.split(";"):
        match = _STYLE_DECLARATION.match(declaration)
        if match is None:
            continue
        property_name = match.group(1).lower()
        raw_value = match.group(2)
        important_match = _IMPORTANT_SUFFIX.search(raw_value)
        important = important_match is not None
        value = _IMPORTANT_SUFFIX.sub("", raw_value).strip().lower()
        previous = effective.get(property_name)
        if previous is None or important or not previous[1]:
            effective[property_name] = (value, important)
    return effective.get("display", ("", False))[0] == "none" or (
        effective.get("visibility", ("", False))[0] == "hidden"
    )


def _select_roots(soup: BeautifulSoup) -> tuple[list[Tag], bool, list[str]]:
    mains = _top_level(soup.find_all("main"), "main")
    if mains:
        return mains, False, []

    articles = _top_level(soup.find_all("article"), "article")
    if articles:
        return articles, False, []

    if soup.body is not None:
        return [soup.body], True, ["No main or article container found; extracted body fallback."]
    return [soup], True, ["No main, article, or body container found; extracted document fallback."]


def _top_level(candidates: list[Tag], tag_name: str) -> list[Tag]:
    return [tag for tag in candidates if tag.find_parent(tag_name) is None]


def _extract_root(root: Tag, capture_id: str, root_index: int) -> tuple[list[Section], list[str]]:
    sections: list[Section] = []
    document_blocks: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    current: _OpenSection | None = None
    intro: _OpenSection | None = None
    heading_ordinal = 0

    for kind, value, level in _iter_blocks(root):
        document_blocks.append(value)
        if kind == "heading":
            if current is not None:
                sections.append(current.to_section(capture_id))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, value))
            current = _OpenSection(
                locator=f"root-{root_index}/heading-{heading_ordinal}",
                level=level,
                heading=value,
                heading_path=tuple(item[1] for item in heading_stack),
            )
            heading_ordinal += 1
            continue

        target = current
        if target is None:
            if intro is None:
                intro = _OpenSection(
                    locator=f"root-{root_index}/intro",
                    level=0,
                    heading="",
                    heading_path=(),
                )
            target = intro
        target.add(kind, value)

    if intro is not None:
        sections.insert(0, intro.to_section(capture_id))
    if current is not None:
        sections.append(current.to_section(capture_id))
    return sections, document_blocks


class _OpenSection:
    def __init__(
        self,
        *,
        locator: str,
        level: int,
        heading: str,
        heading_path: tuple[str, ...],
    ) -> None:
        self.locator = locator
        self.level = level
        self.heading = heading
        self.heading_path = heading_path
        self.blocks: list[str] = []
        self.block_kinds: list[str] = []

    def add(self, kind: str, value: str) -> None:
        self.block_kinds.append(kind)
        self.blocks.append(value)

    def to_section(self, capture_id: str) -> Section:
        text = "\n".join(self.blocks)
        identity = "\0".join((capture_id, self.locator, self.heading, text))
        return Section(
            section_id=f"section-{hashlib.sha256(identity.encode()).hexdigest()}",
            capture_id=capture_id,
            locator=self.locator,
            level=self.level,
            heading=self.heading,
            heading_path=self.heading_path,
            text=text,
            block_kinds=tuple(self.block_kinds),
        )


def _iter_blocks(
    root: Tag,
    *,
    include_loose_text: bool = True,
    loose_kind: str = "paragraph",
) -> Iterator[tuple[str, str, int]]:
    inline_parts: list[str] = []
    for child in root.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            inline_parts.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "br":
            inline_parts.append(" ")
            continue
        if child.name in _HEADINGS:
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            value = normalize_text(_visible_text(child))
            if value:
                yield "heading", value, int(child.name[1])
        elif child.name == "p":
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            value = normalize_text(_visible_text(child))
            if value:
                yield "paragraph", value, 0
        elif child.name == "li":
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            yield from _iter_blocks(child, loose_kind="list_item")
        elif child.name == "tr":
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            value = _table_row_text(child)
            if value:
                yield "table_row", value, 0
        elif _contains_block_boundary(child):
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            yield from _iter_blocks(
                child,
                include_loose_text=include_loose_text,
                loose_kind=loose_kind,
            )
        elif child.name in _BLOCK_CONTAINERS:
            yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)
            if _contains_block_boundary(child):
                yield from _iter_blocks(
                    child,
                    include_loose_text=include_loose_text,
                    loose_kind=loose_kind,
                )
            else:
                value = normalize_text(_visible_text(child))
                if include_loose_text and value:
                    yield loose_kind, value, 0
        else:
            inline_parts.append(_visible_text(child))
    yield from _loose_text_block(inline_parts, include_loose_text, loose_kind)


def _loose_text_block(
    parts: list[str],
    include_loose_text: bool,
    kind: str,
) -> Iterator[tuple[str, str, int]]:
    value = normalize_text("".join(parts))
    parts.clear()
    if include_loose_text and value:
        yield kind, value, 0


def _contains_block_boundary(tag: Tag) -> bool:
    return tag.find(_SEMANTIC_BLOCKS | _BLOCK_CONTAINERS) is not None


def _visible_text(tag: Tag) -> str:
    parts: list[str] = []
    for descendant in tag.descendants:
        if isinstance(descendant, Comment):
            continue
        if isinstance(descendant, NavigableString):
            parts.append(str(descendant))
        elif isinstance(descendant, Tag) and descendant.name == "br":
            parts.append(" ")
    return "".join(parts)


def _table_row_text(row: Tag) -> str:
    cells = row.find_all(("th", "td"), recursive=False)
    return " | ".join(normalize_text(_table_cell_text(cell)) for cell in cells)


def _table_cell_text(cell: Tag) -> str:
    return " / ".join(value for _, value, _ in _iter_blocks(cell))
