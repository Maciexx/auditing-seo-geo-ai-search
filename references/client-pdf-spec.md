# Client PDF specification

Read this file when the audit must be delivered to a client as a PDF.

## Editorial transformation

Do not convert working notes or a long evidence report verbatim. Create a separate client edition that preserves every material conclusion and source while removing crawl logs, raw diagnostics, repeated caveats, internal process commentary, and skill-testing notes.

The client edition must answer, in this order:

1. What is the decision-level conclusion?
2. What works today?
3. Where is the material gap?
4. What evidence supports it?
5. Why does it matter for discovery, understanding, trust, or conversion?
6. What should be implemented first?
7. How will success be checked?
8. What must the owner confirm?

Use plain language in the client's primary language. Preserve evidence labels where uncertainty affects a decision. Keep source numbers or readable links near material claims and provide a complete source page at the end.

Default to 10-15 A4 pages. A useful page budget is:

| Page | Purpose |
|---|---|
| Cover | Client, audit title, scope, date, version, confidentiality |
| 1 | Executive decision and highest-return actions |
| 2 | Business context, positioning, or customer journey |
| 3 | Current search and AI visibility snapshot |
| 4 | Highest-risk findings |
| 5 | Entity consistency and source of truth |
| 6 | Technical SEO, indexability, and bot access |
| 7 | Content, information architecture, and native-language quality |
| 8 | Structured data and the knowledge layer |
| 9 | External authority, listings, reviews, and press |
| 10 | Seasonality or another business-specific issue; omit if irrelevant |
| 11 | Reproducible AI Search benchmark and its limitations |
| 12 | P0/P1/P2 roadmap with dependencies and completion checks |
| 13 | Measurement plan and owner questions |
| 14 | Sources and retrieval dates |

Combine or omit pages when an area is out of scope. Do not add filler to reach a page count. If the evidence genuinely requires more than 15 pages, keep the main client narrative within 15 and move detailed tables to a clearly labeled appendix.

## Visual system

Use the visual system below as the quality reference, not as a content template.

| Element | Standard |
|---|---|
| Format | A4 portrait |
| Cover | Large client-controlled hero image above a warm ivory title panel; use a restrained typographic cover when no suitable image is available |
| Palette | Forest green `#173D34`, muted gold `#A78849`, ivory `#F4F0E7`, warm white `#FBFAF7`, charcoal `#2E3331` |
| Typography | Humanist sans-serif for body and tables; editorial serif for section titles; embed fonts with Polish and required language glyphs |
| Hierarchy | Small gold kicker, large serif page title, compact body, clear subsection headings |
| Tables | Dark green header, white labels, alternating warm rows, repeated header on page breaks |
| Callouts | Pale background with a narrow colored left rule; use risk color only for genuine warnings or unresolved hypotheses |
| Header | Client name plus `AI SEARCH & SEO AUDIT`, separated by a thin gold rule |
| Footer | Version, date, and page number |
| Density | One decision or tightly related finding family per page; retain breathing room |

Use only a photo supplied by the client or clearly controlled by the audited organization. Record its source. Do not use an unrelated stock image merely to decorate the cover.

## Source Markdown profile

Create a concise `client-report.md` before rendering:

```markdown
# Client name: audit title

## Executive decision

Short conclusion and a callout.

### Highest-return actions

1. Action one.
2. Action two.

| Priority | Action | Completion check |
|---|---|---|
| P0 | ... | ... |

## Entity consistency and source of truth

...

## P0/P1/P2 roadmap

...

## Owner questions and required access

...

## Sources

1. [Readable source title](https://example.com/), retrieved DATE.
```

Treat each `##` heading as a new client section and normally a new page. Keep a section within one page where practical. Merge sparse neighboring sections instead of creating nearly empty pages.

Do not include:

- a verdict about the skill or internal test;
- raw tool output, terminal commands, crawler dumps, or internal reference tokens;
- unsupported numeric scores;
- decorative charts without decision value;
- claims that schema, crawler access, or content guarantees rankings or AI citations;
- fabricated screenshots or simulated outputs from unavailable AI products.

## Rendering

Use the bundled renderer from the skill directory:

```bash
python3 scripts/render_client_pdf.py client-report.md client-audit.pdf \
  --client "Client name" \
  --title "AI Search & SEO Audit" \
  --subtitle "Google, ChatGPT, Gemini, Perplexity, Bing/Copilot and generative search" \
  --date "DATE" \
  --version "1.0" \
  --hero path/to/client-controlled-hero.jpg
```

If the workspace provides a bundled Python runtime, use it so `reportlab` and `pypdf` are available. The renderer accepts Markdown tables, headings, paragraphs, block quotes, numbered lists, bullets, links, bold text, and inline code. It replaces Unicode dash characters with ASCII hyphens for PDF reliability.

The renderer is a starting system, not permission to skip judgment. Edit the client Markdown when a table is too dense, a page becomes sparse, or a section spills awkwardly.

## PDF quality assurance

Do not deliver immediately after export.

1. Run `pdfinfo` and confirm A4, expected metadata, readable file size, and plausible page count.
2. Extract text with `pypdf` or `pdftotext`. Confirm the title, client name, owner questions, sources, and required language glyphs. Confirm there are no replacement characters or internal tool tokens.
3. Render every page to PNG with `pdftoppm`.
4. Build a contact sheet and inspect the whole document for rhythm, blank pages, dense pages, orphaned headings, clipped text, broken tables, and inconsistent section transitions.
5. Inspect the cover, densest table, roadmap, owner-questions page, and sources page at full resolution.
6. Correct every visual defect, regenerate the PDF, and repeat the checks.

The PDF is not complete until the latest render has no visible overlap, clipping, unreadable table text, missing glyphs, raw URLs overflowing cells, accidental blank pages, or unprofessional page breaks.

## Delivery files

Unless the user requests a different format, deliver:

- the final client PDF;
- the concise editable `client-report.md` used to generate it;
- the longer working audit only when useful for implementation or evidence review.

Use stable descriptive names such as:

```text
Client_AI_Search_SEO_Audit_PL.pdf
Client_AI_Search_SEO_Audit_PL.md
Client_SEO_GEO_Evidence_Working_Audit.md
```
