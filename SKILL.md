---
name: auditing-seo-geo-ai-search
description: Use when auditing SEO, technical search visibility, GEO, LLMO, answer-engine visibility, AI Search discovery, entity consistency, or preparing a client-ready PDF about a brand's visibility across search and generative systems.
---

# Audit SEO, GEO, and AI Search Visibility

## Core principle

Produce an evidence-led audit of how an entity can be discovered, understood, verified, and selected across conventional search and AI-mediated discovery. Improve the conditions for visibility without promising rankings, citations, recommendations, traffic, or revenue.

Treat SEO, GEO, LLMO, answer-engine optimization, and AI Search as overlapping labels, not separate magic systems. Preserve the foundations: crawlable information, clear entities, useful native-language content, corroborating sources, technical quality, and measurement.

## Operating contract

1. Start with public evidence. Do not make CMS, analytics, Search Console, log, or profile access a prerequisite for useful work.
2. Record the audit date, entity, canonical domain, markets, languages, audience, positioning, conversion path, known competitors, seasonality, available access, exclusions, and limitations.
3. Research current public sources and current platform guidance when a fact, crawler policy, schema rule, product behavior, or AI feature may have changed.
4. Keep an evidence ledger throughout the audit. Never blur a verified fact, direct observation, hypothesis, or access-dependent unknown.
5. Distinguish sampled public inspection from a full crawl. Do not claim sitewide coverage without crawl evidence.
6. Exhaust relevant public research before asking owners for information. Ask only questions that resolve a material contradiction, dependency, or implementation decision.
7. Write the audit in the user's requested language; otherwise use the client's primary language. Evaluate every important site language independently.
8. Keep the client-facing audit concise. Show what exists, where the gap is, why it matters, and what to do.
9. Finish with a visually verified client PDF unless the user explicitly requests analysis only or a different final format. Do not stop at working notes, Markdown, or DOCX.

## Bundled audit engine

This repository includes a deterministic Python engine for bounded public collection, canonical
evidence, findings, readiness dimensions, entity consistency, and report artifacts. When the local
checkout is installed, run:

```bash
ai-search-audit audit https://example.com --output-dir audit-output/example --report-locale en
```

The engine writes `audit.json`, `evidence.jsonl`, `implementation-backlog.csv`, `ai-prompts.json`,
`report-draft.json`, `client-report-data.json`, and `client-report.pdf`. Use these artifacts as an
evidence and implementation layer. A bounded public crawl does not prove sitewide coverage, and
the engine report does not replace the broader client edition defined below.

## Local versioned project workflow

Use the project workflow when the audit should remain linked to later owner answers, first-party
visibility exports, or a validation run. The user-facing commands are:

```text
$auditing-seo-geo-ai-search https://example.com
$auditing-seo-geo-ai-search update project:example
$auditing-seo-geo-ai-search enrich project:example
$auditing-seo-geo-ai-search validate project:example
```

A URL alone starts a new project and creates its immutable `public-v1`. Derive a safe project ID,
confirm the client name and locale, then run `ai-search-audit project create`. All later operations
require the explicit `project:<id>` reference. Before any update, enrich, or validate operation,
load the manifest and perform an explicit domain and entity identity check. Stop on a mismatch.

Map the later commands as follows:

| Command | Project operation |
|---|---|
| `update project:<id>` | Normalize owner answers and create a context version without crawling again. |
| `enrich project:<id>` | Normalize approved first-party visibility exports and add visibility evidence. |
| `validate project:<id>` | Run a fresh like-for-like public audit and compare it with the selected baseline. |

Every successful version must include both `next-audit-data-request_<LOCALE>.md` and
`next-audit-data-request.json`. Tailor them to what is still missing or useful for that project.

The project operation is not complete client delivery when it has only the engine PDF. After the
working audit has been edited into the client edition, render it with an explicit locale, audit ID,
and either a client-controlled hero plus its source or an explicit no-hero reason. Run the complete
PDF QA from [references/client-pdf-spec.md](references/client-pdf-spec.md), calculate the SHA-256 of
that exact reviewed PDF, and run `ai-search-audit project finalize`. The command publishes a new
immutable `reports/<audit-version>/edition-N/` linked to the canonical audit. It rejects the
technical PDF, stale source, wrong project/version, mismatched locale, unreviewed bytes, or a cover
choice that does not match the reviewed PDF. Deliver only the PDF path printed by finalization.

Finalization is a projection and publication boundary. It does not create narrative, findings, or
strategic claims. All client-facing conclusions must already exist in the evidence-reviewed
Markdown. Never overwrite an engine bundle or describe bundle validation as editorial or visual
approval.

The initial deliverable is a free public audit. A client can optionally provide exports for an
optional deeper first-party visibility diagnostic. Keep pricing or sales copy outside the report.
This workflow measures visibility only. For GA4, accept aggregate GA4 Organic Search and GA4 AI
Assistant visits and sessions. Do not retain or analyze leads, revenue, CRM records, conversions,
customer IDs, or user-level events.

Process supplied files export-first. Start the deterministic project command with `--intake-root`
and keep the coordinating CLI process open. It creates an owned temporary intake directory and
prints `OWNED_INTAKE_DIR=<path>`. Use the available local document, spreadsheet, PDF, and image
tools to place the supplied exports there and construct `normalized-intake.json`. Send `READY` on
the command's standard input only after normalization is complete. The same process then validates
and consumes its creator-issued capability. Never copy raw attachments into the client project.
Delete the owned intake directory on success or failure. If a format cannot be normalized reliably,
fail clearly, delete the intake, and ask the user to reattach the source files and rerun with a fresh
export.

Default validation to 90 days after the implementation date. Compare equivalent windows and write
that a change was observed after the work; chronology does not establish causation. During local
development, the stable public checkout remains untouched. Store private client projects outside
all code checkouts.

## Evidence model

Use these labels in working notes and expose them wherever uncertainty affects a finding.

| Label | Meaning | Allowed phrasing |
|---|---|---|
| **Verified fact** | Directly supported by an identified source or authenticated dataset | “The official page states…” / “Search Console shows…” |
| **Observation** | Directly visible in the inspected sample | “On the pages inspected…” |
| **Hypothesis** | Reasoned explanation that requires validation | “A possible cause is…” |
| **Unknown / access-dependent** | Cannot be established with present evidence | “Confirm with a full crawl/logs/CMS access…” |

For each important claim, capture:

```text
claim | label | source or dataset | URL/location | retrieval date/date range |
confidence | conflicting evidence | validation needed
```

Do not silently resolve contradictions. Show the conflict, its operational impact, the provisional canonical value if justified, and who must confirm it.

## Audit workflow

Follow the sequence. Adapt depth to the business and evidence, but do not skip a stage without stating why it is out of scope.

### 1. Frame the entity and business context

- Identify the exact entity being audited: organization, brand, product, place, person, or service.
- Record canonical name, domain, locations, markets, languages, audiences, main offers, differentiators, credentials, conversion paths, and controlled profiles.
- Understand how discovery, validation, and conversion currently interact. Keep social discovery, third-party validation, official-site verification, direct contact, and AI discovery as distinct journey stages.
- Record whether the business is premium, regulated, local, international, seasonal, multi-location, marketplace-dependent, or otherwise constrained.
- Define the audit boundary. Separate public-data coverage from authenticated coverage.

### 2. Build the public source map

Inspect sources appropriate to the entity and market:

- first-party website, language versions, controlled profiles, feeds, policies, and public files;
- search results, map/local results, knowledge panels, cached snippets, and indexed documents;
- major directories, marketplaces, booking or commerce platforms, review platforms, associations, and partner pages;
- reputable press, specialist media, awards, databases, and other independent sources;
- relevant social profiles or creator coverage when they materially influence discovery or verification.

Prioritize primary and authoritative sources. Use first-party sources for canonical business claims and independent sources for corroboration, reputation, and category context. A third-party platform can be influential without being authoritative for every field.

Record the source, retrieval date, claim, evidence label, and contradiction status. Do not repeat volatile ratings, review counts, opening status, prices, policies, rankings, or platform behavior without a date.

### 3. Establish the entity source of truth

Create an entity consistency table. Select fields relevant to the business, including:

```text
canonical name and variants | entity type/category | concise description |
address and coordinates | contact details | locations/service area |
operating dates/hours | seasonal status and reopening date | languages |
offers/products/rooms/services | attributes and amenities | credentials/awards |
founders/owners where public and relevant | official profiles | stable identifiers
```

For each field record the official value, conflicting values, affected sources, proposed canonical wording, evidence, owner, and correction action. Treat material contradictions as visibility and trust defects, not cosmetic inconsistencies.

Define a controlled source-of-truth process: canonical record, accountable owner, update trigger, and downstream profiles that must be synchronized.

### 4. Audit technical discovery and bot access

Inspect what the available tools and access can prove. Cover, where applicable:

- response status, redirect behavior, HTTPS and host consistency;
- `robots.txt`, meta robots, `X-Robots-Tag`, authentication, paywalls, and other access controls;
- current policies for relevant search and AI crawlers, verified against official documentation;
- XML sitemaps, URL discovery, indexability, canonicals, pagination, duplicates, thin/soft-404 pages, and stale URLs;
- crawlable navigation, internal links, orphan risk, breadcrumbs, and information hierarchy;
- language/region URLs, hreflang reciprocity, canonical alignment, and default-language handling;
- server-rendered HTML versus JavaScript-dependent content and rendering failures;
- important facts present as accessible HTML rather than only images, sliders, video, canvas, or PDFs;
- mobile usability, performance risks, Core Web Vitals evidence, accessibility barriers, and intrusive interstitials when relevant.

State whether each conclusion comes from a sample, crawler, index report, Search Console, logs, CMS, or another dataset. Bot permission shows possible access, not indexing, retrieval, citation, or recommendation.

Treat `llms.txt` and similar emerging conventions as optional experiments unless current, authoritative support proves a specific use. Never present them as universal requirements or ranking factors.

### 5. Audit content, information architecture, and languages

Test whether the site answers, in crawlable text:

- What is this entity?
- Where does it operate?
- Who is it for?
- What does it offer?
- Why is it meaningfully different?
- What evidence supports those claims?
- How does a qualified visitor take the next step?

Map important facts and user needs to canonical pages. Look for ambiguous category language, missing decision information, unsupported superlatives, internal contradictions, weak titles/headings, duplication, stale claims, and facts trapped in visual media.

Evaluate each important language version as native content. Check factual parity, market framing, terminology, tone, grammar, cultural fit, conversion path, metadata, internal links, and hreflang. Do not treat literal translation or technical availability as language quality.

Preserve positioning. For premium, private, regulated, or specialist brands, prefer selective authority content and a precise knowledge layer over mass publishing. Recommend content only when it:

- reinforces the entity and its defensible differentiation;
- answers a real high-value question or objection;
- supports verification or the customer journey;
- demonstrates first-hand expertise, evidence, or information gain;
- has a credible owner and maintenance trigger.

Do not prescribe a publishing cadence for its own sake. Freshness means keeping material facts, offers, dates, people, policies, and availability accurate when they change.

### 6. Audit structured data and the knowledge layer

Inspect existing structured data before proposing new markup. Map the real entity and visible page content to the most specific valid schema types supported by current guidance.

Check:

- valid syntax and eligible properties;
- stable `@id` values and consistent identifiers;
- relationships among the organization, locations, people, products, offers, events, articles, reviews, and other relevant sub-entities;
- `sameAs` links to official or authoritative identity profiles;
- address, coordinates, breadcrumbs, images, authorship, dates, offers, and availability where applicable;
- agreement between markup, visible content, canonical source of truth, feeds, and external profiles.

Do not add unsupported ratings, misleading entity types, self-serving review markup, hidden content, invented properties, or schema that the page does not substantiate. Schema can reduce ambiguity; it does not force a ranking or AI citation.

Design a human-readable, machine-readable knowledge layer. Keep key facts concise and elegant, especially on visually minimal or premium sites.

### 7. Audit external authority and corroboration

Assess the entity beyond its own domain:

- completeness and consistency of controlled listings;
- review volume, recency, distribution, response practices, and recurring verified themes;
- reputable press, awards, associations, expert mentions, partner references, and niche directories;
- source quality, relevance, independence, discoverability, and factual accuracy;
- gaps between what customers or media repeatedly verify and what the official site communicates.

Separate authority-building from link accumulation. Do not recommend fake reviews, reputation manipulation, low-quality directories, paid mentions without disclosure, or indiscriminate link schemes.

### 8. Handle seasonal and time-sensitive businesses

Make seasonality explicit wherever it exists. Check official pages, structured data, local profiles, marketplaces, snippets, and external listings for consistent:

```text
season dates | temporary-closure status | reopening date | reservation window |
off-season contact route | year-round versus seasonal services | last-updated signals
```

Prevent “seasonally closed” from being interpreted as abandoned or permanently closed. Recommend event-driven updates before closing, before reopening, and whenever dates change. Compare performance using like-for-like seasonal windows.

### 9. Run a reproducible AI Search benchmark

Benchmark only systems, accounts, modes, locales, and web states that can actually be accessed. Never simulate an unavailable product or imply exhaustive coverage.

Build prompt families appropriate to the buying journey:

- branded fact verification;
- category and non-branded discovery;
- need/problem or use-case discovery;
- local, regional, and multilingual discovery;
- comparisons and alternatives;
- high-intent or constraint-led recommendations.

Use natural prompts a real prospect would ask. Include brand-free prompts to test discovery and branded prompts to test factual understanding. Avoid selecting only prompts designed to produce the client.

For every run record:

| Field | Required record |
|---|---|
| Test conditions | Product/model, mode, account state if relevant, web/browse state, device or interface when material |
| Query context | Exact prompt, language, locale, date, and session/reset method |
| Output | Mention, position only if meaningful, wording, factual accuracy, omissions, competitors |
| Evidence | Citations/links, source types, unsupported claims, screenshots or saved text when permitted |
| Interpretation | Observation, limitation, reproducibility note, and next validation step |

Treat results as dated samples affected by model changes, retrieval indexes, localization, personalization, randomness, and interface behavior. Repeat a stable prompt set over time; do not generalize from one run.

### 10. Synthesize findings and roadmap

Write each material finding with these fields:

| Finding field | Required content |
|---|---|
| Current state | Concise description of what was found |
| Evidence | Label, source/dataset, URL or location, and date |
| Implication | Why it matters to discovery, understanding, verification, or conversion |
| Recommendation | Specific corrective action and intended outcome |
| Delivery | Priority, dependency, effort, risk, owner, and success check |

Assign priorities consistently:

| Priority | Meaning | Typical use |
|---|---|---|
| **P0** | Blocking, misleading, risky, or prerequisite | Access/indexing blocks, wrong canonical facts, severe contradictions, broken localization |
| **P1** | High-value foundation | Entity layer, core page clarity, native-language rewrite, schema graph, major listing repair |
| **P2** | Compounding improvement | Expanded authority pages, selective digital PR, review operations, benchmark refinement |

Prioritize using impact, evidence strength, dependency, effort, and risk. Do not hide uncertainty inside a numeric score. If scores help scanning, publish the rubric and label untested areas rather than assigning false precision.

### 11. Create and verify the client PDF

**REQUIRED SUB-SKILL:** Use `pdf` for PDF creation, rendering, and visual verification.

Keep the working audit as the evidence layer. Create a separate client edition rather than exporting raw notes verbatim. Read [references/client-pdf-spec.md](references/client-pdf-spec.md) and follow its page budget, visual system, editorial rules, source profile, and QA procedure.

Use [scripts/render_client_pdf.py](scripts/render_client_pdf.py) to render the concise client Markdown into a consistent A4 document. Treat the renderer as a layout system, not as a substitute for editing. Rework the source when sections are too dense, sparse, repetitive, or split poorly.

Default delivery artifacts:

```text
Client_AI_Search_SEO_Audit_LANG.pdf
Client_AI_Search_SEO_Audit_LANG.md
Client_SEO_GEO_Evidence_Working_Audit.md   # when useful
```

Use a client-controlled hero image when available and record its source. Otherwise use the restrained typographic cover. Never use an unrelated image only for decoration.

Render every PDF page to images and inspect the complete document. Correct clipping, overflow, broken tables, missing glyphs, internal tokens, weak page rhythm, sparse pages, accidental blank pages, and unreadable sources before delivery.

## Measurement plan

Separate controllable leading indicators from outcome indicators. Choose only metrics supported by available data.

**Leading indicators** may include crawl/index health, canonical accuracy, entity consistency coverage, structured-data validity, important facts present in HTML, language parity/native-quality completion, listing correction rate, and source coverage.

**Outcome indicators** may include qualified organic impressions/clicks, non-branded discovery, assisted and direct conversions, local actions, relevant referring sources, review-theme consistency, AI mention rate across the fixed prompt set, factual accuracy, citation/source mix, and share of tested prompts.

Record baseline, data source, query or prompt set, date range, segment, season, target direction, review cadence, and owner. Use season-adjusted and market/language-specific comparisons. Treat AI benchmark metrics as directional samples, not guaranteed causal outcomes.

## Owner questions

Place this section near the end of the audit. Ask only questions that public research could not answer and that change scope, priority, or implementation. Tie each question to a finding or decision.

Cover only relevant gaps such as:

- canonical business facts or disputed values;
- seasonal dates, operational changes, and update ownership;
- priority markets, languages, audiences, and high-value conversions;
- controlled profiles, feeds, CMS, analytics, Search Console, logs, and governance;
- claims, credentials, press rights, sensitive information, or regulatory constraints;
- how customers currently discover, verify, and convert.

Do not turn the questionnaire into a substitute for public research.

## Client-ready output contract

Produce two layers:

1. **Working audit:** detailed evidence, limitations, diagnostics, and implementation notes.
2. **Client edition:** a concise decision document, normally 10-15 A4 pages including cover and sources, delivered as a polished PDF plus its editable Markdown source.

Default the client edition to this order and scale the length to evidence and business complexity:

1. Executive summary and audit scope
2. Method, evidence labels, and limitations
3. Current visibility and entity understanding
4. Technical SEO and bot access
5. Content, language, and information architecture
6. Structured data and knowledge layer
7. External authority, listings, reviews, and press
8. AI Search benchmark
9. Prioritized findings and P0/P1/P2 roadmap
10. Measurement plan
11. Owner questions
12. Sources and retrieval dates

Make the PDF client-readable while retaining an evidence trail. Separate confirmed public findings from items requiring a full crawl or authenticated access. Use tables only when they improve decisions. Move implementation detail to an appendix or the working audit instead of turning the main document into a long crawler report.

Do not include internal commentary such as a verdict about the skill, test output, tool names, or process notes in the client edition.

## Quality gate

Before delivery, verify all of the following:

- Every material claim has an evidence label and traceable source or dataset.
- Volatile public facts have retrieval dates.
- Sampled observations are not described as sitewide facts.
- Contradictions remain visible until resolved by evidence or an accountable owner.
- Technical, language, schema, external-authority, seasonal, and AI benchmark findings are included or explicitly out of scope.
- AI systems not directly tested are labeled untested.
- Recommendations preserve the entity's positioning and actual customer journey.
- P0/P1/P2 priorities include dependencies and success checks.
- Owner questions come after public research and map to material decisions.
- No sentence guarantees ranking, citation, recommendation, traffic, conversion, or revenue.
- No crawl, analytics result, bot-log finding, review, citation, or AI test was invented.
- No tactic is presented as a universal AI visibility switch.
- The client edition is within the agreed page range or has a justified appendix.
- The final PDF and its concise editable source both exist.
- PDF metadata, A4 size, page count, font embedding, and text extraction were checked.
- Every page was rendered and visually inspected after the latest edit.
- The cover, densest table, roadmap, owner questions, and sources were inspected at full resolution.
- No page contains clipping, overlap, missing glyphs, raw internal tokens, unreadable citations, accidental blank space, or an orphaned heading.
