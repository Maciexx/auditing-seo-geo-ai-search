# Auditing SEO, GEO & AI Search

[![CI](https://github.com/Maciexx/auditing-seo-geo-ai-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Maciexx/auditing-seo-geo-ai-search/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Maciexx/auditing-seo-geo-ai-search)](https://github.com/Maciexx/auditing-seo-geo-ai-search/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Turn a public website into an evidence-led SEO, GEO, AEO, and AI Search audit with a client-ready PDF. Findings separate facts, observations, hypotheses, and unknowns. Missing access stays unavailable instead of becoming a zero score.

## What it does

- Checks crawl and index signals, sitemaps, canonicals, redirects, and bot access.
- Reviews content, information architecture, languages, and citability.
- Compares structured data, entity signals, and the site's public truth.
- Records independent listings, reviews, press, and directory evidence.
- Builds AI prompt families and records observed visibility when observation is available.
- Produces a P0/P1/P2 roadmap, a measurement plan, and questions for owners.

## How it works

The audit starts with public evidence. First-party analytics, search console data, and authenticated platform access are optional. Rules use current official platform guidance and web standards while treating SEO, GEO, LLMO, AEO, and AI Search as overlapping practices.

It does not promise rankings, citations, traffic, or revenue. Every sample is labeled. If a source or check is unavailable, the report says so.

## Install

```bash
git clone https://github.com/Maciexx/auditing-seo-geo-ai-search.git ~/.codex/skills/auditing-seo-geo-ai-search
cd ~/.codex/skills/auditing-seo-geo-ai-search
python3 -m venv .venv
.venv/bin/pip install -e .
```

Other Agent Skills hosts can use the same repository from their configured skill directory. Read [SKILL.md](SKILL.md) before installation because the workflow can access the network, run a bounded crawler, and create local report files.

## Run an audit

Ask an agent:

```text
Use $auditing-seo-geo-ai-search to audit https://example.com and deliver a client-ready PDF.
```

Or run the engine directly:

```bash
.venv/bin/ai-search-audit audit https://example.com --output-dir audit-output/example --report-locale en
```

## Local project workflow

For ongoing work, the agent can keep each audit as a private versioned project. A URL creates the
initial immutable `public-v1`. Later commands use the explicit `project:<id>` reference:

```text
$auditing-seo-geo-ai-search https://example.com
$auditing-seo-geo-ai-search update project:example
$auditing-seo-geo-ai-search enrich project:example
$auditing-seo-geo-ai-search validate project:example
```

`update` adds normalized owner answers. `enrich` adds approved aggregate visibility exports.
`validate` creates a fresh like-for-like audit, normally after 90 days, without claiming that an
observed change was caused by the implementation. Every version includes a tailored
`next-audit-data-request` for the next useful step.

Raw attachments are processed while one coordinating command remains open, then removed. The
project keeps normalized evidence and final reports, not uploaded source files. If processing
fails, attach a fresh export and rerun the same project command.

The engine first saves the technical evidence bundle. The agent then writes the concise client
edition, renders it in the shared editorial design with a client-controlled cover photo (or an
explained typographic cover), and checks every page. `project finalize` saves the reviewed PDF in
`reports/<audit-version>/edition-N/` under that client's private project. Each edition stays linked
to its audit; earlier reports are not overwritten. See the [PDF workflow](references/client-pdf-spec.md).

## Output

The default final deliverable is a concise PDF plus editable Markdown. Detailed evidence and machine-readable audit artifacts remain available for implementation work. The full agent skill turns the engine output into a shorter client edition and visually verifies its pages. The technical bundle, separate from the finalized client edition, contains:

- `audit.json`
- `evidence.jsonl`
- `implementation-backlog.csv`
- `ai-prompts.json`
- `report-draft.json`
- `client-report-data.json`
- `client-report.pdf`

## Limits and privacy

Public inspection is not a replacement for a full authenticated crawl or first-party data. AI Search observations are dated and can vary by product, locale, account, and time.

Never commit live or client output. Repository fixtures must be fabricated and use reserved `.example` domains. See [SECURITY.md](SECURITY.md) for private reporting and [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change.

Before publishing, run `python scripts/validate_public_release.py .`. The optional repeated
`--private-term` arguments check locally supplied names without storing them in the repository;
`--history` checks the branch's committed history too. Keep synthetic tests, not real-client reports.

## License

The project code is available under the [MIT License](LICENSE). Bundled fonts and attributed materials keep their separate terms in [LICENSES](LICENSES/).
