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

## Output

The default final deliverable is a concise PDF plus editable Markdown. Detailed evidence and machine-readable audit artifacts remain available for implementation work. The full agent skill turns the engine output into a shorter client edition and visually verifies its pages. The output directory contains:

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

## License

The project code is available under the [MIT License](LICENSE). Bundled fonts and attributed materials keep their separate terms in [LICENSES](LICENSES/).
