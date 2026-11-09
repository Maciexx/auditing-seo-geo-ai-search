# Google SEO and AI Search checks

Verified: 2026-11-09. Recheck the owning documentation when auditing changing
features. These rules apply to Google Search, not automatically to other AI systems.

## Eligibility and owner evidence

Distinguish four states: public technical access, index eligibility, confirmed indexing,
and observed appearance. HTTP 200 alone proves none of the latter three. Use an
owner-supplied URL Inspection result when confirmed indexing is needed; `site:`
results are incomplete and are not a ranking or index-coverage measurement.

For Search generative AI participation, request dated evidence from Settings →
Search generative AI in Search Console. Record the exact property, selected control
(include, exclude, inherit), effective setting and parent property where inherited.
Unknown parent settings leave the effective setting unknown. Do not infer it from
robots.txt or a missing AI mention. Do not change the setting as part of an audit.
Exclusion affects the documented AI features, not ordinary Search inclusion.

Sources: [technical requirements](https://developers.google.com/search/docs/essentials/technical),
[site operator](https://developers.google.com/search/docs/monitor-debug/search-operators/all-search-site),
[Search generative AI control](https://support.google.com/webmasters/answer/16908024).

## Crawling and preview controls

Check CDN/WAF access and robots separately from metadata. Inspect generic and
Googlebot-specific meta robots and X-Robots-Tag rules without attributing other
crawlers' restrictions to Google. Evaluate the most restrictive applicable rule.
`none` includes noindex; `max-snippet:0` is equivalent to nosnippet. A crawler must
be able to fetch a page to read its metadata; robots disallow does not guarantee
removal from the index.

Record nosnippet, valid max-snippet values, and data-nosnippet regions. Nosnippet
prevents direct content input into AIO/AI Mode; max-snippet limits it. A positive
limit is not a guaranteed exclusion. Keep exceptions for separately granted uses
or structured data visible where applicable. Data-nosnippet is a section control,
not a page-wide noindex. Inspect rendered content if JavaScript changes attributes.
Any removal recommendation depends on publishing intent, not an assumed duty to
maximize AI exposure.

Google-Extended is a product control token without a separate HTTP User-Agent;
it is not a Search inclusion or ranking switch. Keep training/grounding controls
separate from Googlebot and the Search Console participation setting.

Sources: [preview rules](https://developers.google.com/search/docs/crawling-indexing/robots-meta-tag),
[Google crawlers](https://developers.google.com/crawling/docs/crawlers-fetchers/google-common-crawlers#google-extended).

## Content and search presentation

Evaluate the page's purpose and whether readers can complete their task. Prefer
original examples, evidence, experience and accountable expertise over generic
summaries. Check real authorship, sources and review processes, particularly for
YMYL claims. Check generated metadata and image descriptions as carefully as prose.
AI assistance alone is not spam; scaled content produced to manipulate search
without user value is a different issue. Do not fabricate expert identities.

Titles, descriptive anchors, images with useful context and appropriate alt text,
and accessible video content support discovery and comprehension. These are
contextual checks, not a required heading count or word threshold. Do not require
H2, FAQ formatting, special Markdown copies, or small text chunks for Google AI.
Google Search ignores llms.txt; retain it only for separately documented uses.
Do not manufacture pages for imagined fan-out queries or inauthentic mentions.

Sources: [AI optimization guide](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide),
[SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide),
[helpful content](https://developers.google.com/search/docs/fundamentals/creating-helpful-content),
[AI-generated content](https://developers.google.com/search/docs/fundamentals/using-gen-ai-content),
[spam policies](https://developers.google.com/search/docs/essentials/spam-policies).

## Technical and feature-specific interpretation

- Google can render JavaScript. A raw/rendered difference is an observation, not
  proof of failure to index. Check resource access and the actual rendered result.
- A publisher's canonical hint is not proof of Google's selected canonical.
  Normal URL duplication is not automatically a penalty.
- Hreflang can be delivered through HTML, HTTP headers or sitemaps. Inspect the
  applicable methods before reporting absence, and verify reciprocal references.
- Rich-result eligibility is distinct from valid Schema.org. Check current feature
  support. Google retired FAQ rich results in May 2026; do not promise them.
- Core Web Vitals and overall usability matter; a perfect laboratory performance
  score is not a ranking guarantee or the default highest-priority task.
- For commerce/local entities, reconcile visible facts with Merchant Center and
  Business Profile evidence where available. Agentic interactions are a separate,
  optional review, not an SEO eligibility requirement.

Sources: [JavaScript](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics),
[canonicalization](https://developers.google.com/search/docs/crawling-indexing/canonicalization),
[localized versions](https://developers.google.com/search/docs/specialty/international/localized-versions),
[structured data](https://developers.google.com/search/docs/appearance/structured-data/sd-policies),
[feature updates](https://developers.google.com/search/updates),
[page experience](https://developers.google.com/search/docs/appearance/page-experience).

## Measurement and normalization

Use the optional Generative AI performance report (Search) for documented
impressions. Retain source report identity, property, filters, date range, PT time
zone, search type (text-based/multimodal) and aggregation basis. Do not label its
impressions as clicks, CTR, positions, queries or separate AIO/AI Mode measurements.
Discover has a separate report; do not mix it into this Search series.

The chart aggregates by property; page tables aggregate by page. Do not reconcile
the two by summing rows blindly. Generative impressions are included in ordinary
Web data, so preserve separate series and never add them into a combined total.
Mark partial coverage and preliminary periods explicitly. Missing report access
or insufficient impressions are unknown, not measured zero. Exported ~ or -
placeholders become zeros; accept a numerical zero only with source context that
establishes it was actually measured. Otherwise record a nonnumeric state or a
bounded partial series that excludes the ambiguous observations.

Keep report metadata in normalized aggregate evidence and delete raw intake using
the existing owned-intake workflow. Preserve comparison identity: differing report,
filters, search types, timezone or aggregation do not produce a comparable delta.
Use the engine's accepted normalized schema; reject unsupported fields instead
of coercing a report into ordinary Web data.

The current engine accepts monthly aggregate points through the existing
`normalized-intake.json` visibility workflow. Use metric
`gsc.generative_ai.impressions`, unit `impressions`, source report type
`generative-ai-performance`, and explicit source filters:

```text
gsc_property=sc-domain:example.com    # or exact URL-prefix property
search_type=web_text                  # or web_multimodal; never combine the series
timezone=America/Los_Angeles
report_scope=property                 # or page, matching the actual aggregation
zero_semantics=observed               # only after verifying any supplied numeric zeros
```

Do not import daily rows as monthly points. Preserve ordinary source provenance,
date range, coverage and confidence. Query dimensions are not supported by this
report. This is a manual normalized export workflow, not a live Search Console
connector.

For owner participation evidence, use the existing owner-fact workflow with
`search_ai_property`, `search_ai_selected`, `search_ai_effective`, and, only for
inheritance, `search_ai_parent`. Selected values are `include`, `exclude`, `inherit`;
effective values are `include` or `exclude`. Facts must be approved, conflict-free,
bound to the project's domain and have the same observation date. Inheritance
requires a valid ancestor property. The result remains owner-attested, not
API-verified. Incomplete evidence is reported as unknown.

Older AI-feature documentation only describes general Web reporting. Use the
dedicated current Search Console documentation for report/control semantics;
record conflicts and dates rather than silently relying on an older summary.

Sources: [Generative AI performance](https://support.google.com/webmasters/answer/16984139),
[older AI features overview](https://developers.google.com/search/docs/appearance/ai-features).

## Report and history boundaries

Automated structure observations do not predict citations or estimate Google's
internal quality signals. Report evidence, page purpose and uncertainty. Preserve
stored audit versions; new rule versions must not rewrite old scores or claims.
The workflow remains visibility-only: do not import conversions, revenue or
user-level records. A manual AI benchmark is a dated sample, not first-party
impression data.

Source: [third-party tools and advice](https://developers.google.com/search/docs/fundamentals/third-party-seo).
