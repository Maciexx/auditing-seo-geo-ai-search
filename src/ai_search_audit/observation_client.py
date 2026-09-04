"""Frozen factual client language; strategic interpretation remains outside this seam.

Any strategic claims still require the existing reports.py claim guards before
editorial review. This projection cannot infer strategy, causes or consumer reach.
"""

import base64
import json
from datetime import UTC
from typing import Literal
from urllib.parse import quote

from .benchmark import _citation_state, _mention_eligible
from .diagnostic_models import BenchmarkCitation, FrozenDiagnosticModel, _lossless_diagnostic_values
from .diagnostic_observations import ObservationCitationAnnotation
from .measurement_report import _text
from .observation_report import ObservationReport
from .observation_workflow import observation_scope


class ObservationGroup(FrozenDiagnosticModel):
    scope: Literal["branded", "discovery"]
    locale: str
    expected: int
    selected: int
    attempted: int
    eligible: int
    mentioned: int
    citation_eligible: int
    cited: int


class ObservationExcerpt(FrozenDiagnosticModel):
    label: str
    text: str
    sources: tuple[str, ...]


class ObservationClient(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0", "1.1.0"] = "1.0.0"
    report: ObservationReport
    introduction: tuple[str, ...]
    groups: tuple[ObservationGroup, ...]
    notes: tuple[str, ...]
    excerpts: tuple[ObservationExcerpt, ...]


def _bounded_excerpt(
    text: str, annotations: tuple[ObservationCitationAnnotation, ...]
) -> tuple[str, tuple[BenchmarkCitation, ...]]:
    """Shorten before cited spans, never drop a citation from a displayed span."""
    limit = min(600, len(text))
    ordered = sorted(annotations, key=lambda a: (a.start_index, a.end_index, str(a.url)))
    while limit:
        crossing = [a.start_index for a in ordered if a.start_index < limit < a.end_index]
        if crossing:
            limit = min(crossing)
            continue
        urls: list[BenchmarkCitation] = []
        for annotation in ordered:
            if annotation.end_index > limit or annotation.url in urls:
                continue
            if len(urls) == 5:
                limit = annotation.start_index
                break
            urls.append(annotation.url)
        else:
            return text[:limit], tuple(urls)
    return "", ()


def build_observation_client(
    report: ObservationReport, *, projection_version: str = "1.0.0"
) -> ObservationClient:
    if projection_version not in {"1.0.0", "1.1.0"}:
        raise ValueError("unsupported observation projection version")
    report = ObservationReport.model_validate(_lossless_diagnostic_values(report))
    run = report.run
    pl = run.binding.report_locale == "pl"
    selected = set(run.selected_prompt_ids)
    responses = {r.prompt_id: r for r in run.sample.responses}
    attempts = {a.prompt_id: a for a in run.attempts}
    groups = []
    keys = sorted({(observation_scope(p), p.locale) for p in run.worksheet.prompts})
    for scope, locale in keys:
        prompts = [
            p for p in run.worksheet.prompts if (observation_scope(p), p.locale) == (scope, locale)
        ]
        ids = {p.prompt_id for p in prompts}
        eligible = [r for pid, r in responses.items() if pid in ids and _mention_eligible(r)]
        cited = [_citation_state(r, run.sample.approved_domains) for r in eligible]
        groups.append(
            ObservationGroup(
                scope=scope,
                locale=_text(locale),
                expected=len(ids),
                selected=len(ids & selected),
                attempted=len(ids & attempts.keys()),
                eligible=len(eligible),
                mentioned=sum(r.brand_mentioned is True for r in eligible),
                citation_eligible=sum(c is not None for c in cited),
                cited=sum(c is True for c in cited),
            )
        )
    eligible_count = run.sample.metrics.measured
    introduction = [
        (
            f"Próbka. Liczba prób: {len(attempts)}. Wyniki możliwe do oceny: "
            f"{eligible_count}/{run.sample.metrics.expected} pytań z całego zestawu. "
            f"Liczba wybranych pytań: {len(selected)}."
            if pl
            else f"Sample: {len(attempts)} attempts; eligible results: "
            f"{eligible_count}/{run.sample.metrics.expected} whole-pack questions. "
            f"Selected questions: {len(selected)}."
        ),
        (
            "Produkt: OpenAI API web search. To nie jest pomiar interfejsu konsumenckiego ChatGPT "
            "ani Google AI Overviews. Wyniki dotyczą tylko tej próbki, nie całej widoczności marki."
            if pl
            else "Product: OpenAI API web search. This is not a measurement of the ChatGPT "
            "consumer interface or Google AI Overviews. Results describe this sample, not the "
            "brand's overall visibility."
        ),
    ]
    if run.collection_range.start and run.collection_range.end:
        start = run.collection_range.start.astimezone(UTC).date()
        end = run.collection_range.end.astimezone(UTC).date()
        introduction.append(
            ("Zebrano (UTC): " if pl else "Collected (UTC): ")
            + str(start)
            + (f" / {end}" if end != start else "")
            + "."
        )
    else:
        introduction.append("Nie wykonano prób API." if pl else "No API requests were attempted.")
    notes = [
        (
            "Brak danych nie jest wynikiem negatywnym. Wzmianki i cytowania liczymy osobno, "
            "tylko w odpowiedziach, które można ocenić. Niejasne wzmianki pozostają nieznane. "
            "Odwiedzone źródło nie musi być cytowane."
            if pl
            else "Missing data is not a negative result. Mentions and citations are counted "
            "separately, using only answers that can be assessed. Ambiguous mentions remain "
            "unknown. A consulted source is not necessarily cited."
        ),
        (
            "Pojedynczy pomiar nie dowodzi poprawy ani spadku i nie wystarcza do zmiany strategii. "
            "Nie ustalono porównania w takich samych warunkach."
            if pl
            else "One measurement does not prove improvement or decline and is not enough "
            "to change strategy. No comparison under the same conditions has been established."
        ),
        (
            "Następny krok: sprawdzić treść i źródła wskazanych odpowiedzi "
            "przed decyzją o zmianach."
            if pl
            else "Next check: review the indicated answers and sources before deciding on changes."
        ),
    ]
    if not any(g.scope == "discovery" and g.selected for g in groups):
        notes.append(
            "Nie badano odkrywania kategorii bez podania marki."
            if pl
            else "Brand-free category discovery was not tested."
        )
    excerpts: list[ObservationExcerpt] = []
    omitted_excerpt = False
    # First two retained answers in canonical prompt order, not the most favorable answers.
    candidates = [p for p in run.worksheet.prompts if p.prompt_id in responses][:2]
    for prompt in candidates:
        response = responses[prompt.prompt_id]
        annotations = attempts[prompt.prompt_id].citation_annotations or ()
        visible, urls = _bounded_excerpt(response.response_excerpt, annotations)
        if not visible.strip():
            omitted_excerpt = True
            continue
        sources = tuple(
            f"[{_text(url.host)}]({quote(str(url), safe=':/?&=+#%.,;~-')})" for url in urls
        )
        label = (
            ("Fragment odpowiedzi" if pl else "Answer excerpt")
            + f" ({_text(prompt.locale)}): "
            + _text(prompt.text)
        )
        # A prefixed quotation cannot become a Markdown heading or list item.
        text = '"' + _text(visible) + '"'
        if projection_version == "1.1.0":
            # Keep original code-point positions until the terminal literal renderer.
            # This encoding also distinguishes literal backslash escapes from punctuation.
            literal = {
                "text": visible,
                "urls": [str(url) for url in urls],
                "citations": sorted(
                    [
                        [a.start_index, a.end_index, urls.index(a.url)]
                        for a in annotations
                        if a.end_index <= len(visible) and a.url in urls
                    ],
                    key=lambda c: (c[1], c[0], c[2]),
                ),
            }
            encoded = base64.b64encode(
                json.dumps(literal, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).decode("ascii")
            text = "<!-- audit-literal-v1:" + encoded + " -->"
        excerpts.append(ObservationExcerpt(label=label, text=text, sources=sources))
    if excerpts:
        notes.append(
            "Poniżej pierwsze maksymalnie dwa fragmenty w kolejności pytań (do 600 znaków)."
            if pl
            else "Below are the first up to two excerpts in question order (up to 600 characters)."
        )
    if omitted_excerpt:
        notes.append(
            "Fragment pominięto: nie można go skrócić bez odłączenia cytowanych źródeł."
            if pl
            else "An excerpt was omitted because it could not be shortened "
            "without separating its cited sources."
        )
    return ObservationClient(
        schema_version="1.1.0" if projection_version == "1.1.0" else "1.0.0",
        report=report,
        introduction=tuple(introduction),
        groups=tuple(groups),
        notes=tuple(notes),
        excerpts=tuple(excerpts),
    )


def validate_observation_client(
    client: ObservationClient, *, report: ObservationReport
) -> ObservationClient:
    checked = ObservationClient.model_validate(_lossless_diagnostic_values(client))
    expected = build_observation_client(report, projection_version=checked.schema_version)
    if checked != expected:
        raise ValueError("client observation projection differs from trusted evidence")
    return expected
