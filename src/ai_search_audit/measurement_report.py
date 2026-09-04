"""Factual technical-section fragment; full provenance stays in the frozen DTO.

Insert before evidence/visual review. Finalization checks this exact projection,
never appends strategy, changes readiness scores or computes a numeric delta.
"""

import re
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from .diagnostic_models import BenchmarkHash, DiagnosticRunReference, FrozenDiagnosticModel
from .diagnostic_performance import DiagnosticRunV2
from .diagnostic_sources import load_diagnostic_source
from .diagnostic_store import DiagnosticStore
from .diagnostic_workflow import parse_diagnostic_run_reference
from .measurement_profile import _reject_unserialized_fields
from .performance_models import FieldMeasurement, LabMeasurement
from .performance_providers import PerformanceCollection

if TYPE_CHECKING:
    from .knowledge import KnowledgeRegistry

HEADINGS = {"pl": "### Pomiary wydajności (próbka)", "en": "### Performance measurements (sample)"}
_STATES = {
    "pl": {
        "AVAILABLE": "Dostępne",
        "PARTIAL": "Częściowe",
        "UNAVAILABLE": "Niedostępne",
        "FAILED": "Błąd modułu",
        "UNKNOWN": "Nieznane",
    },
    "en": {
        "AVAILABLE": "Available",
        "PARTIAL": "Partial",
        "UNAVAILABLE": "Unavailable",
        "FAILED": "Module failed",
        "UNKNOWN": "Unknown",
    },
}


class MeasurementReport(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    reference: DiagnosticRunReference
    manifest_sha256: BenchmarkHash
    audited_page_count: int = Field(ge=0)
    run: DiagnosticRunV2
    numeric_comparison: Literal["unavailable"] = "unavailable"


def load_measurement_report(
    project_ref: str, *, clients_root: Path, diagnostic_run_ref: str
) -> MeasurementReport:
    reference = parse_diagnostic_run_reference(diagnostic_run_ref)
    source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=reference.source_version
    )
    loaded = DiagnosticStore(clients_root / source.binding.project_id).load(
        reference.source_version, reference.run_id
    )
    if not isinstance(loaded.run, DiagnosticRunV2):
        raise ValueError("measurement report requires a performance diagnostic run")
    return MeasurementReport(
        reference=reference,
        manifest_sha256=loaded.manifest_sha256,
        audited_page_count=len(source.page_urls),
        run=loaded.run,
    )


def _text(value: object) -> str:
    """Data cannot inject Markdown links, columns, layout or renderer tokens."""
    text = " ".join(str(value).split())
    for char in "*`[]|<>@":
        text = text.replace(char, f"\\u{ord(char):04x}")
    return text


def _number(value: float | None, missing: str) -> str:
    return missing if value is None else format(value, ".6g")


def _metrics(
    measurement: LabMeasurement | FieldMeasurement | None, names: tuple[str, ...], missing: str
) -> str:
    values: dict[str, float | None] = (
        {m.name: m.value for m in measurement.metrics} if measurement is not None else {}
    )
    return " / ".join(_number(values.get(name), missing) for name in names)


def _status(collection: PerformanceCollection, locale: str) -> str:
    attempt = collection.attempts[-1]
    return _STATES[locale][attempt.state.value] + (f" ({attempt.reason})" if attempt.reason else "")


def _coverage(collections: list[PerformanceCollection], *, pl: bool) -> str:
    locale = "pl" if pl else "en"
    states = Counter(c.attempts[-1].state.value for c in collections)
    return "; ".join(
        f"{label}: {states[state]}" for state, label in _STATES[locale].items() if states[state]
    ) or ("Moduł wyłączony" if pl else "Module disabled")


def render_measurement_fragment(
    report: MeasurementReport,
    *,
    registry: "KnowledgeRegistry | None" = None,
    as_of: date | None = None,
) -> str:
    """Compile only already guarded strings; finalizers consume this client view."""
    from .benchmark import canonical_hash
    from .knowledge import default_registry_root, load_registry
    from .measurement_client import build_measurement_client, validate_measurement_client

    registry = registry if registry is not None else load_registry(default_registry_root())
    as_of = as_of if as_of is not None else datetime.now(UTC).date()
    client = validate_measurement_client(
        build_measurement_client(report, registry=registry, as_of=as_of),
        report=report,
        registry=registry,
        as_of=as_of,
    )
    locale = client.report.run.binding.report_locale
    pl = locale == "pl"
    # Inert renderer metadata binds equal-valued runs without exposing operator IDs.
    marker = f"<!-- audit-measurement:{canonical_hash(client.model_dump(mode='json'))} -->"
    lines = [HEADINGS[locale], "", marker, ""]
    for paragraph in client.introduction:
        lines.extend([paragraph, ""])
    for heading, headers, rows, notes in (
        (
            "Test laboratoryjny" if pl else "Lab test",
            ("Strona", "Urządzenie", "Źródło", "Wynik testu", "LCP")
            if pl
            else ("Page", "Device", "Source", "Test score", "LCP"),
            client.lab_rows,
            client.lab_notes,
        ),
        (
            "Dane rzeczywistych użytkowników (CrUX)" if pl else "Real-user data (CrUX)",
            ("Zakres i urządzenie", "Okres", "LCP p75", "INP p75", "CLS p75")
            if pl
            else ("Scope and device", "Period", "LCP p75", "INP p75", "CLS p75"),
            client.field_rows,
            client.field_notes,
        ),
    ):
        lines.extend([f"**{heading}**", ""])
        if rows:
            lines.extend(["| " + " | ".join(headers) + " |", "| --- | --- | --- | --- | --- |"])
            lines.extend("| " + " | ".join(row) + " |" for row in rows)
            lines.append("")
        for paragraph in notes:
            lines.extend([paragraph, ""])
    lines.append(
        ("Źródła progów: " if pl else "Threshold sources: ") + ", ".join(client.sources) + "."
    )
    return "\n".join(lines).rstrip() + "\n"


def render_measurement_technical_fragment(report: MeasurementReport) -> str:
    _reject_unserialized_fields(report)
    report = MeasurementReport.model_validate_json(
        report.model_dump_json(serialize_as_any=True, warnings=False)
    )
    run = report.run
    locale = run.binding.report_locale
    pl = locale == "pl"
    missing = "n/d" if pl else "n/a"
    pages = {page.url: f"U{i}" for i, page in enumerate(run.preflight.selected_pages, 1)}
    lab = [c for c in run.collections if c.attempts[0].provider != "crux"]
    field = [c for c in run.collections if c.attempts[0].provider == "crux"]
    lab_count = sum(c.lab is not None for c in lab)
    field_count = sum(c.field is not None for c in field)
    reference = f"{report.reference.source_version}/{report.reference.run_id}"
    slots = len(pages) * len(run.preflight.devices)
    lab_slots = slots if run.preflight.profile.pagespeed_insights else 0
    field_slots = slots if run.preflight.profile.crux else 0
    lines = [
        HEADINGS[locale],
        "",
        f"{'Seria' if pl else 'Run'}: {reference}. "
        f"{'Audyt źródłowy' if pl else 'Source audit'}: {run.binding.audit_id}.",
        "",
    ]
    lines += (
        [
            f"Próbka: {len(pages)}/{report.audited_page_count} audytowanych URL-i, "
            "mobile i desktop. "
            "Nie jest to pełny skan serwisu; języki stron niezweryfikowane.",
            "",
            f"Pary URL/urządzenie z metrykami: lab {lab_count}/{lab_slots}, "
            f"CrUX {field_count}/{field_slots}. To nie oznacza kompletu metryk. "
            f"Lab: {_coverage(lab, pl=True)}. CrUX: {_coverage(field, pl=True)}.",
            "",
            "Pojedynczy pomiar lab, bez wyboru najlepszego wyniku. Lighthouse lokalny zastępuje "
            "niedostępne PSI, nie jest drugim potwierdzeniem. "
            "CrUX to dane rzeczywistych użytkowników. Origin obejmuje cały origin, "
            "nie podstronę; powtórzone origin nie są niezależnymi próbkami.",
            "",
            "LCP, INP, FCP, TBT i SI: ms; CLS: bez jednostki. Brak danych: n/d, nie zero. "
            "Wyświetlane wartości są zaokrąglone. "
            "Lab TBT nie zastępuje terenowego INP. Wynik dostawcy /100 nie jest szansą cytowania. "
            "Niedostępność modułu nie obniża gotowości.",
            "",
            "Porównanie liczbowe niedostępne: brak kontrolowanego porównania w tej edycji. "
            "Zmiana źródła lub ustawień wyklucza kontrolowaną deltę.",
            "",
        ]
        if pl
        else [
            f"Sample: {len(pages)}/{report.audited_page_count} audited URLs, mobile and desktop. "
            "Not a full site scan; page languages are unverified.",
            "",
            f"URL/device pairs with metrics: lab {lab_count}/{lab_slots}, "
            f"CrUX {field_count}/{field_slots}. This does not mean complete metrics. "
            f"Lab: {_coverage(lab, pl=False)}. CrUX: {_coverage(field, pl=False)}.",
            "",
            "Single lab measurement, no best-result selection. "
            "Local Lighthouse replaces unavailable "
            "PSI, not a second confirmation. CrUX describes real users. Origin scope is not a page "
            "result; repeated origins are not independent samples.",
            "",
            "LCP, INP, FCP, TBT and SI: ms; CLS: unitless. Missing: n/a, not zero. "
            "Displayed values are rounded. "
            "Lab TBT does not replace field INP. Provider score /100 is not citation probability. "
            "Module unavailability does not reduce readiness.",
            "",
            "Numeric comparison unavailable: no controlled comparison in this edition. "
            "Changed source or settings excludes a controlled delta.",
            "",
        ]
    )
    if run.collection_range.start is not None and run.collection_range.end is not None:
        lines += [
            ("Czas zbierania danych" if pl else "Collection time")
            + " (UTC): "
            + run.collection_range.start.astimezone(UTC).isoformat(timespec="seconds")
            + " / "
            + run.collection_range.end.astimezone(UTC).isoformat(timespec="seconds")
            + ".",
            "",
        ]
    for url, alias in pages.items():
        lines += [f"{alias}: {_text(url)}", ""]
    if lab:
        lines += [
            "**" + ("Pomiary laboratoryjne" if pl else "Lab measurements") + "**",
            "",
            "| "
            + (
                "URL / profil | Źródło / status | LCP / CLS | FCP / TBT / SI | /100"
                if pl
                else "URL / device | Source / status | LCP / CLS | FCP / TBT / SI | /100"
            )
            + " |",
            "| --- | --- | --- | --- | --- |",
        ]
        stamps = []
        versions = set()
        for c in lab:
            a = c.attempts[0]
            name = (
                "PageSpeed Insights"
                if a.provider == "pagespeed_insights"
                else ("Lighthouse lokalny" if pl else "Local Lighthouse")
            )
            score = c.lab.performance_score if c.lab is not None else None
            lines.append(
                f"| {pages[a.requested_url]} {a.device} | {name}; {_status(c, locale)} | "
                f"{_metrics(c.lab, ('lcp', 'cls'), missing)} | "
                f"{_metrics(c.lab, ('fcp', 'tbt', 'speed_index'), missing)} | "
                f"{_number(score, missing)} |"
            )
            if c.lab is not None:
                stamps.append(c.lab.observed_at.astimezone(UTC).isoformat())
                versions.add(
                    f"{name}: Lighthouse {_text(c.lab.lighthouse_version)}, "
                    f"Chrome {_text(c.lab.chrome_version or missing)}"
                )
        if stamps:
            lines += [
                "",
                f"{'Czas pomiarów lab' if pl else 'Lab measurement times'} (UTC): "
                f"{min(stamps)} / {max(stamps)}.",
                "",
                "; ".join(sorted(versions)) + ".",
            ]
    if field:
        lines += [
            "",
            "**" + ("Dane terenowe CrUX" if pl else "CrUX field data") + "**",
            "",
            "| "
            + (
                "URL / profil | Zakres / status | Okres | LCP / INP / CLS p75 | Odczyt UTC"
                if pl
                else "URL / device | Scope / status | Period | LCP / INP / CLS p75 | Read UTC"
            )
            + " |",
            "| --- | --- | --- | --- | --- |",
        ]
        records = set()
        for c in field:
            a = c.attempts[0]
            f = c.field
            period = f"{f.period.first_date} / {f.period.last_date}" if f else missing
            observed = f.observed_at.astimezone(UTC).date().isoformat() if f else missing
            scope = f.scope if f else missing
            lines.append(
                f"| {pages[a.requested_url]} {a.device} | {scope}; {_status(c, locale)} | "
                f"{period} | {_metrics(f, ('lcp', 'inp', 'cls'), missing)} | {observed} |"
            )
            if f:
                records.add(f"{pages[a.requested_url]} {f.scope}: {_text(f.record_key)}")
        lines += ["", "; ".join(sorted(records))]
    warnings = sum(len(c.lab.warnings) for c in lab if c.lab is not None)
    changes = {
        f"{pages[c.lab.requested_url]}: {_text(c.lab.final_url)}"
        for c in lab
        if c.lab is not None and c.lab.final_url != c.lab.requested_url
    }
    if changes:
        lines += ["", ("Końcowe URL-e: " if pl else "Final URLs: ") + "; ".join(sorted(changes))]
    attempts = sum(len(c.attempts) for c in run.collections)
    lines += [
        "",
        (
            f"Próby dostawców: {attempts}; ostrzeżenia: {warnings}. "
            if pl
            else f"Provider attempts: {attempts}; warnings: {warnings}. "
        )
        + (
            "Pełne dowody, ustawienia, wersje i historia prób: "
            if pl
            else "Full evidence, settings, versions and attempt history: "
        )
        + f"diagnostics/{reference}/evidence.jsonl.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def require_measurement_fragment(markdown: str, fragment: str | None) -> None:
    """Verify consumed pre-reviewed source. Never append or rewrite narrative."""
    headings = sum(markdown.count(heading) for heading in HEADINGS.values())
    if fragment is None:
        if headings:
            raise ValueError("measurement fragment requires an explicit diagnostic run")
        return
    if headings != 1 or markdown.count(fragment) != 1:
        raise ValueError("exact measurement fragment must be inserted before render and review")
    chapter = re.search(r"^## .+", markdown, flags=re.MULTILINE)
    if chapter is None or chapter.start() > markdown.index(fragment):
        raise ValueError("measurement fragment must appear inside an editorial chapter")
