"""Frozen client language derived only from trusted observations and research.

The report snapshot retains exact values, states, evidence IDs and source binding.
No provider prose, mutable registry objects or presentation-time claims cross the guard.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal
from urllib.parse import unquote, urlsplit

from .diagnostic_models import DiagnosticRule, FrozenDiagnosticModel
from .knowledge import KnowledgeRegistry, default_registry_root, load_registry
from .measurement_profile import _reject_unserialized_fields
from .measurement_report import MeasurementReport, _text
from .models import RuleState
from .performance_models import FieldMeasurement, _origin
from .performance_providers import PerformanceCollection

RULE_IDS = ("performance-psi-score-001", "performance-lcp-thresholds-001")
ClientRow = tuple[str, str, str, str, str]


class MeasurementClient(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    report: MeasurementReport
    registry_version: str
    as_of: date
    rules: tuple[DiagnosticRule, DiagnosticRule]
    introduction: tuple[str, ...]
    lab_rows: tuple[ClientRow, ...]
    lab_notes: tuple[str, ...]
    field_rows: tuple[ClientRow, ...]
    field_notes: tuple[str, ...]
    sources: tuple[str, ...]


def _page(url: str, *, pl: bool) -> str:
    parts = urlsplit(url)
    label = (
        unquote(parts.path)
        if parts.path not in {"", "/"}
        else ("Strona główna" if pl else "Homepage")
    )
    if parts.query:
        label += "?" + unquote(parts.query)
    return _text(label)


def _device(device: str, *, pl: bool) -> str:
    return {"mobile": "telefon", "desktop": "komputer"}[device] if pl else device


def _value(value: float | None, *, seconds: bool = False, pl: bool) -> str:
    if value is None:
        return "brak danych" if pl else "unavailable"
    # Decimal keeps positive subnormal milliseconds positive after conversion.
    number = Decimal(str(value)) / (1000 if seconds else 1)
    formatted = (
        format(number, ".3g")
        if 0 < number < Decimal("0.005") or number >= 10000
        else f"{number:.2f}".rstrip("0").rstrip(".")
    )
    if Decimal(formatted) != number:
        formatted = "≈" + formatted
    if pl:
        formatted = formatted.replace(".", ",")
    return formatted + (" s" if seconds else "")


def _band(value: float, *, score: bool, pl: bool) -> str:
    index = (
        (0 if value >= 90 else 1 if value >= 50 else 2)
        if score
        else (0 if value <= 2500 else 1 if value <= 4000 else 2)
    )
    return (("dobry", "wymaga poprawy", "słaby") if pl else ("good", "needs improvement", "poor"))[
        index
    ]


def _metric(value: float | None, rule: DiagnosticRule, *, score: bool, pl: bool) -> str:
    text = _value(value, seconds=not score, pl=pl)
    if value is not None:
        if score:
            text += "/100"
        if rule.state is RuleState.CURRENT:
            text += "; " + _band(value, score=score, pl=pl)
    return text


def _unavailable(collection: PerformanceCollection, *, pl: bool) -> str:
    reason = collection.attempts[-1].reason
    if reason == "missing_key":
        return "brak skonfigurowanego dostępu" if pl else "access not configured"
    if reason in {"invalid_key", "http_401", "http_403"}:
        return "odmowa dostępu" if pl else "access denied"
    return "pomiar niedostępny" if pl else "measurement unavailable"


def _field_key(field: FieldMeasurement) -> tuple[object, ...]:
    # Retrieval time and page attribution do not make a shared origin record independent.
    return (
        field.scope,
        _origin(field.record_key) if field.scope == "origin" else field.record_key,
        field.device,
        field.period.first_date,
        field.period.last_date,
        tuple(sorted((m.name, m.value, m.unit) for m in field.metrics)),
    )


def build_measurement_client(
    report: MeasurementReport,
    *,
    registry: KnowledgeRegistry | None = None,
    as_of: date | None = None,
) -> MeasurementClient:
    """Resolve knowledge at a trusted date, never at a provider-supplied timestamp."""
    registry = registry if registry is not None else load_registry(default_registry_root())
    as_of = as_of if as_of is not None else datetime.now(UTC).date()
    _reject_unserialized_fields(report)
    # Validate lossless values: JSON would silently convert NaN/Inf to null.
    report = MeasurementReport.model_validate(
        report.model_dump(mode="python", serialize_as_any=True, warnings=False)
    )
    if report.reference.source_version != report.run.binding.source_version:
        raise ValueError("measurement report source binding mismatch")
    if report.audited_page_count < len(report.run.preflight.selected_pages):
        raise ValueError("measurement report sample exceeds source coverage")
    try:
        rules = tuple(
            DiagnosticRule.model_validate(registry.resolve(key, as_of=as_of).model_dump())
            for key in RULE_IDS
        )
    except KeyError as exc:
        raise ValueError("measurement interpretation requires a registered rule") from exc
    score_rule, lcp_rule = rules
    run = report.run
    pl = run.binding.report_locale == "pl"
    selected = len(run.preflight.selected_pages)
    devices = ", ".join(_device(device, pl=pl) for device in run.preflight.devices)
    intro = [
        (
            f"Próbka: {selected}/{report.audited_page_count} audytowanych stron; {devices}. "
            "Nie jest to pełny skan serwisu."
            if pl
            else f"Sample: {selected}/{report.audited_page_count} audited pages; {devices}. "
            "Not a full site scan."
        ),
        (
            "Test laboratoryjny pokazuje wydajność w warunkach testu, nie widoczność w AI "
            "ani ocenę Core Web Vitals rzeczywistych użytkowników. Brak danych nie oznacza "
            "zera i nie obniża gotowości. Symbol ≈ oznacza zaokrąglenie."
            if pl
            else "A lab test describes performance under test conditions, not AI visibility "
            "or real-user Core Web Vitals. Missing data is not zero and does not reduce readiness. "
            "The ≈ symbol marks rounding."
        ),
    ]
    if run.collection_range.start is not None and run.collection_range.end is not None:
        start = run.collection_range.start.astimezone(UTC).date()
        end = run.collection_range.end.astimezone(UTC).date()
        intro.append(
            ("Zebrano (UTC): " if pl else "Collected (UTC): ")
            + f"{start}"
            + (f" – {end}" if start != end else "")
            + "."
        )
    lab = [c for c in run.collections if c.attempts[0].provider != "crux"]
    field = [c for c in run.collections if c.attempts[0].provider == "crux"]
    lab_rows: list[ClientRow] = []
    delayed = False
    partial = False
    warnings = False
    replacement_slots = {
        (c.lab.requested_url, c.lab.device)
        for c in lab
        if c.lab is not None and c.lab.provider == "lighthouse_local"
    }
    for collection in lab:
        first = collection.attempts[0]
        if (
            first.provider == "pagespeed_insights"
            and (first.requested_url, first.device) in replacement_slots
        ):
            # The frozen policy allows local results only after PSI has no result.
            # This is one displayed result, not best-score selection or evidence deletion.
            continue
        measurement = collection.lab
        provider = (
            "PageSpeed Insights"
            if first.provider == "pagespeed_insights"
            else ("Lighthouse lokalny" if pl else "Local Lighthouse")
        )
        values = {m.name: m.value for m in measurement.metrics} if measurement else {}
        score = measurement.performance_score if measurement else None
        lcp = values.get("lcp")
        delayed |= lcp is not None and lcp > 2500
        if measurement:
            partial |= collection.attempts[-1].state.value == "PARTIAL"
            warnings |= bool(measurement.warnings)
        else:
            provider += "; " + _unavailable(collection, pl=pl)
        page = _page(first.requested_url, pl=pl)
        if measurement and measurement.final_url != measurement.requested_url:
            page += " → " + _page(measurement.final_url, pl=pl)
        lab_rows.append(
            (
                page,
                _device(first.device, pl=pl),
                provider,
                _metric(score, score_rule, score=True, pl=pl),
                _metric(lcp, lcp_rule, score=False, pl=pl),
            )
        )
    lab_notes = []
    if any(c.lab is not None for c in lab):
        lab_notes.append(
            "Wynik /100 podsumowuje test. LCP to czas pojawienia się największego widocznego "
            "elementu. Ocena LCP odnosi pojedynczy test do progów referencyjnych, "
            "nie rozstrzyga wyniku całego serwisu."
            if pl
            else "The /100 score summarizes the test. LCP is the time until the largest visible "
            "element appears. Its band compares this single test with reference thresholds, "
            "not a sitewide verdict."
        )
        if delayed and lcp_rule.state is RuleState.CURRENT:
            lab_notes.append(
                "Następny krok: sprawdzić w diagnostyce Lighthouse, co opóźnia pojawienie się "
                "największego elementu na wskazanej stronie i urządzeniu. "
                "Sam LCP nie wskazuje przyczyny."
                if pl
                else "Next step: investigate what delays the largest element on the indicated "
                "page and device using Lighthouse diagnostics. "
                "LCP alone does not identify the cause."
            )
        elif score_rule.state is RuleState.CURRENT and any(
            c.lab and c.lab.performance_score is not None and c.lab.performance_score < 90
            for c in lab
        ):
            lab_notes.append(
                "Następny krok: przejrzeć diagnostykę Lighthouse dla wyników wymagających "
                "poprawy, aby ustalić przyczynę przed zmianami."
                if pl
                else "Next step: review Lighthouse diagnostics for scores needing improvement "
                "to identify the cause before making changes."
            )
    elif not lab:
        lab_notes.append(
            "Pomiar laboratoryjny był wyłączony." if pl else "Lab measurement was disabled."
        )
    if any(c.lab and c.lab.provider == "lighthouse_local" for c in lab):
        lab_notes.append(
            "Lighthouse lokalny zastępuje niedostępne PageSpeed Insights; "
            "nie jest niezależnym potwierdzeniem."
            if pl
            else "Local Lighthouse replaces unavailable PageSpeed Insights; "
            "it is not independent confirmation."
        )
    elif any(c.attempts[0].provider == "lighthouse_local" for c in lab):
        lab_notes.append(
            "Próba pomiaru lokalnym Lighthouse nie dostarczyła wyniku zastępczego."
            if pl
            else "The local Lighthouse attempt produced no replacement result."
        )
    if partial or warnings:
        lab_notes.append(
            "Część pomiarów jest niepełna lub zawiera ostrzeżenia dostawcy; "
            "pokazano dostępne wyniki. Wymagają ostrożnej interpretacji."
            if pl
            else "Some measurements are partial or carry provider warnings; "
            "available results are shown and need cautious interpretation."
        )
    if any(rule.state is not RuleState.CURRENT for rule in rules):
        lab_notes.append(
            "Część reguł wymaga ponownej weryfikacji. Zależne oceny i zalecenia wstrzymano; "
            "surowe wyniki pozostają dostępne."
            if pl
            else "Some reference rules need verification. Dependent bands and recommendations "
            "are withheld; raw results remain available."
        )
    field_rows: list[ClientRow] = []
    seen = set()
    for collection in field:
        record = collection.field
        if record is None or _field_key(record) in seen:
            continue
        seen.add(_field_key(record))
        scope = (
            "Origin: " + _text(record.record_key.rstrip("/"))
            if record.scope == "origin"
            else ("Strona: " if pl else "Page: ") + _page(record.record_key, pl=pl)
        )
        metrics = {m.name: m.value for m in record.metrics}
        field_rows.append(
            (
                scope + "; " + _device(record.device, pl=pl),
                f"{record.period.first_date} – {record.period.last_date}",
                _metric(metrics.get("lcp"), lcp_rule, score=False, pl=pl),
                _value(metrics.get("inp"), seconds=True, pl=pl),
                _value(metrics.get("cls"), pl=pl),
            )
        )
    field_notes = []
    no_records = [
        c for c in field if c.field is None and c.attempts[-1].reason in {"no_record", "no_data"}
    ]
    failed = [c for c in field if c.field is None and c not in no_records]
    if field_rows:
        field_notes.append(
            "CrUX opisuje rzeczywistych użytkowników. p75 oznacza, że 75% zarejestrowanych "
            "doświadczeń ma wynik nie większy od podanego. Origin obejmuje strony pod "
            "tym samym protokołem, hostem i portem; identyczne rekordy pokazano raz. "
            "INP opisuje reakcję na "
            "interakcję, CLS stabilność układu. Laboratoryjny TBT nie zastępuje INP."
            if pl
            else "CrUX describes real users. p75 means 75% of recorded experiences have a value "
            "at or below the result. An origin covers pages sharing protocol, host and port; "
            "identical records appear once. INP describes interaction responsiveness; "
            "CLS describes layout stability. Lab TBT does not replace INP."
        )
        if no_records or failed:
            field_notes.append(
                "Dane terenowe obejmują tylko część próbki."
                if pl
                else "Field data covers only part of the sample."
            )
        if any(c.field and c.attempts[-1].state.value == "PARTIAL" for c in field):
            field_notes.append(
                "Część rekordów zawiera niepełne metryki; brakujących wartości nie oceniono."
                if pl
                else "Some records have partial metrics; missing values were not assessed."
            )
    if no_records:
        field_notes.append(
            "Google nie udostępnił danych rzeczywistych użytkowników dla tej części próbki. "
            "Nie obniża to gotowości i nie dowodzi braku ruchu."
            if pl
            else "Google returned no published real-user data for this part of the sample. "
            "This does not reduce readiness or prove there is no traffic."
        )
    if failed:
        reasons = sorted({_unavailable(c, pl=pl) for c in failed})
        field_notes.append(
            (
                "Nie uzyskano części danych CrUX: "
                if pl
                else "Some CrUX data could not be collected: "
            )
            + "; ".join(reasons)
            + (
                ". Nie jest to potwierdzenie braku danych w Google i nie obniża gotowości."
                if pl
                else ". This does not establish that Google has no records "
                "and does not reduce readiness."
            )
        )
    if not field:
        field_notes.append(
            "CrUX był wyłączony. Brak pomiaru nie obniża gotowości."
            if pl
            else "CrUX was disabled. Missing measurement does not reduce readiness."
        )
    return MeasurementClient(
        report=report,
        registry_version=registry.version,
        as_of=as_of,
        rules=(score_rule, lcp_rule),
        introduction=tuple(intro),
        lab_rows=tuple(lab_rows),
        lab_notes=tuple(lab_notes),
        field_rows=tuple(field_rows),
        field_notes=tuple(field_notes),
        sources=tuple(
            f"[{label}]({rule.source_url})"
            for label, rule in zip(("PageSpeed Insights", "LCP"), rules, strict=True)
        ),
    )


def validate_measurement_client(
    client: MeasurementClient,
    *,
    report: MeasurementReport,
    registry: KnowledgeRegistry | None = None,
    as_of: date | None = None,
) -> MeasurementClient:
    """Fail closed on every unchecked change, including fields serializers would drop."""
    _reject_unserialized_fields(client)
    # Preserve invalid numbers until schema validation can reject them.
    checked = MeasurementClient.model_validate(
        client.model_dump(mode="python", serialize_as_any=True, warnings=False)
    )
    expected = build_measurement_client(report, registry=registry, as_of=as_of)
    if checked != expected:
        raise ValueError("client measurement projection differs from trusted evidence or rules")
    return expected
