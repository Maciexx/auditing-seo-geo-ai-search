"""Read-only source-bound projections; never append interpretation after review."""

import json
import re
from datetime import UTC
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from .benchmark import canonical_hash
from .diagnostic_models import (
    BenchmarkHash,
    DiagnosticRunReference,
    FrozenDiagnosticModel,
    _lossless_diagnostic_values,
)
from .diagnostic_observations import DiagnosticObservationRun
from .diagnostic_sources import load_diagnostic_source
from .diagnostic_store import DiagnosticStore
from .diagnostic_workflow import parse_diagnostic_run_reference
from .observation_usage import estimate_attempt

HEADINGS = {
    "pl": "### Obserwacje OpenAI API web search",
    "en": "### OpenAI API web search observations",
}


class ObservationReport(FrozenDiagnosticModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    reference: DiagnosticRunReference
    manifest_sha256: BenchmarkHash
    run: DiagnosticObservationRun

    @model_validator(mode="after")
    def source_binding(self) -> Self:
        if self.reference.source_version != self.run.binding.source_version:
            raise ValueError("observation report source binding mismatch")
        setup = self.run.setup.benchmark_setup
        if setup.provider != "OpenAI" or setup.product != "Responses API":
            raise ValueError("observation projection requires OpenAI Responses API evidence")
        return self


def load_observation_report(
    project_ref: str, *, clients_root: Path, diagnostic_run_ref: str
) -> ObservationReport:
    reference = parse_diagnostic_run_reference(diagnostic_run_ref)
    source = load_diagnostic_source(
        project_ref, clients_root=clients_root, source_version=reference.source_version
    )
    loaded = DiagnosticStore(clients_root / source.binding.project_id).load(
        reference.source_version, reference.run_id
    )
    if not isinstance(loaded.run, DiagnosticObservationRun) or loaded.run.binding != source.binding:
        raise ValueError("observation report requires source-bound API evidence")
    return ObservationReport(
        reference=reference, manifest_sha256=loaded.manifest_sha256, run=loaded.run
    )


def render_observation_fragment(
    report: ObservationReport, *, projection_version: str = "1.0.0"
) -> str:
    from .observation_client import build_observation_client, validate_observation_client

    client = validate_observation_client(
        build_observation_client(report, projection_version=projection_version), report=report
    )
    pl = client.report.run.binding.report_locale == "pl"
    lines = [
        HEADINGS[client.report.run.binding.report_locale],
        "",
        f"<!-- audit-measurement:{canonical_hash(client.model_dump(mode='json'))} -->",
        "",
    ]
    for paragraph in client.introduction:
        lines.extend([paragraph, ""])
    headers = (
        (
            "Zakres / język",
            "Wybrane / wszystkie",
            "Próby",
            "Wzmianki / ocenione",
            "Cytowania / ocenione",
        )
        if pl
        else (
            "Scope / locale",
            "Selected / all",
            "Attempts",
            "Mentions / eligible",
            "Citations / eligible",
        )
    )
    lines.extend(["| " + " | ".join(headers) + " |", "| --- | --- | --- | --- | --- |"])
    for group in client.groups:
        label = (
            ("Pytania o markę" if pl else "Branded")
            if group.scope == "branded"
            else ("Odkrywanie kategorii" if pl else "Category discovery")
        )
        missing = "niedostępne" if pl else "unavailable"
        mentions = f"{group.mentioned}/{group.eligible}" if group.eligible else missing
        citations = (
            f"{group.cited}/{group.citation_eligible}" if group.citation_eligible else missing
        )
        lines.append(
            f"| {label} / {group.locale} | {group.selected}/{group.expected} | "
            f"{group.attempted} | {mentions} | {citations} |"
        )
    lines.append("")
    for note in client.notes:
        lines.extend([note, ""])
    for excerpt in client.excerpts:
        lines.extend([excerpt.label, "", excerpt.text, ""])
        if excerpt.sources:
            lines.extend(
                [
                    (
                        "Źródła cytowane w tym fragmencie: "
                        if pl
                        else "Sources cited in this excerpt: "
                    )
                    + ", ".join(excerpt.sources)
                    + ".",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_observation_technical_fragment(report: ObservationReport) -> str:
    report = ObservationReport.model_validate(_lossless_diagnostic_values(report))
    estimates = [
        estimate_attempt(
            a, report.run.price_provenance, on=a.started_at.astimezone(UTC).date()
        ).model_dump(mode="json")
        for a in report.run.attempts
    ]
    return (
        json.dumps(
            {
                "report": report.model_dump(mode="json"),
                "estimates": estimates,
                "accounting_note": "Estimates are not invoices; missing usage is not zero.",
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def require_observation_fragment(markdown: str, fragment: str | None) -> None:
    headings = sum(markdown.count(heading) for heading in HEADINGS.values())
    if fragment is None:
        if headings:
            raise ValueError("observation fragment requires an explicit diagnostic run")
        return
    if headings != 1 or markdown.count(fragment) != 1:
        raise ValueError("exact observation fragment must be inserted before render and review")
    chapter = re.search(r"^## .+", markdown, flags=re.MULTILINE)
    if chapter is None or chapter.start() > markdown.index(fragment):
        raise ValueError("observation fragment must appear inside an editorial chapter")
