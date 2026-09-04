"""Fabricated .example service/commerce workflows, never a live browser or AI test."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader

from ai_search_audit.content_diagnostics import capture_html
from ai_search_audit.diagnostic_models import DiagnosticIntake
from ai_search_audit.diagnostic_store import DiagnosticStore
from ai_search_audit.models import DataState
from ai_search_audit.project_orchestrator import validate_project_bundle
from tests.test_diagnostic_cli import _handshake

ROOT = Path(__file__).parents[1]

# Only network transport and DNS are substituted. All application code stays real,
# including CLI parsing, canonical collection, raw extraction, intake and PDF binding.
BOOTSTRAP = """
import httpx, json, runpy, socket, sys
from ai_search_audit.crawler import NativeCrawler
site = json.loads(sys.argv.pop(1))
original = NativeCrawler.__init__
def handler(request):
    value = site.get(request.url.path)
    return httpx.Response(200 if value else 404,
        text=value[1] if value else '',
        headers={'content-type': value[0] if value else 'text/plain'}, request=request)
def init(self, *args, **kwargs):
    kwargs['transport'] = httpx.MockTransport(handler)
    kwargs['resolver'] = lambda *args: ['93.184.216.34']
    original(self, *args, **kwargs)
NativeCrawler.__init__ = init
def forbidden(*args, **kwargs):
    raise AssertionError('synthetic fixture attempted live network')
socket.socket.connect = forbidden
socket.create_connection = forbidden
entry = sys.argv.pop(1)
if entry == 'module':
    runpy.run_module('ai_search_audit', run_name='__main__')
else:
    runpy.run_path(entry, run_name='__main__')
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def fixture_site(locale):
    polish = locale == "pl"
    domain = "service.example" if polish else "shop.example"
    client = "Przykładowe Studio" if polish else "Example Store"
    heading = "Zakres usługi" if polish else "Delivery conditions"
    quote = (
        "Realizacja może potrwać 2 dni. Termin nie jest gwarantowany."
        if polish
        else "Delivery may take 2 days. Arrival is not guaranteed."
    )
    overview = (
        "Studio projektuje strony dla małych firm. Zakres ustalamy przed rozpoczęciem prac."
        if polish
        else "Example Store sells reusable bottles. Stock is confirmed before dispatch."
    )
    schema = (
        {"@context": "https://schema.org", "@type": "ProfessionalService", "name": client}
        if polish
        else {"@context": "https://schema.org", "@type": "Product", "name": "Example bottle"}
    )
    html = (
        f'<html lang="{locale}"><head><title>{client}</title>'
        f'<meta name="description" content="{overview}">'
        f'<link rel="canonical" href="https://{domain}/">'
        f'<script type="application/ld+json">{json.dumps(schema)}</script></head>'
        f'<body><nav><a href="/details">Details</a></nav><main><h1>{client}</h1>'
        f"<p>{overview}</p><h2>{heading}</h2><p>{quote}</p>"
        "<table><tr><th>Item</th><th>Value</th></tr><tr><td>Days</td><td>2</td></tr></table>"
        "</main></body></html>"
    )
    site = {
        "/robots.txt": ("text/plain", "User-agent: *\nAllow: /\n"),
        "/sitemap.xml": (
            "application/xml",
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"<url><loc>https://{domain}/</loc></url>"
            f"<url><loc>https://{domain}/details</loc></url></urlset>",
        ),
        "/": ("text/html", html),
        "/details": (
            "text/html",
            html.replace(f'https://{domain}/"', f'https://{domain}/details"'),
        ),
    }
    return domain, client, heading, quote, html, site


def start_cli(work, site, arguments, *, python=None, installed=False):
    python = Path(python or sys.executable)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if not installed:
        environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.Popen(
        [
            str(python),
            "-c",
            BOOTSTRAP,
            json.dumps(site),
            str(python.parent / "ai-search-audit") if installed else "module",
            *arguments,
        ],
        cwd=work,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def finish_cli(process, ready=None, *, pending_delivery=False):
    stdout, stderr = process.communicate(ready, timeout=60)
    assert process.returncode == 0, stdout + stderr
    if pending_delivery:
        assert stderr.startswith("Evidence bundle saved. Client delivery is pending:")
        assert "Do not deliver the technical PDF" in stderr
    else:
        assert stderr == ""
    return stdout.strip()


def client_markdown(locale, client, domain, quote, rendered_quote):
    """Human-reviewed synthetic narrative, not generated by the finalizer."""
    if locale == "pl":
        sections = [
            (
                "Decyzja i zakres",
                f"Obserwacja: {client} opisuje usługę i jej warunki na dwóch "
                "sprawdzonych stronach. W pierwszej kolejności doprecyzuj "
                "początek terminu realizacji "
                "oraz sposób potwierdzania zakresu. Nie ma podstaw do "
                "wnioskowania o całej witrynie.\n\n"
                "> Materiał w całości fikcyjny. Służy sprawdzeniu formatu i przepływu audytu, "
                "nie opisuje działającej firmy.\n\n"
                "| Zakres | Dostęp | Ograniczenie |\n|---|---|---|\n"
                "| Strony publiczne | Dwie próbki | Nie jest to pełny crawl |\n"
                "| Treść po renderowaniu | Jedna próbka | Nie dowodzi dostępu konkretnego bota |\n"
                "| AI Search | Brak odpowiedzi | Brak wyniku, nie zero |",
            ),
            (
                "Metoda i źródła dowodów",
                "Oddzielamy obserwacje w próbce od ocen redakcyjnych. "
                "Publiczny audyt stanowi punkt odniesienia. Dodatkowe "
                "porównanie treści ma własny czas "
                "pozyskania i nie zmienia wcześniejszego raportu.\n\n"
                "| Etykieta | Znaczenie |\n|---|---|\n| Obserwacja | Element obecny w próbce |\n"
                "| Ocena agenta | Interpretacja wymagająca przeglądu |\n"
                "| Nieznane | Brak wystarczających danych |\n\n"
                "Nie testowano rzeczywistej przeglądarki ani produktu AI. Odpowiedzi sieciowe i "
                "zrzut treści są jawnie syntetyczne. Wyniki nie stanowią "
                "pomiaru widoczności firmy.",
            ),
            (
                "Techniczne SEO i dostęp",
                "Obserwacja: w próbkach odpowiedź HTTP ma status 200, "
                "a tekst usługi występuje w HTML. Dokument robots dopuszcza pobieranie. "
                "To warunki możliwego dostępu, nie potwierdzenie indeksacji ani cytowania.\n\n"
                "1. Zweryfikuj indeksację obu adresów w Search Console, "
                "jeśli właściciel udostępni dane.\n"
                "2. Zachowaj dostępność istotnych warunków w HTML.\n"
                "3. Sprawdź adresy kanoniczne po zmianie treści.\n\n"
                "Dane o wydajności, logach botów i pełnym pokryciu indeksu pozostają nieznane.",
            ),
            (
                "Treść, język i architektura informacji",
                f"Źródłowy fragment: „{quote}” [1].\n\n"
                "Ocena agenta: fragment zachowuje warunek i brak gwarancji, lecz nie określa, "
                "od którego zdarzenia liczy się termin. Zalecenie: opisać "
                "moment akceptacji zakresu.\n\n"
                f"Obserwacja porównawcza: „{rendered_quote}” [1] występuje tylko w próbce po "
                "renderowaniu. Możliwy wpływ na dostępność informacji jest hipotezą. Nie dowodzi "
                "to, że konkretny system AI nie odczyta strony.\n\n"
                "Zachowaj polskie znaki i ostrożne sformułowania. Nie "
                "zamieniaj „może” na obietnicę.",
            ),
            (
                "Spójność podmiotu i dane strukturalne",
                "Obserwacja: nazwa studia jest zgodna "
                "w nagłówku i oznaczeniu typu usługi. Dwie próbki nie wystarczają do oceny "
                "spójności katalogów, profili i materiałów zewnętrznych.\n\n"
                "| Pole | Właściciel | Kontrola |\n|---|---|---|\n"
                "| Nazwa | Właściciel marki | Zgodność strony i profili |\n"
                "| Zakres usługi | Osoba prowadząca ofertę | Warunki widoczne na stronie |\n"
                "| Termin | Osoba realizująca usługę | Zgodność treści i umowy |\n\n"
                "Rozszerzaj schema tylko o fakty potwierdzone w widocznej "
                "treści. Nie badano opinii ani prasy.",
            ),
            (
                "Obserwacje AI Search",
                "Nie zebrano odpowiedzi AI. Nie podajemy procentu wzmianek "
                "ani cytowań. Brak pomiaru nie oznacza braku widoczności.\n\n"
                "Do kolejnej próby zachowaj dokładny zestaw pytań, język, rynek, produkt, model, "
                "interfejs, tryb wyszukiwania i sposób rozpoczęcia nowej sesji. "
                "Nieznany model wyklucza kontrolowane porównanie liczbowe.\n\n"
                "Wzmianka o marce i cytowanie domeny są oddzielnymi obserwacjami. "
                "Wyniki API i interfejsu konsumenckiego należy prowadzić w osobnych seriach.",
            ),
            (
                "Plan P0/P1/P2",
                "Nie potwierdzono blokady P0 w próbce. Priorytety wynikają z dowodów "
                "i zależności, nie z automatycznej oceny prawdopodobieństwa cytowania.\n\n"
                "| Priorytet | Działanie | Właściciel | Kontrola zakończenia |\n|---|---|---|---|\n"
                "| P1 | Uściślić początek terminu | Osoba prowadząca ofertę "
                "| Warunek na stronie |\n"
                "| P1 | Sprawdzić treść bez renderowania | Zespół strony | "
                "Kluczowy fragment w HTML |\n"
                "| P2 | Powtórzyć stałe pytania AI | Analityk | Zapis pełnych warunków |\n\n"
                "Przed wdrożeniem potwierdź fakty z osobą odpowiedzialną. Po "
                "zmianie wykonaj ponowną próbkę.",
            ),
            (
                "Pomiar i pytania do właściciela",
                "Mierz pokrycie kluczowych informacji i zgodność "
                "wersji językowych. Obserwacje widoczności zapisuj osobno od kontroli wdrożenia. "
                "Zmiana po wdrożeniu nie dowodzi związku przyczynowego.\n\n"
                "1. Od którego zdarzenia liczy się termin dwóch dni?\n"
                "2. Kto zatwierdza aktualny zakres usługi?\n"
                "3. Czy można udostępnić zbiorcze dane widoczności z Search Console?\n\n"
                "Odpowiedzi powinny rozstrzygać te konkretne decyzje. Nie są "
                "warunkiem ukończenia publicznego audytu.",
            ),
            (
                "Źródła",
                f"1. [Fikcyjna strona usługi](https://{domain}/), próbka z 2026-09-03.\n\n"
                f"2. [Fikcyjne szczegóły oferty](https://{domain}/details), "
                "próbka z 2026-09-03.\n\n"
                "Przytoczone fragmenty zachowano po weryfikacji zgodności ze źródłem. "
                "Cała treść, witryna i grafika okładki zostały wytworzone na potrzeby testu. "
                "Nie wykorzystano danych rzeczywistych klientów.\n\n"
                "Zakres dodatkowej obserwacji obejmuje jedną stronę. Daty i "
                "ograniczenia tej próbki "
                "nie zastępują daty publicznego audytu.",
            ),
        ]
    else:
        sections = [
            (
                "Decision and scope",
                f"Observation: {client} identifies its product and delivery "
                "conditions on the two sampled pages. Clarify when the delivery period starts and "
                "keep stock conditions visible before purchase. The sample "
                "does not establish sitewide coverage.\n\n"
                "> Entirely fabricated specimen. This document tests the "
                "audit workflow and page design; "
                "it does not describe a live business.\n\n"
                "| Area | Evidence | Boundary |\n|---|---|---|\n| Public "
                "pages | Two samples | Not a full crawl |\n"
                "| Rendered content | One supplied sample | No provider access conclusion |\n"
                "| AI observation | One synthetic response | Not a live product test |",
            ),
            (
                "Method and limitations",
                "Source observations and editorial assessments remain separate. "
                "The canonical public audit is the baseline; supplementary "
                "observations carry their own "
                "collection times and do not revise that baseline.\n\n"
                "| Label | Meaning |\n|---|---|\n| Observation | Visible in the supplied sample |\n"
                "| Agent assessment | Reviewed interpretation of a quoted passage |\n"
                "| Unknown | Evidence cannot support a measurement |\n\n"
                "HTTP, browser content and AI text are explicit synthetic "
                "fixtures. No live browser or "
                "AI service was tested. Missing measurements are not failures or zero scores.",
            ),
            (
                "Technical discovery and access",
                "Observation: the two sample URLs return HTTP 200 and "
                "include product conditions in HTML. The robots file permits "
                "crawling. This establishes "
                "possible access only, not indexing, retrieval or citation by an AI system.\n\n"
                "1. Check indexing with Search Console if aggregate evidence becomes available.\n"
                "2. Keep material product and delivery facts visible without interaction.\n"
                "3. Recheck canonical URLs after content changes.\n\n"
                "Performance, bot logs and complete index coverage were not measured.",
            ),
            (
                "Content, language and information architecture",
                f"Source passage: “{quote}” [1].\n\n"
                "Agent assessment: the passage retains uncertainty but "
                "leaves the start of the delivery "
                "period unclear. State whether timing begins at payment "
                "confirmation or dispatch.\n\n"
                f"Observed difference: “{rendered_quote}” [1] appears only in the rendered sample. "
                "A possible accessibility effect is an inference, not proof "
                "that a named AI product "
                "cannot read the page.\n\n"
                "Retain the qualification ‘may’ and the lack of a guarantee. "
                "Concise passages should "
                "remain useful outside the page without inventing stronger promises.",
            ),
            (
                "Entity consistency and structured data",
                "Observation: the sampled product label and "
                "markup describe a reusable bottle. Two pages do not "
                "establish consistency across feeds, "
                "marketplaces or controlled profiles. Review those sources "
                "before making a broader claim.\n\n"
                "| Field | Owner | Completion check |\n|---|---|---|\n| "
                "Product name | Merchandising | Page and feed agree |\n"
                "| Availability | Operations | Visible conditions match stock policy |\n"
                "| Delivery | Fulfilment | Start point and qualifications are explicit |\n\n"
                "Add only schema properties supported by visible content. "
                "External reviews and press are outside this sample.",
            ),
            (
                "AI Search observation",
                "One fabricated grounded response mentions Example Store without "
                "citing its domain. Mention and citation are separate "
                "results; neither establishes factual accuracy. "
                "The frozen prompt set supplies the denominator, including "
                "questions without responses.\n\n"
                "The fixture deliberately leaves the model identifier "
                "unknown. It can retain a dated sample, "
                "but cannot support a controlled numerical change against "
                "another run. API and consumer "
                "interfaces remain separate series.\n\n"
                "This is a test recording, not measured commercial "
                "visibility. Repeat exact prompts only "
                "in a genuinely available product and record the exposed settings.",
            ),
            (
                "P0/P1/P2 roadmap",
                "No P0 blocker was established in the sample. Priorities reflect "
                "the evidence and dependencies. Structural "
                "diagnostics have no scoring weight and do not predict citation probability.\n\n"
                "| Priority | Action | Owner | Completion check |\n|---|---|---|---|\n"
                "| P1 | Explain delivery start point | Fulfilment | Visible qualified condition |\n"
                "| P1 | Verify product facts in raw HTML | Web team | "
                "Source-backed passage present |\n"
                "| P2 | Repeat the frozen AI worksheet | Analyst | Exact "
                "settings and responses recorded |\n\n"
                "Confirm operational facts before changing the page. "
                "Reinspect both representations after implementation.",
            ),
            (
                "Measurement and owner questions",
                "Track the presence and accuracy of material product "
                "facts separately from observed search visibility. A later "
                "change does not prove that "
                "implementation caused it. Compare equivalent periods and "
                "known compatible setups.\n\n"
                "1. Does the two-day period begin at payment or dispatch?\n"
                "2. Who approves stock and delivery wording?\n"
                "3. Can the owner provide aggregate Search Console visibility exports?\n\n"
                "These answers resolve specific decisions. They are not "
                "prerequisites for completing the public audit.",
            ),
            (
                "Sources",
                f"1. [Fabricated product page](https://{domain}/), supplied "
                "sample dated 2026-09-03.\n\n"
                f"2. [Fabricated product details](https://{domain}/details), "
                "supplied sample dated 2026-09-03.\n\n"
                "Quoted passages were checked against their supplied source. "
                "The site, all observations "
                "and cover artwork are synthetic. No real client identity or image is used.\n\n"
                "The supplementary comparison covers one page. Its "
                "collection dates and limitations "
                "remain separate from the canonical audit date.",
            ),
        ]
    # A two-page synthetic site warrants five client pages, not a padded full-audit edition.
    sections[0] = (
        sections[0][0],
        sections[0][1] + "\n\n### " + sections[1][0] + "\n\n" + sections[1][1],
    )
    sections[2] = (
        sections[2][0],
        sections[2][1] + "\n\n### " + sections[4][0] + "\n\n" + sections[4][1],
    )
    sections[7] = (
        sections[7][0],
        sections[7][1] + "\n\n### " + sections[8][0] + "\n\n" + sections[8][1],
    )
    sections[3] = (
        sections[3][0],
        sections[3][1] + "\n\n### " + sections[5][0] + "\n\n" + sections[5][1],
    )
    sections[6] = (
        sections[6][0],
        sections[6][1] + "\n\n### " + sections[7][0] + "\n\n" + sections[7][1],
    )
    sections = [section for index, section in enumerate(sections) if index not in {1, 4, 5, 7, 8}]
    body = "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
    return f"# {client}\n\n" + body.replace("\n\n2. [", "\n2. [") + "\n"


def build_synthetic_delivery(work, locale, *, python=None, installed=False):
    """Reusable by the isolated-wheel smoke; returns final immutable artifact paths."""
    work.mkdir(parents=True, exist_ok=True)
    domain, client, heading, quote, html, site = fixture_site(locale)
    options = {"python": python, "installed": installed}

    def cli(args):
        return start_cli(work, site, args, **options)

    common = ["project:example", "--clients-root", str(work / "clients")]
    bundle = Path(
        finish_cli(
            cli(
                [
                    "project",
                    "create",
                    f"https://{domain}",
                    "--clients-root",
                    str(work / "clients"),
                    "--project-id",
                    "example",
                    "--client-name",
                    client,
                    "--report-locale",
                    locale,
                    "--max-pages",
                    "2",
                ]
            ),
            pending_delivery=True,
        )
    )
    project = work / "clients/example"
    validate_project_bundle(bundle, expected_project_id="example")
    canonical_before = hashes(project)
    worksheet = json.loads(
        finish_cli(
            cli(
                [
                    "project",
                    "benchmark-prepare",
                    *common,
                    "--source-version",
                    "public-v1",
                ]
            )
        )
    )
    assert hashes(project) == canonical_before
    process = cli(
        [
            "project",
            "diagnose",
            *common,
            "--source-version",
            "public-v1",
            "--intake-root",
            str(work / "intake"),
        ]
    )
    owned, contract = _handshake(process)
    assert worksheet == contract.worksheet.model_dump(mode="json")
    rendered_quote = (
        "Zakres wymaga potwierdzenia." if locale == "pl" else "Stock requires confirmation."
    )
    captured = {
        "kind": "rendered",
        "url": f"https://{domain}/",
        "final_url": f"https://{domain}/",
        "html": html.replace("</main>", f"<p>{rendered_quote}</p></main>"),
        "observed_at": datetime.now(UTC).isoformat(),
        "locale": locale,
        "session_key": contract.session_key,
        "consent_state": "none",
        "account_state": "anonymous",
        "collector": "synthetic-browser-fixture/1",
        "viewport": [1280, 800],
        "status_code": 200,
        "complete": True,
        "truncated": False,
    }
    capture = capture_html(**{k: v for k, v in captured.items() if k != "account_state"})
    section = next(s for s in capture.sections if s.heading == heading)
    payload = {
        "expected_binding": contract.source.binding.model_dump(mode="json"),
        "worksheet": worksheet,
        "rendered_captures": [captured],
        "key_passages": [
            {
                "capture_id": capture.capture_id,
                "section_id": section.section_id,
                "quote": rendered_quote,
            }
        ],
        "section_reviews": [
            {
                "capture_id": capture.capture_id,
                "reviews": [
                    {
                        "section_id": section.section_id,
                        "criterion": "conditions",
                        "result": "needs_review",
                        "quotes": [quote],
                        "rationale": "Doprecyzuj początek terminu."
                        if locale == "pl"
                        else "Clarify when the delivery period begins.",
                    }
                ],
            }
        ],
    }
    if locale == "en":
        prompt = next(p for p in contract.worksheet.prompts if p.locale == locale)
        payload.update(
            {
                "worksheet": worksheet,
                "setup": {
                    "provider": "Synthetic fixture",
                    "product": "Synthetic answer fixture",
                    "model_id": None,
                    "interface": "consumer_ui",
                    "search_mode": "enabled",
                    "locale": "en",
                    "market": "GB",
                    "account_state": "anonymous",
                    "reset_method": "fresh synthetic session",
                },
                "responses": [
                    {
                        "prompt_id": prompt.prompt_id,
                        "prompt_text": prompt.text,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "response_text": "Example Store sells bottles.",
                        "grounded": True,
                        "brand_mentioned": True,
                        "citations": [],
                        "citations_complete": True,
                        "complete": True,
                        "response_truncated": False,
                        "inspection_scope": "full_response",
                    }
                ],
            }
        )
    validated = DiagnosticIntake.model_validate(payload)
    (owned / "normalized-intake.json").write_text(validated.model_dump_json(), encoding="utf-8")
    destination = Path(finish_cli(process, "READY\n"))
    assert not owned.exists()
    assert destination == project / "diagnostics/public-v1/run-1"
    loaded = DiagnosticStore(project).load("public-v1", "run-1")
    assert loaded.run.pairs[0].state is DataState.AVAILABLE
    assert loaded.run.pairs[1].state is DataState.UNAVAILABLE
    assert loaded.run.pairs[0].rendered_only_quotes == (rendered_quote,)
    assert loaded.run.section_reviews[0].reviews[0].review.quotes == (quote,)
    assert all(f.rule.scoring_weight == 0 for f in loaded.run.findings)
    if locale == "pl":
        assert loaded.run.module_states.benchmark is DataState.UNAVAILABLE
        assert loaded.run.benchmark.metrics.mention_rate is None
        assert loaded.run.benchmark.metrics.citation_rate is None
        assert loaded.run.benchmark.metrics.measured == 0
    else:
        assert loaded.run.benchmark.worksheet.setup.model_id is None
        assert loaded.run.benchmark.metrics.measured == 1
        assert loaded.run.benchmark.metrics.expected == len(worksheet["prompts"])
        assert loaded.run.benchmark.metrics.citation_rate == 0
    assert loaded.run.comparison is None
    assert all(item.state is DataState.UNKNOWN for item in loaded.run.section_reviews[0].directness)
    assert '"html"' not in (destination / "diagnostics.json").read_text()
    assert {
        name: sha for name, sha in hashes(project).items() if not name.startswith("diagnostics/")
    } == canonical_before
    collection_date = loaded.run.collection_range.start.date().isoformat()
    source = work / "client-report.md"
    source.write_text(
        client_markdown(locale, client, domain, quote, rendered_quote).replace(
            "2026-09-03", collection_date
        ),
        encoding="utf-8",
    )
    hero = work / "synthetic-cover.png"
    artwork = Image.new("RGB", (1600, 900), "#173D34")
    draw = ImageDraw.Draw(artwork)
    draw.rectangle((120, 140, 700, 760), fill="#A78849")
    draw.rectangle((760, 240, 1460, 660), fill="#F4F0E7")
    artwork.save(hero)
    pdf = work / "client-report.pdf"
    renderer = ROOT / "scripts/render_client_pdf.py"
    if installed:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        located = subprocess.run(
            [
                str(python),
                "-c",
                "from importlib.resources import files; "
                "print(files('ai_search_audit').joinpath('assets/render_client_pdf.py'))",
            ],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        renderer = Path(located.stdout.strip())
    command = [
        str(python or sys.executable),
        str(renderer),
        str(source),
        str(pdf),
        "--client",
        client,
        "--title",
        "Audyt SEO i AI Search" if locale == "pl" else "SEO and AI Search Audit",
        "--subtitle",
        "Przykład syntetyczny" if locale == "pl" else "Synthetic specimen",
        "--date",
        collection_date,
        "--version",
        "public-v1",
        "--audit-id",
        contract.source.binding.audit_id,
        "--locale",
        locale,
        "--hero",
        str(hero),
    ]
    result = subprocess.run(
        command,
        cwd=work,
        env=environment if installed else None,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    reader = PdfReader(pdf)
    assert len(reader.pages) == 5
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert quote in source.read_text()
    assert rendered_quote in text and "\ufffd" not in text
    assert ("AUDYT WIDOCZNOŚCI CYFROWEJ" if locale == "pl" else "DIGITAL VISIBILITY AUDIT") in text
    assert len(reader.pages[0].images) == 1
    assert reader.metadata["/ClientEditionLocale"] == locale
    assert reader.metadata["/ClientEditionHeroSHA256"] == digest(hero)
    assert reader.metadata["/ClientEditionSourceSHA256"] == digest(source)
    assert all(abs(float(page.mediabox.width) - 595.28) < 1 for page in reader.pages)
    final = Path(
        finish_cli(
            cli(
                [
                    "project",
                    "finalize",
                    *common,
                    "--version-id",
                    "public-v1",
                    "--markdown",
                    str(source),
                    "--pdf",
                    str(pdf),
                    "--hero",
                    str(hero),
                    "--hero-source",
                    f"https://{domain}/synthetic-cover.png",
                    "--reviewed-pdf-sha256",
                    digest(pdf),
                    "--diagnostic-run",
                    "public-v1/run-1",
                ]
            )
        )
    )
    delivery = json.loads(final.with_name("delivery.json").read_text())
    assert delivery["diagnostic_run"] == {
        "path": "diagnostics/public-v1/run-1",
        "source_audit_id": contract.source.binding.audit_id,
        "manifest_sha256": loaded.manifest_sha256,
        "content_sha256": digest(destination / "diagnostics.json"),
    }
    assert delivery["report_status"] == "PUBLIC_EVIDENCE_DRAFT"
    assert final.read_bytes() == pdf.read_bytes()
    assert final.with_suffix(".md").read_bytes() == source.read_bytes()
    assert {
        name: sha
        for name, sha in hashes(project).items()
        if not name.startswith(("diagnostics/", "reports/"))
    } == canonical_before
    return final


@pytest.mark.parametrize("locale", ["pl", "en"])
def test_synthetic_service_and_commerce_deliver_source_bound_editorial_pdf(tmp_path, locale):
    build_synthetic_delivery(tmp_path / locale, locale)
