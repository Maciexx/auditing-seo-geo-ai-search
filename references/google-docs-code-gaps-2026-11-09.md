# Luki w implementacji po przeglądzie dokumentacji Google

Data: 2026-11-09. Przegląd statyczny; nie zmieniono implementacji ani wag.
Źródła Google wskazane poniżej zostały odczytane w głównym przeglądzie dokumentacji.
Poniższe obserwacje kodu nie oznaczają przeprowadzenia testu integracji z Search Console.

## 1. Rozdzielenie produktów i warunków udziału

**Obserwacja lokalna:** `knowledge/vendors/google-gemini.yaml:2` przechowuje
`google-ai-features-001` z `vendor: google-gemini`, mimo że opis dotyczy Google Search.
Reguła odsyła do starszej strony `appearance/ai-features` i ma datę weryfikacji
2026-08-11. W przeszukanych `src/ai_search_audit`, `knowledge` i `SKILL.md` nie ma
osobnego opisu Search generative AI control, Google-Extended ani kontroli snippetów.

**Propozycja:** osobno opisać Google Search AI, Gemini oraz kontrolę treningu;
do instrukcji audytu dodać sprawdzenie ustawienia i dziedziczenia właściwości w GSC.
Bez dostępu stan ma pozostać nieznany. Nie wnioskować o ustawieniu z robots.txt.
Zmiana identyfikatora lub przypisania reguły wymaga uwzględnienia zgodności historii.

Źródło: [Search generative AI control](https://support.google.com/webmasters/answer/16908024).

## 2. Osobny import raportu generatywnego

**Obserwacja lokalna:** `src/ai_search_audit/adapters.py:25` wymienia
`gsc-generative-ai-reporting` wyłącznie w `FUTURE_CAPABILITIES`.
`visibility_metrics.py:182` definiuje ogólne metryki GSC, bez osobnej metryki
raportu generatywnego. Lista typów w `visibility_metrics.py:320` nie rozróżnia
Web text-based i Web multimodal. Walidacja pochodzenia w `visibility_metrics.py:380`
wymaga obsługiwanego `search_type`, ale nie wymusza identyfikacji raportu generatywnego.
`templates/data-request/en.json:63` i `pl.json:63` proszą o zwykły raport
Performance > Search results. Katalog `data_requests.py:94` nie ma oddzielnej pozycji
dla raportu generatywnego ani ustawienia udziału.

**Propozycja:** dodać opcjonalny eksport z jednoznacznym pochodzeniem, miarą
wyświetleń, zakresem i agregacją. Nie zakładać, że raport dostarcza kliknięcia,
CTR, pozycję lub zapytania. Zachować rozróżnienie agregacji właściwości i stron,
strefę PT, ograniczenia eksportu i brak danych. Nie dodawać jego wyświetleń do
ogólnego Web jako rozłącznego ruchu. W szczególności eksportowane zera mogą
pochodzić z oznaczeń niedostępności w interfejsie: potrzebna informacja o źródle.

Źródło: [Generative AI performance report](https://support.google.com/webmasters/answer/16984139).

## 3. Heurystyka „citation-ready” wymaga korekty znaczenia

**Obserwacja lokalna:** `analyzers.py:319` sprawdza obecność title, meta description,
H1 i H2 oraz minimum 100 znaków tekstu. Brak któregokolwiek trafia do wspólnego
findingu z nazwą klientowską „citation-ready context” (`analyzers.py:367`).
`knowledge/research/content-citability.yaml:11` nadaje tej regule wagę 0.75.
Opis zawiera już zastrzeżenie, że wynik nie przewiduje cytowania; jest ono potrzebne,
ale nie usuwa ryzyka nadinterpretacji samej nazwy i automatycznej oceny.

**Propozycja:** przedstawić wynik jako obserwację struktury treści. Brak H2,
meta description albo krótka strona nie powinny same stanowić dowodu niskiej
jakości lub nieprzydatności dla AI. Ocena powinna uwzględniać cel strony i wartość
dla użytkownika. Ewentualna zmiana wagi wymaga wersjonowania i testów regresji;
nie przeliczać po cichu historycznych wyników.

Źródło: [AI optimization guide](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide).

## 4. llms.txt — doprecyzowanie zakresu dostawcy

**Obserwacja lokalna:** `SKILL.md:204` i
`knowledge/experimental/llms-txt.yaml:4` prawidłowo oznaczają llms.txt jako
eksperyment; waga wynosi zero. Brakuje wyraźnej reguły dla Google Search.

**Propozycja:** zachować ogólne zastrzeżenie dla innych systemów, ale dla Google
Search zapisać wprost brak wykorzystania llms.txt jako mechanizmu poprawy
widoczności. Nie proponować przymusowego chunkingu ani osobnej wersji Markdown
jako warunku obecności w Google AI.

Źródło: [AI optimization guide](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide).
