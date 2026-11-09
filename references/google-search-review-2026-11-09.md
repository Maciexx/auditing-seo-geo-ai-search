# Przegląd dokumentacji Google a skill audytowy

Data weryfikacji: 2026-11-09. Status: analiza i propozycje, bez zmiany skilla lub silnika.

## Zakres i wniosek

Przejrzano SEO Starter Guide oraz powiązaną dokumentację fundamentów, AI Search,
indeksowania, renderowania, kontroli treści, danych strukturalnych, jakości i pomiaru.
To przegląd przekrojowy dokumentacji istotnej dla skilla, nie potwierdzenie przeczytania
każdej strony Search Central ani wszystkich instrukcji branżowych.
Porównanie obejmuje projektowy SKILL.md, references/content-diagnostics.md i wybrane
reguły knowledge/. Zainstalowana kopia skilla nie zawiera projektowej sekcji
Supplementary content diagnostics and AI observations.

Fundament jest dobry: dowody i ograniczenia próby, techniczne SEO, spójność encji,
języki, widoczna treść zgodna ze schematem, brak gwarancji cytowań i powtarzalny benchmark.
Największa potrzeba to aktualizacja zasad Google AI Search, pomiaru oraz ograniczenie
ryzyka traktowania heurystyk struktury tekstu jako wymogów wyszukiwarki.

## Proponowane uzupełnienia

### 1. Kontrola udziału w Google Search generative AI — wysoki priorytet

Google dokumentuje ustawienie Search Console: Settings → Search generative AI.
Obejmuje włączenie, wyłączenie i dziedziczenie ustawienia po nadrzędnej właściwości.
Dokumentacja podaje globalne wdrożenie 31 sierpnia 2026. Wyłączenie tych funkcji
nie jest tym samym co wyłączenie zwykłego Search lub treningu modeli.

Propozycja do etapów 4 i Owner questions: zapisywać efektywne ustawienie,
właściwość, dziedziczenie, datę oraz dowód od właściciela. Bez dostępu: UNKNOWN,
nie domniemane włączenie ani błąd. Nie zmieniać decyzji publikacyjnej w ramach audytu.
[Źródło: kontrola AI](https://support.google.com/webmasters/answer/16908024).

### 2. Osobny pomiar wyświetleń AI — wysoki priorytet

Generative AI performance report (Search) dokumentuje wyświetlenia AI Overviews
i AI Mode, wymiary stron, krajów, dat i urządzeń oraz wyszukiwanie tekstowe/multimodalne.
Nie należy na tej podstawie dopisywać kliknięć, CTR lub pozycji AI.
Brak raportu może oznaczać za mało wyświetleń. Eksport zamienia niedostępne wartości
oznaczone ~ lub - na zera; potrzebny jest kontekst eksportu. Dane należą też do
Web w zwykłym raporcie — nie dodawać ich drugi raz. Daty raportu są w PT.

Propozycja do Measurement i enrich: osobny typ dowodu, zachowanie semantyki braków,
agregacji, filtrów i okresu. Nie zastępować ręcznego benchmarku tym raportem ani odwrotnie.
[Źródło: raport AI](https://support.google.com/webmasters/answer/16984139).

### 3. Rozdzielenie kontroli botów i prezentacji — wysoki priorytet

Uzupełnić kontrolę robots o nosnippet, max-snippet i data-nosnippet, także w nagłówkach.
Nosnippet blokuje użycie treści jako bezpośredniego wejścia do AIO/AI Mode;
max-snippet ogranicza zakres. Uwzględniać wyjątki opisane dla odrębnych zgód
i danych strukturalnych. Odczyt dyrektywy wymaga dostępu crawlera.
[Źródło: robots meta](https://developers.google.com/search/docs/crawling-indexing/robots-meta-tag).

Google-Extended nie ma osobnego HTTP User-Agent i nie steruje obecnością ani rankingiem
w Google Search. Nie zalecać jego odblokowania jako sposobu na AIO.
[Źródło: crawlery](https://developers.google.com/crawling/docs/crawlers-fetchers/google-common-crawlers#google-extended).

### 4. Jakość treści zamiast formatowania „pod AI” — wysoki priorytet

Nowy przewodnik Google odrzuca obowiązkowy chunking, specjalny styl pisania,
specjalny schema markup i llms.txt jako wymagania Google Search. Zaleca treści
wnoszące własne doświadczenie i wiedzę. Nie należy produkować stron na każdą
domniemaną odmianę query fan-out.

Propozycja: oceniać oryginalne dowody, użyteczność i kompletność odpowiedzi;
przegląd sekcji zachować jako ocenę redakcyjną, bez premiowania samych krótkich
akapitów, liczby nagłówków czy formatu FAQ. Reguły Google nie opisują wszystkich
innych systemów AI. Agentic readiness może być modułem opcjonalnym, osobnym od SEO.
[Źródło: AI optimization guide](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide).

### 5. Autorstwo, weryfikacja i masowo generowana treść — średni priorytet

Dodać pytania: kto odpowiada za treść, jak powstała i jaki cel użytkownika realizuje.
Sprawdzać prawdziwość kwalifikacji i materiał dowodowy; nie tworzyć fikcyjnych ekspertów.
E-E-A-T nie jest osobnym czynnikiem rankingowym ani mierzalnym wynikiem Google.
[Źródło: helpful content](https://developers.google.com/search/docs/fundamentals/creating-helpful-content).

Przegląd treści wspomaganej AI powinien obejmować fakty, tytuły, opisy, alty i markup.
Dla e-commerce osobno sprawdzać obowiązujące oznaczenia materiałów AI w Merchant Center.
[Źródło: treści generowane](https://developers.google.com/search/docs/fundamentals/using-gen-ai-content).
Masowa produkcja bez wartości, doorway pages i manipulowanie odpowiedziami generatywnymi
powinny trafiać do oceny ryzyka spamu, a nie rekomendacji skalowania treści.
[Źródło: spam policies](https://developers.google.com/search/docs/essentials/spam-policies).

### 6. Aktualność rich results — średni priorytet

Nie utożsamiać poprawnego Schema.org z obsługiwaną funkcją Google lub gwarancją wyświetlenia.
[Źródło: structured data policies](https://developers.google.com/search/docs/appearance/structured-data/sd-policies).
Google wycofał FAQ rich results od 7 maja 2026, a później usunął dokumentację.
Nie proponować FAQPage w celu uzyskania tej funkcji. Użyteczne pytania i odpowiedzi
mogą nadal mieć wartość dla czytelnika.
[Źródło: changelog](https://developers.google.com/search/updates).

### 7. Doprecyzowanie klasycznego SEO — średni priorytet

- Dodać jawny przegląd tytułów, opisów wyników, opisowych linków, altów i kontekstu
  obrazów/wideo. Nie narzucać liczby słów, idealnej liczby nagłówków ani meta keywords.
  [SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide).
- Nie wyliczać pokrycia indeksu ani rankingu na podstawie site:. Brak wyniku nie
  dowodzi braku indeksacji. [Operator site:](https://developers.google.com/search/docs/monitor-debug/search-operators/all-search-site).
- Oddzielać dostępność, techniczną kwalifikowalność i potwierdzoną indeksację.
  Robots disallow nie jest zamiennikiem noindex.
  [Wymagania techniczne](https://developers.google.com/search/docs/essentials/technical).
- Brak treści w surowym HTML nie dowodzi braku odczytu przez Google: potrzebna jest
  ocena renderowania. [JavaScript SEO](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics).
- Canonical wybrany przez Google i zadeklarowany przez wydawcę to różne informacje.
  Duplikacja nie oznacza automatycznie kary.
  [Canonicalization](https://developers.google.com/search/docs/crawling-indexing/canonicalization).
- Hreflang może być wdrożony w HTML, nagłówkach lub sitemapie; nie zgłaszać braku
  tylko na podstawie HTML. [Wersje językowe](https://developers.google.com/search/docs/specialty/international/localized-versions).
- Dobry wynik wydajności nie gwarantuje wysokiej pozycji; optymalizacja do 100 punktów
  nie powinna wypierać ważniejszych problemów użytkownika.
  [Page experience](https://developers.google.com/search/docs/appearance/page-experience).

## Rozbieżności źródeł i utrzymanie wiedzy

Starsza strona [AI features](https://developers.google.com/search/docs/appearance/ai-features)
(aktualizacja 2025-12-10) opisuje pomiar w Web i brak dodatkowych wymagań.
Nowszy przewodnik i dedykowane instrukcje Search Console dodają kontrolę udziału
i osobny raport wyświetleń. Stary opis nie wystarcza do aktualnej checklisty.
Reguły trzeba aktualizować z zachowaniem historii, daty sprawdzenia i źródła danej funkcji.
Nie należy automatycznie zmieniać wag historycznych audytów.

W knowledge/vendors/google-gemini.yaml reguła google-ai-features-001 dotyczy Google Search,
choć jest przypisana do google-gemini. Proponuję skorygować klasyfikację i sprawdzić
odwołania do identyfikatora przed migracją. Szczegóły kodowe: osobna notatka
google-docs-code-gaps-2026-11-09.md.

W SKILL.md jest też wewnętrzna sprzeczność: workflow ogranicza dane do widoczności,
lecz Measurement wymienia konwersje. Usunąć je z domyślnego zakresu tego narzędzia.
To uwaga z lokalnego przeglądu, nie wymaganie Google.

## Proponowane paczki wdrożenia

1. Checklisty: kontrola udziału AI, dyrektywy preview, rozdzielenie produktów Google,
   kwalifikowalność a potwierdzona indeksacja.
2. Treść: ograniczenia heurystyk, autorstwo i weryfikacja AI, aktualność rich results,
   media i podstawowe SEO.
3. Pomiar: import wyświetleń AI, braki i agregacja, aktualizacja rejestru wiedzy,
   zgodność raportu z zakresem visibility-only.

Najpierw krótki wpis w SKILL.md odsyłający do szczegółowej referencji; kod i import
zmieniać w osobnej paczce z testami zachowania. Nie przenosić całej dokumentacji do
głównego skilla. Narzędziowe oceny pozostają heurystykami, nie metrykami Google.
[Źródło: third-party SEO advice](https://developers.google.com/search/docs/fundamentals/third-party-seo).
