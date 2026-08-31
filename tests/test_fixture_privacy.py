import json
import re
from pathlib import Path
from urllib.parse import urlsplit

FIXTURES = Path(__file__).parents[1] / "fixtures"
PUBLIC_STANDARD_HOSTS = {"schema.org", "www.sitemaps.org"}
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")


def _fixture_directories() -> list[Path]:
    return sorted(path.parent for path in FIXTURES.glob("*/site.json"))


def test_public_fixtures_are_explicitly_synthetic_and_use_reserved_domains() -> None:
    fixtures = _fixture_directories()
    assert fixtures

    for fixture in fixtures:
        site = json.loads((fixture / "site.json").read_text())
        domain = str(site["domain"]).casefold()
        assert site.get("fixture_kind") == "synthetic", fixture
        assert domain.endswith(".example"), fixture


def test_fixture_urls_use_reserved_or_public_standard_hosts() -> None:
    for fixture in _fixture_directories():
        site = json.loads((fixture / "site.json").read_text())
        allowed_hosts = PUBLIC_STANDARD_HOSTS | {str(site["domain"]).casefold()}
        fixture_text = "\n".join(path.read_text() for path in sorted(fixture.glob("*.json")))

        for raw_url in URL_PATTERN.findall(fixture_text):
            hostname = urlsplit(raw_url.rstrip(".,);\\")).hostname
            assert hostname is not None
            assert hostname.casefold() in allowed_hosts or hostname.casefold().endswith(
                ".example"
            ), f"{fixture}: non-reserved fixture URL {raw_url}"
