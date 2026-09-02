import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SKILL_NAME = "auditing-seo-geo-ai-search"
PRIVATE_IDENTIFIERS = (
    "Private" + " Customer Sentinel",
    "private-customer" + ".example",
)
GENERATED_ARTIFACT_NAMES = {
    "ai-prompts.json",
    "audit.json",
    "client-report-data.json",
    "client-report.pdf",
    "evidence.jsonl",
    "implementation-backlog.csv",
    "report-draft.json",
}
PRIVATE_PATH_PARTS = {
    ".staging",
    "audit-output",
    "clients",
    "owned-input",
}
RAW_INTAKE_SUFFIXES = {".csv", ".xlsx"}
SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"(?ix)(?:[\"']?)(?:api[-_]?key|access[-_]?token|client[-_]?secret|"
    rb"private[-_]?key|google_application[-_]?credentials)(?:[\"']?)"
    rb"[ \t]*(?:=|:)[ \t]*(?:[\"']?)([^\s\"',}\]\r\n]+)"
)
BEARER_TOKEN_PATTERN = re.compile(
    rb"(?i)\bauthorization[ \t]*:[ \t]*bearer[ \t]+([A-Za-z0-9._~+/=-]{12,})"
)
PLATFORM_TOKEN_PATTERNS = (
    re.compile(rb"\bAIza[A-Za-z0-9_-]{35}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,255}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,255}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{16,255}\b"),
)
PLACEHOLDER_MARKERS = (
    b"placeholder",
    b"redacted",
    b"not-a-real",
    b"example-token",
    b"example_token",
    b"your-api",
    b"your_api",
    b"<your",
    b"${",
)


def _tracked_paths() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return tuple(Path(path) for path in completed.stdout.decode().split("\0") if path)


def _tracked_payloads() -> tuple[tuple[Path, bytes], ...]:
    return tuple((path, (ROOT / path).read_bytes()) for path in _tracked_paths())


def _is_explicit_placeholder(value: bytes) -> bool:
    folded = value.strip(b"\"'").lower()
    return folded in {b"example", b"dummy", b"changeme"} or any(
        marker in folded for marker in PLACEHOLDER_MARKERS
    )


def _assert_no_secrets(payload: bytes, *, location: str) -> None:
    for match in SECRET_ASSIGNMENT_PATTERN.finditer(payload):
        if not _is_explicit_placeholder(match.group(1)):
            raise AssertionError(f"credential assignment in {location}")
    for match in BEARER_TOKEN_PATTERN.finditer(payload):
        if not _is_explicit_placeholder(match.group(1)):
            raise AssertionError(f"bearer token in {location}")
    for pattern in PLATFORM_TOKEN_PATTERNS:
        for match in pattern.finditer(payload):
            if not _is_explicit_placeholder(match.group(0)):
                raise AssertionError(f"platform secret token in {location}")


def test_tracked_tree_excludes_private_outputs_and_generated_reports() -> None:
    forbidden_parts = PRIVATE_PATH_PARTS | {"output", "tmp"}

    for path in _tracked_paths():
        assert path.name not in GENERATED_ARTIFACT_NAMES, f"forbidden tracked file: {path}"
        assert path.suffix.lower() != ".pdf", f"forbidden tracked PDF: {path}"
        assert path.parts[:2] != ("docs", "superpowers"), f"forbidden tracked documentation: {path}"
        assert forbidden_parts.isdisjoint(path.parts), f"forbidden tracked path: {path}"


def test_tracked_tree_excludes_raw_intake_outside_fabricated_fixture_directories() -> None:
    for path in _tracked_paths():
        suffix = path.suffix.lower()
        if suffix not in RAW_INTAKE_SUFFIXES and suffix != ".json":
            continue
        is_fabricated_fixture = path.parts[:1] == ("fixtures",)
        is_versioned_source = path.parts[:2] in {
            ("src", "ai_search_audit"),
            ("knowledge", "standards"),
            ("knowledge", "vendors"),
            ("knowledge", "research"),
            ("knowledge", "experimental"),
        }
        assert is_fabricated_fixture or is_versioned_source, (
            f"raw intake-like JSON outside an approved source or fabricated fixture: {path}"
        )


def test_raw_intake_guard_rejects_csv_and_xlsx_bypass(monkeypatch) -> None:
    module = sys.modules[__name__]
    for path in (Path("unexpected.csv"), Path("nested/unexpected.xlsx")):
        monkeypatch.setattr(module, "_tracked_paths", lambda path=path: (path,))
        with pytest.raises(AssertionError, match="raw intake-like"):
            test_tracked_tree_excludes_raw_intake_outside_fabricated_fixture_directories()


@pytest.mark.parametrize(
    "payload",
    (
        b"api" + b"_key = live-secret-value",
        b'{"access' + b'_token": "live-json-secret"}',
        b"client" + b"_secret: live-yaml-secret",
        b"Authorization: Bearer " + b"liveBearerToken1234567890",
        b"AI" + b"za12345678901234567890123456789012345",
        b"gh" + b"p_123456789012345678901234567890123456",
        b"sk-" + b"proj-123456789012345678901234567890",
    ),
)
def test_release_secret_guard_rejects_whitespace_structured_and_platform_tokens(
    payload: bytes,
) -> None:
    with pytest.raises(AssertionError, match="credential|token|secret"):
        assert_release_archive_boundary(["ai_search_audit/example.txt"], [payload])


@pytest.mark.parametrize(
    "payload",
    (
        b"api" + b"_key = <YOUR_API_KEY>",
        b'{"access' + b'_token": "REDACTED"}',
        b"Authorization: Bearer example-token-placeholder",
        b"sk-" + b"proj-example-token-placeholder-1234567890",
    ),
)
def test_release_secret_guard_allows_explicit_placeholders(payload: bytes) -> None:
    assert_release_archive_boundary(["ai_search_audit/example.txt"], [payload])


def test_tracked_files_exclude_real_client_identifiers_and_secret_assignments() -> None:
    for path, payload in _tracked_payloads():
        folded = payload.lower()
        for identifier in PRIVATE_IDENTIFIERS:
            assert identifier.casefold().encode() not in folded, (
                f"private client identifier in tracked file: {path}"
            )
        _assert_no_secrets(payload, location=f"tracked file: {path}")


def test_tracked_text_files_exclude_absolute_user_paths() -> None:
    absolute_user_prefix = "/" + "Users" + "/"

    for path in _tracked_paths():
        try:
            text = (ROOT / path).read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        assert absolute_user_prefix not in text, f"absolute user path in tracked file: {path}"


def test_fixture_urls_use_reserved_example_domains() -> None:
    allowed_public_hosts = {"schema.org", "www.sitemaps.org"}
    for path in (ROOT / "fixtures").rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for token in text.replace('"', " ").replace("\\n", " ").split():
            if not token.startswith(("http://", "https://")):
                continue
            host = urlsplit(token.rstrip("\\,);]")).hostname
            assert host is not None
            assert host.endswith(".example") or host in allowed_public_hosts, (
                f"non-reserved fixture host in {path}: {host}"
            )


def assert_release_archive_boundary(names: list[str], payloads: list[bytes]) -> None:
    for name in names:
        path = Path(name)
        assert path.name not in GENERATED_ARTIFACT_NAMES, f"generated report in wheel: {name}"
        assert path.suffix.lower() not in {".pdf", ".csv", ".xlsx"}, (
            f"private/report artifact in wheel: {name}"
        )
        assert PRIVATE_PATH_PARTS.isdisjoint(path.parts), f"private path in wheel: {name}"
        assert path.parts[:2] != ("docs", "superpowers"), f"internal plan in wheel: {name}"

    combined = b"\n".join(payloads).lower()
    assert b"/users/" not in combined
    for identifier in PRIVATE_IDENTIFIERS:
        assert identifier.casefold().encode() not in combined
    for name, payload in zip(names, payloads, strict=True):
        _assert_no_secrets(payload, location=f"wheel file: {name}")


def test_built_wheel_excludes_private_release_boundary(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            subprocess.sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        payloads = [archive.read(name) for name in names if not name.endswith("/")]

    assert_release_archive_boundary(names, payloads)


def test_public_repository_uses_canonical_skill_name() -> None:
    skill = (ROOT / "SKILL.md").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert skill.startswith(f"---\nname: {SKILL_NAME}\n")
    assert project["project"]["name"] == SKILL_NAME


def test_agent_default_prompt_invokes_canonical_skill() -> None:
    data = yaml.safe_load((ROOT / "agents" / "openai.yaml").read_text())

    assert f"${SKILL_NAME}" in data["interface"]["default_prompt"]


def test_public_repository_includes_client_pdf_contract_and_renderer() -> None:
    assert (ROOT / "references" / "client-pdf-spec.md").is_file()
    assert (ROOT / "scripts" / "render_client_pdf.py").is_file()


def test_public_repository_includes_concise_project_entrypoint() -> None:
    required_files = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
    ]
    for filename in required_files:
        assert (ROOT / filename).is_file(), f"missing {filename}"

    readme = (ROOT / "README.md").read_text()
    assert len(readme.splitlines()) <= 150

    required_headings = [
        "## What it does",
        "## How it works",
        "## Install",
        "## Run an audit",
        "## Output",
        "## Limits and privacy",
        "## License",
    ]
    for heading in required_headings:
        assert heading in readme

    required_tokens = [
        "$auditing-seo-geo-ai-search",
        "client-ready PDF",
        "https://example.com",
    ]
    for token in required_tokens:
        assert token in readme


def test_readme_documents_local_projects_with_fabricated_public_examples_only() -> None:
    readme = (ROOT / "README.md").read_text()

    for token in (
        "## Local project workflow",
        "project:<id>",
        "public-v1",
        "update",
        "enrich",
        "validate",
        "next-audit-data-request",
    ):
        assert token in readme

    assert "https://example.com" in readme
    assert "project:example" in readme
    assert "/users/" not in readme.casefold()


def test_license_and_security_policy_expose_required_public_terms() -> None:
    license_text = (ROOT / "LICENSE").read_text()
    security = (ROOT / "SECURITY.md").read_text()

    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Maciexx" in license_text
    assert "/security/advisories/new" in security
    assert "Do not open a public issue" in security
