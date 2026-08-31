import subprocess
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
SKILL_NAME = "auditing-seo-geo-ai-search"


def _tracked_paths() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return tuple(Path(path) for path in completed.stdout.decode().split("\0") if path)


def test_tracked_tree_excludes_private_outputs_and_generated_reports() -> None:
    forbidden_names = {
        "audit.json",
        "evidence.jsonl",
        "implementation-backlog.csv",
        "ai-prompts.json",
        "report-draft.json",
        "client-report-data.json",
        "client-report.pdf",
    }
    forbidden_parts = {"audit-output", "output", "tmp"}

    for path in _tracked_paths():
        assert path.name not in forbidden_names, f"forbidden tracked file: {path}"
        assert path.suffix.lower() != ".pdf", f"forbidden tracked PDF: {path}"
        assert path.parts[:2] != ("docs", "superpowers"), f"forbidden tracked documentation: {path}"
        assert forbidden_parts.isdisjoint(path.parts), f"forbidden tracked path: {path}"


def test_tracked_text_files_exclude_absolute_user_paths() -> None:
    absolute_user_prefix = "/" + "Users" + "/"

    for path in _tracked_paths():
        try:
            text = (ROOT / path).read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        assert absolute_user_prefix not in text, f"absolute user path in tracked file: {path}"


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


def test_license_and_security_policy_expose_required_public_terms() -> None:
    license_text = (ROOT / "LICENSE").read_text()
    security = (ROOT / "SECURITY.md").read_text()

    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Maciexx" in license_text
    assert "/security/advisories/new" in security
    assert "Do not open a public issue" in security
