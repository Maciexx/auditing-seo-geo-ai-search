import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_github_actions_runs_all_required_quality_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "pytest" in workflow
    assert "ruff format --check" in workflow
    assert "ruff check" in workflow
    assert "mypy" in workflow
    assert "scripts/validate_skill.py" in workflow
    assert "poppler-utils" in workflow
    assert "libpango" in workflow
    assert "Report generation smoke" in workflow
    assert "test_same_environment_render_is_byte_identical" in workflow
    assert "Public repository contract" in workflow
    assert "test_public_repository_contract.py" in workflow
    assert "test_client_pdf_script.py" in workflow


def test_renderer_dependencies_are_exactly_pinned() -> None:
    project = (ROOT / "pyproject.toml").read_text()
    assert '"Jinja2==3.1.6"' in project
    assert '"WeasyPrint==69.0"' in project


def test_development_dependencies_include_reproducible_build_frontend() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "build>=1.2,<2" in project["project"]["optional-dependencies"]["dev"]


def test_gitignore_covers_private_local_workflow_artifacts() -> None:
    ignored = set((ROOT / ".gitignore").read_text().splitlines())

    assert {
        "clients/",
        "owned-input/",
        ".staging/",
        "audit-output/",
        "*.pdf",
        "*.csv",
        "*.xlsx",
    } <= ignored


def test_repository_skill_validator_accepts_the_skill() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_skill.py"), str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Skill is valid" in completed.stdout
