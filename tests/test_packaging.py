import shutil
import subprocess
import sys
import tarfile
import textwrap
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    wheel_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return next(wheel_dir.glob("*.whl"))


def test_pdf_reader_is_a_single_runtime_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = project["project"]["dependencies"]
    dev = project["project"]["optional-dependencies"]["dev"]

    assert runtime.count("pypdf==6.16.0") == 1
    assert "pypdf==6.16.0" not in dev


def test_wheel_contains_project_models_report_assets_and_font_license(
    built_wheel: Path,
) -> None:
    with ZipFile(built_wheel) as archive:
        names = archive.namelist()
    assert "ai_search_audit/project_models.py" in names
    assert "ai_search_audit/project_store.py" in names
    assert "ai_search_audit/assets/render_client_pdf.py" in names
    assert "ai_search_audit/client_delivery.py" in names
    assert any(name.endswith("NotoSans-Variable.ttf") for name in names)
    assert any(name.endswith("OFL-NotoSans.txt") for name in names)
    assert any(name.endswith("locales/en.json") for name in names)
    assert any(name.endswith("locales/pl.json") for name in names)
    assert "ai_search_audit/templates/data-request/en.json" in names
    assert "ai_search_audit/templates/data-request/pl.json" in names


def test_wheel_contains_exact_versioned_knowledge_registry(built_wheel: Path) -> None:
    packaged_root = "ai_search_audit/knowledge_registry/"
    with ZipFile(built_wheel) as archive:
        names = set(archive.namelist())
        registry_name = packaged_root + "registry.yaml"
        registry = yaml.safe_load(archive.read(registry_name))
        expected_relative_paths = {"registry.yaml", *registry["files"]}
        packaged_yaml = {
            name.removeprefix(packaged_root)
            for name in names
            if name.startswith(packaged_root) and name.endswith((".yaml", ".yml"))
        }

        assert packaged_yaml == expected_relative_paths
        assert not any(name.startswith("knowledge/") for name in names)
        for relative_path in expected_relative_paths:
            assert (
                archive.read(packaged_root + relative_path)
                == (ROOT / "knowledge" / relative_path).read_bytes()
            )


def test_wheel_contains_exact_lighthouse_runtime_assets(built_wheel: Path) -> None:
    prefix = "ai_search_audit/lighthouse_assets/"
    expected = {
        "runner.mjs",
        "isolation.py",
        "sidecar.py",
        "seccomp.json",
        "package.json",
        "package-lock.json",
        "Dockerfile",
        "Dockerfile.dockerignore",
        "README.md",
        "LICENSE.moby",
    }
    with ZipFile(built_wheel) as archive:
        actual = {
            name.removeprefix(prefix) for name in archive.namelist() if name.startswith(prefix)
        }
        assert actual == expected
        for name in expected:
            assert (
                archive.read(prefix + name)
                == (ROOT / "src/ai_search_audit/lighthouse_assets" / name).read_bytes()
            )


def test_installed_wheel_runs_diagnostic_handshake_and_finalization_outside_checkout(
    built_wheel: Path, tmp_path: Path, monkeypatch
) -> None:
    from tests.test_diagnostics_end_to_end import build_synthetic_delivery

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    poison = tmp_path / "poisoned-pythonpath"
    poison.mkdir()
    marker = tmp_path / "pythonpath-leaked"
    (poison / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unexpected startup path')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    # Every application action is an installed console-command subprocess, with
    # PYTHONPATH removed and cwd outside the checkout; no editable installation.
    final = build_synthetic_delivery(
        tmp_path / "outside-checkout", "en", python=python, installed=True
    )
    assert final.is_file()
    assert not marker.exists(), "an installed subprocess inherited the parent PYTHONPATH"


def test_installed_wheel_exposes_project_models_resources_and_console_command(
    built_wheel: Path, tmp_path: Path
) -> None:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    pip = venv / "bin" / "pip"
    command = venv / "bin" / "ai-search-audit"
    subprocess.run(
        [str(pip), "install", str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from importlib import resources; "
                "from ai_search_audit.project_models import ProjectManifest; "
                "root = resources.files('ai_search_audit'); "
                "assert root.joinpath('templates/client-report/v1/locales/pl.json').is_file(); "
                "assert root.joinpath('templates/client-report/v1/report.html').is_file(); "
                "assert ProjectManifest.__name__ == 'ProjectManifest'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    for args in (("--help",), ("project", "--help")):
        completed = subprocess.run(
            [str(command), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_installed_wheel_runs_mocked_project_audit_outside_repository(
    built_wheel: Path, tmp_path: Path
) -> None:
    venv = tmp_path / "venv"
    work = tmp_path / "outside-repository"
    work.mkdir()
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    pip = venv / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    script = textwrap.dedent(
        """
        import json
        import hashlib
        import subprocess
        import sys
        from datetime import UTC, datetime
        from importlib import resources
        from pathlib import Path

        import httpx

        from ai_search_audit.knowledge import default_registry_root, load_registry
        from ai_search_audit.project_orchestrator import (
            create_project_audit,
            validate_project_bundle,
        )

        calls = []

        def handler(request):
            calls.append(str(request.url))
            pages = {
                "/robots.txt": (
                    "text/plain",
                    "User-agent: *\\nAllow: /\\nSitemap: https://example.com/sitemap.xml\\n",
                ),
                "/sitemap.xml": (
                    "application/xml",
                    '<?xml version="1.0"?><urlset '
                    'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    "<url><loc>https://example.com/</loc></url></urlset>",
                ),
                "/": (
                    "text/html",
                    '<html lang="en"><head><title>Example</title>'
                    '<meta name="description" content="Fabricated example">'
                    '<link rel="canonical" href="https://example.com/">'
                    '<script type="application/ld+json">'
                    '{"@type":"Organization","name":"Example"}'
                    "</script></head><body><h1>Example</h1><h2>Services</h2></body></html>",
                ),
            }
            content_type, body = pages.get(request.url.path, ("text/plain", ""))
            status = 200 if request.url.path in pages else 404
            return httpx.Response(
                status,
                text=body,
                headers={"content-type": content_type},
                request=request,
            )

        clients_root = Path("clients")
        manifest = create_project_audit(
            "https://example.com",
            clients_root=clients_root,
            project_id="example",
            client_name="Example",
            report_locale="en",
            max_pages=1,
            now=datetime(2026, 9, 1, 12, tzinfo=UTC),
            crawler_transport=httpx.MockTransport(handler),
            crawler_resolver=lambda _hostname: ["8.8.8.8"],
        )
        version_root = clients_root / "example" / manifest.versions[-1].relative_path
        validate_project_bundle(version_root, expected_project_id="example")
        audit = json.loads((version_root / "engine" / "audit.json").read_text())
        assert audit["ruleset_version"] == load_registry(default_registry_root()).version
        assert calls == [
            "https://example.com/robots.txt",
            "https://example.com/sitemap.xml",
            "https://example.com/",
        ]
        assert (version_root / "report" / "Example_AI_Search_SEO_Audit_EN_v1.pdf").is_file()
        assert (version_root / "next-audit-data-request_en.md").is_file()

        from ai_search_audit.client_delivery import finalize_client_report

        source = Path("client.md")
        source.write_text("# Example\\n\\n## Decision\\nObserved public evidence only.\\n")
        pdf = Path("client.pdf")
        with resources.as_file(
            resources.files("ai_search_audit").joinpath("assets/render_client_pdf.py")
        ) as renderer:
            subprocess.run([
                sys.executable, str(renderer), str(source), str(pdf),
                "--client", "Example", "--date", "2026-09-02", "--locale", "en",
                "--version", "public-v1", "--audit-id", manifest.versions[-1].audit_id,
                "--no-hero-reason", "Synthetic installation test",
            ], check=True)
        final = finalize_client_report(
            "project:example", clients_root=clients_root, version_id="public-v1",
            markdown_path=source, pdf_path=pdf,
            reviewed_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
            no_hero_reason="Synthetic installation test",
        )
        assert final == clients_root.resolve() / "example/reports/public-v1/edition-1"
        assert (final / "client-report.pdf").read_bytes() == pdf.read_bytes()
        """
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_build_dependency_is_declared_for_the_development_environment() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "build>=1.2,<2" in project["project"]["optional-dependencies"]["dev"]


def test_source_distribution_excludes_private_working_files_without_git_metadata(tmp_path):
    from tests.test_public_repository_contract import assert_release_archive_boundary

    checkout = tmp_path / "source"
    checkout.mkdir()
    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
        .stdout.decode()
        .split("\0")
    )
    for name in filter(None, tracked):
        if name == ".gitignore":
            continue  # Packaging must be safe without VCS ignore discovery.
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for name in (
        "tmp/private-specimen.pdf",
        "clients/example/audit.json",
        "audit-output/example/evidence.jsonl",
        ".staging/source.csv",
    ):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private synthetic artifact")
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path / "dist")],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        names, payloads = [], []
        for member in archive.getmembers():
            if member.isfile():
                names.append(member.name.split("/", 1)[1])
                payloads.append(archive.extractfile(member).read())
    assert "SKILL.md" in names
    assert "scripts/render_client_pdf.py" in names
    assert any(name.startswith("tests/") for name in names)
    assert_release_archive_boundary(names, [b""] * len(names))
