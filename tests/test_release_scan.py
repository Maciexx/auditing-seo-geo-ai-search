import subprocess
import sys
from pathlib import Path

import pytest

SCANNER = Path(__file__).parents[1] / "scripts/validate_public_release.py"


def test_release_scan_detects_private_terms_in_current_files_and_history(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "example.md"
    source.write_text('"Example" + " Private Customer"\n')
    subprocess.run(["git", "add", "example.md"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "initial synthetic fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    command = [
        sys.executable,
        str(SCANNER),
        str(tmp_path),
        "--private-term",
        "Example Private Customer",
    ]
    current = subprocess.run(command, capture_output=True, text=True)
    assert current.returncode == 1
    assert "example.md" in current.stdout
    assert "Example Private Customer" not in current.stdout
    source.write_text("Generic example\n")
    clean = subprocess.run(command, capture_output=True, text=True)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    history = subprocess.run(command + ["--history"], capture_output=True, text=True)
    assert history.returncode == 1
    assert "history" in history.stdout


def test_release_scan_rejects_tracked_reports_without_reading_private_terms(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "client-report.pdf").write_bytes(b"fixture, not a real report")
    subprocess.run(["git", "add", "client-report.pdf"], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(SCANNER), str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "client-report.pdf" in result.stdout


def test_rejected_symlink_does_not_echo_private_name(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    name = "Example-Private-Customer.md"
    (tmp_path / name).symlink_to("missing")
    subprocess.run(["git", "add", name], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(SCANNER), str(tmp_path), "--private-term", "Example Private Customer"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "symlink" in result.stdout
    assert name not in result.stdout


@pytest.mark.parametrize(
    "filename", ["Example-Private-Customer.md", "example_private_customer.bin"]
)
def test_private_names_in_paths_are_rejected_including_binary_files_and_history(tmp_path, filename):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / filename).write_bytes(b"\xff\x00")
    subprocess.run(["git", "add", filename], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    command = [
        sys.executable,
        str(SCANNER),
        str(tmp_path),
        "--private-term",
        "Example Private Customer",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "configured private identifier" in result.stdout
    assert filename not in result.stdout
    subprocess.run(["git", "mv", filename, "generic.bin"], cwd=tmp_path, check=True)
    assert subprocess.run(command, capture_output=True).returncode == 0
    history = subprocess.run(command + ["--history"], capture_output=True, text=True)
    assert history.returncode == 1
    assert "history" in history.stdout
    assert filename not in history.stdout
