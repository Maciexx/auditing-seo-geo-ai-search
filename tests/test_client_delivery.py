from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from tests.test_client_pdf_script import render_edition
from tests.test_project_orchestrator import NOW, _create, _owner_intake


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    manifest = _create(tmp_path, client_name="Example Studio")
    version = manifest.versions[-1]
    root = tmp_path / "clients" / "example"
    result, source, pdf, hero, command = render_edition(tmp_path)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        command + ["--audit-id", version.audit_id], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return (
        root,
        manifest,
        {
            "project_ref": "project:example",
            "clients_root": root.parent,
            "version_id": version.version_id,
            "markdown_path": source,
            "pdf_path": pdf,
            "hero_path": hero,
            "hero_source": "https://studio.example/official-hero.png",
            "reviewed_pdf_sha256": digest(pdf),
        },
    )


def finalize(args):
    from ai_search_audit.client_delivery import finalize_client_report

    return finalize_client_report(**args)


def test_finalization_binds_edition_to_audit_without_mutating_bundle(prepared):
    root, manifest, args = prepared
    bundle = root / manifest.versions[-1].relative_path
    before = {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()}
    destination = finalize(args)
    assert destination == root / "reports/public-v1/edition-1"
    data = json.loads((destination / "delivery.json").read_text())
    assert data["audit_id"] == manifest.latest_audit_id
    assert data["report_status"] == "PUBLIC_EVIDENCE_DRAFT"
    assert data["reviewed_pdf_sha256"] == digest(destination / "client-report.pdf")
    assert (destination / "client-report.md").read_bytes() == args["markdown_path"].read_bytes()
    assert (destination / "cover.png").read_bytes() == args["hero_path"].read_bytes()
    assert len(PdfReader(destination / "client-report.pdf").pages[0].images) == 1
    assert before == {
        str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()
    }
    first = digest(destination / "delivery.json")
    assert finalize(args) == root / "reports/public-v1/edition-2"
    assert digest(destination / "delivery.json") == first


@pytest.mark.parametrize(
    "key,value",
    [
        ("reviewed_pdf_sha256", "0" * 64),
        ("version_id", "context-v99"),
        ("hero_source", ""),
        ("no_hero_reason", "contradictory cover choices"),
    ],
)
def test_invalid_finalization_does_not_publish(prepared, key, value):
    root, _, args = prepared
    args[key] = value
    with pytest.raises(ValueError):
        finalize(args)
    assert not (root / "reports/public-v1/edition-1").exists()


def test_changed_markdown_requires_render_and_review_again(prepared):
    root, _, args = prepared
    args["markdown_path"].write_text("# Example Studio\n\nChanged conclusion\n")
    with pytest.raises(ValueError, match="source"):
        finalize(args)
    assert not (root / "reports/public-v1/edition-1").exists()


def test_technical_pdf_is_not_a_final_client_edition(prepared):
    root, manifest, args = prepared
    args["pdf_path"] = next((root / manifest.versions[-1].relative_path / "report").glob("*.pdf"))
    args["reviewed_pdf_sha256"] = digest(args["pdf_path"])
    with pytest.raises(ValueError, match="editorial"):
        finalize(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("/ClientEditionLocale", "pl"),
        ("/ClientEditionClient", "Other Example"),
        ("/ClientEditionVersion", "context-v2"),
        ("/ClientEditionAuditID", "other-audit"),
        ("/ClientEditionHeroSHA256", "none"),
    ],
)
def test_cross_project_or_stale_renderer_identity_is_rejected(prepared, field, value):
    _, _, args = prepared
    writer = PdfWriter(clone_from=args["pdf_path"])
    writer.add_metadata({field: value})
    writer.write(args["pdf_path"])
    args["reviewed_pdf_sha256"] = digest(args["pdf_path"])
    with pytest.raises(ValueError):
        finalize(args)


def test_output_symlink_is_rejected_without_writing_outside_project(prepared, tmp_path):
    root, _, args = prepared
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "reports").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        finalize(args)
    assert not list(outside.iterdir())


def test_cli_finalizes_explicit_project_version(prepared):
    _, _, args = prepared
    command = [
        sys.executable,
        "-m",
        "ai_search_audit",
        "project",
        "finalize",
        args["project_ref"],
        "--clients-root",
        str(args["clients_root"]),
        "--version-id",
        args["version_id"],
        "--markdown",
        str(args["markdown_path"]),
        "--pdf",
        str(args["pdf_path"]),
        "--hero",
        str(args["hero_path"]),
        "--hero-source",
        args["hero_source"],
        "--reviewed-pdf-sha256",
        args["reviewed_pdf_sha256"],
    ]
    root = Path(__file__).parents[1]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).name == "client-report.pdf"
    assert Path(result.stdout.strip()).is_file()


def test_pdf_content_tampering_is_rejected_even_if_review_hash_is_updated(prepared):
    _, _, args = prepared
    writer = PdfWriter(clone_from=args["pdf_path"])
    writer.remove_page(1)
    writer.write(args["pdf_path"])
    args["reviewed_pdf_sha256"] = digest(args["pdf_path"])
    with pytest.raises(ValueError, match="render"):
        finalize(args)


def test_explicit_no_hero_edition_can_be_finalized(prepared, tmp_path):
    _, manifest, args = prepared
    _, source, pdf, _, command = render_edition(tmp_path, hero=False)
    result = subprocess.run(
        command + ["--audit-id", manifest.latest_audit_id], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args.update(
        markdown_path=source,
        pdf_path=pdf,
        hero_path=None,
        hero_source=None,
        no_hero_reason="No suitable controlled image",
        reviewed_pdf_sha256=digest(pdf),
    )
    destination = finalize(args)
    assert not list(destination.glob("cover.*"))
    assert json.loads((destination / "delivery.json").read_text())["no_hero_reason"]


def test_failed_promotion_preserves_prior_edition_and_removes_partial_staging(
    prepared, monkeypatch
):
    import ai_search_audit.client_delivery as delivery

    root, _, args = prepared
    first = finalize(args)
    before = digest(first / "delivery.json")

    def fail(source, destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(delivery, "_rename_directory_no_replace", fail)
    with pytest.raises(OSError, match="simulated"):
        finalize(args)
    assert digest(first / "delivery.json") == before
    assert sorted(p.name for p in (root / "reports/public-v1").iterdir()) == ["edition-1"]


def test_manifest_version_alias_cannot_override_canonical_bundle_version(prepared, tmp_path):
    from ai_search_audit.project_orchestrator import update_project_context

    root, _, args = prepared
    owned, intake = _owner_intake(tmp_path, entity="Example Studio")
    manifest = update_project_context(
        "project:example",
        clients_root=root.parent,
        intake_dir=owned,
        normalized_intake=intake,
        now=NOW,
    )
    payload = manifest.model_dump(mode="json")
    payload["versions"][-1]["version_id"] = "context-v999"
    (root / "project.json").write_text(json.dumps(payload))
    _, _, _, _, command = render_edition(tmp_path)
    command[command.index("--version") + 1] = "context-v999"
    subprocess.run(command + ["--audit-id", manifest.latest_audit_id], check=True)
    args.update(version_id="context-v999", reviewed_pdf_sha256=digest(args["pdf_path"]))
    with pytest.raises(ValueError, match="canonical"):
        finalize(args)
    assert not (root / "reports").exists()
